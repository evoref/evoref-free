"""順位付け・ゲート・畳み込み (c_16 §7) — 全ストア共通の 1 本

``columns.npz`` のカラム配列に対する numpy ベクトル演算だけで、

1. ゲート = **素の cosine のみ** (§7.1)
2. 順位式 = ``cos × freshness × confidence × store_prior`` (§7.2)
3. 畳み込み = アクティブマスク → ``claim_key`` → origin 優先 → assistant 除外 (§7.3)

を行う。``rag.score_normalization`` / RRF / カートリッジ ``priority`` は使わない。
Python の行ループは持たない (畳み込み後の ≤ k 件でも持たない)。

**lexical スコアを持ち込まないこと** (§6.3)。転置索引は候補生成器であって、
順位は必ずここで cosine から計算し直す。:func:`merge_candidates` は lexical のみで
拾われた行に対して、呼び出し側のコールバックで実 cosine を計算させるための入口。

---

**カラムの前提** (c_16 §5.3。``columns.py`` は別担当が実装中のため、ここでは
必要な配列だけを :class:`RankColumns` として受け取る。名前の突き合わせは結合時)

``flags`` は ``uint8`` のビットフィールドで、本モジュールは以下を仮定する:

===  ==================  ==========================================
bit  定数                 意味
===  ==================  ==========================================
0    ``FLAG_PRIVATE``     private (私的セッション外では呼出側がマスク)
1    ``FLAG_SECRET``      ``confidentiality=secret`` (注入もログも不可)
2    ``FLAG_PINNED``      pinned (保持 GC 除外。順位には効かない)
3    ``FLAG_RETRACTED``   ``veracity=retracted``
4    ``FLAG_SUPERSEDED``  ``superseded_by`` が埋まっている
5    ``FLAG_ASSISTANT``   ``origin=assistant``
===  ==================  ==========================================

``origin`` は ``uint8`` enum で、c_16 §3 のユニオン記載順を値とする
(:class:`Origin`)。優先順位は値ではなく **名前** で与える
(:data:`DEFAULT_ORIGIN_PRIORITY`) ので、別担当の enum と値がずれても
名前が合っていれば再配線できる。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from backend.free.rag.evidence.columns import (
    FLAG_ASSISTANT_ORIGIN,
    FLAG_PINNED,
    FLAG_PRIVATE,
    FLAG_RETRACTED,
    FLAG_SECRET,
    FLAG_SUPERSEDED,
    EvidenceColumns,
)
from backend.free.rag.evidence.types import ORIGIN_IDS
from backend.log_config import get_logger

logger = get_logger("rag.evidence.ranking")


# ------------------------------------------------------------------ フラグ

# ビット表と origin 表は columns.py / types.py が SSOT。ここでは再エクスポートだけ
# 行い、複製を持たない (食い違った複製を避ける)。
FLAG_ASSISTANT: int = FLAG_ASSISTANT_ORIGIN

#: 注入から必ず落ちるフラグ (§7.3-1)。private と失効は呼出側のマスク責務。
INACTIVE_FLAGS: int = FLAG_RETRACTED | FLAG_SUPERSEDED | FLAG_SECRET


class Origin(IntEnum):
    """``origin`` の uint8 enum。値は :data:`types.ORIGIN_IDS` と一致させる。"""

    USER = ORIGIN_IDS["user"]
    ASSISTANT = ORIGIN_IDS["assistant"]
    TOOL = ORIGIN_IDS["tool"]
    DOCUMENT = ORIGIN_IDS["document"]
    WEB = ORIGIN_IDS["web"]
    SYSTEM = ORIGIN_IDS["system"]


#: 同点解決の優先順 (先頭が強い、§7.3-3)。``system`` は §7.3 に記載が無いため末尾。
DEFAULT_ORIGIN_PRIORITY: tuple[str, ...] = (
    "user", "tool", "document", "web", "assistant", "system",
)

#: 1 日の秒数 (epoch 差 → 日数)。
SECONDS_PER_DAY: float = 86400.0


@dataclass(frozen=True)
class RankColumns:
    """順位付けが読むカラム配列 (行位置でそろっている前提)。

    Attributes:
        as_of_epoch: ``float64`` (N)。内容が真だった時点。NaN = null。
        observed_epoch: ``float64`` (N)。こちらが知った時刻。
        half_life_days: ``float32`` (N)。NaN / 非正 = 減衰なし。
        confidence: ``float32`` (N)。
        flags: ``uint8`` (N)。上記ビット表。
        origin: ``uint8`` (N)。:class:`Origin` の値。
        claim_hash64: ``uint64`` (N)。``claim_key`` 先頭 64bit。0 = 鍵なし
            (畳み込み対象外)。
        store_prior_per_row: ``float32`` (N) または ``None``。corpus の
            ``manifest.store_prior_overrides`` のように行ごとに違う場合に使う。
    """

    as_of_epoch: np.ndarray
    observed_epoch: np.ndarray
    half_life_days: np.ndarray
    confidence: np.ndarray
    flags: np.ndarray
    origin: np.ndarray
    claim_hash64: np.ndarray
    store_prior_per_row: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        """行数。"""
        return int(self.observed_epoch.shape[0])

    @classmethod
    def from_columns(
        cls, columns: EvidenceColumns, store_prior_per_row: np.ndarray | None = None,
    ) -> RankColumns:
        """snapshot の :class:`EvidenceColumns` から順位付け用の射影を作る。"""
        return cls(
            as_of_epoch=columns.as_of_epoch,
            observed_epoch=columns.observed_epoch,
            half_life_days=columns.half_life_days,
            confidence=columns.confidence,
            flags=columns.flags,
            origin=columns.origin,
            claim_hash64=columns.claim_hash64,
            store_prior_per_row=store_prior_per_row,
        )


# -------------------------------------------------------------------- ゲート


def gate_by_cosine(cosines: np.ndarray, threshold: float) -> np.ndarray:
    """**素の cosine だけ** で閾値ゲートを掛ける (§7.1)。

    合成スコア (freshness / confidence / store_prior 込み) を閾値に流すと、
    「priority が閾値を偽装する」既知の欠陥に戻る。入るのは cosine のみ。

    Returns:
        ``bool`` マスク。NaN の cosine は ``False``。
    """
    cos = np.asarray(cosines, dtype=np.float64)
    return np.greater_equal(cos, threshold, where=~np.isnan(cos), out=np.zeros(cos.shape, dtype=bool))


# -------------------------------------------------------------------- 順位式


def freshness(
    rows: np.ndarray, columns: RankColumns, now_epoch: float,
) -> np.ndarray:
    """``0.5 ** (age_days / half_life_days)`` (§7.2)。

    - ``half_life_days`` が NaN / 非正なら減衰なし (1.0)
    - age は ``as_of``、無ければ ``observed_at`` から測る
    - 未来の時点 (age < 0) は 0 に丸める (1.0 を超える鮮度を作らない)
    """
    idx = np.asarray(rows, dtype=np.int64)
    as_of = np.asarray(columns.as_of_epoch, dtype=np.float64)[idx]
    observed = np.asarray(columns.observed_epoch, dtype=np.float64)[idx]
    base = np.where(np.isnan(as_of), observed, as_of)
    age_days = (now_epoch - base) / SECONDS_PER_DAY
    age_days = np.where(np.isnan(age_days), 0.0, age_days)
    age_days = np.maximum(age_days, 0.0)

    half_life = np.asarray(columns.half_life_days, dtype=np.float64)[idx]
    decaying = ~np.isnan(half_life) & (half_life > 0.0)
    result = np.ones(idx.shape[0], dtype=np.float64)
    if np.any(decaying):
        safe_hl = np.where(decaying, half_life, 1.0)
        result[decaying] = np.power(0.5, age_days[decaying] / safe_hl[decaying])
    return result


def _resolve_store_prior(
    rows: np.ndarray, columns: RankColumns, store_prior: float | np.ndarray,
) -> np.ndarray:
    """``store_prior`` をスカラ / 行揃え配列 / 全行配列のいずれからも解決する。"""
    idx = np.asarray(rows, dtype=np.int64)
    if np.isscalar(store_prior):
        return np.full(idx.shape[0], float(store_prior), dtype=np.float64)  # type: ignore[arg-type]
    prior = np.asarray(store_prior, dtype=np.float64)
    if prior.ndim == 0:
        return np.full(idx.shape[0], float(prior), dtype=np.float64)
    if prior.shape[0] == idx.shape[0]:
        return prior
    if prior.shape[0] == columns.n_rows:
        return prior[idx]
    raise ValueError(
        f"store_prior length {prior.shape[0]} matches neither candidates "
        f"({idx.shape[0]}) nor rows ({columns.n_rows})"
    )


def score_rows(
    cosines: np.ndarray,
    rows: np.ndarray,
    columns: RankColumns,
    now_epoch: float,
    store_prior: float | np.ndarray,
) -> np.ndarray:
    """``cos × freshness × confidence × store_prior`` (§7.2)。

    Args:
        cosines: ``rows`` と同じ長さの cosine。
        rows: 候補の行番号。
        columns: カラム配列。
        now_epoch: 現在時刻の epoch 秒 (``backend.utils.utc_now_dt().timestamp()``)。
            内部で時刻を取らないのは、同じ候補集合を何度評価しても同じ順位に
            なるようにするため。
        store_prior: スカラ、候補と同じ長さ、または全行と同じ長さの配列。
            ``columns.store_prior_per_row`` があり ``store_prior`` に配列を渡さない
            場合は行ごとの値を掛ける。

    Returns:
        ``float32`` のスコア (``rows`` と同じ並び)。
    """
    idx = np.asarray(rows, dtype=np.int64)
    cos = np.asarray(cosines, dtype=np.float64)
    if cos.shape[0] != idx.shape[0]:
        raise ValueError(f"cosines ({cos.shape[0]}) and rows ({idx.shape[0]}) length mismatch")

    prior = _resolve_store_prior(idx, columns, store_prior)
    if columns.store_prior_per_row is not None and np.isscalar(store_prior):
        prior = prior * np.asarray(columns.store_prior_per_row, dtype=np.float64)[idx]

    confidence = np.asarray(columns.confidence, dtype=np.float64)[idx]
    score = cos * freshness(idx, columns, now_epoch) * confidence * prior
    return score.astype(np.float32, copy=False)


# ------------------------------------------------------------------ 畳み込み


def _origin_rank_table(origin_priority: Sequence[str]) -> np.ndarray:
    """origin enum 値 → 優先順位 (小さいほど強い)。未掲載は最下位。

    ``uint8`` の全域 (256) を張るので、別担当の enum が知らない値を持ち込んでも
    IndexError にならず「最下位」に落ちる。
    """
    table = np.full(256, len(origin_priority), dtype=np.int64)
    for rank, name in enumerate(origin_priority):
        try:
            table[int(Origin[name.upper()])] = rank
        except KeyError:
            logger.debug("unknown origin name in priority list: %s", name)
    return table


def collapse(
    rows: np.ndarray,
    scores: np.ndarray,
    columns: RankColumns,
    *,
    allow_assistant_origin: bool = False,
    origin_priority: Sequence[str] = DEFAULT_ORIGIN_PRIORITY,
) -> tuple[np.ndarray, np.ndarray]:
    """アクティブマスク → ``claim_key`` 畳み込み → assistant 除外 (§7.3)。

    1. ``retracted`` / ``superseded`` / ``secret`` の行を落とす。
       ``private`` と失効 (``valid_until``) は呼出側がマスクで扱う
       (私的セッションかどうかは順位付けの知識ではない)。
    2. 同一 ``claim_hash64`` を 1 件へ畳む。勝者は **confidence 最大 → ``as_of``
       最新 → origin 優先 → 行番号昇順**。``claim_hash64 == 0`` は鍵なしとして
       畳み込み対象外。
    3. ``origin=assistant`` は既定で落とす (``allow_assistant_origin`` で許可)。

    Returns:
        ``(rows int64, scores float32)`` をスコア降順 (同点は行番号昇順) で返す。
    """
    idx = np.asarray(rows, dtype=np.int64)
    val = np.asarray(scores, dtype=np.float32)
    if idx.shape[0] != val.shape[0]:
        raise ValueError(f"rows ({idx.shape[0]}) and scores ({val.shape[0]}) length mismatch")
    if idx.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    flags = np.asarray(columns.flags, dtype=np.uint8)[idx]
    origin = np.asarray(columns.origin, dtype=np.uint8)[idx]

    keep = (flags & INACTIVE_FLAGS) == 0
    if not allow_assistant_origin:
        is_assistant = ((flags & FLAG_ASSISTANT) != 0) | (origin == int(Origin.ASSISTANT))
        keep &= ~is_assistant
    idx = idx[keep]
    val = val[keep]
    if idx.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    claim = np.asarray(columns.claim_hash64, dtype=np.uint64)[idx]
    has_claim = claim != np.uint64(0)

    survivors = idx[~has_claim]
    survivor_scores = val[~has_claim]

    if np.any(has_claim):
        c_idx = idx[has_claim]
        c_val = val[has_claim]
        c_claim = claim[has_claim]
        confidence = np.asarray(columns.confidence, dtype=np.float64)[c_idx]
        as_of = np.asarray(columns.as_of_epoch, dtype=np.float64)[c_idx]
        observed = np.asarray(columns.observed_epoch, dtype=np.float64)[c_idx]
        recency = np.where(np.isnan(as_of), observed, as_of)
        recency = np.where(np.isnan(recency), -np.inf, recency)
        rank_table = _origin_rank_table(origin_priority)
        origin_rank = rank_table[np.asarray(columns.origin, dtype=np.int64)[c_idx]]

        # lexsort は最後のキーが主キー。グループ内の先頭が勝者になる並びを作る。
        order = np.lexsort((c_idx, origin_rank, -recency, -confidence, c_claim))
        sorted_claim = c_claim[order]
        first = np.ones(sorted_claim.shape[0], dtype=bool)
        first[1:] = sorted_claim[1:] != sorted_claim[:-1]
        winners = order[first]
        survivors = np.concatenate([survivors, c_idx[winners]])
        survivor_scores = np.concatenate([survivor_scores, c_val[winners]])

    final_order = np.lexsort((survivors, -survivor_scores.astype(np.float64)))
    return (
        survivors[final_order].astype(np.int64, copy=False),
        survivor_scores[final_order].astype(np.float32, copy=False),
    )


# ------------------------------------------------------------ 候補のマージ


def merge_candidates(
    vector_rows: np.ndarray,
    vector_cos: np.ndarray,
    lexical_rows: np.ndarray,
    cos_for_rows: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """``ベクトル top-k ∪ 転置索引 top-k`` を作る (§6.3)。

    lexical だけで拾われた行は cosine を持っていないので、呼出側の
    ``cos_for_rows`` で **実際の cosine を計算させる**。lexical スコアを
    cosine の代わりに使ってはならない (§6.3: 候補の順位・ゲートは全て cosine)。

    Args:
        vector_rows: ベクトル検索が返した行番号。
        vector_cos: 同じ長さの cosine。
        lexical_rows: 転置索引が返した行番号 (スコアは受け取らない)。
        cos_for_rows: 行番号配列 → cosine 配列のコールバック。

    Returns:
        ``(rows int64, cos float64)``。行番号昇順で重複なし。
    """
    v_rows = np.asarray(vector_rows, dtype=np.int64)
    v_cos = np.asarray(vector_cos, dtype=np.float64)
    if v_rows.shape[0] != v_cos.shape[0]:
        raise ValueError(f"vector_rows ({v_rows.shape[0]}) and vector_cos ({v_cos.shape[0]}) mismatch")
    l_rows = np.unique(np.asarray(lexical_rows, dtype=np.int64))

    # ベクトル側で重複があれば先勝ちで畳む
    v_rows, first_pos = np.unique(v_rows, return_index=True)
    v_cos = v_cos[first_pos]

    only_lexical = l_rows[~np.isin(l_rows, v_rows)]
    if only_lexical.shape[0] == 0:
        return v_rows, v_cos

    extra_cos = np.asarray(cos_for_rows(only_lexical), dtype=np.float64)
    if extra_cos.shape[0] != only_lexical.shape[0]:
        raise ValueError(
            f"cos_for_rows returned {extra_cos.shape[0]} values for "
            f"{only_lexical.shape[0]} rows"
        )
    merged_rows = np.concatenate([v_rows, only_lexical])
    merged_cos = np.concatenate([v_cos, extra_cos])
    order = np.argsort(merged_rows, kind="stable")
    return merged_rows[order], merged_cos[order]
