"""判定点 (Predicate) の第一級化 — 字句・事例・補助タスクを 1 本の契約に束ねる。

このリポジトリの判定は 340 点あり、そのほとんどが *module-level の正規表現 +
``-> bool`` の関数* という形をしている。監査で繰り返し出てくる欠陥は 3 つで、
どれも個々の判定ロジックの誤りではなく **判定が持っていない性質** に起因する:

1. **同一性が無い** — 同じ判定が複数の実装を持つ。「訂正か」は 5 系統
   (``CORRECTION_PATTERNS`` / ``CORRECTION_FORM_TRIGGERS`` / ``WRONG_MARKER_RE`` /
   ``_CORRECTION_NOT_RE`` / ``fact_triggers.yaml``)、日本語文字の判定は 4 実装
   でうち 3 つは文字範囲が違う。片方だけ直すと不整合が残る。
2. **連続量が無い** — 出力が ``bool`` なので「自信が無い」を表現できない。
   結果として閾値も棄権も引けず、「決定論が全部外れたら LLM」という二値の
   カスケードしか組めない。
3. **記録が無い** — 何を根拠に何を選んだかが残らない。340 点のうち
   ``log_decision`` を出しているのは 6 点だけで、較正データも replay の材料も
   貯まらない。

本モジュールはこの 3 つを供給する。``Verdict`` が連続量 (``score``) と棄権
(``band``) と根拠 (``evidence``) を必ず持ち、``PredicateRegistry`` が名前の
一意性を強制し、``CascadePredicate`` が必ず ``log_decision`` を出す。

**置き場が ``core/`` である理由**: 判定点は 4 pillar すべてに散っており
(router は Loop、fact trigger は Mem、RAG 要否は Gen)、どの pillar にも属さない。
``intent_vocab.py`` / ``locale_patterns.py`` と同じ理由でここに置く。したがって
**本モジュールから pillar 配下を import してはいけない** — 依存は常に逆向き。

**既存コードとの関係**: ``LexicalPredicate`` は既存の述語を *そのまま包む* だけで
挙動を変えない。``ExemplarPredicate`` は :mod:`backend.free.agent.tool_gate_knn`
の一般化で、あちらが実ベンチ 137 件で示した結果 (正規表現 recall 66.2% →
kNN k=5 の LOO recall 98.5%) がこの形を採る根拠になっている。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from backend.log_config import get_logger

logger = get_logger("core.predicate")

__all__ = [
    "DISPUTED_EVIDENCE_PREFIX",
    "NEGATIVE_LABEL",
    "AuxPredicate",
    "Band",
    "CascadePredicate",
    "CascadePolicy",
    "Exemplar",
    "ExemplarPredicate",
    "LexicalPredicate",
    "Predicate",
    "PredicateNameConflict",
    "PredicateRegistry",
    "Verdict",
    "calibrate_exemplar_gate",
    "get_predicate",
    "register_predicate",
    "registered_predicates",
    "registry",
]

#: 「どのラベルでもない」を表す予約ラベル。exemplar ファイルの陰性例に使う。
#:
#: カナリア発話を別に持たない理由: 陰性例を同じファイルに入れておけば、
#: 近傍投票が自然に扱い、較正の null 側分布もそこから取れる。corpus の
#: ``CANARY_UTTERANCES`` のような外部リストを判定点ごとに用意する必要がない。
NEGATIVE_LABEL = "none"

#: 判定の帯域。``fire`` = 発火 / ``skip`` = 確信を持って不発 / ``abstain`` = 棄権。
#:
#: ``abstain`` を ``skip`` と分けるのが本設計の核心。従来の ``bool`` は両者を
#: 区別できないため、「判定できなかった」が「判定した結果 False」と同じ扱いに
#: なり、呼出側は縮退先を選べなかった。
Band = Literal["fire", "abstain", "skip"]

#: ``confirm`` 方針で字句と事例が食い違ったときの ``evidence`` 接頭辞。
#: 呼出側はこれを見て「規則は発火したが確認が取れなかった」を判別し、安全側へ
#: 縮退する (文字列を各所で書くと綴り違いで静かに壊れる)。
DISPUTED_EVIDENCE_PREFIX = "disputed:"


@dataclass(frozen=True, slots=True)
class Verdict:
    """1 判定点の結果。

    ``score`` は 0..1 の連続量で、字句判定は 1.0 / 0.0 の 2 値しか取らない
    (規則は「当たったか外れたか」しか言えない)。事例判定は近傍の得票率、
    補助タスク判定はスキーマが返した確信度を入れる。**この連続量があって初めて
    較正と棄権が引ける**。

    ``evidence`` は「何に当たったか」を人が読める 1 語で残す (正規表現の名前 /
    exemplar の id / 却下理由)。正規表現を捨てて事例に移すと判定が読めなくなる、
    という懸念への回答がこれで、``decision.jsonl`` に必ず載る。
    """

    #: 判定結果。二値なら ``True`` / ``False``、多値ならラベル文字列。
    #: 棄権のときは ``None``。
    value: bool | str | None
    #: 0..1 の確信度。
    score: float
    band: Band
    #: 根拠の短い識別子 (英語)。ユーザー入力をそのまま入れない (redaction 対象)。
    evidence: str = ""
    #: 判定点の名前 (``CascadePredicate`` が段ごとに埋める)。
    predicate: str = ""
    #: どの段が返したか (``lexical`` / ``exemplar`` / ``aux``)。
    stage: str = ""

    @property
    def fired(self) -> bool:
        """発火したか。棄権は ``False`` — 呼出側は ``band`` で縮退先を決める。"""
        return self.band == "fire"

    @property
    def decided(self) -> bool:
        """判定が付いたか (``fire`` または ``skip``)。"""
        return self.band != "abstain"

    def as_context(self) -> dict[str, Any]:
        """``DebugLogger.log_decision`` の ``context`` に載せる形。"""
        return {
            "predicate": self.predicate,
            "stage": self.stage,
            "band": self.band,
            "score": round(float(self.score), 4),
            "evidence": self.evidence,
            "value": self.value if isinstance(self.value, (bool, str)) else None,
        }


def _abstain(name: str, evidence: str, *, stage: str = "") -> Verdict:
    return Verdict(
        value=None, score=0.0, band="abstain",
        evidence=evidence, predicate=name, stage=stage,
    )


@runtime_checkable
class Predicate(Protocol):
    """同期で評価できる判定点 (字句・決定論)。"""

    name: str

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        """``text`` を判定する。例外を投げず、判定できなければ棄権を返す。"""
        ...


class LexicalPredicate:
    """既存の字句述語を **挙動を変えずに** 包む。

    ``fn`` は ``(text) -> bool`` か ``(text, ctx) -> bool | str | None``。真値
    (または ``NEGATIVE_LABEL`` 以外の文字列) を返したら ``fire``、偽値なら
    ``skip``、``None`` なら ``abstain`` にする。

    **``None`` を返す既存述語が棄権になる**のは意図的で、「判定できなかった」を
    ``False`` として扱ってきた経路を、呼出側が明示的に選び直せるようにする。
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        evidence: str = "",
        takes_ctx: bool = False,
    ) -> None:
        self.name = name
        self._fn = fn
        self._evidence = evidence or name
        self._takes_ctx = takes_ctx

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        try:
            raw = self._fn(text, ctx) if self._takes_ctx else self._fn(text)
        except Exception as e:  # pragma: no cover - 既存述語の例外を判定へ漏らさない
            logger.warning("Lexical predicate %s raised: %s", self.name, e)
            return _abstain(self.name, "raised", stage="lexical")
        if raw is None:
            return _abstain(self.name, "no_match", stage="lexical")
        if isinstance(raw, str):
            if not raw or raw == NEGATIVE_LABEL:
                return Verdict(
                    value=NEGATIVE_LABEL, score=0.0, band="skip",
                    evidence="no_match", predicate=self.name, stage="lexical",
                )
            return Verdict(
                value=raw, score=1.0, band="fire",
                evidence=self._evidence, predicate=self.name, stage="lexical",
            )
        fired = bool(raw)
        return Verdict(
            value=fired,
            score=1.0 if fired else 0.0,
            band="fire" if fired else "skip",
            evidence=self._evidence if fired else "no_match",
            predicate=self.name,
            stage="lexical",
        )


@dataclass(frozen=True, slots=True)
class Exemplar:
    """事例 1 件。``label`` が ``NEGATIVE_LABEL`` なら陰性例。"""

    text: str
    label: str
    #: 監査 ID / インシデント日付など。``evidence`` としてログに載る。
    id: str = ""

    @property
    def evidence_id(self) -> str:
        return self.id or f"{self.label}:{self.text[:24]}"


def load_exemplars(path: Path) -> list[Exemplar]:
    """JSONL から事例を読む。``{"text": ..., "label": ..., "id": ...}``。

    ``_comment`` キーを持つ行はラベル方針などの注記として読み飛ばす
    (``tool_gate_exemplars.jsonl`` と同じ約束)。壊れた行は 1 行ずつ飛ばし、
    件数を WARNING に出す (永続化規約 c_05 §0.5)。
    """
    if not path.exists():
        logger.warning("Exemplar file not found: %s", path)
        return []
    out: list[Exemplar] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            skipped += 1
            continue
        if not isinstance(rec, dict) or "_comment" in rec:
            continue
        text = rec.get("text") or rec.get("query")
        label = rec.get("label")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        if not isinstance(label, str) or not label:
            skipped += 1
            continue
        out.append(Exemplar(text.strip(), label, str(rec.get("id") or "")))
    if skipped:
        logger.warning("Exemplar file %s: skipped %d malformed records", path, skipped)
    return out


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


#: 較正の正側分位。``corpus/calibration.py`` と同じ値 — 正解の下位 5% を切る。
POSITIVE_QUANTILE = 0.05
#: 較正の null 側分位。陰性例の上位 5% を棒にする。
NULL_QUANTILE = 0.95


def calibrate_exemplar_gate(
    vectors: np.ndarray,
    labels: Sequence[str],
) -> dict[str, float | bool | int]:
    """事例集合だけから **類似度の床** を導く (ラベル無しの実発話は要らない)。

    2 側推定 (``corpus/calibration.py`` と同じ材料) を使う:

    - 正側 = 各行から **同じラベル**の最近傍への top1 の **p05**
      (自分自身は対角を落として除外する)。「票を入れてくれる仲間がどれくらい
      近いか」
    - null 側 = 各行から **違うラベル**の最近傍への top1 の **p95**。
      「最も近い誤答がどれくらい近いか」。陰性クラス (``none``) を持たない
      網羅選択のゲート (層振り分けのように「どれでもない」が存在しない) でも
      この定義なら縮退しない
    - 床 = **分離していれば中点** (谷の真ん中)、**重なっていれば正側 p05**

    .. warning::
       重なったときに ``corpus`` の ``pq_gate`` と **同じ式 (null 側を採る) に
       してはいけない**。両者は別の問題を解いている:

       - ``pq_gate`` は「この問いは corpus に関係があるか」で、索引には陽性
         クラスしか無い。棄権 = 注入しないだけなので、安全側 = 高い床が正しい。
       - 事例ゲートは陽性と陰性が **同じ索引にある**。判別の本体は「どちらの
         クラスに近いか」= 得票率で、床は「事例集合から極端に遠い入力」を弾く
         保険にすぎない。棄権 = ゲートが機能しないので実害が大きい。

       実測 (2026-09-15、bge-m3-q8_0)::

           fact_attribute_slot  正側 p05=0.707  null p95=0.861
           retrieval_skip       正側 p05=0.757  null p95=0.851
           tool_gate            正側 p05=0.676  null p95=0.813

       bge-m3 は自然文どうしの cosine が高く (無関係ペアの中央値 0.459)、
       **絶対類似度に判別力が無い**。null 側を床にすると床が正側分布の上端を
       超え、4 ゲートとも LOO の 95% が棄権した (fact_attribute_slot は 78/82)。
       偽陽性は床ではなく得票率 (``fire_ratio``) で落とす。

    床を下回る入力は、投票が割れていなくても ``abstain`` にする。「事例集合から
    遠い入力」は近傍法が最も間違えるところで、票率だけでは検出できない
    (遠くても k 件の近傍は必ず選ばれ、票は割れない)。

    Returns:
        ``{"ok", "sim_floor", "positive_p05", "null_p95", "overlap",
        "n_positive", "n_negative"}``。
    """
    mat = _l2_normalize(np.asarray(vectors, dtype=np.float32))
    lab = np.asarray([str(x) for x in labels])
    pos_rows = np.flatnonzero(lab != NEGATIVE_LABEL)
    neg_rows = np.flatnonzero(lab == NEGATIVE_LABEL)
    if pos_rows.size < 4:
        return {
            "ok": False, "sim_floor": 0.0, "positive_p05": 0.0, "null_p95": 0.0,
            "overlap": False, "n_positive": int(pos_rows.size),
            "n_negative": int(neg_rows.size),
        }
    # 全行どうしの類似度。同ラベル / 異ラベルの 2 つの分布をここから取る。
    sims = mat @ mat.T
    np.fill_diagonal(sims, -2.0)
    same = lab[:, None] == lab[None, :]

    # 正側 = **同じラベル**の最近傍 (それが票を入れてくれる相手)。ラベルに
    # 仲間が 1 件も無い行は分布に入れない (最近傍が存在しないため)。
    same_top1 = np.where(same, sims, -2.0).max(axis=1)
    pos_mask = (same_top1 > -2.0) & (lab != NEGATIVE_LABEL)
    positive_p05 = (
        float(np.quantile(same_top1[pos_mask], POSITIVE_QUANTILE))
        if pos_mask.any()
        else 0.0
    )

    # null 側 = **違うラベル**の最近傍 = 「最も近い誤答」。陰性クラス
    # (``none``) を持つゲートではその行が、持たない網羅選択のゲート
    # (層振り分けのように「どれでもない」が存在しない) では各クラスの行が
    # 互いにこの分布を作る。以前は ``none`` 行だけから取っていたため、
    # 陰性クラスの無いゲートで null が正側へ縮退していた。
    diff_top1 = np.where(same, -2.0, sims).max(axis=1)
    null_rows = diff_top1 > -2.0
    null_p95 = (
        float(np.quantile(diff_top1[null_rows], NULL_QUANTILE))
        if null_rows.any()
        else positive_p05
    )
    overlap = null_p95 >= positive_p05
    # 重なっているときに null 側を採ると、床が正側分布の上端を超えて自分の
    # 陽性例をほぼ全部棄権する (docstring の実測を参照)。谷が無いなら床は
    # 正側 p05 に置き、クラスの判別は得票率へ任せる。
    sim_floor = positive_p05 if overlap else (null_p95 + positive_p05) / 2.0
    sim_floor = float(min(1.0, max(0.0, sim_floor)))
    return {
        "ok": True,
        "sim_floor": round(sim_floor, 4),
        "positive_p05": round(positive_p05, 4),
        "null_p95": round(null_p95, 4),
        "overlap": bool(overlap),
        "n_positive": int(pos_rows.size),
        "n_negative": int(neg_rows.size),
    }


class ExemplarPredicate:
    """事例の近傍投票で判定する (:mod:`backend.free.agent.tool_gate_knn` の一般化)。

    あちらとの違いは 3 つだけで、設計はそのまま踏襲する:

    - **多値**: ラベルは任意の文字列 (属性スロット名など)。二値は
      ``{"<何か>", NEGATIVE_LABEL}`` の特殊形として扱う。
    - **棄権**: 得票率が ``fire_ratio`` 未満、または top1 類似度が較正で得た
      床を下回るときは ``abstain`` を返す。従来の多数決は必ずどちらかに倒れて
      いたため、precision を落とす方向の誤りを検出できなかった
      (ツール要否ゲートの実測: recall 66.2 → 98.5% と引き換えに
      precision 84.9 → 81.7%)。
    - **較正**: :func:`calibrate_exemplar_gate` を warmup 時に走らせ、床を
      埋め込みモデルごとに引き直す。静的閾値はモデル差し替えで必ず壊れる
      (bge-m3 への切替で注入閾値 0.35 が無関係ペアを通した実例)。

    埋め込みは warmup で 1 回だけ生成する。未 warmup / 次元不一致 / 埋め込み
    失敗はすべて ``abstain`` に倒し、**誤って閉じない**。
    """

    def __init__(
        self,
        name: str,
        embedder: Any,
        *,
        exemplars: Sequence[Exemplar],
        k: int = 5,
        mode: str = "chat",
        fire_ratio: float = 0.6,
        sim_floor: float | None = None,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._exemplars = list(exemplars)
        self._k = max(1, int(k))
        self._mode = mode
        self._fire_ratio = float(fire_ratio)
        self._sim_floor = sim_floor
        self._vectors: np.ndarray | None = None
        self._labels: list[str] = []
        self._ids: list[str] = []
        self._calibration: dict[str, float | bool | int] = {}

    @property
    def calibration(self) -> dict[str, float | bool | int]:
        """warmup で得た較正結果 (未 warmup なら空 dict)。"""
        return dict(self._calibration)

    @property
    def sim_floor(self) -> float:
        return float(self._sim_floor or 0.0)

    def is_ready(self) -> bool:
        return self._vectors is not None and len(self._labels) >= self._k

    def reset(self, embedder: Any = None) -> None:
        """事例ベクトルを捨て、再 :meth:`warmup` できる状態に戻す。

        埋め込みモデルが差し替わると、次元が同じでも幾何が別物になり、
        次元不一致の縮退にも掛からずに投票だけが狂う (``ToolGateKNN.reset``
        と同じ理由)。較正値も同時に捨てる。
        """
        self._vectors = None
        self._labels = []
        self._ids = []
        self._calibration = {}
        if embedder is not None:
            self._embedder = embedder

    async def warmup(self) -> bool:
        """事例を埋め込み、較正する。成功で ``True``。例外は投げない。"""
        if self._vectors is not None:
            return True
        if self._embedder is None or not self._exemplars:
            logger.info(
                "Exemplar predicate %s not warmed up: embedder=%s exemplars=%d",
                self.name, self._embedder is not None, len(self._exemplars),
            )
            return False
        try:
            vecs = await self._embedder.embed(
                [e.text for e in self._exemplars], is_query=True, mode=self._mode,
            )
            self._vectors = _l2_normalize(np.asarray(vecs, dtype=np.float32))
            self._labels = [e.label for e in self._exemplars]
            self._ids = [e.evidence_id for e in self._exemplars]
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.warning("Exemplar predicate %s warmup failed: %s", self.name, e)
            return False
        self._calibration = calibrate_exemplar_gate(self._vectors, self._labels)
        if self._sim_floor is None and self._calibration.get("ok"):
            self._sim_floor = float(self._calibration["sim_floor"])
        self._warn_unreachable_labels()
        logger.info(
            "Exemplar predicate %s ready: %d exemplars, k=%d, sim_floor=%.4f, cal=%s",
            self.name, len(self._labels), self._k, self.sim_floor, self._calibration,
        )
        return True

    def _warn_unreachable_labels(self) -> None:
        """``k × fire_ratio`` 票に **構造的に届かない** ラベルを警告する。

        多値ゲートで見落としやすい罠: 発火には ``ceil(k × fire_ratio)`` 票が
        要るので、事例がそれ未満のラベルは **何を入力しても勝てない**。近傍は
        必ず k 件選ばれるため、票が割れて棄権になるだけで、エラーにも
        ならない。

        実例 (2026-09-15): ``fact_attribute_slot`` を k=7 / ratio=0.6 (4 票必要)
        にしたところ、18 スロット中 15 が事例 3 件以下で発火不能になり、実発話
        928 件に対して **1 度も発火しなかった**。LOO の正解率 0.944 は、ほぼ
        陰性クラス (28 件) だけを測っていた。

        直し方は 2 つで、どちらも実測で決める: ``k`` を下げるか、そのラベルの
        事例を増やす。
        """
        import math

        needed = math.ceil(self._k * self._fire_ratio)
        counts: dict[str, int] = {}
        for label in self._labels:
            counts[label] = counts.get(label, 0) + 1
        unreachable = sorted(
            (label, n) for label, n in counts.items() if n < needed
        )
        if not unreachable:
            return
        logger.warning(
            "Exemplar predicate %s: %d/%d labels cannot reach %d votes "
            "(k=%d x fire_ratio=%.2f) and will never fire: %s. "
            "Lower k or add exemplars.",
            self.name, len(unreachable), len(counts), needed,
            self._k, self._fire_ratio,
            ", ".join(f"{label}={n}" for label, n in unreachable[:12]),
        )

    def unreachable_labels(self) -> dict[str, int]:
        """``k × fire_ratio`` 票に届かないラベル (``{ラベル: 事例数}``)。

        ベンチとテストが同じ判定を使えるように公開する。
        """
        import math

        needed = math.ceil(self._k * self._fire_ratio)
        counts: dict[str, int] = {}
        for label in self._labels:
            counts[label] = counts.get(label, 0) + 1
        return {label: n for label, n in counts.items() if n < needed}

    async def aevaluate(
        self,
        text: str,
        ctx: Mapping[str, Any] | None = None,  # noqa: ARG002 - 段共通の面を保つ
        *,
        query_vec: np.ndarray | None = None,
    ) -> Verdict:
        """近傍投票で判定する。判定できなければ ``abstain``。

        ``ctx`` は使わないが、字句段・補助タスク段と **同じ面** にしておく
        (``CascadePredicate`` が段を差し替えて呼べるのはこのため)。

        ``query_vec`` を渡すと埋め込みの往復を **一切しない**。呼出側が既に
        同じクエリを埋め込んでいる経路 (検索パイプラインは ``query_vec`` を
        引数で受け取る) では、この形にしないとレイテンシが増える。L2 正規化は
        こちらで行うので、呼出側は素のベクトルをそのまま渡してよい。
        """
        if not self.is_ready() or not text.strip():
            return _abstain(self.name, "not_ready", stage="exemplar")
        if query_vec is not None:
            q = np.asarray(query_vec, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(q))
            if norm == 0.0:
                return _abstain(self.name, "zero_vector", stage="exemplar")
            q = q / norm
        else:
            try:
                qv = await self._embedder.embed_query(text, mode=self._mode)
                q = np.asarray(qv, dtype=np.float32)
                norm = float(np.linalg.norm(q))
                if norm == 0.0:
                    return _abstain(self.name, "zero_vector", stage="exemplar")
                q = q / norm
            except Exception as e:
                logger.info("Exemplar predicate %s embed failed: %s", self.name, e)
                return _abstain(self.name, "embed_failed", stage="exemplar")

        assert self._vectors is not None
        if self._vectors.shape[1] != q.shape[0]:
            logger.warning(
                "Exemplar predicate %s dim mismatch (exemplars=%d, query=%d)",
                self.name, self._vectors.shape[1], q.shape[0],
            )
            return _abstain(self.name, "dim_mismatch", stage="exemplar")

        sims = self._vectors @ q
        k = min(self._k, sims.shape[0])
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(-sims[idx])]
        top1 = float(sims[idx[0]])

        votes: dict[str, int] = {}
        for i in idx:
            votes[self._labels[int(i)]] = votes.get(self._labels[int(i)], 0) + 1
        winner = max(votes, key=lambda lab: (votes[lab], lab != NEGATIVE_LABEL))
        share = votes[winner] / float(k)

        if top1 < self.sim_floor:
            return Verdict(
                value=None, score=round(share, 4), band="abstain",
                evidence=f"below_floor:{top1:.3f}",
                predicate=self.name, stage="exemplar",
            )
        if share < self._fire_ratio:
            return Verdict(
                value=None, score=round(share, 4), band="abstain",
                evidence=f"split:{votes[winner]}/{k}",
                predicate=self.name, stage="exemplar",
            )
        evidence = self._ids[int(idx[0])]
        if winner == NEGATIVE_LABEL:
            return Verdict(
                value=NEGATIVE_LABEL, score=round(share, 4), band="skip",
                evidence=evidence, predicate=self.name, stage="exemplar",
            )
        return Verdict(
            value=winner, score=round(share, 4), band="fire",
            evidence=evidence, predicate=self.name, stage="exemplar",
        )

    def leave_one_out(self, *, with_errors: bool = False) -> dict[str, Any]:
        """warmup 済み事例集合に対する LOO 精度 (オフライン評価用)。

        自分自身を除いた近傍投票で各事例を判定し、正解率・棄権率を返す。
        事例を足したときにその場で効果を測るための口で、LLM も外部データも
        要らない (``bench_tool_gate.py`` が 137 件でやっていることの一般形)。

        ``with_errors=True`` で誤分類の中身 (本文 / 期待 / 予測 / 得票率 /
        top1) を ``errors`` に入れる。**どのラベルがどのラベルへ流れたかが
        分からないと「事例を足す」判断が勘になる**。
        """
        if self._vectors is None or not self._labels:
            return {"ok": 0, "n": 0}
        mat = self._vectors
        sims = mat @ mat.T
        np.fill_diagonal(sims, -2.0)
        k = min(self._k, mat.shape[0] - 1)
        correct = abstained = 0
        errors: list[dict[str, Any]] = []
        for row in range(mat.shape[0]):
            idx = np.argpartition(sims[row], -k)[-k:]
            top1 = float(sims[row][idx[np.argmax(sims[row][idx])]])
            votes: dict[str, int] = {}
            for i in idx:
                votes[self._labels[int(i)]] = votes.get(self._labels[int(i)], 0) + 1
            winner = max(votes, key=lambda lab: (votes[lab], lab != NEGATIVE_LABEL))
            share = votes[winner] / float(k)
            if top1 < self.sim_floor or share < self._fire_ratio:
                abstained += 1
                continue
            if winner == self._labels[row]:
                correct += 1
            elif with_errors:
                errors.append({
                    "text": self._exemplars[row].text,
                    "expected": self._labels[row],
                    "predicted": winner,
                    "share": round(share, 3),
                    "top1": round(top1, 3),
                })
        n = mat.shape[0]
        decided = n - abstained
        out: dict[str, Any] = {
            "ok": 1,
            "n": n,
            "decided": decided,
            "abstained": abstained,
            "correct": correct,
            "accuracy_decided": round(correct / decided, 4) if decided else 0.0,
            "accuracy_all": round(correct / n, 4) if n else 0.0,
        }
        if with_errors:
            # 誤分類の中身が無いと「事例を足す」判断が勘になる。どのラベルが
            # どのラベルへ流れたか (混同) を呼出側で数えられるようにする。
            out["errors"] = errors
        return out


class AuxPredicate:
    """棄権帯だけを補助タスク (制約付き JSON) に上げる段。

    **``abstain`` からしか呼ばれない**のが契約。字句や事例で決まった判定に
    LLM を掛け直すことはしない — 遅く (専用スロットでも 10.2 秒、層 5.9 実測
    34〜39 秒)、決定的でなく (同一 prompt 同士で採点が ±2 振れた実測がある)、
    背景併走でチャットの decode を 2 倍遅くするため。

    ``aux_client`` が ``None`` (起動失敗 = degraded) のときは棄権を返し、
    呼出側が従来の答えへ縮退する。
    """

    def __init__(
        self,
        name: str,
        aux_client: Any,
        *,
        purpose: str,
        build_prompt: Callable[[str, Mapping[str, Any] | None], str],
        parse: Callable[[dict], Verdict | None],
        max_tokens: int = 96,
    ) -> None:
        self.name = name
        self._aux = aux_client
        self._purpose = purpose
        self._build_prompt = build_prompt
        self._parse = parse
        self._max_tokens = max_tokens

    async def aevaluate(
        self, text: str, ctx: Mapping[str, Any] | None = None,
    ) -> Verdict:
        if self._aux is None:
            return _abstain(self.name, "aux_unavailable", stage="aux")
        try:
            raw = await self._aux.generate_json(
                self._build_prompt(text, ctx),
                purpose=self._purpose,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            logger.info("Aux predicate %s failed: %s", self.name, e)
            return _abstain(self.name, "aux_failed", stage="aux")
        if not raw:
            return _abstain(self.name, "aux_empty", stage="aux")
        verdict = self._parse(raw)
        if verdict is None:
            return _abstain(self.name, "aux_unparsed", stage="aux")
        return Verdict(
            value=verdict.value, score=verdict.score, band=verdict.band,
            evidence=verdict.evidence or "aux", predicate=self.name, stage="aux",
        )


#: カスケードの合成方針。**判定点ごとに誤りのコストが逆向き**なので 1 つに
#: 決められない。
#:
#: - ``complement``: 字句が不発のときだけ事例に聞く (**recall の穴を埋める**)。
#:   字句の発火は上書きしない。fact trigger のように「当たらないと記憶が 1 件も
#:   残らない」fail-closed な判定点向け。
#: - ``confirm``: 字句が発火したとき事例に確認させ、反対されたら棄権へ倒す
#:   (**precision を守る**)。RAG の skip のように「誤発火が最も重い」判定点向け。
#: - ``shadow``: 常に両方を評価するが **返すのは必ず字句の結果**。不一致だけを
#:   記録する。凍結領域 (``router`` は ``EVOLVABLE_DOMAINS`` から意図的に除外
#:   されている) を測るための方針。
CascadePolicy = Literal["complement", "confirm", "shadow"]


class CascadePredicate:
    """字句 → 事例 → 補助タスクを 1 つの判定点として束ね、必ず記録する。

    ``evaluate`` (同期) は **字句段だけ** を評価する。既存の同期呼出を壊さず、
    かつチャット応答パスで待たせないため (不変則 #1)。事例段と補助タスク段が
    要るときは ``aevaluate`` を使う。

    .. warning::
       各段は **同じラベル語彙** を返すこと。字句段が ``bool`` を返し事例段が
       ラベル文字列を返すと、:func:`_same_decision` が ``"True"`` と
       ``"skip"`` を比べて常に不一致になり、``confirm`` 方針では正しい
       ターンまで棄権へ降りる。二値の判定点でもラベルを揃えるほうが安全。
    """

    def __init__(
        self,
        name: str,
        *,
        lexical: Predicate | None = None,
        exemplar: ExemplarPredicate | None = None,
        aux: AuxPredicate | None = None,
        policy: CascadePolicy = "complement",
        debug_logger: Any = None,
        candidates: Sequence[str] = (),
        scope: str = "request",
    ) -> None:
        self.name = name
        self.policy: CascadePolicy = policy
        self._lexical = lexical
        self._exemplar = exemplar
        self._aux = aux
        self._debug_logger = debug_logger
        self._candidates = list(candidates)
        self._scope = scope

    @property
    def exemplar(self) -> ExemplarPredicate | None:
        return self._exemplar

    def bind_debug_logger(self, debug_logger: Any) -> None:
        """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
        self._debug_logger = debug_logger

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        """字句段だけで判定する (同期・決定論)。"""
        if self._lexical is None:
            return _abstain(self.name, "no_lexical", stage="lexical")
        verdict = self._lexical.evaluate(text, ctx)
        verdict = self._relabel(verdict, stage="lexical")
        self._log(verdict, agreed=None)
        return verdict

    async def aevaluate(
        self,
        text: str,
        ctx: Mapping[str, Any] | None = None,
        *,
        query_vec: np.ndarray | None = None,
    ) -> Verdict:
        """方針にしたがって段を進める。

        ``query_vec`` は事例段へそのまま渡す (呼出側が既に埋め込んでいる
        クエリで、追加の往復を避けたいとき)。
        """
        verdict, _shadow = await self.aevaluate_pair(text, ctx, query_vec=query_vec)
        return verdict

    async def aevaluate_pair(
        self,
        text: str,
        ctx: Mapping[str, Any] | None = None,
        *,
        query_vec: np.ndarray | None = None,
    ) -> tuple[Verdict, Verdict | None]:
        """採択した判定と **事例段の判定** を返す。

        ``shadow`` 方針の呼出側は、採択された規則の結果と並べて事例の結果を
        見たい。同じ入力で 2 度評価すると行列積も埋め込みキャッシュ参照も
        二重に走るので、1 回の評価から両方を返す。事例段まで到達しなかった
        ときは第 2 要素が ``None``。
        """
        lex = (
            self._relabel(self._lexical.evaluate(text, ctx), stage="lexical")
            if self._lexical is not None
            else _abstain(self.name, "no_lexical", stage="lexical")
        )
        if self._exemplar is None:
            self._log(lex, agreed=None)
            return lex, None

        if self.policy == "complement" and lex.band == "fire":
            self._log(lex, agreed=None)
            return lex, None
        if self.policy == "confirm" and lex.band == "abstain":
            self._log(lex, agreed=None)
            return lex, None

        ex = self._relabel(
            await self._exemplar.aevaluate(text, ctx, query_vec=query_vec),
            stage="exemplar",
        )
        agreed = _same_decision(lex, ex)

        if self.policy == "shadow":
            self._log(lex, agreed=agreed, shadow=ex)
            return lex, ex

        if self.policy == "confirm":
            if ex.band == "abstain" or agreed:
                self._log(lex, agreed=agreed, shadow=ex)
                return lex, ex
            # 字句は発火したが事例が反対した → 棄権へ倒す。呼出側は安全側
            # (RAG なら skip をやめて uncertain) へ縮退する。
            out = Verdict(
                value=None, score=ex.score, band="abstain",
                evidence=f"{DISPUTED_EVIDENCE_PREFIX}{ex.evidence}",
                predicate=self.name,
                stage="exemplar",
            )
            self._log(out, agreed=False, shadow=ex)
            return out, ex

        # complement: 字句が不発。事例が決めれば採り、棄権なら補助タスクへ。
        if ex.decided:
            self._log(ex, agreed=agreed)
            return ex, ex
        if self._aux is None:
            self._log(lex, agreed=agreed, shadow=ex)
            return lex, ex
        aux = self._relabel(await self._aux.aevaluate(text, ctx), stage="aux")
        out = aux if aux.decided else lex
        self._log(out, agreed=agreed, shadow=ex)
        return out, ex

    def _relabel(self, verdict: Verdict, *, stage: str) -> Verdict:
        return Verdict(
            value=verdict.value, score=verdict.score, band=verdict.band,
            evidence=verdict.evidence, predicate=self.name,
            stage=verdict.stage or stage,
        )

    def _log(
        self,
        verdict: Verdict,
        *,
        agreed: bool | None,
        shadow: Verdict | None = None,
    ) -> None:
        if self._debug_logger is None:
            return
        context = verdict.as_context()
        context["policy"] = self.policy
        if agreed is not None:
            context["stages_agreed"] = agreed
        if shadow is not None:
            context["shadow_band"] = shadow.band
            context["shadow_value"] = (
                shadow.value if isinstance(shadow.value, (bool, str)) else None
            )
            context["shadow_score"] = round(float(shadow.score), 4)
            context["shadow_evidence"] = shadow.evidence
        try:
            self._debug_logger.log_decision(
                decision_point=self.name,
                chosen=str(verdict.value),
                candidates=self._candidates,
                reason=verdict.evidence,
                context=context,
                scope=self._scope,
            )
        except Exception as e:  # pragma: no cover - ログで判定を落とさない
            logger.debug("Predicate %s decision log failed: %s", self.name, e)


def _same_decision(a: Verdict, b: Verdict) -> bool:
    """2 段の結論が一致しているか (棄権どうしは一致とみなさない)。"""
    if not a.decided or not b.decided:
        return False
    if a.band != b.band:
        return False
    if a.band == "skip":
        return True
    return str(a.value) == str(b.value)


class PredicateNameConflict(RuntimeError):
    """同じ名前の判定点が二重に登録された。

    「同じ判定の二重実装」はこのリポジトリで最も再発している欠陥なので、
    レジストリは**黙って上書きしない**。意図的に差し替えるときは
    :meth:`PredicateRegistry.replace` を使う。
    """


class PredicateRegistry:
    """名前 → 判定点。名前の一意性を強制する。

    解析器 (``scripts/analyze_predicates.py``) はここを走査して発火ビットマップを
    作るので、**登録した判定点だけが重複検査・恒真検査の対象になる**。
    """

    def __init__(self) -> None:
        self._by_name: dict[str, Any] = {}

    def register(self, predicate: Any) -> Any:
        name = getattr(predicate, "name", "")
        if not name:
            raise ValueError("predicate must have a non-empty name")
        if name in self._by_name and self._by_name[name] is not predicate:
            raise PredicateNameConflict(
                f"predicate {name!r} is already registered "
                f"({type(self._by_name[name]).__name__}); "
                "use replace() if the substitution is intentional",
            )
        self._by_name[name] = predicate
        return predicate

    def replace(self, predicate: Any) -> Any:
        self._by_name[getattr(predicate, "name", "")] = predicate
        return predicate

    def get(self, name: str) -> Any | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def clear(self) -> None:
        """テスト用。本番経路からは呼ばない。"""
        self._by_name.clear()


#: プロセス共通のレジストリ。
registry = PredicateRegistry()


def register_predicate(predicate: Any) -> Any:
    """モジュール読み込み時に判定点を登録する。"""
    return registry.register(predicate)


def get_predicate(name: str) -> Any | None:
    return registry.get(name)


def registered_predicates() -> list[Any]:
    return list(registry)


@dataclass(slots=True)
class FiringMatrix:
    """発火ビットマップ (解析器の中間表現)。

    行 = 判定点、列 = 発話。``bits[i, j]`` は判定点 i が発話 j で発火したか。
    """

    names: list[str]
    bits: np.ndarray
    n_texts: int
    abstained: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    def jaccard(self) -> np.ndarray:
        """判定点どうしの Jaccard 類似度行列。"""
        b = self.bits.astype(np.float32)
        inter = b @ b.T
        counts = b.sum(axis=1)
        union = counts[:, None] + counts[None, :] - inter
        union[union == 0] = 1.0
        return (inter / union).astype(np.float32)

    def containment(self) -> np.ndarray:
        """``c[i, j]`` = 判定点 i の発火集合が j に含まれる割合。

        1.0 なら i ⊆ j で、i は j の部分集合 = 冗長かもしれない。
        """
        b = self.bits.astype(np.float32)
        inter = b @ b.T
        counts = b.sum(axis=1)
        counts[counts == 0] = 1.0
        return (inter / counts[:, None]).astype(np.float32)
