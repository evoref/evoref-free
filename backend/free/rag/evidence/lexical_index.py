"""転置索引 (numpy CSR + impact 順 posting) — ``rank-bm25`` の置換 (c_16 §6.2)

``rank_bm25.BM25Plus`` は純 Python でクエリ語ごとに **全文書** を走査するため、
走査量が文書数 N に比例する。本モジュールは同じ BM25+ 寄与を **構築時に前計算** し、
語ごとの posting を寄与 (impact) 降順に並べた CSR 3 配列
(``indptr`` / ``indices`` / ``impacts``) で持つ。クエリ時は

- IDF 上位 ``Q`` 語だけ使う
- 各語の posting 先頭 ``M`` 件だけ読む

の 2 つで走査量を **``Q × M`` に構造的に固定** する (N に依存しない)。
証明用のヘルパは :func:`max_postings_scanned`。

トークナイザは :mod:`backend.free.rag.evidence.tokenize` の
:func:`~backend.free.rag.evidence.tokenize.tokenize_ja` (文字 bi-gram + ASCII 語分割 +
ストップワード) をそのまま流用する。索引側と検索側で切り方がずれると語が一致
しなくなるため、切り方のパラメータは索引に保存し :meth:`LexicalIndex.load` で
復元する。

**スコアの位置付け (c_16 §6.3)**: 本モジュールが返すスコアは
**候補生成の順序付け専用** であって、順位付けには使わない。候補のゲートと順位は
すべて素の cosine から計算する (:mod:`backend.free.rag.evidence.ranking`)。
lexical スコアと cosine を足したり混ぜたりしてはならない。
"""

from __future__ import annotations

import io
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.free.rag.evidence.tokenize import DEFAULT_STOPWORD_BIGRAMS, tokenize_ja
from backend.io import atomic_write_bytes, atomic_write_text
from backend.log_config import get_logger

logger = get_logger("rag.evidence.lexical_index")


#: 索引ファイルの形式版。配列の意味を変える場合に上げる。
LEXICAL_INDEX_VERSION: int = 1

#: :meth:`LexicalIndex.candidates` の疎経路へ切り替える文書数。
#: これ以下なら ``n_docs`` 長の密アキュムレータ (``np.bincount``) を確保する方が速い。
DENSE_ACCUMULATOR_MAX_DOCS: int = 200_000

_INDEX_NPZ_NAME = "index.npz"
_META_JSON_NAME = "meta.json"


@dataclass(frozen=True)
class LexicalParams:
    """索引の構築・検索パラメータ (c_16 §6.2 の表)。

    Attributes:
        q_terms: クエリ語のうち IDF 上位何語を使うか (``Q``)。
        m_postings: 各語の posting を先頭何件まで読むか (``M``)。
        max_df_ratio: ``df > N × 比率`` の語を索引から除外する **相対** 閾値。
            絶対件数にしないのは、コーパスの大きさを替えると意味が変わり
            到達不能な閾値になる事故を繰り返しているため。
        k1 / b / delta: BM25+ のパラメータ。
        use_trigrams / split_ascii / stopwords: トークナイザ設定。索引と
            クエリで必ず一致させる。
    """

    q_terms: int = 32
    m_postings: int = 2000
    max_df_ratio: float = 0.10
    k1: float = 1.2
    b: float = 0.75
    delta: float = 1.0
    use_trigrams: bool = False
    split_ascii: bool = True
    stopwords: frozenset[str] = DEFAULT_STOPWORD_BIGRAMS

    def to_json(self) -> dict[str, object]:
        """JSON 化 (``stopwords`` は決定論的に並べる)。"""
        return {
            "q_terms": self.q_terms,
            "m_postings": self.m_postings,
            "max_df_ratio": self.max_df_ratio,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "use_trigrams": self.use_trigrams,
            "split_ascii": self.split_ascii,
            "stopwords": sorted(self.stopwords),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "LexicalParams":
        """:meth:`to_json` の逆変換。未知キーは無視する。"""
        return cls(
            q_terms=int(data.get("q_terms", 32)),  # type: ignore[arg-type]
            m_postings=int(data.get("m_postings", 2000)),  # type: ignore[arg-type]
            max_df_ratio=float(data.get("max_df_ratio", 0.10)),  # type: ignore[arg-type]
            k1=float(data.get("k1", 1.2)),  # type: ignore[arg-type]
            b=float(data.get("b", 0.75)),  # type: ignore[arg-type]
            delta=float(data.get("delta", 1.0)),  # type: ignore[arg-type]
            use_trigrams=bool(data.get("use_trigrams", False)),
            split_ascii=bool(data.get("split_ascii", True)),
            stopwords=frozenset(data.get("stopwords") or ()),  # type: ignore[arg-type]
        )

    def tokenize(self, text: str) -> list[str]:
        """索引・クエリ共通のトークナイズ。"""
        return tokenize_ja(
            text,
            use_trigrams=self.use_trigrams,
            split_ascii=self.split_ascii,
            stopwords=self.stopwords if self.stopwords else None,
        )


class LexicalIndex:
    """impact 降順 posting を持つ CSR 転置索引。

    行番号は **snapshot の行位置** (``records.jsonl`` の行 index) と一対一。
    id 文字列は持たない (行 → id は ``columns.npz`` の ``ids`` 側の責務)。

    Attributes:
        indptr: ``int64`` (V+1)。語 ``t`` の posting は ``[indptr[t], indptr[t+1])``。
        indices: ``int32`` (nnz)。posting の行番号。impact 降順。
        impacts: ``float32`` (nnz)。BM25+ の (語, 文書) 寄与。
        idf: ``float32`` (V)。クエリ語の選抜に使う。
        vocab: 語 → term_id。剪定後に残った語だけを含む。
        n_docs: 索引時の文書数 N。
        params: 構築・検索パラメータ。
    """

    def __init__(
        self,
        *,
        indptr: np.ndarray,
        indices: np.ndarray,
        impacts: np.ndarray,
        idf: np.ndarray,
        vocab: dict[str, int],
        n_docs: int,
        params: LexicalParams,
    ) -> None:
        self.indptr: np.ndarray = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices: np.ndarray = np.ascontiguousarray(indices, dtype=np.int32)
        self.impacts: np.ndarray = np.ascontiguousarray(impacts, dtype=np.float32)
        self.idf: np.ndarray = np.ascontiguousarray(idf, dtype=np.float32)
        self.vocab: dict[str, int] = vocab
        self.n_docs: int = int(n_docs)
        self.params: LexicalParams = params

    # ------------------------------------------------------------------ 基本

    @property
    def n_terms(self) -> int:
        """索引に残っている語数 (剪定後)。"""
        return len(self.vocab)

    @property
    def nnz(self) -> int:
        """posting の総数。"""
        return int(self.indices.shape[0])

    def posting_len(self, term_id: int) -> int:
        """語 ``term_id`` の posting 長。"""
        return int(self.indptr[term_id + 1] - self.indptr[term_id])

    # ------------------------------------------------------------- クエリ語

    def query_term_ids(self, query: str) -> np.ndarray:
        """クエリを語 id 列に変換し、IDF 上位 ``Q`` 語へ絞る。

        重複語は 1 回だけ数える (クエリ側の tf は使わない)。同点は term_id
        昇順で決定論的に選ぶ。索引に無い語 (剪定済み / 未知語) は落とす。
        """
        seen: dict[int, None] = {}
        for token in self.params.tokenize(query):
            tid = self.vocab.get(token)
            if tid is not None:
                seen[tid] = None
        if not seen:
            return np.empty(0, dtype=np.int64)
        tids = np.fromiter(seen.keys(), dtype=np.int64, count=len(seen))
        # IDF 降順 → term_id 昇順 (lexsort は最後のキーが主キー)
        order = np.lexsort((tids, -self.idf[tids]))
        return tids[order][: self.params.q_terms]

    # --------------------------------------------------------------- 検索

    def candidates(
        self,
        query: str,
        top_k: int,
        *,
        budget_ms: float | None = 20.0,
        row_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """候補行を返す。

        Args:
            query: 生のクエリ文字列。
            top_k: 返す最大件数。
            budget_ms: 走査に使ってよい時間 (ms)。``None`` で無制限。
                超過した時点で **読めた分だけ返す** (degrade)。例外は投げない。
            row_mask: 行番号と同じ長さの bool 配列。``False`` の行を除外する。

        Returns:
            ``(row_indices int64, scores float32)``。スコア降順、同点は行番号昇順。
            スコアは **候補の並べ替え専用** で、cosine と混ぜてはならない (c_16 §6.3)。
        """
        started = time.perf_counter()
        term_ids = self.query_term_ids(query)
        row_parts: list[np.ndarray] = []
        impact_parts: list[np.ndarray] = []
        scanned = 0
        degraded = False

        for tid in term_ids:
            if budget_ms is not None:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if elapsed_ms > budget_ms:
                    degraded = True
                    break
            start = int(self.indptr[tid])
            end = min(int(self.indptr[tid + 1]), start + self.params.m_postings)
            if end <= start:
                continue
            row_parts.append(self.indices[start:end])
            impact_parts.append(self.impacts[start:end])
            scanned += end - start

        if degraded:
            logger.debug(
                "lexical candidates degraded: budget_ms=%.1f terms_used=%d/%d scanned=%d",
                budget_ms if budget_ms is not None else -1.0,
                len(row_parts), int(term_ids.shape[0]), scanned,
            )

        if not row_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        all_rows = np.concatenate(row_parts).astype(np.int64, copy=False)
        all_impacts = np.concatenate(impact_parts).astype(np.float32, copy=False)

        if self.n_docs > DENSE_ACCUMULATOR_MAX_DOCS:
            rows, scores = self._accumulate_sparse(all_rows, all_impacts)
        else:
            rows, scores = self._accumulate_dense(all_rows, all_impacts)

        if row_mask is not None:
            keep = row_mask[rows]
            rows = rows[keep]
            scores = scores[keep]

        if rows.shape[0] == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        return _top_k_desc(rows, scores, top_k)

    def _accumulate_dense(
        self, all_rows: np.ndarray, all_impacts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """密アキュムレータ経路。``O(N)`` メモリだが加算は 1 回。"""
        acc = np.bincount(
            all_rows, weights=all_impacts.astype(np.float64, copy=False),
            minlength=self.n_docs,
        )
        rows = np.flatnonzero(acc)
        return rows.astype(np.int64, copy=False), acc[rows].astype(np.float32, copy=False)

    @staticmethod
    def _accumulate_sparse(
        all_rows: np.ndarray, all_impacts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """疎経路。``n_docs`` 長の確保を避け、出現行だけで畳む。"""
        uniq, inverse = np.unique(all_rows, return_inverse=True)
        sums = np.bincount(
            inverse, weights=all_impacts.astype(np.float64, copy=False),
            minlength=uniq.shape[0],
        )
        return uniq.astype(np.int64, copy=False), sums.astype(np.float32, copy=False)

    # ------------------------------------------------------------ 永続化

    def save(self, directory: Path | str) -> None:
        """``index.npz`` + ``meta.json`` を原子的に書く。"""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            indptr=self.indptr,
            indices=self.indices,
            impacts=self.impacts,
            idf=self.idf,
        )
        atomic_write_bytes(target / _INDEX_NPZ_NAME, buffer.getvalue())
        meta = {
            "lexical_version": LEXICAL_INDEX_VERSION,
            "n_docs": self.n_docs,
            "params": self.params.to_json(),
            "vocab": self.vocab,
        }
        atomic_write_text(
            target / _META_JSON_NAME,
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
        )

    @classmethod
    def load(cls, directory: Path | str) -> "LexicalIndex":
        """:meth:`save` の逆。形式版が新しい場合は読まずに落とす。"""
        source = Path(directory)
        meta = json.loads((source / _META_JSON_NAME).read_text(encoding="utf-8"))
        version = int(meta.get("lexical_version", 0))
        if version > LEXICAL_INDEX_VERSION:
            raise ValueError(
                f"lexical index version {version} is newer than supported "
                f"{LEXICAL_INDEX_VERSION}: {source}"
            )
        with np.load(source / _INDEX_NPZ_NAME) as arrays:
            indptr = arrays["indptr"]
            indices = arrays["indices"]
            impacts = arrays["impacts"]
            idf = arrays["idf"]
        return cls(
            indptr=indptr,
            indices=indices,
            impacts=impacts,
            idf=idf,
            vocab={str(k): int(v) for k, v in meta["vocab"].items()},
            n_docs=int(meta["n_docs"]),
            params=LexicalParams.from_json(meta.get("params") or {}),
        )


class LexicalIndexBuilder:
    """行位置で索引される転置索引を組む (snapshot 生成時に 1 回だけ走る)。

    IDF・文書長正規化・BM25+ 寄与をすべて構築時に前計算し、posting を寄与降順に
    並べる。クエリ時に残る仕事は「先頭 M 件を読んで足す」だけになる。
    """

    def __init__(
        self,
        *,
        q_terms: int = 32,
        m_postings: int = 2000,
        max_df_ratio: float = 0.10,
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0,
        use_trigrams: bool = False,
        split_ascii: bool = True,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.params = LexicalParams(
            q_terms=q_terms,
            m_postings=m_postings,
            max_df_ratio=max_df_ratio,
            k1=k1,
            b=b,
            delta=delta,
            use_trigrams=use_trigrams,
            split_ascii=split_ascii,
            stopwords=(
                DEFAULT_STOPWORD_BIGRAMS if stopwords is None else frozenset(stopwords)
            ),
        )

    def build(self, texts: Sequence[str]) -> LexicalIndex:
        """``texts`` を行位置 (0..N-1) で索引する。

        剪定は **相対**: ``df > max(1, max_df_ratio × N)`` の語を落とす。
        文書が極端に少ないときに全語が消えないよう下限 1 を置く。
        """
        n_docs = len(texts)
        params = self.params
        if n_docs == 0:
            return LexicalIndex(
                indptr=np.zeros(1, dtype=np.int64),
                indices=np.empty(0, dtype=np.int32),
                impacts=np.empty(0, dtype=np.float32),
                idf=np.empty(0, dtype=np.float32),
                vocab={},
                n_docs=0,
                params=params,
            )

        vocab: dict[str, int] = {}
        term_col: list[int] = []
        row_col: list[int] = []
        tf_col: list[int] = []
        doc_len = np.zeros(n_docs, dtype=np.float64)

        for row, text in enumerate(texts):
            tokens = params.tokenize(text or "")
            doc_len[row] = len(tokens)
            if not tokens:
                continue
            for term, tf in Counter(tokens).items():
                tid = vocab.get(term)
                if tid is None:
                    tid = len(vocab)
                    vocab[term] = tid
                term_col.append(tid)
                row_col.append(row)
                tf_col.append(tf)

        if not term_col:
            return LexicalIndex(
                indptr=np.zeros(1, dtype=np.int64),
                indices=np.empty(0, dtype=np.int32),
                impacts=np.empty(0, dtype=np.float32),
                idf=np.empty(0, dtype=np.float32),
                vocab={},
                n_docs=n_docs,
                params=params,
            )

        terms = np.asarray(term_col, dtype=np.int64)
        rows = np.asarray(row_col, dtype=np.int64)
        tfs = np.asarray(tf_col, dtype=np.float64)
        del term_col, row_col, tf_col

        n_terms_raw = len(vocab)
        df = np.bincount(terms, minlength=n_terms_raw).astype(np.float64)

        # 相対剪定 + df == 0 の除去 (後者は語彙構築の性質上起きないが規約どおり明示)
        max_df = max(1.0, params.max_df_ratio * n_docs)
        keep_term = (df > 0) & (df <= max_df)
        n_kept = int(keep_term.sum())
        if n_kept == 0:
            logger.warning(
                "lexical index: every term pruned (n_docs=%d, max_df_ratio=%.3f)",
                n_docs, params.max_df_ratio,
            )
            return LexicalIndex(
                indptr=np.zeros(1, dtype=np.int64),
                indices=np.empty(0, dtype=np.int32),
                impacts=np.empty(0, dtype=np.float32),
                idf=np.empty(0, dtype=np.float32),
                vocab={},
                n_docs=n_docs,
                params=params,
            )

        remap = np.full(n_terms_raw, -1, dtype=np.int64)
        kept_old_ids = np.flatnonzero(keep_term)
        remap[kept_old_ids] = np.arange(n_kept, dtype=np.int64)

        new_terms = remap[terms]
        entry_keep = new_terms >= 0
        new_terms = new_terms[entry_keep]
        rows = rows[entry_keep]
        tfs = tfs[entry_keep]

        kept_df = df[kept_old_ids]
        idf = np.log(1.0 + (n_docs - kept_df + 0.5) / (kept_df + 0.5))

        avgdl = float(doc_len.mean()) or 1.0
        norm = params.k1 * (1.0 - params.b + params.b * (doc_len[rows] / avgdl))
        impacts = idf[new_terms] * (tfs * (params.k1 + 1.0) / (tfs + norm) + params.delta)

        # 語ごとに impact 降順。lexsort は最後のキーが主キー。
        order = np.lexsort((-impacts, new_terms))
        sorted_terms = new_terms[order]
        indices = rows[order].astype(np.int32, copy=False)
        sorted_impacts = impacts[order].astype(np.float32, copy=False)

        indptr = np.zeros(n_kept + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(sorted_terms, minlength=n_kept))

        old_to_term = {tid: term for term, tid in vocab.items()}
        new_vocab = {old_to_term[int(old)]: int(new) for old, new in zip(kept_old_ids, remap[kept_old_ids])}

        logger.info(
            "lexical index built: docs=%d terms=%d (pruned %d) postings=%d",
            n_docs, n_kept, n_terms_raw - n_kept, int(indices.shape[0]),
        )
        return LexicalIndex(
            indptr=indptr,
            indices=indices,
            impacts=sorted_impacts,
            idf=idf.astype(np.float32, copy=False),
            vocab=new_vocab,
            n_docs=n_docs,
            params=params,
        )


class LexicalShardSet:
    """シャード別索引の束 (c_16 §6.2 末尾)。

    シャードは episodic なら ``tier × 月``、semantic なら namespace、corpus なら
    パッケージ。ゲートを通ったシャードだけ読む。

    行番号はシャードごとにローカル (0..n_docs-1) なので、``rows`` で
    **シャードローカル行 → グローバル行** の写像を明示的に持つ。連続領域を
    仮定しない (シャードが飛び飛びの行を持ってよい)。
    """

    def __init__(
        self,
        shards: dict[str, LexicalIndex],
        rows: dict[str, np.ndarray],
    ) -> None:
        missing = set(shards) - set(rows)
        if missing:
            raise ValueError(f"row mapping missing for shards: {sorted(missing)}")
        for name, index in shards.items():
            mapping = np.asarray(rows[name])
            if mapping.shape[0] != index.n_docs:
                raise ValueError(
                    f"shard {name!r}: row mapping has {mapping.shape[0]} entries "
                    f"but index has {index.n_docs} docs"
                )
        self.shards: dict[str, LexicalIndex] = shards
        self.rows: dict[str, np.ndarray] = {
            name: np.asarray(mapping, dtype=np.int64) for name, mapping in rows.items()
        }

    def candidates_in(
        self,
        shards: Iterable[str],
        query: str,
        top_k: int,
        *,
        budget_ms: float | None = 20.0,
        row_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """指定シャードの候補をグローバル行番号で union する。

        ``budget_ms`` は **束全体** の予算。使い切った時点で残りのシャードは
        読まずに、集まった分を返す。

        ``row_mask`` はグローバル行番号で引ける bool 配列。各シャードの写像で
        ローカルへ引き直す。

        スコアはシャードごとの IDF から出るため厳密には可換ではない。
        c_16 §6.3 のとおり **候補の順序付け専用** で、ゲート・順位には使わない。
        """
        started = time.perf_counter()
        row_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []

        for name in shards:
            index = self.shards.get(name)
            if index is None:
                logger.debug("lexical shard not found: %s", name)
                continue
            remaining: float | None = None
            if budget_ms is not None:
                remaining = budget_ms - (time.perf_counter() - started) * 1000.0
                if remaining <= 0.0:
                    logger.debug("lexical shard set: budget exhausted before %s", name)
                    break
            mapping = self.rows[name]
            local_mask = None if row_mask is None else row_mask[mapping]
            local_rows, scores = index.candidates(
                query, top_k, budget_ms=remaining, row_mask=local_mask,
            )
            if local_rows.shape[0] == 0:
                continue
            row_parts.append(mapping[local_rows])
            score_parts.append(scores)

        if not row_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        all_rows = np.concatenate(row_parts).astype(np.int64, copy=False)
        all_scores = np.concatenate(score_parts).astype(np.float32, copy=False)
        return _top_k_desc(all_rows, all_scores, top_k)


def max_postings_scanned(index: LexicalIndex, query: str) -> int:
    """``query`` が実際に読む posting 件数 (走査量の上限証明用)。

    ``Q × M`` を超えないことがテストで検証される。予算超過による打ち切りが
    無い場合の値なので、これが実走査量の上界になる。
    """
    total = 0
    for tid in index.query_term_ids(query):
        total += min(index.posting_len(int(tid)), index.params.m_postings)
    return total


def _top_k_desc(
    rows: np.ndarray, scores: np.ndarray, top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """スコア降順 (同点は行番号昇順) で上位 ``top_k`` を返す。"""
    if top_k <= 0 or rows.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    if rows.shape[0] > top_k:
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        rows = rows[part]
        scores = scores[part]
    order = np.lexsort((rows, -scores))
    return rows[order].astype(np.int64, copy=False), scores[order].astype(np.float32, copy=False)
