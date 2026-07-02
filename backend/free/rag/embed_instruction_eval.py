"""候補検索 instruction の offline 実測評価 (EmbedInstructionEval)

EmbedInstructionEvolver (EvorefLearn) が進化中の候補 instruction の検索品質を
実測するための実体。候補 instruction を ``query_template`` の ``{task}`` に
差し込んで記録済みクエリを再埋め込みし、既存のドキュメントベクトル
(``VectorStore``) と検索して top1 cosine スコアを測る。

本クラスは ``backend.free.optimizer.embed_eval.EmbedEvalProtocol`` を**明示継承
しない** (構造的部分型で満たす)。Gen pillar が Learn pillar の Protocol を
import すると依存方向 (Gen→Learn) を逆転させ pillar 境界を侵すため、duck typing
で満たし wire 時に注入する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.exceptions import VectorDimensionMismatchError
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("rag.embed_instruction_eval")


class EmbedInstructionEval:
    """候補 instruction で query を再埋め込み・再検索し top1 cosine を実測する。"""

    def __init__(
        self,
        embedder: "EmbeddingBackend",
        vector_store: "VectorStore",
        query_template: str = "Instruct: {task}\nQuery: {query}",
        top_k: int = 1,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._query_template = query_template
        self._top_k = max(1, top_k)

    async def score_candidate(
        self, candidate: str, queries: list[str],
    ) -> float | None:
        """``candidate`` を検索 instruction として ``queries`` の top1 平均を返す。

        実測不能 (ベクトルストア空 / query 無し / テンプレート不正 / 埋め込み
        または検索の例外) は ``None`` を返し degrade させる。
        """
        vs = self._vector_store
        if vs is None or vs.count == 0 or not queries:
            return None
        if "{task}" not in self._query_template or "{query}" not in self._query_template:
            # instruction を差し込めないテンプレート (非 instruction-aware モデル)
            return None

        # キャッシュをバイパスして raw embedder を使う (候補ごとの eval 専用埋め込みで
        # ディスクキャッシュを汚さない)。doc_template は instruction-aware モデルでは
        # 空のため、整形済みテキストを is_query=False で素通り送出できる。
        raw = getattr(self._embedder, "inner", self._embedder)
        scores: list[float] = []
        try:
            for q in queries:
                formatted = self._query_template.format(task=candidate, query=q)
                vecs = await raw.embed([formatted], is_query=False)
                hits = vs.search(vecs[0], top_k=self._top_k)
                if hits:
                    scores.append(float(hits[0][1]))
        except (VectorDimensionMismatchError, Exception) as exc:
            logger.warning("embed instruction eval failed, degrading: %s", exc)
            return None

        if not scores:
            return None
        return sum(scores) / len(scores)
