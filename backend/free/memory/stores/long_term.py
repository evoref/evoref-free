"""Layer 3: 長期記憶（RAG ベクトル DB + タグ検索）"""

import numpy as np

from backend.log_config import get_logger
from backend.free.core.text_quality import carries_no_assertion
from backend.free.rag.vector_store import VectorStore

logger = get_logger("memory.long_term")


class LongTermMemory:
    """Layer 3: RAG ベクトル DB + タグ検索（LLM ゼロ）"""

    def __init__(self, vector_store: VectorStore):
        self.vectors = vector_store
        self.note_meta: dict[str, dict] = {}  # {note_id: meta_dict}

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        tag_filter: list[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """ベクトル検索 + オプショナルタグフィルタ

        Returns:
            list of (chunk_id, score, chunk_text)
        """
        candidates = self.vectors.search(query_vec, top_k=top_k * 2)
        logger.debug("LTM search: %d candidates from vector store (fetch_k=%d)", len(candidates), top_k * 2)

        # 問いだけのチャンクは答えを含まないので、検索で当たっても参考情報に
        # ならない。しかも **想起クエリでは必ず最上位に来る** — 過去に同じ質問を
        # していれば意味的に最も近いのはその質問文そのものだから。
        #
        # 実測 (2026-08-09): 「私の趣味は何でしたか？」の LTM 上位 1 位が
        # 0.541「私の趣味は何でしたか？」(過去の同一質問)、2 位が 0.449
        # 「私の趣味はゴルフでしたよね？」。答えを持つチャンクは 3 位以下へ
        # 押し下げられていた。実ストア 272 チャンク中 50 件 (18%) が該当する。
        #
        # 同じ判定は注入層 (MemoryInjector) がファクト / STM ノートに既に
        # 掛けているが、LTM 経路だけ素通りだった。書込側 (absorb_from_short_term)
        # でも弾くが、既存チャンクは reindex しないと消えないため読込時にも掛ける
        # (2026-08-07 の教訓: ゲートは採用時と読込時の両方に要る)。
        before = len(candidates)
        candidates = [c for c in candidates if not carries_no_assertion(c[2])]
        if len(candidates) != before:
            logger.debug(
                "LTM search: assertion gate dropped %d question-only chunks",
                before - len(candidates),
            )

        if tag_filter:
            before = len(candidates)
            filtered = []
            for chunk_id, score, text in candidates:
                meta = self.note_meta.get(chunk_id, {})
                tags = meta.get("tags", [])
                if any(t in tags for t in tag_filter):
                    filtered.append((chunk_id, score, text))
            candidates = filtered
            logger.debug("LTM search: tag filter %s reduced %d -> %d", tag_filter, before, len(candidates))

        return candidates[:top_k]

    def absorb_from_short_term(self, note) -> str | None:
        """STM ノートを LTM に昇格（ベクトル DB に追加）"""
        if note.embedding is None:
            logger.warning("Note %s has no embedding, skipping LTM absorption", note.id)
            return None

        # 問いだけのノートは LTM へ入れない (``search`` の読込時ゲートと対)。
        # 入れても検索ノイズにしかならず、想起クエリでは最上位を占める。
        if carries_no_assertion(getattr(note, "content", "") or ""):
            logger.debug(
                "Note %s carries no assertion, skipping LTM absorption", note.id,
            )
            return None

        # 次元不一致ノートはスキップ
        stored_dim = self.vectors.stored_dim()
        note_dim = int(note.embedding.shape[0])
        if stored_dim is not None and stored_dim != note_dim:
            logger.warning(
                "Note %s embedding dim (%d) != store dim (%d), "
                "skipping LTM absorption. Run 'evoref reindex'.",
                note.id, note_dim, stored_dim,
            )
            return None

        vec = note.embedding.reshape(1, -1)
        # memory ノートは自己完結したチャンクで contextual prefix の価値が薄く、
        # source text も保存しないため has_context=True で登録し、sleep-time
        # Step 5.8 の「source text not found」空振りスキャンから除外する。
        chunk_ids = self.vectors.add_vectors(
            vec,
            [note.content],
            source=f"memory:{note.session_id}",
            category="memory",
            has_context=True,
        )

        chunk_id = chunk_ids[0]
        # EvorefMem: trace_id を note_meta に保存し
        # ファクトと episodic LTM チャンクを ``trace_id`` で連結可能にする。
        self.note_meta[chunk_id] = {
            "note_id": note.id,
            "tags": note.tags,
            "keywords": note.keywords,
            "session_id": note.session_id,
            "trace_id": getattr(note, "trace_id", None),
            # 発話者 (user / assistant / rag / system)。STM ノートは全件持って
            # いるのに LTM へ移す時に落ちており、検索結果では「ユーザーが述べた
            # 事実」と「アシスタント自身が過去に答えた内容」が区別できなかった。
            #
            # 実測 (2026-08-09): STM 99 件の内訳は user 49 / assistant 50 で、
            # **半分が自分の出力**。「私の趣味は？」の LTM 上位には自分の過去の
            # 回答が並び、古い値 (0.444) が新しい値 (0.438) を上回っていた。
            # まず素性を残す — 検索側でどう扱うかは、これが観測できるように
            # なってから実測で決める。
            "source": getattr(note, "source", None),
        }
        logger.info("Absorbed note %s to LTM as chunk %s", note.id, chunk_id)
        return chunk_id
