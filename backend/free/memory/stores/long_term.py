"""Layer 3: 長期記憶（RAG ベクトル DB + タグ検索）"""

import numpy as np

from backend.log_config import get_logger
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
        chunk_ids = self.vectors.add_vectors(
            vec,
            [note.content],
            source=f"memory:{note.session_id}",
            category="memory",
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
        }
        logger.info("Absorbed note %s to LTM as chunk %s", note.id, chunk_id)
        return chunk_id
