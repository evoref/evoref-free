"""テキストチャンク分割"""

import re

import tiktoken

from backend.log_config import get_logger

logger = get_logger("rag.chunker")

# cl100k_base エンコーディング（日英混在テキストに対応）
_encoding = tiktoken.get_encoding("cl100k_base")

# 文境界パターン（優先順位順）
SENTENCE_BOUNDARIES = [
    re.compile(r"\n\n+"),                    # 段落区切り
    re.compile(r"(?<=[。！？…])\s*"),         # 日本語文末
    re.compile(r"(?<=[.!?])\s+"),            # 英語文末
    re.compile(r"\n"),                        # 改行
]


class SemanticChunker:
    """セマンティック境界ベースのチャンク分割"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk: int = 64,
        max_chunk: int = 512,
        strategy: str = "semantic",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.strategy = strategy

    def chunk(self, text: str) -> list[str]:
        """テキストをチャンクに分割"""
        if not text.strip():
            logger.debug("chunk: empty text, returning []")
            return []

        logger.debug(
            "chunk: strategy=%s, text_length=%d, chunk_size=%d, overlap=%d",
            self.strategy, len(text), self.chunk_size, self.chunk_overlap,
        )

        if self.strategy == "fixed":
            result = self._fixed_chunk(text)
        else:
            result = self._semantic_chunk(text)

        logger.debug(
            "chunk: produced %d chunks, sizes=[%s]",
            len(result),
            ", ".join(str(len(c)) for c in result[:10])
            + ("..." if len(result) > 10 else ""),
        )
        return result

    def _semantic_chunk(self, text: str) -> list[str]:
        """セマンティック分割: 文境界で分割し、トークン予算に収める"""
        # まず文に分割
        paragraphs = re.split(r"\n\n+", text)
        para_count = len([p for p in paragraphs if p.strip()])
        sentences = self._split_sentences(text)
        if not sentences:
            logger.debug("_semantic_chunk: no sentences extracted")
            return []
        avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        logger.debug(
            "_semantic_chunk: paragraphs=%d, sentences=%d, avg_sentence_len=%.1f",
            para_count, len(sentences), avg_len,
        )

        chunks = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            # 1文がmax_chunkを超える場合は固定長で分割
            if sent_tokens > self.max_chunk:
                logger.debug(
                    "_semantic_chunk: sentence exceeds max_chunk (%d > %d), "
                    "falling back to fixed chunking",
                    sent_tokens, self.max_chunk,
                )
                # 現在のバッファをフラッシュ
                if current_sentences:
                    chunks.append("".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # 長い文を固定長で分割
                sub_chunks = self._fixed_chunk(sentence)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sent_tokens > self.max_chunk and current_sentences:
                # バッファをフラッシュ
                chunks.append("".join(current_sentences))
                # オーバーラップ: 最後の文を引き継ぐ
                last = current_sentences[-1] if current_sentences else ""
                current_sentences = [last] if self._estimate_tokens(last) < self.chunk_overlap else []
                current_tokens = self._estimate_tokens("".join(current_sentences))

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # 残りをフラッシュ
        if current_sentences:
            chunk_text = "".join(current_sentences)
            if self._estimate_tokens(chunk_text) >= self.min_chunk or not chunks:
                chunks.append(chunk_text)
            elif chunks:
                # min_chunk未満なら最後のチャンクに結合
                chunks[-1] += chunk_text

        return [c.strip() for c in chunks if c.strip()]

    def _fixed_chunk(self, text: str) -> list[str]:
        """固定長分割（トークン単位）"""
        tokens = _encoding.encode(text)
        if not tokens:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = _encoding.decode(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += step

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """文境界で分割"""
        # 段落 → 文の2段階で分割
        paragraphs = re.split(r"\n\n+", text)
        sentences = []

        for para in paragraphs:
            if not para.strip():
                continue
            # 日本語・英語の文末で分割
            parts = re.split(r"(?<=[。！？.!?])\s*", para)
            for part in parts:
                if part.strip():
                    sentences.append(part.strip() + " ")

        return sentences

    def _estimate_tokens(self, text: str) -> int:
        """tiktoken による正確なトークン数計算"""
        if not text:
            return 0
        return max(1, len(_encoding.encode(text)))
