"""BM25 検索（rank-bm25 + 文字 n-gram + ASCII 分割 + ストップワード）"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from rank_bm25 import BM25Plus

from backend.log_config import get_logger

logger = get_logger("rag.bm25_retriever")


# 接頭辞・接尾辞になりやすい高頻度のつなぎ語のみを絞って列挙し、
# 内容語バイグラムを誤って除去しないよう保守的に維持する。
DEFAULT_STOPWORD_BIGRAMS: frozenset[str] = frozenset(
    [
        "のは", "のが", "のに", "のを", "のと", "のも", "のか", "ので", "のよ",
        "はの", "はが", "はに", "はを", "はと", "はま",
        "がの", "がは", "がに", "がを", "がと", "がで",
        "にの", "には", "にが", "にを", "にと", "にし", "にな", "にお",
        "をの", "をは", "をが", "をに", "をと", "をし", "をお",
        "とが", "とは", "とに", "とを", "との", "とし",
        "です", "ます", "でし", "まし", "だっ", "った",
        "する", "した", "して", "しま", "され", "せる",
        "ある", "あり", "いる", "いま", "なる", "なっ", "なり",
        "この", "その", "あの", "どの", "これ", "それ", "あれ", "どれ",
    ]
)


def _split_ascii_token(raw: str) -> list[str]:
    """camelCase / snake_case / 連続数字の ASCII トークンをサブトークンに分割する。

    元トークン自体は呼び出し元で別途追加される。本関数はサブトークン
    （小文字化済み）のみを返す。元と同一になるサブトークンは除外。
    """
    subs: list[str] = []
    # まず underscore で区切る
    for part in re.split(r"_+", raw):
        if not part:
            continue
        # camelCase/PascalCase/連続数字を分解
        for m in re.finditer(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", part):
            sub = m.group(0).lower()
            if sub:
                subs.append(sub)
    return subs


def tokenize_ja(
    text: str,
    *,
    use_trigrams: bool = False,
    split_ascii: bool = True,
    stopwords: Iterable[str] | None = None,
) -> list[str]:
    """MeCab 不要の日本語トークナイザ。

    - NFKC 正規化
    - ASCII トークンは大小文字保持した上で小文字化したトークンを出力
    - ``split_ascii`` が真なら camelCase / snake_case を追加トークン化（元トークンも保持）
    - 日本語部分は文字 bi-gram を生成、``use_trigrams`` が真なら tri-gram も併用
    - ``stopwords`` に含まれる n-gram は除外
    """
    nfkc = unicodedata.normalize("NFKC", text)
    # ASCII span を case-preserving で抽出
    ascii_spans = re.findall(r"[A-Za-z0-9_]+", nfkc)
    ascii_tokens: list[str] = []
    for span in ascii_spans:
        low = span.lower()
        ascii_tokens.append(low)
        if split_ascii:
            for sub in _split_ascii_token(span):
                if sub != low:
                    ascii_tokens.append(sub)

    # 日本語バイグラム: ASCII / 空白 / underscore を除いた連続領域から
    lower = nfkc.lower()
    ja_text = re.sub(r"[a-z0-9_\s]", "", lower)

    stop_set = frozenset(stopwords) if stopwords is not None else frozenset()

    bigrams = [ja_text[i : i + 2] for i in range(len(ja_text) - 1)]
    if stop_set:
        bigrams = [b for b in bigrams if b not in stop_set]

    tokens = ascii_tokens + bigrams

    if use_trigrams:
        trigrams = [ja_text[i : i + 3] for i in range(len(ja_text) - 2)]
        if stop_set:
            trigrams = [t for t in trigrams if t not in stop_set]
        tokens += trigrams

    return tokens


# 旧名を後方互換用に残す（内部利用のみ、テスト/外部参照は tokenize_ja を推奨）。
_tokenize_ja = tokenize_ja


class BM25Retriever:
    """BM25Plus ベースの検索。

    トークナイズ挙動 (trigram / ASCII 分割 / stopword) と
    BM25 パラメータ (k1 / b / delta) を外部から注入可能にした。
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        use_trigrams: bool = False,
        split_ascii: bool = True,
        stopwords: Iterable[str] | None = None,
    ):
        self._bm25: BM25Plus | None = None
        self._chunk_ids: list[str] = []
        self._chunks: list[str] = []
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._use_trigrams = use_trigrams
        self._split_ascii = split_ascii
        # stopwords は None / [] / 明示リストを区別したいので frozenset に固定化
        self._stopwords: frozenset[str] = (
            frozenset(stopwords) if stopwords is not None else frozenset()
        )

    def _tokenize(self, text: str) -> list[str]:
        return tokenize_ja(
            text,
            use_trigrams=self._use_trigrams,
            split_ascii=self._split_ascii,
            stopwords=self._stopwords if self._stopwords else None,
        )

    def build(self, chunk_ids: list[str], chunks: list[str]) -> None:
        """BM25 インデックスを構築"""
        self._chunk_ids = chunk_ids
        self._chunks = chunks
        tokenized = [self._tokenize(c) for c in chunks]
        self._bm25 = BM25Plus(tokenized, k1=self._k1, b=self._b, delta=self._delta)
        vocab: set[str] = set()
        for tokens in tokenized:
            vocab.update(tokens)
        logger.info("BM25 index built: %d documents", len(chunks))
        logger.debug(
            "BM25 build stats: docs=%d, vocab_size=%d, total_tokens=%d, "
            "k1=%.3f, b=%.3f, delta=%.3f, trigrams=%s, split_ascii=%s, stopwords=%d",
            len(chunks), len(vocab), sum(len(t) for t in tokenized),
            self._k1, self._b, self._delta,
            self._use_trigrams, self._split_ascii, len(self._stopwords),
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """BM25 検索。(chunk_id, score) のリストを返す"""
        if self._bm25 is None or not self._chunk_ids:
            logger.debug("BM25 search: index not built, returning []")
            return []

        tokenized_query = self._tokenize(query)
        ascii_count = len(re.findall(r"[A-Za-z0-9_]+", query))
        ngram_count = len(tokenized_query) - ascii_count
        logger.debug(
            "BM25 search: query=%r, tokens=%d (ascii_like=%d, ngram=%d)",
            query[:50], len(tokenized_query), ascii_count, ngram_count,
        )
        scores = self._bm25.get_scores(tokenized_query)

        k = min(top_k, len(scores))
        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))

        logger.debug(
            "BM25 search: %d results (of %d docs), scores=[%s]",
            len(results), self.count,
            ", ".join(f"{s:.3f}" for _, s in results[:5]),
        )
        return results

    @property
    def count(self) -> int:
        return len(self._chunk_ids)
