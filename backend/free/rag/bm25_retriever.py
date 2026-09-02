"""BM25 検索（rank-bm25 + 文字 n-gram + ASCII 分割 + ストップワード）"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import TYPE_CHECKING

from rank_bm25 import BM25Plus

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

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


#: :meth:`BM25Retriever.rare_query_tokens` の既定閾値。コーパスの 1% 未満の
#: ドキュメントにしか現れないトークンを「希少」とみなす。
DEFAULT_RARE_TOKEN_DF_RATIO: float = 0.01


#: 日本語 n-gram を切る「連続領域」。単語構成文字 (かな / 漢字 / 長音 等) の
#: 連続で、ASCII 英数字・underscore・空白・句読点・記号で途切れる。
_JA_RUN_RE = re.compile(r"[^\W\d_a-z]+")


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

    # 日本語 n-gram: ASCII / 空白 / 記号で区切った **連続領域ごと** に生成する。
    # 以前は除去後の文字列を 1 本に繋いでから bi-gram を切っていたため、
    # 「〜を使うで Python を〜」→「うで」、「〜です。次に〜」→「。次」のような
    # 境界をまたぐ偽トークンが生まれていた。偽トークンは df が極端に小さく、
    # ``rare_query_tokens`` で「希少語」として拾われて floor 免除の根拠になる。
    lower = nfkc.lower()
    ja_runs = _JA_RUN_RE.findall(lower)

    stop_set = frozenset(stopwords) if stopwords is not None else frozenset()

    bigrams = [
        run[i : i + 2] for run in ja_runs for i in range(len(run) - 1)
    ]
    if stop_set:
        bigrams = [b for b in bigrams if b not in stop_set]

    tokens = ascii_tokens + bigrams

    if use_trigrams:
        trigrams = [
            run[i : i + 3] for run in ja_runs for i in range(len(run) - 2)
        ]
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
        debug_logger: "DebugLogger | None" = None,
    ):
        self._debug_logger = debug_logger
        self._bm25: BM25Plus | None = None
        self._chunk_ids: list[str] = []
        self._chunks: list[str] = []
        #: トークン -> 出現ドキュメント数。``rare_query_tokens`` が使う。
        self._df: dict[str, int] = {}
        #: chunk_id -> トークン集合。``lexical_anchors`` が使う (再トークナイズ回避)。
        self._doc_tokens: dict[str, frozenset[str]] = {}
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
        self._df = {}
        self._doc_tokens = {}
        for cid, tokens in zip(chunk_ids, tokenized):
            uniq = frozenset(tokens)
            vocab.update(uniq)
            self._doc_tokens[cid] = uniq
            for t in uniq:
                self._df[t] = self._df.get(t, 0) + 1
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

    def rare_query_tokens(
        self, query: str, max_df_ratio: float = DEFAULT_RARE_TOKEN_DF_RATIO,
    ) -> frozenset[str]:
        """クエリ中の **希少な** トークン (出現ドキュメント比が閾値未満) を返す。

        「ユーザーが珍しい文字列を名指しした」ことの決定論的な証拠として使う。
        型番・ファイルパス・固有名詞・エラーコードのような literal は密ベクトルが
        最も苦手とする一方、BM25 では df が極端に小さいトークンとして必ず立つ。

        閾値は **相対** (コーパス全体に対する比) なので、コーパスの大きさや
        埋め込みモデルを替えても意味が変わらない。絶対値の閾値がモデル差し替えで
        到達不能になる事故を繰り返しているため、ここでは絶対値を持たない。

        索引未構築なら空集合 (呼出側は「証拠なし」として扱えばよい)。
        """
        if not self._df or not self._chunk_ids:
            return frozenset()
        cutoff = max(1, int(len(self._chunk_ids) * max_df_ratio))
        return frozenset(
            t for t in set(self._tokenize(query))
            if 0 < self._df.get(t, 0) <= cutoff
        )

    def lexical_anchors(
        self,
        query: str,
        chunk_ids: Iterable[str],
        max_df_ratio: float = DEFAULT_RARE_TOKEN_DF_RATIO,
    ) -> frozenset[str]:
        """``chunk_ids`` のうち、クエリの希少トークンを実際に含むものを返す。

        「珍しい語で名指しされ、その語を本当に持っているチャンク」だけが残る。
        コサイン類似度のフロアを免除する根拠として使う (:mod:`search_pipeline`)。
        """
        rare = self.rare_query_tokens(query, max_df_ratio)
        if not rare:
            return frozenset()
        return frozenset(
            cid for cid in chunk_ids
            if self._doc_tokens.get(cid, frozenset()) & rare
        )

    @property
    def count(self) -> int:
        return len(self._chunk_ids)


def build_index_from_vector_store(bm25: BM25Retriever, vector_store) -> int:
    """``vector_store`` の全チャンクから BM25 索引を張り直す。

    索引の作り方 (contextual prefix 付きのテキストを使う) を 1 箇所に集約する。
    以前は起動時の配線と sleep-time のプレフィックス生成後の 2 箇所に同じ 4 行が
    複写されており、**チャット応答経路が使い始めた後もどちらか片方しか
    更新されない**危険があった。

    Returns:
        索引に入れたチャンク数 (メタデータが空なら 0、索引はそのまま)。
    """
    metadata = getattr(vector_store, "metadata", None) or []
    if not metadata:
        return 0
    chunk_ids = [m["id"] for m in metadata]
    chunks = [vector_store.get_contextual_text(cid) for cid in chunk_ids]
    bm25.build(chunk_ids, chunks)
    return len(chunk_ids)
