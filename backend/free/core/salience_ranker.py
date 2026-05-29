"""BudgetMem 式サリエンスランカー: RAG チャンクのトークン予算内情報量最大化

BudgetMem (arXiv:2511.04919) のサリエンススコアを numpy + 文字列処理のみで実装。
LLM 不要。推論パスへの追加遅延: < 1ms。

5因子スコアリング:
  1. クエリ関連度 — 既存の検索スコア（BM25+Vector RRF）を流用
  2. TF-IDF 重要度 — チャンク内の高頻度語密度
  3. エンティティ密度 — 固有表現（数値、コード識別子等）の出現率
  4. 情報密度 — ユニーク語彙率（冗長な繰り返しを減点）
  5. 位置バイアス — ドキュメント先頭/末尾の重み付け
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from backend.log_config import get_logger
from backend.policy_helpers import get_policy_value
from backend.utils import estimate_tokens

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("core.salience_ranker")

# デフォルト重み（ポリシー未設定時）
_DEFAULT_W_QUERY_RELEVANCE = 0.35
_DEFAULT_W_TFIDF = 0.20
_DEFAULT_W_ENTITY_DENSITY = 0.15
_DEFAULT_W_INFO_DENSITY = 0.20
_DEFAULT_W_POSITION_BIAS = 0.10

# エンティティ検出パターン（コード・技術テキスト特有のトークン）
_ENTITY_PATTERN = re.compile(
    r"(?:"
    r"[A-Za-z_]\w*[_.]\w+"  # ドット/アンダースコア含む識別子 (e.g. os.path, my_var)
    r"|[0-9]+(?:\.[0-9]+)+"  # バージョン番号・IP等 (e.g. 3.14.159)
    r"|[0-9]{2,}"  # 数値（2桁以上）
    r"|https?://\S+"  # URL
    r"|[A-Z][A-Z0-9_]{2,}"  # 定数（大文字3文字以上, e.g. MAX_SIZE）
    r"|`[^`]+`"  # バッククォート囲みコード
    r")"
)

# 単語トークナイズパターン
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff\u3040-\u30ff]")


class SalienceRanker:
    """BudgetMem 式サリエンスランカー

    RAG チャンクを5因子でスコアリングし、トークン予算内で
    情報量を最大化するチャンク集合を選別する。
    """

    def __init__(
        self,
        policy: PolicyInterpreter | None = None,
        mode: str = "chat",
    ):
        self._policy = policy
        self._mode = mode

        # 重みをポリシーから取得（フォールバック: デフォルト値）
        self._w1 = self._p("salience_w_query_relevance", _DEFAULT_W_QUERY_RELEVANCE)
        self._w2 = self._p("salience_w_tfidf", _DEFAULT_W_TFIDF)
        self._w3 = self._p("salience_w_entity_density", _DEFAULT_W_ENTITY_DENSITY)
        self._w4 = self._p("salience_w_info_density", _DEFAULT_W_INFO_DENSITY)
        self._w5 = self._p("salience_w_position_bias", _DEFAULT_W_POSITION_BIAS)

    def _p(self, key: str, default: float) -> float:
        """ポリシーからパラメータ取得（フォールバック付き）"""
        return get_policy_value(
            self._policy, "search", key, default,
            mode=self._mode, coerce_type=float,
        )

    def rank(
        self,
        chunks: list[tuple[str, float, str]],
        budget_tokens: int,
    ) -> list[str]:
        """チャンクをサリエンススコアで評価し、予算内で最適な集合を返す。

        Args:
            chunks: 検索結果 [(chunk_id, search_score, text), ...]
            budget_tokens: RAG チャンクに割り当てられるトークン予算

        Returns:
            予算内で情報量を最大化するチャンクテキストのリスト（挿入順）
        """
        if not chunks or budget_tokens <= 0:
            return []

        n = len(chunks)

        # 全チャンクの TF-IDF 用コーパス統計を計算
        corpus_df = _compute_document_frequencies(chunks)
        corpus_size = n

        # 各チャンクのサリエンススコアとトークンコストを計算
        scored: list[tuple[int, float, int, str]] = []  # (index, salience, tokens, text)
        for i, (chunk_id, search_score, text) in enumerate(chunks):
            tokens = estimate_tokens(text)
            if tokens <= 0:
                continue

            s1 = _normalize_query_relevance(search_score, chunks)
            s2 = _compute_tfidf_density(text, corpus_df, corpus_size)
            s3 = _compute_entity_density(text)
            s4 = _compute_info_density(text)
            s5 = _compute_position_bias(i, n)

            salience = (
                self._w1 * s1
                + self._w2 * s2
                + self._w3 * s3
                + self._w4 * s4
                + self._w5 * s5
            )
            scored.append((i, salience, tokens, text))

        if not scored:
            return []

        # 効率順（salience / token）でソートしてグリーディ充填
        scored.sort(key=lambda x: x[1] / max(1, x[2]), reverse=True)

        selected: list[tuple[int, str]] = []  # (original_index, text)
        remaining = budget_tokens
        for idx, salience, tokens, text in scored:
            if tokens > remaining:
                continue
            selected.append((idx, text))
            remaining -= tokens

        # 元のドキュメント順序を保持（検索結果の自然な流れを維持）
        selected.sort(key=lambda x: x[0])

        result = [text for _, text in selected]

        logger.debug(
            "SalienceRanker: %d/%d chunks selected, "
            "%d/%d tokens used, efficiency=%.1f%%",
            len(result), len(chunks),
            budget_tokens - remaining, budget_tokens,
            (budget_tokens - remaining) / max(1, budget_tokens) * 100,
        )

        return result


def _normalize_query_relevance(
    score: float,
    chunks: list[tuple[str, float, str]],
) -> float:
    """検索スコアを [0, 1] に正規化する"""
    if not chunks:
        return 0.0
    max_score = max(s for _, s, _ in chunks)
    min_score = min(s for _, s, _ in chunks)
    if max_score == min_score:
        return 1.0
    return (score - min_score) / (max_score - min_score)


def _compute_document_frequencies(
    chunks: list[tuple[str, float, str]],
) -> dict[str, int]:
    """コーパス全体の文書頻度（DF）を計算する"""
    df: dict[str, int] = {}
    for _, _, text in chunks:
        words = set(_WORD_PATTERN.findall(text.lower()))
        for w in words:
            df[w] = df.get(w, 0) + 1
    return df


def _compute_tfidf_density(
    text: str,
    corpus_df: dict[str, int],
    corpus_size: int,
) -> float:
    """チャンク内の TF-IDF 上位語の密度を [0, 1] で返す"""
    words = _WORD_PATTERN.findall(text.lower())
    if not words:
        return 0.0

    # TF 計算
    tf: dict[str, int] = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1

    # TF-IDF スコア
    tfidf_scores: list[float] = []
    for w, count in tf.items():
        tf_val = count / len(words)
        df_val = corpus_df.get(w, 1)
        idf_val = math.log((corpus_size + 1) / (df_val + 1)) + 1
        tfidf_scores.append(tf_val * idf_val)

    if not tfidf_scores:
        return 0.0

    # 上位 30% の TF-IDF スコアの平均を密度として使用
    tfidf_scores.sort(reverse=True)
    top_n = max(1, len(tfidf_scores) * 3 // 10)
    top_avg = sum(tfidf_scores[:top_n]) / top_n

    # [0, 1] に正規化（TF-IDF 値の実用的上限は ~3.0）
    return min(1.0, top_avg / 3.0)


def _compute_entity_density(text: str) -> float:
    """固有表現（数値、コード識別子等）の密度を [0, 1] で返す"""
    words = _WORD_PATTERN.findall(text)
    if not words:
        return 0.0
    entities = _ENTITY_PATTERN.findall(text)
    raw = len(entities) / len(words)
    # sigmoid 風の正規化（0.3 付近で 0.5）
    return min(1.0, raw / 0.6)


def _compute_info_density(text: str) -> float:
    """ユニーク語彙率（冗長な繰り返しを減点）を [0, 1] で返す"""
    words = _WORD_PATTERN.findall(text.lower())
    if not words:
        return 0.0
    unique = len(set(words))
    return unique / len(words)


def _compute_position_bias(index: int, total: int) -> float:
    """ドキュメント先頭/末尾の重み付けを [0, 1] で返す

    先頭と末尾に高い重みを付け、中央部分を軽くする（U字カーブ）。
    """
    if total <= 1:
        return 1.0
    # [0, 1] に正規化された位置
    pos = index / (total - 1)
    # U 字カーブ: 先頭(1.0) → 中央(0.5) → 末尾(0.8)
    return 1.0 - 0.5 * math.sin(pos * math.pi)
