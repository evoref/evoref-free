"""学習済みパターンストア: 会話から学習したパターン検出キーワードの管理

訂正・言い直し検出時に会話コンテキストからキーワードを抽出し、
次回以降のパターン検出に活用する。LLM 呼び出しなし。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.agent.learned_pattern_store import LearnedPatternRepository
from backend.free.agent.learned_patterns_types import LearnedPattern
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("agent.learned_patterns")

# デフォルト設定
DEFAULT_MAX_PATTERNS = 200
DEFAULT_INITIAL_WEIGHT = 0.5
DEFAULT_DECAY_RATE = 0.05
DEFAULT_BOOST_AMOUNT = 0.15
DEFAULT_MIN_WEIGHT = 0.1
DEFAULT_MATCH_THRESHOLD = 0.3

# 意図キーワード抽出用パターン（動詞・指示語を重点抽出）
_INTENT_PATTERNS = [
    # 日本語動詞・指示語（「〜して」「〜しろ」「〜してください」等の語幹）
    re.compile(r"([\u4e00-\u9fff]{1,4})[しさせ]て"),
    re.compile(r"([\u4e00-\u9fff]{1,4})[しさせ]ろ"),
    re.compile(r"([\u4e00-\u9fff]{1,4})[しさせ]てください"),
    # カタカナ動作語
    re.compile(r"([\u30a0-\u30ff]{2,})して"),
    re.compile(r"([\u30a0-\u30ff]{2,})する"),
    # 英語動詞（先頭の動詞を抽出）
    re.compile(r"\b(add|remove|delete|update|fix|change|create|append|insert|replace|move|copy|rename|merge)\b", re.IGNORECASE),
]

# 汎用キーワード抽出（NoteBuilder 互換）
_KEYWORD_PATTERNS = [
    re.compile(r"[A-Za-z][A-Za-z0-9_.-]+"),
    re.compile(r"[\u4e00-\u9fff]{2,8}"),
    re.compile(r"[\u30a0-\u30ff]{2,}"),
]

# ストップワード（パターンとして学習しない一般的な語）
_STOPWORDS = frozenset({
    # 日本語
    "これ", "それ", "あれ", "ここ", "そこ", "この", "その",
    "ある", "いる", "なる", "する", "できる", "ない", "です",
    "ます", "した", "して", "から", "まで", "より", "ため",
    "こと", "もの", "ところ", "とき", "よう", "ほう",
    # 英語
    "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that",
    "with", "from", "for", "not", "but", "and", "or",
})

# long_form カテゴリ専用の学習除外語 (CJK)。出力先指定などで頻出するが、
# 文書(散文)の種別を示さない汎用ファイル操作語。ASCII トークンは
# ``_LONG_FORM_ASCII_FILEISH_RE`` 側で一律除外するためここには含めない。
_LONG_FORM_NONLEARNABLE_EXACT = frozenset({
    "出力", "出力先", "保存", "保存先", "ファイル",
    "書き出し", "書き込み", "配下", "エクスポート", "セーブ",
})
# ASCII のみのトークン (パス片 / URL 片 / ホスト名 / 形式名 / 識別子) を表す。
# long_form の文書種別シグナルとしては信頼できないため学習対象から外す。
_LONG_FORM_ASCII_FILEISH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


__all__ = ["LearnedPattern", "LearnedPatternStore"]


class LearnedPatternStore:
    """学習済みパターン検出キーワードの永続化管理

    訂正・言い直し検出時に会話から抽出したキーワードを蓄積し、
    次回以降のフィードバック検出で活用する。
    """

    def __init__(
        self,
        config: dict | None = None,
        policy: PolicyInterpreter | None = None,
    ):
        self._policy = policy
        cfg = (config or {}).get("learning", {})
        self.max_patterns: int = cfg.get("pattern_max_patterns", DEFAULT_MAX_PATTERNS)
        self.initial_weight: float = cfg.get("pattern_initial_weight", DEFAULT_INITIAL_WEIGHT)
        self.decay_rate: float = cfg.get("pattern_decay_rate", DEFAULT_DECAY_RATE)
        self.boost_amount: float = cfg.get("pattern_boost_amount", DEFAULT_BOOST_AMOUNT)
        self.min_weight: float = cfg.get("pattern_min_weight", DEFAULT_MIN_WEIGHT)
        self.match_threshold: float = cfg.get("pattern_match_threshold", DEFAULT_MATCH_THRESHOLD)

        self._patterns: dict[str, LearnedPattern] = {}

    @property
    def count(self) -> int:
        return len(self._patterns)

    @property
    def patterns(self) -> dict[str, LearnedPattern]:
        return self._patterns

    def add_pattern(
        self,
        keyword: str,
        category: str = "correction",
        weight: float | None = None,
    ) -> LearnedPattern:
        """パターンを追加または既存のパターンを強化する

        Args:
            keyword: 学習するキーワード
            category: カテゴリ ("correction" | "intent" | "rephrase")
            weight: 初期重み（None でデフォルト値）

        Returns:
            追加/更新されたパターン
        """
        key = keyword.lower()
        if key in _STOPWORDS or len(key) < 2:
            return LearnedPattern(keyword=keyword)

        now = time.time()

        if key in self._patterns:
            pattern = self._patterns[key]
            pattern.source_count += 1
            pattern.last_seen = now
            # 重みをブースト（上限 1.0）
            pattern.weight = min(1.0, pattern.weight + self.boost_amount)
            logger.debug(
                "Boosted pattern '%s': weight=%.3f, sources=%d",
                key, pattern.weight, pattern.source_count,
            )
        else:
            pattern = LearnedPattern(
                keyword=keyword,
                category=category,
                weight=weight if weight is not None else self.initial_weight,
                source_count=1,
                first_seen=now,
                last_seen=now,
            )
            self._patterns[key] = pattern
            logger.info("Learned new pattern: '%s' (category=%s)", keyword, category)

        # 上限超過時は低重みパターンを削除
        self._enforce_limit()
        return pattern

    def match(
        self,
        text: str,
        category: str | None = None,
    ) -> list[tuple[str, float]]:
        """テキストに対して学習済みパターンをマッチングする

        Args:
            text: マッチング対象テキスト
            category: フィルタするカテゴリ（None で全カテゴリ）

        Returns:
            マッチしたパターンのリスト [(keyword, weight), ...]
        """
        if not self._patterns:
            return []

        text_lower = text.lower()
        matches: list[tuple[str, float]] = []

        for key, pattern in self._patterns.items():
            if pattern.weight < self.match_threshold:
                continue
            if category is not None and pattern.category != category:
                continue
            if key in text_lower:
                matches.append((pattern.keyword, pattern.weight))
                # ヒットカウントを更新
                pattern.hit_count += 1
                pattern.last_hit = time.time()

        return sorted(matches, key=lambda x: -x[1])

    def boost(self, keyword: str, amount: float | None = None) -> None:
        """指定パターンの重みをブーストする

        Args:
            keyword: ブースト対象のキーワード
            amount: ブースト量（None でデフォルトの boost_amount）
        """
        key = keyword.lower()
        if key not in self._patterns:
            return
        amt = amount if amount is not None else self.boost_amount
        pattern = self._patterns[key]
        pattern.weight = min(1.0, pattern.weight + amt)
        logger.debug("Boosted pattern '%s': weight=%.3f (+%.3f)", key, pattern.weight, amt)

    def decay_one(self, keyword: str, amount: float | None = None) -> None:
        """指定パターンの重みを減衰させる

        min_weight を下回ったパターンは削除する。

        Args:
            keyword: 減衰対象のキーワード
            amount: 減衰量（None でデフォルトの decay_rate）
        """
        key = keyword.lower()
        if key not in self._patterns:
            return
        amt = amount if amount is not None else self.decay_rate
        pattern = self._patterns[key]
        pattern.weight -= amt
        if pattern.weight < self.min_weight:
            del self._patterns[key]
            logger.debug("Removed pattern '%s' (below min_weight after decay)", key)
        else:
            logger.debug("Decayed pattern '%s': weight=%.3f (-%.3f)", key, pattern.weight, amt)

    def decay_all(self) -> int:
        """全パターンの重みを減衰させる（sleep-time で実行）

        Returns:
            削除されたパターン数
        """
        removed = 0
        to_remove: list[str] = []

        for key, pattern in self._patterns.items():
            pattern.weight -= self.decay_rate
            if pattern.weight < self.min_weight:
                to_remove.append(key)

        for key in to_remove:
            del self._patterns[key]
            removed += 1

        if removed:
            logger.info("Decayed and removed %d patterns", removed)
        return removed

    def decrement_source_count(self, keyword: str) -> None:
        """パターンの source_count を減算する（カートリッジ unload 用）

        source_count が 0 になったパターンは削除する。
        """
        key = keyword.lower()
        if key not in self._patterns:
            return
        pattern = self._patterns[key]
        pattern.source_count -= 1
        if pattern.source_count <= 0:
            del self._patterns[key]
            logger.debug("Removed pattern '%s' (source_count=0)", key)

    def extract_intent_keywords(self, text: str) -> list[str]:
        """テキストから意図キーワードを抽出する（LLM 不要）

        ユーザーの指示文から動作指示語を抽出し、パターン候補として返す。

        Args:
            text: ユーザーのクエリテキスト

        Returns:
            抽出されたキーワードリスト
        """
        keywords: list[str] = []

        # 意図パターンからの抽出（動詞・指示語重視）
        for pattern in _INTENT_PATTERNS:
            for m in pattern.finditer(text):
                kw = m.group(1) if m.lastindex else m.group(0)
                if kw.lower() not in _STOPWORDS and len(kw) >= 2:
                    keywords.append(kw)

        # 汎用キーワード抽出（NoteBuilder 互換、補完用）
        for pattern in _KEYWORD_PATTERNS:
            for m in pattern.finditer(text):
                kw = m.group(0)
                if kw.lower() not in _STOPWORDS and len(kw) >= 2:
                    keywords.append(kw)

        # 重複排除（大文字小文字無視）
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            lower = kw.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(kw)

        return result[:10]

    @staticmethod
    def is_long_form_learnable(keyword: str) -> bool:
        """``keyword`` を ``category="long_form"`` として学習してよいか判定する (pure)。

        long_form は文書(散文)生成の意図シグナルを学習するカテゴリ。ファイルパス片
        (Users / Desktop / aa)・URL 片 (https / soccer.yahoo.co.jp / wcup)・汎用ファイル
        操作語 (出力 / ファイル / 保存) は文書種別を示さないため除外する。CJK の
        文書/意図語 (仕様書 / レポート / ガイドライン 等) は通す。

        割り切り: ASCII のみのトークン (python / Excel / wcup 等) は文書種別シグナルとして
        信頼できないため一律除外する (英語 long_form 語の学習は犠牲にする)。
        ``correction`` / ``tool_routing`` 等の他カテゴリには適用しない (ASCII 語が正当)。
        """
        kw = keyword.strip()
        if len(kw) < 2:
            return False
        if kw.lower() in _LONG_FORM_NONLEARNABLE_EXACT:
            return False
        return not _LONG_FORM_ASCII_FILEISH_RE.match(kw)

    def _enforce_limit(self) -> None:
        """パターン数上限を超えた場合、低重みパターンを削除"""
        if len(self._patterns) <= self.max_patterns:
            return

        # 重みの低い順にソートして超過分を削除
        sorted_patterns = sorted(
            self._patterns.items(),
            key=lambda x: x[1].weight,
        )
        overflow = len(self._patterns) - self.max_patterns
        for key, _ in sorted_patterns[:overflow]:
            del self._patterns[key]
        logger.info("Removed %d low-weight patterns (limit: %d)", overflow, self.max_patterns)

    def get_stats(self) -> dict:
        """統計情報を返す"""
        if not self._patterns:
            return {"count": 0, "categories": {}}

        categories: dict[str, int] = {}
        for p in self._patterns.values():
            categories[p.category] = categories.get(p.category, 0) + 1

        weights = [p.weight for p in self._patterns.values()]
        return {
            "count": len(self._patterns),
            "categories": categories,
            "weight_avg": sum(weights) / len(weights),
            "weight_min": min(weights),
            "weight_max": max(weights),
            "total_hits": sum(p.hit_count for p in self._patterns.values()),
        }

    def save(self, path: str | Path) -> None:
        """JSON ファイルに永続化する (infra 層に委譲)"""
        LearnedPatternRepository.save(self._patterns, path)

    def load(self, path: str | Path) -> None:
        """JSON ファイルからロードする (infra 層に委譲)

        ドメインルール: ストップワード / 空 keyword のエントリは
        ロード時に除外する。
        """
        loaded = LearnedPatternRepository.load(path)
        if loaded is None:
            return

        self._patterns.clear()
        for key, pattern in loaded.items():
            if key and key not in _STOPWORDS:
                self._patterns[key] = pattern
        logger.info("Loaded %d learned patterns from %s", len(self._patterns), path)
