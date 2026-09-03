"""Reactive エージェント: パターンマッチ + キャッシュで即応答（< 0.1秒）"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from backend.free.core.intent_vocab import (
    GREETING_PUNCTUATION_EN,
    GREETING_PUNCTUATION_JA,
    exact_greeting_pattern,
)
from backend.free.core.locale_patterns import select_locale_variant
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("agent.reactive")


@dataclass
class ReactiveResponse:
    """Reactive 層の応答"""
    content: str
    source: str  # "pattern" | "cache"
    elapsed_ms: float = 0.0


class ResponseCache:
    """LRU キャッシュ: 直近の応答を保持"""

    def __init__(self, max_size: int = 100, ttl_sec: int = 300):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_sec = ttl_sec

    def get(self, key: str) -> str | None:
        """キャッシュから応答を取得（TTL チェック付き）"""
        if key not in self._cache:
            return None
        value, ts = self._cache[key]
        if time.time() - ts > self._ttl_sec:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, key: str, value: str) -> None:
        """キャッシュに応答を格納"""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()


# 定型応答パターン: (regex, i18n キー)。応答文はユーザーに見える UI テキストなので
# backend/i18n/<locale>.json の ``agent.reactive.greeting.*`` から ``msg()`` で引く
# (CLAUDE.md §6 #6)。
#
# **照合は両リスト、文面は locale** (``_pattern_match`` 参照)。パターンは入力の
# 言語に、``msg()`` は UI の言語に従う。両方を locale で切り替えていた頃は、
# locale='en' のまま「こんにちは」と打つと reactive 層が応答せず LLM へ
# エスカレートしていた。
_GREETING_KEY = "agent.reactive.greeting."
GREETING_RESPONSES: list[tuple[re.Pattern, str]] = [
    (re.compile(exact_greeting_pattern(r"こんにち[はわ]", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "hello"),
    (re.compile(exact_greeting_pattern(r"おはよう(?:ございます)?", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "good_morning"),
    (re.compile(exact_greeting_pattern(r"こんばんは", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "good_evening"),
    (re.compile(exact_greeting_pattern(r"やあ|ども|hi|hello|hey", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "casual"),
    (re.compile(exact_greeting_pattern(r"ありがと[うございます]*|thanks|thank you", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "thanks"),
    (re.compile(exact_greeting_pattern(r"おやすみ(?:なさい)?", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "good_night"),
    (re.compile(exact_greeting_pattern(r"さようなら|bye|goodbye", punctuation=GREETING_PUNCTUATION_JA), re.IGNORECASE), _GREETING_KEY + "bye"),
]

# GREETING_RESPONSES の英語版。GUI 左下の言語設定が 'en' の場合のみ使う
# (パターンの語彙が違うため独立したリストにする。文面は同じ i18n キー)。
# "good morning"/"good night" は日本語版の「おはよう」「おやすみ」相当だが、
# 元の JA パターンには hi/hello/hey が既に含まれているため、
# ここでも同様の口語挨拶を独立エントリとして残す。
GREETING_RESPONSES_EN: list[tuple[re.Pattern, str]] = [
    (re.compile(exact_greeting_pattern(r"hi|hello|hey", punctuation=GREETING_PUNCTUATION_EN), re.IGNORECASE), _GREETING_KEY + "hello"),
    (re.compile(exact_greeting_pattern(r"good\s*morning", punctuation=GREETING_PUNCTUATION_EN), re.IGNORECASE), _GREETING_KEY + "good_morning"),
    (re.compile(exact_greeting_pattern(r"good\s*(?:evening|afternoon)", punctuation=GREETING_PUNCTUATION_EN), re.IGNORECASE), _GREETING_KEY + "good_evening"),
    (re.compile(exact_greeting_pattern(r"good\s*night", punctuation=GREETING_PUNCTUATION_EN), re.IGNORECASE), _GREETING_KEY + "good_night"),
    (re.compile(exact_greeting_pattern(r"thanks|thank\s*you", punctuation=GREETING_PUNCTUATION_EN), re.IGNORECASE), _GREETING_KEY + "thanks"),
    (re.compile(exact_greeting_pattern(r"bye|goodbye", punctuation=GREETING_PUNCTUATION_EN), re.IGNORECASE), _GREETING_KEY + "bye"),
]


class ReactiveAgent:
    """Reactive 層: パターンマッチ + キャッシュで即応答

    LLM を呼び出さず、ルールベースのみで応答する。
    目標応答時間: < 0.1秒
    """

    def __init__(self, cache_max_size: int = 100, cache_ttl_sec: int = 300):
        self.cache = ResponseCache(max_size=cache_max_size, ttl_sec=cache_ttl_sec)

    def process(self, query: str) -> ReactiveResponse | None:
        """Reactive 層で応答を試みる (ルールベース + キャッシュのみ、LLM ゼロ)。

        Returns:
            ReactiveResponse: 応答できた場合 (挨拶パターン / キャッシュ命中)
            None: 応答できない場合（上位層にエスカレーション）
        """
        start = time.perf_counter()

        # 1. パターンマッチ（挨拶等）
        response = self._pattern_match(query)
        if response is not None:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Pattern match hit: %.1fms", elapsed)
            return ReactiveResponse(content=response, source="pattern", elapsed_ms=elapsed)

        # 2. キャッシュ検索 (上位層の応答を cache_response() で蓄積したもの)
        cached = self.cache.get(self._cache_key(query))
        if cached is not None:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Cache hit: %.1fms", elapsed)
            return ReactiveResponse(content=cached, source="cache", elapsed_ms=elapsed)

        # Reactive 層 (ルールベース) では対応不可 → 上位層へエスカレート
        return None

    def cache_response(self, query: str, response: str) -> None:
        """応答をキャッシュに保存（上位層の応答をキャッシュする用途）"""
        self.cache.put(self._cache_key(query), response)

    def _pattern_match(self, query: str) -> str | None:
        """定型応答パターンで照合

        **照合は JA / EN 両方**、**文面は GUI locale** で決める。パターンは入力の
        言語に、``msg()`` は UI の言語に従うべきもので、両方を locale で切り替えて
        いたため locale='en' のまま「こんにちは」と打つと reactive 層が応答せず
        LLM へエスカレートしていた (0.1 秒で返せる挨拶に往復が乗る)。

        locale 側のリストを **先に** 見るのは、両リストが hi / hello / thanks /
        bye を別々の i18n キーで持つため — 順序を入れ替えると locale='en' の
        "hello" が JA 側の ``casual`` キーに当たって文面が変わる。
        locale が優先順位を、union が網羅を決める。
        """
        stripped = query.strip()
        primary, secondary = select_locale_variant(
            (GREETING_RESPONSES, GREETING_RESPONSES_EN),
            (GREETING_RESPONSES_EN, GREETING_RESPONSES),
        )
        for pattern, key in (*primary, *secondary):
            if pattern.match(stripped):
                return msg(key)
        return None

    def _cache_key(self, query: str) -> str:
        """クエリからキャッシュキーを生成（正規化）"""
        return query.strip().lower()
