"""Reactive エージェント: パターンマッチ + キャッシュで即応答（< 0.1秒）"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from backend.log_config import get_logger

logger = get_logger("agent.reactive")


@dataclass
class ReactiveResponse:
    """Reactive 層の応答"""
    content: str
    source: str  # "pattern" | "cache" | "rag" | "memory"
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


# 日付・時刻クエリ判定パターン (reactive 即応答用)。
# Router の executable_query regex (router.py:101) より絞り込んだ集合: ここに
# マッチしたら datetime tool 経由を経ずに reactive layer が rule-based で
# 即応答する (二重防護の第 1 段)。「今日|明日|昨日」単独追加は誤検出するため
# 「明日.*何月|明日.*日付」のように疑問語と組み合わせた場合のみマッチ。
_DATETIME_QUERY_RE: re.Pattern = re.compile(
    r"(?:何月|何日|何曜日|何時|現在時刻|現在の時刻|現在の時間|"
    r"今日.*日付|明日.*日付|date|time)",
    re.IGNORECASE,
)


# 定型応答パターン: (regex, response)
GREETING_RESPONSES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:こんにち[はわ])\s*[!！。.]?\s*$", re.IGNORECASE), "こんにちは！何かお手伝いできることはありますか？"),
    (re.compile(r"^(?:おはよう(?:ございます)?)\s*[!！。.]?\s*$", re.IGNORECASE), "おはようございます！今日は何をしましょうか？"),
    (re.compile(r"^(?:こんばんは)\s*[!！。.]?\s*$", re.IGNORECASE), "こんばんは！何かお手伝いしましょうか？"),
    (re.compile(r"^(?:やあ|ども|hi|hello|hey)\s*[!！。.]?\s*$", re.IGNORECASE), "こんにちは！お気軽にどうぞ。"),
    (re.compile(r"^(?:ありがと[うございます]*|thanks|thank you)\s*[!！。.]?\s*$", re.IGNORECASE), "どういたしまして！他に何かあればお気軽にどうぞ。"),
    (re.compile(r"^(?:おやすみ(?:なさい)?)\s*[!！。.]?\s*$", re.IGNORECASE), "おやすみなさい。良い夜を！"),
    (re.compile(r"^(?:さようなら|bye|goodbye)\s*[!！。.]?\s*$", re.IGNORECASE), "さようなら！またいつでもどうぞ。"),
]


class ReactiveAgent:
    """Reactive 層: パターンマッチ + キャッシュで即応答

    LLM を呼び出さず、ルールベースのみで応答する。
    目標応答時間: < 0.1秒
    """

    def __init__(self, cache_max_size: int = 100, cache_ttl_sec: int = 300):
        self.cache = ResponseCache(max_size=cache_max_size, ttl_sec=cache_ttl_sec)

    def process(
        self,
        query: str,
        rag_results: list[tuple[str, float, str]] | None = None,
        memory_context: list[dict] | None = None,
    ) -> ReactiveResponse | None:
        """Reactive 層で応答を試みる

        Returns:
            ReactiveResponse: 応答できた場合
            None: 応答できない場合（上位層にエスカレーション）
        """
        start = time.perf_counter()

        # 1. パターンマッチ（挨拶等）
        response = self._pattern_match(query)
        if response is not None:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Pattern match hit: %.1fms", elapsed)
            return ReactiveResponse(content=response, source="pattern", elapsed_ms=elapsed)

        # 1.5. 日付・時刻クエリ → Python 側で datetime を取得して即応答
        # (LLM 呼び出し / tool 呼び出し無し。router.py の executable 経路と
        # 二重防護)
        dt_response = self._handle_datetime_query(query)
        if dt_response is not None:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Datetime query hit: %.1fms", elapsed)
            return ReactiveResponse(
                content=dt_response, source="datetime", elapsed_ms=elapsed,
            )

        # 2. キャッシュ検索
        cached = self.cache.get(self._cache_key(query))
        if cached is not None:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Cache hit: %.1fms", elapsed)
            return ReactiveResponse(content=cached, source="cache", elapsed_ms=elapsed)

        # 3. 高スコア RAG ヒット（score > 0.8）→ 直接応答
        if rag_results:
            top_score = rag_results[0][1]
            if top_score > 0.8:
                content = rag_results[0][2]
                elapsed = (time.perf_counter() - start) * 1000
                self.cache.put(self._cache_key(query), content)
                logger.info("High-score RAG hit (%.2f): %.1fms", top_score, elapsed)
                return ReactiveResponse(content=content, source="rag", elapsed_ms=elapsed)

        # Reactive 層では対応不可
        return None

    def cache_response(self, query: str, response: str) -> None:
        """応答をキャッシュに保存（上位層の応答をキャッシュする用途）"""
        self.cache.put(self._cache_key(query), response)

    def _pattern_match(self, query: str) -> str | None:
        """定型応答パターンで照合"""
        stripped = query.strip()
        for pattern, response in GREETING_RESPONSES:
            if pattern.match(stripped):
                return response
        return None

    def _handle_datetime_query(self, query: str) -> str | None:
        """日付・時刻クエリに対する rule-based 即応答。

        :data:`_DATETIME_QUERY_RE` にマッチすれば
        :func:`backend.utils.utc_now_dt` で UTC tz-aware datetime を取得して
        ローカル時刻に変換し、``「今日は YYYY 年 M 月 D 日 (曜日)、現在時刻は HH:MM です。」``
        形式で返す。LLM 呼び出し無しで数 ms 応答。マッチしない場合は ``None``。
        """
        if not _DATETIME_QUERY_RE.search(query):
            return None
        from backend.utils import utc_now_dt
        now = utc_now_dt().astimezone()  # ローカル tz に変換
        weekday_jp = "月火水木金土日"[now.weekday()]
        return (
            f"今日は {now.year} 年 {now.month} 月 {now.day} 日 "
            f"({weekday_jp}曜日)、現在時刻は {now.hour:02d}:{now.minute:02d} です。"
        )

    def _cache_key(self, query: str) -> str:
        """クエリからキャッシュキーを生成（正規化）"""
        return query.strip().lower()
