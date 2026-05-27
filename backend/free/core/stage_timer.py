"""ステージ別タイミング計測ユーティリティ

リクエスト処理の各ステージ（検索・埋め込み・リランク・LLM）の
経過時間をミリ秒で記録し、requests.jsonl の timing フィールドとして出力する。
"""

from __future__ import annotations

import time


class StageTimer:
    """リクエスト処理のステージ別タイミングを計測する

    使い方:
        timer = StageTimer()
        timer.start("search_ms")
        result = await run_search(...)
        timer.stop("search_ms")
        timer.start("llm_total_ms")
        ...
        timer.stop("llm_total_ms")
        print(timer.to_dict())  # {"search_ms": 123.4, "llm_total_ms": 567.8}
    """

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._stages: dict[str, float] = {}

    def start(self, name: str) -> None:
        """ステージの計測を開始"""
        self._starts[name] = time.monotonic()

    def stop(self, name: str) -> None:
        """ステージの計測を終了し、経過時間をミリ秒で記録"""
        t0 = self._starts.pop(name, None)
        if t0 is not None:
            self._stages[name] = round((time.monotonic() - t0) * 1000, 1)

    def set(self, name: str, ms: float) -> None:
        """計測済みの値を直接セット"""
        self._stages[name] = round(ms, 1)

    def to_dict(self) -> dict[str, float]:
        """計測結果を辞書で返す（ミリ秒単位）"""
        return dict(self._stages)
