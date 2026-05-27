"""会話履歴の共通ユーティリティ

日時パース・フォーマット、文字列スニペット生成など、
history_manager / api / cli で横断的に使われるヘルパー関数。
"""

from __future__ import annotations

from datetime import datetime


def parse_iso(s: str) -> datetime | None:
    """ISO 8601 文字列を datetime にパース

    空文字列やパース失敗時は None を返す。
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_datetime(iso_str: str) -> str:
    """ISO 8601 → 読みやすい日時文字列（例: 2026-03-07 14:30）"""
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_str[:16]


def format_duration(seconds: int) -> str:
    """秒数 → 読みやすい時間文字列（例: 2m 5s）"""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


def snippet_around(text: str, query: str, context: int = 40) -> str:
    """query の前後 context 文字を切り出してスニペットを生成

    テキスト内に query が見つからない場合は空文字列を返す。
    """
    pos = text.lower().find(query.lower())
    if pos < 0:
        return ""
    start = max(0, pos - context)
    end = min(len(text), pos + len(query) + context)
    snippet = text[start:end].replace("\n", " ")
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"
