"""多言語対応ヘルパー（バックエンド + CLI 共用）"""

import json
from pathlib import Path

_BASE_DIR = Path(__file__).parent / "i18n"
_messages: dict[str, dict] = {}
_locale: str = "ja"
_fallback: str = "ja"


def init_i18n(locale: str = "ja", fallback: str = "ja") -> None:
    """config.yaml 読込み後に呼び出す"""
    global _locale, _fallback
    _locale = locale
    _fallback = fallback
    for f in _BASE_DIR.glob("*.json"):
        lang = f.stem
        _messages[lang] = json.loads(f.read_text(encoding="utf-8"))


def set_locale(locale: str) -> None:
    global _locale
    _locale = locale


def get_locale() -> str:
    return _locale


def msg(key: str, **kwargs) -> str:
    """
    メッセージ取得。ドット区切りのキーでネストをたどる。
    例: msg("cli.welcome", version="0.1.0", mode="coding")
    """
    keys = key.split(".")
    value = _resolve(keys, _locale)
    if value is None:
        value = _resolve(keys, _fallback)
    if value is None:
        return key
    for k, v in kwargs.items():
        value = value.replace(f"{{{k}}}", str(v))
    return value


def _resolve(keys: list[str], locale: str) -> str | None:
    obj = _messages.get(locale)
    if obj is None:
        return None
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj if isinstance(obj, str) else None


def available_locales() -> list[str]:
    """利用可能なロケール一覧"""
    return sorted(_messages.keys())
