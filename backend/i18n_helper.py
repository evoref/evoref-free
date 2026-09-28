"""多言語対応ヘルパー（バックエンド + CLI 共用）"""

import json
import re
from collections.abc import Sequence
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


def prompt_locale(default: str = "ja") -> str:
    """LLM プロンプトへ埋め込む固定文の言語 (``i18n.prompt_locale``、既定 ja)。

    UI の ``_locale`` とは独立で、config がロードされていない (テスト / CLI
    単体) 場合や未知の値は ``default`` に落とす。呼出側は
    ``LABELS.get(prompt_locale(), LABELS["ja"])`` の形で辞書から引く
    (``backend.free.api.chat.chat._REFERENCE_BLOCK_DIRECTIVES`` と同じ選び方)。
    """
    try:
        from backend.config import get_config

        value = (get_config().get("i18n") or {}).get("prompt_locale", default)
    except Exception:
        return default
    return value if isinstance(value, str) and value else default


# LLM 生成物 prose の出力言語指示に使う言語名 (ja 表記, en 表記)。
_PROSE_LANGUAGE_NAMES: dict[str, tuple[str, str]] = {
    "ja": ("日本語", "Japanese"),
    "en": ("英語", "English"),
}


def prose_language_name(*, english: bool = False) -> str:
    """成果物 prose の出力言語名 (生成時点の locale を反映)。

    日本語テンプレートへは既定 (日本語表記)、英語テンプレートへは
    ``english=True`` (英語表記) で埋め込む。未知 locale は英語へフォールバック。
    import 時に焼き込まず、必ず生成時点で呼ぶこと。
    """
    names = _PROSE_LANGUAGE_NAMES.get(get_locale(), _PROSE_LANGUAGE_NAMES["en"])
    return names[1] if english else names[0]


def msg(key: str, **kwargs) -> str:
    """
    メッセージ取得。ドット区切りのキーでネストをたどる。
    例: msg("cli.welcome", version="0.1.0", mode="create")
    """
    return msg_for_locale(_locale, key, **kwargs)


def msg_for_locale(locale: str, key: str, **kwargs) -> str:
    """``locale`` を明示してメッセージを取得する (``msg`` は UI の ``_locale`` 固定)。

    LLM プロンプトへ埋め込む文 (``agent.event_reminder``) は UI 言語ではなく
    ``prompt_locale()`` に従うため、キーは i18n に置いたまま locale だけ差し替える。
    見つからなければ ``_fallback`` → キー文字列。
    """
    keys = key.split(".")
    value = _resolve(keys, locale)
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


_template_line_cache: dict[tuple[str, ...], tuple[re.Pattern[str], ...]] = {}


def template_line_patterns(
    keys: Sequence[tuple[str, str]], *, group: str,
) -> tuple[re.Pattern[str], ...]:
    """読み込み済みの全 locale の定型文を、その文面だけの 1 行に当たる正規表現にする。

    システム自身が出した定型文 (「{path} に書き込んだ内容:」) を会話履歴・記憶の
    ノートから読み戻すときに使う。**文面の SSOT は i18n** なので正規表現はそこから
    組む。``keys`` は ``(キー, 差込み名)`` の組で、差込み名の位置を名前付きグループ
    ``group`` に、それ以外の差込みを ``.+?`` にする。読み手 (``agent.file_ledger`` の
    書込み報告の復元 / ``memory`` の提示ブロックの除外) で読み方を 1 本にする
    (2026-09-28 レビュー L2)。
    """
    templates: list[tuple[str, str]] = []
    for locale in available_locales():
        for key, slot in keys:
            template = msg_for_locale(locale, key)
            if template and template != key and f"{{{slot}}}" in template:
                templates.append((template, slot))
    cache_key = (group, *(t for t, _ in templates))
    cached = _template_line_cache.get(cache_key)
    if cached is not None:
        return cached
    patterns: list[re.Pattern[str]] = []
    for template, slot in templates:
        body = re.escape(template)
        body = body.replace(re.escape(f"{{{slot}}}"), rf"(?P<{group}>\S.*?)")
        body = re.sub(r"\\\{[a-z_]+\\\}", r".+?", body)
        patterns.append(re.compile(rf"^\s*{body}\s*$"))
    compiled = tuple(patterns)
    _template_line_cache[cache_key] = compiled
    return compiled
