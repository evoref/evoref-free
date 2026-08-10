"""エディタタブ名 (ファイル名) のアシスト導出ヘルパ

クリエイトモードで生成したコード/仕様書を Pro エディタへタブ表示する際、
タブ名 (= ファイル名) が日本語にならないよう、生成内容から **ASCII snake_case
の stem (拡張子なし)** をアシストモデルで導出する。long_form 経路
(`api/chat/chat_streaming.py`) と meta_cognitive 経路
(`agent/meta_cognitive.py`) の双方がこのヘルパを共用する。

設計:
- アシスト未接続 (degraded) / 呼出失敗 / 不正応答でも、言語別の決定論的
  ASCII フォールバック stem を返し、**常に非空・ASCII** を保証する。
- LLM が日本語/記号/拡張子を返しても ``_to_ascii_slug`` で正規化する。
- 拡張子は付与しない (言語に応じた拡張子は呼出側が付ける)。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.free.llm.assist_client import assist_ready
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient

logger = get_logger("llm.editor_filename")

# ASCII slug 化: 英数字 + `_` + `-` 以外を `_` 化 (SPLIT モードの
# `_slug_for_split_file` と同方針)。
_ASCII_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_STEM_LEN = 32
_CONTENT_PREVIEW_LEN = 1500
_HINT_PREVIEW_LEN = 200

# 言語別の決定論的フォールバック stem (アシスト不在/失敗時)。
_FALLBACK_STEM_BY_LANGUAGE: dict[str, str] = {
    "markdown": "document",
    "python": "script",
    "typescript": "module",
    "javascript": "script",
    "html": "index",
    "css": "styles",
    "json": "data",
    "yaml": "config",
    "xml": "data",
    "sql": "query",
    "bash": "script",
}
_DEFAULT_FALLBACK_STEM = "output"

_PROMPT_TEMPLATE = """\
以下のファイル内容に最適な英語のファイル名 (拡張子なし) を 1 つ提案してください。

制約:
- 英小文字 + 数字 + アンダースコア (snake_case) のみ。日本語・空白・記号・拡張子は禁止。
- 32 文字以内。内容を端的に表す名前にする
  (例: grid_management / game_of_life / api_client / user_service / data_loader)。
- 言語: {language}

【ユーザー指示】{hint}

【ファイル内容 (先頭抜粋)】
{content}

JSON のみ出力: {{"file_name": "..."}}"""


def _fallback_stem(language: str) -> str:
    """言語別の決定論的フォールバック stem を返す。"""
    return _FALLBACK_STEM_BY_LANGUAGE.get(
        (language or "").lower(), _DEFAULT_FALLBACK_STEM,
    )


def _to_ascii_slug(raw: str) -> str:
    """任意文字列を ASCII snake_case stem に正規化する (空なら ``""``)。"""
    slug = _ASCII_SAFE_RE.sub("_", raw or "").strip("_-")
    slug = slug[:_MAX_STEM_LEN].rstrip("_-")
    # 数字始まりは識別子として扱いづらいため接頭辞を付ける。
    if slug and slug[0].isdigit():
        slug = f"file_{slug}"
    return slug


async def derive_editor_filename_stem(
    assist_client: "AssistModelClient | None",
    *,
    content: str,
    hint: str,
    language: str,
) -> str:
    """生成内容から ASCII snake_case の stem (拡張子なし) を導出する。

    Args:
        assist_client: アシストモデルクライアント。``None`` (degraded) なら
            言語別フォールバック stem を返す。
        content: 生成本文 (先頭 1500 字をプロンプトに渡す)。
        hint: ユーザー指示文 (補助コンテキスト)。
        language: 言語識別子 (``python`` / ``markdown`` 等)。フォールバック
            stem の選択に使う。

    Returns:
        ASCII snake_case の stem。常に非空。拡張子は含まない。
    """
    fallback = _fallback_stem(language)
    if not assist_ready(assist_client, "editor_filename"):
        return fallback
    prompt = _PROMPT_TEMPLATE.format(
        language=language or "text",
        hint=(hint or "").strip()[:_HINT_PREVIEW_LEN],
        content=(content or "")[:_CONTENT_PREVIEW_LEN],
    )
    try:
        result = await assist_client.generate_json(
            prompt,
            max_tokens=24,
            temperature=0.2,
            purpose="editor_filename",
        )
    except Exception as e:
        logger.warning(
            "editor_filename derivation failed (%s); using fallback '%s'",
            type(e).__name__, fallback,
        )
        return fallback
    stem = _to_ascii_slug(str(result.get("file_name", "")))
    return stem or fallback
