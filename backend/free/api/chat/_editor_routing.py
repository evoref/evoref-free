"""クリエイトモードの生成コードをエディタ／チャットのどちらへ出すか判定する。

クリエイトモードでは生成したプログラムを既定でエディタエリアへ流す
(``"editor"``)。ただしユーザーが「エディタには出さずチャットに表示して」と
明確に指示した場合のみチャットに残す (``"chat"``)。

判定は :mod:`backend.free.api.chat.chat` の ``chat()`` から呼ばれ、結果は
SSE ``editor_route`` フレームでフロントエンドへ通知される。LLM を介さない
決定的なキーワード判定で、チャット応答パスにレイテンシを足さない。
"""

from __future__ import annotations

import re

__all__ = ["detect_editor_route"]


# 「エディタに出さずチャットに表示して」を示すヒント。"chat" へ分岐するシグナル。
# 日本語: エディタ出力の否定 (「エディタ(に|へ|は)…(出力|表示)しない/不要/なし」)、
#         または「チャット/ここ/この場/画面(に|へ/で)…(表示/出力/出して/貼って/返して)」。
# 英語: in (the) chat / here、don't / no / without (use) (the) editor など。
_CHAT_ROUTE_HINT_RE = re.compile(
    r"(?:"
    r"エディタ(?:エリア)?(?:に|へ|は|を)[^。!?\n]{0,12}?(?:出力|表示|書き?出)[^。!?\n]{0,6}?(?:しない|せず|不要|なし|不用)"
    r"|エディタ(?:エリア)?(?:は|を)?[^。!?\n]{0,6}?(?:使わ(?:ない|ず)|不要|なし)"
    r"|(?:チャット|ここ|この場|画面)(?:エリア)?(?:に|へ|で|の方に)[^。!?\n]{0,12}?(?:表示|出力|出し|貼っ|返し|書い)"
    r"|in\s+(?:the\s+)?chat"
    r"|(?:don'?t|do\s+not|no|without)\s+(?:use\s+)?(?:the\s+)?editor"
    r"|here\s+in\s+(?:the\s+)?chat"
    r")",
    re.IGNORECASE,
)


def detect_editor_route(message: str) -> str:
    """生成コードをエディタ／チャットのどちらへ出すかを判定する。

    Args:
        message: ユーザー指示文

    Returns:
        ``"chat"`` (エディタ出力不要の明示指示がある場合) または
        ``"editor"`` (既定)。
    """
    if _CHAT_ROUTE_HINT_RE.search(message):
        return "chat"
    return "editor"
