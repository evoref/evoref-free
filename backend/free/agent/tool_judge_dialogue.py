"""判定へ渡す会話文脈の取り出し (純粋関数)

判定プロンプト / グラウンディング検証 / ネイティブ tool calling が共有する
「会話履歴をどこまで・どう切り出すか」の定義を 1 箇所に集める。
"""

from __future__ import annotations

#: 判定プロンプトへ載せる会話 1 メッセージあたりの文字数上限。切り詰め側と
#: 復元側で同じ定数を共有する (別々に持つと片方の変更で復元が効かなくなる)。
_JUDGE_CONTEXT_CHARS = 100
#: 層5.2 の数値グラウンディングに使う直近ターン数。長く取るほど無関係な数値を
#: 拾いやすくなるため短く保つ。
_CALCULATE_CONTEXT_TURNS = 4
def _dialogue_text(
    conversation: list[dict] | None, turns: int | None = None,
) -> str:
    """会話本文を連結して返す (純粋関数)。

    「対話に現れた数値」を数えるために使う。role は問わない (換算率は
    アシスタント発話側に出る)。``turns`` を渡すと末尾その数のターンに絞る。
    """
    if not conversation:
        return ""
    window = conversation[-turns:] if turns else conversation
    parts = [
        str(turn.get("content") or "")
        for turn in window
        if isinstance(turn, dict)
    ]
    return "\n".join(p for p in parts if p)


def _recent_dialogue_text(conversation: list[dict] | None) -> str:
    """直近 ``_CALCULATE_CONTEXT_TURNS`` ターンの本文を連結して返す (純粋関数)。

    層5.2 の事前フィルタと合成式のグラウンディング検証で使う。式を合成する
    層なので、捏造の余地を絞るために窓は狭く保つ。
    """
    return _dialogue_text(conversation, _CALCULATE_CONTEXT_TURNS)


def _recent_dialogue_messages(
    conversation: list[dict] | None,
    turns: int = _CALCULATE_CONTEXT_TURNS,
) -> list[dict]:
    """直近 ``turns`` 件を messages 配列として返す (純粋関数)。

    ネイティブ tool calling へ渡す最小の文脈。「そのファイルを読んで」のような
    照応をモデルが解けるように直近だけ載せ、prefill を膨らませない
    (1 メッセージ ``_JUDGE_CONTEXT_CHARS`` 文字で切り詰め)。
    """
    if not conversation:
        return []
    messages: list[dict] = []
    for turn in conversation[-turns:]:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        messages.append({"role": role, "content": content[:_JUDGE_CONTEXT_CHARS]})
    return messages
