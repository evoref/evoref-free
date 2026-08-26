"""判定へ渡す会話文脈の取り出し (純粋関数)

判定プロンプト / グラウンディング検証 / ネイティブ tool calling が共有する
「会話履歴をどこまで・どう切り出すか」の定義を 1 箇所に集める。
"""

from __future__ import annotations

#: 判定プロンプトへ載せる会話 1 メッセージあたりの文字数上限。切り詰め側と
#: 復元側で同じ定数を共有する (別々に持つと片方の変更で復元が効かなくなる)。
_JUDGE_CONTEXT_CHARS = 100
#: 計算クエリの判定に使う直近ターン数。長く取るほど無関係な数値を拾いやすく
#: なるため短く保つ。
#:
#: ⚠ **これは「判定」用であって「合成」用ではない。** 式合成 (層5.95) にこの
#: 窓を渡すと、基準値が窓の外にある差分クエリで必ず間違える。実インシデント
#: (2026-08-26): 在庫 12→9→14→12 を追った後の「最初の在庫からいくつ減りま
#: したか？」で合成器に直近 4 ターンしか渡らず、窓内の最古の値が 14 だった
#: ため ``14 - 12`` を合成して 4/4 とも「2台」と誤答した (正解 0)。
#: 合成には :data:`_SYNTHESIS_CONTEXT_TURNS` を使うこと。
_CALCULATE_CONTEXT_TURNS = 4

#: 式合成 (層5.95) に渡す会話のターン数。
#:
#: **合成する範囲と検証する範囲を一致させる。** 合成した式は
#: ``_ungrounded_numbers`` が **会話全体** を許可リストにして検証する
#: (``_suppress_ungrounded_calculate`` 参照)。合成側だけ窓を狭めると、
#: 検証は通るのに被演算子を見ていない式が作られる — 上記の ``14 - 12`` が
#: まさにそれで、14 も 12 も会話にある実数値なのでグラウンディング検証は
#: 素通りした。「捏造でないこと」は「正しい被演算子であること」を意味しない。
#:
#: 「最初の〜」は会話の先頭を指しうるので上限は実質的に会話全体。1 メッセージ
#: あたり ``_JUDGE_CONTEXT_CHARS`` 文字へ切り詰めた上での件数上限なので、
#: 長い会話でもプロンプトは有界に収まる。層5.95 は分類器が no_tool を返した
#: 差分クエリでしか走らないため、prefix キャッシュへの影響も限定的。
_SYNTHESIS_CONTEXT_TURNS = 120
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
