"""現在ターンの user メッセージへテキストを付与する共通プリミティブ。

会話パイプラインは「最後の user メッセージを探して content を書き換える」
操作を複数箇所で行う (動的ブロックの前置 / 各種注記の後置 / ツール実行結果の
後置)。以前はこの走査と置換が 4 箇所で独立に書かれており、要素を mutate する
かどうか・元の dict のキーを保つかどうかも実装ごとに揺れていた。

**なぜ system ではなく user 末尾か**: system へ足すと llama-server の prefix KV
キャッシュが毎ターン無効化される (system は静的に保つ設計)。加えて、生クエリの
直後は小型モデルの指示追従が最も効く位置でもある。

**付与順序の契約** (この順に後ろへ積まれる):

1. 動的ブロック (few-shot / RAG / 記憶) — ``prepend_to_last_user`` で **前置**。
   生クエリは末尾に残る。
   例外: 生クエリが直前の出力を指す照応 (「上の内容を」「さっきの話」) を含む
   ターンだけは ``append_to_last_user`` で **生クエリの後ろ** へ回す。前置すると
   注入ブロックが指示語の参照先を奪うため。system へは回さない (prefix KV
   キャッシュが全損する)。
2. 最新ターン切り詰めの注記 / 現在日付 / 人格 / 文字数上限の注記 —
   ``append_to_last_user`` で後置
   (``core.inference.build_messages`` / ``build_messages_for_loop``)。
3. ツール実行結果 + 話題再フォーカス — 後置 (``agent.deliberative``)。

3 は 1・2 が済んだ messages を後から受け取る (ツール実行が完了して初めて内容が
決まる) ため、単一の組み立て器で一度に組むことはできない。順序はこの契約に
依存しており、生クエリが動的ブロックと注記の間に挟まる形を崩さないこと。

**最後の user メッセージのレイアウト** (送信時ガード
``LocalClient._enforce_context_budget`` が境界単位で切り詰めるための契約)::

    [動的ブロック + DYNAMIC_CONTEXT_DELIMITER]   ← 前置 (既定)
    生クエリ
    [DYNAMIC_CONTEXT_TRAILING_DELIMITER + 動的ブロック]   ← 後置 (照応ターンのみ)
    [注記 …]                                     ← 日付 / 人格 / 文字数 / 計測
    [TOOL_RESULT_HEADER + ツール結果 + 接地指示]   ← deliberative

境界マーカーは本モジュールが唯一の出所で、:func:`split_last_user` が
この 3 層へ分解する。生クエリと注記の間にはマーカーが無い (注記は
ユーザー発言の直後に置くことに意味があり、区切り語を足すと毎ターン
再プリフィルされる) ので、前置レイアウトでは ``raw_query`` に注記が含まれる。
"""

from __future__ import annotations

from collections.abc import Callable

# 動的コンテキストブロックと生クエリの境界に挟む固定文 (``i18n.prompt_locale`` 別)。
# few-shot 例 / 参考情報をユーザー発言と混同させないための区切り。
# 「無関係なら言及せず自分の知識で普通に答える」等の指示本文は
# ``agent.prompt_manager.REFERENCE_BLOCK_DIRECTIVES`` (system 側) が持つ。
# **ここに置いた文字は毎ターン再プリフィルされる** ので短く保つ
# (``core.tests.test_inference`` がトークン上限を固定している)。
DYNAMIC_CONTEXT_DELIMITERS: dict[str, str] = {
    "ja": "\n\n---\n[ここまで参考枠 / ここからユーザーの発言]\n",
    "en": "\n\n---\n[End of reference material / user message follows]\n",
}
#: 既定 (ja) の区切り。旧 ``core.inference._DYNAMIC_CONTEXT_DELIMITER`` と同一。
DYNAMIC_CONTEXT_DELIMITER = DYNAMIC_CONTEXT_DELIMITERS["ja"]

# 照応を含むターンで動的ブロックを **生クエリの後ろ** へ回すときの区切り。
# 前置版と違い「上の」「さっき」の参照先が直前のやり取りであることを明示する
# 必要がある (後置しても、指示語が直後のブロックを掴む余地は残るため)。この
# 指示は **配置に依存する** ので system へは移さない。発火は実測 232 ターン中
# 2 件 (1%) で、再プリフィルの寄与も小さい。
DYNAMIC_CONTEXT_TRAILING_DELIMITERS: dict[str, str] = {
    "ja": (
        "\n\n---\n"
        "以下はシステムが用意した参考枠であり、ユーザーの発言ではありません。"
        "上のユーザー発言に含まれる「上の」「先ほど」「さっき」「直前の」等の指示語は、"
        "この参考枠ではなく **直前までのやり取り** を指します。\n\n"
    ),
    "en": (
        "\n\n---\n"
        "The following is reference material prepared by the system, not the "
        "user's message. Words such as \"the above\", \"earlier\", or \"the "
        "previous\" in the user's message above refer to **the preceding "
        "exchange**, not to this reference material.\n\n"
    ),
}
DYNAMIC_CONTEXT_TRAILING_DELIMITER = DYNAMIC_CONTEXT_TRAILING_DELIMITERS["ja"]

#: ツール実行結果ブロックの見出し (``agent.deliberative`` が後置する)。
#: 接地指示の本文が「上記の ## ツール実行結果 は…」とこの見出しを名指しする
#: ため、locale で変えない (``[関連する記憶]`` / ``[参考情報]`` と同じ扱い)。
TOOL_RESULT_HEADER = "\n\n## ツール実行結果\n"

#: 注入ブロックの行頭ラベル (両 locale)。``[関連する記憶]`` の各行は
#: ``- (過去の記録) …`` / ``- （訂正済み） …`` のように装飾されるため、
#: ``[参考情報]`` 側の生テキストと突き合わせるときはここに載るラベルを剥がす
#: (``core.inference._normalize_for_frame_dedup``)。新しいラベル (locale 版を
#: 含む) を出す側はここへ追記すること — 検出側はこの集合から派生する。
FRAME_LINE_LABELS: frozenset[str] = frozenset({
    "過去の記録", "past record", "past records",
    "訂正済み", "corrected",
    "競合", "conflict", "conflicting",
})


def _all_delimiters(table: dict[str, str]) -> list[str]:
    """locale 辞書の区切りを長い順に並べる (接頭辞関係があっても長い方を先に試す)。"""
    return sorted(set(table.values()), key=len, reverse=True)


def split_last_user(content: str) -> tuple[str, str, str]:
    """最後の user メッセージの content を ``(prefix_blocks, raw_query, suffix)`` へ分解する。

    ``prefix_blocks + raw_query + suffix == content`` を常に満たす (純粋関数)。

    - ``prefix_blocks``: 前置された動的ブロック **と区切り** (無ければ空)。
    - ``raw_query``: ユーザーの生クエリ。前置レイアウトでは後置注記を含む
      (注記との間にマーカーが無いため)。後置レイアウト / ツール結果ありでは
      その直前まで。
    - ``suffix``: 後置の区切り + 動的ブロック、注記、``TOOL_RESULT_HEADER``
      以降のすべて (無ければ空)。

    送信時ガードはこの順で落とす: ``prefix_blocks`` (参考枠) → ``suffix`` の
    ツール結果本文の切り詰め → 最後まで ``raw_query`` を残す。
    """
    head, suffix = content, ""
    idx = content.find(TOOL_RESULT_HEADER)
    if idx >= 0:
        head, suffix = content[:idx], content[idx:]
    for delim in _all_delimiters(DYNAMIC_CONTEXT_TRAILING_DELIMITERS):
        pos = head.find(delim)
        if pos >= 0:
            return "", head[:pos], head[pos:] + suffix
    for delim in _all_delimiters(DYNAMIC_CONTEXT_DELIMITERS):
        pos = head.rfind(delim)
        if pos >= 0:
            cut = pos + len(delim)
            return head[:cut], head[cut:], suffix
    return "", head, suffix


def edit_last_user(
    messages: list[dict], transform: Callable[[str], str],
) -> bool:
    """最後の user メッセージの content に ``transform`` を適用する。

    該当要素を **新しい dict で置換** する (入力要素は mutate しない)。
    ``_trim_history`` が返す未圧縮ターンは呼び出し元の history と dict 参照を
    共有するため、in-place 更新すると会話履歴そのものを汚染する。
    ``role`` 以外のキーも保つ。

    Returns:
        user メッセージが見つかり書き換えたら ``True``。見つからなければ
        ``False`` (呼び出し側が system への fallback 等を判断する)。
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            original = messages[i].get("content", "")
            messages[i] = {**messages[i], "content": transform(original)}
            return True
    return False


def append_to_last_user(
    messages: list[dict], text: str, *, separator: str = "\n\n",
) -> bool:
    """最後の user メッセージ末尾へ ``text`` を追記する。"""
    return edit_last_user(messages, lambda c: f"{c}{separator}{text}")


def prepend_to_last_user(
    messages: list[dict], text: str, *, separator: str,
) -> bool:
    """最後の user メッセージ先頭へ ``text`` を前置する (生クエリは末尾に残る)。"""
    return edit_last_user(messages, lambda c: f"{text}{separator}{c}")
