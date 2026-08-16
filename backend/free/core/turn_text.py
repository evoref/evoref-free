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
"""

from __future__ import annotations

from collections.abc import Callable


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
