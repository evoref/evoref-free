"""分類器スロットの共有接頭辞 (KV 接頭辞キャッシュを 1 スロットで複数タスクが保つための規約)。

llama-server の接頭辞キャッシュはスロットごとに「直前のプロンプト」との最長共通
接頭辞しか再利用せず、hybrid recurrent モデル (Qwen3.5/3.8) では **system
メッセージが前回と一致しないと 0** になる (実測 2026-09-11: system の末尾を延ばした
だけの 2 プロンプトは token LCP 438 でも cache_n=0、system を同一にして user を
替えると 438 再利用)。分類器スロット (``classifier_slot``) はツール分類器・日付
抽出器・式合成 (EvorefLoop) と、チャット応答パスで同期発火する補助タスク
(``AuxClient`` の ``CHAT_PATH_PURPOSES``) が共有するため、**同じスロットの全
プロンプトを同じ system で始める** 必要がある。

規約:

- system は分類器の system (役割 + ツールメニュー) と **byte 一致** させる。
- 各タスク本来の system は、区切り文 (``separator``) を頭に付けた **user**
  メッセージとして system 直後に置く。
- 接頭辞の供給元は分類器 (ツールメニューを持つ EvorefLoop 側) で、
  ``LocalClient.set_classifier_slot_prefix`` を通じて公開する。``AuxClient`` は
  分類器スロットへ送る直前に :func:`apply_shared_prefix` で整形する。
"""

from __future__ import annotations

from backend.free.core.locale_patterns import select_locale_variant

#: 接頭辞として付けるときの区切り。上のツール一覧は参照情報でこの判定では
#: 使わないと明示する。
SLOT_PREFIX_SEPARATOR_JA = (
    "[別の判定] 上のツール一覧は共有の前置き (参照情報) で、この判定では使わない。"
    "以下の指示に従うこと。\n\n"
)
SLOT_PREFIX_SEPARATOR_EN = (
    "[Separate task] The tool list above is a shared preamble (reference only); "
    "do not use it here. Follow the instructions below.\n\n"
)


def slot_prefix_separator() -> str:
    """現在の locale の区切り文。"""
    return select_locale_variant(SLOT_PREFIX_SEPARATOR_JA, SLOT_PREFIX_SEPARATOR_EN)


def shared_prefix_messages(prefix: str, task_system: str) -> list[dict]:
    """共有 system + (区切り + タスク指示) の user、の 2 メッセージ。"""
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": slot_prefix_separator() + task_system},
    ]


def apply_shared_prefix(prefix: str, messages: list[dict]) -> list[dict]:
    """``messages`` を共有接頭辞で始まる形へ整形して返す (冪等)。

    - 先頭が既に ``prefix`` と一致する system ならそのまま返す
      (分類器 / 日付抽出器のように整形済みのプロンプト)。
    - 先頭が別の system なら、その内容を区切り付き user に降ろし、system を
      ``prefix`` に差し替える。
    - system が無ければ ``prefix`` の system を挿入し、続く最初の user の先頭に
      区切りを付ける。

    元のリストは変更しない。
    """
    if not prefix or not messages:
        return messages
    first = messages[0]
    if first.get("role") == "system":
        if first.get("content") == prefix:
            return messages
        return [
            *shared_prefix_messages(prefix, str(first.get("content") or "")),
            *messages[1:],
        ]
    rest = [dict(m) for m in messages]
    if rest[0].get("role") == "user":
        rest[0]["content"] = slot_prefix_separator() + str(rest[0].get("content") or "")
    else:
        rest.insert(0, {"role": "user", "content": slot_prefix_separator().rstrip()})
    return [{"role": "system", "content": prefix}, *rest]
