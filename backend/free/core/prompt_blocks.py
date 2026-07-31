"""プロンプトへ注入するブロックのラベル定数と組み立て関数 (emit 側の SSOT)。

注入ブロックのラベルは **出す側と剥がす側の両方** で使われる。剥がす側
(``agent.meta_cognitive_utils`` の足場除去 / ``learning.fewshot_pool`` の
内部語彙検出) がラベルを独自にリテラル再記述していると、出す側の文言を変えた
瞬間に検出が黙って空振りする。ラベルはここを唯一の出所とし、検出側はこの定数
から派生させること。

``EXISTING_CONTENT_BLOCK_HEADING`` / ``FETCHED_DATA_BLOCK_HEADING``
(``agent.meta_cognitive_utils``) は既にこの形で共有されている。本モジュールは
pillar 非依存の ``core/`` に置くため、``core.inference`` (チャット応答パス) と
``agent.*`` (Meta-Cognitive パス) の双方から参照できる。
"""

from __future__ import annotations

#: 現在日時ブロックの角括弧ラベル。チャット応答パス
#: (``core.inference._current_date_note``) と生成パス
#: (``agent.meta_cognitive._inject_current_date``) の両方が出し、
#: ``agent.meta_cognitive_utils._PROMPT_SCAFFOLD_MARKERS`` が剥がす。
CURRENT_DATETIME_LABEL = "[現在日時 (UTC基準)]"


def current_datetime_block(guidance: str) -> str:
    """``[現在日時 (UTC基準)] YYYY-MM-DD (X曜)。<guidance>`` を返す。

    日付スタンプの体裁 (ラベル / 書式 / 曜日の和名) を 1 箇所に集約する。
    後続の指示文は用途ごとに異なる (チャット応答は「過去か未来かの判断」まで
    求め、成果物生成は「相対表現の解釈」のみ) ため引数で受け取る。

    内部時刻不変則 (naive datetime 禁止) に従い ``utc_now_dt()`` を使う
    (純粋関数ではない)。
    """
    from backend.utils import utc_now_dt

    now = utc_now_dt()
    weekday = "月火水木金土日"[now.weekday()]
    return f"{CURRENT_DATETIME_LABEL} {now:%Y-%m-%d} ({weekday}曜)。{guidance}"
