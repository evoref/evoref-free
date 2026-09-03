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
CURRENT_DATETIME_LABEL = "[現在日時]"


def current_datetime_block(guidance: str) -> str:
    """``[現在日時] YYYY-MM-DD (X曜)。<guidance>`` を返す。

    日付スタンプの体裁 (ラベル / 書式 / 曜日の和名) を 1 箇所に集約する。
    後続の指示文は用途ごとに異なる (チャット応答は「過去か未来かの判断」まで
    求め、成果物生成は「相対表現の解釈」のみ) ため引数で受け取る。

    暦日は **ローカル基準**。以前は UTC 基準で出していたが、``run_command``
    系ツールはローカル時刻を返すため、両方が同じプロンプトに並ぶと base が
    「ローカルの日付 + UTC の曜日」を接合して曜日を取り違えた (実インシデント
    2026-08-04 ライブ監査: ローカル 08-04(火) と UTC 08-03(月) から
    「2026年8月4日（月曜）」と回答)。ユーザーが「今日」と言うのはローカルの
    今日なので、暦日をローカルに揃えて接合の余地を無くす。**ズレるのは
    UTC と暦日が食い違う時間帯だけ** (JST なら 00:00-09:00) なので、検証は
    その時間帯か単体テストで行うこと。

    時刻 (HH:MM) やオフセットは載せない。日付の基準を与えるのが役目で、
    時刻が要る質問はツールが答えるため。旧 UTC 版からの差分をラベルと
    暦日の基準だけに留める意図もある。

    内部時刻不変則 (naive datetime 禁止) は ``utc_now_dt()`` を
    ``astimezone()`` で変換することで保つ (tz-aware のまま。純粋関数ではない)。
    """
    from backend.utils import utc_now_dt

    now = utc_now_dt().astimezone()
    weekday = "月火水木金土日"[now.weekday()]
    return (
        f"{CURRENT_DATETIME_LABEL} {now:%Y-%m-%d} ({weekday}曜)。{guidance}"
        f"{_EXPLICIT_DATE_PRECEDENCE}"
    )


#: **ユーザーが明示した日付を、この基準日で上書きしない。** 実インシデント
#: (2026-09-03 ライブ監査 T8#10): 「**11月の3連休**に金沢へ2泊3日」と指定した
#: 行程が、最終出力で「2026年9月4日(金)〜9月6日(日)」= 注入した基準日の翌日から
#: 採番された日付に化けた (しかも 3 連休ですらない)。基準日は相対表現を解くための
#: ものであって、明示された月日を置き換える権限は無い。ラベルと同じ 1 箇所で
#: 全経路 (チャット応答 / 成果物生成) に効かせる。
_EXPLICIT_DATE_PRECEDENCE = (
    "ユーザーが月や日付を明示している場合は、その指定を最優先で使い、"
    "この基準日で置き換えないこと。"
)
