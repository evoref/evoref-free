"""ツール呼び出し判定の値オブジェクト

``ToolCallJudge`` の各判定層とガード列が共有する唯一のデータ構造。ガードを
純粋関数として切り出すために、判定本体 (``tool_call_judge``) とは別モジュール
へ置く (逆向きに参照すると循環 import になる)。
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ToolJudgement:
    """ツール呼び出し判定結果"""
    tool_needed: bool
    tool_name: str = ""
    tool_args: dict = None  # type: ignore[assignment]
    #: 判定を確定させた層。``rule`` (決定論層 / URL リコール) / ``cartridge`` /
    #: ``learned`` / ``recall`` (executable command リコール。curator が再学習から
    #: 除外するために区別する) / ``classifier`` (層 5.9 の文法制約分類・層 5.95 の
    #: 式合成)。
    source: str = "rule"
    #: calculate の式に含まれる、対話から辿れない数値リテラル。式を捨てずに
    #: 実行したときだけ入り、回答側でその値の出所を開示させるために使う
    #: (``_suppress_ungrounded_calculate`` 参照)。
    unexplained_numbers: tuple[str, ...] = ()
    #: calculate の式の組み方が対話の表記と食い違う疑い (``expression_sanity_issues``)。
    #: 接地と同じく式は捨てず、回答側で検算と開示を求める。
    expression_issues: tuple[str, ...] = ()
    #: 日付演算を求められたのに、実行するコマンドが日付演算をしていない
    #: (現在日時だけを返す) か。``unexplained_numbers`` と同じく **格下げはせず**
    #: 印だけ立て、回答側で「ツールで検証していない」ことと数え方の前提を
    #: 開示させる (``_flag_ungrounded_date_math`` / 2026-09-08 監査 F-06)。
    unexplained_date_math: bool = False
    #: このターンで「状態を変える操作を選んだが実行できなかった」か。
    #:
    #: 以前は ``ToolCallJudge`` のインスタンス属性 (``_action_blocked``) に置き、
    #: 呼出側は ``state.tool_call_judge`` を後から読んでいた。判定器は
    #: **プロセス唯一の共有インスタンス** で ``judge()`` の中に ``await`` が
    #: あるため、チャットが 2 本重なると片方の ``judge()`` が他方の読み取り前に
    #: フラグをリセットする。読み手は 4 箇所 (reactive-light ゲート / 経験記録 /
    #: deliberative の注記 2 箇所) にあり、どれも judge() 完了後の別タイミング。
    #: 失われるのは「やっていない操作を完了と言わせない」ガードなので、
    #: リクエスト毎の値は判定結果そのものに載せる。
    action_blocked: bool = False
    #: このターンで「実測しようとしたが実行できなかった」か (上と同じ理由)。
    measurement_blocked: bool = False
    #: 分類器の ``calculate`` が門 (桁の取り違え / 再計算の置き換え・期間の食い違い) で
    #: 落ち、組み直しも回数の置き換えも通らず no_tool にしたか (上と同じ理由で結果に
    #: 載せる)。回答側は「計算を検証できなかった」と注記する (docs/f_03 §3.1 / §3.5)。
    calculation_rejected: bool = False
    #: ``calculation_rejected`` のうち、分類器の式が calculate の式として成り立たず
    #: (未知の名前 / 構文、``validate_expression``) 組み直しも通らなかった回か。回答側は
    #: 「計算ツールが使えない」ではなく「式を組めなかった」と述べ、:attr:`stated_numbers`
    #: を挙げて利用者に確認を求める (2026-10-03 再実行 D08#4、docs/f_03 §3.1 / §3.5)。
    calculation_unbuildable: bool = False
    #: 会話で利用者が述べた数 (出現順・重複なし)。``calculation_unbuildable`` の回だけ入る。
    stated_numbers: tuple[str, ...] = ()
    #: 窓内想起のガードが「答えはこの会話の窓にある」と判断して履歴検索を
    #: 止めたか。回答側は「過去の会話を検索していない」ではなく「対象はこの
    #: 会話の中」と注記する (2026-09-26 監査 C05#5、docs/f_03 §3.5)。
    recall_in_window: bool = False
    #: 尋ねた属性が **すべて** ``[関連する記憶]`` に載っている (尋ねた ⊆ 注入済み)
    #: ので履歴検索を止めたか (``DeliberativeAgent._suppress_redundant_history_search``)。
    #: 回答側は「検索していない。無ければ確認できていないと言え」ではなく「記憶の
    #: 値で答えよ」と注記する (2026-09-27 監査 F9、docs/f_03 §3.5)。
    answered_from_memory: bool = False
    #: 尋ねた属性の **一部だけ** が載った状態で履歴検索を止めたか (抑止の条件は
    #: 尋ねた ∩ 注入済み ≠ ∅)。回答側は「載っている事柄はその値で、載っていない
    #: 事柄は確認できていないと伝えよ」と注記する (独立レビュー H1)。
    partially_answered_from_memory: bool = False

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}
