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

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}
