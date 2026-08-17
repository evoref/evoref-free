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
    source: str = "rule"  # "llm" | "rule" | "cartridge" | "learned"
    #: calculate の式に含まれる、対話から辿れない数値リテラル。式を捨てずに
    #: 実行したときだけ入り、回答側でその値の出所を開示させるために使う
    #: (``_suppress_ungrounded_calculate`` 参照)。
    unexplained_numbers: tuple[str, ...] = ()

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}
