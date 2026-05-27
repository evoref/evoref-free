"""Meta-Cognitive ループのステップ結果コンパクター（ルールベースコンテキスト圧縮）"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.log_config import get_logger
from backend.utils import estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("agent.step_compactor")


@dataclass
class StepResult:
    """ツール実行ステップの結果"""

    tool_name: str
    output: str
    iteration: int
    compressed: bool = False


class StepCompactor:
    """Meta-Cognitive ループのステップ結果を段階的に圧縮する。

    LLM 呼び出しゼロ、O(n) で処理完了。
    最新反復は全文保持、古い反復からツール種類別ルールで圧縮する。
    """

    def __init__(self, config: dict, policy: PolicyInterpreter | None = None):
        self._policy = policy
        ac = config.get("agent", {})
        self.rag_lines: int = self._p(
            "step_compaction_rag_lines",
            ac.get("step_compaction_rag_lines", 2),
        )
        self.cmd_head_tail: int = self._p(
            "step_compaction_command_head_tail",
            ac.get("step_compaction_command_head_tail", 5),
        )
        self.file_skeleton_threshold: int = self._p(
            "file_skeleton_threshold", 30,
        )

    def _p(self, key: str, default: int | float) -> int | float:
        """ポリシーからパラメータ取得（フォールバック付き）"""
        if self._policy is None:
            return default
        try:
            return self._policy.get("agent", key)
        except KeyError:
            return default

    def compact(
        self, steps: list[StepResult], budget: int
    ) -> list[StepResult]:
        """ステップ結果リストを予算内に収める。

        最新反復は全文保持、古い反復から段階的に圧縮。
        """
        if not steps:
            return steps

        before_tokens = self._total_tokens(steps)
        if before_tokens <= budget:
            return steps

        # 段階1: 古い反復をツール種類別ルールで圧縮
        result: list[StepResult] = []
        for i, step in enumerate(steps):
            is_latest = i == len(steps) - 1
            result.append(step if is_latest else self._compress_step(step))

        self._total_tokens(result)
        dropped = 0

        # 段階2: それでも超過 → 最古を1行サマリに、さらに削除
        while self._total_tokens(result) > budget and len(result) > 1:
            result[0] = self._to_one_liner(result[0])
            if self._total_tokens(result) > budget:
                result.pop(0)
                dropped += 1

        final_tokens = self._total_tokens(result)

        logger.debug(
            "StepCompactor: before=%d, budget=%d", before_tokens, budget
        )
        logger.debug(
            "StepCompactor: after=%d, steps_kept=%d, steps_dropped=%d",
            final_tokens,
            len(result),
            dropped,
        )

        # E9003: 圧縮後も予算超過（ステップが1件のみの場合）
        if final_tokens > budget and len(result) == 1:
            logger.warning(
                "Step compaction: budget exceeded after full compaction, "
                "dropping all prior steps"
            )

        return result

    def _compress_step(self, step: StepResult) -> StepResult:
        """ツール種類別ルールで圧縮"""
        compressors: dict = {
            "rag_search": self._compress_rag,
            "search_history": self._compress_rag,
            "search_code": self._compress_code_search,
            "run_command": self._compress_command,
            "read_file": self._compress_file,
            "calculate": lambda x: x,
        }
        fn = compressors.get(step.tool_name, self._compress_generic)
        compressed_output = fn(step.output)

        original_tokens = _estimate_tokens(step.output)
        compressed_tokens = _estimate_tokens(compressed_output)
        if original_tokens > 0:
            ratio = round(
                (1 - compressed_tokens / original_tokens) * 100
            )
        else:
            ratio = 0

        logger.debug(
            "StepCompactor: step[%d] %s %d->%d (%d%%)",
            step.iteration,
            step.tool_name,
            original_tokens,
            compressed_tokens,
            ratio,
        )

        return StepResult(
            tool_name=step.tool_name,
            output=compressed_output,
            iteration=step.iteration,
            compressed=True,
        )

    def _compress_rag(self, output: str) -> str:
        """各チャンクの先頭N行 + スコア行のみ"""
        lines = output.split("\n")
        result: list[str] = []
        line_count = 0
        for line in lines:
            if line.startswith("[score:") or line.startswith("---"):
                result.append(line)
                line_count = 0
            elif line_count < self.rag_lines:
                result.append(line)
                line_count += 1
        return "\n".join(result)

    def _compress_code_search(self, output: str) -> str:
        """ファイルパス + マッチ行のみ"""
        return "\n".join(
            line
            for line in output.split("\n")
            if ":" in line and not line.startswith(" ")
        )

    def _compress_command(self, output: str) -> str:
        """先頭M行 + 末尾M行"""
        lines = output.split("\n")
        m = self.cmd_head_tail
        if len(lines) <= m * 2 + 2:
            return output
        omitted = len(lines) - m * 2
        return "\n".join(
            lines[:m] + [f"... ({omitted} lines omitted) ..."] + lines[-m:]
        )

    def _compress_file(self, output: str) -> str:
        """30行以下はそのまま、超過時は定義行スケルトン"""
        lines = output.split("\n")
        if len(lines) <= self.file_skeleton_threshold:
            return output
        skeleton = [
            line
            for line in lines
            if re.match(r"\s*(def |class |import )", line)
        ]
        return f"[file structure ({len(lines)} lines)]\n" + "\n".join(
            skeleton
        )

    def _compress_generic(self, output: str) -> str:
        """先頭100文字 + 文字数"""
        s = output[:100].replace("\n", " ")
        if len(output) > 100:
            return s + f"... ({len(output)} chars)"
        return s

    def _to_one_liner(self, step: StepResult) -> StepResult:
        """1行サマリに圧縮"""
        preview = step.output[:40].replace("\n", " ")
        return StepResult(
            tool_name=step.tool_name,
            output=f"[{step.tool_name} executed: {preview}...]",
            iteration=step.iteration,
            compressed=True,
        )

    def _total_tokens(self, steps: list[StepResult]) -> int:
        """ステップリスト全体のトークン数を推定"""
        return sum(_estimate_tokens(s.output) for s in steps)
