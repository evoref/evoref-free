"""Agentic ループ: LLM のツール呼び出しを解析・実行するループ制御"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.free.agent.meta_cognitive_utils import (
    iter_balanced_brace_substrings,
    try_parse_tool_dict,
)
from backend.log_config import get_logger

logger = get_logger("agent.agentic_loop")


@dataclass
class LoopStep:
    """ループの1ステップ"""
    step: int
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    llm_response: str = ""
    is_final: bool = False


@dataclass
class LoopResult:
    """ループの実行結果"""
    final_response: str
    steps: list[LoopStep] = field(default_factory=list)
    total_steps: int = 0
    hit_limit: bool = False


class AgenticLoop:
    """LLM のツール呼び出しを反復的に解析・実行するループ

    MetaCognitiveAgent が使用する。ステップ上限で無限ループを防止。
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    async def run(
        self,
        messages: list[dict],
        llm_client,
        tools_registry,
        on_step=None,
    ) -> LoopResult:
        """ループを実行

        Args:
            messages: 初期メッセージ配列
            llm_client: LocalClient
            tools_registry: ToolsRegistry
            on_step: ステップごとのコールバック (step: LoopStep) -> None

        Returns:
            LoopResult
        """
        steps: list[LoopStep] = []
        current_messages = list(messages)

        for step_num in range(1, self.max_steps + 1):
            # LLM に推論させる
            result = await llm_client.generate(
                current_messages, stream=False, max_tokens=512,
                id_slot=getattr(llm_client, 'background_slot', -1),
            )
            text = result["choices"][0]["message"]["content"].strip()

            # ツール呼び出しをパース
            tool_call = self._parse_tool_call(text)

            if tool_call is None:
                # ツール呼び出しではない → 最終応答
                step = LoopStep(
                    step=step_num,
                    llm_response=text,
                    is_final=True,
                )
                steps.append(step)
                if on_step:
                    await _call_on_step(on_step, step)

                logger.info("Loop completed at step %d (final response)", step_num)
                return LoopResult(
                    final_response=text,
                    steps=steps,
                    total_steps=step_num,
                )

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # ツール実行
            tool_result = await self._execute_tool(
                tools_registry, tool_name, tool_args,
            )

            step = LoopStep(
                step=step_num,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                llm_response=text,
            )
            steps.append(step)

            if on_step:
                await _call_on_step(on_step, step)

            logger.info("Step %d: tool=%s", step_num, tool_name)

            # ツール結果をメッセージに追加して再ループ
            current_messages.append({"role": "assistant", "content": text})
            current_messages.append({
                "role": "user",
                "content": self._format_tool_result(tool_name, tool_result),
            })

        # ステップ上限到達
        logger.warning("Loop hit step limit (%d)", self.max_steps)
        final_text = steps[-1].tool_result if steps else "Step limit reached."
        return LoopResult(
            final_response=final_text,
            steps=steps,
            total_steps=self.max_steps,
            hit_limit=True,
        )

    async def _execute_tool(
        self, tools_registry, name: str, args: dict,
    ) -> str:
        """ツールを実行し結果を文字列で返す"""
        if not tools_registry.has(name):
            return f"Error: Unknown tool '{name}'"
        try:
            result = await tools_registry.execute(name, **args)
            return str(result)
        except Exception as e:
            logger.warning("Tool execution error: %s - %s", name, e)
            return f"Error: {e}"

    def _parse_tool_call(self, text: str) -> dict | None:
        """LLM 応答からツール呼び出し JSON をパース

        旧実装はネスト 7 だったが、共通ヘルパー
        ``iter_balanced_brace_substrings`` / ``try_parse_tool_dict`` に
        切り出してネスト 2 まで平坦化済み。
        """
        # 全体が JSON の場合
        direct = try_parse_tool_dict(text)
        if direct is not None:
            return direct

        # テキスト中に埋め込まれた JSON を探す
        for candidate in iter_balanced_brace_substrings(text):
            parsed = try_parse_tool_dict(candidate)
            if parsed is not None:
                return parsed

        return None

    def _format_tool_result(self, name: str, result: str) -> str:
        """ツール結果をメッセージ用に整形"""
        return f"Tool '{name}' returned:\n{result}"


async def _call_on_step(on_step, step: LoopStep) -> None:
    """on_step コールバックを呼び出す（async/sync 両対応）"""
    import inspect
    if inspect.iscoroutinefunction(on_step):
        await on_step(step)
    else:
        on_step(step)
