"""ツール定義・登録・実行レジストリ"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.log_config import get_logger

logger = get_logger("agent.tools_registry")


@dataclass
class ToolDefinition:
    """ツール定義"""
    name: str
    func: Callable
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    modes: list[str] = field(default_factory=lambda: ["chat", "coding"])


class ToolsRegistry:
    """ツールの登録・取得・実行を管理"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict[str, Any] | None = None,
        modes: list[str] | None = None,
    ) -> None:
        """ツールを登録"""
        self._tools[name] = ToolDefinition(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {},
            modes=modes or ["chat", "coding"],
        )
        logger.info("Registered tool: %s", name)

    def has(self, name: str) -> bool:
        """ツールが登録済みかどうか"""
        return name in self._tools

    def get(self, name: str) -> ToolDefinition | None:
        """ツール定義を取得"""
        return self._tools.get(name)

    def get_descriptions_text(self, mode: str | None = None) -> str:
        """ツール説明をテキスト形式で返す（プロンプト注入用）

        各ツールについて、シグネチャ行に続けて parameter ごとに required/optional
        と description を 1 行ずつ展開する。LLM が optional 引数を省略する判断や、
        ISO 日付などの例示を読めるようにするため。
        """
        lines = []
        for tool in self._tools.values():
            if mode and mode not in tool.modes:
                continue
            required = self._required_param_names(tool)
            sig_parts = []
            for k, v in tool.parameters.items():
                ptype = v.get("type", "any")
                marker = "" if k in required else "?"
                sig_parts.append(f"{k}{marker}: {ptype}")
            sig_text = ", ".join(sig_parts)
            lines.append(f"- {tool.name}({sig_text}): {tool.description}")
            for k, v in tool.parameters.items():
                desc = v.get("description", "")
                if not desc:
                    continue
                req_marker = "required" if k in required else "optional"
                lines.append(f"    - {k} ({req_marker}): {desc}")
        return "\n".join(lines)

    @staticmethod
    def _required_param_names(tool: ToolDefinition) -> set[str]:
        """ツール関数シグネチャから default なしパラメータ名を抽出"""
        try:
            sig = inspect.signature(tool.func)
        except (ValueError, TypeError):
            return set()
        required: set[str] = set()
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.default is inspect.Parameter.empty:
                required.add(name)
        return required

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """ツールを実行

        引数不足時は TypeError を発生させず、エラー文字列を返す。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        # 必須引数のバリデーション（全ツール共通）
        validation_error = self._validate_args(tool, kwargs)
        if validation_error:
            logger.warning("Tool arg validation failed: %s - %s", name, validation_error)
            return f"Error: {validation_error}"

        logger.info("Executing tool: %s(%s)", name, kwargs)

        if inspect.iscoroutinefunction(tool.func):
            return await tool.func(**kwargs)
        # 同期関数はスレッドプールで実行し、イベントループをブロックしない
        return await asyncio.to_thread(tool.func, **kwargs)

    @staticmethod
    def _validate_args(tool: ToolDefinition, kwargs: dict[str, Any]) -> str | None:
        """関数シグネチャに基づいて必須引数を検証する

        Returns:
            エラーメッセージ。問題なければ None。
        """
        try:
            sig = inspect.signature(tool.func)
        except (ValueError, TypeError):
            return None  # シグネチャ取得不可ならスキップ

        missing: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            # デフォルト値なし = 必須引数
            if param.default is inspect.Parameter.empty:
                if param_name not in kwargs or kwargs[param_name] is None:
                    missing.append(param_name)

        if not missing:
            return None

        # LLM が引数を修正できるようにシグネチャ情報を含める
        expected_parts: list[str] = []
        for p_name, p in sig.parameters.items():
            if p_name in ("self", "cls"):
                continue
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            ann = (
                p.annotation.__name__
                if p.annotation is not inspect.Parameter.empty
                and hasattr(p.annotation, "__name__")
                else "any"
            )
            if p.default is inspect.Parameter.empty:
                expected_parts.append(f"{p_name}: {ann}")
            else:
                expected_parts.append(f"{p_name}: {ann} = {p.default!r}")

        return (
            f"Missing required argument(s): {', '.join(missing)}. "
            f"Expected: {tool.name}({', '.join(expected_parts)})"
        )

    @property
    def count(self) -> int:
        return len(self._tools)
