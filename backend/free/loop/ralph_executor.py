"""

Geoffrey Huntley "ralph wiggum as a software engineer" 由来の **反復駆動
executor**。単純なサイクルを延々と回し、失敗を ``failure_pattern`` として
蓄積することで、複雑な計画よりも反復回数で品質を稼ぐ設計。

サイクル (1 タスクあたり):

    messages = harness.prepare(task_fact)          # failure/policy/fewshot 注入
    response = assist_model.generate(messages)
    actions  = harness.parse(response)
    for action in actions[:max_actions_per_task]:
        result = action_runner.run_one(action)
        harness.observe(action, result)
    gate_outcome = run_quality_gates(gates)
    return ExecutionOutcome(...)

SemMem への ``failure_pattern`` / ``progress_marker`` / ``task.status`` 書き
込みは本層では行わず、``LoopDriver`` が ``ExecutionOutcome`` を見て一元
管理する (副作用の所在を単純化)。

- 手法の源流は上記 Huntley のブログ。コードそのものは取り込まず手法名のみ
  参照している
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.free.harness.action import Action, ActionResult, NoopAction
from backend.free.harness.base import Harness
from backend.free.harness.parser import ParseError
from backend.free.loop.action_runner import ActionRunner
from backend.free.loop.executor import (
    ExecutionOutcome,
    ExecutionStatus,
)
from backend.free.loop.quality_gate import (
    QualityGate,
    QualityGateOutcome,
    run_quality_gates,
)
from backend.free.memory.types import SemanticFact, make_fact
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.loop.driver import TaskFactView

logger = get_logger("loop.ralph_executor")


POLICY_TEMPERATURE_KEYS = ("temperature", "assist_temperature")
POLICY_MAX_TOKENS_KEYS = ("max_tokens", "assist_max_tokens")
POLICY_TOP_K_KEYS = ("top_k", "assist_top_k")

DEFAULT_TEMPERATURE = 0.4
# 1024 tokens (~4000 chars) では actions JSON が途中切断され parse 失敗する
# 事象が発生したため 2048 に引き上げる。assist_client の ralph_loop timeout
# (120s) に対し 2048 tokens 生成は実測 ~10s で十分収まる。
DEFAULT_MAX_TOKENS = 2048


@dataclass
class RalphExecutor:
    """LLM + Harness + ActionRunner + QualityGate を結線した TaskExecutor。

    Args:
        harness: `DefaultHarness` またはテスト用スタブ (`Harness` Protocol)
        action_runner: 書込先制限付き ActionRunner
        assist_client: llama-server (assist) クライアント。None の場合は
            LLM 呼び出しをスキップし、Action 0 件 (skipped) を返す
            (デグラデーションモード)。
        quality_gates: 各サイクル後に走らせる ``QualityGate`` の列 (空可)
        max_actions_per_task: 1 サイクルあたり実行する Action 最大数
        policy_provider: ``mode -> dict`` を返す callable。温度 / top_k /
            max_tokens を LLM 呼び出しに反映するために参照する。None 可。
        mode: 通常 ``"coding"``。policy lookup の key。
        purpose: ``AssistModelClient.generate(purpose=...)`` に渡すラベル。
    """

    harness: Harness
    action_runner: ActionRunner
    assist_client: "AssistModelClient | None"
    quality_gates: Sequence[QualityGate] = ()
    max_actions_per_task: int = 10
    policy_provider: object | None = None  # Callable[[str], dict | None]
    mode: str = "coding"
    purpose: str = "ralph_loop"

    name: str = "ralph"

    async def execute(self, task: "TaskFactView") -> ExecutionOutcome:
        """1 タスクを LLM 駆動で消化する。"""
        task_fact = _view_to_fact(task)
        messages = self.harness.prepare(task_fact)
        if self.assist_client is None:
            logger.warning(
                "RalphExecutor: assist_client is None — skipping task=%s "
                "(degradation mode)", task.task_id,
            )
            return ExecutionOutcome(
                status="skipped",
                error="assist_client unavailable",
                notes={"executor": self.name, "reason": "no_assist_client"},
            )

        policy = self._lookup_policy()
        temperature = _policy_float(
            policy, POLICY_TEMPERATURE_KEYS, DEFAULT_TEMPERATURE,
        )
        max_tokens = _policy_int(
            policy, POLICY_MAX_TOKENS_KEYS, DEFAULT_MAX_TOKENS,
        )

        t0 = time.perf_counter()
        try:
            resp = await self.assist_client.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=self.purpose,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RalphExecutor: assist generate failed for task=%s: %s",
                task.task_id, exc,
            )
            return ExecutionOutcome(
                status="failure",
                error=f"assist_generate_error: {exc}",
                notes={"executor": self.name, "stage": "generate"},
            )
        gen_ms = (time.perf_counter() - t0) * 1000.0
        response_text = _extract_response_text(resp)

        try:
            actions = self.harness.parse(response_text)
        except ParseError as exc:
            logger.warning(
                "RalphExecutor: parse failed for task=%s: %s",
                task.task_id, exc,
            )
            return ExecutionOutcome(
                status="failure",
                error=f"parse_error: {exc}",
                notes={
                    "executor": self.name,
                    "stage": "parse",
                    "response_head": response_text[:1000],
                    "response_length": str(len(response_text)),
                    "generate_ms": f"{gen_ms:.0f}",
                },
            )

        trimmed: list[Action] = list(actions)[: self.max_actions_per_task]
        if len(actions) > self.max_actions_per_task:
            logger.info(
                "RalphExecutor: truncated actions %d -> %d (task=%s)",
                len(actions), self.max_actions_per_task, task.task_id,
            )

        # 実行
        results: list[ActionResult] = []
        for action in trimmed:
            result = self.action_runner.run_one(action)
            self.harness.observe(action, result)
            results.append(result)
            if not result.success:
                # 早期失敗: 残りの Action はスキップ (暴走防止)
                logger.info(
                    "RalphExecutor: action failed, stopping early "
                    "(task=%s, kind=%s, err=%s)",
                    task.task_id, action.kind, result.error,
                )
                break

        artifacts = tuple(self.action_runner.collect_artifacts(results))
        action_failed = any(not r.success for r in results)

        # actions が noop のみ、または空 → skipped 扱い
        is_all_noop = bool(trimmed) and all(
            isinstance(a, NoopAction) for a in trimmed
        )
        if not trimmed or is_all_noop:
            reason = "no_actions" if not trimmed else "all_noop"
            return ExecutionOutcome(
                status="skipped",
                actions=tuple(trimmed),
                action_results=tuple(results),
                artifacts=artifacts,
                error=None,
                notes={
                    "executor": self.name,
                    "reason": reason,
                    "generate_ms": f"{gen_ms:.0f}",
                },
            )

        # 品質ゲート (Action 失敗時も走らせる: 部分変更の影響を観測するため)
        gate_outcome: QualityGateOutcome | None = None
        if self.quality_gates:
            try:
                gate_outcome = await asyncio.to_thread(
                    run_quality_gates, list(self.quality_gates),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RalphExecutor: quality_gates failed to run: %s", exc,
                )

        gate_failed = gate_outcome is not None and not gate_outcome.ok
        status: ExecutionStatus
        error: str | None = None
        if action_failed:
            status = "failure"
            failing = next((r for r in results if not r.success), None)
            error = (
                f"action_failed: {failing.error}" if failing and failing.error
                else "action_failed"
            )
        elif gate_failed:
            status = "failure"
            assert gate_outcome is not None  # for type-checker
            failed_names = ", ".join(gate_outcome.failed)
            error = f"quality_gate_failed: {failed_names}"
        else:
            status = "success"

        return ExecutionOutcome(
            status=status,
            actions=tuple(trimmed),
            action_results=tuple(results),
            gate_outcome=gate_outcome,
            artifacts=artifacts,
            error=error,
            notes={
                "executor": self.name,
                "generate_ms": f"{gen_ms:.0f}",
                "temperature": f"{temperature:.3f}",
                "max_tokens": str(max_tokens),
            },
        )

    def _lookup_policy(self) -> dict[str, object] | None:
        if self.policy_provider is None:
            return None
        try:
            result = self.policy_provider(self.mode)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            logger.debug("policy_provider lookup failed: %s", exc)
            return None
        if isinstance(result, dict):
            return result
        return None


# ──────────────────────────────────────────────────────────────────────────
# ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def _view_to_fact(view: "TaskFactView") -> SemanticFact:
    """``TaskFactView`` から ``prompt_builder`` 互換の ``SemanticFact`` を組み立てる。

    ``build_messages`` は ``SemanticFact`` を受けるため、view のフィールドを
    再エンコードして最小限のファクトを作る。元ストアから引き直さないのは、
    executor 呼び出し後に task.status が遷移しても view の snapshot を
    使うため。
    """
    from backend.free.loop.driver import (
        TASK_PREDICATE,
        TASK_SUBJECT_PREFIX,
        encode_task_object,
    )

    object_json = encode_task_object(
        task_id=view.task_id,
        title=view.title,
        description=view.description,
        depends_on=view.depends_on,
        salience=view.salience,
        status=view.status,
        source_path=view.source_path,
    )
    return make_fact(
        subject=f"{TASK_SUBJECT_PREFIX}{view.task_id}",
        predicate=TASK_PREDICATE,
        object_=object_json,
        type="task",
        scope=SemanticFact.make_project_scope(view.project_id),
        mode_origin="coding",
        confidence=1.0,
    )


def _extract_response_text(resp: dict | str) -> str:
    """llama-server /v1/chat/completions レスポンスから content を取り出す。"""
    if isinstance(resp, str):
        return resp
    try:
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def _policy_float(
    policy: dict[str, object] | None,
    keys: tuple[str, ...],
    default: float,
) -> float:
    if not policy:
        return default
    for key in keys:
        if key in policy:
            try:
                return float(policy[key])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return default


def _policy_int(
    policy: dict[str, object] | None,
    keys: tuple[str, ...],
    default: int,
) -> int:
    if not policy:
        return default
    for key in keys:
        if key in policy:
            try:
                return int(policy[key])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return default


__all__ = ["RalphExecutor"]
