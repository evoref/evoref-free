"""

``Harness`` はエージェントが自律実行する際の中核制御層を抽象化する。
``SemanticFactStore`` 直参照を廃止し、SemMem 読取は
:class:`~backend.free.memory.views.harness.HarnessFactView` (read-only) 経由
に統一した。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from backend.free.harness.action import Action, ActionResult
from backend.free.harness.parser import parse_actions
from backend.free.harness.prompt_builder import build_messages
from backend.free.memory.views.harness import HarnessFactView

if TYPE_CHECKING:
    from backend.free.memory.types import SemanticFact

# PolicyInterpreter.get_all 互換シグネチャ: ``mode`` を受けて policy dict を返す
PolicyProvider = Callable[[str], dict[str, object] | None]


class Harness(Protocol):
    """task → action 変換の中核制御層 Protocol。"""

    def prepare(self, task: "SemanticFact") -> list[dict[str, str]]:
        """task ファクトを LLM 入力 messages に展開する。"""
        ...

    def parse(self, response: str) -> list[Action]:
        """LLM 応答テキストを ``Action`` 列に変換する。"""
        ...

    def observe(self, action: Action, result: ActionResult) -> None:
        """Action 実行結果を次サイクルのコンテキストへ注入する。"""
        ...


class DefaultHarness:
    """``Harness`` Protocol の既定実装。

    - ``prepare`` は :func:`build_messages` に委譲
    - ``parse`` は :func:`parse_actions` に委譲
    - ``observe`` は直近 ``observation_window`` 件の結果を保持し、次回
      ``prepare`` 呼び出し時に ``extra_user_context`` として展開する
    """

    def __init__(
        self,
        *,
        harness_view: HarnessFactView,
        mode: str = "create",
        max_failures: int = 5,
        max_fewshot: int = 3,
        observation_window: int = 3,
        policy_provider: PolicyProvider | None = None,
    ) -> None:
        """Args:
            harness_view: SemMem (global + project スコープ) への read-only View。
                prompt_builder が failure_pattern / fewshot を引く際に使う。
            mode: ``"chat"`` / ``"create"`` 等。fewshot subject 前方一致に使用。
            max_failures: 注入する failure_pattern の最大件数。
            max_fewshot: 注入する fewshot の最大件数。
            observation_window: 直近観測結果のリングバッファサイズ。
            policy_provider: ``mode -> dict`` を返す callable (任意)。
        """
        self._harness_view = harness_view
        self._mode = mode
        self._max_failures = max_failures
        self._max_fewshot = max_fewshot
        self._observation_window = max(0, observation_window)
        self._observations: list[ActionResult] = []
        self._policy_provider = policy_provider

    def prepare(self, task: "SemanticFact") -> list[dict[str, str]]:
        policy = (
            self._policy_provider(self._mode)
            if self._policy_provider is not None
            else None
        )
        return build_messages(
            task,
            harness_view=self._harness_view,
            mode=self._mode,
            max_failures=self._max_failures,
            max_fewshot=self._max_fewshot,
            policy_overrides=policy,
            extra_user_context=self._render_observations(),
        )

    def parse(self, response: str) -> list[Action]:
        return parse_actions(response)

    def observe(self, action: Action, result: ActionResult) -> None:  # noqa: ARG002
        if self._observation_window == 0:
            return
        self._observations.append(result)
        if len(self._observations) > self._observation_window:
            self._observations = self._observations[-self._observation_window:]

    # ── 観測ヘルパ (テスト・統計用) ───────────────────────────────────

    @property
    def recent_observations(self) -> list[ActionResult]:
        """直近の観測結果スナップショット (read-only コピー)"""
        return list(self._observations)

    def _render_observations(self) -> str | None:
        if not self._observations:
            return None
        lines: list[str] = []
        for i, obs in enumerate(self._observations, 1):
            status = "OK" if obs.success else "FAIL"
            head = f"[{i}] {status} {obs.action.kind}"
            lines.append(head)
            if obs.diff_summary:
                lines.append(f"    diff: {obs.diff_summary}")
            if obs.error:
                lines.append(f"    error: {obs.error}")
            elif obs.output:
                snippet = obs.output[:200].replace("\n", " ")
                lines.append(f"    output: {snippet}")
        return "\n".join(lines)
