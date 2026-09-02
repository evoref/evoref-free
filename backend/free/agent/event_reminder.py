"""イベント駆動リマインダーシステム（Instruction Fade-Out 対策）"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from backend.free.agent.safety_patterns import DANGEROUS_PATTERNS
from backend.free.agent.agent_state import AgentState
from backend.free.core.turn_text import append_to_last_user
from backend.i18n_helper import msg_for_locale, prompt_locale
from backend.log_config import get_logger

logger = get_logger("agent.event_reminder")


@dataclass
class ReminderEvent:
    """検知されたイベント"""

    event_type: str
    severity: str  # 'info' | 'warning' | 'critical'
    context: dict


class EventDetector:
    """ルールベースのイベント検知。LLM 呼び出しゼロ。"""

    def detect(self, state: AgentState) -> list[ReminderEvent]:
        """AgentState からイベントを検知"""
        events: list[ReminderEvent] = []

        # ツール連続失敗
        if state.consecutive_tool_failures >= 2:
            events.append(
                ReminderEvent(
                    "tool_repeated_failure",
                    "warning",
                    {
                        "fail_count": state.consecutive_tool_failures,
                        "last_tool": state.last_tool_name,
                        "last_error": state.last_error[:80],
                    },
                )
            )

        # 最終反復（Meta-Cognitive 層のみ）
        if (
            state.current_iteration == state.max_iterations - 1
            and state.agent_layer == "meta_cognitive"
        ):
            events.append(
                ReminderEvent(
                    "last_iteration",
                    "warning",
                    {"iteration": state.current_iteration + 1},
                )
            )

        # 危険コマンド検知
        if state.pending_tool == "run_command":
            cmd = state.pending_args.get("command", "")
            if any(re.search(p, cmd) for p in DANGEROUS_PATTERNS):
                events.append(
                    ReminderEvent(
                        "dangerous_command",
                        "critical",
                        {"command": cmd[:100]},
                    )
                )

        # フォーマット逸脱
        if state.expected_format and state.last_output:
            if not self._matches_format(
                state.last_output, state.expected_format
            ):
                preview = state.last_output[:50].replace("\n", " ")
                logger.debug(
                    "EventDetector: format deviation detected, "
                    "expected=%s, got=%s",
                    state.expected_format,
                    preview,
                )
                events.append(
                    ReminderEvent(
                        "format_deviation",
                        "warning",
                        {"expected": state.expected_format},
                    )
                )

        # コンテキスト圧迫
        if state.context_usage_pct > 85:
            events.append(
                ReminderEvent(
                    "context_pressure",
                    "info",
                    {"usage_pct": state.context_usage_pct},
                )
            )

        return events

    def _matches_format(self, output: str, expected: str) -> bool:
        """出力が期待フォーマットに合致するか判定"""
        if expected == "json":
            try:
                json.loads(output)
                return True
            except json.JSONDecodeError:
                return False
        if expected == "unified_diff":
            return output.strip().startswith(("---", "@@", "diff"))
        return True


class EventReminderSystem:
    """イベント検知 -> リマインダー注入の統合システム。"""

    def __init__(self, config: dict):
        self.detector = EventDetector()
        ac = config.get("agent", {})
        self.enabled: bool = ac.get("reminders_enabled", True)
        self.max_reminders: int = ac.get("max_reminders_per_turn", 2)
        self.block_dangerous: bool = ac.get(
            "dangerous_command_block", True
        )

    def inject(
        self, messages: list[dict], state: AgentState
    ) -> list[dict]:
        """messages にリマインダーを注入して返す。

        critical は必ず注入、warning/info は上限まで。
        注入位置: **最後の user メッセージの末尾** (``turn_text`` の付与順序の
        契約に従う注記の 1 つ)。以前は独立した user メッセージを直前へ挿入して
        いたが、user ロールが 2 連続になり gemma 系テンプレートが 400 を返す
        (2026-09-02 監査 P-A8)。user メッセージが無い場合だけ末尾へ user として
        足す (情報を落とさないための縮退)。入力の list と要素は変更しない。
        """
        if not self.enabled:
            return messages

        events = self.detector.detect(state)
        if not events:
            return messages

        # 優先度順にソート
        priority = {"critical": 0, "warning": 1, "info": 2}
        events.sort(key=lambda e: priority.get(e.severity, 9))

        # critical は常に注入、non-critical は上限あり
        selected: list[ReminderEvent] = []
        non_critical = 0
        for event in events:
            if event.severity == "critical":
                selected.append(event)
            elif non_critical < self.max_reminders:
                selected.append(event)
                non_critical += 1

        # リマインダーテキスト生成
        reminder_text = "\n".join(
            self._render(e) for e in selected
        )

        # ログ出力
        event_types = [e.event_type for e in selected]
        tokens = max(1, len(reminder_text) // 2)
        logger.debug(
            "EventDetector: detected %d events: [%s]",
            len(events),
            ", ".join(e.event_type for e in events),
        )
        logger.debug(
            "EventReminderSystem: injected %d reminders (%d tokens), "
            "events=[%s]",
            len(selected),
            tokens,
            ", ".join(event_types),
        )

        # 危険コマンドブロックのログ
        for event in selected:
            if event.event_type == "dangerous_command":
                logger.warning(
                    "EventReminderSystem: BLOCKED dangerous command: %s",
                    event.context.get("command", ""),
                )

        result = list(messages)
        if not append_to_last_user(result, reminder_text):
            result.append({"role": "user", "content": reminder_text})
        return result

    def _render(self, event: ReminderEvent) -> str:
        """i18n テンプレートを解決する (言語は UI ではなく ``prompt_locale``)。

        リマインダーは LLM へ渡すプロンプト文なので、他の注記と同じく
        ``i18n.prompt_locale`` に従う。キーは i18n に置いたまま locale だけ
        差し替える (``msg`` は UI の locale 固定)。
        """
        return msg_for_locale(
            prompt_locale(), f"agent.reminder.{event.event_type}", **event.context,
        )
