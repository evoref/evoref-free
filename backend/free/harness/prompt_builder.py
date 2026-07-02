"""

task ファクト + 関連 failure_pattern + 適用 policy + 関連 fewshot を SemMem
から取得し、LLM 入力 messages を構築する。

``SemanticFactStore`` 直参照を廃止し、読取は
:class:`~backend.free.memory.views.harness.HarnessFactView` (read-only) 経由
に統一した。

設計方針:

- llama-server の /v1/chat/completions に渡す前提のため、テンプレート適用は
  サーバー側に委譲する (CLAUDE.md 設計原則 4)
- failure_pattern / fewshot は HarnessFactView から最大件数つきで取得
- policy は呼び出し側が ``policy_overrides`` を dict で渡す
- LLM 出力フォーマット (``<actions>...</actions>``) はシステムメッセージ末尾に
  常に明示する
"""

from __future__ import annotations

import json

from backend.free.memory.types import SemanticFact
from backend.free.memory.views.harness import HarnessFactView


SYSTEM_PROMPT_HEADER = (
    "You are evoref's autonomous harness. Convert the given task into a "
    "minimal, ordered sequence of concrete actions.\n\n"
    "Respond with ONLY a JSON array wrapped in <actions>...</actions>. "
    "Do not include any other text outside the tags.\n\n"
    "Each array element must be one of:\n"
    "- {\"kind\": \"edit_file\", \"path\": str, \"old\": str, \"new\": str}\n"
    "- {\"kind\": \"run_command\", \"command\": [str, ...], \"cwd\": str | null}\n"
    "- {\"kind\": \"search\", \"query\": str}\n"
    "- {\"kind\": \"noop\", \"reason\": str}\n\n"
    "STRICT: \"path\" is a string (NOT an array). \"old\" and \"new\" are "
    "top-level keys of the edit_file object (NOT nested inside \"path\"). "
    "Only \"command\" is an array.\n\n"
    "Correct example (a single edit_file action):\n"
    "<actions>[{\"kind\": \"edit_file\", \"path\": \"backend/foo.py\", "
    "\"old\": \"return 1\", \"new\": \"return 2\"}]</actions>\n\n"
    "Use \"noop\" when the task is already satisfied or you need more "
    "information."
)


def build_messages(
    task: SemanticFact,
    *,
    harness_view: HarnessFactView,
    mode: str = "coding",
    max_failures: int = 5,
    max_fewshot: int = 3,
    policy_overrides: dict[str, object] | None = None,
    extra_user_context: str | None = None,
) -> list[dict[str, str]]:
    """task ファクト + 関連ファクトから LLM messages を構築する。

    Args:
        task: 対象タスクファクト (``type=="task"``)。``content`` (= ``object``)
            を user メッセージ本体として埋め込む。
        harness_view: SemMem への read-only View。failure_pattern / fewshot を
            引くために使う。
        mode: ``"chat"`` / ``"coding"`` 等。fewshot subject 前方一致に使用。
        max_failures: 注入する failure_pattern の最大件数。
        max_fewshot: 注入する fewshot の最大件数。
        policy_overrides: policy パラメータの dict。``None`` または空 dict
            なら policy セクションは生成しない。
        extra_user_context: user メッセージの冒頭に挟みたい追加文脈。

    Returns:
        ``[{"role": ..., "content": ...}, ...]`` 形式の messages 配列。

    Raises:
        ValueError: ``task.type`` が ``"task"`` でない場合。
    """
    if task.type != "task":
        raise ValueError(
            f"build_messages expects task fact, got type={task.type!r}",
        )

    failures = _collect_failures(harness_view, mode, max_failures)
    fewshots = _collect_fewshots(harness_view, mode, max_fewshot)

    system_content = _render_system(
        policy_overrides=policy_overrides or {},
        failures=failures,
        fewshots=fewshots,
    )
    user_content = _render_user(task, extra_user_context)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ──────────────────────────────────────────────────────────────────────────
# 収集 (HarnessFactView 委譲)
# ──────────────────────────────────────────────────────────────────────────


def _collect_failures(
    harness_view: HarnessFactView, mode: str, limit: int,  # noqa: ARG001
) -> list[SemanticFact]:
    """HarnessFactView から最近の failure_pattern を recency 順に返す。"""
    if limit <= 0:
        return []
    return harness_view.get_recent_failures(mode=None, limit=limit)


def _collect_fewshots(
    harness_view: HarnessFactView, mode: str, limit: int,
) -> list[SemanticFact]:
    """HarnessFactView から ``mode`` の fewshot を recency 順に返す。"""
    if limit <= 0:
        return []
    facts = harness_view.get_active_fewshots(mode=mode)
    facts.sort(key=lambda f: f.created_at, reverse=True)
    return facts[:limit]


# ──────────────────────────────────────────────────────────────────────────
# レンダリング
# ──────────────────────────────────────────────────────────────────────────


def _render_system(
    *,
    policy_overrides: dict[str, object],
    failures: list[SemanticFact],
    fewshots: list[SemanticFact],
) -> str:
    parts: list[str] = [SYSTEM_PROMPT_HEADER]

    if policy_overrides:
        parts.append("\n## Active Policy")
        parts.append(
            "```json\n"
            + json.dumps(policy_overrides, ensure_ascii=False, indent=2)
            + "\n```",
        )

    if failures:
        parts.append("\n## Recent Failure Patterns")
        for f in failures:
            sig = f.failure_signature or "?"
            parts.append(f"- [{sig}] {f.predicate}: {f.object}")

    if fewshots:
        parts.append("\n## Few-shot Examples")
        for i, f in enumerate(fewshots, 1):
            parts.append(f"### Example {i}")
            parts.append(f.object)

    return "\n".join(parts)


def _render_user(task: SemanticFact, extra_context: str | None) -> str:
    lines: list[str] = []
    if extra_context:
        lines.append("## Observation Context")
        lines.append(extra_context.strip())
        lines.append("")
    lines.append("## Task")
    lines.append(f"subject: {task.subject}")
    lines.append(f"predicate: {task.predicate}")
    lines.append(f"goal: {task.object}")
    return "\n".join(lines)
