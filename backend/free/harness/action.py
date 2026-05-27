"""

Harness レイヤーで生成・実行される行為の型安全な表現。

LLM 出力 (:func:`backend.free.harness.parser.parse_actions`) からは ``kind``
タグ付き JSON を経由してこのデータクラスへ復元される。実 executor は
``isinstance`` ベースのディスパッチで処理する想定。

設計方針:

- ``frozen=True`` で不変オブジェクト
- ``slots=True`` でメモリ効率
- ``kind`` を ``Literal`` でタグ付けし、``Action`` ユニオンで型を絞り込む
- ``Any`` 禁止 (.claude/rules/backend.md)
- 後方互換は不要 (CLAUDE.md "後方互換は不要")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ActionKind = Literal["edit_file", "run_command", "search", "noop"]


@dataclass(frozen=True, slots=True)
class EditFileAction:
    """ファイル編集 action。

    ``old`` を ``new`` で置換する単純な置換編集を表現する。新規ファイル作成は
    ``old=""`` (空文字列) を許容することで表現する想定。
    """

    path: str
    old: str
    new: str
    kind: Literal["edit_file"] = "edit_file"


@dataclass(frozen=True, slots=True)
class RunCommandAction:
    """シェルコマンド実行 action。

    ``command`` は引数列。シェル文字列としての展開を避けるため list[str] に固定。
    """

    command: tuple[str, ...]
    cwd: str | None = None
    kind: Literal["run_command"] = "run_command"


@dataclass(frozen=True, slots=True)
class SearchAction:
    """RAG / grep など検索 action。"""

    query: str
    kind: Literal["search"] = "search"


@dataclass(frozen=True, slots=True)
class NoopAction:
    """何もしない action。LLM が「タスク完了」「情報不足」などを表明した
    ケースの placeholder として用いる。"""

    reason: str = ""
    kind: Literal["noop"] = "noop"


Action = EditFileAction | RunCommandAction | SearchAction | NoopAction


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Action 実行結果。Harness.observe() が受け取る。"""

    action: Action
    success: bool
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    diff_summary: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# シリアライズ / デシリアライズ
# ──────────────────────────────────────────────────────────────────────────


def action_to_dict(action: Action) -> dict[str, object]:
    """Action を JSON-serializable な dict に変換する。

    parser から復元可能な形式を返す。``kind`` は必ず含む。
    """
    match action:
        case EditFileAction():
            return {
                "kind": "edit_file",
                "path": action.path,
                "old": action.old,
                "new": action.new,
            }
        case RunCommandAction():
            return {
                "kind": "run_command",
                "command": list(action.command),
                "cwd": action.cwd,
            }
        case SearchAction():
            return {"kind": "search", "query": action.query}
        case NoopAction():
            return {"kind": "noop", "reason": action.reason}


def action_from_dict(data: dict[str, object]) -> Action:
    """``kind`` タグ付き dict から ``Action`` を復元する。

    Raises:
        ValueError: 必須フィールド欠落 / 未知 kind / 型不整合
    """
    kind = data.get("kind")
    match kind:
        case "edit_file":
            return EditFileAction(
                path=_require_str(data, "path"),
                old=_require_str(data, "old"),
                new=_require_str(data, "new"),
            )
        case "run_command":
            command = data.get("command")
            if not isinstance(command, list) or not all(
                isinstance(c, str) for c in command
            ):
                raise ValueError("run_command.command must be list[str]")
            cwd_raw = data.get("cwd")
            if cwd_raw is not None and not isinstance(cwd_raw, str):
                raise ValueError("run_command.cwd must be str or null")
            return RunCommandAction(command=tuple(command), cwd=cwd_raw)
        case "search":
            return SearchAction(query=_require_str(data, "query"))
        case "noop":
            reason_raw = data.get("reason", "")
            if not isinstance(reason_raw, str):
                raise ValueError("noop.reason must be str")
            return NoopAction(reason=reason_raw)
        case _:
            raise ValueError(f"unknown action kind: {kind!r}")


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing or non-str field: {key}")
    return value
