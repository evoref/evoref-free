"""

``Harness`` が LLM 出力から抽出した ``Action`` 列を実際にローカル環境で
実行する層。sandbox 制限 (書込先 allowlist / コマンド allowlist) と
成果物メタデータの収集を担う。

設計方針:

- **書込先制限**: ``edit_file`` Action の ``path`` は ``Path.resolve()`` 後に
  ``allowed_write_roots`` のいずれかの配下である必要がある。パストラバーサル
  (``..``) や絶対パスによる sandbox 脱出は拒否。
- **コマンド allowlist**: ``run_command`` の引数列先頭 (``command[0]``) が
  ``allowed_commands`` に含まれる必要がある。``["*"]`` のみを含む場合は
  全許可 (テスト / 開発用; 本番運用では明示推奨)。``shell=False`` 固定。
- **artifact メタデータ**: ``edit_file`` が実行された場合、変更後のファイル
  内容から ``diff_sha1`` (SHA1 先頭 12 文字) / ``lines_added`` / ``lines_removed``
  を算出し ``ActionResult.metadata`` に格納する。``ExecutionOutcome.artifacts``
  への集計は RalphExecutor 側で行う。
- **純粋関数優先**: sandbox 判定 (``_resolve_under_roots`` / ``_is_command_allowed``)
  は純粋関数として分離。I/O はメソッド内にまとめる。
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from backend.free.harness.action import (
    Action,
    ActionResult,
    EditFileAction,
    NoopAction,
    RunCommandAction,
    SearchAction,
)
from backend.free.loop.executor import ArtifactEntry
from backend.log_config import get_logger

logger = get_logger("loop.action_runner")


#: ``diff_sha1`` に保持する文字数 (SHA1 hexdigest の先頭)
DIFF_SHA1_LEN = 12

#: ``output`` / ``error`` に保持する末尾バイト数
OUTPUT_TAIL_BYTES = 4000

#: ``["*"]`` が allowed_commands に含まれる場合は全許可
WILDCARD = "*"


SubprocessRunner = Callable[..., subprocess.CompletedProcess]
"""``subprocess.run`` 互換 callable (テスト差し替え用)"""


class ActionRunnerError(RuntimeError):
    """ActionRunner の構築・実行で発生したエラー"""


@dataclass
class ActionRunnerConfig:
    """ActionRunner の sandbox 設定 (config.yaml から構築)"""

    allowed_write_roots: tuple[Path, ...]
    allowed_commands: tuple[str, ...]
    command_timeout_sec: float = 120.0


@dataclass
class ActionRunner:
    """``Action`` を逐次実行して ``ActionResult`` を返すランナー。

    同一インスタンスを複数 task で再利用可能 (状態を持たない)。sandbox 設定は
    コンストラクタで固定する。``runner`` を差し替えるとサブプロセス呼び出しを
    mock できる (テスト用)。
    """

    config: ActionRunnerConfig
    repo_root: Path
    runner: SubprocessRunner = field(default=subprocess.run)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        resolved: list[Path] = []
        for raw in self.config.allowed_write_roots:
            p = Path(raw)
            if not p.is_absolute():
                p = self.repo_root / p
            resolved.append(p.resolve())
        self._resolved_roots: tuple[Path, ...] = tuple(resolved)
        self._allowed_cmds: tuple[str, ...] = tuple(self.config.allowed_commands)

    # ── 公開 API ──────────────────────────────────────────────────────

    def run_one(self, action: Action) -> ActionResult:
        """1 件の Action を実行する。例外は送出せず ``success=False`` に畳む。"""
        t0 = time.perf_counter()
        try:
            match action:
                case EditFileAction():
                    return self._run_edit_file(action, t0)
                case RunCommandAction():
                    return self._run_command(action, t0)
                case SearchAction():
                    return self._run_search(action, t0)
                case NoopAction():
                    return self._run_noop(action, t0)
        except Exception as exc:
            logger.warning(
                "ActionRunner: unexpected error running %s: %s",
                action.kind, exc,
            )
            return ActionResult(
                action=action,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def collect_artifacts(
        self, results: Iterable[ActionResult],
    ) -> list[ArtifactEntry]:
        """``ActionResult`` 列から ``ArtifactEntry`` を集計する。

        ``edit_file`` が success した ``ActionResult`` のみを対象とする。
        """
        out: list[ArtifactEntry] = []
        for r in results:
            if not r.success:
                continue
            if not isinstance(r.action, EditFileAction):
                continue
            meta = r.metadata or {}
            sha = str(meta.get("diff_sha1") or "")
            if not sha:
                continue
            try:
                added = int(meta.get("lines_added") or 0)
                removed = int(meta.get("lines_removed") or 0)
            except (TypeError, ValueError):
                added = removed = 0
            out.append(
                ArtifactEntry(
                    path=r.action.path,
                    diff_sha1=sha,
                    lines_added=added,
                    lines_removed=removed,
                    action_kind="edit_file",
                ),
            )
        return out

    # ── edit_file ─────────────────────────────────────────────────────

    def _run_edit_file(
        self, action: EditFileAction, t0: float,
    ) -> ActionResult:
        target = self._resolve_write_path(action.path)
        existed = target.exists()
        old_bytes = target.read_bytes() if existed else b""
        old_text = _safe_decode(old_bytes)

        new_text = self._apply_replace(old_text, action.old, action.new)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text, encoding="utf-8", newline="\n")

        diff_sha1 = hashlib.sha1(
            (old_text + "\0" + new_text).encode("utf-8", errors="replace"),
        ).hexdigest()[:DIFF_SHA1_LEN]
        lines_added, lines_removed = _count_line_diff(old_text, new_text)
        diff_summary = f"+{lines_added}/-{lines_removed} {action.path}"
        metadata: dict[str, str] = {
            "executor": "action_runner",
            "diff_sha1": diff_sha1,
            "lines_added": str(lines_added),
            "lines_removed": str(lines_removed),
            "existed_before": "true" if existed else "false",
        }
        logger.info(
            "ActionRunner.edit_file: path=%s added=%d removed=%d sha1=%s",
            target, lines_added, lines_removed, diff_sha1,
        )
        return ActionResult(
            action=action,
            success=True,
            output=f"wrote {target} ({lines_added}+/{lines_removed}-)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            diff_summary=diff_summary,
            metadata=metadata,
        )

    def _apply_replace(self, old_text: str, old: str, new: str) -> str:
        if old == "":
            # 新規ファイル作成 or 末尾への追記 (既存内容を保持したまま new を上書き)
            return new
        if old not in old_text:
            raise ActionRunnerError(
                f"edit_file: `old` not found in target file "
                f"(len(old)={len(old)}, len(target)={len(old_text)})",
            )
        return old_text.replace(old, new, 1)

    def _resolve_write_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise ActionRunnerError("edit_file: path must be non-empty")
        p = Path(raw_path)
        if not p.is_absolute():
            p = self.repo_root / p
        resolved = p.resolve() if p.exists() else _lexical_resolve(p)
        if not _is_under_any(resolved, self._resolved_roots):
            raise ActionRunnerError(
                f"edit_file: path outside sandbox: {resolved} "
                f"(allowed: {[str(r) for r in self._resolved_roots]})",
            )
        return resolved

    # ── run_command ───────────────────────────────────────────────────

    def _run_command(
        self, action: RunCommandAction, t0: float,
    ) -> ActionResult:
        if not action.command:
            raise ActionRunnerError("run_command: command must be non-empty")
        head = action.command[0]
        if not _is_command_allowed(head, self._allowed_cmds):
            raise ActionRunnerError(
                f"run_command: {head!r} not in allowed_commands "
                f"({list(self._allowed_cmds)})",
            )
        cwd: Path
        if action.cwd:
            p = Path(action.cwd)
            if not p.is_absolute():
                p = self.repo_root / p
            cwd_resolved = p.resolve() if p.exists() else _lexical_resolve(p)
            if not _is_under_any(cwd_resolved, self._resolved_roots + (self.repo_root,)):
                raise ActionRunnerError(
                    f"run_command: cwd outside sandbox: {cwd_resolved}",
                )
            cwd = cwd_resolved
        else:
            cwd = self.repo_root
        logger.info(
            "ActionRunner.run_command: cmd=%s cwd=%s timeout=%.1fs",
            list(action.command), cwd, self.config.command_timeout_sec,
        )
        try:
            cp = self.runner(
                list(action.command),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_sec,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ActionResult(
                action=action,
                success=False,
                error=f"timeout after {self.config.command_timeout_sec:.0f}s",
                output=_tail(getattr(exc, "stdout", None)),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"executor": "action_runner", "timeout": "true"},
            )
        except (FileNotFoundError, OSError) as exc:
            return ActionResult(
                action=action,
                success=False,
                error=f"exec error: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"executor": "action_runner", "exec_error": "true"},
            )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        ok = cp.returncode == 0
        return ActionResult(
            action=action,
            success=ok,
            output=_tail(cp.stdout),
            error=(None if ok else _tail(cp.stderr) or f"rc={cp.returncode}"),
            duration_ms=duration_ms,
            metadata={
                "executor": "action_runner",
                "returncode": str(cp.returncode if cp.returncode is not None else ""),
            },
        )

    # ── search / noop ─────────────────────────────────────────────────

    def _run_search(
        self, action: SearchAction, t0: float,
    ) -> ActionResult:
        # RAG 検索は LoopDriver / RalphExecutor 側で別途注入する想定。
        # ActionRunner 単体では「未サポート」を示して呼び出し側に委譲する。
        return ActionResult(
            action=action,
            success=True,
            output=f"search query={action.query!r} (deferred)",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={"executor": "action_runner", "deferred": "true"},
        )

    def _run_noop(
        self, action: NoopAction, t0: float,
    ) -> ActionResult:
        return ActionResult(
            action=action,
            success=True,
            output=f"noop reason={action.reason!r}",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            metadata={"executor": "action_runner"},
        )


# ──────────────────────────────────────────────────────────────────────────
# 純粋ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def _lexical_resolve(p: Path) -> Path:
    """存在しないパスも含めて ``..`` を解決する。"""
    parts: list[str] = []
    for part in p.parts:
        if part == "..":
            if parts and parts[-1] not in ("", "/"):
                parts.pop()
        elif part == ".":
            continue
        else:
            parts.append(part)
    if not parts:
        return Path(p.anchor) if p.anchor else Path(".")
    return Path(*parts)


def _is_under_any(path: Path, roots: Iterable[Path]) -> bool:
    try:
        path_resolved = path.resolve() if path.exists() else _lexical_resolve(path)
    except OSError:
        path_resolved = path
    for root in roots:
        try:
            path_resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_command_allowed(head: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return False
    if WILDCARD in allowed:
        return True
    basename = Path(head).name.lower()
    # Windows の ``python.exe`` / ``npm.cmd`` 等は拡張子を剥がして比較
    if "." in basename:
        stem = basename.rsplit(".", 1)[0]
    else:
        stem = basename
    lowered = {a.lower() for a in allowed}
    return stem in lowered or basename in lowered


def _safe_decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _count_line_diff(old_text: str, new_text: str) -> tuple[int, int]:
    """素朴な行数差分を返す (``difflib`` で unified diff を数えるより軽量)。

    ``lines_added`` = new で増えた行数, ``lines_removed`` = old から消えた行数
    (厳密な LCS ではなく、共通ハッシュセットの相補で計算)。
    """
    import difflib

    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = 0
    removed = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            added += j2 - j1
        elif op == "delete":
            removed += i2 - i1
        elif op == "replace":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def _tail(text: str | None) -> str:
    if text is None:
        return ""
    if len(text) <= OUTPUT_TAIL_BYTES:
        return text
    return "...<truncated>...\n" + text[-OUTPUT_TAIL_BYTES:]


# ──────────────────────────────────────────────────────────────────────────
# 設定ビルダ
# ──────────────────────────────────────────────────────────────────────────


def build_action_runner_config(
    raw_cfg: dict[str, object] | None,
) -> ActionRunnerConfig:
    """``config.yaml`` の ``loop.sandbox`` dict から ``ActionRunnerConfig`` を作る。

    ``LoopSandboxConfig`` から生成される dict を想定。"""
    cfg = raw_cfg or {}
    roots = cfg.get("allowed_write_roots") or ["local/loop_sandbox"]
    cmds = cfg.get("allowed_commands") or []
    timeout = cfg.get("command_timeout_sec") or 120.0
    if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
        raise ActionRunnerError("loop.sandbox.allowed_write_roots must be list[str]")
    if not isinstance(cmds, list) or not all(isinstance(c, str) for c in cmds):
        raise ActionRunnerError("loop.sandbox.allowed_commands must be list[str]")
    return ActionRunnerConfig(
        allowed_write_roots=tuple(Path(r) for r in roots),
        allowed_commands=tuple(cmds),
        command_timeout_sec=float(timeout),
    )


__all__ = [
    "ActionRunner",
    "ActionRunnerConfig",
    "ActionRunnerError",
    "build_action_runner_config",
]

# mypy / type: ignore 要請用 (未使用 Literal を警告させないため)
_ = Literal
