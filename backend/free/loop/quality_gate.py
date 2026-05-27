"""

EvorefMem 統合仕様 における自律実行ループの **品質ゲート層** を実装する
ループ 1 イテレーションで生成された変更が「コミット可能か」を判定するための
アダプタを提供する。

スコープ:

1. ``QualityGate`` 抽象 + 具象アダプタ
   - ``PytestGate``     : ``python scripts/safe_pytest.py [scope...]`` を呼ぶ
   - ``TypecheckGate``  : ``frontend/`` で ``npm run check`` を呼ぶ
   - ``LintGate``       : ``frontend/`` で ``npm run lint`` を呼ぶ (デフォルト OFF)
2. ``run_quality_gates(gates)`` — 順次実行して ``QualityGateOutcome`` を返す
3. ``decide_gate_action(outcome, policy, retries_so_far, retry_limit)``
   — ``loop.on_gate_fail`` (``retry`` / ``skip`` / ``abort``) と ``retry_limit_per_task``
   から次アクションを決定する純粋関数
4. ``rollback_working_tree(repo_root)`` — 失敗時に作業ツリーを破棄する helper
   (``git reset --hard HEAD`` + ``git clean -fd``)。**コミットは行われない** ため
   未コミット変更だけが対象になる。
5. ``build_default_gates(config, repo_root, frontend_dir)``
   — ``LoopQualityGatesConfig`` から有効なゲートのリストを構築

設計原則 (CLAUDE.md / .claude/rules/backend.md):
- Python 3.12+ 型表現
- ``subprocess.run`` を ``runner`` で差し替え可能にしてユニットテストでの mock を許容
- I/O 副作用は ``run`` メソッド内に閉じ込める。判定ロジックは純粋関数
- 後方互換不要
- 失敗時はコミットしない (本モジュールはコミットを一切行わない)

LoopDriver への接続は executor を実装する際に行う
本フェーズでは品質ゲート層のみを単独の依存として提供する。
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from backend.schemas import LoopQualityGatesConfig
from backend.log_config import get_logger

logger = get_logger("loop.quality_gate")


# ──────────────────────────────────────────────────────────────────────────
# 型エイリアス
# ──────────────────────────────────────────────────────────────────────────

OnGateFailPolicy = Literal["retry", "skip", "abort"]
"""``loop.on_gate_fail`` の取り得る値 (LoopConfig と同期)"""

SubprocessRunner = Callable[..., subprocess.CompletedProcess]
"""``subprocess.run`` 互換 callable (テスト差し替え用)"""

STDOUT_TAIL_BYTES = 4000
"""``GateResult.stdout_tail`` / ``stderr_tail`` に保持する末尾バイト数"""


# ──────────────────────────────────────────────────────────────────────────
# 結果データ型
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateResult:
    """単一品質ゲートの実行結果。

    ``ok=True`` かつ ``skipped=False`` で「合格」。``skipped=True`` の場合は
    ゲート自体が無効化されているか、依存ツール (例: ``npm``) が見つからずに
    実行を回避したことを示し、合格にも失敗にもカウントしない。
    """

    name: str
    ok: bool
    skipped: bool
    returncode: int | None
    duration_ms: int
    stdout_tail: str
    stderr_tail: str
    error: str | None = None
    skip_reason: str | None = None

    def is_failure(self) -> bool:
        """ゲート実行が失敗とみなされる (= 後段のリトライ判定対象) か"""
        return not self.ok and not self.skipped


@dataclass(frozen=True)
class QualityGateOutcome:
    """品質ゲート群の集約結果"""

    ok: bool
    results: tuple[GateResult, ...]
    failed: tuple[str, ...]
    skipped: tuple[str, ...]


class GateAction(StrEnum):
    """``decide_gate_action`` が返す次アクション"""

    PROCEED = "proceed"
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"


# ──────────────────────────────────────────────────────────────────────────
# ゲート抽象
# ──────────────────────────────────────────────────────────────────────────


class QualityGate(Protocol):
    """品質ゲートインターフェース。

    ``name`` は ``GateResult.name`` / ``QualityGateOutcome.failed`` の識別子に使う。
    ``enabled=False`` のゲートは ``run_quality_gates`` で skip 扱いになる。
    """

    name: str
    enabled: bool

    def run(self) -> GateResult: ...


@dataclass
class _SubprocessGateBase:
    """``subprocess.run`` ベースのゲート共通実装。

    具象クラスは ``name`` / ``build_command`` / ``cwd`` / ``timeout_sec`` を提供する。
    """

    name: str
    enabled: bool = True
    cwd: Path | None = None
    timeout_sec: float = 600.0
    runner: SubprocessRunner = field(default=subprocess.run)
    env_extra: dict[str, str] | None = None

    # サブクラスでオーバーライド
    def build_command(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def precheck_skip_reason(self) -> str | None:
        """実行可能性チェック。スキップ理由 (str) を返すと skip 扱い"""
        return None

    def run(self) -> GateResult:
        if not self.enabled:
            return GateResult(
                name=self.name,
                ok=True,
                skipped=True,
                returncode=None,
                duration_ms=0,
                stdout_tail="",
                stderr_tail="",
                skip_reason="disabled",
            )
        skip = self.precheck_skip_reason()
        if skip is not None:
            logger.info("quality_gate %s skipped: %s", self.name, skip)
            return GateResult(
                name=self.name,
                ok=True,
                skipped=True,
                returncode=None,
                duration_ms=0,
                stdout_tail="",
                stderr_tail="",
                skip_reason=skip,
            )

        cmd = self.build_command()
        cwd = str(self.cwd) if self.cwd is not None else None
        logger.info(
            "quality_gate %s start: cmd=%s cwd=%s timeout=%.0fs",
            self.name, cmd, cwd, self.timeout_sec,
        )
        t0 = time.monotonic()
        try:
            cp = self.runner(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "quality_gate %s timeout after %dms", self.name, duration_ms,
            )
            return GateResult(
                name=self.name,
                ok=False,
                skipped=False,
                returncode=None,
                duration_ms=duration_ms,
                stdout_tail=_tail_text(getattr(exc, "stdout", None)),
                stderr_tail=_tail_text(getattr(exc, "stderr", None)),
                error=f"timeout after {self.timeout_sec:.0f}s",
            )
        except (FileNotFoundError, OSError) as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "quality_gate %s exec error: %s", self.name, exc,
            )
            return GateResult(
                name=self.name,
                ok=False,
                skipped=False,
                returncode=None,
                duration_ms=duration_ms,
                stdout_tail="",
                stderr_tail=str(exc),
                error=f"exec error: {exc}",
            )

        duration_ms = int((time.monotonic() - t0) * 1000)
        ok = cp.returncode == 0
        logger.info(
            "quality_gate %s done: returncode=%s ok=%s duration=%dms",
            self.name, cp.returncode, ok, duration_ms,
        )
        return GateResult(
            name=self.name,
            ok=ok,
            skipped=False,
            returncode=cp.returncode,
            duration_ms=duration_ms,
            stdout_tail=_tail_text(cp.stdout),
            stderr_tail=_tail_text(cp.stderr),
        )


def _tail_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = repr(data)
    else:
        text = data
    if len(text) <= STDOUT_TAIL_BYTES:
        return text
    return "...<truncated>...\n" + text[-STDOUT_TAIL_BYTES:]


# ──────────────────────────────────────────────────────────────────────────
# 具象ゲート
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PytestGate(_SubprocessGateBase):
    """``python scripts/safe_pytest.py`` を呼ぶゲート。

    ``scope`` を指定すると ``safe_pytest.py`` の位置引数として渡される。
    ``repo_root`` は ``cwd`` を兼ねる。
    ``python_executable`` で Python 実行形を差し替え可能 (デフォルト ``python``)。
    """

    name: str = "pytest"
    repo_root: Path | None = None
    scope: tuple[str, ...] = ()
    python_executable: str = "python"
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cwd is None and self.repo_root is not None:
            self.cwd = self.repo_root

    def build_command(self) -> list[str]:
        if self.repo_root is None:
            raise ValueError("PytestGate.repo_root must be set")
        script = self.repo_root / "scripts" / "safe_pytest.py"
        cmd: list[str] = [self.python_executable, str(script)]
        cmd.extend(self.extra_args)
        cmd.extend(self.scope)
        return cmd

    def precheck_skip_reason(self) -> str | None:
        if self.repo_root is None:
            return "repo_root unset"
        script = self.repo_root / "scripts" / "safe_pytest.py"
        if not script.exists():
            return f"safe_pytest.py not found at {script}"
        return None


@dataclass
class TypecheckGate(_SubprocessGateBase):
    """``cd frontend && npm run check`` を呼ぶゲート"""

    name: str = "typecheck"
    frontend_dir: Path | None = None
    npm_executable: str | None = None
    npm_args: tuple[str, ...] = ("run", "check")

    def __post_init__(self) -> None:
        if self.cwd is None and self.frontend_dir is not None:
            self.cwd = self.frontend_dir

    def _resolve_npm(self) -> str | None:
        if self.npm_executable:
            return self.npm_executable
        # Windows では npm.cmd / npm.bat 経由で解決される
        return shutil.which("npm")

    def build_command(self) -> list[str]:
        npm = self._resolve_npm()
        if npm is None:
            raise FileNotFoundError("npm executable not found")
        return [npm, *self.npm_args]

    def precheck_skip_reason(self) -> str | None:
        if self.frontend_dir is None:
            return "frontend_dir unset"
        if not self.frontend_dir.exists():
            return f"frontend_dir not found: {self.frontend_dir}"
        if self._resolve_npm() is None:
            return "npm not found in PATH"
        return None


@dataclass
class LintGate(TypecheckGate):
    """``cd frontend && npm run lint`` を呼ぶゲート (デフォルト OFF)"""

    name: str = "lint"
    enabled: bool = False
    npm_args: tuple[str, ...] = ("run", "lint")


# ──────────────────────────────────────────────────────────────────────────
# Runner / Decision / Rollback
# ──────────────────────────────────────────────────────────────────────────


def run_quality_gates(gates: Sequence[QualityGate]) -> QualityGateOutcome:
    """与えられたゲートを宣言順に実行して集約結果を返す。

    ある 1 つのゲートが失敗しても残りのゲートは実行する (ループ運用上、複数の
    失敗を一度に確認できた方が retry コストが低いため)。
    すべてのゲートが ``ok or skipped`` のとき outcome.ok = True。
    """
    results: list[GateResult] = []
    failed: list[str] = []
    skipped: list[str] = []
    for gate in gates:
        result = gate.run()
        results.append(result)
        if result.skipped:
            skipped.append(result.name)
        elif not result.ok:
            failed.append(result.name)
    ok = len(failed) == 0
    return QualityGateOutcome(
        ok=ok,
        results=tuple(results),
        failed=tuple(failed),
        skipped=tuple(skipped),
    )


def decide_gate_action(
    outcome: QualityGateOutcome,
    *,
    policy: OnGateFailPolicy,
    retries_so_far: int,
    retry_limit: int,
) -> GateAction:
    """``loop.on_gate_fail`` ポリシーから次アクションを決定する純粋関数。

    | policy | outcome.ok | retries_so_far < retry_limit | action  |
    |--------|------------|------------------------------|---------|
    | -      | True       | -                            | PROCEED |
    | retry  | False      | True                         | RETRY   |
    | retry  | False      | False                        | ABORT   |
    | skip   | False      | -                            | SKIP    |
    | abort  | False      | -                            | ABORT   |

    ``retries_so_far`` は今回失敗を含まない既往リトライ回数 (= 次に retry する
    場合は ``retries_so_far + 1`` 回目になる)。
    """
    if outcome.ok:
        return GateAction.PROCEED
    if policy == "abort":
        return GateAction.ABORT
    if policy == "skip":
        return GateAction.SKIP
    if policy == "retry":
        if retries_so_far < retry_limit:
            return GateAction.RETRY
        return GateAction.ABORT
    raise ValueError(f"unknown on_gate_fail policy: {policy!r}")


def rollback_working_tree(
    repo_root: Path,
    *,
    runner: SubprocessRunner = subprocess.run,
    include_untracked: bool = True,
    timeout_sec: float = 60.0,
) -> None:
    """作業ツリーの未コミット変更を破棄する。

    ループの 1 イテレーションが品質ゲートで失敗した場合に呼ばれることを想定する。
    本モジュールは **コミットを一切行わない** ため、ロールバック対象は常に
    未コミット変更のみ (HEAD はそのまま)。

    実行する git コマンド:

    1. ``git reset --hard HEAD`` — トラッキング済みファイルを HEAD に戻す
    2. ``git clean -fd`` (``include_untracked=True`` のみ) — 未追跡ファイル/ディレクトリを削除

    Raises:
        RuntimeError: ``git`` コマンドが見つからない、または失敗した場合
    """
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found in PATH")
    if not (repo_root / ".git").exists():
        raise RuntimeError(f"not a git repository: {repo_root}")

    cmds: list[list[str]] = [[git, "reset", "--hard", "HEAD"]]
    if include_untracked:
        cmds.append([git, "clean", "-fd"])

    for cmd in cmds:
        logger.warning(
            "rollback_working_tree: running %s in %s", cmd, repo_root,
        )
        cp = runner(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if cp.returncode != 0:
            raise RuntimeError(
                f"git rollback failed ({cmd}): rc={cp.returncode} "
                f"stderr={_tail_text(cp.stderr)}"
            )


# ──────────────────────────────────────────────────────────────────────────
# ファクトリ
# ──────────────────────────────────────────────────────────────────────────


def build_default_gates(
    config: LoopQualityGatesConfig,
    *,
    repo_root: Path,
    frontend_dir: Path | None = None,
    pytest_scope: Sequence[str] = (),
    runner: SubprocessRunner | None = None,
) -> list[QualityGate]:
    """``LoopQualityGatesConfig`` から有効なゲートのリストを組み立てる。

    ``frontend_dir`` 省略時は ``repo_root / 'frontend'`` を採用する。
    ``runner`` を渡すとすべてのゲートで共通の subprocess runner を使う
    (テスト用)。
    """
    fdir = frontend_dir if frontend_dir is not None else (repo_root / "frontend")
    gates: list[QualityGate] = []

    pytest_kwargs: dict[str, object] = {
        "enabled": config.pytest,
        "repo_root": repo_root,
        "scope": tuple(pytest_scope),
    }
    if runner is not None:
        pytest_kwargs["runner"] = runner
    gates.append(PytestGate(**pytest_kwargs))  # type: ignore[arg-type]

    typecheck_kwargs: dict[str, object] = {
        "enabled": config.typecheck,
        "frontend_dir": fdir,
    }
    if runner is not None:
        typecheck_kwargs["runner"] = runner
    gates.append(TypecheckGate(**typecheck_kwargs))  # type: ignore[arg-type]

    lint_kwargs: dict[str, object] = {
        "enabled": config.lint,
        "frontend_dir": fdir,
    }
    if runner is not None:
        lint_kwargs["runner"] = runner
    gates.append(LintGate(**lint_kwargs))  # type: ignore[arg-type]

    return gates


__all__ = [
    "GateAction",
    "GateResult",
    "LintGate",
    "OnGateFailPolicy",
    "PytestGate",
    "QualityGate",
    "QualityGateOutcome",
    "TypecheckGate",
    "build_default_gates",
    "decide_gate_action",
    "rollback_working_tree",
    "run_quality_gates",
]
