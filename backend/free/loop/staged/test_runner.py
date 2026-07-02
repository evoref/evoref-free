"""staged コーディングの test 工程: 生成テストをサンドボックスで実行する。

**安全上の不変則**: evoref 自身のテストスイートを起動しうる ``PytestGate`` /
``scripts/safe_pytest.py`` / ``build_default_gates`` は **使わない**。これらは
``repo_root`` 固定・``cwd=REPO_ROOT`` 固定で evoref の test を収集するため、
誤って system/e2e (PC フリーズ/強制再起動リスク) に触れる。

代わりにワークスペースを ``repo_root``/``cwd``/``rootdir`` とした専用
:class:`ActionRunner` で ``python -m pytest <単一テストファイル>`` を実行する。

**evoref 本体テストに到達しない実際の仕組み (正確な記述)**:

- 収集を限定しているのは ``--rootdir`` ではなく、**位置引数で単一テストファイルを
  明示指定する**こと。pytest は指定されたファイルだけを収集するため、evoref の
  ``tests/system`` 等は対象にならない。``run()`` は常に 1 ファイルのみを渡す
  (この不変則が崩れると保証も崩れるので、ディレクトリ指定に変えてはいけない)。
- ワークスペース既定は ``local/coding/`` で **repo 配下**にあるため、pytest は祖先を
  遡って evoref のルート ``conftest.py`` / ``pytest.ini`` を **読み込む**
  (``--rootdir`` は rootdir 算出を変えるだけで conftest/ini 探索は止めない)。これは
  害ではなく、ルート ``conftest.py`` の subprocess ガードが生成テスト側の危険な
  サブプロセス起動を **追加で抑止**する防御として機能する (defense-in-depth)。
- 継承される ``addopts`` (``-m "not system and not e2e"`` 等) は無印の生成テストに
  影響しない。将来 repo の addopts にグローバル ``-m`` / プラグインが増えた場合は
  収集が変わりうるため、その時は ``--confcutdir`` / ``-c`` での明示隔離を検討する。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from backend.free.harness.action import RunCommandAction
from backend.free.loop.action_runner import ActionRunner, ActionRunnerConfig
from backend.free.loop.quality_gate import GateResult
from backend.free.loop.staged.workspace import WorkspaceManager, _safe_rel
from backend.io.atomic import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("loop.staged.test_runner")

# 生成テストが src/ のモジュールを ``import`` できるようにする bootstrap。
# ワークスペース直下の conftest.py として書き出す (sys.path に src/ を足すだけ)。
# 注: ワークスペースが repo 配下にある場合 evoref のルート conftest も併せて
# 読み込まれるが、収集対象は run() が渡す単一ファイルに限定されるため evoref
# 本体テストは走らない (詳細はモジュール docstring 参照)。
_CONFTEST_BODY = """\
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
"""

_SUMMARY_RE = re.compile(
    r"(?:(\d+) failed)?(?:, )?(?:(\d+) passed)?(?:, )?(?:(\d+) error)?",
)


@dataclass
class StagedTestRunner:
    """ワークスペース限定で pytest を実行する。

    Args:
        workspace: 対象 :class:`WorkspaceManager`。
        test_timeout_sec: pytest サブプロセスのタイムアウト。
        python_exe: 使用する python 実行ファイル (既定: 現行インタプリタ)。
    """

    workspace: WorkspaceManager
    test_timeout_sec: float = 120.0
    python_exe: str = field(default_factory=lambda: sys.executable or "python")

    def _ensure_bootstrap(self) -> None:
        conftest = self.workspace.path("conftest.py")
        if not conftest.exists():
            with AtomicWriter(conftest) as f:
                f.write(_CONFTEST_BODY)

    def _build_runner(self) -> ActionRunner:
        cfg = ActionRunnerConfig(
            allowed_write_roots=(self.workspace.root,),
            allowed_commands=("python",),
            command_timeout_sec=self.test_timeout_sec,
        )
        return ActionRunner(config=cfg, repo_root=self.workspace.root)

    def _pytest_command(self, target) -> tuple[str, ...]:
        """ワークスペース限定の pytest コマンドを組む。

        収集を限定する実体は **位置引数 ``target`` (単一テストファイル)**。pytest は
        指定ファイルのみ収集するため evoref の system/e2e には到達しない。``--rootdir``
        は rootdir 算出を揃えるためで、conftest/ini 探索を止めるものではない
        (詳細はモジュール docstring 参照)。
        """
        return (
            self.python_exe, "-m", "pytest",
            "-p", "no:cacheprovider", "-q",
            "--rootdir", str(self.workspace.root),
            str(target),
        )

    def run(self, *, test_logical_path: str) -> GateResult:
        """ワークスペース内の単一テストファイルに対して pytest を実行する。

        この呼び出しは同期 (subprocess.run)。executor からは
        ``await asyncio.to_thread(runner.run, ...)`` で呼ぶこと。
        """
        self._ensure_bootstrap()
        ar = self._build_runner()
        test_ws_rel = f"tests/{_safe_rel(test_logical_path)}"
        target = self.workspace.root / test_ws_rel
        cmd = self._pytest_command(target)
        action = RunCommandAction(command=cmd, cwd=str(self.workspace.root))
        result = ar.run_one(action)

        out = (result.output or "") + "\n" + (result.error or "")
        rc_raw = (result.metadata or {}).get("returncode", "")
        try:
            returncode: int | None = int(rc_raw) if rc_raw != "" else None
        except ValueError:
            returncode = None
        failed, passed, errors = _parse_pytest_summary(out)
        # rc 5 = no tests collected (pytest 規約)。テスト工程としては失敗扱い。
        no_tests = returncode == 5
        ok = bool(result.success) and not no_tests
        summary = _summarize(failed, passed, errors, no_tests, returncode)
        return GateResult(
            name="staged_pytest",
            ok=ok,
            skipped=False,
            returncode=returncode,
            duration_ms=int(result.duration_ms),
            stdout_tail=result.output or "",
            stderr_tail=result.error or "",
            error=None if ok else summary,
        )


def _parse_pytest_summary(text: str) -> tuple[int, int, int]:
    """pytest 末尾サマリから (failed, passed, error) 件数を best-effort 抽出。"""
    failed = passed = errors = 0
    for m in _SUMMARY_RE.finditer(text):
        f, p, e = m.group(1), m.group(2), m.group(3)
        if f:
            failed = max(failed, int(f))
        if p:
            passed = max(passed, int(p))
        if e:
            errors = max(errors, int(e))
    return failed, passed, errors


def _summarize(
    failed: int, passed: int, errors: int, no_tests: bool, rc: int | None,
) -> str:
    if no_tests:
        return "no tests collected"
    parts: list[str] = []
    if failed:
        parts.append(f"{failed} failed")
    if errors:
        parts.append(f"{errors} error")
    if passed:
        parts.append(f"{passed} passed")
    if not parts:
        parts.append(f"rc={rc}")
    return ", ".join(parts)
