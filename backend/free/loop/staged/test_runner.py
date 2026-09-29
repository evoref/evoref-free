"""staged クリエイトの test 工程: 生成テストをサンドボックスで実行する。

**安全上の不変則**: evoref 自身のテストスイートを起動しうる
``scripts/safe_pytest.py`` は **使わない**。``cwd=REPO_ROOT`` 固定で evoref の
test を収集するため、誤って system/e2e (PC フリーズ/強制再起動リスク) に触れる。

代わりにワークスペースを ``repo_root``/``cwd``/``rootdir`` とした専用
:class:`ActionRunner` で ``python -m pytest <単一テストファイル>`` を実行する。

**evoref 本体テストに到達しない実際の仕組み (正確な記述)**:

- 収集を限定しているのは ``--rootdir`` ではなく、**位置引数で単一テストファイルを
  明示指定する**こと。pytest は指定されたファイルだけを収集するため、evoref の
  ``tests/system`` 等は対象にならない。``run()`` は常に 1 ファイルのみを渡す
  (この不変則が崩れると保証も崩れるので、ディレクトリ指定に変えてはいけない)。
- ワークスペース既定は ``<data_root>/g1/store/create/`` で **repo 配下** (既定のデータ根 ``userdata/``) にあるため、pytest は祖先を
  遡って evoref のルート ``conftest.py`` / ``pytest.ini`` を **読み込む**
  (``--rootdir`` は rootdir 算出を変えるだけで conftest/ini 探索は止めない)。これは
  害ではなく、ルート ``conftest.py`` の subprocess ガードが生成テスト側の危険な
  サブプロセス起動を **追加で抑止**する防御として機能する (defense-in-depth)。
- 継承される ``addopts`` (``-m "not system and not e2e"`` 等) は無印の生成テストに
  影響しない。将来 repo の addopts にグローバル ``-m`` / プラグインが増えた場合は
  収集が変わりうるため、その時は ``--confcutdir`` / ``-c`` での明示隔離を検討する。
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.core import fs_sandbox
from backend.free.core.check_outcome import UncheckedReason
from backend.free.harness.action import RunCommandAction
from backend.free.loop.action_runner import ActionRunner, ActionRunnerConfig
from backend.free.loop.quality_gate import GateResult
from backend.free.loop.staged.workspace import WorkspaceManager, _safe_rel
from backend.io.atomic import AtomicWriter
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("loop.staged.test_runner")

# collection error 中の欠落モジュール抽出 (環境要因判定用)。
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
# stdlib 名 (py3.10+)。`_curses` → `curses` の私的名正規化は判定側で行う。
_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()))

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

""" + fs_sandbox.CONFTEST_GUARD

_PYTEST_LAUNCHER = "run_pytest.py"
#: 第 1 引数は作業フォルダの根。pytest の表示 (``tests/test_x.py::test_a``) を根からの相対に保つため
#: 根で起動し、設定が済んだら (テストの収集・実行の前に) 実行ごとの空の CWD へ戻る。
_PYTEST_LAUNCHER_BODY = (
    "# evoref: pytest をスクリプトとして起動する (-m では audit hook を入れられない、f_10 §11.1-4)\n"
    "import os\nimport sys\n\nimport pytest\n\n"
    "_CWD = os.getcwd()\n\n\n"
    "class _BackToScratchCwd:\n"
    "    @staticmethod\n"
    "    def pytest_configure(config):\n"
    "        os.chdir(_CWD)\n\n\n"
    "os.chdir(sys.argv[1])\n"
    "sys.exit(pytest.main(sys.argv[2:], plugins=[_BackToScratchCwd()]))\n"
)


def _is_link(path: Path) -> bool:
    """シンボリックリンクかジャンクションか (どちらも中へ入らない)。"""
    return os.path.islink(path) or bool(getattr(os.path, "isjunction", lambda _p: False)(path))


def _unlink_link(path: Path) -> None:
    """リンクそのものを消す (ジャンクション・ディレクトリへのリンクは rmdir、先は触らない)。"""
    try:
        os.unlink(path)
    except (IsADirectoryError, PermissionError):
        os.rmdir(path)


def _prune_untracked(directory: Path, src_real: Path, tracked: set[Path]) -> None:
    """``directory`` の下の manifest に無いファイルと空のフォルダを消す (リンクは辿らない)。"""
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        logger.debug("staged test state reset: could not list %s: %s", directory, exc)
        return
    for entry in entries:
        path = Path(entry.path)
        try:
            if _is_link(path):
                _unlink_link(path)
                continue
            if not path.resolve().is_relative_to(src_real):
                continue
            if entry.is_dir(follow_symlinks=False):
                _prune_untracked(path, src_real, tracked)
                if not any(path.iterdir()):
                    path.rmdir()
            elif path.resolve() not in tracked:
                path.unlink()
        except OSError as exc:
            logger.debug("staged test state reset: could not remove %s: %s", path, exc)


#: pytest の要約の失敗行 (``FAILED tests/test_x.py::test_a - AssertionError: …``)。
_FAILED_LINE_RE = re.compile(r"^(FAILED|ERROR) (\S+?)(?: - (.*))?$", re.MULTILINE)

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
        debug_logger: pytest 不合格時の詳細 (stdout/stderr tail) を
            ``long_form.jsonl`` へ構造化記録する (任意)。
    """

    workspace: WorkspaceManager
    test_timeout_sec: float = 120.0
    python_exe: str = field(default_factory=lambda: sys.executable or "python")
    debug_logger: "DebugLogger | None" = None

    def _ensure_bootstrap(self) -> None:
        # 既存の作業フォルダ (再接続・v1 の再開) の古い conftest にも隔離の備えを入れる
        conftest = self.workspace.path("conftest.py")
        try:
            if conftest.read_text(encoding="utf-8") == _CONFTEST_BODY:
                return
        except OSError:
            pass
        with AtomicWriter(conftest) as f:
            f.write(_CONFTEST_BODY)

    def _reset_run_state(self) -> None:
        """前回の実行が残した状態を消す (実行ごとに同じ状態から始める、f_10 §11.1-4)。

        ``src/`` の下で manifest に無いファイル (テストが作った保存ファイル等) と、
        ``.sandbox/`` のホーム・一時フォルダ・CWD。K01 では修正前の版が壊した ``todos.json`` を
        修正後の版が読み、4 件とも偽の不合格になった。manifest が読めなければ何も消さない。
        シンボリックリンク・ジャンクションは中へ入らず、リンクそのものだけを消す
        (``rglob`` はジャンクションを辿り、作業フォルダの外の中身を消した — 独立レビュー)。
        """
        fs_sandbox.reset_state(self.workspace.root)
        src = self.workspace.root / "src"
        if not src.is_dir() or _is_link(src):
            return
        try:
            tracked = {
                (self.workspace.root / f.workspace_path).resolve() for f in self.workspace.list_files()
            }
        except Exception as exc:  # noqa: BLE001 - 読めなければ消さない (安全側)
            logger.debug("staged test state reset skipped: %s", exc)
            return
        if not tracked:
            return
        _prune_untracked(src, src.resolve(), tracked)

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
        # pytest はスクリプトとして起動する: ``-m`` だと audit hook を入れた途端に
        # import が壊れる (f_10 §11.1-4)。入口は ``.sandbox/run_pytest.py``。
        launcher = fs_sandbox.sandbox_dir(self.workspace.root) / _PYTEST_LAUNCHER
        return (
            self.python_exe, str(launcher), str(self.workspace.root),
            "-p", "no:cacheprovider", "-q",
            # 失敗トレースを圧縮し、spec 見直しループの evidence 窓 (末尾 3000
            # chars) に複数の失敗が収まるようにする。
            "--tb=short",
            "--rootdir", str(self.workspace.root),
            str(target),
        )

    def run(self, *, test_logical_path: str) -> GateResult:
        """ワークスペース内の単一テストファイルに対して pytest を実行する。

        この呼び出しは同期 (subprocess.run)。executor からは
        ``await asyncio.to_thread(runner.run, ...)`` で呼ぶこと。
        """
        self._ensure_bootstrap()
        self._reset_run_state()
        ar = self._build_runner()
        test_ws_rel = f"tests/{_safe_rel(test_logical_path)}"
        target = self.workspace.root / test_ws_rel
        cmd = self._pytest_command(target)
        # 書込みを許すのは src/ と .sandbox の home / tmp / cwd だけ (孫プロセスにも継承、f_10 §11.1-4)。
        # CWD は実行ごとの空フォルダ (CWD 相対の保存ファイルを次の実行へ持ち越さない)
        fs_sandbox.write_script(self.workspace.root, _PYTEST_LAUNCHER, _PYTEST_LAUNCHER_BODY)
        env = fs_sandbox.sandbox_env(self.workspace.root, write_dirs=[self.workspace.root / "src"])
        action = RunCommandAction(
            command=cmd, cwd=str(fs_sandbox.sandbox_cwd(self.workspace.root)), env=env,
        )
        result = ar.run_one(action)
        violations = fs_sandbox.violation_paths(fs_sandbox.read_violations(self.workspace.root))

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

        if violations:
            # 外への書込みを止めた結果の合否はコードの誤りの証拠にならない — 未検査
            # (不合格にすると作り直し・偽の不合格を生む、2026-09-27 ライブ監査 K01)
            logger.warning(
                "staged test %s tried to write outside the workspace: %s",
                test_logical_path, violations[:3],
            )
            gate = GateResult(
                name="staged_pytest",
                ok=False,
                skipped=True,
                returncode=returncode,
                duration_ms=int(result.duration_ms),
                stdout_tail=result.output or "",
                stderr_tail=result.error or "",
                error=summary,
                skip_reason=f"sandbox violation: {', '.join(violations[:3])}",
                skip_kind=UncheckedReason.SANDBOX_VIOLATION.value,
                skip_detail=", ".join(violations[:3]),
            )
            self._log_gate_result(gate, test_logical_path)
            return gate

        # collection error の原因が「非生成物・非 stdlib モジュールの
        # ModuleNotFoundError」なら環境要因 (外部依存未インストール) であり、
        # コードの欠陥ではない。import スモークが同ケースを warning に分類する
        # のと整合させ、skipped=True で返して呼出側 (executor) が spec 見直し
        # トリガ・failure 集計・テスト未合格警告に乗せないようにする
        # (2026-07-06 live: pygame 未インストールで見直しループが誤発火)。
        if not ok:
            env_dep = _detect_env_missing_dependency(
                out, self._src_stems(), self._internal_names(),
                src_imports=self._src_imports(),
            )
            if env_dep:
                gate = GateResult(
                    name="staged_pytest",
                    ok=False,
                    skipped=True,
                    returncode=returncode,
                    duration_ms=int(result.duration_ms),
                    stdout_tail=result.output or "",
                    stderr_tail=result.error or "",
                    error=summary,
                    skip_reason=f"外部依存 '{env_dep}' が未インストール (環境要因)",
                    skip_kind=UncheckedReason.MISSING_DEPENDENCY.value,
                    skip_detail=env_dep,
                )
                self._log_gate_result(gate, test_logical_path)
                return gate
        gate = GateResult(
            name="staged_pytest",
            ok=ok,
            skipped=False,
            returncode=returncode,
            duration_ms=int(result.duration_ms),
            stdout_tail=result.output or "",
            stderr_tail=result.error or "",
            error=None if ok else summary,
        )
        self._log_gate_result(gate, test_logical_path)
        return gate

    def _log_gate_result(self, gate: GateResult, test_logical_path: str) -> None:
        """pytest 不合格時のみ詳細 (stdout/stderr tail) を JSONL へ記録する。

        合格時は manifest 側 (record_test_result) に残るため、ここでは
        develop=investigate/evolve での障害解析に要る不合格ケースに絞る。
        """
        if gate.ok or self.debug_logger is None:
            return
        try:
            self.debug_logger.log_long_form_event({
                "phase": "staged_test",
                "test_logical_path": test_logical_path,
                "ok": gate.ok,
                "skipped": gate.skipped,
                "returncode": gate.returncode,
                "error": gate.error,
                "skip_reason": gate.skip_reason,
                "stdout_tail": (gate.stdout_tail or "")[-2000:],
                "stderr_tail": (gate.stderr_tail or "")[-2000:],
            })
        except Exception as exc:
            logger.debug("staged test gate result log failed: %s", exc)

    def _src_stems(self) -> set[str]:
        """生成 src のモジュール stem 集合 (欠落依存の環境要因判定用)。

        ネストしたパス (``core/game.py``) はトップレベルのパッケージ名 (``core``)
        も含める (import スモークの ``_stems`` と判定基準を揃える)。
        """
        try:
            stems: set[str] = set()
            for f in self.workspace.list_files(kind="src"):
                path = Path(f.logical_path)
                stems.add(path.stem)
                if len(path.parts) > 1:
                    stems.add(path.parts[0])
            return stems
        except Exception:  # noqa: BLE001 - 判定不能時は空集合 (env-skip しない側へ)
            return set()

    def _src_imports(self) -> frozenset[str] | None:
        """生成 src が import するモジュールの先頭名 (読めなければ ``None`` = 判定しない)。

        import スモークは外部依存の欠落を警告で通すので、本物の外部依存なら src 側にも
        同じ import がある。テストだけが import する名前をコードの欠陥にする判定用 (f_10)。
        """
        try:
            names: set[str] = set()
            for f in self.workspace.list_files(kind="src"):
                if not f.logical_path.endswith(".py"):
                    continue
                code = self.workspace.read_file(f.logical_path, kind="src")
                if code is None:
                    return None
                for node in ast.walk(ast.parse(code)):
                    if isinstance(node, ast.Import):
                        names.update(a.name.split(".")[0] for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names.add(node.module.split(".")[0])
            return frozenset(names)
        except Exception:  # noqa: BLE001 - 判定不能時は従来どおり (env-skip しうる側)
            return None

    def _internal_names(self) -> frozenset[str]:
        """spec が宣言する内部契約名 (幻覚内部 import を env-skip させない判定用)。"""
        try:
            from backend.free.loop.staged.spec_parts import internal_contract_names
            return internal_contract_names(self.workspace.read_spec() or "")
        except Exception:  # noqa: BLE001 - 判定不能時は空 (env-skip 側へ)
            return frozenset()


def _detect_env_missing_dependency(
    out: str, src_stems: set[str],
    internal_names: frozenset[str] = frozenset(),
    *,
    src_imports: frozenset[str] | None = None,
) -> str | None:
    """collection error の原因が環境要因の外部依存欠落なら、そのモジュール名を返す。

    条件: 「ERROR collecting」を含み、欠落モジュールがすべて
    (a) 生成物の stem でない (cross-file 欠落や幻覚 import はコード欠陥) かつ
    (b) stdlib でない (ターゲット OS で起動不能な欠陥) かつ
    (c) spec の内部契約名 (Component 名 / 正準モジュール stem、case-insensitive)
    でない (幻覚内部 import はコード欠陥。2026-07-07 live: `from game import
    Game` が env-skip され偽 success で配信された) こと。
    ``src_imports`` を渡したときは (d) 生成 src のどれかが import している名前で
    あること (テストだけが import する名前は生成されなかった兄弟か幻覚、2026-09-26
    trace 74893037692b の ``import storage``)。
    1 つでもコード欠陥側の欠落が混ざる場合は None (通常の失敗扱い)。
    """
    if "ERROR collecting" not in out:
        return None
    missing_names = {
        m.group(1).split(".")[0] for m in _MISSING_MODULE_RE.finditer(out)
    }
    if not missing_names:
        return None
    lowered_internal = {n.lower() for n in internal_names}
    env_only: list[str] = []
    for name in sorted(missing_names):
        if name in src_stems:
            return None
        if name in _STDLIB_MODULES or name.lstrip("_") in _STDLIB_MODULES:
            return None
        if name.lower() in lowered_internal:
            return None
        if src_imports is not None and name not in src_imports:
            return None
        env_only.append(name)
    return env_only[0] if env_only else None


def failure_lines(gate: GateResult, limit: int = 3) -> list[str]:
    """失敗したテストの要約 (pytest の ``FAILED …`` / ``ERROR …`` 行、先頭 ``limit`` 件)。

    ``FAILED tests/test_examples.py::test_example_3 - json.decoder.JSONDecodeError:...`` →
    ``test_example_3 — json.decoder.JSONDecodeError:...``。
    """
    text = ((gate.stdout_tail or "") + "\n" + (gate.stderr_tail or "")).replace("\r", "")
    out: list[str] = []
    for m in _FAILED_LINE_RE.finditer(text):
        name = m.group(2).rsplit("::", 1)[-1]
        item = f"{name} — {m.group(3).strip()}" if m.group(3) else name
        if item not in out:
            out.append(item)
    return out[:limit]


def failed_count(gate: GateResult) -> int:
    """不合格の件数 (failed + error)。合格・未検査は 0。"""
    if gate.ok or gate.skipped:
        return 0
    failed, _passed, errors = _parse_pytest_summary(gate.error or "")
    return failed + errors


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
