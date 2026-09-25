"""evoref reset サブコマンド — データ根を初期状態へ戻す (c_05 §0.2 / §0.3)。

消すもの: このリリースの世代フォルダの ``<data_root>/g<N>/store/`` (記憶・履歴・学習・
corpus・モデル切替状態) と ``g<N>/cache/``、データ根直下の ``logs/`` ``tmp/`` の中身。

残すもの: ``outputs/`` ``themes/`` ``profiles/`` (利用者の成果物と上書き)、
``store/pro/`` と世代印 (``--include-pro`` を付けない限り。Pro の形式の版を
世代印が覚えているので、片方だけ消すと次の起動が readonly になる)、
他の世代フォルダ (``g<M>/``。消すのは ``evoref data prune`` の役目、G2 から)、
台帳で ``keep_on_reset`` を宣言した形式のファイル (作り直しが高価な derived。
補助タスクの timeout 較正・履歴要約の埋め込み。``--include-cache`` で消す)。

選択的な初期化 (G1 設計 §16.2 #13):

- ``--learning``: ``store/learning/`` (``--include-pro`` なら ``store/pro/learning/`` も)
  と、semantic ストアの ``learn.*`` ファクト。ファクトは記憶のファイルを消さずに
  EvidenceStore の ``retract`` で取り消す (書き手ロックの下、停止中に)。
- ``--memory``: ``store/memory/`` だけ。semantic ストアの ``learn.*`` ファクト
  (学習の結果) は残す — 消す前に取り出し、空の semantic ストアへ ``create`` し直す。
  取り消し (``retract``) で済ませないのは、記憶の本文が事象ログに残るため。

どちらも ``logs/`` ``tmp/`` ``cache/`` と世代印には触れない。

単一書き手ロック (``store/.writer.lock``) を取ってから消すので、serve の稼働中は
拒否する。確認は ``--yes`` で省く。Develop の初期化 API
(``POST /api/develop/reset-local-data``) は本モジュールを
``python -m backend.free.cli.reset_command --yes --include-pro --stop-services``
でデタッチ起動する (同じ経路で消す)。``--stop-services`` はロックを持つ backend と
llama-server / frontend をポートで止めてからロックを取る。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import PathResolver
from backend.data_root import DataRootError, data_path, generation_root, resolve_data_root, store_root
from backend.io import ledger_files
from backend.io.generation_seal import GITKEEP_FILENAME, SEAL_FILENAME
from backend.io.writer_lock import LOCK_FILENAME, WriterLockHeld, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("cli.reset")

#: 中身を消す (ディレクトリ自体は残す) ディレクトリ (``cache`` は世代フォルダの下、:func:`data_path`)。
WIPED_DIRS: tuple[str, ...] = ("logs", "tmp", "cache")
#: ``--include-pro`` を付けない限り ``store/`` に残す項目。
PRO_KEPT: tuple[str, ...] = ("pro", SEAL_FILENAME)
#: ``--learning`` で取り消す semantic ファクトの namespace。
LEARNING_NAMESPACE = "learn"
#: 取り消し事象の書き手 (事象ログの ``by``)。
RETRACT_BY = "cli.reset"

# evoref-ctl.bat が立てるウィンドウタイトル (start "<title>" ...)。
_WINDOW_TITLES = ("llama-server", "evoref-backend", "evoref-frontend")
_FRONTEND_PORT = 5173
_DEFAULT_PORTS = [8000, 8080, 8082]


@dataclass(slots=True)
class ResetReport:
    """初期化の結果 (消した項目数と、消せなかった項目)。"""

    removed: int = 0
    kept_pro: bool = False
    #: ``keep_on_reset`` の形式のため残したファイルの数。
    kept_cache: int = 0
    #: ``--learning`` で取り消した ``learn.*`` ファクトの数。
    retracted: int = 0
    #: ``--memory`` で残した (作り直したストアへ入れ直した) ``learn.*`` ファクトの数。
    kept_learning: int = 0
    errors: list[str] = field(default_factory=list)


def _remove(path: Path, *, retries: int, delay: float) -> str | None:
    """``path`` (ファイル / ディレクトリ) を消す。Windows のロック残りに備えて軽く再試行する。"""
    for attempt in range(retries):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            return None
        except OSError as e:
            if attempt == retries - 1:
                return f"{path}: {e}"
            time.sleep(delay)
    return None


def kept_on_reset_classifier() -> ledger_files.Classifier:
    """データ根からの相対 posix パス → 形式の関数 (``keep_on_reset`` の判定に使う)。"""
    from backend.formats import load_all_formats

    load_all_formats()
    return ledger_files.classifier()


def _kept_files(data_root: Path, bases: list[Path], classify: ledger_files.Classifier) -> set[Path]:
    """``bases`` の下で残すファイル (形式が ``keep_on_reset`` を宣言しているもの)。"""
    kept: set[Path] = set()
    generation_dir = generation_root(data_root)
    for base in bases:
        # 台帳 (path_key) は世代フォルダの中だけ。logs/ tmp/ に keep_on_reset の形式は無い
        if not base.is_dir() or not base.is_relative_to(generation_dir):
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            spec = classify(path.relative_to(generation_dir).as_posix())
            if spec is not None and spec.keep_on_reset:
                kept.add(path)
    return kept


def _targets_in(directory: Path, kept: set[Path], kept_dirs: set[Path], skip: set[str]) -> list[Path]:
    """``directory`` の中で消すもの。残すファイルを含むディレクトリは中へ降りる。"""
    out: list[Path] = []
    for entry in directory.iterdir():
        if entry.name in skip or entry in kept:
            continue
        if entry in kept_dirs:
            out += _targets_in(entry, kept, kept_dirs, set())
        else:
            out.append(entry)
    return out


def _open_semantic_store(data_root: Path):
    """停止中の semantic ストアを開く (readonly なら例外)。無ければ ``None``。"""
    from backend.free.memory.semantic.store import STORE_DIRNAME
    from backend.free.rag.evidence.store import EvidenceStore, EvidenceStoreReadonlyError

    store_dir = PathResolver.layout_path(data_root, "memory_dir") / STORE_DIRNAME
    if not store_dir.is_dir():
        return None
    store = EvidenceStore(store_dir, store_name="semantic", by=RETRACT_BY)
    store.load()
    if store.readonly:
        reason = store.readonly_reason
        store.close()
        raise EvidenceStoreReadonlyError(f"semantic store is read-only ({reason})")
    return store


def retract_learning_facts(data_root: Path) -> int:
    """semantic ストアの ``learn.*`` ファクトを取り消す (記憶のファイルは消さない)。

    停止中に書き手ロックの下で呼ぶ (呼出側が持つ)。取り消しは事象ログへの
    ``retract`` で、次の起動の replay で反映される。ストアが readonly (新しい版 /
    読めない manifest) なら例外で止める。

    Returns:
        取り消したファクトの数。
    """
    store = _open_semantic_store(data_root)
    if store is None:
        return 0
    try:
        ids = [
            record.id for record in store.iter_records()
            if record.namespace == LEARNING_NAMESPACE and record.veracity != "retracted"
        ]
        for record_id in ids:
            store.retract(record_id, "reset --learning", by=RETRACT_BY)
        if ids:
            store.save_manifest()
        return len(ids)
    finally:
        store.close()


def take_learning_facts(data_root: Path) -> list:
    """semantic ストアの取り消されていない ``learn.*`` ファクトを取り出す (ストアは変えない)。"""
    store = _open_semantic_store(data_root)
    if store is None:
        return []
    try:
        return [
            record for record in store.iter_records()
            if record.namespace == LEARNING_NAMESPACE and record.veracity != "retracted"
        ]
    finally:
        store.close()


def restore_learning_facts(data_root: Path, records: list) -> int:
    """消した後の空の semantic ストアへ ``learn.*`` ファクトを ``create`` し直す。

    埋め込みは次の snapshot が作る。
    """
    from backend.free.memory.semantic.store import STORE_DIRNAME
    from backend.free.rag.evidence.store import EvidenceStore

    if not records:
        return 0
    store_dir = PathResolver.layout_path(data_root, "memory_dir") / STORE_DIRNAME
    store_dir.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(store_dir, store_name="semantic", by=RETRACT_BY)
    store.load()
    try:
        for record in records:
            store.create(record, by=RETRACT_BY)
        store.save_manifest()
    finally:
        store.close()
    return len(records)


def reset_data_root(
    data_root: Path,
    *,
    include_pro: bool = False,
    learning: bool = False,
    memory: bool = False,
    include_cache: bool = False,
    retries: int = 8,
    retry_delay: float = 0.5,
) -> ResetReport:
    """データ根を初期状態へ戻す (書き手ロックは呼出側が持つ)。

    ``learning`` / ``memory`` のどちらも偽なら全体、どちらかが真ならその範囲だけ。
    """
    report = ResetReport()
    store = store_root(data_root)
    selective = learning or memory
    # (消す範囲の根, 根の直下で飛ばす名前)
    scopes: list[tuple[Path, set[str]]] = []
    if selective:
        if learning:
            scopes.append((PathResolver.layout_path(data_root, "learning_dir"), set()))
            if include_pro:
                scopes.append((store / "pro" / "learning", set()))
        if memory:
            scopes.append((PathResolver.layout_path(data_root, "memory_dir"), set()))
    else:
        skip = {LOCK_FILENAME}
        if not include_pro and (store / "pro").exists():
            skip.update(PRO_KEPT)
            report.kept_pro = True
        scopes.append((store, skip))
        scopes += [(data_path(data_root, name), set()) for name in WIPED_DIRS]

    if learning and not memory:
        try:
            report.retracted = retract_learning_facts(data_root)
        except Exception as e:  # noqa: BLE001 — 報告して続ける (ファイルの削除は進める)
            report.errors.append(f"learn.* facts: {e}")
    kept_learning: list = []
    if memory and not learning:
        try:
            kept_learning = take_learning_facts(data_root)
        except Exception as e:  # noqa: BLE001 — 取り出せないなら記憶を消さない
            report.errors.append(f"learn.* facts: {e}")
            return report

    bases = [base for base, _ in scopes if base.is_dir()]
    # ディレクトリ構成の印 (リポジトリで追跡する .gitkeep、c_03 §10.1) は消した後に戻す
    gitkeeps = [path for base in bases for path in base.rglob(GITKEEP_FILENAME)]
    kept = set() if include_cache else _kept_files(data_root, bases, kept_on_reset_classifier())
    report.kept_cache = len(kept)
    kept_dirs = {parent for path in kept for parent in path.parents}
    targets: list[Path] = []
    for base, skip in scopes:
        if base.is_dir():
            targets += _targets_in(base, kept, kept_dirs, skip)
    for path in targets:
        error = _remove(path, retries=retries, delay=retry_delay)
        if error is None:
            report.removed += 1
        else:
            report.errors.append(error)
    for marker in gitkeeps:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch(exist_ok=True)
        except OSError as e:
            report.errors.append(f"{marker}: {e}")
    if kept_learning:
        try:
            report.kept_learning = restore_learning_facts(data_root, kept_learning)
        except Exception as e:  # noqa: BLE001 — 報告する (学習の結果は失われる)
            report.errors.append(f"learn.* facts: {e}")
    return report


# ────────────────────────────────────────────
# --stop-services (Develop の初期化 API の経路)
# ────────────────────────────────────────────


def _config_ports(project_root: Path) -> list[int]:
    """config.yaml から停止対象ポートを集める。読めなければ既定ポート。"""
    from backend.free.cli.pid_manager import collect_configured_ports

    try:
        import yaml

        cfg = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8")) or {}
        ports = list(collect_configured_ports(cfg)) or list(_DEFAULT_PORTS)
    except Exception as e:  # noqa: BLE001 — config が読めなくても停止は続ける
        print(f"[reset] WARNING: failed to read config ports ({e}); using defaults")
        ports = list(_DEFAULT_PORTS)
    if _FRONTEND_PORT not in ports:
        ports.append(_FRONTEND_PORT)
    return ports


def _taskkill_window_titles() -> None:
    """Windows: evoref-ctl.bat が立てたウィンドウ単位でツリー kill する。"""
    if sys.platform != "win32":
        return
    commands = [["taskkill", "/F", "/T", "/FI", f"WINDOWTITLE eq {t}"] for t in _WINDOW_TITLES]
    commands.append(["taskkill", "/F", "/IM", "llama-server.exe"])
    for cmd in commands:
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass


def stop_services(project_root: Path, *, wait_timeout: float = 30.0) -> bool:
    """全サービス (backend + llama-server + frontend) を止め、ポートの解放を待つ。"""
    from backend.free.cli.pid_manager import find_port_occupants, kill_port_occupants

    ports = _config_ports(project_root)
    _taskkill_window_titles()
    own_pid = os.getpid()
    kill_port_occupants([o for o in find_port_occupants(ports) if o.pid != own_pid])
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if not find_port_occupants(ports):
            return True
        time.sleep(0.5)
    return not find_port_occupants(ports)


# ────────────────────────────────────────────
# entrypoint
# ────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    from backend.i18n_helper import msg

    parser = argparse.ArgumentParser(prog="evoref reset", description=msg("cli.help_reset"))
    parser.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    parser.add_argument(
        "--include-pro", action="store_true",
        help="Also delete store/pro/ (Pro learning data such as LoRA adapters)",
    )
    parser.add_argument(
        "--learning", action="store_true",
        help="Reset learning only: store/learning/ and learn.* facts in the semantic store",
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="Reset memory only: store/memory/",
    )
    parser.add_argument(
        "--include-cache", action="store_true",
        help=(
            "Also delete expensive derived caches that reset keeps "
            "(aux timeout calibration, history summary embeddings)"
        ),
    )
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root to reset (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    parser.add_argument(
        "--stop-services", action="store_true",
        help="Stop the backend, llama-server and frontend first (they hold the writer lock)",
    )
    parser.add_argument("--delay", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def _confirm_key(args: argparse.Namespace) -> str:
    """確認文の i18n キー (初期化の範囲で変わる)。"""
    if args.learning and args.memory:
        return "cli.reset_confirm_learning_memory"
    if args.learning:
        return "cli.reset_confirm_learning"
    if args.memory:
        return "cli.reset_confirm_memory"
    return "cli.reset_confirm_with_pro" if args.include_pro else "cli.reset_confirm"


def _confirmed(console, data_root: Path, args: argparse.Namespace) -> bool:
    from backend.i18n_helper import msg

    if not sys.stdin or not sys.stdin.isatty():
        from backend.free.cli.renderer import render_error

        render_error(console, msg("cli.reset_needs_yes"))
        return False
    answer = console.input(msg(_confirm_key(args), data_root=str(data_root)) + " [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def run_reset(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref reset``)。"""
    from backend.free.cli.config_loader import _find_project_root
    from backend.free.cli.renderer import create_console, render_error, render_info
    from backend.i18n_helper import init_i18n, msg

    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console(no_color=not (sys.stdout and sys.stdout.isatty()))
    project_root = _find_project_root()
    try:
        # 不正なデータ根 (models/ の中など) はサービスを止める前に拒否する。
        data_root = resolve_data_root(args.data_root, root=project_root)
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return 1
    if not args.yes and not _confirmed(console, data_root, args):
        render_info(console, msg("cli.reset_aborted"))
        return 1
    if args.stop_services:
        if args.delay > 0:
            time.sleep(args.delay)  # 202 応答がブラウザへ届くまでの猶予
        if not stop_services(project_root):
            print("[reset] WARNING: some ports are still occupied")
    try:
        lock = acquire_writer_lock(store_root(data_root))
    except WriterLockHeld as e:
        render_error(console, msg("cli.reset_locked", detail=str(e)))
        return 1
    try:
        report = reset_data_root(
            data_root, include_pro=args.include_pro, learning=args.learning,
            memory=args.memory, include_cache=args.include_cache,
        )
    finally:
        lock.release()
    render_info(console, msg("cli.reset_done", data_root=str(data_root), count=report.removed))
    if report.retracted:
        render_info(console, msg("cli.reset_retracted_learning", count=report.retracted))
    if report.kept_learning:
        render_info(console, msg("cli.reset_kept_learning", count=report.kept_learning))
    if report.kept_pro:
        render_info(console, msg("cli.reset_kept_pro"))
    if report.kept_cache:
        render_info(console, msg("cli.reset_kept_cache", count=report.kept_cache))
    if report.errors:
        render_error(
            console,
            msg("cli.reset_errors", count=len(report.errors), paths="; ".join(report.errors)),
        )
        return 1
    return 0


def _redirect_output_if_detached() -> None:
    """pythonw で起動されると stdout / stderr が ``None`` なので temp のファイルへ向ける。

    データ根の ``logs/`` は消す対象なので使わない。
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    import tempfile

    try:
        log = open(Path(tempfile.gettempdir()) / "evoref_reset.log", "a", encoding="utf-8")  # noqa: SIM115
    except OSError:
        return
    sys.stdout = log
    sys.stderr = log


if __name__ == "__main__":
    _redirect_output_if_detached()
    sys.exit(run_reset(sys.argv[1:]))
