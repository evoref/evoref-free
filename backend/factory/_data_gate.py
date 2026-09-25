"""起動ゲート (c_05 §0.9)。pillar 配線の前に呼ぶ。

必須段 (配線前、50ms 未満):

1. ``data_root`` の置き場を判定する (ネットワーク / OneDrive → 拒否、FAT / exFAT → 警告)
2. 世代フォルダ ``g<N>/`` を調べる (c_05 §0.2)。自分の世代が無く新しい世代だけあれば
   拒否、古い世代だけあれば readonly (移行は G2 の ``evoref data migrate``)。自分の世代と
   新しい世代が両方あれば警告 (ここでの変更は新しい世代へ持ち越されない)
3. 単一書き手ロックを取る (取れなければ拒否)
4. 世代印を読む。読めなければ readonly
5. ``pending`` に自分の読む形式があれば拒否 (G1 は移行器を持たない)
6. コードとの版比較。``newer`` (と、移行器の無い ``older``) があれば readonly
7. readonly でなければ世代印を現行へ書き直す (新しい形式・書き手・最後のエディション)
8. readonly でなければ取り残しの ``*.tmp`` (kill された原子的書き込みの一時ファイル) を
   ``store/`` から消す。ロックの下なので年齢判定は要らない (書いている他プロセスは無い)

世代印が無いまま ``store/`` に中身があるのは不整合なので readonly にする (黙って
新規扱いにすると、既存データの版を確かめないまま書き足すことになる)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config import PathResolver
from backend.data_location import DataLocation, inspect_location
from backend.data_root import DATA_GENERATION, GENERATION_DIRNAME, generation_dirs, store_root
from backend.io.format_registry import FormatRegistry
from backend.io.generation_seal import (
    SEAL_FILENAME,
    FormatState,
    check_seal,
    updated_seal,
    write_seal,
)
from backend.io.writer_lock import WriterLock, WriterLockHeld, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("factory.data_gate")

class DataGateRefused(RuntimeError):
    """起動を止める理由 (``code`` は UI / CLI の案内の鍵)。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(slots=True)
class DataGateResult:
    data_root: Path
    location: DataLocation
    lock: WriterLock
    readonly: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    states: dict[str, FormatState] = field(default_factory=dict)
    #: 世代印の ``last_edition`` と今のエディションが違えば (前回, 今回)。UI へ 1 回出す。
    edition_switched_from: str | None = None
    edition_switched_to: str | None = None
    #: 起動時に消した取り残しの一時ファイルの数。
    stale_tmp_removed: int = 0
    edition: str = ""
    app_version: str = ""
    #: 実行中エディションが読む形式のコード側の版 (起動ログに出す)。
    format_versions: dict[str, int] = field(default_factory=dict)


def check_generation_dirs(data_root: Path) -> tuple[list[str], list[str]]:
    """世代フォルダを調べる。戻り値は (readonly の理由, 警告)。起動を止めるなら例外。"""
    gens = generation_dirs(data_root)
    newer = sorted(g for g in gens if g > DATA_GENERATION)
    older = sorted(g for g in gens if g < DATA_GENERATION)
    if DATA_GENERATION in gens:
        warnings = [
            f"a newer data generation exists (g{g}); changes made here are not carried "
            f"forward to it" for g in newer
        ]
        return [], warnings
    if newer:
        raise DataGateRefused(
            "data_generation_newer",
            f"{data_root} holds only newer data generation(s) "
            f"({', '.join(f'g{g}' for g in newer)}); this version reads {GENERATION_DIRNAME}",
        )
    if older:
        return [
            f"data generation {GENERATION_DIRNAME} is missing and only older generation(s) "
            f"exist ({', '.join(f'g{g}' for g in older)}); they need a migration this "
            "version does not have",
        ], []
    return [], []



def run_data_gate(
    data_root: Path,
    registry: FormatRegistry,
    *,
    edition: str,
    app_version: str,
    started_at: str = "",
    allow_unsafe: bool = False,
) -> DataGateResult:
    """起動ゲートを通す。止めるべきなら :class:`DataGateRefused`。"""
    location = inspect_location(data_root)
    warnings: list[str] = []
    if location.unsafe:
        where = "a network drive" if location.remote else "OneDrive"
        if not allow_unsafe:
            raise DataGateRefused(
                "data_root_unsafe",
                f"data_root {data_root} is on {where}; move it with --data-root or start with "
                "--allow-unsafe-data-root",
            )
        warnings.append(f"data_root is on {where} (allowed by --allow-unsafe-data-root)")
    if location.weak_fs:
        warnings.append(f"data_root is on {location.fs_type}; rename/fsync ordering is weak")
    generation_reasons, generation_warnings = check_generation_dirs(data_root)
    warnings += generation_warnings

    store_dir = store_root(data_root)
    try:
        lock = acquire_writer_lock(store_dir, started_at=started_at)
    except WriterLockHeld as e:
        raise DataGateRefused("data_root_locked", str(e)) from e

    result = DataGateResult(
        data_root=data_root, location=location, lock=lock, warnings=warnings,
        edition=edition, app_version=app_version,
        format_versions={s.format_id: s.version for s in registry.read_by(edition)},  # type: ignore[arg-type]
    )
    if generation_reasons:
        result.readonly = True
        result.reasons.extend(generation_reasons)
    try:
        _check_generation(result, store_dir, registry, edition=edition, app_version=app_version)
    except BaseException:
        lock.release()
        raise
    if not result.readonly:
        # Free は Pro のデータ (store/pro/) に入らない (c_05 §0.4.2)。
        pro_dir = None if edition == "pro" else PathResolver.layout_path(data_root, "pro_dir")
        result.stale_tmp_removed = sweep_stale_tmp(store_dir, skip=pro_dir)
    return result


def sweep_stale_tmp(store_dir: Path, *, skip: Path | None = None) -> int:
    """``store/`` 配下の取り残しの ``*.tmp`` を消す (書き手ロックの下で呼ぶ)。

    ``AtomicWriter`` の一時ファイルは宛先と同じディレクトリに ``<name>.<rand>.tmp``
    で作られ、kill されると残る。消せなかったものは数えずに次の起動へ回す。
    ``skip`` のディレクトリには入らない。
    """
    import os

    removed = 0
    stack = [store_dir]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if skip is not None and Path(entry.path) == skip:
                    continue
                stack.append(Path(entry.path))
            elif entry.name.endswith(".tmp"):
                try:
                    os.unlink(entry.path)
                    removed += 1
                except OSError:
                    continue
    return removed


def _check_generation(
    result: DataGateResult,
    store_dir: Path,
    registry: FormatRegistry,
    *,
    edition: str,
    app_version: str,
) -> None:
    seal_path = store_dir / SEAL_FILENAME
    check = check_seal(store_dir, registry, edition)
    if check.unreadable is None and check.pending:
        raise DataGateRefused(
            "migration_pending",
            f"a data migration is in progress for {', '.join(check.pending)} and this version "
            "cannot finish it",
        )
    reasons = check.readonly_reasons(store_dir)
    if reasons:
        result.readonly = True
        result.reasons.extend(reasons)
    if check.unreadable is not None or check.missing_with_data:
        return
    seal = check.seal
    result.states = check.states
    if seal is not None and seal.last_edition and seal.last_edition != edition:
        result.edition_switched_from = seal.last_edition
        result.edition_switched_to = edition
    if not result.readonly:
        write_seal(seal_path, updated_seal(seal, registry, edition=edition, app_version=app_version))


def open_data_root(state: Any, project_root: Path) -> DataGateResult:
    """lifespan から呼ぶ入口。ゲートを通し、結果を ``state`` へ載せる。

    readonly なら書き込み層の最後の砦を有効にし、``state.data_readonly_reason`` を
    立てる (学習と sleep-time はこれを見て止まる)。止めるべきなら
    :class:`DataGateRefused` を送出する (起動を止める)。
    """
    import os

    from backend.data_root import ALLOW_UNSAFE_ENV, resolve_data_root
    from backend.formats import load_all_formats
    from backend.io.format_registry import FORMATS
    from backend.io.readonly import enable_readonly
    from backend.io.versioned import producer_edition
    from backend.utils import utc_now
    from backend.version import get_runtime_version

    data_root = resolve_data_root(root=project_root)
    load_all_formats()
    try:
        result = run_data_gate(
            data_root,
            FORMATS,
            edition=producer_edition(),
            app_version=str(get_runtime_version()),
            started_at=utc_now(),
            allow_unsafe=os.environ.get(ALLOW_UNSAFE_ENV) == "1",
        )
    except DataGateRefused as e:
        logger.error("Startup refused by the data gate (%s): %s", e.code, e)
        raise
    state.data_gate = result
    if result.readonly:
        reason = "; ".join(result.reasons)
        state.data_readonly_reason = reason
        enable_readonly(store_root(data_root), reason)
    return result


def _warn_edition_downgrade(previous: str) -> None:
    """Pro → Free の切替を知らせる (世代印の ``last_edition`` と今のエディションで判定)。

    G0 は ``edition_state.json`` を別に持っていたが、G1 は世代印が前回の
    エディションを覚えている (c_05 §0.4.6)。
    """
    from backend.i18n_helper import msg
    from backend.io.versioned import producer_edition

    current = producer_edition()
    if previous != "pro" or current != "free":
        return
    logger.warning("Edition downgrade detected: %s -> %s", previous, current)
    logger.warning(msg("warning.edition.downgrade", old=previous.upper(), new=current.upper()))
    logger.warning(msg("warning.edition.lora_unavailable"))


def report_data_gate(state: Any) -> None:
    """ゲートの結果をログへ出す (ログの設定が済んだ配線の後に呼ぶ)。"""
    result: DataGateResult | None = state.data_gate
    if result is None:
        return
    for warning in result.warnings:
        logger.warning("Data root %s: %s", result.data_root, warning)
    if result.edition_switched_from:
        logger.info("Edition switched from %s to %s", result.edition_switched_from,
                    result.edition_switched_to)
        _warn_edition_downgrade(result.edition_switched_from)
    if result.stale_tmp_removed:
        logger.info("Removed %d stale temporary file(s) under %s", result.stale_tmp_removed,
                    store_root(result.data_root))
    if result.readonly:
        logger.error("Data root %s is read-only: %s", result.data_root, "; ".join(result.reasons))
    logger.info(
        "Data root %s: app_version=%s edition=%s %s formats=%s fs=%s readonly=%s",
        result.data_root, result.app_version, result.edition, _generations_text(result.edition),
        _format_versions_text(result.format_versions), result.location.fs_type or "unknown",
        result.readonly,
    )


def _generations_text(edition: str) -> str:
    """``generation=N`` (Pro なら `` pro_generation=M`` も)。"""
    from backend.io.format_lock import current_generations

    gens = current_generations()
    text = f"generation={gens['generation']}"
    if edition == "pro" and "pro_generation" in gens:
        text += f" pro_generation={gens['pro_generation']}"
    return text


def _format_versions_text(versions: dict[str, int]) -> str:
    """形式の版を 1 語に畳む: 揃っていれば ``all v1 (N)``、違う形式だけ名前を並べる。"""
    if not versions:
        return "none"
    base, count = Counter(versions.values()).most_common(1)[0]
    if count == len(versions):
        return f"all v{base} ({count})"
    others = ",".join(f"{fid}=v{v}" for fid, v in sorted(versions.items()) if v != base)
    return f"v{base} ({count})+{others}"


__all__ = [
    "DataGateRefused",
    "DataGateResult",
    "check_generation_dirs",
    "open_data_root",
    "report_data_gate",
    "run_data_gate",
    "sweep_stale_tmp",
]
