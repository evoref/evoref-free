"""環境調整の実行 (c_16 §7.2.3 ``runner.py``) — CLI / API / 起動の 3 つの入口が共有する。

登録項目を依存順に実行し、項目ごとに ``cache/auto_tune.json`` へ保存する (途中で落ちても済んだ項目は
残る)。1 項目の失敗は他を止めない。環境起因の失敗 (一時的な起動失敗・空きの不足等) と例外は保存せず
``failed`` に理由を返す (次の実行でまた測る)。

測る前に必ず :func:`backend.free.core.tuning.gate.should_measure` を見る。環境移行の確認が済んでいない
(``pending`` / ``declined``) なら何も測らない。``force`` (手動の実行) は常に測り、``accepted`` を記録する。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.core.tuning.gate import GateState, load_gate, record_decision, should_measure
from backend.free.core.tuning.hardware import HardwareProbes, HardwareProfile, probe_hardware
from backend.free.core.tuning.items import (
    TuneContext,
    TuneItem,
    TuneOutcome,
    TuneRegistry,
    TuneSpec,
    applied_for,
    load_builtin_tuners,
)
from backend.free.core.tuning.store import (
    AutoTuneRecord,
    TunePaths,
    load_auto_tune,
    resolve_tune_paths,
    save_item,
)
from backend.log_config import get_logger

logger = get_logger("core.tuning.runner")

#: 進捗のコールバック ``(phase, current, total, key)``。phase は ``item`` / ``done`` / ``blocked``。
ProgressFn = Callable[[str, int, int, str], None]


@dataclass
class AutoTuneResult:
    """1 回の実行の結果。"""

    gate: GateState
    #: 項目を走らせたか (確認待ちで止めたなら偽)。
    measured: bool
    #: :func:`should_measure` の理由 (``forced`` / ``fresh`` / ``tune_pending`` 等)。
    reason: str
    #: この回に保存した項目。
    items: dict[str, TuneItem] = field(default_factory=dict)
    #: 保存しなかった失敗 (環境起因・例外) の理由。
    failed: dict[str, str] = field(default_factory=dict)
    #: 保存に失敗した項目 (書き込みの失敗。値は ``items`` に残る)。
    unsaved: list[str] = field(default_factory=list)
    record: AutoTuneRecord | None = None


def build_item(spec: TuneSpec, outcome: TuneOutcome, cfg: dict[str, Any], stamp: str) -> TuneItem:
    """保存する 1 項目 (環境起因の失敗は呼び手が除く)。``applied`` / 理由は config から決める。

    明示の値の項目は ``reason`` を ``manual`` にする (画面と CLI が「手動設定のため未適用」と出す)。
    このとき結果の理由に含まれる前提の印 (:attr:`TuneSpec.basis`) も落ちるので、後で config を
    ``auto`` に戻すと :func:`~backend.free.core.tuning.resolve.resolve_tuned` は見積り直す (安全側)。
    """
    applied, not_applied = applied_for(spec, cfg)
    if outcome.status == "ok":
        source = spec.method
        reason = outcome.reason if applied else not_applied
    else:
        source = outcome.status
        applied = False
        reason = outcome.reason or not_applied
    return TuneItem(
        key=spec.key, value=outcome.value, source=source, reason=reason,
        config_key=spec.config_key, applied=applied,
        requires_restart=spec.requires_restart, measured_at=stamp,
    )


def run(
    cfg: dict[str, Any],
    project_root: Path,
    *,
    only: Iterable[str] | None = None,
    force: bool = False,
    progress: ProgressFn | None = None,
    probes: HardwareProbes | None = None,
    registry: TuneRegistry | None = None,
    paths: TunePaths | None = None,
    now: Callable[[], str] | None = None,
) -> AutoTuneResult:
    """登録項目を依存順に実行して結果を保存する。

    ``only`` は項目の絞り込み (未知のキーは ``ValueError``)。``force`` は手動の実行で、確認状態に
    かかわらず測り、現在の PC に ``accepted`` を記録する。
    """
    from backend.free.rag.rerank_selftest import collect_pc_info
    from backend.utils import utc_now

    clock = now or utc_now
    specs = (registry or load_builtin_tuners()).ordered(only)
    resolved = paths or resolve_tune_paths(cfg, project_root)
    hardware: HardwareProfile = probe_hardware(probes)
    current = collect_pc_info(hardware.gpu_names)
    if force:
        record_decision(resolved.auto_tune, "accepted", current, now=clock())
    gate = load_gate(cfg, project_root, current, paths=resolved)
    ok, why = should_measure(gate, force=force)
    total = len(specs)
    if not ok:
        logger.info("Auto-tune not run: %s (changed axes: %s)", why, ", ".join(gate.changed_axes) or "-")
        if progress is not None:
            progress("blocked", 0, total, "")
        return AutoTuneResult(gate, False, why)

    result = AutoTuneResult(gate, True, why, record=load_auto_tune(resolved.auto_tune)[0])
    for index, spec in enumerate(specs):
        if progress is not None:
            progress("item", index, total, spec.key)
        ctx = TuneContext(cfg, project_root, force=force, hardware_fn=lambda: hardware)
        try:
            outcome = spec.run(ctx)
        except Exception as e:  # noqa: BLE001 - 1 項目の失敗で他の項目を止めない
            logger.warning("Auto-tune item %s failed: %s", spec.key, e, exc_info=True)
            result.failed[spec.key] = f"error:{type(e).__name__}"
            continue
        if outcome.status == "failed" and outcome.environmental:
            logger.info("Auto-tune item %s not saved (environmental: %s)", spec.key, outcome.reason)
            result.failed[spec.key] = outcome.reason or "environmental"
            continue
        stamp = clock()
        item = build_item(spec, outcome, cfg, stamp)
        result.items[spec.key] = item
        # 開始時に読んだレコードを上書きせず、書く直前に最新を読み直して 1 項目だけ足す (その間に
        # 起動スクリプトの見積り・予約・確認の答えが書いたものを消さない)。別の PC で測った項目は
        # 捨て、稼働中の画面の実行が残した予約はこの実行で果たした (``store.save_item``)
        saved = save_item(resolved.auto_tune, item, current, stamp)
        if saved is None:
            result.unsaved.append(spec.key)
        else:
            result.record = saved
    if progress is not None:
        progress("done", total, total, "")
    return result


__all__ = ["AutoTuneResult", "ProgressFn", "build_item", "run"]
