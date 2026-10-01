"""稼働中の backend から環境調整を走らせる実行状態 (c_16 §7.2.3 の API の入口、c_06 §2.7)。

``AppState.auto_tune_runner`` が 1 つ持つ。実行本体 (:func:`~backend.free.core.tuning.runner.run`) は
ブロッキングなのでスレッドで走らせ、状態 (``idle`` / ``running`` / ``done`` / ``failed``)・進捗・
最後の結果をここに持つ。FastAPI には依存しない (router は ``backend/free/api/system/auto_tune.py``)。

稼働中に走らせてよいのは ``TuneSpec.safe_while_running`` の項目だけ (一時ポートで測るもの)。
それ以外は実行せず、次の 2 つに分けて結果に載せる (c_16 §7.2.3):

- **予約** (``reason="scheduled"``) — llama-server を起こさずに決まる base 向けの見積り (ctx / ngl /
  VRAM 予算 …)。稼働中は自分の base が VRAM / RAM を占めていて空きが過小に見えるので、
  ``cache/auto_tune.json`` の ``recompute`` に載せ、次の起動 / 画面の再起動 (base が止まった時点) で測る
- **要停止** (``reason="requires_stop"``) — 本番のポートでサーバを起こして測る項目 (リランカー)。停止中の CLI だけ
- **明示指名のみ** (``reason="requires_explicit"``) — 一時ポートでサーバを起こして測る項目 (埋め込みの配置)。
  稼働中はチャットのアイドルを待たずに CPU / GPU の一時サーバを起こすので、``only`` で名指ししたときだけ走らせる
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.core.tuning.gate import answer_migration, load_gate
from backend.free.core.tuning.hardware import HardwareProbes, probe_hardware
from backend.free.core.tuning.items import TuneItem, TuneRegistry, TuneSpec, load_builtin_tuners
from backend.free.core.tuning.runner import AutoTuneResult, run
from backend.free.core.tuning.resolve import backend_pc
from backend.free.core.tuning.store import TunePaths, load_auto_tune, schedule_recompute
from backend.free.rag.rerank_selftest import PcInfo, collect_pc_info
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

logger = get_logger("core.tuning.service")

#: 稼働中に実行できない項目の理由 (結果の ``reason``)。
REQUIRES_STOP = "requires_stop"
#: 予約して次の起動 / 再起動で再計算する項目の理由 (結果の ``reason``)。
SCHEDULED = "scheduled"
#: 稼働中は ``only`` で名指ししたときだけ走らせる項目の理由 (結果の ``reason``)。
REQUIRES_EXPLICIT = "requires_explicit"

#: 確認の答えのうち API が受けるもの (``auto`` / ``pending`` は答えではない)。
ANSWERS: tuple[str, ...] = ("accepted", "declined", "unchanged")


class AutoTuneBusy(Exception):
    """実行中に別の実行 / 確認の答えを受けた。"""


@dataclass(frozen=True)
class RunPlan:
    """``only`` を稼働中に実行できる項目 / 予約する項目 / 停止中の CLI 限定の項目に分けたもの。"""

    runnable: tuple[str, ...]
    blocked: tuple[str, ...]
    scheduled: tuple[str, ...] = ()
    #: 一時サーバを起こす項目のうち ``only`` で名指しされなかったもの (走らせない)。
    explicit_only: tuple[str, ...] = ()


def _placeholder(spec_key: str, config_key: str, requires_restart: bool, *, source: str, reason: str) -> TuneItem:
    """保存しない項目 (実行しなかった / 環境起因で測れなかった) の表示用。"""
    return TuneItem(
        key=spec_key, source=source, reason=reason, config_key=config_key,
        applied=False, requires_restart=requires_restart,
    )


def item_to_dict(item: TuneItem) -> dict[str, Any]:
    """API の ``items`` の 1 要素 (契約のキー集合)。"""
    return {
        "key": item.key, "value": item.value, "source": item.source, "reason": item.reason,
        "config_key": item.config_key, "applied": item.applied,
        "requires_restart": item.requires_restart, "measured_at": item.measured_at,
    }


@dataclass
class AutoTuneService:
    """環境調整の実行状態。``run_fn`` / ``probes`` / ``registry`` はテストで差し替える。"""

    run_fn: Callable[..., AutoTuneResult] = run
    probes: HardwareProbes | None = None
    registry: TuneRegistry | None = None

    state: str = "idle"
    progress: dict[str, Any] | None = None
    error: str | None = None
    restart_required: bool = False
    #: 最後の実行で保存しなかった項目 (``requires_stop`` / 環境起因の失敗)。
    extra_items: dict[str, TuneItem] = field(default_factory=dict)
    #: ``--list-devices`` まで読んだ完全な PC (実行 / 確認の答えの後に持つ)。
    pc: PcInfo | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)

    def _registry(self) -> TuneRegistry:
        return self.registry or load_builtin_tuners()

    def plan(self, only: Iterable[str] | None) -> RunPlan:
        """``only`` (``None`` は全項目) を依存順に並べ、稼働中に走らせてよいかで分ける。

        稼働中に一時サーバを起こして測る項目 (``safe_while_running`` かつ ``needs_servers``) は、
        ``only`` で名指ししたときだけ走らせる (全項目の実行ではチャットと同時に判別を始めない)。
        未知のキーは ``ValueError`` (呼び手が 422 にする)。
        """
        specs = self._registry().ordered(only)
        named = set(only) if only is not None else set()

        def explicit(s: TuneSpec) -> bool:
            return s.safe_while_running and s.needs_servers and s.key not in named

        return RunPlan(
            runnable=tuple(s.key for s in specs if s.safe_while_running and not explicit(s)),
            blocked=tuple(s.key for s in specs if not s.safe_while_running and s.needs_servers),
            scheduled=tuple(s.key for s in specs if not s.safe_while_running and not s.needs_servers),
            explicit_only=tuple(s.key for s in specs if explicit(s)),
        )

    # ── 実行 ──────────────────────────────────────────

    async def start(
        self,
        cfg: dict[str, Any],
        project_root: Path,
        paths: TunePaths,
        *,
        only: Iterable[str] | None = None,
        force: bool = False,
        on_finished: Callable[[], None] | None = None,
    ) -> RunPlan:
        """裏で実行を始める (実行中は :class:`AutoTuneBusy`、未知の ``only`` は ``ValueError``)。"""
        if self.lock.locked():
            raise AutoTuneBusy
        plan = self.plan(list(only) if only is not None else None)
        await self.lock.acquire()  # 空いていれば待たずに取れる (check と取得の間に await は挟まない)
        self.state, self.error, self.restart_required = "running", None, False
        self.progress = {"phase": "start", "current": 0, "total": len(plan.runnable)}
        registry = self._registry()
        self.extra_items = {
            key: _placeholder(
                key, registry.get(key).config_key, registry.get(key).requires_restart,  # type: ignore[union-attr]
                source="skipped", reason=reason,
            )
            for keys, reason in (
                (plan.blocked, REQUIRES_STOP), (plan.scheduled, SCHEDULED), (plan.explicit_only, REQUIRES_EXPLICIT),
            )
            for key in keys
        }
        self._task = asyncio.get_running_loop().create_task(
            self._execute(cfg, project_root, paths, plan, force, on_finished),
        )
        return plan

    def _on_progress(self, phase: str, current: int, total: int, key: str) -> None:
        """runner の進捗 (ワーカースレッドから呼ばれる)。項目の実行中は ``phase`` が項目のキー。"""
        self.progress = {"phase": key if phase == "item" and key else phase, "current": current, "total": total}

    def _run_blocking(
        self, cfg: dict[str, Any], project_root: Path, paths: TunePaths, plan: RunPlan, force: bool,
    ) -> tuple[AutoTuneResult, dict[str, Any]]:
        before = load_auto_tune(paths.auto_tune)[0]
        previous = {k: v.value for k, v in before.items.items()} if before is not None else {}
        result = self.run_fn(
            cfg, project_root, only=list(plan.runnable), force=force, progress=self._on_progress,
            probes=self.probes, registry=self._registry(), paths=paths,
        )
        return result, previous

    async def _execute(
        self,
        cfg: dict[str, Any],
        project_root: Path,
        paths: TunePaths,
        plan: RunPlan,
        force: bool,
        on_finished: Callable[[], None] | None,
    ) -> None:
        try:
            if plan.scheduled:
                loop = asyncio.get_running_loop()
                if await run_in_executor_with_context(
                    loop, None, schedule_recompute, paths.auto_tune, list(plan.scheduled),
                ):
                    logger.info("Auto-tune: scheduled %s for the next start", ", ".join(plan.scheduled))
            if plan.runnable:
                loop = asyncio.get_running_loop()
                result, previous = await run_in_executor_with_context(
                    loop, None, self._run_blocking, cfg, project_root, paths, plan, force,
                )
                registry = self._registry()
                for key, reason in result.failed.items():
                    spec = registry.get(key)
                    if spec is not None:
                        self.extra_items[key] = _placeholder(
                            key, spec.config_key, spec.requires_restart, source="failed", reason=reason,
                        )
                if result.unsaved:
                    logger.warning("Auto-tune results could not be saved: %s", ", ".join(result.unsaved))
                self.restart_required = any(
                    item.applied and item.requires_restart and previous.get(key) != item.value
                    for key, item in result.items.items()
                )
                if result.measured:  # 確認待ちで測らなかった回は runner の "blocked" を残す
                    self.progress = {"phase": "done", "current": len(plan.runnable), "total": len(plan.runnable)}
            else:
                self.progress = {"phase": "done", "current": 0, "total": 0}
            self.state = "done"
        except Exception as e:  # noqa: BLE001 - 失敗は state=failed で画面に返す (サーバは止めない)
            logger.warning("Auto-tune run failed: %s", e, exc_info=True)
            self.state, self.error = "failed", f"auto-tune failed: {type(e).__name__}"
        finally:
            self.lock.release()
            if on_finished is not None:
                try:
                    on_finished()
                except Exception as e:  # noqa: BLE001 - 状態表示の更新に失敗しても結果は返す
                    logger.warning("Auto-tune status refresh failed: %s", e)

    # ── 確認の答え ────────────────────────────────────

    def _current_pc(self) -> PcInfo:
        """GPU 名込みの現在の PC (``--list-devices`` を読む。ブロッキング)。"""
        pc = collect_pc_info(probe_hardware(self.probes).gpu_names)
        self.pc = pc
        return pc

    async def answer(
        self, paths: TunePaths, decision: str, *, cfg: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """確認の答えを残す。``(ok, 拒否の理由)``。実行中は :class:`AutoTuneBusy`。

        ``unchanged`` は GPU 名込みの PC が要る (指紋に入る) ので、スレッドで ``--list-devices`` を読む。
        ``cfg`` を渡すと ``unchanged`` は使っている機能の結果だけを照合する (確認状態と同じ範囲)。
        """
        if self.lock.locked():
            raise AutoTuneBusy

        def work() -> tuple[bool, str]:
            outcome = answer_migration(paths, decision, self._current_pc(), cfg=cfg)  # type: ignore[arg-type]
            return outcome.ok, outcome.reason

        async with self.lock:
            return await run_in_executor_with_context(asyncio.get_running_loop(), None, work)

    # ── 読み出し ──────────────────────────────────────

    def snapshot(self, cfg: dict[str, Any], project_root: Path, paths: TunePaths) -> dict[str, Any]:
        """``GET /api/system/auto-tune`` の本文 (保存済みの結果 + 実行状態。測らない)。"""
        record = load_auto_tune(paths.auto_tune)[0]
        # GPU 名込みの PC (確認の答えで読み直したもの、無ければ backend が起動時に読んだもの) で、
        # 起動スクリプト・``/api/status`` と同じ完全な指紋で照合する
        pc = self.pc if self.pc is not None else backend_pc()
        gate = load_gate(cfg, project_root, pc, paths=paths)
        saved = dict(record.items) if record is not None else {}
        scheduled = list(record.recompute) if record is not None else []
        for key in scheduled:  # 予約中は保存値 (前の提案) を見せつつ、状態は「再起動時に再計算」
            spec = self._registry().get(key)
            if spec is not None:
                item = _placeholder(key, spec.config_key, spec.requires_restart, source="skipped", reason=SCHEDULED)
                if key in saved:
                    item.value, item.measured_at = saved[key].value, saved[key].measured_at
                saved[key] = item
        merged: dict[str, TuneItem] = {}
        for key in self._registry().keys():
            if key in saved:
                merged[key] = saved[key]
            elif key in self.extra_items:
                merged[key] = self.extra_items[key]
        for key, item in {**self.extra_items, **saved}.items():
            merged.setdefault(key, item)
        return {
            "state": self.state,
            "progress": dict(self.progress) if self.progress is not None else None,
            "pc": {
                "hostname": pc.hostname, "cpu": pc.cpu, "logical_cores": pc.logical_cores,
                "memory_gb": pc.memory_gb, "gpus": list(pc.gpus),
            },
            "items": [item_to_dict(i) for i in merged.values()],
            "decision": gate.decision,
            "changed_axes": list(gate.changed_axes),
            "error": self.error,
            # 予約は再起動で果たす (状態が残っている間は再起動が要る)
            "restart_required": self.restart_required or bool(scheduled),
        }


__all__ = ["ANSWERS", "REQUIRES_EXPLICIT", "REQUIRES_STOP", "SCHEDULED", "AutoTuneBusy", "AutoTuneService", "RunPlan", "item_to_dict"]
