"""環境移行の確認 (c_16 §7.2.3 ``gate.py``) — 起動経路の全入口が同じ 1 実装を見る。

「環境移行」= 保存済みの結果 (``embed_placement`` / ``rerank_selftest`` / ``auto_tune``) の指紋が
現在の PC と違うこと。保存済みの結果が **どれも無い** (新規インストール) のは移行ではない (``fresh``)。

- :func:`evaluate_gate` — 現在の PC と保存済みの指紋群から :class:`GateState` を決める (純関数)
- :func:`should_measure` — 測り直すかの唯一の判定 (埋め込み・リランカー・runner が共通に使う)
- :func:`record_decision` — 確認の答え (``accepted`` / ``declined`` / ``pending`` / ``unchanged``) を残す
- :func:`restamp` — ``unchanged`` (同じ PC。ホスト名だけが変わった) のとき、測定値を保ったまま
  3 ファイルの指紋と PC を現在の値に書き直す。ホスト名以外の軸が変わっていれば拒否する

GPU 名は ``llama-server --list-devices`` (subprocess) でしか取れないので、backend は起動時に 1 回だけ
読んで持ち (``resolve.prime_backend_gpus``)、起動スクリプトと同じ GPU 名込みの完全な指紋で照合する
(GPU だけが変わった移行も ``/api/status`` に出す)。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.free.core.tuning.items import MANUAL, NOT_AUTO
from backend.free.core.tuning.store import (
    RECORD_LOCK,
    AutoTuneRecord,
    Decision,
    TunePaths,
    load_auto_tune,
    resolve_tune_paths,
    save_auto_tune,
    update_auto_tune,
)
from backend.free.rag.rerank_selftest import PcInfo
from backend.log_config import get_logger

logger = get_logger("core.tuning.gate")

GateKind = Literal["fresh", "unchanged_env", "migrated"]

#: PC の指紋の軸 (``PcInfo.digest`` の材料と同じ。増やさない、c_16 §7.2.3)。
AXES: tuple[str, ...] = ("hostname", "cpu", "logical_cores", "memory_gb", "gpus")

#: 起動スクリプトが測り直す理由のうち、確認を挟むもの。PC が変わった (``fingerprint_changed``) と
#: 結果が無い (``no_result``。新規なら ``fresh`` で従来どおり測る) だけ。モデルの変更・GPU の起動失敗の
#: 再試行・明示の設定の変更は同じ PC の出来事なので確認しない。
GATED_REASONS: frozenset[str] = frozenset({"fingerprint_changed", "no_result"})

#: 3 ファイルの名前 (``SavedPc.source``)。
SOURCE_EMBED = "embed_placement"
SOURCE_RERANK = "rerank_selftest"
SOURCE_AUTO_TUNE = "auto_tune"


@dataclass(frozen=True)
class SavedPc:
    """保存済みの結果 1 つの指紋と PC。"""

    source: str
    fingerprint: str
    pc: PcInfo


@dataclass(frozen=True)
class GateState:
    """確認状態。``changed_axes`` は食い違うファイルの PC と現在の PC の差の和 (:data:`AXES` の順)。"""

    state: GateKind
    decision: Decision
    changed_axes: tuple[str, ...] = ()
    #: 指紋が食い違う保存済みの結果 (``SavedPc.source``)。
    mismatched: tuple[str, ...] = ()
    #: 現在の PC の指紋。
    fingerprint: str = ""


# ── 純関数 ────────────────────────────────────────────────


def changed_axes(saved: PcInfo, current: PcInfo, *, compare_gpus: bool = True) -> list[str]:
    """2 つの PC の違う軸 (:data:`AXES` の順)。ホスト名は大小無視、GPU は並び順を見ない。"""
    out: list[str] = []
    if saved.hostname.strip().lower() != current.hostname.strip().lower():
        out.append("hostname")
    if saved.cpu.strip() != current.cpu.strip():
        out.append("cpu")
    if saved.logical_cores != current.logical_cores:
        out.append("logical_cores")
    if saved.memory_gb != current.memory_gb:
        out.append("memory_gb")
    if compare_gpus and sorted(g.strip() for g in saved.gpus) != sorted(g.strip() for g in current.gpus):
        out.append("gpus")
    return out


def _matches(saved: SavedPc, current: PcInfo, *, compare_gpus: bool) -> bool:
    """保存済みの結果が現在の PC のものか。GPU を比べられるなら指紋で、比べられなければ軸で見る。"""
    if compare_gpus:
        return saved.fingerprint == current.digest
    return not changed_axes(saved.pc, current, compare_gpus=False)


def recorded_decision(
    record: AutoTuneRecord | None, current: PcInfo, *, compare_gpus: bool = True,
) -> Decision | None:
    """現在の PC に対して記録済みの確認 (別の PC の確認は効かないので ``None``)。"""
    if record is None or not record.decided_fingerprint or record.decision == "auto":
        return None
    if compare_gpus:
        same = record.decided_fingerprint == current.digest
    else:
        same = not changed_axes(record.decided_pc, current, compare_gpus=False)
    return record.decision if same else None


def evaluate_gate(
    current: PcInfo,
    saved: Iterable[SavedPc | None],
    record: AutoTuneRecord | None = None,
    *,
    compare_gpus: bool = True,
) -> GateState:
    """確認状態を決める (純関数)。

    - 保存済みの結果 (指紋を持つもの) が 1 つも無い → ``fresh`` (新規。確認なしで測る)
    - 全て現在の PC と一致 → ``unchanged_env``
    - どれかが食い違う → ``migrated`` (確認が要る)

    ``decision`` は現在の PC に記録された確認があればそれ、無ければ ``migrated`` は ``pending``、
    それ以外は ``auto``。``auto_tune`` は項目を測ったレコードだけを指紋として数える (確認だけを
    書いたレコードは結果ではない) — 呼び手が ``saved`` に入れる (:func:`saved_pcs`)。
    """
    present = [s for s in saved if s is not None and s.fingerprint]
    digest = current.digest
    decided = recorded_decision(record, current, compare_gpus=compare_gpus)
    if not present:
        return GateState("fresh", decided or "auto", fingerprint=digest)
    mismatched = [s for s in present if not _matches(s, current, compare_gpus=compare_gpus)]
    if not mismatched:
        return GateState("unchanged_env", decided or "auto", fingerprint=digest)
    axes: set[str] = set()
    for s in mismatched:
        axes.update(changed_axes(s.pc, current, compare_gpus=compare_gpus))
    return GateState(
        "migrated", decided or "pending",
        changed_axes=tuple(a for a in AXES if a in axes),
        mismatched=tuple(s.source for s in mismatched),
        fingerprint=digest,
    )


def should_measure(gate: GateState, *, force: bool = False) -> tuple[bool, str]:
    """測るか、とその理由 — 起動経路 (埋め込み・リランカー・runner) はこれだけを見る。

    - ``force`` (手動の実行) → 常に測る (呼び手が ``accepted`` を記録する)
    - ``fresh`` / ``unchanged_env`` → 測る (従来どおり、各ファイルの要否に任せる)
    - ``migrated`` + ``accepted`` → 測る
    - ``migrated`` + ``pending`` / ``declined`` / ``unchanged`` → 測らない (``tune_<decision>``。
      未調整のまま縮退して起動する: 埋め込みは CPU、リランカーは無効)
    """
    if force:
        return True, "forced"
    if gate.state != "migrated":
        return True, gate.state
    if gate.decision == "accepted":
        return True, "accepted"
    decision = gate.decision if gate.decision != "auto" else "pending"
    return False, f"tune_{decision}"


def restamp_check(saved: Iterable[SavedPc | None], current: PcInfo) -> tuple[bool, str]:
    """``unchanged`` (指紋だけを書き直す) を許すか。食い違う結果の全てで、違う軸がホスト名だけのとき。

    指紋が違うのに PC の軸に差が見えない (記録が欠けている) ものは確かめられないので拒否する。
    """
    for s in saved:
        if s is None or not s.fingerprint or s.fingerprint == current.digest:
            continue
        axes = changed_axes(s.pc, current, compare_gpus=True)
        if not axes:
            return False, f"{s.source}: fingerprint differs but the recorded PC does not show how"
        others = [a for a in axes if a != "hostname"]
        if others:
            return False, f"{s.source}: {', '.join(others)} changed (only a hostname change can be kept)"
    return True, ""


# ── ファイル ──────────────────────────────────────────────


def relevant_sources(cfg: dict[str, Any]) -> tuple[bool, bool]:
    """(埋め込みの配置を見るか, リランカーを見るか)。使っていない機能の古い結果で移行と言わない。"""
    from backend.schemas.rag import rerank_mode_of

    embed = (cfg.get("embedding") or {}).get("gpu_layers") == "auto"
    # 既定 on でもモデル未設定なら使っていない
    rerank = rerank_mode_of(cfg) != "off" and bool((cfg.get("model_paths") or {}).get("rerank_model"))
    return embed, rerank


def saved_pcs(
    paths: TunePaths, *, embed: bool = True, rerank: bool = True,
) -> tuple[list[SavedPc], AutoTuneRecord | None]:
    """保存済みの 3 ファイルの指紋と、``auto_tune`` のレコード (確認状態の読み出し用)。読めないものは無いとみなす。"""
    from backend.free.rag.embed_placement import load_placement_result
    from backend.free.rag.rerank_selftest import load_selftest_result

    out: list[SavedPc] = []
    if embed:
        placed, _ = load_placement_result(paths.embed_placement)
        if placed is not None:
            out.append(SavedPc(SOURCE_EMBED, placed.fingerprint, placed.pc))
    if rerank:
        tested, _ = load_selftest_result(paths.rerank_selftest)
        if tested is not None:
            out.append(SavedPc(SOURCE_RERANK, tested.fingerprint, tested.pc))
    record, _ = load_auto_tune(paths.auto_tune)
    if record is not None and record.measured_at and record.fingerprint:
        out.append(SavedPc(SOURCE_AUTO_TUNE, record.fingerprint, record.pc))
    return out, record


def load_gate(
    cfg: dict[str, Any],
    project_root: Path,
    current: PcInfo,
    *,
    compare_gpus: bool = True,
    paths: TunePaths | None = None,
) -> GateState:
    """config と保存済みのファイルから確認状態を決める (起動スクリプトと backend の共通の入口)。"""
    embed, rerank = relevant_sources(cfg)
    saved, record = saved_pcs(paths or resolve_tune_paths(cfg, project_root), embed=embed, rerank=rerank)
    return evaluate_gate(current, saved, record, compare_gpus=compare_gpus)


def record_decision(path: Path, decision: Decision, current: PcInfo, *, now: str | None = None) -> bool:
    """確認の答えを現在の PC に対して残す (他の内容は保つ。最新を読み直して書く)。書けたら ``True``。"""
    from backend.utils import utc_now

    stamp = now or utc_now()

    def answer(record: AutoTuneRecord) -> None:
        record.decision = decision
        record.decided_at = stamp
        record.decided_fingerprint = current.digest
        record.decided_pc = current

    ok = update_auto_tune(path, answer) is not None
    if not ok:
        logger.warning("Failed to record the auto-tune decision %s to %s", decision, path)
    return ok


@dataclass(frozen=True)
class RestampResult:
    """:func:`restamp` の結果。拒否なら ``ok`` が偽で ``reason`` に理由 (英語)。"""

    ok: bool
    reason: str = ""
    rewritten: tuple[str, ...] = field(default_factory=tuple)


def restamp(
    paths: TunePaths, current: PcInfo, *, cfg: dict[str, Any] | None = None,
) -> RestampResult:
    """``unchanged``: 測らずに保存済みの結果の指紋と PC を現在の値へ書き直す (測定値は保つ)。

    ``current`` は GPU 名を含む完全な PC (``--list-devices`` を読んだ起動スクリプト / CLI の値) で
    なければならない (指紋に GPU 名が入るため)。ホスト名以外の軸が変わっていれば何も書かずに拒否し、
    書き直したら ``unchanged`` を記録する。``cfg`` を渡すと、確認状態 (:func:`load_gate`) と同じく
    使っている機能の結果だけを照合し・書き直す (使っていない機能の古い結果で拒否しない。その結果は
    この PC で測ったものではないので指紋も書き換えない)。
    """
    with RECORD_LOCK:
        return _restamp(paths, current, cfg)


def _restamp(paths: TunePaths, current: PcInfo, cfg: dict[str, Any] | None) -> RestampResult:
    from backend.free.rag import embed_placement as ep
    from backend.free.rag import rerank_selftest as st

    embed, rerank = relevant_sources(cfg) if cfg is not None else (True, True)
    saved, record = saved_pcs(paths, embed=embed, rerank=rerank)
    ok, why = restamp_check(saved, current)
    if not ok:
        return RestampResult(False, why)
    digest = current.digest
    rewritten: list[str] = []
    placed, _ = ep.load_placement_result(paths.embed_placement) if embed else (None, "")
    if placed is not None and placed.fingerprint != digest:
        placed.fingerprint, placed.pc = digest, current
        if not ep.save_placement_result(paths.embed_placement, placed):
            return RestampResult(False, f"failed to write {paths.embed_placement}", tuple(rewritten))
        rewritten.append(SOURCE_EMBED)
    tested, _ = st.load_selftest_result(paths.rerank_selftest) if rerank else (None, "")
    if tested is not None and tested.fingerprint != digest:
        tested.fingerprint, tested.pc = digest, current
        if not st.save_selftest_result(paths.rerank_selftest, tested):
            return RestampResult(False, f"failed to write {paths.rerank_selftest}", tuple(rewritten))
        rewritten.append(SOURCE_RERANK)
    if record is not None and record.measured_at and record.fingerprint and record.fingerprint != digest:
        record.fingerprint, record.pc = digest, current
        if not save_auto_tune(paths.auto_tune, record):
            return RestampResult(False, f"failed to write {paths.auto_tune}", tuple(rewritten))
        rewritten.append(SOURCE_AUTO_TUNE)
    if not record_decision(paths.auto_tune, "unchanged", current):
        return RestampResult(False, f"failed to write {paths.auto_tune}", tuple(rewritten))
    logger.info("Auto-tune restamp: kept the measurements of %s for this PC", ", ".join(rewritten) or "nothing")
    return RestampResult(True, "", tuple(rewritten))


def answer_migration(
    paths: TunePaths, decision: Decision, current: PcInfo, *, cfg: dict[str, Any] | None = None,
) -> RestampResult:
    """確認の答えを反映する入口 (CLI / API が使う)。``unchanged`` は :func:`restamp`、他は記録だけ。"""
    if decision == "unchanged":
        return restamp(paths, current, cfg=cfg)
    if decision not in ("accepted", "declined", "pending"):
        return RestampResult(False, f"unknown decision: {decision}")
    if not record_decision(paths.auto_tune, decision, current):
        return RestampResult(False, f"failed to write {paths.auto_tune}")
    return RestampResult(True)


# ── backend の読み取り (``/api/status.auto_tune``) ─────────────


@dataclass(frozen=True)
class AutoTuneStatus:
    """``/api/status`` に出す環境調整の状態 (起動時に 1 回決める)。

    ``state``: ``fresh`` (結果なし) / ``ok`` (同じ PC) / ``pending`` / ``declined`` / ``accepted`` /
    ``unchanged`` (移行の確認状態) / ``unknown`` (読めない)。
    """

    state: str = "unknown"
    reason: str = ""
    changed_axes: tuple[str, ...] = ()
    items_summary: dict[str, int] = field(default_factory=dict)
    measured_at: str | None = None


def summarize_items(record: AutoTuneRecord | None) -> dict[str, int]:
    """項目の件数 (``total`` / ``applied`` / ``manual`` / ``failed``)。"""
    items = list(record.items.values()) if record is not None else []
    return {
        "total": len(items),
        "applied": sum(1 for i in items if i.applied),
        "manual": sum(1 for i in items if not i.applied and i.reason in (MANUAL, NOT_AUTO)),
        "failed": sum(1 for i in items if i.source == "failed"),
    }


def resolve_auto_tune_status(gate: GateState, record: AutoTuneRecord | None) -> AutoTuneStatus:
    """確認状態とレコードから ``/api/status`` の値を決める (純関数)。"""
    common = {
        "items_summary": summarize_items(record),
        "measured_at": (record.measured_at or None) if record is not None else None,
    }
    if gate.state == "fresh":
        return AutoTuneStatus("fresh", "no_result", **common)
    if gate.state == "unchanged_env":
        return AutoTuneStatus("ok", "", **common)
    decision = gate.decision if gate.decision != "auto" else "pending"
    return AutoTuneStatus(decision, "pc_changed", changed_axes=gate.changed_axes, **common)


def load_auto_tune_status(
    cfg: dict[str, Any], project_root: Path, *, paths: TunePaths | None = None, current: PcInfo | None = None,
) -> AutoTuneStatus:
    """backend の起動時 (と実行・確認の答えの後) に読む。読めなければ ``unknown`` (起動は止めない)。

    PC は ``current`` (無ければ backend が起動時に読んだ GPU 名込みの PC、``resolve.backend_pc``) で、
    起動スクリプトと同じ完全な指紋で照合する — GPU だけが変わった移行も ``pending`` として出し、
    ``changed_axes`` に ``gpus`` を載せる (「変更なし」はホスト名だけのときしか選べない)。
    """
    from backend.free.core.tuning.resolve import backend_pc

    try:
        resolved = paths or resolve_tune_paths(cfg, project_root)
        embed, rerank = relevant_sources(cfg)
        saved, record = saved_pcs(resolved, embed=embed, rerank=rerank)
        gate = evaluate_gate(current or backend_pc(), saved, record, compare_gpus=True)
    except Exception as e:  # noqa: BLE001 - 状態の表示だけなので起動を止めない
        logger.warning("Auto-tune state unreadable: %s", e)
        return AutoTuneStatus("unknown", f"unreadable: {type(e).__name__}")
    return resolve_auto_tune_status(gate, record)


__all__ = [
    "AXES",
    "GATED_REASONS",
    "SOURCE_AUTO_TUNE",
    "SOURCE_EMBED",
    "SOURCE_RERANK",
    "AutoTuneStatus",
    "GateKind",
    "GateState",
    "RestampResult",
    "SavedPc",
    "answer_migration",
    "changed_axes",
    "evaluate_gate",
    "load_auto_tune_status",
    "load_gate",
    "record_decision",
    "recorded_decision",
    "relevant_sources",
    "resolve_auto_tune_status",
    "restamp",
    "restamp_check",
    "saved_pcs",
    "should_measure",
    "summarize_items",
]
