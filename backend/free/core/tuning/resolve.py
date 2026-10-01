"""調整結果を起動時に反映する (c_16 §7.2.3「config が ``auto`` の項目だけ起動時に反映する」)。

全項目に共通の規則 (:func:`resolve_tuned`):

1. config の値が自動を意味しない (明示値) → その値をそのまま返す (``manual``。保存済みの提案値は使わない)
2. 保存済みの項目があり、指紋が今の PC と一致し、前提の印 (:attr:`TuneSpec.basis`、例: base モデル) も
   一致する → 保存値 (``saved``)。指紋の違う保存値 (環境移行で未確認) は使わない
3. 環境移行の確認待ち (``pending`` / ``declined``) → 保守側の値 (``fallback``)。ただし起動スクリプトは
   ``base_model.resolve_or_provisional`` を通し、ctx / ngl / VRAM 予算は **保存しない一時の見積り** を使う。
   確認待ちで「測らない」のはサーバを起こす測定 (埋め込みの配置 / リランカー) で、GGUF と
   ``--list-devices`` を読むだけの見積りは安全 (保守側の固定 ngl 999 は小さい VRAM で OOM する)
4. llama-server を起こさずに決まる項目 (``needs_servers`` が偽) → その場で見積もる (``estimated``)。
   見積りは **保存する** — backend は測らずに保存値を読むので、起動スクリプトがその場で決めた値を
   残さないと、backend (例: 文脈長からのトークン予算) と llama-server の ``-c`` が食い違う
5. それもできなければ保守側の値

稼働中の画面の実行は、base に効く項目を測らずに **予約** (``AutoTuneRecord.recompute``) する。予約された
項目は保存値があっても、base が止まっているとき (起動前 / 再起動の直前、``can_recompute``) に 2 の代わりに
4 で測り直し、保存して予約から外す (再計算に失敗したら前の保存値のまま、予約は残す)。

backend (``allow_decide=False``) は測らない: 起動時に 1 回だけ読んだ GPU 名 (:func:`prime_backend_gpus`)
込みの PC の指紋で保存値を照合し (起動スクリプト・``gate.load_auto_tune_status`` と同じ完全な指紋)、
使える保存値が無ければ保守側の値を返す。起動スクリプトがその場の見積りを保存できなかったときも、
backend は保守側 (小さい ctx) に倒れるだけで llama-server の窓を超える側には倒れない。

``allow_decide`` を省略した呼び出し (起動スクリプトの ``_tuned`` 等) は :func:`decide_scope` の指定に
従う (既定は決めてよい)。backend のプロセス内で起動コマンドを組む経路 (``server_control`` / モード切替)
は、base を止めた後の組み直しだけ ``decide_scope(True, hardware=...)`` で見積りを許し、それ以外は
``decide_scope(False)`` で保存値を読むだけにする。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.free.core.tuning.hardware import HardwareProbes, HardwareProfile, memory_squeezed, probe_hardware
from backend.free.core.tuning.items import (
    MANUAL,
    TuneContext,
    TuneItem,
    TuneOutcome,
    TuneRegistry,
    TuneSpec,
    config_value,
    is_auto,
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

logger = get_logger("core.tuning.resolve")

# ── backend の PC (GPU 名込み) ─────────────────────────────

#: backend が起動時に読む ``--list-devices`` のタイムアウト (秒)。超えたら GPU なし (起動スクリプトの
#: 失敗時と同じ扱い) で照合する。
BACKEND_LIST_DEVICES_TIMEOUT_SEC = 15.0

#: backend プロセスの GPU 名 (起動時に 1 回だけ読む)。``None`` は未読 (最初の参照で読む)。
_BACKEND_GPUS: tuple[str, ...] | None = None


def _backend_list_devices() -> str:
    from scripts.launch_llama import _list_llama_devices

    return _list_llama_devices(timeout=BACKEND_LIST_DEVICES_TIMEOUT_SEC)


def probe_backend_gpus() -> tuple[str, ...]:
    """``--list-devices`` から GPU 名を読む (ブロッキング)。失敗・空・縮退は ``()`` (起動スクリプトと同じ)。"""
    try:
        hw = probe_hardware(HardwareProbes(list_devices=_backend_list_devices))
    except Exception as e:  # noqa: BLE001 - 読めなければ GPU なしで照合する (起動は止めない)
        logger.warning("Auto-tune: listing the GPUs for the backend failed: %s", e)
        return ()
    return tuple(hw.gpu_names)


def prime_backend_gpus(gpus: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """backend の GPU 名を決める (``gpus`` を渡せばそれ、無ければ読む)。backend の起動時に executor で 1 回呼ぶ。"""
    global _BACKEND_GPUS
    _BACKEND_GPUS = tuple(gpus) if gpus is not None else probe_backend_gpus()
    return _BACKEND_GPUS


def reset_backend_gpus() -> None:
    """未読へ戻す (テスト用)。"""
    global _BACKEND_GPUS
    _BACKEND_GPUS = None


def backend_pc() -> Any:
    """backend の現在の PC (GPU 名込みの完全な指紋の材料)。未読なら読む (起動時に読んでいれば読まない)。"""
    from backend.free.rag.rerank_selftest import collect_pc_info

    gpus = _BACKEND_GPUS if _BACKEND_GPUS is not None else prime_backend_gpus()
    return collect_pc_info(list(gpus))


# ── 決めてよいかの範囲指定 ─────────────────────────────────

#: (見積ってよいか, 共有する輪郭)。``None`` は指定なし (決めてよい)。
_DECIDE: ContextVar[tuple[bool, HardwareProfile | None] | None] = ContextVar("auto_tune_decide", default=None)


@contextmanager
def decide_scope(allow: bool, *, hardware: HardwareProfile | None = None) -> Iterator[None]:
    """この範囲の ``resolve_tuned`` (``allow_decide`` 省略) が見積ってよいかと、共有する輪郭。

    ``hardware`` を渡すと項目ごとに ``--list-devices`` を読み直さない (1 回の組み立てで共有する)。
    """
    token = _DECIDE.set((allow, hardware))
    try:
        yield
    finally:
        _DECIDE.reset(token)


def decide_allowed(explicit: bool | None = None) -> bool:
    """見積ってよいか。明示の指定があればそれ、無ければ :func:`decide_scope` の指定 (どちらも無ければ可)。"""
    if explicit is not None:
        return explicit
    scope = _DECIDE.get()
    return scope[0] if scope is not None else True


@contextmanager
def launch_scope(allow: bool) -> Iterator[None]:
    """backend のプロセス内で起動コマンドを組む範囲 (``server_control`` / プロセス管理)。

    許可しない (既定) なら保存値を読むだけ (``--list-devices`` も見積りも走らない)。許可する (base を
    止めた後の組み直し) なら輪郭を 1 回だけ読み、項目の間で共有する。ブロッキングなので呼び手は
    executor で走らせる。
    """
    with decide_scope(allow, hardware=probe_hardware() if allow else None):
        yield

#: 値の出どころ。
SOURCE_MANUAL = "manual"
SOURCE_SAVED = "saved"
SOURCE_ESTIMATED = "estimated"
SOURCE_FALLBACK = "fallback"

#: 起動スクリプトの 1 回の起動で同じ項目を何度も解決する (``-c`` / slots / VRAM 見積り…) ので、
#: 解決結果を config の dict に覚える (``launch_llama`` の ``__resolved_slots__`` と同じ流儀)。
CACHE_KEY = "__auto_tune_resolved__"

#: 保存済みの項目のうち反映に使えるもの (``skipped`` / ``failed`` は値を持たない)。
_USABLE_SOURCES = frozenset({"measured", "estimated"})

#: ``auto_tune.json`` の読み取りのキャッシュ (パス → ((mtime_ns, size, ino), レコード))。backend はチャットの
#: 経路で文脈長を何度も引くので、ファイルが変わらない限り読み直さない。
_RECORD_CACHE: dict[str, tuple[tuple[int, int, int], AutoTuneRecord | None]] = {}


@dataclass(frozen=True)
class Resolved:
    """解決した値と出どころ (``manual`` / ``saved`` / ``estimated`` / ``fallback``) と理由 (英語)。"""

    key: str
    value: Any
    source: str
    reason: str = ""

    @property
    def applied(self) -> bool:
        """調整の結果が効いているか (明示値なら偽)。"""
        return self.source != SOURCE_MANUAL


def _read_record(path: Path) -> AutoTuneRecord | None:
    """保存済みのレコード (ファイルの mtime と大きさが変わらなければキャッシュを返す)。"""
    try:
        st = path.stat()
    except OSError:
        _RECORD_CACHE.pop(str(path), None)
        return None
    # 書き込みは置き換え (AtomicWriter) なので、ファイルの番号 (st_ino) も変わる
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    cached = _RECORD_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    record, _status = load_auto_tune(path)
    _RECORD_CACHE[str(path)] = (stamp, record)
    return record


def _saved_item(record: AutoTuneRecord | None, spec: TuneSpec, basis: str | None) -> TuneItem | None:
    """反映に使える保存済みの項目 (指紋の照合は呼び手)。前提の印が今と違えば ``None``。"""
    if record is None:
        return None
    item = record.items.get(spec.key)
    if item is None or item.source not in _USABLE_SOURCES:
        return None
    if basis is not None and basis not in (item.reason or ""):
        return None
    return item


def _scheduled(cfg: dict[str, Any], key: str, project_root: Path | None, paths: TunePaths | None) -> bool:
    """``key`` が再計算の予約に載っているか (メモより予約を優先する: 長く動く backend の config には
    前の解決が残るので、画面の再起動で予約を果たせなくなる)。"""
    if project_root is None:
        return False
    record = _read_record((paths or resolve_tune_paths(cfg, project_root)).auto_tune)
    return record is not None and key in record.recompute


def _fallback(spec: TuneSpec, reason: str) -> Resolved:
    return Resolved(spec.key, spec.fallback, SOURCE_FALLBACK, reason)


def _persist(
    paths: TunePaths, spec: TuneSpec, outcome: Any, cfg: dict[str, Any], current: Any, now: str,
) -> None:
    """その場の見積りを 1 項目だけ保存する (最新を読み直してキー単位で書き足す。予約からも外す)。"""
    from backend.free.core.tuning.runner import build_item

    if save_item(paths.auto_tune, build_item(spec, outcome, cfg, now), current, now) is None:
        logger.warning("Auto-tune: failed to save the estimated %s to %s", spec.key, paths.auto_tune)


def _resolve_auto(
    spec: TuneSpec,
    cfg: dict[str, Any],
    project_root: Path | None,
    *,
    hardware: HardwareProfile | None,
    probes: HardwareProbes | None,
    paths: TunePaths | None,
    allow_decide: bool,
    persist: bool,
    now: Callable[[], str] | None,
    can_recompute: Callable[[], bool] | None,
) -> Resolved:
    from backend.free.core.tuning.gate import load_gate, should_measure
    from backend.free.rag.rerank_selftest import collect_pc_info

    if project_root is None:
        return _fallback(spec, "no_project_root")
    resolved_paths = paths or resolve_tune_paths(cfg, project_root)
    record = _read_record(resolved_paths.auto_tune)
    basis = spec.basis(cfg, project_root) if spec.basis is not None else None
    saved = _saved_item(record, spec, basis)

    if not allow_decide:
        # 起動スクリプトと同じ完全な指紋 (GPU 名込み) で照合する。GPU だけが変わった PC で保存値を
        # 使うと、確認待ちで保守側 (-c 8192) に倒れた llama-server と backend の値が食い違う
        if saved is not None and record is not None and record.fingerprint == backend_pc().digest:
            return Resolved(spec.key, saved.value, SOURCE_SAVED, saved.reason)
        return _fallback(spec, "not_tuned" if saved is None else "pc_changed")

    hw = hardware if hardware is not None else probe_hardware(probes)
    current = collect_pc_info(hw.gpu_names)
    usable = saved if saved is not None and record is not None and record.fingerprint == current.digest else None
    # 稼働中の実行が予約した項目は、base が止まっているとき (起動前 / 再起動の直前) だけ測り直す。
    # base が載っている間に見積ると空きが過小になるので、そのときは保存値のまま予約を残す。
    recompute = bool(record is not None and spec.key in record.recompute) and (
        can_recompute is None or can_recompute()
    )
    if usable is not None and not recompute:
        return Resolved(spec.key, usable.value, SOURCE_SAVED, usable.reason)
    ok, why = should_measure(load_gate(cfg, project_root, current, paths=resolved_paths))
    if not ok:
        return _fallback(spec, why)
    if spec.needs_servers:
        return _fallback(spec, "needs_servers")
    squeezed = memory_squeezed(hw)
    if squeezed is not None:
        # 一時的に握られた空きで見積った値を恒久化しない: 保存値があればそれ (予約は残す)、
        # 無ければこの回だけの見積り (空きが小さいので保守側に倒れる) を保存せずに使う
        logger.warning("Auto-tune: free memory looks held by something else (%s); not saving %s", squeezed, spec.key)
        if usable is not None:
            return Resolved(spec.key, usable.value, SOURCE_SAVED, usable.reason)
        persist = False
    try:
        outcome = spec.run(TuneContext(cfg, project_root, hardware_fn=lambda: hw))
    except Exception as e:  # noqa: BLE001 - 見積りの失敗で起動を止めない (保守側へ倒す)
        logger.warning("Auto-tune: estimating %s failed: %s", spec.key, e, exc_info=True)
        outcome = TuneOutcome("failed", reason=f"error:{type(e).__name__}", environmental=True)
    if outcome.status != "ok":
        if usable is not None:  # 予約の再計算に失敗: 前の保存値で起動し、予約は次へ残す
            return Resolved(spec.key, usable.value, SOURCE_SAVED, usable.reason)
        return _fallback(spec, outcome.reason or outcome.status)
    if persist:
        from backend.utils import utc_now

        _persist(resolved_paths, spec, outcome, cfg, current, (now or utc_now)())
    return Resolved(spec.key, outcome.value, SOURCE_ESTIMATED, outcome.reason)


def resolve_tuned(
    cfg: dict[str, Any],
    key: str,
    *,
    project_root: Path | None = None,
    hardware: HardwareProfile | None = None,
    probes: HardwareProbes | None = None,
    paths: TunePaths | None = None,
    registry: TuneRegistry | None = None,
    allow_decide: bool | None = None,
    persist: bool = True,
    use_cache: bool = True,
    now: Callable[[], str] | None = None,
    can_recompute: Callable[[], bool] | None = None,
) -> Resolved:
    """項目 ``key`` の起動時の値を決める (規則はモジュールの説明)。

    ``hardware`` / ``probes`` は輪郭の注入 (無ければ :func:`decide_scope` の輪郭、それも無ければ
    ``--list-devices`` 等を読む)。``allow_decide=False`` は backend の読み取り (測らない・書かない)、
    省略は :func:`decide_scope` の指定 (無ければ決めてよい)。``use_cache`` は起動スクリプトの 1 回の
    起動の中で同じ答えを返すための config へのメモ (``allow_decide`` のときだけ使う)。``can_recompute``
    は予約 (``recompute``) を今果たしてよいか (base が止まっているか) で、``None`` は常に可。
    未知の ``key`` は ``ValueError``。
    """
    scope = _DECIDE.get()
    allow_decide = decide_allowed(allow_decide)
    if hardware is None and scope is not None:
        hardware = scope[1]
    spec = (registry or load_builtin_tuners()).get(key)
    if spec is None:
        raise ValueError(f"unknown tune item: {key!r}")
    if not is_auto(spec, cfg):
        return Resolved(key, config_value(cfg, spec.config_key, spec.config_default), SOURCE_MANUAL, MANUAL)
    memo = cfg.get(CACHE_KEY) if use_cache and allow_decide else None
    if isinstance(memo, dict) and isinstance(memo.get(key), dict) and not _scheduled(cfg, key, project_root, paths):
        return Resolved(**memo[key])
    result = _resolve_auto(
        spec, cfg, project_root, hardware=hardware, probes=probes, paths=paths,
        allow_decide=allow_decide, persist=persist, now=now, can_recompute=can_recompute,
    )
    if use_cache and allow_decide:
        cfg.setdefault(CACHE_KEY, {})[key] = asdict(result)
    return result


__all__ = [
    "CACHE_KEY",
    "SOURCE_ESTIMATED",
    "SOURCE_FALLBACK",
    "SOURCE_MANUAL",
    "SOURCE_SAVED",
    "BACKEND_LIST_DEVICES_TIMEOUT_SEC",
    "Resolved",
    "backend_pc",
    "decide_allowed",
    "decide_scope",
    "launch_scope",
    "prime_backend_gpus",
    "probe_backend_gpus",
    "reset_backend_gpus",
    "resolve_tuned",
]
