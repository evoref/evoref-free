"""調整項目のレジストリ (c_16 §7.2.3 ``items.py``)。

1 項目 = 1 モジュール (``backend/free/core/tuning/tuners/<key>.py``)。各モジュールは import 時に
:func:`register` で :class:`TuneSpec` を 1 つ登録する。読み込む一覧は :data:`TUNER_MODULES` で、
片落ち (置いたのに一覧に無い / 一覧にあるのに無い) はテストが落とす (``backend/formats.py`` の
``DECLARING_MODULES`` と同じ流儀)。同名の項目は 2 つ作らない (登録で拒否する)。

項目を 1 つ足す手順: ``tuners/<key>.py`` に ``run`` 関数と ``register(TuneSpec(...))`` を書き、
:data:`TUNER_MODULES` に 1 行足す。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from backend.io.codec import persisted

if TYPE_CHECKING:
    from backend.free.core.tuning.hardware import HardwareProfile

TuneMethod = Literal["measured", "estimated"]
#: 項目の値の出どころ。``measured`` / ``estimated`` は今回求めた値、``skipped`` は対象外 (機能が off 等)、
#: ``failed`` は PC の性質として確定した失敗 (環境起因の失敗は保存しない)。
TuneSource = Literal["measured", "estimated", "skipped", "failed"]
OutcomeStatus = Literal["ok", "skipped", "failed"]

#: 利用者の明示値のため反映しない理由。
MANUAL = "manual"
#: 「自動」を意味しない既定 (``null`` で従来の既定に従う等) のため反映しない理由。
NOT_AUTO = "not_auto"


@persisted()
@dataclass
class TuneItem:
    """1 項目の結果 (``cache/auto_tune.json`` の ``items`` の値)。

    ``value`` は提案値 (反映しない ``manual`` の項目でも残す)。``applied`` は config がその項目を
    自動に任せている (反映される) か。
    """

    key: str
    value: Any = None
    source: str = ""
    reason: str = ""
    config_key: str = ""
    applied: bool = False
    requires_restart: bool = False
    measured_at: str = ""
    _extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class TuneOutcome:
    """項目の ``run`` の戻り値 (純粋な結果。``applied`` は runner が config から決める)。

    ``environmental`` が真の失敗は PC の性質ではない (一時的な起動失敗・空きの不足等) ので
    保存しない (次の実行でまた測る、c_16 §7.2.3)。
    """

    status: OutcomeStatus
    value: Any = None
    reason: str = ""
    environmental: bool = False


@dataclass
class TuneContext:
    """項目の ``run`` に渡す文脈。"""

    cfg: dict[str, Any]
    project_root: Path
    #: 手動の実行 (保存済みの結果があっても測り直す)。
    force: bool = False
    #: 遅延して測る輪郭 (使う項目だけが :meth:`hardware` で求める)。
    hardware_fn: Callable[[], HardwareProfile] | None = None
    _hardware: HardwareProfile | None = field(default=None, repr=False)

    def hardware(self) -> HardwareProfile:
        """輪郭を 1 回だけ求める。"""
        if self._hardware is None:
            if self.hardware_fn is None:
                from backend.free.core.tuning.hardware import probe_hardware

                self._hardware = probe_hardware()
            else:
                self._hardware = self.hardware_fn()
        return self._hardware


@dataclass(frozen=True)
class TuneSpec:
    """調整項目の宣言。

    ``auto_values`` は config の値のうち「自動に任せる」を意味するもの (``auto`` / ``None`` /
    threads の ``0`` 等、項目ごとに宣言する)。キーが無いときは ``config_default`` を値とみなす。
    """

    key: str
    config_key: str
    method: TuneMethod
    requires_restart: bool
    run: Callable[[TuneContext], TuneOutcome]
    auto_values: tuple[Any, ...] = ("auto", None)
    config_default: Any = None
    #: 先に走らせる項目 (例: リランカーは埋め込みが VRAM を取った後に測る)。
    depends_on: tuple[str, ...] = ()
    description: str = ""
    #: 稼働中の backend の API から走らせてよいか。一時ポートで測る項目だけ真 (本番ポートを使う項目は
    #: 停止中の CLI に限る)。
    safe_while_running: bool = False
    #: 測るのに llama-server を起こす項目 (埋め込みの配置 / リランカー)。起動前の自動実行
    #: (``evoref tune --startup-check``) と :func:`~backend.free.core.tuning.resolve.resolve_tuned` の
    #: その場の見積りは、偽の項目 (見積りだけで決まるもの) に限る。
    needs_servers: bool = False
    #: 決められないとき (保存結果が無く見積りもできない / 環境移行の確認待ち) に使う保守側の値。
    fallback: Any = None
    #: 保存結果が前提とした条件 (例: base モデルのファイル名) の印。``run`` は結果の ``reason`` に
    #: この印を含め、:func:`~backend.free.core.tuning.resolve.resolve_tuned` は印が今の条件と
    #: 違う保存結果を使わない (モデルを替えたら見積り直す)。``None`` は条件を持たない項目。
    basis: Callable[[dict[str, Any], Path], str | None] | None = None


class TuneRegistry:
    """項目の台帳。同名の登録を拒否する。"""

    def __init__(self) -> None:
        self._specs: dict[str, TuneSpec] = {}

    def register(self, spec: TuneSpec) -> TuneSpec:
        if spec.key in self._specs:
            raise ValueError(f"tune item {spec.key!r} is already registered")
        self._specs[spec.key] = spec
        return spec

    def get(self, key: str) -> TuneSpec | None:
        return self._specs.get(key)

    def keys(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[TuneSpec]:
        return list(self._specs.values())

    def ordered(self, only: Iterable[str] | None = None) -> list[TuneSpec]:
        """依存順 (登録順を保った位相順)。``only`` は絞り込み (依存先は足さない)。

        未知のキー・未登録の依存先・循環は ``ValueError``。
        """
        for spec in self._specs.values():
            missing = [d for d in spec.depends_on if d not in self._specs]
            if missing:
                raise ValueError(f"tune item {spec.key!r} depends on unknown items: {missing}")
        wanted = list(self._specs) if only is None else list(dict.fromkeys(only))
        unknown = [k for k in wanted if k not in self._specs]
        if unknown:
            raise ValueError(f"unknown tune items: {unknown}")
        order: list[str] = []
        visiting: set[str] = set()

        def visit(key: str) -> None:
            if key in order:
                return
            if key in visiting:
                raise ValueError(f"tune items have a dependency cycle at {key!r}")
            visiting.add(key)
            for dep in self._specs[key].depends_on:
                visit(dep)
            visiting.discard(key)
            order.append(key)

        # 登録順は import の順 (起動スクリプトやテストが項目モジュールを先に import すると変わる) なので、
        # 組み込みの項目は TUNER_MODULES の並びを正とする (それ以外は登録順のまま、sorted は安定)
        rank = {name: i for i, name in enumerate(TUNER_MODULES)}
        for key in sorted(self._specs, key=lambda k: rank.get(getattr(self._specs[k].run, "__module__", ""), len(rank))):
            visit(key)
        chosen = set(wanted)
        return [self._specs[k] for k in order if k in chosen]


#: 既定の台帳 (``tuners/`` の各モジュールが import 時に登録する)。
REGISTRY = TuneRegistry()

#: 項目を登録するモジュール。1 項目 = 1 モジュール。足したらここにも足す
#: (``backend/free/core/tuning/tests/test_items.py`` が片落ちを検出する)。
TUNER_MODULES: tuple[str, ...] = (
    # base の見積り (ctx → ngl) と VRAM 予算は、埋め込み / リランカーの判別 (サーバを起こして
    # VRAM を取る) より先に、何も載っていない空きで決める。
    "backend.free.core.tuning.tuners.ctx",
    # b・ub は ctx の KV に上乗せして決め、ngl はその ub の計算バッファで層数を決める (ctx → b・ub → ngl)
    "backend.free.core.tuning.tuners.batch",
    "backend.free.core.tuning.tuners.ngl",
    "backend.free.core.tuning.tuners.ram_params",
    "backend.free.core.tuning.tuners.vram_budget",
    "backend.free.core.tuning.tuners.embed_placement",
    "backend.free.core.tuning.tuners.embed_params",
    "backend.free.core.tuning.tuners.rerank",
    # 配置 (ngl / 埋め込み / リランカー) が決まった後でコアを配分する
    "backend.free.core.tuning.tuners.threads",
    "backend.free.core.tuning.tuners.tps",
    "backend.free.core.tuning.tuners.background_budget",
    "backend.free.core.tuning.tuners.startup_timeouts",
)


def register(spec: TuneSpec) -> TuneSpec:
    """既定の台帳へ登録する (``tuners/*`` が import 時に呼ぶ)。"""
    return REGISTRY.register(spec)


def load_builtin_tuners() -> TuneRegistry:
    """:data:`TUNER_MODULES` を全て import して既定の台帳を揃える。"""
    for name in TUNER_MODULES:
        importlib.import_module(name)
    return REGISTRY


# ── 反映の可否 ────────────────────────────────────────────


def config_value(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """``"a.b.c"`` を辿った config の値 (途中が無ければ ``default``)。"""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _same(value: Any, candidate: Any) -> bool:
    """型も含めて同じか (``0`` と ``False``、``1`` と ``True`` を混ぜない)。"""
    if candidate is None or value is None:
        return value is candidate
    return type(value) is type(candidate) and value == candidate


def is_auto(spec: TuneSpec, cfg: dict[str, Any]) -> bool:
    """config がこの項目を自動に任せているか (値が ``auto_values`` のどれか)。"""
    raw = config_value(cfg, spec.config_key, spec.config_default)
    return any(_same(raw, v) for v in spec.auto_values)


def applied_for(spec: TuneSpec, cfg: dict[str, Any]) -> tuple[bool, str]:
    """(反映するか, 反映しない理由)。自動を意味する値のときだけ ``(True, "")``。

    明示の値 → ``(False, "manual")`` (提案値は ``TuneItem.value`` に残す)。自動を意味しない
    ``null`` (従来の既定に従う項目) → ``(False, "not_auto")``。
    """
    if is_auto(spec, cfg):
        return True, ""
    raw = config_value(cfg, spec.config_key, spec.config_default)
    return False, NOT_AUTO if raw is None else MANUAL


__all__ = [
    "MANUAL",
    "NOT_AUTO",
    "REGISTRY",
    "TUNER_MODULES",
    "TuneContext",
    "TuneItem",
    "TuneMethod",
    "TuneOutcome",
    "TuneRegistry",
    "TuneSource",
    "TuneSpec",
    "applied_for",
    "config_value",
    "is_auto",
    "load_builtin_tuners",
    "register",
]
