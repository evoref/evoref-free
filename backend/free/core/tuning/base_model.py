"""base モデル (``model_paths.base_model``) の見積りの材料 — 項目 ctx / ngl が共有する (c_16 §7.2.3)。

GGUF の読み取りと KV の見積りは起動スクリプトの既存実装 (``read_gguf_metadata`` /
``estimate_kv_cache_mb`` / プロファイルの ``context_size``) を lazy import して使う (式を二重に持たない)。
起動スクリプトが無い配布物 (backend だけ) では材料が取れず、項目は環境起因の失敗として保守側へ倒れる。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: config にもプロファイルにも context_size が無いときの base の ``-c``
#: (``scripts/launch_llama.py::_CONTEXT_SIZE_DEFAULTS["base"]`` / ``backend.config._CONTEXT_SIZE_FALLBACK``
#: と同じ値)。ctx の上限 (プロファイル未宣言のとき) と保守側の値に使う。
DEFAULT_CTX = 8192
#: base の計算バッファの見積り (MiB)。実測ではなく見積り (埋め込み / リランカーの配置判定と同じ値。
#: ``-b 512`` の計算バッファは語彙 15 万級のモデルで数百 MiB なので、多めに見ておく)。
COMPUTE_MARGIN_MIB = 1024
#: :data:`COMPUTE_MARGIN_MIB` が前提にする物理バッチ ``-ub`` (llama-server の既定)。
BASE_UBATCH = 512
#: 単体 GPU で空きから引くヘッドルーム (MiB)。ドライバと他のプロセスの揺れの分 (c_16 §7.2.3
#: 「dGPU: 数百 MiB」)。iGPU は ``runtime.vulkan_host_buffer_headroom_mib`` (:func:`gpu_headroom_mib`)。
DGPU_HEADROOM_MIB = 512
#: CPU / iGPU (共有メモリ) の配置で空き RAM から残す余白 (MiB)。同居する backend・埋め込み・
#: フロントエンドと OS の揺れの分。見積り (実測ではない)。
RAM_RESERVE_MIB = 4096
#: ``llama.slots: auto`` の解決 (``launch_llama.resolve_base_slots``) と同じ閾値と本数。
_LONG_FORM_SLOT_MIN_CTX = 16384
_SLOTS_BASE = 3
_SLOTS_WITH_LONG_FORM = 4


def no_nested_recompute() -> bool:
    """項目の中で別の項目 (ctx / ngl / batch) を引くときの ``can_recompute``: 予約を果たさない。

    予約の再計算は起動経路の最上位 (``-c`` / ``-ngl`` を決める呼び出し、base が止まっているとき) の
    役目で、入れ子の参照が base の稼働中に予約を消してはいけない (保存値があればそれを使う)。
    """
    return False


@dataclass(frozen=True)
class BaseModelInfo:
    """見積りに使う base モデルの性質。

    ``kv_mb(ctx)`` は文脈長 ``ctx`` の文脈メモリ (KV + hybrid の再帰状態、MiB)。スロット数は
    config の ``llama.slots`` (``auto`` は ctx で 3 / 4) から決めて渡す。見積れなければ ``None``。
    """

    name: str
    model_mb: int
    n_layers: int | None
    ctx_train: int | None
    profile_ctx: int | None
    kv_mb: Callable[[int], int | None]
    compute_mb: int = COMPUTE_MARGIN_MIB

    @property
    def basis(self) -> str:
        """保存結果の前提の印 (:attr:`TuneSpec.basis`)。モデルを替えたら見積り直す。"""
        return basis_tag(self.name)

    @property
    def ctx_upper(self) -> int:
        """ctx の上限: プロファイルの既定 (無ければ :data:`DEFAULT_CTX`) とモデルの学習 ctx の小さい方。"""
        upper = self.profile_ctx or DEFAULT_CTX
        if self.ctx_train:
            upper = min(upper, int(self.ctx_train))
        return upper


def compute_mb_for_ubatch(ubatch: Any) -> int:
    """物理バッチ ``ubatch`` の計算バッファの見積り (MiB)。項目 ctx / ngl / b・ub が共有する 1 実装。

    :data:`BASE_UBATCH` 以下 (と未指定) は :data:`COMPUTE_MARGIN_MIB`。超えた分は ub に比例させる
    (計算バッファの主成分 = 活性・注意の中間値は ub 行ぶん確保されるので線形とみなす。実測ではなく
    多めの見積り)。ctx / ngl の見積りに足す計算バッファはこの値 1 つで、b・ub の増分を別に足さない。
    """
    if not isinstance(ubatch, int) or isinstance(ubatch, bool) or ubatch <= BASE_UBATCH:
        return COMPUTE_MARGIN_MIB
    return -(-COMPUTE_MARGIN_MIB * ubatch // BASE_UBATCH)


def explicit_ubatch(cfg: dict[str, Any]) -> int | None:
    """``llama.ubatch_size`` の明示値 (``auto`` / ``null`` / 無しは ``None``)。"""
    raw = (cfg.get("llama") or {}).get("ubatch_size")
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def base_placement(ngl: int, n_layers: int | None, *, has_gpu: bool) -> str:
    """base の配置 (``gpu`` 全層 / ``partial`` 一部 / ``cpu``) を ``-ngl`` から決める (純関数)。

    層数が分からない部分値は ``partial`` (CPU を使う側) に倒す。
    """
    if not has_gpu or ngl == 0:
        return "cpu"
    if ngl < 0 or ngl >= 999 or (n_layers and ngl >= n_layers):
        return "gpu"
    return "partial"


def resolve_base_ngl(cfg: dict[str, Any], project_root: Path, hw: Any) -> int:
    """base の ``-ngl`` (明示の整数はそのまま、``auto`` は項目 ngl の解決値、決まらなければ 999)。

    項目 ``threads`` / ``ram_params`` が base の配置を知るために使う (保存も resolve_tuned の規則どおり)。
    """
    raw = (cfg.get("llama") or {}).get("gpu_layers", 999)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    from backend.free.core.tuning.resolve import resolve_tuned

    value = resolve_tuned(
        cfg, "ngl", project_root=project_root, hardware=hw, use_cache=False, can_recompute=no_nested_recompute,
    ).value
    return value if isinstance(value, int) and not isinstance(value, bool) else 999


#: 環境移行の確認待ち (gate の ``tune_pending`` / ``tune_declined``) の間も、起動スクリプトが **保存しない
#: 一時の見積り** を使う項目。どれも llama-server を起こさない見積り (GGUF と ``--list-devices`` / RAM を
#: 読むだけ) で、測り直しの確認が要る「測定」(埋め込みの配置 / リランカーの自己テスト = サーバを起こす) とは
#: 違う。確認待ちに保守側の固定値 (ctx 8192 / ngl 999) を使うと、VRAM の小さい PC で base が全層 GPU に
#: 載って OOM する (以前の ``gpu_layers: auto`` は毎回の起動で見積もっていた)。
PROVISIONAL_KEYS: frozenset[str] = frozenset({"ctx", "ngl", "vram_budget"})
#: 一時の見積りの出どころ (``resolve.Resolved.source``)。保存しない。
SOURCE_PROVISIONAL = "provisional"
#: ``resolve_tuned`` のメモ (``resolve.CACHE_KEY``) と同じ流儀で、1 回の起動の中の一時の見積りを覚える
#: (``--list-devices`` を何度も読まない)。
PROVISIONAL_CACHE_KEY = "__auto_tune_provisional__"


def _pending(reason: str) -> bool:
    """``resolve_tuned`` の保守側の理由が、環境移行の確認待ち (gate の ``tune_<decision>``) か。"""
    return str(reason or "").startswith("tune_")


def resolve_or_provisional(
    cfg: dict[str, Any], key: str, *, project_root: Path | None, hardware: Any = None,
    use_cache: bool = True, **kwargs: Any,
) -> Any:
    """``resolve_tuned`` に、確認待ちの間の **一時の見積り** (保存しない) を足したもの。

    ``resolve_tuned`` が確認待ちのため保守側の値を返した :data:`PROVISIONAL_KEYS` の項目だけ、その場で
    ``spec.run`` を走らせて値を返す (出どころ :data:`SOURCE_PROVISIONAL`)。``auto_tune.json`` には書かない
    (確認の前に保存すると、確認の答えを待たずに新しい PC の結果として扱われる)。それ以外は
    ``resolve_tuned`` の答えのまま。backend の読み取り (``allow_decide=False``) も素通し (測らない)。

    - ctx は backend が確認待ちの間に使う保守側 (``fallback`` = 8192) を下回らない — backend のトークン予算が
      llama-server の ``-c`` を超えないように (見積りの上限は従来どおりプロファイル)
    - 見積りも取れない (輪郭やモデルが読めない) ときの ngl は、全層 (999) ではなく全層の 40%
      (:func:`backend.free.core.tuning.tuners.ngl.conservative_ngl`)
    """
    from backend.free.core.tuning import resolve as rs
    from backend.free.core.tuning.items import TuneContext, TuneOutcome, load_builtin_tuners

    resolved = rs.resolve_tuned(cfg, key, project_root=project_root, hardware=hardware, use_cache=use_cache, **kwargs)
    if (
        key not in PROVISIONAL_KEYS or resolved.source != rs.SOURCE_FALLBACK or not _pending(resolved.reason)
        or project_root is None or not rs.decide_allowed(kwargs.get("allow_decide"))
    ):
        return resolved
    memo = cfg.get(PROVISIONAL_CACHE_KEY) if use_cache else None
    if isinstance(memo, dict) and isinstance(memo.get(key), dict):
        return rs.Resolved(**memo[key])
    spec = (kwargs.get("registry") or load_builtin_tuners()).get(key)
    hw = hardware if hardware is not None else rs.probe_hardware(kwargs.get("probes"))
    try:
        outcome = spec.run(TuneContext(cfg, project_root, hardware_fn=lambda: hw))
    except Exception as e:  # noqa: BLE001 - 一時の見積りの失敗で起動を止めない (保守側へ倒す)
        outcome = TuneOutcome("failed", reason=f"error:{type(e).__name__}", environmental=True)
    value: Any
    if outcome.status == "ok":
        value, why = outcome.value, f"unsaved estimate while {resolved.reason}: {outcome.reason}"
        if key == "ctx" and isinstance(value, int) and isinstance(spec.fallback, int):
            value = max(value, spec.fallback)
    elif key == "ngl":
        from backend.free.core.tuning.tuners.ngl import conservative_ngl

        value = conservative_ngl(cfg, project_root)
        why = f"conservative while {resolved.reason} (estimate {outcome.status}: {outcome.reason})"
    else:
        return resolved
    result = rs.Resolved(key, value, SOURCE_PROVISIONAL, why)
    if use_cache:
        cfg.setdefault(PROVISIONAL_CACHE_KEY, {})[key] = {
            "key": key, "value": value, "source": SOURCE_PROVISIONAL, "reason": why,
        }
    return result


def basis_tag(model_name: str) -> str:
    """前提の印の表記 (``model=<ファイル名>``)。"""
    return f"model={model_name}"


def slots_for(raw_slots: Any, ctx: int) -> int:
    """``llama.slots`` の実数 (``auto`` は ctx が 16384 以上で 4、未満で 3)。"""
    if raw_slots == "auto" or raw_slots is None:
        return _SLOTS_WITH_LONG_FORM if ctx >= _LONG_FORM_SLOT_MIN_CTX else _SLOTS_BASE
    return max(1, int(raw_slots))


def gpu_headroom_mib(cfg: dict[str, Any], kind: str) -> int:
    """空きから引くヘッドルーム。iGPU は Vulkan host buffer の予約 (現行の
    ``runtime.vulkan_host_buffer_headroom_mib``、既定 4096) を踏襲、単体 GPU は :data:`DGPU_HEADROOM_MIB`。
    """
    if kind == "igpu":
        return int((cfg.get("runtime") or {}).get("vulkan_host_buffer_headroom_mib", 4096))
    return DGPU_HEADROOM_MIB


def base_model_path(cfg: dict[str, Any], project_root: Path) -> Path | None:
    """``model_paths.base_model`` の絶対パス (未設定なら ``None``。存在は見ない)。"""
    raw = (cfg.get("model_paths") or {}).get("base_model")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def base_basis(cfg: dict[str, Any], project_root: Path) -> str | None:
    """今の config の base モデルの印 (:attr:`TuneSpec.basis` の実装)。未設定なら ``None``。"""
    path = base_model_path(cfg, project_root)
    return basis_tag(path.name) if path is not None else None


def load_base_model_info(cfg: dict[str, Any], project_root: Path) -> BaseModelInfo | None:
    """base モデルの材料を GGUF とプロファイルから読む。モデル無し / 起動スクリプト無しは ``None``。"""
    path = base_model_path(cfg, project_root)
    if path is None or not path.is_file():
        return None
    try:
        # ``from scripts import launch_llama`` ではなく名前で引く: 起動スクリプトをスクリプトとして
        # 起動したときは ``scripts`` パッケージが無く、``scripts.launch_llama`` だけが登録される
        # (``launch_llama._tuned``)
        ll = importlib.import_module("scripts.launch_llama")
    except ImportError:
        return None
    size = ll._file_size_mb(path)
    if size is None:
        return None
    meta = ll._read_gguf_metadata_cached(path)
    lc = cfg.get("llama") or {}
    raw_slots = lc.get("slots", "auto")

    def kv_mb(ctx: int) -> int | None:
        return ll.estimate_kv_cache_mb(
            meta, ctx, lc.get("cache_type_k"), lc.get("cache_type_v"), n_seq=slots_for(raw_slots, ctx),
        )

    layers = meta.get("block_count") or ll._read_gguf_layer_count(path)
    return BaseModelInfo(
        name=path.name, model_mb=int(size), n_layers=int(layers) if layers else None,
        ctx_train=meta.get("context_length"),
        profile_ctx=ll._profile_context_size_for_model(path, project_root),
        kv_mb=kv_mb,
        # 明示の ``-ub`` はその計算バッファで見る (``auto`` の ub は項目 ngl が b・ub の解決値で差し替える)
        compute_mb=compute_mb_for_ubatch(explicit_ubatch(cfg)),
    )


__all__ = [
    "BASE_UBATCH",
    "COMPUTE_MARGIN_MIB",
    "DEFAULT_CTX",
    "DGPU_HEADROOM_MIB",
    "PROVISIONAL_CACHE_KEY",
    "PROVISIONAL_KEYS",
    "RAM_RESERVE_MIB",
    "SOURCE_PROVISIONAL",
    "BaseModelInfo",
    "base_basis",
    "base_model_path",
    "base_placement",
    "basis_tag",
    "compute_mb_for_ubatch",
    "explicit_ubatch",
    "gpu_headroom_mib",
    "load_base_model_info",
    "no_nested_recompute",
    "resolve_base_ngl",
    "resolve_or_provisional",
    "slots_for",
]
