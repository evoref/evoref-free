"""項目 ``embed_params`` — 埋め込みの batch / ubatch / HTTP バッチ件数 / timeout (c_16 §7.2.3「埋め込み batch」)。

材料は埋め込みの配置の判別結果 (``cache/embed_placement.json``、§7.2.2) の 1 要求の p50 と輪郭 (RAM / GPU)。
p50 を既知のクエリの代表トークン数で割って **ms/トークン** にし (単一クエリの値なので、バッチの実効速度より
遅く出る = 保守側)、

- ``timeout``: HTTP バッチの最小件数 (:data:`HTTP_BATCH_MIN`) の文書と、``max_length`` トークンの単一入力の
  どちらも timeout の半分に収まる秒数。下限は従来の 30 秒、上限 :data:`TIMEOUT_MAX`
- HTTP バッチ件数: 1 回の HTTP (文書 ``rag.chunk_size`` トークン × 件数) が timeout の半分に収まる件数
  (:data:`HTTP_BATCH_MIN` 〜 従来の :data:`HTTP_BATCH_MAX`)
- ``query_timeout``: 単一クエリの p95 の見積り (p50 × :data:`QUERY_P95_OVER_P50`。判別結果は p50 しか
  残さない) × :data:`QUERY_TIMEOUT_FACTOR`。下限は従来の 3.0 秒、上限 :data:`QUERY_TIMEOUT_MAX`
- ``batch_size`` / ``ubatch_size``: ``max_length`` 以上 (単一入力が ubatch を超えると llama-server が 500)。
  文脈長が ``max_length`` より大きければ、計算バッファの見積りが空きに収まる範囲で 2 の冪まで広げる

を決める。判別結果が無い (未判別 / PC が違う / GPU が無く測っていない) ときは、遅い CPU を仮定した
:data:`ASSUMED_MS_PER_TOKEN` で同じ式を通す (保守側)。

反映は **キーごと**: ``embedding.batch_size`` / ``ubatch_size`` / ``timeout`` / ``query_timeout`` が ``auto`` /
``null`` のものだけ調整値を使い、明示値はそのまま使う (:func:`effective_embed_params`)。HTTP バッチ件数は
config のキーを持たない (常に調整値)。``ubatch >= max_length`` の保証は調整値にだけ掛け、明示値が破って
いれば警告だけ出す (明示値は利用者の選択)。

既定の出どころは本モジュール 1 か所 (:data:`FALLBACK` と定数)。起動スクリプト (``-b`` / ``-ub``)・
``LlamaCppEmbedder`` (HTTP バッチ / timeout)・EvidenceStore / sleep-time の埋め込みバッチが
:func:`resolve_embed_params` の同じ解決値を使う。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.free.core.tuning.hardware import HardwareProfile
from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, register

KEY = "embed_params"

#: 1 回の HTTP に載せる件数の上限 (従来の ``_MAX_HTTP_BATCH`` / ``EMBED_BATCH_SIZE``) と下限。
HTTP_BATCH_MAX = 64
HTTP_BATCH_MIN = 4
#: ``embedding.timeout`` の下限 (従来の既定) と上限 (秒)。
TIMEOUT_MIN = 30.0
TIMEOUT_MAX = 120.0
#: 1 回の HTTP が timeout のこの割合 (半分) に収まるように件数と timeout を決める。
TIMEOUT_SHARE = 0.5
#: ``embedding.query_timeout`` の下限 (従来の既定) と上限 (秒)。上限はチャットの TTFT に前置きされる分。
QUERY_TIMEOUT_MIN = 3.0
QUERY_TIMEOUT_MAX = 10.0
#: 単一クエリの p95 / p50 の見積り。判別結果は p50 しか残さないので比で置く。実測 (c_16 §7.2.2、
#: 890M): base の prefill 中のクエリ p50 は単独の約 3 倍 (66 → 214 ms)。
QUERY_P95_OVER_P50 = 3.0
#: p95 に掛ける余裕。
QUERY_TIMEOUT_FACTOR = 2.0
#: 判別の既知のクエリ (``embed_placement.PROBE_TEXTS``) の代表トークン数 (``estimate_tokens`` の中央値)。
#: p50 はこの長さの 1 要求の時間とみなす。
PROBE_TOKENS = 48
#: 測っていないときに仮定する ms/トークン (純 CPU で約 400 tok/s。実測の純 CPU は約 595 tok/s、
#: c_16 §7.2.2 の表。それより遅い側に置く)。
ASSUMED_MS_PER_TOKEN = 2.5
#: 文書 1 件のトークン数の既定 (``rag.chunk_size`` の既定と同じ)。
DEFAULT_DOC_TOKENS = 512
#: llama-server の既定 ``-ub`` (これより小さくはしない)。
UBATCH_MIN = 512
#: ``max_length`` の既定 (``EmbeddingConfig.max_length``)。
DEFAULT_MAX_LENGTH = 8192
#: 非因果の埋め込みの計算バッファの見積り: ubatch² × この値 (バイト)。注意の重み (16 ヘッド × fp32)。
ATTN_BYTES_PER_PAIR = 64
#: 計算バッファに使ってよい空き (RAM / VRAM) の割合 (base・リランカーと分け合う)。
MEMORY_SHARE = 0.25
#: config のキーを持つ調整値 (HTTP バッチ件数は常に調整値)。
CONFIG_FIELDS: tuple[str, ...] = ("batch_size", "ubatch_size", "timeout", "query_timeout")

_MIB = 1024 * 1024


@dataclass(frozen=True)
class EmbedTiming:
    """判別結果から取り出した配置と 1 要求の p50 (ms)。"""

    placement: str
    p50_ms: float


@dataclass(frozen=True)
class EmbedParams:
    """埋め込みの実効値 (明示値と調整値を合わせたもの)。

    ``sources`` はキー → ``manual`` (明示値) / ``tuned`` (調整値)。``warnings`` は明示値が
    不変条件を破っているときの英語の文 (呼び手がログ / 標準エラーへ出す)。
    """

    batch_size: int
    ubatch_size: int
    http_batch: int
    timeout: float
    query_timeout: float
    sources: dict[str, str]
    warnings: tuple[str, ...] = ()


# ── 純関数 ────────────────────────────────────────────────


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ms_per_token(timing: EmbedTiming | None) -> float:
    """1 トークンあたりの ms。測っていなければ :data:`ASSUMED_MS_PER_TOKEN`。"""
    if timing is None or timing.p50_ms <= 0:
        return ASSUMED_MS_PER_TOKEN
    return timing.p50_ms / PROBE_TOKENS


def ubatch_floor(max_length: int) -> int:
    """不変条件 ``ubatch >= max_length`` を満たす最小の ubatch (llama-server の既定 512 未満にはしない)。"""
    return max(int(max_length), UBATCH_MIN)


def compute_buffer_mib(ubatch: int) -> int:
    """ubatch の計算バッファの見積り (MiB)。"""
    return math.ceil(ubatch * ubatch * ATTN_BYTES_PER_PAIR / _MIB)


def _memory_pool(hw: HardwareProfile, placement: str | None) -> tuple[int, str] | None:
    """(計算バッファに使ってよい MiB, 見た先)。GPU 配置なら最大空きの GPU、それ以外は RAM。取れなければ ``None``。"""
    gpu = hw.best_gpu
    if placement == "gpu" and gpu is not None and gpu.free_mib > 0:
        return int(gpu.free_mib * MEMORY_SHARE), "vram"
    if hw.free_ram_mib > 0:
        return int(hw.free_ram_mib * MEMORY_SHARE), "ram"
    return None


def decide_ubatch(
    hw: HardwareProfile, placement: str | None, *, max_length: int, context_size: int,
) -> tuple[int, str]:
    """(ubatch, 理由)。``max_length`` 以上 (2 の冪で文脈長まで) のうち計算バッファが空きに収まる最大。

    下限 (``max_length``) すら収まらなくても下限を返す (不変条件が先。理由に ``warn`` を残す)。
    """
    floor = ubatch_floor(max_length)
    candidates = [floor]
    size = 1 << (floor - 1).bit_length()
    if size == floor:
        size *= 2
    while size <= int(context_size):
        candidates.append(size)
        size *= 2
    pool = _memory_pool(hw, placement)
    if pool is None:
        return floor, f"ubatch {floor} (= max_length floor; memory unknown)"
    free, where = pool
    for ub in reversed(candidates):
        need = compute_buffer_mib(ub)
        if need <= free:
            return ub, f"ubatch {ub} ({where}: compute ~{need} MiB <= {free} MiB)"
    # 不変条件 (単一入力 <= ubatch) は崩さない。計算バッファを減らせるのは max_length (と context_size) を
    # 下げることだけなので、それを提案する (backend の切り詰めも max_length を読むので自動では下げない)
    half = ubatch_floor(max(UBATCH_MIN, int(max_length) // 2))
    hint = (
        f"; consider embedding.max_length / context_size {half} (compute ~{compute_buffer_mib(half)} MiB)"
        if half < floor else ""
    )
    return floor, (
        f"warn: ubatch {floor} ({where}: compute ~{compute_buffer_mib(floor)} MiB > {free} MiB, "
        f"kept for max_length){hint}"
    )


def decide_embed_params(
    hw: HardwareProfile,
    timing: EmbedTiming | None,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    context_size: int | None = None,
    doc_tokens: int = DEFAULT_DOC_TOKENS,
    basis: str = "",
) -> TuneOutcome:
    """配置の測定 (無ければ保守側) と輪郭から埋め込みの調整値を決める (純関数)。"""
    mpt = ms_per_token(timing)
    doc = max(1, int(doc_tokens))
    one_doc_sec = doc * mpt / 1000.0
    longest_sec = max(HTTP_BATCH_MIN * one_doc_sec, int(max_length) * mpt / 1000.0)
    timeout = float(_clamp(math.ceil(longest_sec / TIMEOUT_SHARE), TIMEOUT_MIN, TIMEOUT_MAX))
    http_batch = int(_clamp(math.floor(timeout * TIMEOUT_SHARE / one_doc_sec), HTTP_BATCH_MIN, HTTP_BATCH_MAX))
    if timing is None:
        query_timeout = QUERY_TIMEOUT_MIN
    else:
        p95_sec = timing.p50_ms * QUERY_P95_OVER_P50 / 1000.0
        query_timeout = round(_clamp(p95_sec * QUERY_TIMEOUT_FACTOR, QUERY_TIMEOUT_MIN, QUERY_TIMEOUT_MAX), 1)
    ubatch, ub_why = decide_ubatch(
        hw, timing.placement if timing is not None else None,
        max_length=max_length, context_size=context_size if context_size is not None else max_length,
    )
    measured = (
        f"{timing.placement} p50 {timing.p50_ms:.0f} ms" if timing is not None
        else f"not measured, assumed {ASSUMED_MS_PER_TOKEN} ms/token"
    )
    reason = f"{measured}: {mpt:.2f} ms/token, doc {doc} tok; {ub_why}"
    if basis:
        reason = f"{reason}; {basis}"
    return TuneOutcome("ok", value={
        "batch_size": ubatch, "ubatch_size": ubatch, "http_batch": http_batch,
        "timeout": timeout, "query_timeout": query_timeout,
        "ms_per_token": round(mpt, 3), "placement": timing.placement if timing is not None else None,
    }, reason=reason)


def placement_timing(setting: Any, saved: Any, current_digest: str) -> EmbedTiming | None:
    """判別結果 (``EmbedPlacementResult`` / ``None``) から、今の配置の p50 を取り出す (純関数)。

    指紋が今の PC と違う結果は使わない。配置は ``embedding.gpu_layers`` が整数ならその値 (0 は CPU)、
    ``null`` は CPU、``auto`` は判別結果の配置。その配置の p50 が無ければ (GPU が無く CPU を測って
    いない等) ``None`` (保守側)。
    """
    if saved is None or saved.fingerprint != current_digest:
        return None
    if setting == "auto":
        placement = saved.placement
    elif setting is None:
        placement = "cpu"
    else:
        placement = "gpu" if int(setting) != 0 else "cpu"
    p50 = saved.gpu_p50_ms if placement == "gpu" else saved.cpu_p50_ms
    if p50 is None or p50 <= 0:
        return None
    return EmbedTiming(placement, float(p50))


def http_batch_of(backend: object, cap: int = HTTP_BATCH_MAX) -> int:
    """埋め込みバックエンドの解決済みの HTTP バッチ件数 (``http_batch_size()``) を ``cap`` 以下で返す。

    持たない (テストの簡易実装等) / 整数でなければ ``cap``。EvidenceStore と sleep-time が 1 回の
    ``embed()`` に渡す件数をこれで揃える。
    """
    fn = getattr(backend, "http_batch_size", None)
    value = fn() if callable(fn) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return min(value, cap)
    return cap


def is_auto_value(value: Any) -> bool:
    """config の値が ``auto`` (``null`` / キー無しを含む) か。"""
    return value is None or value == "auto"


def manual_fields(emb_cfg: dict[str, Any]) -> list[str]:
    """明示値を持つキー (調整値を反映しないもの)。"""
    return [k for k in CONFIG_FIELDS if not is_auto_value(emb_cfg.get(k))]


#: 決められないとき (確認待ち・材料なし) の保守側の値 = 測っていない仮定で既定の ``max_length`` を通した値。
#: ``batch_size`` / ``ubatch_size`` は :func:`effective_embed_params` が今の ``max_length`` 以上へ引き上げる。
FALLBACK: dict[str, Any] = dict(
    decide_embed_params(HardwareProfile(), None).value,  # type: ignore[arg-type]
)


def effective_embed_params(emb_cfg: dict[str, Any], tuned: Any) -> EmbedParams:
    """明示値と調整値 (``tuned``、無ければ :data:`FALLBACK`) を合わせた実効値 (純関数)。

    明示値はそのまま使う。調整値の ``batch_size`` / ``ubatch_size`` は今の ``max_length`` 以上へ
    引き上げる (保存時と ``max_length`` が違っても不変条件を守る)。明示値が ``ubatch < max_length`` /
    ``batch < ubatch`` なら ``warnings`` に英語の文を入れる (値は変えない)。
    """
    base = dict(FALLBACK)
    if isinstance(tuned, dict):
        base.update({k: v for k, v in tuned.items() if v is not None})
    max_length = int(emb_cfg.get("max_length") or DEFAULT_MAX_LENGTH)
    sources: dict[str, str] = {}
    picked: dict[str, Any] = {}
    for key in CONFIG_FIELDS:
        raw = emb_cfg.get(key)
        if is_auto_value(raw):
            picked[key], sources[key] = base[key], "tuned"
        else:
            picked[key], sources[key] = raw, "manual"
    sources["http_batch"] = "tuned"
    batch, ubatch = int(picked["batch_size"]), int(picked["ubatch_size"])
    if sources["ubatch_size"] == "tuned":
        ubatch = max(ubatch, ubatch_floor(max_length))
    if sources["batch_size"] == "tuned":
        batch = max(batch, ubatch)
    warnings: list[str] = []
    if ubatch < max_length:
        warnings.append(
            f"embedding.ubatch_size={ubatch} < max_length={max_length}: a single input of "
            f"{max_length} tokens makes llama-server return 500. Set ubatch_size to auto or >= max_length.",
        )
    if batch < ubatch:
        warnings.append(
            f"embedding.batch_size={batch} < ubatch_size={ubatch}: llama-server requires -b >= -ub. "
            "Set batch_size to auto or >= ubatch_size.",
        )
    http_batch = int(_clamp(int(base["http_batch"]), HTTP_BATCH_MIN, HTTP_BATCH_MAX))
    return EmbedParams(
        batch_size=batch, ubatch_size=ubatch, http_batch=http_batch,
        timeout=float(picked["timeout"]), query_timeout=float(picked["query_timeout"]),
        sources=sources, warnings=tuple(warnings),
    )


# ── 解決 (起動スクリプト / backend 共通の入口) ────────────────


def resolve_embed_params(
    cfg: dict[str, Any], project_root: Path | None, *, allow_decide: bool = False, use_cache: bool = True,
) -> EmbedParams:
    """``resolve_tuned`` の調整値と config の明示値を合わせた実効値。

    backend は ``allow_decide=False`` (保存値を読むだけ。無ければ保守側)。起動スクリプトは
    ``allow_decide=True`` (保存値 → その場の見積り (保存する) → 保守側)。
    """
    from backend.free.core.tuning.resolve import resolve_tuned

    emb_cfg = cfg.get("embedding") or {}
    resolved = resolve_tuned(
        cfg, KEY, project_root=project_root, allow_decide=allow_decide, use_cache=use_cache,
    )
    return effective_embed_params(emb_cfg, resolved.value)


# ── 項目 ──────────────────────────────────────────────────


def _saved_placement(cfg: dict[str, Any], project_root: Path) -> Any:
    """保存済みの埋め込みの配置の判別結果 (無い・読めなければ ``None``)。

    置き場は ``resolve_tuned`` と同じ解決 (``resolve.resolve_tune_paths``) で引く。
    """
    from backend.free.core.tuning.resolve import resolve_tune_paths
    from backend.free.rag.embed_placement import load_placement_result

    return load_placement_result(resolve_tune_paths(cfg, project_root).embed_placement)[0]


def _placement_tag(cfg: dict[str, Any], project_root: Path) -> str:
    """判別結果の印 (配置と判別した時刻)。判別し直したら見積り直す。"""
    saved = _saved_placement(cfg, project_root)
    return "none" if saved is None else f"{saved.placement}@{saved.decided_at}"


def _doc_tokens(cfg: dict[str, Any]) -> int:
    return int((cfg.get("rag") or {}).get("chunk_size") or DEFAULT_DOC_TOKENS)


def embed_basis(cfg: dict[str, Any], project_root: Path) -> str:
    """保存結果の前提の印 (:attr:`TuneSpec.basis`): max_length / 文脈長 / 文書長 / 配置設定 / 判別結果。"""
    emb = cfg.get("embedding") or {}
    max_length = int(emb.get("max_length") or DEFAULT_MAX_LENGTH)
    context = int(emb.get("context_size") or max_length)
    return (
        f"embed_basis[L={max_length},c={context},d={_doc_tokens(cfg)},"
        f"gl={emb.get('gpu_layers')},p={_placement_tag(cfg, project_root)}]"
    )


def run(ctx: TuneContext) -> TuneOutcome:
    """保存済みの判別結果 (この PC) と輪郭から調整値を見積もる。サーバは起こさない。"""
    from backend.free.rag.rerank_selftest import collect_pc_info

    emb = ctx.cfg.get("embedding") or {}
    if emb.get("backend", "llama-cpp") != "llama-cpp":
        return TuneOutcome("skipped", reason="embedding_backend_not_llama_cpp")
    hw = ctx.hardware()
    saved = _saved_placement(ctx.cfg, ctx.project_root)
    timing = placement_timing(emb.get("gpu_layers"), saved, collect_pc_info(hw.gpu_names).digest)
    max_length = int(emb.get("max_length") or DEFAULT_MAX_LENGTH)
    outcome = decide_embed_params(
        hw, timing, max_length=max_length, context_size=int(emb.get("context_size") or max_length),
        doc_tokens=_doc_tokens(ctx.cfg), basis=embed_basis(ctx.cfg, ctx.project_root),
    )
    manual = manual_fields(emb)
    if manual:
        # 明示値のキーは反映しない (他のキーと HTTP バッチ件数は反映するので項目は ok のまま)
        outcome = TuneOutcome("ok", value={**outcome.value, "manual": manual}, reason=outcome.reason)
    return outcome


SPEC = register(TuneSpec(
    key=KEY,
    config_key="",
    method="estimated",
    requires_restart=True,
    run=run,
    fallback=FALLBACK,
    depends_on=("embed_placement",),
    needs_servers=False,
    safe_while_running=True,  # 保存済みの判別結果を読むだけ
    basis=embed_basis,
    description="embedding batch / ubatch / HTTP batch / timeouts from the placement probe, c_16 §7.2.3",
))


__all__ = [
    "FALLBACK",
    "HTTP_BATCH_MAX",
    "HTTP_BATCH_MIN",
    "KEY",
    "EmbedParams",
    "EmbedTiming",
    "decide_embed_params",
    "decide_ubatch",
    "effective_embed_params",
    "embed_basis",
    "http_batch_of",
    "manual_fields",
    "placement_timing",
    "resolve_embed_params",
    "ubatch_floor",
]
