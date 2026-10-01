"""埋め込みサーバの配置 (GPU / CPU) の判別 (c_16 §7.2.2) — PC が変わったときだけ測る。

``embedding.gpu_layers: auto`` のとき、GPU が候補 (空き・予算が足りる) なら起動スクリプトが一時ポートで
埋め込みサーバを CPU (``-ngl 0``) と GPU (``-ngl 999``) で起こし、既知のクエリを送って

1. ベクトルが有限・非ゼロ・次元一致か
2. 同じ文の CPU と GPU のベクトルの cosine の最小 ≥ :data:`MIN_COSINE` か (既存ストアと混ぜて困らないか)
3. 1 要求の時間の中央値 (p50) が GPU < CPU か

を見て配置を決め、``cache/embed_placement.json`` (derived・keep_on_reset、c_05 §0.7.1) に残す。
判定は純関数 (:func:`check_vectors` / :func:`decide_embed_placement` / :func:`placement_needed` /
:func:`resolve_embed_placement_status`) で、HTTP は :func:`measure_embed_server` の薄い層だけ。

**PC の指紋はリランカーの自己テストと同じもの** (:func:`backend.free.rag.rerank_selftest.collect_pc_info`)
を使う。llama.cpp の版は記録だけする。埋め込みモデルの ``model_key`` は指紋に入れないが、保存結果と
違えば判別し直す (埋め込みは必須機能で、モデルで VRAM の要りようが変わるため。リランカーとの違い)。
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from backend.free.rag.rerank_selftest import PcInfo, pc_mismatch
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger

logger = get_logger("rag.embed_placement")

#: 同じ文の CPU と GPU のベクトルの cosine の最小がこれ以上なら「同じ埋め込み」とみなす。
MIN_COSINE = 0.999
#: 暖機の巡回数 (計測に入れない)。
WARM_ROUNDS = 1
#: 計測の巡回数 (1 巡 = 既知のクエリを 1 本ずつ全部送る)。
MEASURE_ROUNDS = 5
#: 1 要求の HTTP timeout (秒)。
REQUEST_TIMEOUT_SEC = 30.0
#: 環境起因の失敗 (次の起動でまた判別する) の理由。
SERVER_UNHEALTHY = "server_unhealthy"
#: GPU の一時サーバが起動しない (``server_unhealthy``) のがこの回数続いたら、PC の性質として
#: ``gpu_failed:server_unhealthy`` を保存する (それまでは回数だけ残して次の起動でまた判別する)。
GPU_UNHEALTHY_LIMIT = 3
#: GPU の一時サーバの起動失敗を数えている途中の記録の理由 (保存するが、次の起動でまた判別する)。
GPU_UNHEALTHY_RETRY = "gpu_unhealthy_retry"
#: GPU 候補外の理由のうち、PC の性質として保存するもの (空きの不足・予算は状態なので保存しない)。
SAVED_NON_CANDIDATE_REASONS = frozenset({"no_gpu_device"})

EmbedPlacementKind = Literal["gpu", "cpu"]

EMBEDDINGS_PATH = "/v1/embeddings"

#: 既知のクエリ (短・中・長)。チャット応答パスの単一クエリ埋め込みに寄せる。
PROBE_TEXTS: tuple[str, ...] = (
    "明日の会議は何時からですか？",
    "Python の asyncio で複数のコルーチンを並行に動かし、全部の結果がそろうまで待つにはどうすればよいですか？",
    "先週まとめた旅行の計画について、宿の候補を三つに絞った理由と、それぞれの予算、移動時間、"
    "子ども連れで気をつける点をもう一度整理して、最終的にどこを予約するのがよいか意見をください。",
)


# ── 判定 (純関数) ─────────────────────────────────────────


def check_vectors(vectors: Sequence[Sequence[float]], *, expected_dim: int, n_texts: int) -> str:
    """ベクトル列の健全性。問題が無ければ空文字列、あれば理由。

    理由: ``count_mismatch`` / ``dim_mismatch`` / ``non_finite`` / ``zero_vector``。
    """
    if len(vectors) != n_texts:
        return "count_mismatch"
    for vec in vectors:
        if len(vec) != expected_dim:
            return "dim_mismatch"
        if not all(math.isfinite(x) for x in vec):
            return "non_finite"
        if not any(x != 0.0 for x in vec):
            return "zero_vector"
    return ""


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """2 本のベクトルの cosine (どちらかがゼロなら 0.0)。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def min_cosine(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """同じ位置の組の cosine の最小。"""
    if not left or len(left) != len(right):
        raise ValueError("vector lists must be non-empty and of the same length")
    return min(cosine(a, b) for a, b in zip(left, right, strict=True))


@dataclass
class EmbedProbe:
    """1 つの配置で測った生の結果 (HTTP 層の出力)。"""

    #: 最後の巡のベクトル (:data:`PROBE_TEXTS` の順)。
    vectors: list[list[float]] = field(default_factory=list)
    timings_ms: list[float] = field(default_factory=list)
    #: 失敗の理由 (``server_unhealthy`` / ``http_500`` / ``http_error:*`` / ``bad_response``)。成功なら空。
    error: str = ""

    @property
    def p50_ms(self) -> float | None:
        return statistics.median(self.timings_ms) if self.timings_ms else None


@dataclass(frozen=True)
class PlacementDecision:
    """判別の結論。``save`` が偽なら環境起因の失敗で、保存せずこの回だけ CPU で起こす。"""

    placement: EmbedPlacementKind
    gpu_layers: int
    reason: str
    save: bool
    cpu_p50_ms: float | None = None
    gpu_p50_ms: float | None = None
    cosine_min: float | None = None
    #: GPU の一時サーバが続けて起動しなかった回数 (この回を含む)。
    gpu_unhealthy_streak: int = 0


def is_environmental_failure(reason: str) -> bool:
    """環境起因の失敗 (PC の性質ではない) か。保存せず次の起動でまた判別する。"""
    return reason == SERVER_UNHEALTHY or reason.startswith("http_error:")


def decide_embed_placement(
    cpu: EmbedProbe | None,
    gpu: EmbedProbe | None,
    *,
    gpu_candidate_reason: str,
    expected_dim: int,
    n_texts: int = len(PROBE_TEXTS),
    min_cos: float = MIN_COSINE,
    prior_unhealthy_streak: int = 0,
) -> PlacementDecision:
    """CPU と GPU の測定から配置を決める。

    ``gpu`` が ``None`` なら GPU は候補外 (``gpu_candidate_reason`` がその理由) で、CPU は測らなくて
    よい (``cpu=None``)。保存するのは PC の性質 (``no_gpu_device``) だけで、空きの不足や予算超過は
    状態なので保存しない。
    GPU を採るのは、GPU が成功・ベクトル健全・CPU との cosine の最小 ≥ ``min_cos``・p50 が CPU より
    小さいときだけ。CPU 側の失敗は比べる基準が無いので保存しない (この回は CPU)。
    GPU の一時サーバが起動しない (``server_unhealthy``) のは ``prior_unhealthy_streak`` + 1 回目として
    数え、:data:`GPU_UNHEALTHY_LIMIT` 回に届いたら ``gpu_failed:server_unhealthy`` を保存し、
    それまでは :data:`GPU_UNHEALTHY_RETRY` (回数だけ残して次の起動でまた判別) にする。
    """
    if gpu is None:
        return PlacementDecision(
            "cpu", 0, gpu_candidate_reason,
            save=gpu_candidate_reason in SAVED_NON_CANDIDATE_REASONS,
            cpu_p50_ms=cpu.p50_ms if cpu is not None else None,
        )
    if cpu is None:
        raise ValueError("the CPU probe is required when the GPU was measured")
    cpu_bad = cpu.error or check_vectors(cpu.vectors, expected_dim=expected_dim, n_texts=n_texts)
    if cpu_bad:
        return PlacementDecision("cpu", 0, f"cpu_probe_failed:{cpu_bad}", save=False)
    cpu_p50 = cpu.p50_ms
    if gpu.error == SERVER_UNHEALTHY:
        streak = max(0, prior_unhealthy_streak) + 1
        final = streak >= GPU_UNHEALTHY_LIMIT
        return PlacementDecision(
            "cpu", 0, f"gpu_failed:{SERVER_UNHEALTHY}" if final else GPU_UNHEALTHY_RETRY,
            save=True, cpu_p50_ms=cpu_p50, gpu_unhealthy_streak=streak,
        )
    if gpu.error:
        return PlacementDecision(
            "cpu", 0, f"gpu_failed:{gpu.error}",
            save=not is_environmental_failure(gpu.error), cpu_p50_ms=cpu_p50,
        )
    gpu_bad = check_vectors(gpu.vectors, expected_dim=expected_dim, n_texts=n_texts)
    gpu_p50 = gpu.p50_ms
    if gpu_bad:
        return PlacementDecision(
            "cpu", 0, f"gpu_bad_vectors:{gpu_bad}", save=True, cpu_p50_ms=cpu_p50, gpu_p50_ms=gpu_p50,
        )
    cos = min_cosine(cpu.vectors, gpu.vectors)
    common = {"cpu_p50_ms": cpu_p50, "gpu_p50_ms": gpu_p50, "cosine_min": cos}
    if cos < min_cos:
        return PlacementDecision("cpu", 0, "cosine_mismatch", save=True, **common)
    if cpu_p50 is None or gpu_p50 is None or gpu_p50 >= cpu_p50:
        return PlacementDecision("cpu", 0, "gpu_not_faster", save=True, **common)
    return PlacementDecision("gpu", 999, "gpu_faster", save=True, **common)


# ── 結果の永続化 ──────────────────────────────────────────


@persisted()
@dataclass
class EmbedPlacementResult:
    """判別 1 回の結果 (``cache/embed_placement.json`` の payload)。"""

    fingerprint: str
    placement: EmbedPlacementKind = "cpu"
    gpu_layers: int = 0
    #: 判別の理由 (``gpu_faster`` / ``gpu_not_faster`` / ``cosine_mismatch`` / ``no_gpu_device`` 等)。
    reason: str = ""
    cpu_p50_ms: float | None = None
    gpu_p50_ms: float | None = None
    cosine_min: float | None = None
    dim: int = 0
    decided_at: str = ""
    #: GPU の一時サーバが続けて起動しなかった回数 (:data:`GPU_UNHEALTHY_RETRY` の間だけ増える)。
    gpu_unhealthy_streak: int = 0
    #: 判別した埋め込みモデル (指紋には入れないが、違えば判別し直す)。
    model_key: str = ""
    #: 記録だけ。
    model_file: str = ""
    llama_server_version: str = ""
    pc: PcInfo = field(default_factory=PcInfo)
    _extra: dict[str, Any] | None = None


#: 形式 (c_05 §0.7.1)。PC 固有の測定値で、測り直せる (derived) が高価 (一時サーバを 2 回起こす)
#: ので ``evoref reset`` でも残す (``--include-cache`` まで)。
EMBED_PLACEMENT_FORMAT = register_format(FormatSpec(
    format_id="cache.embed_placement",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="cache/embed_placement.json",
    retention=(
        "one record, rewritten under embedding.gpu_layers: auto when the PC fingerprint or the "
        "embedding model_key changes (or while a GPU start failure is being retried)"
    ),
    export=False,
    keep_on_reset=True,
    records=(EmbedPlacementResult,),
))


class EmbedPlacementFile(VersionedJsonFile):
    """``cache/embed_placement.json`` の読み書き (derived: 読めなければ捨てて判別し直す)。"""

    FORMAT = EMBED_PLACEMENT_FORMAT
    _state_logger = logger

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.result: EmbedPlacementResult | None = None

    def _to_payload(self) -> Any:
        if self.result is None:
            raise ValueError("no embed placement result to save")
        return codec_for(EmbedPlacementResult).encode(self.result)

    def _from_payload(self, payload: Any) -> None:
        self.result = codec_for(EmbedPlacementResult).decode(payload)


def load_placement_result(path: Path | str) -> tuple[EmbedPlacementResult | None, str]:
    """保存済みの結果と、読んだときの分類 (``absent`` / ``current`` / ``corrupt`` 等)。"""
    store = EmbedPlacementFile(path)
    if store.load():
        return store.result, str(store.last_status or "current")
    return None, str(store.last_status or "absent")


def save_placement_result(path: Path | str, result: EmbedPlacementResult) -> bool:
    """結果を書く (AtomicWriter)。書けたら ``True``。"""
    store = EmbedPlacementFile(path)
    store.result = result
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return store.save()


def model_changed(saved: EmbedPlacementResult, model_key: str | None) -> bool:
    """保存結果の埋め込みモデルと ``model_key`` が違うか (どちらかが不明なら違わないとみなす)。"""
    return bool(model_key and saved.model_key and model_key != saved.model_key)


def placement_needed(
    saved: EmbedPlacementResult | None,
    fingerprint: str,
    *,
    force: bool = False,
    model_key: str | None = None,
) -> tuple[bool, str]:
    """判別するか、とその理由。

    理由: ``forced`` / ``no_result`` / ``fingerprint_changed`` / ``model_changed`` /
    ``gpu_retry_pending``。読めない結果 (新しい版・壊れ) は ``saved=None`` で渡る (``no_result``)。
    埋め込みモデルが変わったら判別し直す — リランカー (WARNING だけ) と違い、埋め込みは必須機能で、
    モデルで VRAM の要りようが大きく変わりうるため (c_16 §7.2.2)。
    """
    if force:
        return True, "forced"
    if saved is None:
        return True, "no_result"
    if saved.fingerprint != fingerprint:
        return True, "fingerprint_changed"
    if model_changed(saved, model_key):
        return True, "model_changed"
    if saved.reason == GPU_UNHEALTHY_RETRY:
        return True, "gpu_retry_pending"
    return False, ""


def prior_unhealthy_streak(
    saved: EmbedPlacementResult | None, fingerprint: str, model_key: str | None,
) -> int:
    """続けて数える GPU の起動失敗の回数 (同じ PC・同じモデルの再試行中の記録があるときだけ)。"""
    if (
        saved is None or saved.reason != GPU_UNHEALTHY_RETRY or saved.fingerprint != fingerprint
        or model_changed(saved, model_key)
    ):
        return 0
    return saved.gpu_unhealthy_streak


# ── 設定と保存結果から配置を決める (起動の全経路と backend で共有) ──


@dataclass(frozen=True)
class EmbedPlacementStatus:
    """埋め込みサーバの配置 (``/api/status`` の ``embed_placement``)。

    ``setting`` は ``auto`` / ``explicit`` (整数) / ``default`` (``null``)。
    """

    setting: str = "default"
    placement: EmbedPlacementKind = "cpu"
    gpu_layers: int = 0
    #: 理由 (``explicit_gpu_layers`` / ``default_cpu`` / ``not_decided`` / ``stale_fingerprint`` /
    #: ``model_changed`` / 判別の理由)。
    reason: str = "default_cpu"
    decided_at: str | None = None
    cpu_p50_ms: float | None = None
    gpu_p50_ms: float | None = None
    cosine_min: float | None = None


def gpu_layers_setting(raw: object) -> Literal["auto"] | int | None:
    """``embedding.gpu_layers`` の生の値を ``"auto"`` / 整数 / ``None`` に正規化する。

    スキーマ (``Literal["auto"]``) と同じく ``auto`` は小文字の完全一致だけ。他の文字列は ``ValueError``。
    """
    if raw is None:
        return None
    if raw == "auto":
        return "auto"
    return int(raw)  # type: ignore[call-overload]


def resolve_embed_placement_status(
    setting: object,
    saved: EmbedPlacementResult | None,
    *,
    current_pc: PcInfo | None = None,
    current_model_key: str | None = None,
) -> EmbedPlacementStatus:
    """設定と保存結果から配置を決める (純関数、subprocess も HTTP も使わない)。

    - 整数 → その値 (0 は CPU、それ以外は GPU)。``null`` → CPU (従来の既定)。保存結果は見ない。
    - ``auto`` で結果が無い / 読めない → CPU (``not_decided``)。
    - ``auto`` で ``current_pc`` と保存時の PC が食い違う → CPU (``stale_fingerprint``。GPU 名は比べない)。
    - ``auto`` で ``current_model_key`` と保存時のモデルが違う → CPU (``model_changed``。持ち主の
      プロセスの次の起動で判別し直す)。
    - それ以外は保存結果の配置。
    """
    value = gpu_layers_setting(setting)
    if value is None:
        return EmbedPlacementStatus()
    if value != "auto":
        return EmbedPlacementStatus(
            setting="explicit", placement="gpu" if value != 0 else "cpu",
            gpu_layers=value, reason="explicit_gpu_layers",
        )
    if saved is None:
        return EmbedPlacementStatus(setting="auto", reason="not_decided")
    common = {
        "decided_at": saved.decided_at or None, "cpu_p50_ms": saved.cpu_p50_ms,
        "gpu_p50_ms": saved.gpu_p50_ms, "cosine_min": saved.cosine_min,
    }
    if current_pc is not None and pc_mismatch(saved.pc, current_pc):
        return EmbedPlacementStatus(setting="auto", reason="stale_fingerprint", **common)
    if model_changed(saved, current_model_key):
        return EmbedPlacementStatus(setting="auto", reason="model_changed", **common)
    return EmbedPlacementStatus(
        setting="auto", placement=saved.placement, gpu_layers=saved.gpu_layers,
        reason=saved.reason, **common,
    )


# ── HTTP (薄い層) ─────────────────────────────────────────


def _parse_embeddings(body: Any, n: int) -> list[list[float]]:
    """``/v1/embeddings`` の応答から ``index`` 順のベクトル列を取り出す。形が違えば ``ValueError``。"""
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != n:
        raise ValueError("unexpected embeddings response")
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    return [[float(x) for x in item["embedding"]] for item in ordered]


def measure_embed_server(
    base_url: str,
    *,
    post: Callable[..., httpx.Response] = httpx.post,
    clock: Callable[[], float] = time.perf_counter,
    texts: Sequence[str] = PROBE_TEXTS,
    warm_rounds: int = WARM_ROUNDS,
    measure_rounds: int = MEASURE_ROUNDS,
    timeout: float = REQUEST_TIMEOUT_SEC,
) -> EmbedProbe:
    """起動済みの埋め込みサーバへ ``texts`` を 1 本ずつ ``warm_rounds + measure_rounds`` 巡送る。

    再試行しない (測定なので失敗は隠さない)。失敗した時点で ``error`` を入れて返す。
    ベクトルは最後の巡の値。
    """
    url = f"{base_url}{EMBEDDINGS_PATH}"
    result = EmbedProbe()
    for rnd in range(warm_rounds + measure_rounds):
        vectors: list[list[float]] = []
        for text in texts:
            started = clock()
            try:
                resp = post(url, json={"input": [text]}, timeout=timeout)
                resp.raise_for_status()
                vec = _parse_embeddings(resp.json(), 1)[0]
            except httpx.HTTPStatusError as e:
                result.error = f"http_{e.response.status_code}"
                return result
            except httpx.HTTPError as e:
                result.error = f"http_error:{type(e).__name__}"
                return result
            except (ValueError, KeyError, TypeError):
                result.error = "bad_response"
                return result
            elapsed_ms = (clock() - started) * 1000.0
            vectors.append(vec)
            if rnd >= warm_rounds:
                result.timings_ms.append(elapsed_ms)
        result.vectors = vectors
    return result


__all__ = [
    "EMBED_PLACEMENT_FORMAT",
    "GPU_UNHEALTHY_LIMIT",
    "GPU_UNHEALTHY_RETRY",
    "MIN_COSINE",
    "PROBE_TEXTS",
    "SERVER_UNHEALTHY",
    "EmbedPlacementFile",
    "EmbedPlacementResult",
    "EmbedPlacementStatus",
    "EmbedProbe",
    "PlacementDecision",
    "check_vectors",
    "cosine",
    "decide_embed_placement",
    "gpu_layers_setting",
    "is_environmental_failure",
    "load_placement_result",
    "measure_embed_server",
    "min_cosine",
    "model_changed",
    "placement_needed",
    "prior_unhealthy_streak",
    "resolve_embed_placement_status",
    "save_placement_result",
]
