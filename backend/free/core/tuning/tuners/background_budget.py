"""項目 ``background_budget`` — sleep-time の 1 サイクルあたりの件数 (c_16 §7.2.3)。

sleep-time の各段 (疑似クエリ・要約・競合解決・ノート進化・curator) の件数は、
「27B で 1 件 20 秒級」という開発機の実測を前提にした固定値だった。CPU・低スペックでは 1 サイクルが
数時間に伸びて Full が終わらず、高スペックでは件数を使い切らずに溜まる。そこで実測の decode tps
(:mod:`backend.free.llm.tps_calibration`) を基準の速さ (:data:`REFERENCE_DECODE_TPS`) と比べた倍率
``scale`` (:data:`MIN_SCALE` 〜 :data:`MAX_SCALE`) を各件数の既定値に掛ける。

- 未測定 (:data:`~backend.free.llm.tps_calibration.MIN_SAMPLES` に届いていない) なら ``scale = 1.0``
  で、件数は現行の既定のまま (挙動不変)。
- 掛けるのは **利用者が schema の既定から変えていない値だけ** (既定と違う値は明示 = ``manual`` と
  みなして不変。``tps_calibration.is_explicit_agent_timeout`` と同じ流儀)。
- backend (sleep-time) は **サイクルの開始時** に :func:`background_scale` で
  プロセスの実測の台帳から引き直す (tps は走りながら貯まるので起動時 1 回にしない)。測りも書きもしない。
- 学習 (Level 2 の SPSA 反復数・評価件数) は **伸縮しない**。LoRA の学習・評価の所要時間は decode tps と
  比例せず、サイクルごとに評価件数が変わると候補間・サイクル間の評価の比較可能性が下がる。

この項目の ``run`` は画面 / CLI 向けの要約で、保存済みの実測 (``cache/tps_calibration.json``) から
``{scale, decode_tps, samples}`` を返すだけ (サーバを起こさない)。
"""

from __future__ import annotations

import math
from typing import Any

from backend.free.core.tuning.items import TuneContext, TuneOutcome, TuneSpec, config_value, register

KEY = "background_budget"

#: 現行の件数の既定が前提にした decode の速さ (tok/s)。既定は「27B で 1 件 20 秒級」「1 問 ≈ 7 秒」
#: (``rag.pseudo_query`` のコメント) の開発機で決めた値で、その機の 27B の decode は単独で
#: 200〜218 ms/tok (2026-09-03 監査、``tps_calibration.TIMEOUT_HEADROOM`` の根拠と同じ実測)
#: ≈ 4.6〜5.0 tok/s。1 問 ≈ 35 トークンを 7 秒で出す速さとも合う。
REFERENCE_DECODE_TPS = 5.0
#: 倍率の下限 / 上限。遅い PC でも 1 サイクルに 1/4 は進め、速い PC でも 2 倍までに留める
#: (件数を増やすと 1 サイクルの壁時計とチャットへの横取りの機会が比例して増えるため)。
MIN_SCALE = 0.25
MAX_SCALE = 2.0


# ── 判定 (純関数) ─────────────────────────────────────────


def scale_for_decode_tps(decode_tps: float | None) -> float:
    """実測の decode tps から件数の倍率を求める。未測定 (``None`` / 0 以下) は ``1.0``。"""
    if decode_tps is None or not decode_tps > 0:
        return 1.0
    return min(MAX_SCALE, max(MIN_SCALE, decode_tps / REFERENCE_DECODE_TPS))


def scaled_count(default: int, scale: float, minimum: int = 1) -> int:
    """件数の既定 ``default`` に ``scale`` を掛けて四捨五入する (``minimum`` を下回らない)。

    ``default`` が 0 以下 (「その段を止める」の意味) はそのまま返す。
    """
    if default <= 0:
        return default
    return max(minimum, math.floor(default * scale + 0.5))


def scaled_config_keys() -> tuple[tuple[str, int], ...]:
    """伸縮する config のキー (ドット区切り) と schema の既定値。

    壁時計の予算 (``rag.pseudo_query.budget_seconds``) は伸縮しない — 時間で畳む段は件数が縮まなくても
    時間で止まるので、tps で予算まで縮めると遅い PC で二重に削れる。学習 (``learning.*``) も対象外
    (モジュールの説明を参照)。
    """
    from backend.schemas._common import HistoryConfig
    from backend.schemas.memory import ConflictResolverConfig, MemoryConfig
    from backend.schemas.rag import PseudoQueryConfig

    def default(model: Any, name: str) -> int:
        return int(model.model_fields[name].default)

    return (
        ("rag.pseudo_query.max_per_cycle", default(PseudoQueryConfig, "max_per_cycle")),
        ("rag.pseudo_query.backfill_per_cycle", default(PseudoQueryConfig, "backfill_per_cycle")),
        ("history.summary_batch_size", default(HistoryConfig, "summary_batch_size")),
        ("memory.conflict_batch_size", default(MemoryConfig, "conflict_batch_size")),
        ("memory.conflict_resolver.max_per_cycle", default(ConflictResolverConfig, "max_per_cycle")),
        ("memory.note_evolution_batch", default(MemoryConfig, "note_evolution_batch")),
    )


def is_explicit_count(cfg: dict[str, Any], dotted: str, default: int) -> bool:
    """``dotted`` が schema の既定から変えられているか (変えていれば伸縮しない)。

    検証後の config は既定で埋まっているので「書いたか」は区別できない。既定と違う値を利用者の明示と
    みなす。キーが無ければ既定扱い。
    """
    raw = config_value(cfg, dotted, None)
    if raw is None:
        return False
    try:
        return int(raw) != default
    except (TypeError, ValueError):
        return True


def manual_keys(cfg: dict[str, Any]) -> list[str]:
    """伸縮の対象のうち、明示値のため伸縮しないキー。"""
    return [k for k, d in scaled_config_keys() if is_explicit_count(cfg, k, d)]


def scaled_background_config(cfg: dict[str, Any], scale: float) -> dict[str, Any]:
    """既定のままの件数に ``scale`` を掛けた config を返す (元の ``cfg`` は書き換えない)。

    ``scale == 1.0`` なら ``cfg`` そのものを返す (未測定の間は挙動不変)。書き換えるのは値を入れる
    経路の dict だけで、他の節は元と共有する。
    """
    if scale == 1.0:
        return cfg
    out = dict(cfg)
    copied = {id(out)}
    for dotted, default in scaled_config_keys():
        if is_explicit_count(cfg, dotted, default):
            continue
        *parents, leaf = dotted.split(".")
        node = out
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict) or id(child) not in copied:
                child = dict(child) if isinstance(child, dict) else {}
                copied.add(id(child))
                node[part] = child
            node = child
        node[leaf] = scaled_count(default, scale)
    return out


# ── 読み取り (backend) ────────────────────────────────────


def live_decode_tps(client: Any = None) -> float | None:
    """プロセスの実測の台帳から decode tps を引く (未測定 / 台帳が未設定なら ``None``)。

    ``client`` (``AuxClient`` / ``LocalClient``) を渡せば、それが今話しているモデルの実測を引く。
    無ければ active の base モデルの実測。backend は測らず、読むだけ。
    """
    from backend.free.llm.tps_calibration import get_tps_tracker, measured_tps_of

    for candidate in (client, getattr(client, "local", None)):
        estimate = measured_tps_of(candidate)
        if estimate is not None and estimate.decode_tps is not None:
            return estimate.decode_tps
    tracker = get_tps_tracker()
    if tracker is None:
        return None
    try:
        from backend.config import get_path_resolver

        model_key = get_path_resolver().active_model_key
    except Exception:  # noqa: BLE001 - config 未ロード等。件数は現行の既定のまま
        return None
    estimate = tracker.estimate(model_key)
    return estimate.decode_tps if estimate is not None else None


def background_scale(client: Any = None) -> float:
    """サイクルの開始時に引く件数の倍率 (未測定なら ``1.0``)。"""
    return scale_for_decode_tps(live_decode_tps(client))


# ── 項目 ──────────────────────────────────────────────────


def run(ctx: TuneContext) -> TuneOutcome:
    """保存済みの実測 (この PC・base モデル) から倍率を求める。"""
    from backend.config import PathResolver
    from backend.free.llm.tps_calibration import MIN_SAMPLES, estimate_of, load_tps_calibration
    from backend.free.rag.rerank_selftest import collect_pc_info, pc_mismatch

    resolver = PathResolver(ctx.cfg, ctx.project_root)
    base_model = config_value(ctx.cfg, "model_paths.base_model", "")
    model_key = resolver.model_key_for(base_model) if base_model else ""
    record = load_tps_calibration(resolver.resolve_local("tps_calibration_file"))
    entry = None
    if record is not None and not pc_mismatch(record.pc, collect_pc_info(ctx.hardware().gpu_names)):
        entry = record.models.get(model_key)
    estimate = estimate_of(entry)
    decode = estimate.decode_tps if estimate is not None else None
    value: dict[str, Any] = {
        "scale": round(scale_for_decode_tps(decode), 3),
        "decode_tps": round(decode, 2) if decode is not None else None,
        "samples": {"decode": entry.decode_samples if entry is not None else 0, "min": MIN_SAMPLES},
    }
    manual = manual_keys(ctx.cfg)
    if manual:
        # 明示値の件数だけ伸縮しない (他は伸縮するので項目は反映のまま)
        value["manual"] = manual
    return TuneOutcome("ok", value=value, reason="measured" if decode is not None else "no_samples")


SPEC = register(TuneSpec(
    key=KEY,
    config_key="",
    method="estimated",
    requires_restart=False,
    run=run,
    needs_servers=False,
    safe_while_running=True,  # 保存済みの実測を読むだけ
    description="per-cycle counts of sleep-time scaled by the measured decode tok/s, c_16 §7.2.3",
))
