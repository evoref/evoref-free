"""`/api/learning` ハンドラから抽出した純粋な収集ロジック

`backend/free/api/learning.py` の `_build_scheduler_status` ハンドラ (100 行)
内に直書きされていた以下のロジックを抽出した純粋関数群:
- raw dict → `Level1ResultEntry` マッピング
- `_executed_phases` 抽出 (raw_l1 から取り出し)
- raw dict → `PolicyEvolverDomainStatus` マッピング
- raw dict → `FitnessPoint` 履歴マッピング
- raw dict → `PriorityRequestEntry` マッピング
- raw dict → `ActiveSessionInfo` マッピング
- mode_data → `ExperienceByModeModel` マッピング

レイヤー責務:
- `learning.py` (API 層)            — HTTP / Pydantic / scheduler 取得 / Pro/Free ガード
- `_learning_collectors.py` (helper) — 純粋なマッピング (raw dict → Pydantic)

すべて引数のみに依存し、FastAPI / app_state / グローバル設定 / scheduler オブ
ジェクト / I/O に触れない pure 関数。Pro/Free ガードは呼び出し側 (handler) の
責務として、helper 自体は edition-agnostic。
"""

from __future__ import annotations

from backend.free.api.learning._learning_schemas import (
    ActiveSessionInfo,
    ExperienceByModeModel,
    FitnessPoint,
    Level1ResultEntry,
    Level2GatesModel,
    Level2StatusModel,
    Level2TargetStatus,
    PolicyEvolverDomainStatus,
    PriorityRequestEntry,
)


# ── タイムスタンプ整形 ──────────────────────────────────────────────────


def ts_to_iso(ts: float | None) -> str | None:
    """float タイムスタンプ (epoch 秒) を `YYYY-MM-DDTHH:MM:SSZ` 形式に変換。

    `None` / 0 以下の値は `None` を返す。
    """
    import time as _time

    if ts is None or ts <= 0:
        return None
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(ts))


def latest_level2_run(raw_last_level2_run: object) -> float:
    """target 別 dict (``{"base": ts, "assist": ts}``) から最新の実行時刻を取る。

    ``LearningScheduler.get_status()`` の ``last_level2_run`` は 2026-07-18 の
    修正で単一 float から target 別 dict へ変わった。API 層はまだ単一の
    「最終 Level 2 実行時刻」表示のままなので、target 別の最大値に潰して返す
    (旧フォーマットの単一 float / 未設定値との後方互換も保つ)。
    """
    if isinstance(raw_last_level2_run, dict):
        values = [v for v in raw_last_level2_run.values() if isinstance(v, (int, float))]
        return max(values) if values else 0.0
    if isinstance(raw_last_level2_run, (int, float)):
        return float(raw_last_level2_run)
    return 0.0


# ── Level 1 結果マッピング ────────────────────────────────────


def map_level1_results(raw_l1: dict | None) -> dict[str, Level1ResultEntry]:
    """raw `last_level1_results` から `_executed_phases` を除外した上で
    `Level1ResultEntry` マップを構築する純粋関数。

    `raw_l1` が None / 空の場合は空 dict。dict でないキー値はスキップする。
    元 handler は破壊的に `pop("_executed_phases")` していたが、本関数は
    非破壊で `_executed_phases` を無視する。
    """
    if not raw_l1:
        return {}
    result: dict[str, Level1ResultEntry] = {}
    for key, val in raw_l1.items():
        if key == "_executed_phases":
            continue
        if not isinstance(val, dict):
            continue
        result[key] = Level1ResultEntry(
            improved=val.get("improved", False),
            fitness_before=val.get("fitness_before"),
            fitness_after=val.get("fitness_after"),
        )
    return result


def extract_executed_phases(raw_l1: dict | None) -> list[str]:
    """raw `last_level1_results` から `_executed_phases` リストを取り出す
    非破壊な純粋関数。値が list でない / 欠損なら空リスト。
    """
    if not raw_l1:
        return []
    phases = raw_l1.get("_executed_phases")
    if not isinstance(phases, list):
        return []
    return list(phases)


# ── Pro: PolicyEvolver 状態マッピング ────────────────────────


def map_policy_evolver_status(
    pro_raw: dict | None,
) -> dict[str, PolicyEvolverDomainStatus]:
    """Pro の `get_pro_status()` 結果から `policy_evolver` 配下の dict を
    `PolicyEvolverDomainStatus` のマップに変換する純粋関数。

    Pro/Free 判定は呼び出し側の責務。`pro_raw` が None / `policy_evolver`
    キー欠損なら空 dict。
    """
    if not pro_raw:
        return {}
    raw_evolvers = pro_raw.get("policy_evolver", {}) or {}
    if not isinstance(raw_evolvers, dict):
        return {}
    result: dict[str, PolicyEvolverDomainStatus] = {}
    for key, val in raw_evolvers.items():
        if not isinstance(val, dict):
            continue
        result[key] = PolicyEvolverDomainStatus(
            current_fitness=val.get("current_fitness"),
            best_fitness=val.get("best_fitness", 0.0),
            decline_count=val.get("decline_count", 0),
            sigma=val.get("sigma", 0.0),
            phase=val.get("phase", ""),
        )
    return result


# ── Pro: Level 2 (base/assist) 状態マッピング ────────────────


def _map_level2_target(raw: object) -> Level2TargetStatus:
    """raw target dict を `Level2TargetStatus` に変換する純粋関数。"""
    if not isinstance(raw, dict):
        return Level2TargetStatus()
    return Level2TargetStatus(
        method=str(raw.get("method", "")),
        bootstrap_enabled=bool(raw.get("bootstrap_enabled", False)),
        adapter_exists=bool(raw.get("adapter_exists", False)),
        version=int(raw.get("version", 0) or 0),
        experiences_current=int(raw.get("experiences_current", 0) or 0),
        bootstrap_min=int(raw.get("bootstrap_min", 0) or 0),
        spsa_min=int(raw.get("spsa_min", 0) or 0),
        cvector_min=int(raw.get("cvector_min", 0) or 0),
        block_reason=str(raw.get("block_reason", "") or ""),
    )


def map_level2_status(raw: object | None) -> Level2StatusModel | None:
    """raw `level2` dict を `Level2StatusModel` に変換する純粋関数。

    `raw` が dict でない場合 (Free / Level2Runner 未注入) は `None`。
    Pro/Free 判定は呼び出し側ではなく raw の有無で吸収する。
    """
    if not isinstance(raw, dict):
        return None
    raw_gates = raw.get("gates", {})
    if isinstance(raw_gates, dict):
        gates = Level2GatesModel(
            active_minutes=float(raw_gates.get("active_minutes", 5.0) or 5.0),
            overdue_hours=float(raw_gates.get("overdue_hours", 24.0) or 24.0),
            recheck_interval_sec=float(
                raw_gates.get("recheck_interval_sec", 300.0) or 300.0,
            ),
        )
    else:
        gates = Level2GatesModel()
    return Level2StatusModel(
        running_target=raw.get("running_target"),
        next_target=str(raw.get("next_target", "base")),
        base=_map_level2_target(raw.get("base")),
        assist=_map_level2_target(raw.get("assist")),
        gates=gates,
    )


# ── Fitness 履歴マッピング ──────────────────────────────────────────────


def map_fitness_history(
    raw_history: dict | None,
) -> dict[str, list[FitnessPoint]]:
    """raw fitness 履歴を `{key: [FitnessPoint, ...]}` に変換する純粋関数。

    各ポイントが dict でない場合はスキップする。
    """
    if not raw_history:
        return {}
    result: dict[str, list[FitnessPoint]] = {}
    for key, points in raw_history.items():
        if not isinstance(points, list):
            continue
        result[key] = [
            FitnessPoint(run=p.get("run", 0), fitness=p.get("fitness", 0.0))
            for p in points
            if isinstance(p, dict)
        ]
    return result


# ── 優先キュー / SUSPENDED session マッピング ────────────────


def map_priority_queue(raw_queue: list | None) -> list[PriorityRequestEntry]:
    """raw 優先キューを `PriorityRequestEntry` のリストに変換する純粋関数。

    `raw_queue` が None / 空 / list でない場合は空リスト。dict でない要素は
    スキップする。`requested_at` / `relax_ratio` は float キャストで型耐性。
    """
    if not raw_queue or not isinstance(raw_queue, list):
        return []
    return [
        PriorityRequestEntry(
            reason=item.get("reason", ""),
            requested_at=float(item.get("requested_at", 0.0) or 0.0),
            relax_ratio=float(item.get("relax_ratio", 1.0) or 1.0),
            payload=item.get("payload"),
        )
        for item in raw_queue
        if isinstance(item, dict)
    ]


def map_active_session(raw_session: object | None) -> ActiveSessionInfo | None:
    """raw active session dict を `ActiveSessionInfo` に変換する純粋関数。

    `raw_session` が dict でない場合は `None`。
    """
    if not isinstance(raw_session, dict):
        return None
    return ActiveSessionInfo(
        session_id=str(raw_session.get("session_id", "")),
        started_at=float(raw_session.get("started_at", 0.0) or 0.0),
        reason=str(raw_session.get("reason", "")),
        completed_phases=list(raw_session.get("completed_phases", []) or []),
        yield_count=int(raw_session.get("yield_count", 0) or 0),
        cartridge_snapshot=list(raw_session.get("cartridge_snapshot", []) or []),
        experience_count=int(raw_session.get("experience_count", 0) or 0),
    )


# ── モード別経験数 ──────────────────────────────────────────────────────


def map_experience_by_mode(mode_data: dict | None) -> ExperienceByModeModel:
    """raw `experience_by_mode` dict を Pydantic に変換する純粋関数。

    None / 欠損キーは 0 を返す。
    """
    if not mode_data or not isinstance(mode_data, dict):
        return ExperienceByModeModel()
    return ExperienceByModeModel(
        chat=int(mode_data.get("chat", 0) or 0),
        create=int(mode_data.get("create", 0) or 0),
    )
