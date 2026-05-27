"""`/api/status` ハンドラから抽出した純粋な収集ロジック

`backend/free/api/status.py` の `_collect_debug_info` ハンドラ内に直書きされて
いた以下のロジックを抽出した純粋関数群:
- ログディレクトリの解決 (相対パス → 絶対パス)
- ログディレクトリ全体のディスク使用量計算 (rglob + stat)
- backend.log の ERROR 行数カウント
- 埋め込みキャッシュヒット率の算出
- `LearningScheduler` から `LearningBriefStatus` への抽出

レイヤー責務:
- `status.py` (API 層)            — HTTP / Pydantic / 状態取得 / オーケストレーション
- `_status_collectors.py` (helper) — 純粋な計算 / パス解決 / ファイル I/O ベース集計

`compute_cache_hit_rate` と `resolve_log_dir` は完全な純粋関数。
`compute_log_disk_usage_mb` / `count_recent_errors` / `extract_learning_brief` は
ファイル I/O または scheduler オブジェクトに依存するが、副作用は読み込みのみ。
"""

from __future__ import annotations

from pathlib import Path

from backend.free.api.schemas import LearningBriefStatus


# ── ログディレクトリ ────────────────────────────────────────────────────


def resolve_log_dir(log_dir_str: str, project_root: Path) -> Path:
    """設定値の `log_dir` (相対 / 絶対) を絶対パスとして解決する純粋関数。

    既に絶対パスならそのまま `Path` 化、相対パスなら `project_root` 配下に解決。
    """
    log_dir = Path(log_dir_str)
    if log_dir.is_absolute():
        return log_dir
    return project_root / log_dir


def compute_log_disk_usage_mb(log_dir: Path) -> float:
    """ログディレクトリ配下の全ファイルサイズ合計を MB で返す。

    ディレクトリが存在しない場合は 0.0。読み取り不可ファイルは無視する
    (元 handler の `OSError` 寛容と同じ挙動)。
    """
    if not log_dir.exists():
        return 0.0
    total = 0
    for f in log_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            total += f.stat().st_size
        except OSError:
            continue
    return round(total / (1024 * 1024), 2)


def count_recent_errors(backend_log: Path) -> int:
    """`backend.log` 内の `[ERROR]` 行を数える。

    ファイルが存在しない / 読み込めない場合は 0 を返す (元 handler 互換)。
    """
    if not backend_log.exists():
        return 0
    try:
        text = backend_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return text.count(" [ERROR] ")


# ── 埋め込みキャッシュヒット率 ──────────────────────────────────────────


def compute_cache_hit_rate(cache_stats: object) -> float:
    """埋め込みキャッシュ統計からヒット率 (0.0 - 1.0) を算出する純粋関数。

    `cache_stats` が `dict` 以外、または hits + misses が 0 の場合は 0.0。
    元 handler の `int(... or 0)` による寛容な型変換を保持。
    """
    if not isinstance(cache_stats, dict):
        return 0.0
    hits = int(cache_stats.get("hits", 0) or 0)
    misses = int(cache_stats.get("misses", 0) or 0)
    total = hits + misses
    if total <= 0:
        return 0.0
    return round(hits / total, 4)


def extract_embedder_cache_hit_rate(embedder: object | None) -> float:
    """`state.embedder` から `cache_stats` を取り出してヒット率を算出する。

    `embedder` が `None`、または `cache_stats` 属性を持たない場合は 0.0。
    `compute_cache_hit_rate` への薄いラッパ。
    """
    if embedder is None or not hasattr(embedder, "cache_stats"):
        return 0.0
    return compute_cache_hit_rate(embedder.cache_stats)


# ── 学習スケジューラ → LearningBriefStatus ─────────────────────────────


def extract_learning_brief(scheduler: object | None) -> LearningBriefStatus:
    """`LearningScheduler` から `LearningBriefStatus` を構築する。

    `scheduler` が `None`、または `get_status()` が例外を投げた場合は
    既定値 (`running=False, experience_count=0, conditions_met=False`) を返す。
    """
    if scheduler is None:
        return LearningBriefStatus()
    try:
        status = scheduler.get_status()
    except Exception:
        return LearningBriefStatus()
    return LearningBriefStatus(
        running=status.get("running", False),
        experience_count=status.get("experience_count", 0),
        conditions_met=status.get("conditions_met", False),
    )
