"""Level 1 セッション + 優先キューのデータクラスと永続化ヘルパー

f_04 §5.4 / §5.5 の設計に従い、Level 1 の中断・再開を可能にする
セッション単位の進捗管理と、アイドル判定をバイパスして Level 1 を要求する
優先キューを提供する。

このモジュールはデータ構造と JSON I/O のみを担当し、スケジューラの
実行ループからは `LearningScheduler` 経由で利用される
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("learning.level1_session")


# ── PriorityRequest（f_04 §5.4）─────────────────────────────


@dataclass
class PriorityRequest:
    """通常のアイドル判定をバイパスして Level 1 を要求するエントリ

    `reason` ごとに最新値で上書きされる（同 reason の重複は許容しない）。
    """

    reason: str
    requested_at: float
    relax_ratio: float = 1.0  # 経験数閾値の緩和率（0.5 = 半分）
    payload: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PriorityRequest":
        return cls(
            reason=str(data["reason"]),
            requested_at=float(data.get("requested_at", time.time())),
            relax_ratio=float(data.get("relax_ratio", 1.0)),
            payload=data.get("payload"),
        )


# ── Level1Session（f_04 §5.5）───────────────────────────────


@dataclass
class Level1Session:
    """Level 1 の 1 セッション分の進捗

    1 セッションは「カートリッジ snapshot + 経験 snapshot + フェーズ進捗」を持つ。
    yield された場合は `level1_session_active.json` に保存され、再開時に
    そのまま続きから実行できる。
    """

    session_id: str
    started_at: float
    cartridge_snapshot: list[str]      # JSON シリアライズ可能形式
    experience_snapshot: list[dict]    # 開始時に固定された経験リスト
    completed_phases: list[str] = field(default_factory=list)
    phase_state: dict[str, dict] = field(default_factory=dict)
    reason: str = "idle"
    yield_count: int = 0

    @classmethod
    def new(
        cls,
        cartridge_ids: list[str] | set[str] | frozenset[str],
        experiences: list[dict],
        reason: str = "idle",
    ) -> "Level1Session":
        return cls(
            session_id=str(uuid.uuid4()),
            started_at=time.time(),
            cartridge_snapshot=sorted(cartridge_ids),
            experience_snapshot=list(experiences),
            completed_phases=[],
            phase_state={},
            reason=reason,
            yield_count=0,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Level1Session":
        return cls(
            session_id=str(data["session_id"]),
            started_at=float(data.get("started_at", time.time())),
            cartridge_snapshot=list(data.get("cartridge_snapshot", [])),
            experience_snapshot=list(data.get("experience_snapshot", [])),
            completed_phases=list(data.get("completed_phases", [])),
            phase_state=dict(data.get("phase_state", {})),
            reason=str(data.get("reason", "idle")),
            yield_count=int(data.get("yield_count", 0)),
        )


# ── 永続化ヘルパー ────────────────────────────────────────────


def save_active_session(path: Path, session: Level1Session) -> None:
    """SUSPENDED な Level1Session を JSON ファイルへアトミック保存する"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    logger.debug("Active session saved: %s", path)


def load_active_session(path: Path) -> Level1Session | None:
    """active session ファイルがあればロードする。無ければ None"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Level1Session.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load active session %s: %s", path, e)
        return None


def archive_session(active_path: Path, history_dir: Path, session: Level1Session) -> Path:
    """完了した session を history へ移動する。

    active ファイルを削除し、`history_dir/{session_id}.json` に書き出す。
    返り値はアーカイブ後のパス。
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    target = history_dir / f"{session.session_id}.json"
    target.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if active_path.exists():
        try:
            active_path.unlink()
        except OSError as e:
            logger.warning("Failed to remove active session file %s: %s", active_path, e)
    logger.info("Session archived: %s", target)
    return target


def discard_active_session(path: Path) -> None:
    """active session ファイルを破棄する（destructive cancel 後の復旧不能ケース等）"""
    if path.exists():
        try:
            path.unlink()
            logger.info("Active session discarded: %s", path)
        except OSError as e:
            logger.warning("Failed to discard active session %s: %s", path, e)
