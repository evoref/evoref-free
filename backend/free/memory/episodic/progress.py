"""ノート化済みターンの進捗 (``episodic/progress.json``)。

sleep-time のノート生成は「会話履歴のうち、まだノートにしていないターン」を
入力にする。同じターンを二度ノートにしないための **唯一の判定材料** がこの
ファイルで、セッションごとに「最後にノート化したターンの id と件数」を持つ。

位置 (件数) だけでは足りない — 履歴は圧縮・復元でターン列の前方が縮むこと
があり、そのとき件数基準だと未処理のターンを飛ばす。id が見つかればそれを
優先し、見つからないときだけ件数へ落ちる。

エンベロープ (``schema_version`` / ``written_at`` / ``producer`` / ``payload``)
と「未対応の新しい版は読まず書き戻さない」規約は c_05 §0.5 のとおり。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.io import AtomicWriter
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("memory.episodic.progress")

PROGRESS_FILE = "progress.json"

#: 1 セッションあたりで覚えておくターン数の上限は持たない — 保持しているのは
#: 「最後の 1 件の id と件数」だけなので、セッション数に比例した定数サイズ。
SCHEMA_VERSION = 1


class EpisodicProgress:
    """セッション → 最後にノート化したターン。

    Attributes:
        sessions: ``{session_id: {"last_turn_id": str, "turn_count": int,
            "updated_at": ISO}}``。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.sessions: dict[str, dict[str, Any]] = {}
        #: 未対応の新しい版を読んだ = 書き戻すと壊すので保存しない。
        self.readonly: bool = False
        self._dirty = False

    # ── 読み書き ──

    def load(self) -> bool:
        """進捗を読む。ファイル不在 / 破損 / 新しい版では ``False``。"""
        if not self.path.exists():
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read episodic progress %s: %s", self.path, e)
            return False
        version = int((raw or {}).get("schema_version") or 0)
        if version > SCHEMA_VERSION:
            logger.warning(
                "Episodic progress %s is schema_version=%d (> %d); refusing to "
                "read or write it back", self.path, version, SCHEMA_VERSION,
            )
            self.readonly = True
            return False
        payload = (raw or {}).get("payload") or {}
        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            logger.warning("Episodic progress %s has no sessions map", self.path)
            return False
        self.sessions = {
            str(sid): dict(entry)
            for sid, entry in sessions.items()
            if isinstance(entry, dict)
        }
        return True

    def save(self) -> None:
        """進捗を原子的に書き出す (変更が無ければ何もしない)。"""
        if self.readonly:
            logger.warning(
                "Skipping save of episodic progress %s: on-disk file is newer",
                self.path,
            )
            return
        if not self._dirty:
            return
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "written_at": utc_now(),
            "producer": {"component": "memory.episodic.progress"},
            "payload": {"sessions": self.sessions},
        }
        with AtomicWriter(self.path) as f:
            f.write(json.dumps(envelope, ensure_ascii=False, indent=2))
        self._dirty = False

    # ── 判定 ──

    def start_index(self, session_id: str, turns: list[dict]) -> int:
        """``turns`` のうち、ノート化を始めるべき添字を返す。

        最後にノート化した ``turn_id`` が列の中に見つかればその次から、
        見つからなければ記録済みの件数から始める (どちらも無ければ 0)。
        """
        entry = self.sessions.get(session_id)
        if not entry:
            return 0
        last_turn_id = str(entry.get("last_turn_id") or "")
        if last_turn_id:
            for index, turn in enumerate(turns):
                if str(turn.get("turn_id") or "") == last_turn_id:
                    return index + 1
        count = int(entry.get("turn_count") or 0)
        return min(count, len(turns))

    def mark(self, session_id: str, turns: list[dict], processed_upto: int) -> None:
        """``turns[:processed_upto]`` までノート化した、と記録する。"""
        if processed_upto <= 0:
            return
        last = turns[processed_upto - 1]
        self.sessions[session_id] = {
            "last_turn_id": str(last.get("turn_id") or ""),
            "turn_count": int(processed_upto),
            "updated_at": utc_now(),
        }
        self._dirty = True

    def forget(self, session_id: str) -> None:
        """セッションの進捗を落とす (履歴削除に追従する)。"""
        if self.sessions.pop(session_id, None) is not None:
            self._dirty = True


__all__ = ["PROGRESS_FILE", "SCHEMA_VERSION", "EpisodicProgress"]
