"""ノート化済みターンの進捗 (``episodic/progress.json``)。

sleep-time のノート生成は「会話履歴のうち、まだノートにしていないターン」を
入力にする。同じターンを二度ノートにしないための **唯一の判定材料** がこの
ファイルで、セッションごとに「最後にノート化したターンの id と件数」を持つ。

位置 (件数) だけでは足りない — 履歴は圧縮・復元でターン列の前方が縮むこと
があり、そのとき件数基準だと未処理のターンを飛ばす。id が見つかればそれを
優先し、見つからないときだけ件数へ落ちる。

封筒 (形式 ``episodic.progress``、:class:`~backend.io.versioned.VersionedJsonFile`)
と「新しい版 / G1 の封筒でないファイルは読まず書き戻さない (readonly)、壊れた
ファイルは退避する」規約は c_05 §0.4 / §0.5 のとおり。readonly の間はノート化を
止める (進捗を残せないまま取り込むと、再起動のたびに同じターンを二度ノートにする
— :func:`~backend.free.memory.episodic.ingest.ingest_new_turns`)。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("memory.episodic.progress")

PROGRESS_FILE = "progress.json"


@persisted()
@dataclass
class SessionProgress:
    """1 セッションの進捗 (最後にノート化したターンの id と件数)。"""

    last_turn_id: str = ""
    turn_count: int = 0
    updated_at: str = ""
    #: この版が知らないキー (書き戻しでそのまま戻す)。
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class ProgressPayload:
    """``progress.json`` の payload (``sessions`` の無いファイルは壊れている)。"""

    sessions: dict[str, SessionProgress]
    _extra: dict[str, Any] | None = None


_PAYLOAD_CODEC = codec_for(ProgressPayload)

#: 1 セッションあたりで覚えておくターン数の上限は持たない — 保持しているのは
#: 「最後の 1 件の id と件数」だけなので、セッション数に比例した定数サイズ。
EPISODIC_PROGRESS_FORMAT = register_format(FormatSpec(
    format_id="episodic.progress",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/memory/episodic/progress.json",
    retention="follows forget; one entry per session",
    export=True,
    records=(ProgressPayload,),
))


class EpisodicProgress(VersionedJsonFile):
    """セッション → 最後にノート化したターン。

    Attributes:
        sessions: ``{session_id: SessionProgress}``。
        readonly: ディスク上のファイルを書き戻すと壊す (新しい版 / G1 の封筒で
            ない / 退避に失敗した)。``save`` は書かない。
    """

    FORMAT = EPISODIC_PROGRESS_FORMAT
    RAISE_ON_SAVE_ERROR = True
    _state_logger = logger

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.sessions: dict[str, SessionProgress] = {}
        #: payload の未知キー (書き戻しでそのまま戻す)。
        self._extra: dict[str, Any] | None = None
        self._dirty = False

    # ── 読み書き ──

    def save(self, path: Path | str | None = None) -> bool:
        """進捗を封筒付きで原子的に書き出す (変更が無ければ何もしない)。

        readonly ならログに出して ``False``。書き込みの失敗は従来どおり送出する。
        """
        if self.readonly:
            logger.warning(
                "Skipping save of episodic progress %s: the on-disk file must "
                "not be overwritten (%s)", self.path, self.last_status,
            )
            return False
        if not self._dirty:
            return True
        if not super().save(path):
            return False
        self._dirty = False
        return True

    def _to_payload(self) -> dict[str, Any]:
        return _PAYLOAD_CODEC.encode(ProgressPayload(sessions=self.sessions, _extra=self._extra))

    def _from_payload(self, payload: Any) -> None:
        data = _PAYLOAD_CODEC.decode(payload)
        self.sessions = data.sessions
        self._extra = data._extra

    # ── 判定 ──

    def start_index(self, session_id: str, turns: list[dict]) -> int:
        """``turns`` のうち、ノート化を始めるべき添字を返す。

        最後にノート化した ``turn_id`` が列の中に見つかればその次から、
        見つからなければ記録済みの件数から始める (どちらも無ければ 0)。
        """
        entry = self.sessions.get(session_id)
        if entry is None:
            return 0
        if entry.last_turn_id:
            for index, turn in enumerate(turns):
                if str(turn.get("turn_id") or "") == entry.last_turn_id:
                    return index + 1
        return min(entry.turn_count, len(turns))

    def mark(self, session_id: str, turns: list[dict], processed_upto: int) -> None:
        """``turns[:processed_upto]`` までノート化した、と記録する (未知キーは残す)。"""
        if processed_upto <= 0:
            return
        last = turns[processed_upto - 1]
        self.sessions[session_id] = replace(
            self.sessions.get(session_id) or SessionProgress(),
            last_turn_id=str(last.get("turn_id") or ""),
            turn_count=int(processed_upto),
            updated_at=utc_now(),
        )
        self._dirty = True

    def forget(self, session_id: str) -> None:
        """セッションの進捗を落とす (履歴削除に追従する)。"""
        if self.sessions.pop(session_id, None) is not None:
            self._dirty = True


__all__ = ["EPISODIC_PROGRESS_FORMAT", "PROGRESS_FILE", "EpisodicProgress"]
