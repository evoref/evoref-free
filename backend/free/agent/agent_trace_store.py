"""MDP トレースの常設ストア (develop フラグ非依存)

``AgentTracer`` が出す begin / step / end イベントを、``DebugLogger`` の
デバッグ JSONL とは別に **通常起動でも** 追記する。エピソード記憶
(``mdp_ingester`` → episodic LTM) の入力はこちら。

- 1 日 1 ファイル ``agent_trace_YYYY-MM-DD.jsonl`` (UTC)。``MDPIngester`` /
  ``MDPTraceExtractor`` はディレクトリを ``agent_trace*.jsonl`` でグロブする
  ので、デバッグ JSONL と同じ命名にしておけば読み手を変えずに済む。
- 各行には ``trace_id`` (contextvars) を付け、``DebugLogger`` と同じ redaction
  を通す。develop=evolve では同じイベントがデバッグ JSONL にも出る (観測用、
  取り込みはしない)。
- ``retention_days`` を過ぎたファイルは追記時に消す (取り込み済みの古い
  トレースをディスクに溜め続けない)。``MDPIngester`` の offset は残るが、
  存在しないファイルは読み手側で無視される。
- 応答パス (イベントループ上) から同期で呼ばれるので、イベントごとに
  open/close せず当日のハンドルを持ち続け、書き込みは OS バッファへの
  ``write`` + ``flush`` だけにする (fsync はしない)。日付が変わったら
  ローテーションし、shutdown で :meth:`close` する。
"""

from __future__ import annotations

import json
import re
import threading
from datetime import timedelta
from pathlib import Path
from typing import TextIO

from backend.log_config import get_logger
from backend.structlog_config import redact_payload
from backend.trace_context import get_trace_id
from backend.utils import utc_now_dt

logger = get_logger("agent.trace_store")

FILE_PREFIX = "agent_trace_"
DEFAULT_RETENTION_DAYS = 30

_FILE_DATE_RE = re.compile(rf"^{FILE_PREFIX}(\d{{4}}-\d{{2}}-\d{{2}})\.jsonl$")


class AgentTraceStore:
    """``agent_trace_YYYY-MM-DD.jsonl`` への追記ストア。"""

    def __init__(
        self, directory: Path | str, *, retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.directory = Path(directory)
        self.retention_days = max(0, int(retention_days))
        self._lock = threading.Lock()
        self._pruned_for: str | None = None
        self._handle: TextIO | None = None
        self._handle_path: Path | None = None

    def current_path(self) -> Path:
        """今日 (UTC) の出力ファイル。"""
        return self.directory / f"{FILE_PREFIX}{utc_now_dt().date().isoformat()}.jsonl"

    def append(self, event: dict) -> None:
        """1 イベントを追記する。失敗は WARNING に留めて応答経路を止めない。"""
        payload = dict(event)
        if not payload.get("trace_id"):
            trace_id = get_trace_id()
            if trace_id:
                payload["trace_id"] = trace_id
        # private はレコード自身の印を優先する。contextvar はリクエスト単位で、
        # executor / バックグラウンド境界を越えると落ちるため、そこで書かれた
        # step が unmasked のまま永続化され episodic LTM へ入っていた。
        line = json.dumps(
            redact_payload(payload, force_private=bool(payload.get("private"))),
            ensure_ascii=False,
        )
        with self._lock:
            try:
                handle = self._open_for_today()
                handle.write(line + "\n")
                handle.flush()
                self._prune_if_day_changed(self.current_path().name)
            except OSError as exc:
                self._close_handle()
                logger.warning("agent trace append failed (%s): %s", self.directory, exc)

    def _open_for_today(self) -> TextIO:
        path = self.current_path()
        if self._handle is not None and self._handle_path == path:
            return self._handle
        self._close_handle()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._handle_path = path
        return self._handle

    def _close_handle(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
        self._handle = None
        self._handle_path = None

    def close(self) -> None:
        """常駐ハンドルを閉じる (shutdown 用。以後の append は再度開く)。"""
        with self._lock:
            self._close_handle()

    def _prune_if_day_changed(self, today_name: str) -> None:
        if self._pruned_for == today_name:
            return
        self._pruned_for = today_name
        self.prune()

    def prune(self) -> int:
        """``retention_days`` より古いファイルを削除し、件数を返す。"""
        if self.retention_days <= 0 or not self.directory.exists():
            return 0
        cutoff = (utc_now_dt() - timedelta(days=self.retention_days)).date().isoformat()
        removed = 0
        for path in self.directory.glob(f"{FILE_PREFIX}*.jsonl"):
            m = _FILE_DATE_RE.match(path.name)
            if m is None or m.group(1) >= cutoff:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("agent trace prune failed (%s): %s", path, exc)
        if removed:
            logger.info("Pruned %d agent trace file(s) older than %s", removed, cutoff)
        return removed
