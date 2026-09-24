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
- ``retention_days`` を過ぎたファイルは日付が変わって最初の追記で消す (取り込み
  済みの古いトレースをディスクに溜め続けない)。``MDPIngester`` の offset は残るが、
  存在しないファイルは読み手側で無視される。
- 応答パス (イベントループ上) から同期で呼ばれるので、ここでは 1 行を直列化して
  チャット経路の書き手スレッド (c_05 §0.5.9、:mod:`backend.io.writer_thread`) へ
  出すだけ。ハンドルは書き手スレッドが持ち続け、fsync はターンの終わりか 1 秒ごと。
  **耐久性の契約: 最後の 1 秒 (または終わっていないターン) のイベントは失ってよい**
  (取り込み待ちの入力で、落ちたエピソードは ``end`` の無い保留として捨てられる)。
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

from backend.io.format_registry import FormatSpec, register_format
from backend.io.jsonl_store import ROW_VERSION_FIELD
from backend.io.readonly import DataReadonlyError
from backend.io.writer_thread import ChatWriter, default_writer
from backend.log_config import get_logger
from backend.structlog_config import redact_payload
from backend.trace_context import get_trace_id
from backend.utils import utc_now_dt

logger = get_logger("agent.trace_store")

FILE_PREFIX = "agent_trace_"
DEFAULT_RETENTION_DAYS = 30

_FILE_DATE_RE = re.compile(rf"^{FILE_PREFIX}(\d{{4}}-\d{{2}}-\d{{2}})\.jsonl$")

AGENT_TRACE_FORMAT = register_format(FormatSpec(
    format_id="agent_trace",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/agent_trace/agent_trace_<yyyy-mm-dd>.jsonl",
    retention="retention_days (30) after the file's day; may lose the last second on a crash",
    encodings=("jsonl",),
))


class AgentTraceStore:
    """``agent_trace_YYYY-MM-DD.jsonl`` への追記ストア (書き手スレッド経由)。"""

    def __init__(
        self, directory: Path | str, *, retention_days: int = DEFAULT_RETENTION_DAYS,
        writer: ChatWriter | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.retention_days = max(0, int(retention_days))
        self._writer = writer
        self._pruned_for: str | None = None

    @property
    def writer(self) -> ChatWriter:
        return self._writer if self._writer is not None else default_writer()

    def current_path(self) -> Path:
        """今日 (UTC) の出力ファイル。"""
        return self.directory / f"{FILE_PREFIX}{utc_now_dt().date().isoformat()}.jsonl"

    def append(self, event: dict) -> None:
        """1 イベントを追記する (enqueue だけ)。失敗は応答経路を止めない。

        行の先頭に現行の版 ``_v`` を刻む (c_05 §0.5.1。読んだ行の版は運ばない)。
        """
        payload = {ROW_VERSION_FIELD: AGENT_TRACE_FORMAT.version}
        payload.update((k, v) for k, v in event.items() if k != ROW_VERSION_FIELD)
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
        path = self.current_path()
        try:
            self.writer.append(path, [line], format_id=AGENT_TRACE_FORMAT.format_id)
        except DataReadonlyError:
            logger.debug("agent trace not written: data root is read-only")
            return
        self._prune_if_day_changed(path.name)

    def close(self) -> None:
        """互換の入口 (ハンドルは書き手スレッドが持ち、停止時に閉じる)。"""

    def _prune_if_day_changed(self, today_name: str) -> None:
        if self._pruned_for == today_name:
            return
        self._pruned_for = today_name
        stale = self._stale_files()
        if not stale:
            return
        try:
            self.writer.call(
                lambda: self._remove(stale), format_id=AGENT_TRACE_FORMAT.format_id, paths=stale,
            )
        except DataReadonlyError:
            logger.debug("agent trace prune skipped: data root is read-only")

    def prune(self) -> int:
        """``retention_days`` より古いファイルを削除し、件数を返す (完了を待つ)。"""
        stale = self._stale_files()
        if not stale:
            return 0
        return self.writer.call(
            lambda: self._remove(stale), format_id=AGENT_TRACE_FORMAT.format_id, paths=stale,
        ).result()

    def _stale_files(self) -> list[Path]:
        if self.retention_days <= 0 or not self.directory.exists():
            return []
        cutoff = (utc_now_dt() - timedelta(days=self.retention_days)).date().isoformat()
        out: list[Path] = []
        for path in self.directory.glob(f"{FILE_PREFIX}*.jsonl"):
            m = _FILE_DATE_RE.match(path.name)
            # ISO 日付 (YYYY-MM-DD) 同士の比較。時刻の文字列ではない。
            if m is not None and m.group(1) < cutoff:
                out.append(path)
        return out

    def _remove(self, paths: list[Path]) -> int:
        """(書き手スレッド) 古いファイルを消す。"""
        removed = 0
        for path in paths:
            try:
                if self.writer.remove_now(path):
                    removed += 1
            except OSError as exc:
                logger.warning("agent trace prune failed (%s): %s", path, exc)
        if removed:
            logger.info("Pruned %d agent trace file(s)", removed)
        return removed
