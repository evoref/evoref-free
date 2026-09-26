"""staged クリエイトの run レコード + 追記イベントログ (f_10 §7、2026-09-18)。

``WorkspaceManager`` (``manifest.json`` / 生成物) とは別の面として、1 staged
実行 (= 1 workspace = 1 run) の **事実**を ``run.json`` に、**時系列イベント**を
``events.jsonl`` に持つ。「事実を永続し、表示状態は導出する」設計 —
``derive_run_status`` が読み出し時に ``working`` / ``needs_input`` / ``done`` /
``incomplete`` / ``failed`` / ``cancelled`` / ``timeout`` を導出するので、表示状態そのものは
どこにも保存しない。

書き手は ``chat_stream_staged.run_staged_pipeline`` (staged) と
``generation.harness.LongFormHarness`` (longform、f_08 §2.3) の開始 / 終端だけ。
読み手は ``/api/create/runs*`` (再接続の口、backend のみ) と sleep-time の
GC (Step 5.89)。

ディレクトリ配置は :mod:`backend.free.loop.staged.workspace` と同じ
``create_workspace_dir/{run_id}/`` (run_id == workspace_id)。
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.io import JSONLAppendStore
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import id_pattern, is_valid_id
from backend.io.jsonl_store import ROW_VERSION_FIELD
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger
from backend.utils import parse_utc, utc_now, utc_now_dt

logger = get_logger("loop.staged.run_record")

RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"

#: ``.trash-`` へ改名してから消す GC の接頭辞 (corpus の版 GC と同じ作法、c_16 §5.4)。
TRASH_PREFIX = ".trash-"

ActivityState = Literal["running", "blocked", "exited"]
#: ``resumed`` は blocked (needs_input) から次ターンで再開されたときに旧 run が
#: 終端する exit_kind (Phase 3b、f_10 §7)。
ExitKind = Literal["done", "timeout", "cancelled", "disconnected", "error", "resumed"]
#: ``incomplete`` は ``exit_kind=done`` だが ``tasks_failed > 0`` (流れたが欠けた、f_10 §7)。
RunStatus = Literal[
    "working", "needs_input", "done", "incomplete", "failed", "cancelled", "timeout",
]

#: run_id (= workspace_id) の接頭辞 (ID 台帳 ``run_``、c_05 §0.5.5)。
RUN_ID_PREFIX = "run_"
#: run_id の文法 (発番時と API 入口の検査用)。
RUN_ID_PATTERN = id_pattern(RUN_ID_PREFIX)

#: 時刻が壊れて読めない run を GC のソート先頭 (= 最初に削除対象) へ寄せる番兵。
_EPOCH_MIN = datetime.min.replace(tzinfo=timezone.utc)


def _sort_key(iso: str) -> datetime:
    return parse_utc(iso) or _EPOCH_MIN


def is_valid_run_id(run_id: str) -> bool:
    """``run_id`` が ID 台帳の文法に合うか (API 入口の検査。traversal 防止を兼ねる)。"""
    return is_valid_id(run_id, RUN_ID_PREFIX)


def _is_run_dir(entry: Path) -> bool:
    """``create_dir`` 直下の run の作業場か (読み手の判定。長さでは弾かない、c_05 §0.5.5)。"""
    return entry.is_dir() and entry.name.startswith(RUN_ID_PREFIX)


# ===========================================================================
# RunRecord (run.json)
# ===========================================================================


@dataclass
class RunRecord:
    """1 staged 実行の事実 (``run.json`` の payload)。"""

    run_id: str
    session_id: str
    request_id: str
    mode: str
    query: str
    output_target: str
    started_at: str
    ended_at: str | None = None
    activity_state: ActivityState = "running"
    exit_kind: ExitKind | None = None
    last_event_seq: int = 0
    brief_tokens: int = 0
    #: 構成テンプレートで seed した場合の来歴鍵 (``<package_id>@<version>:
    #: <entry_id>``、c_05 §0.6 / c_16 §4.5.2)。seed していない run は空文字。
    template: str = ""
    #: パイプラインが流れた上で欠けたもの (失敗タスク / 未生成モジュール / 落ちた
    #: 契約テスト) の件数。``exit_kind`` は「どう終わったか」でこれとは別の軸
    #: (f_10 §7、2026-09-26)。任意フィールドなので形式の版は上げない。
    tasks_failed: int = 0
    #: 未知キー退避 (c_05 §0.5.2)。次の保存で原形のままトップレベルへ復元する。
    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """API 応答 / 永続化ペイロード共用の dict 化 (``_extra`` はトップレベルへ展開)。"""
        out = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "_extra"
        }
        out.update(self._extra)
        return out


_RUN_RECORD_FIELDS = {f.name for f in dataclasses.fields(RunRecord) if f.name != "_extra"}
_RUN_RECORD_REQUIRED = {
    f.name
    for f in dataclasses.fields(RunRecord)
    if f.name != "_extra"
    and f.default is dataclasses.MISSING
    and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
}


CREATE_RUN_FORMAT = register_format(FormatSpec(
    format_id="create.run",
    version=1,
    klass="sot",
    writers=frozenset({"free", "pro"}),
    path_key=f"store/create/<run>/{RUN_FILE}",
    retention="kept with the run workspace",
))

#: 作業場の工程間ハンドオフ (``WorkspaceManager`` の ``manifest.json``、封筒付き)。
#: ``create.workspace`` の木より具体的な path_key なので台帳ではこちらに分類される。
CREATE_WORKSPACE_MANIFEST_FORMAT = register_format(FormatSpec(
    format_id="create.workspace_manifest",
    version=1,
    klass="sot",
    writers=frozenset({"free", "pro"}),
    path_key="store/create/<run>/manifest.json",
    retention="kept with the run workspace",
))

#: run の作業ディレクトリの残り (隔離 SemMem ``.semmem``・仕様 / 計画・生成物)。create の run は export しない (G1 設計 §8.6)。
CREATE_WORKSPACE_FORMAT = register_format(FormatSpec(
    format_id="create.workspace",
    version=1,
    klass="sot",
    writers=frozenset({"free", "pro"}),
    path_key="store/create/<run>/**",
    retention="create.runs_keep newest runs (by ended_at); running runs are never removed",
    encodings=("dir",),
))


class RunRecordStore(VersionedJsonFile):
    """``run.json`` (封筒付き、SoT なので fsync)。"""

    FORMAT = CREATE_RUN_FORMAT
    RAISE_ON_SAVE_ERROR = True
    _state_logger = logger

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(Path(workspace_root) / RUN_FILE)
        self.record: RunRecord | None = None

    # ── 書き手 API (run_staged_pipeline / LongFormHarness のみ) ──────────

    def start(
        self,
        *,
        run_id: str,
        session_id: str,
        request_id: str,
        mode: str,
        query: str,
        output_target: str,
        brief_tokens: int = 0,
        resume_of: str | None = None,
    ) -> RunRecord:
        """run 開始を記録し即座に永続化する。

        ``resume_of`` は blocked (needs_input) から再開した run の元 run_id
        (Phase 3b、f_10 §7)。``_extra`` へ退避し ``RunRecord.to_dict()`` で
        API 応答へそのまま透過する。
        """
        self.record = RunRecord(
            run_id=run_id, session_id=session_id, request_id=request_id,
            mode=mode, query=query, output_target=output_target,
            started_at=utc_now(), activity_state="running",
            brief_tokens=int(brief_tokens),
            _extra={"resume_of": resume_of} if resume_of else {},
        )
        self.save()
        return self.record

    def block(self, question: str) -> None:
        """制作ステージの問い返し (needs_input) を記録し即座に永続化する。

        ``question`` は ``_extra`` へ退避し ``RunRecord.to_dict()`` で API
        応答へそのまま透過する (Phase 3b、f_10 §7)。``start`` 未実行なら no-op。
        """
        if self.record is None:
            return
        self.record.activity_state = "blocked"
        self.record._extra["question"] = question
        self.save()

    def finish(
        self, exit_kind: ExitKind, *, last_event_seq: int = 0, template: str = "",
        tasks_failed: int = 0,
    ) -> None:
        """run 終端を記録し即座に永続化する。``start`` 未実行なら no-op。

        ``template`` は構成テンプレートで seed した場合の来歴鍵 (c_05 §0.6)。
        """
        if self.record is None:
            return
        self.record.ended_at = utc_now()
        self.record.activity_state = "exited"
        self.record.exit_kind = exit_kind
        self.record.last_event_seq = int(last_event_seq)
        self.record.tasks_failed = max(0, int(tasks_failed))
        if template:
            self.record.template = template
        self.save()

    # ── VersionedJsonFile 抽象メソッド ──────────────────────────────────────

    def _to_payload(self) -> dict[str, Any]:
        return self.record.to_dict() if self.record is not None else {}

    def _from_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("run record payload must be an object")
        missing = _RUN_RECORD_REQUIRED - payload.keys()
        if missing:
            raise ValueError(f"run record missing required fields: {sorted(missing)}")
        kwargs = {k: payload[k] for k in _RUN_RECORD_FIELDS if k in payload}
        kwargs["_extra"] = {k: v for k, v in payload.items() if k not in _RUN_RECORD_FIELDS}
        self.record = RunRecord(**kwargs)


# ===========================================================================
# RunEventLog (events.jsonl)
# ===========================================================================


@persisted()
@dataclass
class RunEvent:
    """1 イベント行 (``{_v, seq, at, kind, payload}`` の ``_v`` 以外。未知キーは ``_extra``)。"""

    seq: int
    at: str
    kind: str
    payload: dict[str, Any]
    #: この版が知らないキー (追記し直すときにそのまま戻す)。
    _extra: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """API 応答の形 (``{seq, at, kind, payload}``)。"""
        return {"seq": self.seq, "at": self.at, "kind": self.kind, "payload": self.payload}


#: run の事象ログ (``events.jsonl``、行は ``{"_v", ...}`` の追記だけ)。``create.workspace``
#: の木より具体的な path_key なので台帳ではこちらに分類される。
CREATE_RUN_EVENTS_FORMAT = register_format(FormatSpec(
    format_id="create.run_events",
    version=1,
    klass="sot",
    writers=frozenset({"free", "pro"}),
    path_key=f"store/create/<run>/{EVENTS_FILE}",
    retention="append-only; kept with the run workspace",
    encodings=("jsonl",),
    records=(RunEvent,),
))
_RUN_EVENT_CODEC = codec_for(RunEvent)


def _serialize_event(evt: RunEvent) -> str:
    row = {ROW_VERSION_FIELD: CREATE_RUN_EVENTS_FORMAT.version, **_RUN_EVENT_CODEC.encode(evt)}
    return json.dumps(row, ensure_ascii=False)


def _deserialize_event(line: str) -> RunEvent:
    obj = json.loads(line)
    obj.pop(ROW_VERSION_FIELD, None)  # 版はストアが確かめ済み
    return _RUN_EVENT_CODEC.decode(obj)


class RunEventLog:
    """``events.jsonl`` — 追記専用 (:class:`JSONLAppendStore` 経由)。

    ``seq`` はこのログ内で **1 始まり** で単調増加するファイル内カウンタ
    (位置カウンタではない、c_05 §0.5.4)。0 を「まだ 1 件も読んでいない /
    追記していない」の番兵として空けておくことで、再接続 API の既定
    ``after=0`` (= 先頭から全部読む) がそのまま成立する。再開時は末尾を
    読んで続きから採番する。
    """

    def __init__(
        self, workspace_root: Path | str, *, debug_logger: Any | None = None,
    ) -> None:
        self._store: JSONLAppendStore[RunEvent] = JSONLAppendStore(
            Path(workspace_root) / EVENTS_FILE,
            serialize=_serialize_event,
            deserialize=_deserialize_event,
            key_of=lambda evt: str(evt.seq),
            debug_logger=debug_logger,
            row_version=CREATE_RUN_EVENTS_FORMAT.version,
        )
        existing = self._store.load_all().values()
        self._next_seq = max((e.seq for e in existing), default=0) + 1

    def append(self, kind: str, payload: dict[str, Any]) -> RunEvent:
        """1 イベントを追記し、その行を返す。"""
        evt = RunEvent(seq=self._next_seq, at=utc_now(), kind=kind, payload=dict(payload))
        self._store.append(evt)
        self._next_seq += 1
        return evt

    @property
    def last_seq(self) -> int:
        """最後に追記した seq (未追記なら -1)。"""
        return self._next_seq - 1

    def read(self, *, after: int = 0) -> list[RunEvent]:
        """``seq > after`` のイベントを seq 昇順で返す (watermark 付き読み)。"""
        events = [e for e in self._store.load_all().values() if e.seq > after]
        events.sort(key=lambda e: e.seq)
        return events

    def latest(self) -> RunEvent | None:
        """最新 (seq 最大) のイベント。1 件も無ければ ``None``。"""
        events = list(self._store.load_all().values())
        return max(events, key=lambda e: e.seq) if events else None


# ===========================================================================
# 導出状態
# ===========================================================================

_EXIT_KIND_TO_STATUS: dict[str, RunStatus] = {
    "done": "done",
    "timeout": "timeout",
    "cancelled": "cancelled",
    # disconnected / error は derive_run_status の戻り値集合に専用の値が
    # 無いため failed に畳む (ユーザー起因のキャンセルでもタイムアウトでも
    # ないという意味で「異常終了」側)。
    "disconnected": "failed",
    "error": "failed",
    # resumed は次ターンで新しい run へ引き継がれた旧 run の終端 (異常終了ではない)。
    "resumed": "done",
}

#: exit_kind が未記録のまま exited (プロセス強制終了等) だった場合の最終フォール
#: バック。最新イベントの kind から推測する。
_EVENT_KIND_TO_STATUS: dict[str, RunStatus] = {
    "cancel": "cancelled",
    "cancelled": "cancelled",
    "timeout": "timeout",
    "disconnect": "failed",
}


#: ``running`` のまま ``started_at`` からこの秒数を過ぎた run は「書き手のプロセスが消えた」
#: とみなし ``failed`` に導出する (create.turn_timeout_sec の既定 3600 + 余裕 600)。
STALE_RUNNING_AFTER_SEC = 3600.0 + 600.0


def stale_after_from_config(config: dict | None) -> float:
    """``create.turn_timeout_sec`` + 600 秒 (config が無ければ既定) を返す。"""
    try:
        turn = float(((config or {}).get("create") or {}).get("turn_timeout_sec", 3600.0))
    except (TypeError, ValueError):
        turn = 3600.0
    return turn + 600.0


def _tasks_failed(run: RunRecord) -> int:
    """``tasks_failed`` を整数で読む (手で壊された値は 0 = 欠けていない扱い)。"""
    try:
        return int(run.tasks_failed or 0)
    except (TypeError, ValueError):
        return 0


def derive_run_status(
    run: RunRecord,
    last_event: RunEvent | None,
    *,
    stale_after_sec: float = STALE_RUNNING_AFTER_SEC,
    now: datetime | None = None,
) -> RunStatus:
    """``run.json`` + 最新イベントから表示状態を導出する (優先順: activity_state →
    exit_kind → 最新 event)。表示状態そのものは保存しない (f_10 §7)。

    ``exit_kind=done`` でも ``tasks_failed > 0`` なら ``incomplete`` (2026-09-26)。
    ``running`` は ``started_at`` から ``stale_after_sec`` を過ぎていれば ``failed``
    (書き手のプロセスがクラッシュ / kill されると終端が書かれないため。2026-09-19)。
    """
    if run.activity_state == "running":
        started = parse_utc(run.started_at)
        current = now or utc_now_dt()
        if started is not None and (current - started).total_seconds() > stale_after_sec:
            return "failed"
        return "working"
    if run.activity_state == "blocked":
        return "needs_input"
    if run.exit_kind:
        status = _EXIT_KIND_TO_STATUS.get(run.exit_kind, "failed")
        if status == "done" and run.exit_kind == "done" and _tasks_failed(run) > 0:
            return "incomplete"
        return status
    if last_event is not None:
        return _EVENT_KIND_TO_STATUS.get(last_event.kind, "failed")
    return "failed"


# ===========================================================================
# 読み手 API (再接続の口 / GC が使う)
# ===========================================================================


def list_runs(
    create_dir: Path | str, session_id: str | None = None,
    *, stale_after_sec: float = STALE_RUNNING_AFTER_SEC,
) -> list[tuple[RunRecord, RunStatus]]:
    """``create_dir`` 配下の run 一覧を ``started_at`` 降順で返す (壊れた run はスキップ)。"""
    root = Path(create_dir)
    if not root.is_dir():
        return []
    out: list[tuple[RunRecord, RunStatus]] = []
    skipped = 0
    for entry in sorted(root.iterdir()):
        if not _is_run_dir(entry):
            continue
        # run.json の無い作業場は run record 導入前の workspace で、壊れた run ではない。
        if not (entry / RUN_FILE).is_file():
            continue
        loaded = _load_run(entry, stale_after_sec=stale_after_sec)
        if loaded is None:
            skipped += 1
            continue
        record, status = loaded
        if session_id is not None and record.session_id != session_id:
            continue
        out.append((record, status))
    if skipped:
        logger.warning("list_runs: skipped %d unreadable run(s) under %s", skipped, root)
    out.sort(key=lambda t: _sort_key(t[0].started_at), reverse=True)
    return out


def load_run(
    create_dir: Path | str, run_id: str, *, stale_after_sec: float = STALE_RUNNING_AFTER_SEC,
) -> tuple[RunRecord, RunStatus] | None:
    """1 run の記録 + 導出状態。存在しない / 不正 id / 壊れている場合は ``None``。"""
    if not is_valid_run_id(run_id):
        return None
    return _load_run(Path(create_dir) / run_id, stale_after_sec=stale_after_sec)


def read_events(
    create_dir: Path | str, run_id: str, *, after: int = 0,
) -> list[RunEvent] | None:
    """1 run のイベントを watermark 付きで読む。run が存在しなければ ``None``。"""
    if not is_valid_run_id(run_id):
        return None
    workspace_root = Path(create_dir) / run_id
    if not workspace_root.is_dir():
        return None
    return RunEventLog(workspace_root).read(after=after)


def _load_run(
    workspace_root: Path, *, stale_after_sec: float = STALE_RUNNING_AFTER_SEC,
) -> tuple[RunRecord, RunStatus] | None:
    store = RunRecordStore(workspace_root)
    if not store.load() or store.record is None:
        return None
    last_event = RunEventLog(workspace_root).latest()
    return store.record, derive_run_status(
        store.record, last_event, stale_after_sec=stale_after_sec,
    )


# ===========================================================================
# GC (sleep-time Step 5.89、c_05 §0.5.6)
# ===========================================================================


def gc_old_runs(
    create_dir: Path | str, *, keep: int = 20,
    stale_after_sec: float = STALE_RUNNING_AFTER_SEC,
) -> list[str]:
    """``ended_at`` の古い順に ``keep`` を超えた exited run を刈る。

    走行中 (``activity_state == "running"``) は対象外。``.trash-`` へ改名して
    から中身を消す (corpus 版 GC と同じ作法、c_16 §5.4) — 掴まれていれば改名
    ごと失敗し次回へ回る。
    """
    if keep < 1:
        return []
    root = Path(create_dir)
    if not root.is_dir():
        return []
    removed: list[str] = []
    # 前回の掃き残し (改名はできたが中身を消せなかった run) を先に回収する。
    for leftover in root.glob(f"{TRASH_PREFIX}*"):
        shutil.rmtree(str(leftover), ignore_errors=True)

    exited: list[tuple[str, str, Path]] = []  # (ended_at, run_id, dir)
    for entry in root.iterdir():
        if not _is_run_dir(entry):
            continue
        store = RunRecordStore(entry)
        if not store.load() or store.record is None:
            continue
        record = store.record
        # 走行中は消さない。ただし陳腐化した running (書き手が消えた) は failed と
        # 導出されるので GC 対象に含める (f_10 §7)。
        if record.activity_state == "running" and derive_run_status(
            record, None, stale_after_sec=stale_after_sec,
        ) == "working":
            continue
        exited.append((record.ended_at or record.started_at, record.run_id, entry))

    if len(exited) <= keep:
        return []
    exited.sort(key=lambda t: _sort_key(t[0]))
    for _ended_at, run_id, path in exited[: len(exited) - keep]:
        trash = root / f"{TRASH_PREFIX}{path.name}"
        try:
            path.rename(trash)
        except OSError as e:
            logger.warning("run GC deferred (in use?): %s: %s", run_id, e)
            continue
        shutil.rmtree(str(trash), ignore_errors=True)
        removed.append(run_id)
    if removed:
        logger.info("GC'd %d old staged create run(s): %s", len(removed), ", ".join(removed))
    return removed


__all__ = [
    "EVENTS_FILE",
    "RUN_FILE",
    "RunEvent",
    "RunEventLog",
    "RunRecord",
    "RunRecordStore",
    "RunStatus",
    "derive_run_status",
    "gc_old_runs",
    "is_valid_run_id",
    "list_runs",
    "load_run",
    "read_events",
]
