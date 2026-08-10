"""staged クリエイトの temp ワークスペース管理。

工程間 (spec→code→test) の中間成果物 (spec.md / 生成コード / 生成テスト) と
進捗 manifest を実ファイルとして保持し、後続工程が読み戻して整合を担保する。
SemMem (artifact/progress/failure ファクト) は内容を持たない進捗・学習シグナル
専用とし、**ファイル内容と manifest の真実はこのワークスペースが持つ**。

ディレクトリ構成 (``create_workspace_dir/{workspace_id}/``)::

    manifest.json            工程間ハンドオフ (atomic, fsync)
    spec.md                  spec 工程の設計仕様
    src/<logical_path>       生成コード (パス忠実)
    tests/<logical_path>     生成テスト (パス忠実)
    tests/_runs/<task>.<n>.json  テスト実行の詳細ログ

全書き込みは :class:`AtomicWriter` 経由 (crash 耐性)。manifest の更新は
in-process Lock 下の read-modify-write + fsync で原子化する。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from backend.io.atomic import AtomicWriter
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("loop.staged.workspace")

SCHEMA_VERSION = 1

FileKind = Literal["src", "test", "spec"]
Stage = Literal["spec", "code", "test"]

_KIND_SUBDIR: dict[str, str] = {"src": "src", "test": "tests", "spec": "."}


@dataclass(frozen=True)
class WorkspaceFile:
    """manifest の files エントリ (1 ファイル)。"""

    logical_path: str
    kind: FileKind
    workspace_path: str
    sha256: str
    bytes: int
    produced_by_task: str
    stage: Stage
    last_updated: float
    covers: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageTestResult:
    """テスト工程 1 実行分の結果。

    ``kind`` は記録の出所。``"pytest"`` は生成テストを実際に実行した結果、
    ``"smoke"`` は import/準拠の静的ゲート (テストは 1 つも走っていない)。
    両方を同じ ``test_results`` に入れているため、区別が無いと「スモーク合格」
    が「テスト合格」として集計される (実インシデント 2026-08-04 ライブ監査:
    テストファイル 0 件・UI は「生成ユニットテストは未実行です」と表示して
    いるのに manifest は ``tests_passing: true``)。空文字は出所不明として
    ``tests_passing`` の集計から外す (旧 manifest 互換、安全側)。
    """

    task_id: str
    passed: bool
    failed_count: int
    attempt: int
    summary: str
    output_tail: str
    ran_at: float
    run_ref: str = ""
    kind: str = ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_rel(logical_path: str) -> str:
    """logical_path をワークスペース外へ脱出できない相対パスに正規化する。"""
    p = logical_path.strip().replace("\\", "/")
    parts = [seg for seg in p.split("/") if seg and seg not in (".", "..")]
    return "/".join(parts) or "module"


@dataclass
class WorkspaceManager:
    """1 クリエイトセッション = 1 ワークスペース。

    ``open_or_create`` で生成 / 再アタッチする。manifest 変更は ``_update_manifest``
    に集約し、Lock + AtomicWriter(fsync) で原子化する。
    """

    root: Path
    workspace_id: str
    debug_logger: "DebugLogger | None" = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── 生成 / 位置解決 ────────────────────────────────────────────────
    @staticmethod
    def derive_workspace_id(*, project_id: str | None, session_id: str) -> str:
        """継続ターンで同一ワークスペースに再アタッチできる決定的 ID を返す。"""
        if project_id:
            return _safe_rel(project_id).replace("/", "_")
        return hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def open_or_create(
        cls,
        create_workspace_dir: Path | str,
        *,
        workspace_id: str,
        session_id: str,
        project_id: str,
        goal: str = "",
        debug_logger: "DebugLogger | None" = None,
    ) -> "WorkspaceManager":
        """``{create_workspace_dir}/{workspace_id}`` を生成 / 再アタッチする。"""
        root = Path(create_workspace_dir) / workspace_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").mkdir(exist_ok=True)
        (root / "tests" / "_runs").mkdir(parents=True, exist_ok=True)
        mgr = cls(root=root, workspace_id=workspace_id, debug_logger=debug_logger)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            now = time.time()
            mgr._write_manifest_raw({
                "schema_version": SCHEMA_VERSION,
                "workspace_id": workspace_id,
                "project_id": project_id,
                "session_id": session_id,
                "goal": goal,
                "created_at": now,
                "updated_at": now,
                "tasks": [],
                "files": {},
                "spec": None,
                "stage_notes": {},
                "test_results": {},
                "progress": {
                    "tasks_total": 0, "tasks_done": 0, "tasks_failed": 0,
                    "files_written": 0, "tests_passing": None,
                },
            })
            logger.info("staged workspace created: %s", root)
        else:
            logger.info("staged workspace re-attached: %s", root)
        return mgr

    def path(self, rel: str) -> Path:
        return self.root / rel

    def cleanup(self) -> None:
        """ワークスペース (含 .semmem / 生成物 / manifest) を削除する。

        ``create.staged.cleanup_workspace=true`` のときにリクエスト完了後に呼ぶ。
        失敗は握り潰す (一時ファイルのため致命的でない)。
        """
        import shutil

        try:
            shutil.rmtree(self.root, ignore_errors=True)
            logger.info("staged workspace cleaned up: %s", self.root)
        except Exception as exc:
            logger.warning("staged workspace cleanup failed: %s", exc)

    # ── spec ──────────────────────────────────────────────────────────
    def write_spec(self, content: str, *, task_id: str) -> WorkspaceFile:
        sha = _sha256(content)
        with AtomicWriter(self.root / "spec.md") as f:
            f.write(content)
        now = time.time()
        rec = {
            "workspace_path": "spec.md", "sha256": sha,
            "produced_by_task": task_id, "last_updated": now,
        }
        self._update_manifest(lambda m: m.__setitem__("spec", rec))
        return WorkspaceFile(
            logical_path="spec.md", kind="spec", workspace_path="spec.md",
            sha256=sha, bytes=len(content.encode("utf-8")),
            produced_by_task=task_id, stage="spec", last_updated=now,
        )

    def read_spec(self) -> str | None:
        p = self.root / "spec.md"
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    # ── flowchart (設計フローチャート mermaid) ─────────────────────────
    def write_flowchart(self, mermaid: str, *, task_id: str) -> WorkspaceFile:
        """設計フローチャート (mermaid) を flowchart.md に保存し manifest に記録。"""
        sha = _sha256(mermaid)
        with AtomicWriter(self.root / "flowchart.md") as f:
            f.write(mermaid)
        now = time.time()
        rec = {
            "workspace_path": "flowchart.md", "sha256": sha,
            "produced_by_task": task_id, "last_updated": now,
        }
        self._update_manifest(lambda m: m.__setitem__("flowchart", rec))
        return WorkspaceFile(
            logical_path="flowchart.md", kind="spec", workspace_path="flowchart.md",
            sha256=sha, bytes=len(mermaid.encode("utf-8")),
            produced_by_task=task_id, stage="spec", last_updated=now,
        )

    def read_flowchart(self) -> str | None:
        p = self.root / "flowchart.md"
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    # ── code / test ファイル (パス忠実) ───────────────────────────────
    def write_file(
        self,
        logical_path: str,
        content: str,
        *,
        kind: FileKind,
        stage: Stage,
        task_id: str,
        covers: tuple[str, ...] = (),
    ) -> WorkspaceFile:
        rel = _safe_rel(logical_path)
        ws_rel = f"{_KIND_SUBDIR[kind]}/{rel}" if kind != "spec" else rel
        target = self.root / ws_rel
        with AtomicWriter(target) as f:
            f.write(content)
        sha = _sha256(content)
        now = time.time()
        wf = WorkspaceFile(
            logical_path=rel, kind=kind, workspace_path=ws_rel, sha256=sha,
            bytes=len(content.encode("utf-8")), produced_by_task=task_id,
            stage=stage, last_updated=now, covers=tuple(covers),
        )

        def _mut(m: dict) -> None:
            m["files"][rel] = {
                "kind": kind, "workspace_path": ws_rel, "sha256": sha,
                "bytes": wf.bytes, "produced_by_task": task_id, "stage": stage,
                "last_updated": now, "covers": list(covers),
            }

        self._update_manifest(_mut)
        return wf

    def read_file(self, logical_path: str, *, kind: FileKind) -> str | None:
        rel = _safe_rel(logical_path)
        ws_rel = f"{_KIND_SUBDIR[kind]}/{rel}" if kind != "spec" else rel
        p = self.root / ws_rel
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    def file_map(self) -> dict[str, WorkspaceFile]:
        """logical_path -> :class:`WorkspaceFile` (整合性参照面)。"""
        m = self.read_manifest()
        out: dict[str, WorkspaceFile] = {}
        for lp, rec in (m.get("files") or {}).items():
            out[lp] = WorkspaceFile(
                logical_path=lp, kind=rec["kind"],
                workspace_path=rec["workspace_path"], sha256=rec["sha256"],
                bytes=int(rec.get("bytes", 0)),
                produced_by_task=rec.get("produced_by_task", ""),
                stage=rec.get("stage", "code"),
                last_updated=float(rec.get("last_updated", 0.0)),
                covers=tuple(rec.get("covers") or ()),
            )
        return out

    def list_files(
        self, *, kind: FileKind | None = None, stage: Stage | None = None,
    ) -> list[WorkspaceFile]:
        files = self.file_map().values()
        return [
            f for f in files
            if (kind is None or f.kind == kind)
            and (stage is None or f.stage == stage)
        ]

    # ── manifest: tasks / notes / test 結果 ───────────────────────────
    def upsert_task(
        self, *, task_id: str, title: str, stage: Stage, status: str,
        depends_on: list[str] | None = None, last_error: str | None = None,
    ) -> None:
        def _mut(m: dict) -> None:
            now = time.time()
            tasks = m["tasks"]
            for t in tasks:
                if t.get("task_id") == task_id:
                    t.update(title=title, stage=stage, status=status,
                             last_error=last_error, updated_at=now)
                    if depends_on is not None:
                        t["depends_on"] = list(depends_on)
                    if status in ("done", "failed"):
                        pass
                    return
            tasks.append({
                "task_id": task_id, "title": title, "stage": stage,
                "status": status, "depends_on": list(depends_on or []),
                "attempts": 0, "last_error": last_error, "updated_at": now,
            })

        self._update_manifest(_mut)

    def bump_task_attempt(self, task_id: str) -> None:
        def _mut(m: dict) -> None:
            for t in m["tasks"]:
                if t.get("task_id") == task_id:
                    t["attempts"] = int(t.get("attempts", 0)) + 1
                    t["updated_at"] = time.time()
                    return

        self._update_manifest(_mut)

    def record_stage_notes(
        self, task_id: str, *, decisions: list[str],
        open_questions: list[str] | None = None,
    ) -> None:
        def _mut(m: dict) -> None:
            m["stage_notes"][task_id] = {
                "decisions": list(decisions),
                "open_questions": list(open_questions or []),
                "note_ref": None,
            }

        self._update_manifest(_mut)

    def record_test_result(self, result: StageTestResult) -> None:
        run_ref = f"tests/_runs/{_safe_rel(result.task_id)}.{result.attempt}.json"
        with AtomicWriter(self.root / run_ref) as f:
            f.write(json.dumps({
                "task_id": result.task_id, "passed": result.passed,
                "failed_count": result.failed_count, "attempt": result.attempt,
                "summary": result.summary, "output_tail": result.output_tail,
                "ran_at": result.ran_at, "kind": result.kind,
            }, ensure_ascii=False, indent=2))

        def _mut(m: dict) -> None:
            m["test_results"][result.task_id] = {
                "passed": result.passed, "failed_count": result.failed_count,
                "attempt": result.attempt, "summary": result.summary,
                "output_tail": result.output_tail, "run_ref": run_ref,
                "ran_at": result.ran_at, "kind": result.kind,
            }

        self._update_manifest(_mut)

    def get_test_result(self, task_id: str) -> StageTestResult | None:
        m = self.read_manifest()
        rec = (m.get("test_results") or {}).get(task_id)
        if rec is None:
            return None
        return StageTestResult(
            task_id=task_id, passed=bool(rec.get("passed")),
            failed_count=int(rec.get("failed_count", 0)),
            attempt=int(rec.get("attempt", 0)),
            summary=str(rec.get("summary", "")),
            output_tail=str(rec.get("output_tail", "")),
            ran_at=float(rec.get("ran_at", 0.0)),
            run_ref=str(rec.get("run_ref", "")),
            kind=str(rec.get("kind", "")),
        )

    # ── manifest I/O ──────────────────────────────────────────────────
    def read_manifest(self) -> dict:
        p = self.root / "manifest.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_manifest_raw(self, manifest: dict) -> None:
        with AtomicWriter(self.root / "manifest.json", fsync=True) as f:
            f.write(json.dumps(manifest, ensure_ascii=False, indent=2))

    def _update_manifest(self, mutate: Callable[[dict], None]) -> dict:
        """Lock 下で read-modify-write (+ progress 再計算 + fsync)。"""
        with self._lock:
            m = self.read_manifest()
            if not m:
                logger.warning("manifest missing/corrupt during update: %s", self.root)
                return {}
            m.setdefault("tasks", [])
            m.setdefault("files", {})
            m.setdefault("stage_notes", {})
            m.setdefault("test_results", {})
            mutate(m)
            m["updated_at"] = time.time()
            m["progress"] = self._recompute_progress(m)
            self._write_manifest_raw(m)
            return m

    @staticmethod
    def _recompute_progress(m: dict) -> dict:
        tasks = m.get("tasks") or []
        files = m.get("files") or {}
        results = m.get("test_results") or {}
        # 実際にテストを走らせた記録だけを数える。静的スモークゲートも同じ
        # ``test_results`` に入るため、全件を見ると「テスト 0 件でも合格」に
        # なる。1 件も走っていなければ True/False ではなく None (未実行)。
        runs = [r for r in results.values() if r.get("kind") == "pytest"]
        return {
            "tasks_total": len(tasks),
            "tasks_done": sum(1 for t in tasks if t.get("status") == "done"),
            "tasks_failed": sum(1 for t in tasks if t.get("status") == "failed"),
            "files_written": sum(1 for r in files.values() if r.get("kind") == "src"),
            "tests_passing": all(r.get("passed") for r in runs) if runs else None,
        }
