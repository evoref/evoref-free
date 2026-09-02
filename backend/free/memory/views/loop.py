"""

EvorefLoop pillar が書き込む ``task`` / ``progress_marker`` /
``failure_pattern`` / ``artifact`` の管理、および Loop が reader として
読む ``policy`` / ``fewshot`` / ``personal_fact`` / ``preference`` / ``decision`` /
``commitment`` / ``project`` / ``create`` / ``create_task`` の読取 API を提供する。

追加した骨格に加えて、既存 ``LoopDriver`` /
``DefaultHarness`` / ``bootstrap_project_context`` / ``make_loop_artifact_hook``
等が依存していたタスク管理 / 失敗パターン統計 / artifact 一覧 / orphan 回収 /
user profile 収集 API を本 View に集約する。

注: 本 View は loop 所有 FactType の sanctioned な書込/読取 API 面であり、
全メソッドが現時点で production から呼ばれているわけではない (例:
``count_artifact_facts`` は API 面として提供するが現状呼出元なし)。
``write_failure_pattern`` は ralph loop に加え sleep-time の MDP 抽出
(``extraction._persist_failure_patterns_via_view``) からも利用される。

## スコープと store 分離

Loop driver は複数スコープ (``global`` + ``project:<id>``) を横断する。
本 View は:

- ``stores``: 読取対象の全スコープ (``list[SemanticFactStoreProtocol]``)
- ``writeback_store``: 書込先 (通常は project スコープ)

の 2 系統を受け取る。

## subject 命名

subject namespace を ``loop.*`` に一括移行済

- ``task``: ``loop.task.<task_id>``
- ``progress_marker``: ``loop.progress.<task_id>``
- ``failure_pattern``: ``loop.failure.<signature>``
- ``artifact``: ``loop.artifact.<task_id>.<path_sha1[:12]>``
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from backend.free.memory.ownership import Pillar
from backend.free.memory.protocols import SemanticFactStoreProtocol
from backend.free.memory.types import (
    FactType,
    MemoryMode,
    SemanticFact,
    TaskStatus,
    make_fact,
)
from backend.free.memory.views.base import (
    FactViewBase,
    merge_active_facts_across_stores,
    safe_json_loads as _safe_json_loads,
)

# ──────────────────────────────────────────────────────────────────────────
# 定数 (loop/ モジュールと整合)
# ──────────────────────────────────────────────────────────────────────────

TASK_SUBJECT_PREFIX = "loop.task."
"""task ファクトの subject prefix"""

PROGRESS_MARKER_PREFIX = "loop.progress."
"""progress_marker ファクトの subject prefix"""

FAILURE_SUBJECT_PREFIX = "loop.failure."
"""failure_pattern ファクトの subject prefix"""

ARTIFACT_SUBJECT_PREFIX = "loop.artifact."
"""artifact ファクトの subject prefix"""

TASK_PREDICATE = "defines"
PROGRESS_PREDICATE = "reached"
FAILURE_PREDICATE = "failed_with"
ARTIFACT_PREDICATE = "produced"

_PATH_SHA1_PREFIX_LEN = 12

_VALID_TASK_STATUSES: frozenset[str] = frozenset(
    {"open", "in_progress", "done", "failed"},
)

_USER_PROFILE_TYPES: tuple[FactType, ...] = ("personal_fact", "preference")
"""``@self`` 仮想カートリッジが集める Mem owned ファクト型。readers に ``"loop"`` を含む。"""


# ──────────────────────────────────────────────────────────────────────────
# BootstrapResult
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoopBootstrapResult:
    """:meth:`LoopFactView.bootstrap_context` の返却型。

    Attributes:
        project_id: 対象プロジェクト ID。
        active_policies: confidence 閾値を満たす有効 policy 群。
        pinned_global: global スコープの pinned ファクト。
        pinned_project: project スコープの pinned ファクト。
        candidate_failure_patterns: 参照対象の failure_pattern 群。
        skipped_below_confidence: confidence 閾値未満でスキップした policy 数。
        policy_activation_min_confidence: policy 有効化の下限 confidence。
        artifact_count: project スコープの artifact 総数 (観測用)。
    """

    project_id: str
    active_policies: list[SemanticFact] = field(default_factory=list)
    pinned_global: list[SemanticFact] = field(default_factory=list)
    pinned_project: list[SemanticFact] = field(default_factory=list)
    candidate_failure_patterns: list[SemanticFact] = field(default_factory=list)
    skipped_below_confidence: int = 0
    policy_activation_min_confidence: float = 0.7
    artifact_count: int = 0

    def tier1_facts(self) -> list[SemanticFact]:
        """Tier 1 (prompt injection) 対象 (active policy + pinned) を id 重複排除して返す。"""
        seen: set[str] = set()
        out: list[SemanticFact] = []
        for fact in (
            *self.active_policies,
            *self.pinned_global,
            *self.pinned_project,
        ):
            if fact.id in seen:
                continue
            seen.add(fact.id)
            out.append(fact)
        return out

    def total_tier1(self) -> int:
        return len(self.tier1_facts())

    def as_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "active_policies": len(self.active_policies),
            "pinned_global": len(self.pinned_global),
            "pinned_project": len(self.pinned_project),
            "candidate_failure_patterns": len(self.candidate_failure_patterns),
            "skipped_below_confidence": self.skipped_below_confidence,
            "policy_activation_min_confidence": (
                self.policy_activation_min_confidence
            ),
            "total_tier1": self.total_tier1(),
            "artifact_count": self.artifact_count,
        }


# ──────────────────────────────────────────────────────────────────────────
# 内部ユーティリティ (JSON パース / subject 生成)
# ──────────────────────────────────────────────────────────────────────────


def _parse_task_status(fact: SemanticFact) -> TaskStatus:
    status = _safe_json_loads(fact.object).get("status", "open")
    if status in _VALID_TASK_STATUSES:
        return status  # type: ignore[return-value]
    return "open"


def _parse_task_salience(fact: SemanticFact) -> float:
    try:
        return float(_safe_json_loads(fact.object).get("salience", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _parse_task_id(fact: SemanticFact) -> str:
    """``fact.object`` から task_id を取り出す (JSON 壊れ時は subject prefix 除去)。"""
    payload = _safe_json_loads(fact.object)
    tid = payload.get("task_id")
    if isinstance(tid, str) and tid:
        return tid
    if fact.subject.startswith(TASK_SUBJECT_PREFIX):
        return fact.subject[len(TASK_SUBJECT_PREFIX):]
    return ""


def _parse_task_depends_on(fact: SemanticFact) -> list[str]:
    raw = _safe_json_loads(fact.object).get("depends_on", [])
    if not isinstance(raw, list):
        return []
    return [str(d) for d in raw]


def _path_sha1_prefix(file_path: str) -> str:
    digest = hashlib.sha1(file_path.encode("utf-8")).hexdigest()
    return digest[:_PATH_SHA1_PREFIX_LEN]


def build_artifact_subject(task_id: str, file_path: str) -> str:
    """``loop.artifact.<task_id>.<path_sha1_prefix>`` を返す"""
    if not task_id:
        raise ValueError("task_id must be non-empty")
    if not file_path:
        raise ValueError("file_path must be non-empty")
    return f"{ARTIFACT_SUBJECT_PREFIX}{task_id}.{_path_sha1_prefix(file_path)}"


# ──────────────────────────────────────────────────────────────────────────
# LoopFactView
# ──────────────────────────────────────────────────────────────────────────


class LoopFactView(FactViewBase):
    """EvorefLoop pillar の Fact View。

    Attributes:
        pillar: ``"loop"`` 固定。
    """

    pillar: Pillar = "loop"

    def __init__(
        self,
        *,
        stores: Iterable[SemanticFactStoreProtocol],
        writeback_store: SemanticFactStoreProtocol,
    ) -> None:
        """Args:
            stores: 読取対象の全スコープ (global + project)。``writeback_store``
                と重複して含まれていても構わない。
            writeback_store: Loop 所有 fact の書込先ストア (通常は project スコープ)。
        """
        self._stores: list[SemanticFactStoreProtocol] = list(stores)
        if not self._stores:
            raise ValueError("LoopFactView requires at least one store")
        self._writeback_store: SemanticFactStoreProtocol = writeback_store

    # ──────────────────────────────────────────────────────────────────
    # 汎用アクセサ (低レベル)
    # ──────────────────────────────────────────────────────────────────

    def get_fact(self, fact_id: str) -> SemanticFact | None:
        """writeback_store から ID でファクトを取得する (未存在は ``None``)。"""
        return self._writeback_store.get_fact(fact_id)

    @property
    def writeback_store(self) -> SemanticFactStoreProtocol:
        """**テスト専用** の writeback_store 参照 (本番コードは view 経由で使うこと)。"""
        return self._writeback_store

    # ──────────────────────────────────────────────────────────────────
    # 読取系 — task
    # ──────────────────────────────────────────────────────────────────

    def list_task_facts(
        self,
        project_id: str,
        *,
        status_filter: set[str] | None = None,
    ) -> list[SemanticFact]:
        """指定 project の task ファクトを全ストア横断で収集する (superseded 除外)。

        ``status_filter`` に ``{"open", "in_progress"}`` 等を指定するとその
        status を持つファクトのみを返す。
        """
        self._assert_read("task")
        project_scope = SemanticFact.make_project_scope(project_id)
        out: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("task"):
                if fact.id in seen or fact.scope != project_scope:
                    continue
                if status_filter is not None:
                    if _parse_task_status(fact) not in status_filter:
                        continue
                out.append(fact)
                seen.add(fact.id)
        return out

    def get_open_tasks(self, project_id: str) -> list[SemanticFact]:
        """``project_id`` の ``open`` / ``in_progress`` 状態の task ファクトを返す。"""
        return self.list_task_facts(
            project_id, status_filter={"open", "in_progress"},
        )

    def get_next_task(self, project_id: str) -> SemanticFact | None:
        """次に実行すべき task ファクトを返す (salience 降順 → created_at 昇順)。

        依存関係 (``depends_on``) は考慮しない単純版。依存グラフを考慮する
        場合は :meth:`pick_next_task_with_deps` を用いる。
        """
        opens = self.get_open_tasks(project_id)
        opens.sort(key=lambda f: (-_parse_task_salience(f), f.created_at))
        return opens[0] if opens else None

    def pick_next_task_with_deps(self, project_id: str) -> SemanticFact | None:
        """依存グラフを考慮して次タスクを選定する。

        アルゴリズム:

        1. ``status=open`` のタスクのみ対象
        2. ``depends_on`` の task_id がすべて ``status=done`` のものに絞る
           (依存先が ``failed`` / ``in_progress`` / ``open`` / 未登録なら除外)
        3. 残り候補から ``salience`` 最大 (同点は ``created_at`` / ``task_id`` 昇順)
        4. 候補ゼロなら ``None``
        """
        all_tasks = self.list_task_facts(project_id)
        by_task_id: dict[str, SemanticFact] = {
            _parse_task_id(t): t for t in all_tasks
        }
        done_ids = {
            _parse_task_id(t)
            for t in all_tasks
            if _parse_task_status(t) == "done"
        }
        candidates: list[SemanticFact] = []
        for task in all_tasks:
            if _parse_task_status(task) != "open":
                continue
            deps = _parse_task_depends_on(task)
            unmet = [d for d in deps if d not in done_ids]
            missing = [d for d in deps if d not in by_task_id]
            if unmet or missing:
                continue
            candidates.append(task)
        if not candidates:
            return None
        candidates.sort(
            key=lambda f: (
                -_parse_task_salience(f),
                f.created_at,
                _parse_task_id(f),
            ),
        )
        return candidates[0]

    # ──────────────────────────────────────────────────────────────────
    # 読取系 — failure_pattern
    # ──────────────────────────────────────────────────────────────────

    def get_matching_failure_patterns(
        self,
        *,
        project_id: str,
        signature: str,
    ) -> list[SemanticFact]:
        """``failure_signature`` が一致する失敗パターンを返す (superseded 除外)。"""
        self._assert_read("failure_pattern")
        project_scope = SemanticFact.make_project_scope(project_id)
        result: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("failure_pattern"):
                if fact.id in seen:
                    continue
                if fact.scope != project_scope:
                    continue
                if fact.failure_signature != signature:
                    continue
                result.append(fact)
                seen.add(fact.id)
        return result

    def list_failure_pattern_facts(
        self,
        project_id: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """``project_id`` スコープの failure_pattern ファクトを全ストアから収集する。"""
        self._assert_read("failure_pattern")
        project_scope = SemanticFact.make_project_scope(project_id)
        out: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type(
                "failure_pattern", include_superseded=include_superseded,
            ):
                if fact.id in seen or fact.scope != project_scope:
                    continue
                out.append(fact)
                seen.add(fact.id)
        return out

    def aggregate_failure_patterns(
        self,
        project_id: str,
    ) -> tuple[int, dict[str, int]]:
        """project の failure_pattern を ``error_type`` 別に集計する。

        Returns:
            ``(total_occurrences, {error_type: occurrences})``。
            ``error_type`` 欠損ファクトは ``"unknown"`` キーに加算される。
        """
        facts = self.list_failure_pattern_facts(project_id)
        total = 0
        by_type: dict[str, int] = {}
        for fact in facts:
            payload = _safe_json_loads(fact.object)
            try:
                occurrences = int(payload.get("occurrences") or 1)
            except (TypeError, ValueError):
                occurrences = 1
            if occurrences < 1:
                occurrences = 1
            error_type = str(payload.get("error_type") or "").strip() or "unknown"
            by_type[error_type] = by_type.get(error_type, 0) + occurrences
            total += occurrences
        return total, by_type

    # ──────────────────────────────────────────────────────────────────
    # 読取系 — progress_marker
    # ──────────────────────────────────────────────────────────────────

    def list_progress_marker_facts(
        self,
        project_id: str,
    ) -> list[SemanticFact]:
        """指定 project の progress_marker ファクトを全ストアから収集する (superseded 除外)。"""
        self._assert_read("progress_marker")
        project_scope = SemanticFact.make_project_scope(project_id)
        out: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("progress_marker"):
                if fact.id in seen or fact.scope != project_scope:
                    continue
                out.append(fact)
                seen.add(fact.id)
        return out

    # ──────────────────────────────────────────────────────────────────
    # 読取系 — artifact
    # ──────────────────────────────────────────────────────────────────

    def list_artifact_facts(
        self,
        project_id: str,
        *,
        task_id: str | None = None,
    ) -> list[SemanticFact]:
        """指定 project の artifact ファクトを全ストアから収集する (superseded 除外)。

        ``task_id`` を指定すると subject 前方一致 (``loop.artifact.<task_id>.``)
        に絞る。
        """
        self._assert_read("artifact")
        project_scope = SemanticFact.make_project_scope(project_id)
        prefix: str | None = None
        if task_id is not None:
            prefix = f"{ARTIFACT_SUBJECT_PREFIX}{task_id}."
        out: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.search_by_type("artifact"):
                if fact.id in seen or fact.scope != project_scope:
                    continue
                if prefix is not None and not fact.subject.startswith(prefix):
                    continue
                out.append(fact)
                seen.add(fact.id)
        return out

    def count_artifact_facts(self, project_id: str) -> int:
        """``project_id`` スコープの artifact 件数 (superseded 除外)。"""
        return len(self.list_artifact_facts(project_id))

    # ──────────────────────────────────────────────────────────────────
    # 読取系 — policy / fewshot (他 pillar 所有だが Loop が reader)
    # ──────────────────────────────────────────────────────────────────

    def get_active_policies(
        self,
        *,
        min_confidence: float = 0.7,
        mode: MemoryMode | None = None,
    ) -> list[SemanticFact]:
        """Loop が参照する有効 policy 群 (superseded 除外)。"""
        self._assert_read("policy")
        return merge_active_facts_across_stores(
            self._stores, "policy", min_confidence=min_confidence, mode=mode,
        )

    def get_active_fewshots(
        self,
        *,
        mode: MemoryMode | None = None,
        min_confidence: float = 0.0,
    ) -> list[SemanticFact]:
        """Loop が参照する有効 fewshot 群 (superseded 除外)。"""
        self._assert_read("fewshot")
        return merge_active_facts_across_stores(
            self._stores, "fewshot", min_confidence=min_confidence, mode=mode,
        )

    # ──────────────────────────────────────────────────────────────────
    # 読取系 — pinned / user profile
    # ──────────────────────────────────────────────────────────────────

    def get_pinned_facts(self) -> list[SemanticFact]:
        """全ストアの pinned ファクトを結合して返す (重複 ID は除外)。"""
        result: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for fact in store.pinned_facts():
                if fact.id in seen:
                    continue
                result.append(fact)
                seen.add(fact.id)
        return result

    def get_user_profile(self, *, limit: int = 20) -> list[SemanticFact]:
        """``personal_fact`` / ``preference`` を全ストアから収集する (superseded 除外)。

        readers に ``"loop"`` を追加済 (``personal_fact`` / ``preference``)。
        並び順: pinned 優先 → access_count 降順 → accessed_at 降順 → id 昇順。
        ``limit<=0`` なら全件。
        """
        for ftype in _USER_PROFILE_TYPES:
            self._assert_read(ftype)
        out: list[SemanticFact] = []
        seen: set[str] = set()
        for store in self._stores:
            for ftype in _USER_PROFILE_TYPES:
                for fact in store.search_by_type(
                    ftype, include_superseded=False,
                ):
                    if fact.id in seen:
                        continue
                    out.append(fact)
                    seen.add(fact.id)
        out.sort(
            key=lambda f: (
                0 if f.pinned else 1,
                -f.access_count,
                -f.accessed_at,
                f.id,
            ),
        )
        return out[:limit] if limit > 0 else out

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — task
    # ──────────────────────────────────────────────────────────────────

    def add_facts(self, facts: Iterable[SemanticFact]) -> list[SemanticFact]:
        """Loop 所有のファクト群 (合成済み task graph 等) を writeback_store へ追加する。

        全件の owner / subject namespace 検証を **先に** 通してから書く
        (途中で 1 件だけ弾かれて半端な graph が残らないように)。
        ``chat_stream_staged`` が隔離ストアへ ``store.add_fact`` を直呼びして
        いた経路の受け皿 (2026-09-02 監査 M22)。

        Raises:
            WriteOwnershipError: いずれかの ``fact.type`` の owner が Loop でない。
            SubjectNamespaceError: いずれかの ``fact.subject`` が ``mem.*`` /
                ``learn.*`` namespace を持つ。
        """
        items = list(facts)
        for fact in items:
            self._assert_write(fact.type)
            self._assert_subject_owner(fact.subject)
        return [self._writeback_store.add_fact(fact) for fact in items]

    def update_task_status(
        self,
        fact_id: str,
        status: TaskStatus,
    ) -> SemanticFact:
        """既存 task ファクトの ``status`` を更新する。

        ``object`` (JSON) の ``status`` フィールドのみ差し替える。
        ``accessed_at`` は store 側で自動更新。

        Raises:
            KeyError: ``fact_id`` が存在しない。
            ValueError: 対象ファクトの ``type`` が ``task`` でない / ``status``
                が許容外。
            WriteOwnershipError: owner が Loop でない (実装上は必ず Loop)。
        """
        self._assert_write("task")
        if status not in _VALID_TASK_STATUSES:
            raise ValueError(
                f"invalid task status: {status!r} "
                f"(expected one of {sorted(_VALID_TASK_STATUSES)!r})",
            )
        fact = self._writeback_store.get_fact(fact_id)
        if fact is None:
            raise KeyError(fact_id)
        if fact.type != "task":
            raise ValueError(
                f"expected task fact, got type={fact.type!r} (id={fact_id})",
            )
        payload = _safe_json_loads(fact.object)
        payload["status"] = status
        return self._writeback_store.update_fact(
            fact_id,
            object=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — progress_marker
    # ──────────────────────────────────────────────────────────────────

    def write_progress_marker(
        self,
        *,
        project_id: str,
        task_id: str,
        title: str = "",
        status: str = "done",
        iteration: int | None = None,
        trace_id: str | None = None,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> SemanticFact:
        """進捗マーカーを冪等に書き込む (既存 subject があれば occurrences 加算)。

        subject は ``loop.progress.<task_id>``
        """
        self._assert_write("progress_marker")
        if not project_id:
            raise ValueError("project_id must be non-empty")
        if not task_id:
            raise ValueError("task_id must be non-empty")
        t = time.time() if now is None else float(now)
        subject = f"{PROGRESS_MARKER_PREFIX}{task_id}"
        project_scope = SemanticFact.make_project_scope(project_id)

        existing = [
            f
            for f in self._writeback_store.search_by_subject(subject)
            if f.scope == project_scope and f.type == "progress_marker"
        ]
        if existing:
            latest = sorted(
                existing,
                key=lambda f: (-f.accessed_at, -f.created_at, f.id),
            )[0]
            prev_payload = _safe_json_loads(latest.object)
            prev_occurrences = int(prev_payload.get("occurrences") or 0)
            merged_iteration = (
                iteration if iteration is not None
                else prev_payload.get("iteration")
            )
            merged_trace_id = (
                trace_id if trace_id is not None
                else prev_payload.get("trace_id")
            )
            merged_title = title or str(prev_payload.get("title") or "")
            payload: dict[str, Any] = {
                "task_id": task_id,
                "title": merged_title,
                "status": status,
                "completed_at": t,
                "occurrences": prev_occurrences + 1,
            }
            if merged_iteration is not None:
                try:
                    payload["iteration"] = int(merged_iteration)
                except (TypeError, ValueError):
                    pass
            if merged_trace_id is not None:
                payload["trace_id"] = str(merged_trace_id)
            return self._writeback_store.update_fact(
                latest.id,
                object=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )

        payload = {
            "task_id": task_id,
            "title": title,
            "status": status,
            "completed_at": t,
            "occurrences": 1,
        }
        if iteration is not None:
            payload["iteration"] = int(iteration)
        if trace_id is not None:
            payload["trace_id"] = trace_id

        fact = make_fact(
            subject=subject,
            predicate=PROGRESS_PREDICATE,
            object_=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            type="progress_marker",
            scope=project_scope,
            mode_origin="create",
            confidence=confidence,
            now=t,
            trace_id=trace_id,
        )
        return self._writeback_store.add_fact(fact)

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — failure_pattern
    # ──────────────────────────────────────────────────────────────────

    def write_failure_pattern(
        self,
        *,
        project_id: str,
        signature: str,
        error_type: str,
        normalized_file_path: str,
        last_actions: list[str],
        mitigation: str | None = None,
        outcome_label: str | None = None,
        output_tail: str | None = None,
        confidence: float = 0.7,
        trace_id: str | None = None,
        now: float | None = None,
    ) -> SemanticFact:
        """失敗パターンを冪等に書き込む (``signature`` で重複検出 + occurrences 加算)。

        低レベルプリミティブ。``GateResult`` から signature を計算するのは
        呼び出し側の責務。``outcome_label`` を渡すと ``outcomes_history`` に追記する。
        ``output_tail`` を渡すと最新の gate stdout/stderr 末尾を ``object`` に
        含める (次イテレーションの LLM に失敗の生出力を観測させる用途)。
        """
        self._assert_write("failure_pattern")
        t = time.time() if now is None else float(now)
        subject = f"{FAILURE_SUBJECT_PREFIX}{signature}"
        project_scope = SemanticFact.make_project_scope(project_id)

        payload: dict[str, Any] = {
            "error_type": error_type,
            "normalized_file_path": normalized_file_path,
            "last_actions": list(last_actions)[-3:],
            "occurrences": 1,
            "outcomes_history": [outcome_label] if outcome_label else [],
        }
        if mitigation is not None:
            payload["mitigation"] = mitigation
        if output_tail:
            payload["output_tail"] = output_tail

        existing = self._writeback_store.search_by_subject(subject)
        active = [f for f in existing if not f.superseded_by]
        if active:
            prev = active[0]
            prev_payload = _safe_json_loads(prev.object)
            payload["occurrences"] = int(prev_payload.get("occurrences", 1)) + 1
            prev_outcomes = prev_payload.get("outcomes_history")
            if isinstance(prev_outcomes, list):
                merged_outcomes = [str(o) for o in prev_outcomes]
                if outcome_label and outcome_label not in merged_outcomes:
                    merged_outcomes.append(outcome_label)
                payload["outcomes_history"] = merged_outcomes[-5:]
            if "mitigation" not in payload and "mitigation" in prev_payload:
                payload["mitigation"] = prev_payload["mitigation"]
            # output_tail は最新失敗のものを優先 (古いものは捨てる)。
            # 新しい値が無い場合のみ過去の値を維持する。
            if "output_tail" not in payload and "output_tail" in prev_payload:
                payload["output_tail"] = prev_payload["output_tail"]
            return self._writeback_store.update_fact(
                prev.id,
                object=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                confidence=confidence,
            )

        fact = make_fact(
            subject=subject,
            predicate=FAILURE_PREDICATE,
            object_=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            type="failure_pattern",
            scope=project_scope,
            mode_origin="create",
            confidence=confidence,
            now=t,
            trace_id=trace_id,
            failure_signature=signature,
        )
        return self._writeback_store.add_fact(fact)

    # ──────────────────────────────────────────────────────────────────
    # 書込系 — artifact
    # ──────────────────────────────────────────────────────────────────

    def write_artifact(
        self,
        *,
        project_id: str,
        task_id: str,
        file_path: str,
        diff_sha1: str,
        lines_added: int,
        lines_removed: int,
        gate_passed: bool,
        action_kind: str = "edit_file",
        related_progress_marker: str = "",
        iteration: int | None = None,
        trace_id: str | None = None,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> SemanticFact | None:
        """成果物 (ファイル編集) のメタデータを冪等に書き込む。

        同一 subject (``loop.artifact.<task_id>.<path_sha1>``) かつ同一
        ``diff_sha1`` の artifact が既に存在する場合は重複書込をスキップし
        ``None`` を返す。
        """
        self._assert_write("artifact")
        if not project_id:
            raise ValueError("project_id must be non-empty")
        if not task_id:
            raise ValueError("task_id must be non-empty")
        t = time.time() if now is None else float(now)
        subject = build_artifact_subject(task_id, file_path)
        project_scope = SemanticFact.make_project_scope(project_id)

        existing = [
            f
            for f in self._writeback_store.search_by_subject(subject)
            if f.scope == project_scope and f.type == "artifact"
        ]
        for fact in existing:
            prev_payload = _safe_json_loads(fact.object)
            if prev_payload.get("diff_sha1") == diff_sha1:
                return None

        related_progress = (
            related_progress_marker
            or f"{PROGRESS_MARKER_PREFIX}{task_id}"
        )
        payload: dict[str, Any] = {
            "file_path": file_path,
            "diff_sha1": diff_sha1,
            "lines_added": int(lines_added),
            "lines_removed": int(lines_removed),
            "gate_passed": bool(gate_passed),
            "related_task_id": task_id,
            "related_progress_marker": related_progress,
            "created_at": t,
            "action_kind": action_kind,
        }
        if iteration is not None:
            payload["iteration"] = int(iteration)
        if trace_id is not None:
            payload["trace_id"] = trace_id

        fact = make_fact(
            subject=subject,
            predicate=ARTIFACT_PREDICATE,
            object_=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            type="artifact",
            scope=project_scope,
            mode_origin="create",
            confidence=confidence,
            now=t,
            trace_id=trace_id,
        )
        return self._writeback_store.add_fact(fact)

    # ──────────────────────────────────────────────────────────────────
    # 統合系 / orphan 回収
    # ──────────────────────────────────────────────────────────────────

    def consolidate_failure_patterns(
        self,
        *,
        project_id: str | None = None,
    ) -> dict[str, int]:
        """同一 ``failure_signature`` の failure_pattern を統合する (Step 13 相当)。

        各グループで最新の ``accessed_at`` を持つファクトを kept として残し、
        他を supersede する。さらに以下のペイロードマージを行う:

        - ``occurrences``: 全ファクトの合計
        - ``outcomes_history``: 全ファクトのマージ (順序保持 + 末尾 5 件に制限)
        - ``mitigation``: 候補 (``accessed_at`` 降順) の最初の非 ``None`` 値を採用

        Args:
            project_id: 指定時はその project スコープのみ対象。``None`` の
                場合は writeback_store の全 failure_pattern が対象。

        Returns:
            ``{"groups_scanned", "merged_groups", "superseded_facts",
            "kept_facts", "mitigation_updated"}`` のサマリ dict。
        """
        self._assert_write("failure_pattern")
        scope_filter = (
            SemanticFact.make_project_scope(project_id)
            if project_id is not None else None
        )
        groups: dict[str, list[SemanticFact]] = {}
        for fact in self._writeback_store.search_by_type("failure_pattern"):
            if fact.superseded_by:
                continue
            if scope_filter is not None and fact.scope != scope_filter:
                continue
            sig = fact.failure_signature or ""
            if not sig:
                continue
            groups.setdefault(sig, []).append(fact)

        groups_scanned = len(groups)
        merged_groups = 0
        superseded_facts = 0
        kept_facts = 0
        mitigation_updated = 0
        for facts in groups.values():
            if len(facts) <= 1:
                kept_facts += 1
                continue
            facts.sort(
                key=lambda f: (-f.accessed_at, -f.created_at, f.id),
            )
            keep = facts[0]
            losers = facts[1:]
            kept_facts += 1

            # ペイロードのマージ (occurrences + outcomes_history + mitigation)
            keep_payload = _safe_json_loads(keep.object)
            total_occ = 0
            merged_outcomes: list[str] = []
            merged_last_actions: list[str] = list(
                keep_payload.get("last_actions") or [],
            )
            mitigation_candidates: list[tuple[float, str]] = []
            prev_mitigation = keep_payload.get("mitigation")
            if prev_mitigation is not None:
                mitigation_candidates.append(
                    (keep.accessed_at, str(prev_mitigation)),
                )
            fallback_error_type = str(keep_payload.get("error_type") or "")
            fallback_file_path = str(
                keep_payload.get("normalized_file_path") or "",
            )
            for fact in facts:
                p = _safe_json_loads(fact.object)
                total_occ += max(1, int(p.get("occurrences") or 1))
                outcomes = p.get("outcomes_history") or []
                if isinstance(outcomes, list):
                    for o in outcomes:
                        if o and str(o) not in merged_outcomes:
                            merged_outcomes.append(str(o))
                if not fallback_error_type and p.get("error_type"):
                    fallback_error_type = str(p["error_type"])
                if (
                    not fallback_file_path
                    and p.get("normalized_file_path")
                ):
                    fallback_file_path = str(p["normalized_file_path"])
                if not merged_last_actions and p.get("last_actions"):
                    la = p["last_actions"]
                    if isinstance(la, list):
                        merged_last_actions = [str(a) for a in la]
                if fact is keep:
                    continue
                if p.get("mitigation") is not None:
                    mitigation_candidates.append(
                        (fact.accessed_at, str(p["mitigation"])),
                    )

            new_mitigation = prev_mitigation
            if mitigation_candidates:
                mitigation_candidates.sort(key=lambda t: t[0], reverse=True)
                new_mitigation = mitigation_candidates[0][1]
            mitigation_changed = new_mitigation != prev_mitigation

            new_payload: dict[str, Any] = {
                "error_type": fallback_error_type,
                "normalized_file_path": fallback_file_path,
                "last_actions": merged_last_actions[-3:],
                "occurrences": total_occ,
                "outcomes_history": merged_outcomes[-5:],
            }
            if new_mitigation is not None:
                new_payload["mitigation"] = new_mitigation

            self._writeback_store.update_fact(
                keep.id,
                object=json.dumps(
                    new_payload, ensure_ascii=False, sort_keys=True,
                ),
            )
            if mitigation_changed:
                mitigation_updated += 1

            for old in losers:
                self._writeback_store.supersede(old.id, keep.id)
                superseded_facts += 1
            merged_groups += 1
        return {
            "groups_scanned": groups_scanned,
            "merged_groups": merged_groups,
            "superseded_facts": superseded_facts,
            "kept_facts": kept_facts,
            "mitigation_updated": mitigation_updated,
        }

    def reopen_orphan_in_progress_tasks(
        self,
        project_id: str,
    ) -> list[SemanticFact]:
        """``in_progress`` のまま残存している orphan task を ``open`` に戻す。

        driver プロセスが ``executor.execute`` await 中に強制終了 (OS kill /
        未捕捉例外等) した場合、task ファクトは ``in_progress`` のまま SemMem
        に残る。新 driver の ``pick_next_task`` は ``status=open`` しか拾わない
        ため、本メソッドを bootstrap / driver.start の直後に呼ぶことで orphan
        を回収する

        Returns:
            open に戻された task ファクトのリスト (決定的順)。
        """
        self._assert_write("task")
        targets = self.list_task_facts(
            project_id, status_filter={"in_progress"},
        )
        targets.sort(key=lambda f: (f.created_at, _parse_task_id(f)))
        reopened: list[SemanticFact] = []
        for task in targets:
            try:
                updated = self.update_task_status(task.id, "open")
            except (KeyError, ValueError):
                continue
            reopened.append(updated)
        return reopened

    def bootstrap_context(
        self,
        *,
        project_id: str,
        policy_activation_min_confidence: float = 0.7,
    ) -> LoopBootstrapResult:
        """Tier 1 コンテキスト素材 (policy / pinned / failure_pattern) を収集する。"""
        if not project_id:
            raise ValueError("project_id must be non-empty")
        self._assert_read("policy")
        self._assert_read("failure_pattern")

        project_scope = SemanticFact.make_project_scope(project_id)

        active_policies: list[SemanticFact] = []
        pinned_global: list[SemanticFact] = []
        pinned_project: list[SemanticFact] = []
        candidate_failures: list[SemanticFact] = []
        skipped = 0
        artifact_count = 0

        seen_policy: set[str] = set()
        seen_failure: set[str] = set()
        seen_pinned: set[str] = set()
        seen_artifact: set[str] = set()

        for store in self._stores:
            for fact in store.search_by_type("policy"):
                if fact.id in seen_policy or fact.superseded_by:
                    continue
                if fact.confidence < policy_activation_min_confidence:
                    skipped += 1
                    continue
                active_policies.append(fact)
                seen_policy.add(fact.id)
            for fact in store.search_by_type("failure_pattern"):
                if fact.id in seen_failure or fact.superseded_by:
                    continue
                if fact.scope != project_scope:
                    continue
                candidate_failures.append(fact)
                seen_failure.add(fact.id)
            for fact in store.search_by_type("artifact"):
                if fact.id in seen_artifact:
                    continue
                if fact.scope == project_scope and not fact.superseded_by:
                    artifact_count += 1
                    seen_artifact.add(fact.id)
            for fact in store.pinned_facts():
                if fact.id in seen_pinned or fact.superseded_by:
                    continue
                if fact.scope == "global":
                    pinned_global.append(fact)
                    seen_pinned.add(fact.id)
                elif fact.scope == project_scope:
                    pinned_project.append(fact)
                    seen_pinned.add(fact.id)

        active_policies.sort(key=lambda f: (f.created_at, f.id))
        pinned_global.sort(key=lambda f: (f.created_at, f.id))
        pinned_project.sort(key=lambda f: (f.created_at, f.id))
        candidate_failures.sort(
            key=lambda f: (-f.accessed_at, -f.created_at, f.id),
        )

        return LoopBootstrapResult(
            project_id=project_id,
            active_policies=active_policies,
            pinned_global=pinned_global,
            pinned_project=pinned_project,
            candidate_failure_patterns=candidate_failures,
            skipped_below_confidence=skipped,
            policy_activation_min_confidence=policy_activation_min_confidence,
            artifact_count=artifact_count,
        )


__all__ = [
    "ARTIFACT_PREDICATE",
    "ARTIFACT_SUBJECT_PREFIX",
    "FAILURE_PREDICATE",
    "FAILURE_SUBJECT_PREFIX",
    "LoopBootstrapResult",
    "LoopFactView",
    "PROGRESS_MARKER_PREFIX",
    "PROGRESS_PREDICATE",
    "TASK_PREDICATE",
    "TASK_SUBJECT_PREFIX",
    "build_artifact_subject",
]
