"""Sleep-time update sub-modules

sleep-time cycle (Steps 6-13) の各工程を責務別に独立 module へ分離した。
呼び出し元 (``backend.free.memory.sleep_update.SleepTimeWorker``) は本パッケージの
関数を順次呼び出すオーケストレータとして機能し、実ロジックは以下の module に
閉じ込める。

Module 一覧:

- :mod:`.extraction` — Step 8: SemanticFact Extractor (Chat / Coding / MDP)
- :mod:`.promotion` — Step 9: history summary → ``decision`` / ``commitment`` 昇格
- :mod:`.gc` — Step 9: ``semmem_limits`` に基づく SemMem GC 実行
- :mod:`.failure_consolidator` — Step 13: ``failure_pattern`` 統合
- :mod:`.summarize` — Step 8-9: 未要約セッションの LLM 要約生成

本パッケージは EvorefMem pillar 内部扱い
(``backend/free/memory/`` 配下) のため、EvorefMem が所有する全 FactType に
対する書込操作を :class:`~backend.free.memory.semantic.store.SemanticFactStore`
に直接行うことを許容する (MemFactView の overhead を避けるため)。
外部 pillar (EvorefLoop / EvorefLearn / Harness) は本パッケージを import しては
ならない — pillar boundary test で検出する

Step 11 (Critique-Synthesis) / Step 12 (Policy Evolver writeback) / Step 14
(Few-shot GC) は EvorefLearn pillar (:mod:`backend.free.learning.scheduler`)
に所属する Level 1 session-end 工程であり、本パッケージでは扱わない。
"""

from __future__ import annotations

from backend.free.memory.sleep.archive import archive_inactive_projects
from backend.free.memory.sleep.contextual import (
    generate_contextual_prefixes,
    generate_prefixes_for_store,
)
from backend.free.memory.sleep.extraction import (
    extract_semantic_facts,
    persist_facts,
)
from backend.free.memory.sleep.failure_consolidator import (
    consolidate_failure_patterns_for_project,
)
from backend.free.memory.sleep.gc import run_semmem_gc
from backend.free.memory.sleep.mdp_ingest import (
    ensure_mdp_ingester,
    ingest_mdp_traces,
)
from backend.free.memory.sleep.promotion import (
    classify_summary_type,
    promote_history_to_semmem,
)
from backend.free.memory.sleep.summarize import summarize_unsummarized_sessions

__all__ = [
    "archive_inactive_projects",
    "classify_summary_type",
    "consolidate_failure_patterns_for_project",
    "ensure_mdp_ingester",
    "extract_semantic_facts",
    "generate_contextual_prefixes",
    "generate_prefixes_for_store",
    "ingest_mdp_traces",
    "persist_facts",
    "promote_history_to_semmem",
    "run_semmem_gc",
    "summarize_unsummarized_sessions",
]
