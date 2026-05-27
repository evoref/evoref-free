"""

STM ノートおよび ``agent_trace*.jsonl`` を入力に SemanticFact を生成する
抽出器 3 種を提供する。Trigger B (アイドル ``full_idle_minutes`` 後) でのみ
``SleepTimeWorker._step8_extract_facts`` から呼び出される想定。

- :class:`ChatExtractor` — チャットモード由来の personal_fact / world_fact /
  preference / emotion / opinion を抽出
- :class:`CodingExtractor` — コーディングモード由来の project / decision /
  commitment / task / coding を抽出
- :class:`MDPTraceExtractor` — agent_trace*.jsonl (日付付きファイル) からエピソード単位で
  failure_pattern / decision を抽出

設計原則 (CLAUDE.md / .claude/rules/backend.md):
- LLM 呼び出しなし。ルールベースのみ (アシストモデル拡張は別 Phase)
- ``private`` ノートはスキップ
- セッションあたり抽出上限を尊重 (chat 10 / coding 5、pinned は無制限)
- ``MemoryNote.extracted_fact_ids`` に書き戻すことで二重抽出を防ぐ
"""

from backend.free.memory.extractors.base import (
    BaseExtractor,
    ExtractionContext,
    ExtractionResult,
)
from backend.free.memory.extractors.chat import ChatExtractor
from backend.free.memory.extractors.coding import CodingExtractor
from backend.free.memory.extractors.mdp_trace import MDPTraceExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionContext",
    "ExtractionResult",
    "ChatExtractor",
    "CodingExtractor",
    "MDPTraceExtractor",
]
