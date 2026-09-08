"""EvorefMem — 構造化事実 (SemMem) の統一ストア (c_16 §4.2)。

スコープ別ディレクトリ (``global/`` / ``projects/<id>/``) を廃し、``Evidence``
(``kind="fact"`` / ``"claim"``) 1 型 + ``scope`` フィールドへ畳んだ。実体は
``local/memory/semantic/`` の :class:`~backend.free.rag.evidence.EvidenceStore`
1 つで、シャードは namespace (``mem`` / ``know`` / ``idx`` / ``loop`` / ``learn``)。

- :mod:`~backend.free.memory.semantic.fact` — ``SemanticFact`` ↔ ``Evidence``
- :mod:`~backend.free.memory.semantic.namespaces` — namespace 規則 (競合 / 減衰 / 注入)
- :mod:`~backend.free.memory.semantic.store` — :class:`SemanticStore` と
  スコープ束縛ビュー :class:`ScopedSemanticStore`
- :mod:`~backend.free.memory.semantic.sources` — ``know.*`` の取得元 / 取得単位
- :mod:`~backend.free.memory.semantic.subject_key` — subject の分類
- :mod:`~backend.free.memory.semantic.pin_manager` — pin の保護期間
- :mod:`~backend.free.memory.semantic.gc` — 件数超過の削除候補選定

書き手は sleep-time (``SleepTimeWorker``) だけ。例外は ``artifact`` ファクトの
即時書込 (ラルフループ) の 1 つ。
"""

from backend.free.memory.semantic.fact import (
    FACT_ATTR_FIELDS,
    FactRecordError,
    evidence_to_fact,
    fact_to_evidence,
)
from backend.free.memory.semantic.model_id import normalize_embedding_model_id
from backend.free.memory.semantic.namespaces import (
    NAMESPACE_POLICIES,
    NamespacePolicy,
    competes,
    is_injectable,
    know_half_life_days,
    namespace_of,
    policy_for,
)
from backend.free.memory.semantic.sources import (
    ItemRegistry,
    KnowledgeIngest,
    KnowledgeItem,
    KnowledgeSource,
    SourceRegistry,
)
from backend.free.memory.semantic.store import (
    ScopedSemanticStore,
    SemanticHit,
    SemanticStore,
)
from backend.free.memory.semantic.subject_key import (
    ALL_PILLARS,
    SubjectKey,
    SubjectKeyError,
    SubjectPillar,
)

__all__ = [
    "ALL_PILLARS",
    "FACT_ATTR_FIELDS",
    "NAMESPACE_POLICIES",
    "FactRecordError",
    "ItemRegistry",
    "KnowledgeIngest",
    "KnowledgeItem",
    "KnowledgeSource",
    "NamespacePolicy",
    "ScopedSemanticStore",
    "SemanticHit",
    "SemanticStore",
    "SourceRegistry",
    "SubjectKey",
    "SubjectKeyError",
    "SubjectPillar",
    "competes",
    "evidence_to_fact",
    "fact_to_evidence",
    "is_injectable",
    "know_half_life_days",
    "namespace_of",
    "normalize_embedding_model_id",
    "policy_for",
]
