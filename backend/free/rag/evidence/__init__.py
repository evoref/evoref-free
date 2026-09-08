"""Evidence Store — 注入材料の統一永続層 (docs/c_16_evidence_store.md)

会話由来ノート / 構造化事実 / 世界知識 / 文書チャンクを 1 つのレコード型
:class:`Evidence` と 3 つのストア (episodic / semantic / corpus) に統一する
基盤。EvorefGen pillar に属し、**他 pillar (memory / loop / learning / agent)
を参照しない**。

phase 1a で入っているのは以下:

- :mod:`~backend.free.rag.evidence.types` — レコード型・claim_key・confidence
- :mod:`~backend.free.rag.evidence.events` — 追記のみの事象ログ (§5.2)
- :mod:`~backend.free.rag.evidence.columns` — ``columns.npz`` (§5.3)
- :mod:`~backend.free.rag.evidence.snapshot` — 畳み込み / 版の読み書き
- :mod:`~backend.free.rag.evidence.manifest` — ``manifest.json`` (§5.1)
- :mod:`~backend.free.rag.evidence.store` — :class:`EvidenceStore` 基盤

転置索引 (§6.2) と順位付け (§7) は後続の段階。
"""

from __future__ import annotations

from backend.free.rag.evidence.columns import (
    COLUMNS_FILE,
    FLAG_ASSISTANT_ORIGIN,
    FLAG_PINNED,
    FLAG_PRIVATE,
    FLAG_RETRACTED,
    FLAG_SECRET,
    FLAG_SUPERSEDED,
    EvidenceColumns,
    active_mask,
    build_columns,
    freshness,
    load_columns,
    save_columns,
)
from backend.free.rag.evidence.events import (
    EVENT_VERSION,
    EventOp,
    EventPosition,
    EvidenceEventLog,
)
from backend.free.rag.evidence.lexical_index import (
    LexicalIndex,
    LexicalIndexBuilder,
    LexicalParams,
    LexicalShardSet,
    max_postings_scanned,
)
from backend.free.rag.evidence.ranking import (
    DEFAULT_ORIGIN_PRIORITY,
    Origin,
    RankColumns,
    collapse,
    gate_by_cosine,
    merge_candidates,
    score_rows,
)
from backend.free.rag.evidence.manifest import (
    DEFAULT_RETENTION,
    MANIFEST_FILE,
    EvidenceManifest,
)
from backend.free.rag.evidence.snapshot import (
    OFFSETS_FILE,
    RECORDS_FILE,
    SnapshotReader,
    SnapshotWriter,
    apply_patch,
    list_versions,
    prune_snapshots,
    read_snapshot,
    snapshot_dir,
    version_name,
)
from backend.free.rag.evidence.store import (
    DEFAULT_EMBED_MODE,
    DEFAULT_SHARD,
    EMBED_AS_QUERY_ATTR,
    EMBED_MODE_ATTR,
    LEXICAL_DEFAULTS,
    LEXICAL_DIR,
    SHARD_ROWS_FILE,
    SHARDS_FILE,
    EvidenceStore,
    UsageBuffer,
    embed_side_of,
    is_active,
    shard_dirname,
)
from backend.free.rag.evidence.types import (
    RECORD_VERSION,
    REQUIRED_KEYS,
    Confidentiality,
    Evidence,
    EvidenceRecordError,
    Kind,
    Origin,
    StoreName,
    Tier,
    Veracity,
    claim_hash64,
    compute_claim_key,
    compute_claim_key_structured,
    corroboration_count,
    derive_confidence,
    from_record,
    new_evidence_id,
    normalize_claim_text,
    validate_attrs,
)

__all__ = [
    "COLUMNS_FILE",
    "DEFAULT_EMBED_MODE",
    "DEFAULT_RETENTION",
    "DEFAULT_SHARD",
    "EMBED_AS_QUERY_ATTR",
    "EMBED_MODE_ATTR",
    "EVENT_VERSION",
    "LEXICAL_DEFAULTS",
    "LEXICAL_DIR",
    "SHARDS_FILE",
    "SHARD_ROWS_FILE",
    "FLAG_ASSISTANT_ORIGIN",
    "FLAG_PINNED",
    "FLAG_PRIVATE",
    "FLAG_RETRACTED",
    "FLAG_SECRET",
    "FLAG_SUPERSEDED",
    "MANIFEST_FILE",
    "OFFSETS_FILE",
    "RECORDS_FILE",
    "RECORD_VERSION",
    "REQUIRED_KEYS",
    "Confidentiality",
    "EventOp",
    "EventPosition",
    "Evidence",
    "EvidenceColumns",
    "EvidenceEventLog",
    "EvidenceManifest",
    "EvidenceRecordError",
    "EvidenceStore",
    "Kind",
    "Origin",
    "SnapshotReader",
    "SnapshotWriter",
    "StoreName",
    "Tier",
    "UsageBuffer",
    "Veracity",
    "active_mask",
    "apply_patch",
    "build_columns",
    "claim_hash64",
    "compute_claim_key",
    "compute_claim_key_structured",
    "corroboration_count",
    "derive_confidence",
    "embed_side_of",
    "freshness",
    "from_record",
    "is_active",
    "list_versions",
    "load_columns",
    "new_evidence_id",
    "normalize_claim_text",
    "prune_snapshots",
    "read_snapshot",
    "save_columns",
    "shard_dirname",
    "snapshot_dir",
    "validate_attrs",
    "version_name",
    "LexicalIndex",
    "LexicalIndexBuilder",
    "LexicalParams",
    "LexicalShardSet",
    "max_postings_scanned",
    "DEFAULT_ORIGIN_PRIORITY",
    "Origin",
    "RankColumns",
    "collapse",
    "gate_by_cosine",
    "merge_candidates",
    "score_rows",
]
