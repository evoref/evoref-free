"""

`SemanticFactStore` を提供する。`global` / `project:<id>` の物理分離は
ストアインスタンス単位 (ルートディレクトリ単位) で表現する。

"""

from backend.free.memory.semantic.embedding_store import (
    DEFAULT_MODEL_ID,
    EMBEDDINGS_DIRNAME,
    ROW_TO_ID_FILENAME,
    VECTORS_FILENAME,
    EmbeddingStore,
    embeddings_root,
    list_stored_models,
    register_new_model,
    swap_active_model_id,
)
from backend.free.memory.semantic.manifest import (
    DEFAULT_COMPONENT_VERSIONS,
    DEFAULT_INDEX_GENERATIONS,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    EmbeddingManifest,
    Manifest,
    ensure_manifest,
    load_manifest,
    manifest_path,
    normalize_embedding_model_id,
    update_manifest,
    write_manifest,
)
from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.semantic.subject_key import (
    ALL_PILLARS,
    SubjectKey,
    SubjectKeyError,
    SubjectPillar,
)
from backend.free.memory.semantic.subject_migration import (
    FactPredicate,
    SubjectCategoryRenameMigration,
)

__all__ = [
    "ALL_PILLARS",
    "DEFAULT_COMPONENT_VERSIONS",
    "DEFAULT_INDEX_GENERATIONS",
    "DEFAULT_MODEL_ID",
    "EMBEDDINGS_DIRNAME",
    "EmbeddingManifest",
    "EmbeddingStore",
    "FactPredicate",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "ROW_TO_ID_FILENAME",
    "SemanticFactStore",
    "SubjectCategoryRenameMigration",
    "SubjectKey",
    "SubjectKeyError",
    "SubjectPillar",
    "VECTORS_FILENAME",
    "embeddings_root",
    "ensure_manifest",
    "list_stored_models",
    "load_manifest",
    "manifest_path",
    "normalize_embedding_model_id",
    "register_new_model",
    "swap_active_model_id",
    "update_manifest",
    "write_manifest",
]
