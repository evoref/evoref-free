"""corpus — 文書由来チャンクのパッケージ形式と実行時ストア (c_16 §4.3)

EvorefGen pillar。``.evocart`` パッケージ (`package.py`) を決定論チャンク
(`chunking.py`) にして、パッケージ版ごとに 1 つの
:class:`~backend.free.rag.evidence.store.EvidenceStore` を持つ実行時ストア
(`store.py`) へ入れる。旧カートリッジ (``CartridgeManager`` + 個別
``VectorStore`` + ``registry.json``) の置き換え。
"""

from __future__ import annotations

from backend.free.rag.corpus.chunking import (
    CHUNKER_VERSION,
    chunk_documents,
    chunk_evidence_id,
    document_headings,
    list_documents,
)
from backend.free.rag.corpus.package import (
    DOCS_DIR,
    PACKAGE_FILE,
    PACKAGE_SCHEMA_VERSION,
    PACKAGE_SUFFIX,
    PREBUILT_DIR,
    PackageContents,
    PackageError,
    PackageMeta,
    PrebuiltInfo,
    compute_content_digest,
    iter_prebuilt_chunks,
    meta_from_record,
    package_filename,
    read_package,
    validate_package_id,
    validate_version,
    write_package,
    write_package_meta,
)
from backend.free.rag.corpus.store import (
    CENTROID_FILE,
    CORPUS_DIR_NAME,
    DEFAULT_CORPUS_STORE_PRIOR,
    PACKAGES_DIR,
    CorpusHit,
    CorpusInstallCancelled,
    CorpusManifest,
    CorpusPackage,
    CorpusStore,
    InstallResult,
    adapt_embedding_backend,
    merge_rag_evidence_config,
    resolve_cartridge_gate_threshold,
)

__all__ = [
    "CENTROID_FILE",
    "CHUNKER_VERSION",
    "CORPUS_DIR_NAME",
    "DEFAULT_CORPUS_STORE_PRIOR",
    "DOCS_DIR",
    "PACKAGES_DIR",
    "PACKAGE_FILE",
    "PACKAGE_SCHEMA_VERSION",
    "PACKAGE_SUFFIX",
    "PREBUILT_DIR",
    "CorpusHit",
    "CorpusInstallCancelled",
    "CorpusManifest",
    "CorpusPackage",
    "CorpusStore",
    "InstallResult",
    "PackageContents",
    "PackageError",
    "PackageMeta",
    "PrebuiltInfo",
    "adapt_embedding_backend",
    "chunk_documents",
    "chunk_evidence_id",
    "compute_content_digest",
    "document_headings",
    "iter_prebuilt_chunks",
    "list_documents",
    "merge_rag_evidence_config",
    "meta_from_record",
    "package_filename",
    "read_package",
    "resolve_cartridge_gate_threshold",
    "validate_package_id",
    "validate_version",
    "write_package",
    "write_package_meta",
]
