"""EvorefMem semantic ストア一元化マニフェスト

現状分散していた「ストア全体のメタ情報」(schema_version / embedding モデル /
index generation / created_at 等) を ``<memory_dir>/semantic/manifest.json``
に一元化する。これにより ``embeddings/<model_id>/vectors.npy`` の次元と
実モデルが divergent になる silent 事故を防ぎ、``_extra`` round-trip /
embedding model-agnostic 化 / CLI の共通土台となる

## 位置付け

EvorefMem の版番号 source of truth は引き続き
``local/memory/semantic/SCHEMA_VERSION`` マーカー
``manifest.json`` は **補助情報** として以下を保持する:

- ``schema_version``: マーカーと同値 (重複コピー)
- ``component_versions``: fact / subject_ns / embedding / index 各領域の
  独立バージョン
- ``embedding``: アクティブな埋め込みモデルの ``model_id`` / ``dim`` /
  ``normalized``。embedding model-agnostic 化に伴う atomic swap の伏線
- ``created_at`` / ``last_migrated_at`` / ``last_compacted_at``: 監査タイムスタンプ
- ``index_generations``: 索引再構築のための世代番号

## 設計原則

- manifest は **frozen dataclass** として immutable に扱い、更新時は
  :func:`update_manifest` が新インスタンスを生成して atomic 書き込みする
- 読み込み失敗 (破損 JSON / IO エラー) は :func:`load_manifest` が
  ``None`` を返し WARN ログを出す。呼び出し側は既存 manifest が読めない
  場合でも ``ensure_manifest()`` で復旧できる
- :func:`ensure_manifest` は **idempotent**。既存 manifest があれば
  そのまま返し、無い場合は引数の embedding 情報から初期 manifest を生成
- atomic 書き込みは一時ファイル + :func:`os.replace` で実現
  (Windows / POSIX ともに同一 FS 上では atomic)
- pillar 境界 上は EvorefMem 内部モジュール。他 pillar から
  本モジュールを直接 import してはならない
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from backend.io import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("memory.semantic.manifest")

# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────

MANIFEST_FILENAME = "manifest.json"
"""``<memory_dir>/semantic/`` 直下に置くマニフェストファイル名。"""

MANIFEST_SCHEMA_VERSION = 1
"""現行の manifest 自体のスキーマバージョン (SCHEMA_VERSION マーカーと同値)。"""

DEFAULT_COMPONENT_VERSIONS: dict[str, int] = {
    "fact": 1,
    "subject_ns": 1,
    "embedding": 1,
    "index": 2,
}
"""初期 component_versions

リリース前段階では 4 component とも `1` に揃えた状態を起点とする。
将来レイアウトに破壊的変更を入れる component のみを個別にバンプする。

- ``fact``: SemanticFact レコード形式 (``_version`` / ``_extra`` を含む現行形式)
- ``subject_ns``: subject namespace 規約 (``loop.*`` / ``learn.*`` / ``mem.*`` の 3 prefix)
- ``embedding``: 埋め込みストアのレイアウト
  (``embeddings/<model_id>/vectors.npy`` + ``row_to_id.json`` の model_id 別ディレクトリ構成)
- ``index``: 索引ファイルのフォーマット (v1=4 ファイル、v2=統合 ``index.jsonl``)

``SCHEMA_VERSION`` は現状 1 のまま (``init_evorefmem.SCHEMA_VERSION`` が SSOT)。
統合 ``index.jsonl`` は ``SemanticFactStore._reconcile_index`` が起動時に
facts.jsonl から自己修復するため、bump 無しで運用する。``IndexV1ToV2Migration``
(from=1/to=2) は将来の破壊的変更で bump する際の経路として休眠登録のまま。
"""

DEFAULT_INDEX_GENERATIONS: dict[str, int] = {
    "index_jsonl": 1,
}
"""索引再構築の世代番号。0 になることはない (初期値 1)。

M5-d で 4 索引 (facts_by_subject/type/pillar.idx + pinned.idx) を統合
``index.jsonl`` 1 本に簡素化した。
"""


# ──────────────────────────────────────────────────────────────────────────
# dataclass
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbeddingManifest:
    """``manifest.embedding`` 相当の埋め込みメタ情報。"""

    model_id: str
    dim: int
    normalized: bool = True


@dataclass(frozen=True)
class Manifest:
    """``semantic/manifest.json`` の in-memory 表現。

    frozen にして不変オブジェクトとして扱う。更新は :func:`update_manifest`
    経由で新インスタンスを生成する。
    """

    schema_version: int
    component_versions: dict[str, int]
    embedding: EmbeddingManifest
    created_at: str
    last_migrated_at: str | None = None
    last_compacted_at: str | None = None
    index_generations: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_INDEX_GENERATIONS),
    )

    # ── シリアライズ ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON 書込用の dict 表現を返す (キー順は挿入順保証)。"""
        return {
            "schema_version": self.schema_version,
            "component_versions": dict(self.component_versions),
            "embedding": {
                "model_id": self.embedding.model_id,
                "dim": self.embedding.dim,
                "normalized": self.embedding.normalized,
            },
            "created_at": self.created_at,
            "last_migrated_at": self.last_migrated_at,
            "last_compacted_at": self.last_compacted_at,
            "index_generations": dict(self.index_generations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """dict から Manifest を復元する (欠損キーはデフォルト値で補完)。"""
        emb_raw = data.get("embedding") or {}
        if not isinstance(emb_raw, Mapping):
            raise ValueError("manifest.embedding must be an object")
        model_id = emb_raw.get("model_id")
        dim = emb_raw.get("dim")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("manifest.embedding.model_id must be a non-empty string")
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("manifest.embedding.dim must be a positive int")
        normalized = bool(emb_raw.get("normalized", True))

        schema_version = int(data.get("schema_version", MANIFEST_SCHEMA_VERSION))
        comp_versions_raw = data.get("component_versions") or {}
        if not isinstance(comp_versions_raw, Mapping):
            raise ValueError("manifest.component_versions must be an object")
        component_versions = {**DEFAULT_COMPONENT_VERSIONS}
        for k, v in comp_versions_raw.items():
            component_versions[str(k)] = int(v)

        created_at = str(data.get("created_at") or "")
        if not created_at:
            raise ValueError("manifest.created_at must be a non-empty string")

        last_migrated_at = data.get("last_migrated_at")
        if last_migrated_at is not None and not isinstance(last_migrated_at, str):
            raise ValueError("manifest.last_migrated_at must be string or null")
        last_compacted_at = data.get("last_compacted_at")
        if last_compacted_at is not None and not isinstance(last_compacted_at, str):
            raise ValueError("manifest.last_compacted_at must be string or null")

        idx_gen_raw = data.get("index_generations") or {}
        if not isinstance(idx_gen_raw, Mapping):
            raise ValueError("manifest.index_generations must be an object")
        index_generations = {**DEFAULT_INDEX_GENERATIONS}
        for k, v in idx_gen_raw.items():
            index_generations[str(k)] = int(v)

        return cls(
            schema_version=schema_version,
            component_versions=component_versions,
            embedding=EmbeddingManifest(
                model_id=model_id, dim=dim, normalized=normalized,
            ),
            created_at=created_at,
            last_migrated_at=last_migrated_at,
            last_compacted_at=last_compacted_at,
            index_generations=index_generations,
        )


# ──────────────────────────────────────────────────────────────────────────
# パス
# ──────────────────────────────────────────────────────────────────────────


def manifest_path(memory_dir: Path) -> Path:
    """``<memory_dir>/semantic/manifest.json`` のフルパスを返す。"""
    return Path(memory_dir) / "semantic" / MANIFEST_FILENAME


# ──────────────────────────────────────────────────────────────────────────
# 時刻ユーティリティ
# ──────────────────────────────────────────────────────────────────────────


def _utc_iso8601(now: float | None = None) -> str:
    """UTC ISO 8601 文字列 (秒精度 + ``Z`` 末尾) を返す。

    サーバ実行 tz に依存しない監査タイムスタンプ用 (legacy backup
    などと同じ規約)
    """
    t = time.time() if now is None else now
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ──────────────────────────────────────────────────────────────────────────
# embedding model_id の正規化
# ──────────────────────────────────────────────────────────────────────────

_EMBEDDING_MODEL_SUFFIXES: tuple[str, ...] = (
    ".gguf", ".bin", ".safetensors", ".onnx", ".pt", ".pth",
)


def normalize_embedding_model_id(model_name: str) -> str:
    """``config.yaml::embedding.model_name`` を manifest 向け ``model_id`` に正規化する。

    例:
    - ``"bge-m3-q8_0.gguf"`` → ``"bge-m3-q8_0"``
    - ``"models/qwen3-embedding-4b.onnx"`` → ``"qwen3-embedding-4b"``
    - ``"BGE-M3"`` → ``"bge-m3"``

    - basename を取り出す (``models/`` 等のディレクトリ部を除去)
    - 既知のモデル拡張子を末尾から 1 回だけ剥がす
    - ASCII 小文字化 (比較しやすさ優先)

    なお本関数はあくまで初期値生成向けの best-effort。
    既存 manifest がある場合はそちらが優先され、ここの出力は上書きしない。
    """
    name = (model_name or "").strip()
    if not name:
        raise ValueError("embedding model_name must be non-empty")
    base = Path(name).name.lower()
    for suffix in _EMBEDDING_MODEL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or name.lower()


# ──────────────────────────────────────────────────────────────────────────
# 読み込み / 書き込み
# ──────────────────────────────────────────────────────────────────────────


def load_manifest(memory_dir: Path) -> Manifest | None:
    """manifest.json を読み込む。存在しない / 破損時は ``None`` を返し WARN ログ。"""
    path = manifest_path(memory_dir)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("semantic manifest unreadable at %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning(
            "semantic manifest must be a JSON object, got %s at %s",
            type(data).__name__, path,
        )
        return None
    try:
        return Manifest.from_dict(data)
    except (TypeError, ValueError) as exc:
        logger.warning("semantic manifest malformed at %s: %s", path, exc)
        return None


def write_manifest(memory_dir: Path, manifest: Manifest) -> Path:
    """manifest.json を atomic 書き込みする (:class:`AtomicWriter` 委譲)。

    途中中断で壊れた manifest が残ることは無い (同一 FS 上で atomic)。
    Windows ``PermissionError`` は :mod:`backend.io._retry` の retry
    ポリシーで吸収される。
    """
    path = manifest_path(memory_dir)
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
    with AtomicWriter(path) as f:
        f.write(payload)
    logger.debug("wrote semantic manifest: %s", path)
    return path


def ensure_manifest(
    memory_dir: Path,
    *,
    embedding_model_id: str,
    embedding_dim: int,
    normalized: bool = True,
    now: float | None = None,
) -> Manifest:
    """manifest.json を冪等に用意する。

    - 既存の manifest.json が **読み込み可能** ならそのまま返す (no-op)
    - 存在しない / 読み込み失敗時は現在時刻と引数値で新規生成して書き込む

    既存 manifest と引数値の不一致検出 (embedding model_id /
    dim 等) は **WARN ログのみ** 出して既存 manifest を優先する。
    自動 swap は別層の責務
    """
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    model_id = (embedding_model_id or "").strip()
    if not model_id:
        raise ValueError("embedding_model_id must be non-empty")

    existing = load_manifest(memory_dir)
    if existing is not None:
        if (
            existing.embedding.model_id != model_id
            or existing.embedding.dim != embedding_dim
            or existing.embedding.normalized != normalized
        ):
            logger.warning(
                "semantic manifest embedding drift: "
                "manifest=(model_id=%s, dim=%d, normalized=%s) vs "
                "config=(model_id=%s, dim=%d, normalized=%s). "
                "atomic swap will reconcile this mismatch.",
                existing.embedding.model_id, existing.embedding.dim,
                existing.embedding.normalized,
                model_id, embedding_dim, normalized,
            )
        return existing

    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        component_versions=dict(DEFAULT_COMPONENT_VERSIONS),
        embedding=EmbeddingManifest(
            model_id=model_id, dim=embedding_dim, normalized=normalized,
        ),
        created_at=_utc_iso8601(now),
        last_migrated_at=None,
        last_compacted_at=None,
        index_generations=dict(DEFAULT_INDEX_GENERATIONS),
    )
    write_manifest(memory_dir, manifest)
    logger.info(
        "semantic manifest created: %s (model_id=%s, dim=%d)",
        manifest_path(memory_dir), model_id, embedding_dim,
    )
    return manifest


def update_manifest(memory_dir: Path, /, **patch: Any) -> Manifest:
    """既存 manifest.json を部分更新する (atomic rewrite)。

    指定可能なキー:

    - ``schema_version`` (int)
    - ``component_versions`` (dict[str, int]) — 既存値に merge
    - ``embedding`` (EmbeddingManifest または対応 dict) — 丸ごと差替
    - ``created_at`` / ``last_migrated_at`` / ``last_compacted_at`` (str | None)
    - ``index_generations`` (dict[str, int]) — 既存値に merge

    manifest が未作成の場合は :class:`FileNotFoundError`。
    """
    current = load_manifest(memory_dir)
    if current is None:
        raise FileNotFoundError(
            f"semantic manifest not found under {memory_dir}/semantic/. "
            "Call ensure_manifest() first.",
        )

    changes: dict[str, Any] = {}
    for key, value in patch.items():
        if key == "component_versions":
            if not isinstance(value, Mapping):
                raise TypeError("component_versions must be a mapping")
            merged = dict(current.component_versions)
            for k, v in value.items():
                merged[str(k)] = int(v)
            changes[key] = merged
        elif key == "index_generations":
            if not isinstance(value, Mapping):
                raise TypeError("index_generations must be a mapping")
            merged = dict(current.index_generations)
            for k, v in value.items():
                merged[str(k)] = int(v)
            changes[key] = merged
        elif key == "embedding":
            if isinstance(value, EmbeddingManifest):
                changes[key] = value
            elif isinstance(value, Mapping):
                changes[key] = EmbeddingManifest(
                    model_id=str(value["model_id"]),
                    dim=int(value["dim"]),
                    normalized=bool(value.get("normalized", True)),
                )
            else:
                raise TypeError(
                    "embedding must be EmbeddingManifest or mapping, "
                    f"got {type(value).__name__}",
                )
        elif key in {
            "schema_version",
            "created_at",
            "last_migrated_at",
            "last_compacted_at",
        }:
            changes[key] = value
        else:
            raise KeyError(f"unknown manifest field: {key!r}")

    updated = replace(current, **changes)
    write_manifest(memory_dir, updated)
    return updated


__all__ = [
    "DEFAULT_COMPONENT_VERSIONS",
    "DEFAULT_INDEX_GENERATIONS",
    "EmbeddingManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "ensure_manifest",
    "load_manifest",
    "manifest_path",
    "normalize_embedding_model_id",
    "update_manifest",
    "write_manifest",
]
