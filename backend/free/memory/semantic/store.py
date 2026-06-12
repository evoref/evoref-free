"""

EvorefMem 統合仕様 における意味記憶 (SemMem) の永続化層を提供する
1 ストアインスタンス = 1 スコープ (`global` または `project:<id>`) で物理分離する。

ファイルレイアウト::

    <root>/
    ├── facts.jsonl                         # 追記式の生ログ (last-write-wins on id)
    ├── index.jsonl                         # fact_id 正規化形の統合索引 (subject/type/pillar/pinned)
    └── embeddings/
        └── <model_id>/
            ├── vectors.npy                 # numpy 2D float32 (N, dim)
            └── row_to_id.json              # JSON: {"row_to_id": [fact_id, ...]}

設計方針:
- `facts.jsonl` は追記式とし、同一 ID の複数行が存在し得る。読み込み時は
  最後に書かれた行で上書きされる (last-write-wins)。
- `index.jsonl` は :class:`backend.io.IncrementalIndexUpdater` 上に
  ``{fact_id: {subject, type, pillar, pinned}}`` を持つ正規化形の統合索引。
  add / update / delete のたびに差分 1 行を append し、閾値超過で tombstone を
  畳んで compact する。旧 4 索引 (``facts_by_subject.idx`` /
  ``facts_by_type.idx`` / ``facts_by_pillar.idx`` / ``pinned.idx``) を統合した形。
  ``index.jsonl`` は起動時 :meth:`_reconcile_index` が facts.jsonl
  (source of truth) と突合して自己修復する (未生成なら生成、欠落分は backfill)
  ため、SCHEMA_VERSION の bump 無しに常に facts.jsonl と整合する。
  ``IndexV1ToV2Migration`` (from=1/to=2) は将来の bump 用に休眠登録のまま。
- supersession は `superseded_by` / `supersedes` フィールドで表現する。
  検索系 API はデフォルトで `superseded_by` が立っているファクトを除外する。
- 後方互換は提供しない

## pillar 索引

統合 ``index.jsonl`` の ``pillar`` 属性は 3 pillar namespace
(`loop.*` / `learn.*` / `mem.*`) の subject 前方一致検索を高速化する。

索引キーは **pillar (3 値まで)** とし、必ず 3 バケット以内に収まる。
prefix 検索はヒットした pillar バケットを走査し、``fact.subject.startswith``
で最終フィルタする (pillar 内のファクト数は高々数千〜数万オーダーなので
線形フィルタで十分)。索引キーの分類は :class:`SubjectKey` に一本化。

API:
- `add_fact` / `get_fact` / `update_fact`
- `search_by_subject` / `search_by_type` / `search_by_pillar_prefix`
- `search_by_embedding` (cosine similarity, top_k)
- `supersede(old_id, new_id)`
- `for_global(root)` / `for_project(root, project_id)` 補助コンストラクタ
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.io import AtomicWriter, IncrementalIndexUpdater
from backend.free.memory.semantic.embedding_store import (
    EmbeddingStore,
)
from backend.free.memory.semantic.manifest import Manifest
from backend.free.memory.semantic.subject_key import SubjectKey
from backend.free.memory.types import (
    FactType,
    SemanticFact,
    deserialize_fact_jsonl,
    serialize_fact_jsonl,
)
from backend.log_config import get_logger

logger = get_logger("memory.semantic.store")


# ──────────────────────────────────────────────────────────────────────────
# ファイル名定数
# ──────────────────────────────────────────────────────────────────────────

FACTS_FILENAME = "facts.jsonl"
SUBJECT_IDX_FILENAME = "facts_by_subject.idx"
TYPE_IDX_FILENAME = "facts_by_type.idx"
PILLAR_IDX_FILENAME = "facts_by_pillar.idx"
PINNED_IDX_FILENAME = "pinned.idx"

#: M5-c1: fact_id 正規化形の統合索引ファイル名。
#: 1 行 = 1 fact 属性レコード ``{"fact_id":..., "subject":..., "type":...,
#: "pillar":..., "pinned":...}``。dual write 期間中は旧 4 索引と並存し、
#: M5-d の lazy migration 完了後は **唯一の永続索引** となる。
INDEX_JSONL_FILENAME = "index.jsonl"

PILLAR_SUBJECT_PREFIXES: tuple[str, ...] = ("loop.", "learn.", "mem.")
"""3 pillar namespace の subject 前方一致索引対象 prefix

`facts_by_pillar.idx` はこの 3 prefix を共通索引する。旧
`HARNESS_SUBJECT_PREFIX = "harness."` は 全廃済み
"""


def _matches_pillar_prefix(subject: str) -> bool:
    """``subject`` が 3 pillar namespace のいずれかに前方一致するか。"""
    return any(subject.startswith(p) for p in PILLAR_SUBJECT_PREFIXES)


# ──────────────────────────────────────────────────────────────────────────
# 統合索引 (index.jsonl) のシリアライザ / 属性抽出
# ──────────────────────────────────────────────────────────────────────────


def _fact_to_index_attrs(fact: SemanticFact) -> dict[str, Any]:
    """SemanticFact から index.jsonl 用の属性 dict を抽出する。

    pillar は ``SubjectKey.try_parse(subject).pillar`` で導出する (subject から
    決定論的に決まるため厳密には保存不要だが、外部 inspect ツールでの再計算
    コストを避けるため明示的に保存する)。
    """
    key = SubjectKey.try_parse(fact.subject)
    return {
        "subject": fact.subject,
        "type": str(fact.type),
        "pillar": key.pillar if key is not None else None,
        "pinned": bool(fact.pinned),
    }


def _serialize_index_entry(fact_id: str, attrs: dict[str, Any]) -> str:
    """index.jsonl 用の 1 行 JSON 文字列を作る (改行を含めない)。"""
    payload = {"fact_id": fact_id, **attrs}
    return json.dumps(payload, ensure_ascii=False)


def _deserialize_index_entry(line: str) -> tuple[str, dict[str, Any]]:
    """index.jsonl の 1 行から ``(fact_id, attrs)`` を復元する。"""
    obj = json.loads(line)
    fact_id = obj["fact_id"]
    attrs = {k: v for k, v in obj.items() if k != "fact_id"}
    return fact_id, attrs


class SemanticFactStore:
    """1 スコープ (global または project:<id>) の SemanticFact を管理する永続ストア。

    ルートディレクトリは呼び出し側で指定する。`for_global` / `for_project`
    で標準レイアウト (`local/memory/semantic/{global,projects/<id>}/`) に
    沿ったコンストラクタが利用できる。
    """

    def __init__(
        self, root_dir: Path, *, manifest: Manifest | None = None,
    ) -> None:
        """Store を初期化する.

        Args:
            root_dir: スコープ別ルートディレクトリ
                (``<semantic_root>/global`` or ``<semantic_root>/projects/<id>``)
                場合は ``embeddings/<model_id>/vectors.npy`` の次元と
                ``manifest.embedding.dim`` を照合し、不一致なら WARN ログを
                出す
                None の場合は照合をスキップ (テスト fixture 等での直接
                instantiation 互換)。
        """
        self.root_dir = Path(root_dir)
        self._manifest = manifest
        self._facts: dict[str, SemanticFact] = {}
        self._by_subject: dict[str, set[str]] = {}
        self._by_type: dict[str, set[str]] = {}
        self._by_pillar: dict[str, set[str]] = {}
        self._pinned: set[str] = set()
        # model_id 別ディレクトリで管理する。
        self._embedding_store: EmbeddingStore = self._init_embedding_store()
        # M5-c1: 統合索引 index.jsonl の dual write 用 updater。
        # `_load()` 中の rebuild 時には dual write をスキップするため、
        # 初期化フラグを False にしておく。`_load()` 完了後に True に切り替え、
        # 以降の add/update/delete で旧 4 索引と並行して書き込まれる。
        self._index_updater: IncrementalIndexUpdater[dict[str, Any]] = (
            IncrementalIndexUpdater(
                self.root_dir / INDEX_JSONL_FILENAME,
                serialize_entry=_serialize_index_entry,
                deserialize_entry=_deserialize_index_entry,
                flush_interval_writes=50,
            )
        )
        self._index_dual_write_enabled = False
        self._load()
        self._index_dual_write_enabled = True
        # index.jsonl を facts.jsonl (source of truth) と突合して自己修復する。
        # 新規ストア (index.jsonl 未生成) や前回 shutdown での pending 喪失を、
        # SCHEMA_VERSION の bump 無しに毎起動で埋める。
        self._reconcile_index()
        self._verify_embedding_dim_against_manifest()

    def _init_embedding_store(self) -> EmbeddingStore:
        """アクティブ model_id を manifest から解決して EmbeddingStore を構築する.

        優先順:
        1. manifest.embedding.model_id が与えられていればそれを採用
        2. 与えられていない場合は ``embeddings/`` 配下を走査して自動判定
        3. どちらも解決できなければ ``DEFAULT_MODEL_ID`` (test fixture 用)
        """
        manifest_model_id: str | None = None
        if self._manifest is not None:
            manifest_model_id = self._manifest.embedding.model_id
        return EmbeddingStore.active(
            self.root_dir, manifest_model_id=manifest_model_id,
        )

    # ── 補助コンストラクタ ────────────────────────────────────────────

    @classmethod
    def for_global(
        cls, semantic_root: Path, *, manifest: Manifest | None = None,
    ) -> SemanticFactStore:
        """`<semantic_root>/global/` を root とするストアを作る"""
        return cls(Path(semantic_root) / "global", manifest=manifest)

    @classmethod
    def for_project(
        cls,
        semantic_root: Path,
        project_id: str,
        *,
        manifest: Manifest | None = None,
    ) -> SemanticFactStore:
        """`<semantic_root>/projects/<project_id>/` を root とするストアを作る"""
        if not project_id:
            raise ValueError("project_id must be non-empty")
        return cls(
            Path(semantic_root) / "projects" / project_id, manifest=manifest,
        )

    # ── manifest 照合 ────────────────────────────────────────────────

    def _verify_embedding_dim_against_manifest(self) -> None:
        """``manifest.embedding.dim`` と実 EmbeddingStore の次元を照合する.

        不一致時は WARN ログのみ。model_id
        別ディレクトリに格納されているため、manifest で指定された
        model_id の vectors.npy との次元整合を見る形に変わっている。
        """
        if self._manifest is None:
            return
        actual_dim = self._embedding_store.dim
        if actual_dim is None:
            return
        expected_dim = self._manifest.embedding.dim
        if expected_dim != actual_dim:
            logger.warning(
                "semantic embeddings dim mismatch at %s: manifest=%d (%s) "
                "vs vectors.npy=%d. atomic swap will reconcile this.",
                self.root_dir, expected_dim,
                self._manifest.embedding.model_id, actual_dim,
            )

    # ── パス ──────────────────────────────────────────────────────────

    def _facts_path(self) -> Path:
        return self.root_dir / FACTS_FILENAME

    # ── ロード ────────────────────────────────────────────────────────

    def _load(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._facts_path()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        fact = deserialize_fact_jsonl(line)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "Skipping malformed fact line in %s: %s", path, exc,
                        )
                        continue
                    # last-write-wins on id
                    self._facts[fact.id] = fact
        self._rebuild_indexes()
        # 埋め込みは EmbeddingStore 側で既にロード済 (__init__ 内)。

    def _rebuild_indexes(self) -> None:
        self._by_subject.clear()
        self._by_type.clear()
        self._by_pillar.clear()
        self._pinned.clear()
        for fact in self._facts.values():
            self._add_to_indexes(fact)

    def _add_to_indexes(self, fact: SemanticFact) -> None:
        self._by_subject.setdefault(fact.subject, set()).add(fact.id)
        self._by_type.setdefault(fact.type, set()).add(fact.id)
        # 索引キーは SubjectKey.pillar に一本化
        # (subject 全文ではなく "loop" / "learn" / "mem" の 3 値まで)
        key = SubjectKey.try_parse(fact.subject)
        if key is not None:
            self._by_pillar.setdefault(key.pillar, set()).add(fact.id)
        if fact.pinned:
            self._pinned.add(fact.id)
        # M5-c1: 統合索引 index.jsonl への dual write
        # (起動時の rebuild 中は False で skip、起動完了後の add/update で True)
        if self._index_dual_write_enabled:
            self._index_updater.upsert(fact.id, _fact_to_index_attrs(fact))

    def _remove_from_indexes(self, fact: SemanticFact) -> None:
        self._discard(self._by_subject, fact.subject, fact.id)
        self._discard(self._by_type, fact.type, fact.id)
        key = SubjectKey.try_parse(fact.subject)
        if key is not None:
            self._discard(self._by_pillar, key.pillar, fact.id)
        self._pinned.discard(fact.id)
        # M5-c1: 統合索引 index.jsonl への dual write
        if self._index_dual_write_enabled:
            self._index_updater.remove(fact.id)

    @staticmethod
    def _discard(idx: dict[str, set[str]], key: str, fact_id: str) -> None:
        bucket = idx.get(key)
        if not bucket:
            return
        bucket.discard(fact_id)
        if not bucket:
            del idx[key]

    # ── 永続化 ────────────────────────────────────────────────────────

    def _append_fact_line(self, fact: SemanticFact) -> None:
        line = serialize_fact_jsonl(fact)
        with self._facts_path().open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def flush_index(self) -> None:
        """``index.jsonl`` の pending dual-write を永続化する (lifespan shutdown 用)。

        :class:`IncrementalIndexUpdater` は ``flush_interval_writes`` 未満の
        pending op を in-memory buffer に保持するため、明示 flush しないと
        プロセス終了で最大 ``flush_interval-1`` 件が失われる。次回起動の
        :meth:`_reconcile_index` が facts.jsonl から復元するが、shutdown で
        flush しておけば余分な再構築 I/O を避けられる。
        """
        self._index_updater.flush()

    def _reconcile_index(self) -> None:
        """``index.jsonl`` を facts.jsonl (source of truth) と突合して自己修復する。

        起動時、index.jsonl の pending 喪失 / 未生成 (新規ストア) で facts.jsonl と
        乖離している分を upsert / remove で埋め、差分があるときだけ flush する
        (clean 起動では I/O ゼロ)。``_index_dual_write_enabled=True`` の状態で
        呼ぶこと (upsert/remove が index.jsonl に反映される)。
        """
        indexed = self._index_updater.load()
        changed = False
        for fact_id, fact in self._facts.items():
            attrs = _fact_to_index_attrs(fact)
            if indexed.get(fact_id) != attrs:
                self._index_updater.upsert(fact_id, attrs)
                changed = True
        for stale_id in indexed.keys() - self._facts.keys():
            self._index_updater.remove(stale_id)
            changed = True
        if changed:
            self._index_updater.flush()
            logger.debug(
                "index.jsonl reconciled against facts.jsonl at %s "
                "(facts=%d)", self.root_dir, len(self._facts),
            )

    def _upsert_embedding(self, fact_id: str, vec: np.ndarray) -> None:
        """EmbeddingStore に委譲する薄いラッパ (内部 API の見掛け互換用)."""
        self._embedding_store.upsert(fact_id, vec)

    # ── CRUD ──────────────────────────────────────────────────────────

    def add_fact(self, fact: SemanticFact) -> SemanticFact:
        """新規ファクトを追加する。

        - `id` が未設定なら新規発番する
        - 既存 ID と衝突したら `ValueError`
        - `created_at` / `accessed_at` が 0 なら現在時刻で埋める
        - `embedding` が設定されていれば EmbeddingStore にも追記する
        """
        if not fact.id:
            fact.id = SemanticFact.new_id()
        if fact.id in self._facts:
            raise ValueError(f"fact id already exists: {fact.id}")
        now = time.time()
        if fact.created_at == 0.0:
            fact.created_at = now
        if fact.accessed_at == 0.0:
            fact.accessed_at = fact.created_at
        self._facts[fact.id] = fact
        self._add_to_indexes(fact)
        self._append_fact_line(fact)
        if fact.embedding is not None:
            self._upsert_embedding(fact.id, fact.embedding)
        logger.debug(
            "add_fact: id=%s subject=%s type=%s scope=%s",
            fact.id, fact.subject, fact.type, fact.scope,
        )
        return fact

    def get_fact(self, fact_id: str) -> SemanticFact | None:
        """ID でファクトを取得する。存在しなければ None"""
        return self._facts.get(fact_id)

    def update_fact(self, fact_id: str, **changes: Any) -> SemanticFact:
        """既存ファクトのフィールドを差分更新する。

        指定可能なフィールドは `SemanticFact` の dataclass フィールドのみ。
        `id` の変更は許可しない。インデックスに影響するフィールド
        (`subject` / `type` / `pinned`) を変更した場合は索引を再構築する。
        `embedding` を含めると EmbeddingStore にも反映する。
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(fact_id)
        if "id" in changes:
            raise ValueError("cannot change fact id")
        for key in changes:
            if not hasattr(fact, key):
                raise AttributeError(f"unknown SemanticFact field: {key}")

        self._remove_from_indexes(fact)
        for key, val in changes.items():
            setattr(fact, key, val)
        fact.accessed_at = time.time()
        self._add_to_indexes(fact)
        self._append_fact_line(fact)
        if "embedding" in changes:
            # embedding を明示的に None へクリアした場合は EmbeddingStore からも
            # 除去する。さもないと vectors.npy に stale ベクトルが残り、
            # search_by_embedding が更新後も古いベクトルでヒットし続ける。
            if fact.embedding is not None:
                self._upsert_embedding(fact.id, fact.embedding)
            else:
                self._remove_embedding(fact.id)
        logger.debug("update_fact: id=%s changes=%s", fact_id, sorted(changes.keys()))
        return fact

    def delete_fact(self, fact_id: str) -> bool:
        """ファクトを物理削除する

        - in-memory 状態 / 索引 / pinned 集合 / 埋め込み行列から取り除く
        - ``facts.jsonl`` を残存ファクトのみで書き換える (rewrite)
        - 索引 / 埋め込みファイルも持続化を更新する

        Returns:
            実際に削除された場合 ``True``、未存在の場合 ``False``。
        """
        fact = self._facts.pop(fact_id, None)
        if fact is None:
            return False
        self._remove_from_indexes(fact)
        self._remove_embedding(fact_id)
        self._rewrite_facts_log()
        logger.debug("delete_fact: id=%s subject=%s", fact_id, fact.subject)
        return True

    def _remove_embedding(self, fact_id: str) -> None:
        """EmbeddingStore からの削除に委譲する."""
        self._embedding_store.delete(fact_id)

    def _rewrite_facts_log(self) -> None:
        """``facts.jsonl`` を現在の in-memory ファクトのみで書き直す。

        delete_fact 後に追記式 jsonl から削除済みファクトの行が消えないと、
        次回ロード時に last-write-wins で復活してしまうため、削除時には
        全行を書き直す必要がある。書き込みは :class:`AtomicWriter` 経由
        (Windows ``PermissionError`` retry 込み)。
        """
        path = self._facts_path()
        with AtomicWriter(path) as f:
            for fact in self._facts.values():
                f.write(serialize_fact_jsonl(fact) + "\n")

    def count_by_type(self, fact_type: FactType, *, include_superseded: bool = False) -> int:
        """``type`` に該当するファクト数を返す"""
        ids = self._by_type.get(fact_type, set())
        if include_superseded:
            return len(ids)
        return sum(1 for fid in ids if not self._facts[fid].superseded_by)

    def supersede(self, old_id: str, new_id: str) -> None:
        """`old_id` を `new_id` で置き換える supersession チェーンを構築する。

        `old.superseded_by = new_id` と `new.supersedes += [old_id]` を更新。
        既に supersede 済の old を再 supersede しようとすると `ValueError`。
        """
        old = self._facts.get(old_id)
        new = self._facts.get(new_id)
        if old is None:
            raise KeyError(f"old fact not found: {old_id}")
        if new is None:
            raise KeyError(f"new fact not found: {new_id}")
        if old_id == new_id:
            raise ValueError("cannot supersede a fact by itself")
        if old.superseded_by:
            raise ValueError(
                f"fact {old_id} already superseded by {old.superseded_by}",
            )
        self.update_fact(old_id, superseded_by=new_id)
        merged: list[str] = list(new.supersedes)
        if old_id not in merged:
            merged.append(old_id)
        self.update_fact(new_id, supersedes=merged)
        logger.info("supersede: %s -> %s", old_id, new_id)

    # ── 検索 ──────────────────────────────────────────────────────────

    def search_by_subject(
        self,
        subject: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """`subject` に完全一致するファクトを返す"""
        return self._collect(self._by_subject.get(subject, set()), include_superseded)

    def search_by_type(
        self,
        fact_type: FactType,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """`type` に完全一致するファクトを返す"""
        return self._collect(self._by_type.get(fact_type, set()), include_superseded)

    def search_by_pillar_prefix(
        self,
        prefix: str,
        *,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        """3 pillar namespace (``loop.`` / ``learn.`` / ``mem.``) の subject
        前方一致でファクトを返す

        ``prefix`` は ``loop.`` / ``learn.`` / ``mem.`` のいずれかで始まる
        完全な前方一致パターンを期待する (例: ``learn.policy.coding.``)。
        索引 ``facts_by_pillar.idx`` を参照して高速化する。
        """
        if not _matches_pillar_prefix(prefix):
            raise ValueError(
                "pillar prefix must start with one of "
                f"{PILLAR_SUBJECT_PREFIXES!r}, got {prefix!r}",
            )
        # _by_pillar は pillar 名 ("loop"/"learn"/"mem")
        # をキーとし、値はそのバケット内の全 fact_id。prefix から pillar を
        # 取り出して該当バケットを走査し、subject.startswith で最終フィルタ。
        pillar = prefix.split(".", 1)[0]
        bucket = self._by_pillar.get(pillar)
        if not bucket:
            return []
        ids: set[str] = set()
        for fact_id in bucket:
            fact = self._facts.get(fact_id)
            if fact is not None and fact.subject.startswith(prefix):
                ids.add(fact_id)
        return self._collect(ids, include_superseded)

    def search_by_embedding(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        include_superseded: bool = False,
    ) -> list[tuple[SemanticFact, float]]:
        """埋め込みベクトルで cosine similarity 検索する。

        Returns:
            (fact, score) のリスト。score は cosine similarity (-1.0〜1.0)。
        """
        # EmbeddingStore は fact filter を知らないため、superseded 除外 /
        # 存在チェックの分だけ多めに引いて top_k を満たす。
        store_count = self._embedding_store.count
        if store_count == 0:
            return []
        raw = self._embedding_store.search(query, top_k=store_count)
        results: list[tuple[SemanticFact, float]] = []
        for fid, score in raw:
            fact = self._facts.get(fid)
            if fact is None:
                continue
            if not include_superseded and fact.superseded_by:
                continue
            results.append((fact, score))
            if len(results) >= top_k:
                break
        return results

    # ── 内部ヘルパ ────────────────────────────────────────────────────

    def _collect(
        self,
        ids: Iterable[str],
        include_superseded: bool,
    ) -> list[SemanticFact]:
        out: list[SemanticFact] = []
        for fid in ids:
            fact = self._facts.get(fid)
            if fact is None:
                continue
            if not include_superseded and fact.superseded_by:
                continue
            out.append(fact)
        return out

    # ── 観測ヘルパ (テスト・統計用) ───────────────────────────────────

    def __len__(self) -> int:
        return len(self._facts)

    def all_facts(self, *, include_superseded: bool = True) -> list[SemanticFact]:
        if include_superseded:
            return list(self._facts.values())
        return [f for f in self._facts.values() if not f.superseded_by]

    def pinned_facts(self) -> list[SemanticFact]:
        return [self._facts[fid] for fid in self._pinned if fid in self._facts]
