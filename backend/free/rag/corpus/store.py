"""`CorpusStore` — 文書由来チャンクの実行時ストア (c_16 §4.3 / §7)

**パッケージは配布単位 (`package.py`)、corpus は実行時ストア** の二層のうち
実行時側。ディスク形は:

```
local/memory/corpus/
├── manifest.json                     # {active: {<id>: <version>},
│                                     #  store_prior_overrides: {<id>: float},
│                                     #  loaded: [<id>]}
└── packages/<id>/<version>/          # ← ここが 1 つの EvidenceStore
    ├── package.json
    ├── docs/
    ├── manifest.json                 # EvidenceStore の manifest (c_16 §5.1)
    ├── snapshot/v0001/               # records.jsonl / offsets / columns / lexical
    ├── embeddings/<model_id>/
    └── centroid.npy
```

パッケージ版のディレクトリが **そのまま EvidenceStore のディレクトリ** で、
snapshot は 1 版だけ持つ。corpus は事象ログを持たない — 版が履歴そのもので、
内容は不変 (c_16 §2.1)。install / rebuild だけが書き手で、チャット応答パスは
読むだけ。

## 順位付け

ゲートは 2 段。まずパッケージ centroid との素の cosine (旧カートリッジゲートの
規則をそのまま引き継ぐ)、通ったパッケージの中で ``EvidenceStore.search`` が
``score = cos × freshness × confidence × store_prior`` (c_16 §7.2) を計算する。
``store_prior`` は既定 1.0 (2026-09-08 A/B) で、``manifest.store_prior_overrides`` が PC 固有に
上書きできる。旧 ``priority`` は廃止 — 掛けた値を閾値に流すと閾値を偽装する
(2026-09-02 監査 S-A4) ので、返り値では cosine と store_prior を分けて渡す。
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.rag.corpus.chunking import CHUNKER_VERSION, chunk_documents
from backend.free.rag.corpus.package import (
    DOCS_DIR,
    PACKAGE_FILE,
    PREBUILT_DIR,
    PREBUILT_EMBEDDINGS_DIR,
    PackageError,
    PackageMeta,
    PrebuiltInfo,
    compute_content_digest,
    iter_prebuilt_chunks,
    meta_from_record,
    read_package,
    write_package_meta,
    write_prebuilt_build,
    write_prebuilt_chunks,
)
from backend.free.rag.corpus.calibration import (
    CANARY_UTTERANCES,
    calibration_signature,
    compute_corpus_calibration,
    load_corpus_calibration,
    save_corpus_calibration,
)
from backend.free.rag.corpus.pseudo_queries import PSEUDO_QUERIES_DIR, PseudoQueryIndex
from backend.free.rag.evidence._json_state import JsonStateFile
from backend.free.rag.evidence.config import merge_rag_evidence_config
from backend.free.rag.evidence.ranking import RankColumns, score_rows
from backend.free.rag.evidence.store import EvidenceStore, UsageBuffer
from backend.free.rag.vector_store import dequantize_int8
from backend.free.rag.evidence.types import (
    Evidence,
    EvidenceRecordError,
    from_record,
)
from backend.io import AtomicWriter
from backend.log_config import get_logger
from backend.utils import format_utc, utc_now, utc_now_dt

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("rag.corpus.store")

#: 転置索引の上位から疑似クエリの lazy 生成対象へ積む件数 (パッケージごと、f_01 §6.4)。
LEXICAL_SEED_K = 3
#: 取りこぼした問いのヒント (f_01 §6.4 の misses) の上限。
PQ_HINTS_PER_CHUNK = 3
PQ_HINTS_MAX_CHUNKS = 500
#: misses の候補 1 件あたり、同じ節の兄弟チャンクへ広げる件数 (f_01 §6.4)。
PQ_MISS_SIBLINGS = 8
#: 見出しの先頭の番号 (「6.3 検索」→ 6)。同じ大節の判定に使う。
_HEADING_NUMBER_RE = re.compile(r"^\s*(\d+)")

CORPUS_DIR_NAME = "corpus"
CORPUS_MANIFEST_FILE = "manifest.json"
PACKAGES_DIR = "packages"
CENTROID_FILE = "centroid.npy"
EMBEDDINGS_DIR = "embeddings"

#: 順位式の ``store_prior`` — corpus の既定 (c_16 §7.2)。
DEFAULT_CORPUS_STORE_PRIOR = 1.0

#: ``rag.cartridge_gate.threshold`` が未指定でプロファイルにも無いときの既定。
DEFAULT_CARTRIDGE_GATE_THRESHOLD = 0.3

#: 保持するパッケージ版数 (c_16 §5.4)。
DEFAULT_VERSIONS_KEEP = 2


# ── centroid ゲートの閾値解決 (旧 cartridge_manager から移設) ──────────


def _profile_cartridge_gate_threshold() -> float | None:
    """有効な埋め込みモデルプロファイルの ``embedding.rag.cartridge_gate_threshold``。

    解決は :func:`backend.free.rag.memory_threshold_calibration.profile_embedding_threshold`
    (注入ゲートと同じ読み手)。不在 / 失敗は ``None`` (呼出側が 0.3 に倒す)。
    """
    from backend.free.rag.memory_threshold_calibration import (
        profile_embedding_threshold,
    )

    return profile_embedding_threshold("rag", "cartridge_gate_threshold")


def resolve_cartridge_gate_threshold(gate_cfg: dict | None) -> float:
    """``rag.cartridge_gate.threshold`` の実効値を決める。

    優先順: 明示値 → 有効な埋め込みプロファイルの値 → 0.3。cos の絶対値は
    埋め込みモデルごとに分布が違う (無関係ペアの中央値が LFM2.5 0.105 /
    Qwen3 0.273 / bge-m3 0.459) ため、未指定 (``None``) はモデル固有値に
    追随させ、固定の 0.3 を転写しない。
    """
    explicit = (gate_cfg or {}).get("threshold")
    if explicit is not None:
        return float(explicit)
    from_profile = _profile_cartridge_gate_threshold()
    if from_profile is not None:
        logger.info(
            "Corpus centroid gate threshold resolved from embedding profile: %.2f",
            from_profile,
        )
        return from_profile
    return DEFAULT_CARTRIDGE_GATE_THRESHOLD


# ── 埋め込みバックエンドのアダプタ ──────────────────────────────────────


class _AttrEmbedder:
    """``EmbeddingBackend`` (メソッド形) を属性形へ薄く包む。

    :class:`EvidenceStore` は ``backend.model_name`` / ``backend.dim`` を
    **属性** として読む (``getattr(backend, "model_name", "")``) が、
    :class:`~backend.free.rag.embedding_backend.EmbeddingBackend` Protocol は
    ``model_name()`` / ``dim()`` を **メソッド** で定義している。素で渡すと
    束縛メソッドの ``repr`` がモデル名としてディレクトリ名に化けるため、ここで
    橋渡しする。既に属性形のもの (テスト用の fake) はそのまま返す。
    """

    def __init__(self, backend: "EmbeddingBackend") -> None:
        self._backend = backend

    async def embed(self, texts: list[str], **kwargs: Any) -> Any:
        return await self._backend.embed(texts, **kwargs)

    async def embed_query(self, query: str, **kwargs: Any) -> Any:
        return await self._backend.embed_query(query, **kwargs)

    @property
    def dim(self) -> int:
        return int(self._backend.dim())

    @property
    def model_name(self) -> str:
        return str(self._backend.model_name())

    @property
    def backend_type(self) -> str:
        return str(self._backend.backend_type())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def adapt_embedding_backend(backend: Any) -> Any:
    """必要なら :class:`_AttrEmbedder` で包む (属性形ならそのまま)。"""
    if backend is None:
        return None
    if callable(getattr(backend, "model_name", None)):
        return _AttrEmbedder(backend)
    return backend


def embedding_model_id_of(backend: Any) -> str:
    """埋め込みバックエンドのモデル id (取れなければ空文字)。"""
    adapted = adapt_embedding_backend(backend)
    return "" if adapted is None else str(getattr(adapted, "model_name", "") or "")


# ── corpus manifest ────────────────────────────────────────────────────


class CorpusManifest(JsonStateFile):
    """``corpus/manifest.json`` (c_16 §4.3)。

    ``active`` だけが版切替の唯一の権威。install は最後にここを書き替える。
    """

    SCHEMA_VERSION = 1

    def __init__(self, corpus_dir: Path | str) -> None:
        super().__init__(Path(corpus_dir) / CORPUS_MANIFEST_FILE, fsync=True)
        self.active: dict[str, str] = {}
        self.store_prior_overrides: dict[str, float] = {}
        self.loaded: list[str] = []

    def _to_payload(self) -> dict[str, Any]:
        return {
            "active": dict(self.active),
            "store_prior_overrides": {
                k: float(v) for k, v in self.store_prior_overrides.items()
            },
            "loaded": list(self.loaded),
        }

    def _from_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("corpus manifest payload must be an object")
        active = payload.get("active")
        self.active = {
            str(k): str(v) for k, v in (active or {}).items()
        } if isinstance(active, dict) else {}
        overrides = payload.get("store_prior_overrides")
        self.store_prior_overrides = {
            str(k): float(v) for k, v in (overrides or {}).items()
            if isinstance(v, (int, float))
        } if isinstance(overrides, dict) else {}
        loaded = payload.get("loaded")
        self.loaded = [str(v) for v in loaded] if isinstance(loaded, list) else []


# ── 1 パッケージ版 ─────────────────────────────────────────────────────


@dataclass(slots=True)
class CorpusPackage:
    """インストール済みパッケージの 1 版 (= 1 EvidenceStore)。"""

    meta: PackageMeta
    directory: Path
    store: EvidenceStore
    centroid: np.ndarray | None = None
    doc_count: int = 0
    chunk_count: int = 0
    size_mb: float = 0.0
    installed_at: str = ""
    loaded: bool = False
    #: 疑似クエリ索引 (f_01 §6)。版ディレクトリ配下の独立した EvidenceStore。
    pseudo_queries: PseudoQueryIndex | None = None

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def version(self) -> str:
        return self.meta.version

    @property
    def tool_hints(self) -> list[dict[str, Any]]:
        return self.meta.tool_hints

    @property
    def embedding_model_id(self) -> str:
        return self.store.manifest.embedding_model_id

    @property
    def embedding_dim(self) -> int:
        return int(self.store.manifest.embedding_dim)


@dataclass(slots=True, frozen=True)
class CorpusHit:
    """検索 1 件 (c_16 §7.2 の score と素の cosine を分けて持つ)。"""

    package_id: str
    evidence_id: str
    cosine: float
    score: float
    text: str
    heading: str = ""
    doc_id: str = ""
    #: 疑似クエリ由来のとき、最良の問いが取りこぼした問いの言い換えか (f_01 §6.3)。
    hinted: bool = False


@dataclass(slots=True)
class InstallResult:
    """install / rebuild の結果 (進捗 UI とレスポンスが読む)。"""

    package: CorpusPackage
    reused_prebuilt: bool = False
    embedded: int = 0
    errors: list[str] = field(default_factory=list)


class CorpusInstallCancelled(Exception):
    """インストール処理がユーザー要求でキャンセルされた。"""


ProgressCallback = Callable[[dict], Any]
CancelCheck = Callable[[], bool]


async def _emit(callback: ProgressCallback | None, frame: dict) -> None:
    """進捗コールバックが ``None`` でないときだけ呼ぶ (await 可能なら await)。"""
    if callback is None:
        return
    result = callback(frame)
    if hasattr(result, "__await__"):
        await result


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise CorpusInstallCancelled("Corpus package install cancelled")


def version_sort_key(version: str) -> tuple:
    """semver をおおまかに並べる鍵 (数値部分だけ見る)。

    prerelease の厳密な順序は要らない — GC が「古い版から捨てる」ために使う
    だけで、同値になっても active 版は必ず保護されるため。
    """
    core = version.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return (tuple(parts[:3]), version)


class CorpusStore:
    """文書由来チャンクの実行時ストア (c_16 §4.3)。

    Args:
        corpus_dir: ``local/memory/corpus``。
        embedding_backend: 埋め込みバックエンド。``None`` でも既存パッケージの
            検索はできる (ベクトルはディスク上にある)。install / rebuild には
            必要。:meth:`set_embedding_backend` で後から差せる。
        rag_config: ``rag`` セクション (+ c_16 §9 の ``memory.evidence.*`` を
            マージしたもの)。``quantization`` / ``memmap_threshold`` /
            ``cluster_index`` / ``cartridge_gate`` / ``ranking.store_prior`` /
            ``lexical`` を読む。
        debug_logger: ``rag.jsonl`` へゲートの採否を残す。
    """

    def __init__(
        self,
        corpus_dir: Path | str,
        embedding_backend: "EmbeddingBackend | None" = None,
        rag_config: Any = None,
        debug_logger: "DebugLogger | None" = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.packages_dir = self.corpus_dir / PACKAGES_DIR
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.rag_config = rag_config if rag_config is not None else {}
        self._debug_logger = debug_logger
        self._embedding_backend = adapt_embedding_backend(embedding_backend)

        self.manifest = CorpusManifest(self.corpus_dir)
        self.manifest.load()

        gate_cfg = self._section("cartridge_gate")
        self._gate_enabled = bool(self._get(gate_cfg, "enabled", True))
        self._gate_threshold = resolve_cartridge_gate_threshold(
            {"threshold": self._get(gate_cfg, "threshold", None)},
        )
        self._gate_max_packages = int(self._get(gate_cfg, "max_cartridges", 10))
        self._gate_fallback_when_empty = bool(
            self._get(gate_cfg, "fallback_when_empty", False),
        )
        self._max_loaded = int(self._top("max_loaded_cartridges", 20))
        self._large_warn_chunks = int(self._top("large_cartridge_warn_chunks", 50000))

        #: id → パッケージ (active 版のみ)。OrderedDict の末尾 = 最近使用。
        self._packages: OrderedDict[str, CorpusPackage] = OrderedDict()
        #: 原文の見出し階層 ``[(深さ, 題)]`` (``heading_path`` 用、文書ごと)。
        self._outlines: dict[tuple[str, str], list[tuple[int, str]]] = {}
        self._on_change_callbacks: list[Callable[[str, str], None]] = []
        #: 応答パスで採用した corpus チャンク id (``<pkg>:<ev>``) のプロセス内
        #: バッファ。疑似クエリの lazy 生成対象 (f_01 §6.4)。書くのは sleep-time。
        self.pq_hits = UsageBuffer()
        #: 取りこぼした問い (f_01 §6.4 の misses) の対象チャンク。hits より先に
        #: 処理し、既に問いを持つチャンクも対象にする。
        self.pq_misses = UsageBuffer()
        #: chunk id → ユーザーの問い (ヒント)。1 チャンク 3 件、全体 500 件で打ち切る。
        self._pq_hints: dict[str, list[str]] = {}
        self._pq_hints_lock = threading.Lock()
        #: corpus 側の較正結果 (f_01 §6.6)。``None`` = 未較正 (記憶側の棒へ倒す)。
        self._calibration: dict[str, Any] | None = None
        self._discover()
        self._load_calibration_cache()

    # ── 設定の読み出し (dict / pydantic どちらでも) ──

    @staticmethod
    def _get(section: Any, key: str, default: Any) -> Any:
        if section is None:
            return default
        value = (
            section.get(key) if isinstance(section, dict)
            else getattr(section, key, None)
        )
        return default if value is None else value

    def _section(self, name: str) -> Any:
        return self._get(self.rag_config, name, None)

    def _top(self, key: str, default: Any) -> Any:
        return self._get(self.rag_config, key, default)

    def store_prior_for(self, package_id: str) -> float:
        """パッケージの ``store_prior`` (override → 設定 → 0.9)。"""
        override = self.manifest.store_prior_overrides.get(package_id)
        if override is not None:
            return float(override)
        ranking = self._section("ranking")
        priors = self._get(ranking, "store_prior", None)
        value = self._get(priors, "corpus", None)
        try:
            return float(value) if value is not None else DEFAULT_CORPUS_STORE_PRIOR
        except (TypeError, ValueError):
            return DEFAULT_CORPUS_STORE_PRIOR

    def set_store_prior_override(self, package_id: str, prior: float | None) -> None:
        """PC 固有の ``store_prior`` 上書き (``None`` で解除)。"""
        if prior is None:
            self.manifest.store_prior_overrides.pop(package_id, None)
        else:
            self.manifest.store_prior_overrides[package_id] = float(prior)
        self.manifest.save()

    # ── 埋め込みバックエンド ──

    @property
    def embedding_backend(self) -> Any:
        return self._embedding_backend

    def set_embedding_backend(self, backend: "EmbeddingBackend | None") -> None:
        """埋め込みバックエンドを差し替える (起動順の都合で後から入る)。"""
        adapted = adapt_embedding_backend(backend)
        self._embedding_backend = adapted
        for package in self._packages.values():
            package.store.embedding_backend = adapted
            if package.pseudo_queries is not None:
                package.pseudo_queries.set_embedding_backend(adapted)

    @property
    def embedding_model_id(self) -> str:
        backend = self._embedding_backend
        return "" if backend is None else str(getattr(backend, "model_name", "") or "")

    # ── 起動時の走査 ──

    def package_dir(self, package_id: str, version: str) -> Path:
        return self.packages_dir / package_id / version

    def installed_versions(self, package_id: str) -> list[str]:
        """ディスク上に残っている版 (古い順)。"""
        root = self.packages_dir / package_id
        if not root.is_dir():
            return []
        versions = [
            p.name for p in root.iterdir()
            if p.is_dir() and (p / PACKAGE_FILE).exists()
        ]
        versions.sort(key=version_sort_key)
        return versions

    def _discover(self) -> None:
        """``manifest.active`` の版を開く (壊れている版は active から外す)。"""
        stale: list[str] = []
        for package_id, version in list(self.manifest.active.items()):
            directory = self.package_dir(package_id, version)
            if not (directory / PACKAGE_FILE).exists():
                logger.warning(
                    "Corpus package %s v%s is missing on disk; dropping from active",
                    package_id, version,
                )
                stale.append(package_id)
                continue
            try:
                package = self._open_package(directory)
            except (PackageError, OSError, ValueError) as e:
                logger.warning("Failed to open corpus package %s: %s", directory, e)
                stale.append(package_id)
                continue
            package.loaded = package_id in self.manifest.loaded
            self._packages[package_id] = package
        if stale:
            for package_id in stale:
                self.manifest.active.pop(package_id, None)
                if package_id in self.manifest.loaded:
                    self.manifest.loaded.remove(package_id)
            self.manifest.save()
        logger.info(
            "CorpusStore ready: %d package(s), %d loaded (dir=%s)",
            len(self._packages), sum(1 for p in self._packages.values() if p.loaded),
            self.corpus_dir,
        )

    def _open_package(self, directory: Path) -> CorpusPackage:
        """版ディレクトリを開いて :class:`CorpusPackage` にする。"""
        meta = meta_from_record(
            json.loads((directory / PACKAGE_FILE).read_text(encoding="utf-8")),
        )
        store = self._make_store(directory, meta.id)
        store.load()
        return CorpusPackage(
            meta=meta,
            directory=directory,
            store=store,
            centroid=_load_centroid(directory),
            doc_count=_count_documents(directory / DOCS_DIR),
            chunk_count=len(store),
            size_mb=_dir_size_mb(directory / EMBEDDINGS_DIR),
            installed_at=_installed_at(directory),
            pseudo_queries=self._open_pseudo_queries(directory, meta.id, store),
        )

    def _open_pseudo_queries(
        self, directory: Path, package_id: str, store: EvidenceStore,
    ) -> PseudoQueryIndex:
        """版ディレクトリの疑似クエリ索引を開く (無ければ空のまま)。

        本体 snapshot への生存判定を束ね、対応チャンクの無い問い (孤児) を
        検索 / 充足率から外す (f_01 §6.2)。
        """
        index = PseudoQueryIndex(
            directory, package_id,
            embedding_backend=self._embedding_backend,
            rag_config=self.rag_config,
        )

        def is_live(target_id: str) -> bool:
            snapshot = store.snapshot
            return snapshot is not None and snapshot.row_of(target_id) is not None

        index.bind_targets(is_live)
        try:
            index.load()
        except (OSError, ValueError) as e:
            # 索引が壊れていてもパッケージ本体の検索は生かす。
            logger.warning(
                "Failed to open pseudo-query index for %s: %s", package_id, e,
            )
        return index

    def _make_store(self, directory: Path, package_id: str) -> EvidenceStore:
        """版ディレクトリ用の :class:`EvidenceStore` を作る。

        シャード鍵はパッケージ id 固定 — 1 版 = 1 パッケージなので、転置索引は
        1 シャードで足りる (c_16 §6.2 の「corpus はパッケージ別シャード」を
        版ディレクトリ単位で満たす)。
        """
        return EvidenceStore(
            directory,
            store_name="corpus",
            embedding_backend=self._embedding_backend,
            rag_config=self.rag_config,
            by="corpus_install",
            shard_key_for=lambda _record, key=package_id: key,
        )

    # ── 目録 ──

    def list_packages(self) -> list[CorpusPackage]:
        """インストール済みパッケージ (active 版) の一覧。"""
        return list(self._packages.values())

    def get(self, package_id: str) -> CorpusPackage | None:
        return self._packages.get(package_id)

    @property
    def loaded(self) -> dict[str, CorpusPackage]:
        """検索対象のパッケージ (LRU 順、末尾 = 最近使用)。"""
        return OrderedDict(
            (pid, pkg) for pid, pkg in self._packages.items() if pkg.loaded
        )

    @property
    def loaded_ids(self) -> list[str]:
        return [pid for pid, pkg in self._packages.items() if pkg.loaded]

    def get_tool_hints(self) -> list[dict[str, Any]]:
        """ロード済みパッケージの ``tool_hints`` を集約する。"""
        hints: list[dict[str, Any]] = []
        for package in self._packages.values():
            if package.loaded:
                hints.extend(package.tool_hints)
        return hints

    def on_change(self, callback: Callable[[str, str], None]) -> None:
        """``(event, package_id)`` を受けるコールバックを登録する。

        ``event`` は ``load`` / ``unload`` / ``uninstall``。
        """
        self._on_change_callbacks.append(callback)

    def _notify(self, event: str, package_id: str) -> None:
        for callback in self._on_change_callbacks:
            try:
                callback(event, package_id)
            except Exception as e:  # noqa: BLE001 — 通知先の失敗で本処理を止めない
                logger.warning("Corpus change callback failed: %s", e)

    # ── load / unload ──

    def load(self, package_id: str) -> CorpusPackage:
        """パッケージを検索対象に加える。"""
        package = self._packages.get(package_id)
        if package is None:
            raise KeyError(f"Corpus package '{package_id}' not found")
        if package.loaded:
            self._packages.move_to_end(package_id)
            return package
        self._evict_before_insert(package_id)
        package.loaded = True
        self._packages.move_to_end(package_id)
        self._warn_if_large(package)
        self._persist_loaded()
        logger.info(
            "Loaded corpus package %s v%s (centroid=%s)",
            package_id, package.version,
            "yes" if package.centroid is not None else "no",
        )
        self._notify("load", package_id)
        return package

    def close(self) -> None:
        """開いている全パッケージの :class:`EvidenceStore` を手放す。

        パッケージ 1 版 = 1 EvidenceStore で、それぞれが memmap を握る。
        Windows は掴んだままのファイルを消せないので、閉じないと版の GC
        (``versions_keep``) が黙って失敗する (CLAUDE.md §10)。
        """
        for package in self._packages.values():
            try:
                _release_package(package)
            except Exception as e:  # noqa: BLE001 — shutdown を止めない
                logger.warning(
                    "Failed to release corpus package %s: %s", package.meta.id, e,
                )

    def unload(self, package_id: str) -> CorpusPackage:
        """パッケージを検索対象から外す (ディスクには残す)。"""
        package = self._packages.get(package_id)
        if package is None:
            raise KeyError(f"Corpus package '{package_id}' not found")
        package.loaded = False
        self._persist_loaded()
        logger.info("Unloaded corpus package %s", package_id)
        self._notify("unload", package_id)
        return package

    def _persist_loaded(self) -> None:
        self.manifest.loaded = self.loaded_ids
        self.manifest.save()

    def _evict_before_insert(self, about_to_load: str) -> None:
        """``max_loaded_cartridges`` を保つため LRU で最古参を unload する。"""
        if self._max_loaded <= 0:
            return
        while len(self.loaded_ids) >= self._max_loaded:
            victim = next(
                (pid for pid in self.loaded_ids if pid != about_to_load), None,
            )
            if victim is None:
                return
            self._packages[victim].loaded = False
            logger.info(
                "Corpus LRU eviction: %s (max_loaded=%d)", victim, self._max_loaded,
            )
            self._notify("unload", victim)

    def _warn_if_large(self, package: CorpusPackage) -> None:
        if self._large_warn_chunks > 0 and package.chunk_count >= self._large_warn_chunks:
            logger.warning(
                "Large corpus package loaded: %s has %d chunks (>= %d). "
                "Search latency may degrade until the cluster index is built.",
                package.id, package.chunk_count, self._large_warn_chunks,
            )

    # ── install ──

    async def install(
        self,
        zip_path: Path | str,
        *,
        progress_cb: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> InstallResult:
        """`.evocart` をインストールする (c_16 §4.3)。

        手順 (順序が意味を持つ):

        1. 版ディレクトリへ展開する
        2. ``docs/`` から ``content_digest`` を計算し直し、宣言値と突き合わせる
        3. ``prebuilt/`` の ``chunker_version`` と埋め込みモデルが両方一致すれば
           チャンクと埋め込みをそのまま取り込む。違えば ``docs/`` から作り直す
        4. ``EvidenceStore.put`` → ``create_snapshot()`` で索引を組む
        5. ``centroid.npy`` を書く
        6. **最後に** ``manifest.active[id] = version`` を書く

        6 を最後にするのは、途中で落ちても manifest が「完成している版」を
        指したままにするため。
        """
        source = Path(zip_path)
        await _emit(progress_cb, {"phase": "extract", "status": "running"})
        _check_cancel(cancel_check)

        contents = read_package(source)
        meta = contents.meta
        directory = self.package_dir(meta.id, meta.version)
        if directory.exists():
            # 未完成の版が残っている / 同じ版の入れ直し。active は最後まで
            # 旧版を指しているので、ここで消しても検索は生きている。
            shutil.rmtree(str(directory), ignore_errors=True)
        contents = read_package(source, directory)
        meta = contents.meta
        await _emit(
            progress_cb,
            {"phase": "extract", "status": "done", "detail": meta.id},
        )

        docs_dir = directory / DOCS_DIR
        actual_digest = compute_content_digest(docs_dir)
        if not actual_digest:
            shutil.rmtree(str(directory), ignore_errors=True)
            raise PackageError(f"package '{meta.id}' has no documents")
        if meta.content_digest and meta.content_digest != actual_digest:
            shutil.rmtree(str(directory), ignore_errors=True)
            raise PackageError(
                f"package '{meta.id}' content_digest mismatch: declared "
                f"{meta.content_digest[:16]}…, actual {actual_digest[:16]}…",
            )
        meta = replace(meta, content_digest=actual_digest)
        write_package_meta(directory, meta)

        result = await self._build_version(
            directory, meta, contents.prebuilt,
            progress_cb=progress_cb, cancel_check=cancel_check,
        )

        previous = self.manifest.active.get(meta.id)
        self.manifest.active[meta.id] = meta.version
        if meta.id not in self.manifest.loaded:
            self.manifest.loaded.append(meta.id)
        self.manifest.save()

        package = result.package
        package.loaded = True
        self._evict_before_insert(meta.id)
        self._packages[meta.id] = package
        self._packages.move_to_end(meta.id)
        self._warn_if_large(package)

        logger.info(
            "Installed corpus package %s v%s (was %s): %d docs, %d chunks, "
            "prebuilt=%s",
            meta.id, meta.version, previous or "-", package.doc_count,
            package.chunk_count, "reused" if result.reused_prebuilt else "rebuilt",
        )
        return result

    async def _build_version(
        self,
        directory: Path,
        meta: PackageMeta,
        prebuilt: PrebuiltInfo | None,
        *,
        progress_cb: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> InstallResult:
        """展開済みの版ディレクトリに snapshot / 埋め込み / centroid を作る。"""
        docs_dir = directory / DOCS_DIR
        model_id = self.embedding_model_id
        reused = self._adopt_prebuilt(directory, prebuilt, model_id)

        _check_cancel(cancel_check)
        await _emit(progress_cb, {"phase": "chunk_embed", "status": "running"})
        if reused:
            records = list(self._read_prebuilt_chunks(directory))
        else:
            def report(index: int, total: int, doc_id: str) -> None:
                logger.debug("chunking %s (%d/%d)", doc_id, index, total)

            records = chunk_documents(
                docs_dir,
                package_id=meta.id,
                package_version=meta.version,
                content_digest=meta.content_digest,
                language=meta.language,
                rag_config=self.rag_config,
                on_document=report,
            )
        if not records:
            raise PackageError(
                f"package '{meta.id}' produced no chunks from {docs_dir}",
            )

        store = self._make_store(directory, meta.id)
        store.load()
        # 版ディレクトリの manifest に「どの chunker 版で作ったか」を刻む
        # (c_16 §5.1)。刻んでおかないと、規則を変えたあとに再構築が要る版を
        # 見分けられない。
        store.manifest.chunker_version = CHUNKER_VERSION
        for record in records:
            store.put(record, by="corpus_install")
        await _emit(
            progress_cb,
            {
                "phase": "chunk_embed", "status": "done",
                "current": len(records), "total": len(records),
            },
        )

        _check_cancel(cancel_check)
        await _emit(progress_cb, {"phase": "index", "status": "running"})
        await store.create_snapshot()
        # corpus は事象ログを持たない (c_16 §2.1) — 版が履歴そのもので内容は
        # 不変なので、畳み込んだ後の events/ は本文の二重持ちにしかならない。
        # manifest の folded_through は残るが、読み手は「その位置以降の事象は
        # 無い」と見るだけなので整合する。
        shutil.rmtree(str(directory / "events"), ignore_errors=True)
        await _emit(progress_cb, {"phase": "index", "status": "done"})

        centroid = _save_centroid(directory, store)
        package = CorpusPackage(
            meta=meta,
            directory=directory,
            store=store,
            centroid=centroid,
            doc_count=_count_documents(docs_dir),
            chunk_count=len(store),
            size_mb=_dir_size_mb(directory / EMBEDDINGS_DIR),
            installed_at=utc_now(),
            pseudo_queries=self._open_pseudo_queries(directory, meta.id, store),
        )
        return InstallResult(
            package=package,
            reused_prebuilt=reused,
            embedded=0 if reused else len(records),
        )

    def _adopt_prebuilt(
        self, directory: Path, prebuilt: PrebuiltInfo | None, model_id: str,
    ) -> bool:
        """``prebuilt/`` を採用できるなら埋め込みを版ディレクトリへ移す。

        採用条件は **chunker 版と埋め込みモデルの両方一致** (c_16 §4.3)。
        どちらか違えば ``docs/`` から作り直す — チャンクの id は
        ``(content_digest, chunker_version, doc_id, position)`` の決定論
        ハッシュなので、作り直しても同じ id に落ち着く。

        埋め込みは ``embeddings/<model_id>/`` へ複写しておくだけでよい。
        :meth:`EvidenceStore.embed_and_index_snapshot` が ``text_hash`` の
        一致で int8 の行を流用するので、埋め込み呼び出しは 0 回になる。
        """
        if prebuilt is None:
            return False
        if prebuilt.chunker_version != CHUNKER_VERSION:
            logger.info(
                "Ignoring prebuilt for %s: chunker version %d != %d",
                directory.name, prebuilt.chunker_version, CHUNKER_VERSION,
            )
            return False
        if not model_id or prebuilt.embedding_model_id != model_id:
            logger.info(
                "Ignoring prebuilt for %s: embedding model %r != %r",
                directory.name, prebuilt.embedding_model_id, model_id,
            )
            return False

        source = (
            directory / PREBUILT_DIR / PREBUILT_EMBEDDINGS_DIR
            / prebuilt.embedding_model_id
        )
        if not source.is_dir():
            logger.warning(
                "Prebuilt for %s declares model %s but has no embeddings dir",
                directory.name, prebuilt.embedding_model_id,
            )
            return False
        target = directory / EMBEDDINGS_DIR / model_id
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source), str(target))
        logger.info("Adopted prebuilt embeddings for %s (%s)", directory.name, model_id)
        return True

    @staticmethod
    def _read_prebuilt_chunks(directory: Path) -> list[Evidence]:
        """``prebuilt/chunks.jsonl`` を :class:`Evidence` にする。"""
        records: list[Evidence] = []
        skipped = 0
        for raw in iter_prebuilt_chunks(directory / PREBUILT_DIR):
            try:
                records.append(from_record(raw))
            except EvidenceRecordError:
                skipped += 1
        if skipped:
            logger.warning("prebuilt chunks: skipped %d unreadable record(s)", skipped)
        return records

    # ── rebuild / uninstall / GC ──

    async def rebuild(
        self,
        package_id: str,
        *,
        progress_cb: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> InstallResult:
        """``docs/`` からチャンクと埋め込みを作り直す (埋め込みモデル切替後)。

        ``package.json`` と ``docs/`` は残し、派生物 (snapshot / 索引 /
        埋め込み / centroid / EvidenceStore manifest) だけを捨ててから組み直す。
        """
        package = self._packages.get(package_id)
        if package is None:
            raise KeyError(f"Corpus package '{package_id}' not found")
        directory = package.directory
        docs_dir = directory / DOCS_DIR
        if not docs_dir.is_dir() or not any(p.is_file() for p in docs_dir.rglob("*")):
            raise PackageError(f"No documents found for corpus package '{package_id}'")

        was_loaded = package.loaded
        try:
            old_chunker = int(package.store.manifest.chunker_version)
        except (AttributeError, TypeError, ValueError):
            old_chunker = CHUNKER_VERSION
        # メモリ上の参照を先に落とす (Windows で memmap を握ったままだと
        # embeddings/ を消せない)。
        self._packages.pop(package_id, None)
        _release_package(package)
        _clean_derived(directory)
        if old_chunker != CHUNKER_VERSION:
            # チャンク id が全部変わるので疑似クエリ索引は孤児になる。残すと
            # 充足率を水増しし検索では読み飛ばされるだけ (f_01 §3.3 の 6)。
            shutil.rmtree(str(directory / PSEUDO_QUERIES_DIR), ignore_errors=True)
            logger.info(
                "Dropped the pseudo-query index of %s (chunker v%d -> v%d changes "
                "every chunk id)", package_id, old_chunker, CHUNKER_VERSION,
            )

        meta = replace(
            package.meta, content_digest=compute_content_digest(docs_dir),
        )
        write_package_meta(directory, meta)
        result = await self._build_version(
            directory, meta, None,
            progress_cb=progress_cb, cancel_check=cancel_check,
        )
        rebuilt = result.package
        rebuilt.loaded = was_loaded
        self._packages[package_id] = rebuilt
        self._persist_loaded()
        if rebuilt.pseudo_queries is not None and rebuilt.pseudo_queries.orphan_count():
            # 本文が変わったチャンクは id も変わる → その問いは孤児。生まれた
            # 時点で版から落とす (残すと充足率を水増しし、上位 k を押し出す)。
            try:
                await rebuilt.pseudo_queries.commit()
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    "Failed to drop orphaned pseudo queries of %s: %s", package_id, e,
                )
        logger.info(
            "Rebuilt corpus package %s v%s: %d docs, %d chunks",
            package_id, meta.version, rebuilt.doc_count, rebuilt.chunk_count,
        )
        return result

    def uninstall(self, package_id: str) -> None:
        """全版をディスクから消す。"""
        package = self._packages.get(package_id)
        if package is None and package_id not in self.manifest.active:
            raise KeyError(f"Corpus package '{package_id}' not found")
        if package is not None:
            _release_package(package)
        self._packages.pop(package_id, None)
        root = self.packages_dir / package_id
        if root.exists():
            shutil.rmtree(str(root), ignore_errors=True)
        self.manifest.active.pop(package_id, None)
        self.manifest.store_prior_overrides.pop(package_id, None)
        if package_id in self.manifest.loaded:
            self.manifest.loaded.remove(package_id)
        self.manifest.save()
        logger.info("Uninstalled corpus package %s", package_id)
        self._notify("uninstall", package_id)

    def gc_old_versions(self, keep: int = DEFAULT_VERSIONS_KEEP) -> list[tuple[str, str]]:
        """非 active の古い版を消す (直近 ``keep`` 版を保持、c_16 §5.4)。

        active 版は件数に関わらず必ず残す — 消すと検索が空になる。
        """
        if keep < 1:
            return []
        removed: list[tuple[str, str]] = []
        for root in sorted(p for p in self.packages_dir.iterdir() if p.is_dir()):
            package_id = root.name
            versions = self.installed_versions(package_id)
            active = self.manifest.active.get(package_id)
            keepers = set(versions[-keep:])
            if active:
                keepers.add(active)
            for version in versions:
                if version in keepers:
                    continue
                shutil.rmtree(str(root / version), ignore_errors=True)
                removed.append((package_id, version))
        if removed:
            logger.info(
                "GC'd %d old corpus package version(s): %s",
                len(removed),
                ", ".join(f"{pid}@{ver}" for pid, ver in removed),
            )
        return removed

    # ── prebuilt の書き出し (Pro のパッケージ作成が使う) ──

    async def build_prebuilt(
        self, staging_dir: Path | str, meta: PackageMeta,
    ) -> PrebuiltInfo | None:
        """``staging/docs`` から ``staging/prebuilt/`` を作る (c_16 §4.3)。

        作成側 (Pro) が現在の埋め込みモデルで事前構築しておくと、同じモデルを
        使う受け手は install 時に埋め込みを 1 回も呼ばずに済む。埋め込み
        バックエンドが無ければ ``None`` (prebuilt 無しのパッケージになる)。
        """
        backend = self._embedding_backend
        if backend is None:
            return None
        staging = Path(staging_dir)
        docs_dir = staging / DOCS_DIR
        digest = compute_content_digest(docs_dir)
        if not digest:
            return None
        records = chunk_documents(
            docs_dir,
            package_id=meta.id,
            package_version=meta.version,
            content_digest=digest,
            language=meta.language,
            rag_config=self.rag_config,
        )
        if not records:
            return None

        import tempfile

        with tempfile.TemporaryDirectory(prefix="evocart-prebuilt-") as tmp:
            work = Path(tmp)
            store = self._make_store(work, meta.id)
            store.load()
            for record in records:
                store.put(record, by="corpus_prebuild")
            await store.create_snapshot()
            model_id = store.manifest.embedding_model_id or self.embedding_model_id
            dimension = int(store.manifest.embedding_dim)
            source = work / EMBEDDINGS_DIR / model_id
            prebuilt_dir = staging / PREBUILT_DIR
            target = prebuilt_dir / PREBUILT_EMBEDDINGS_DIR / model_id
            _release_store(store)
            if target.exists():
                shutil.rmtree(str(target), ignore_errors=True)
            if source.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(source), str(target))

        write_prebuilt_chunks(prebuilt_dir, [r.to_record() for r in records])
        info = PrebuiltInfo(
            chunker_version=CHUNKER_VERSION,
            embedding_model_id=model_id,
            embedding_dim=dimension,
        )
        write_prebuilt_build(prebuilt_dir, info)
        logger.info(
            "Built prebuilt for %s: %d chunk(s), model=%s dim=%d",
            meta.id, len(records), model_id, dimension,
        )
        return info

    # ── 検索 (c_16 §6.3 / §7) ──

    def centroid_gate(
        self, query_vec: np.ndarray, package_ids: Sequence[str],
    ) -> list[str]:
        """パッケージ centroid との素の cosine で候補を絞る。

        - centroid 未構築のパッケージは常に通す (段階移行のため)
        - 全件不通過時は ``fallback_when_empty`` で挙動を切替。既定 (False) は
          空を返して corpus 検索自体を skip する — 雑談・ファイル生成依頼など
          ロード中パッケージと無関係な発話で chunk が混入するのを防ぐ
        - 元の LRU 順を保って返す (ラウンドロビンの決定性維持)
        """
        ids = list(package_ids)
        if not self._gate_enabled or not ids:
            return ids
        query = np.asarray(query_vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm < 1e-9:
            return ids
        query = query / norm

        scored: list[tuple[str, float]] = []
        rejected: list[tuple[str, float]] = []
        uncomputed: list[str] = []
        for package_id in ids:
            package = self._packages.get(package_id)
            centroid = None if package is None else package.centroid
            if centroid is None:
                uncomputed.append(package_id)
                continue
            if centroid.shape[0] != query.shape[0]:
                # 次元が違う = 別モデルで作った centroid。ここで落とさず通し、
                # 次元検査 (check_dimension_consistency) の担当にする。
                uncomputed.append(package_id)
                continue
            similarity = float(np.dot(query, centroid))
            if similarity >= self._gate_threshold:
                scored.append((package_id, similarity))
            else:
                rejected.append((package_id, similarity))

        if rejected and self._debug_logger is not None:
            self._debug_logger.log_rag_selection(
                query="corpus_centroid_gate", quality="corpus_centroid_gate",
                floor=self._gate_threshold, kept=scored, rejected=rejected,
            )

        if self._gate_max_packages > 0:
            scored.sort(key=lambda item: -item[1])
            scored = scored[: self._gate_max_packages]

        passed = {package_id for package_id, _ in scored} | set(uncomputed)
        if not passed:
            if self._gate_fallback_when_empty:
                logger.debug(
                    "Corpus gate: 0 passed of %d (threshold=%.2f), falling back to all",
                    len(ids), self._gate_threshold,
                )
                return ids
            logger.debug(
                "Corpus gate: 0 passed of %d (threshold=%.2f), skipping corpus search",
                len(ids), self._gate_threshold,
            )
            return []
        return [package_id for package_id in ids if package_id in passed]

    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int = 5,
        *,
        per_package_k: int | None = None,
        threshold: float = 0.0,
        now: float | None = None,
        include_private: bool = False,
    ) -> list[CorpusHit]:
        """ロード済みパッケージ横断検索 (c_16 §6.3 / §7)。

        Args:
            query_text: 転置索引に渡す生のクエリ。空文字ならベクトル候補だけを
                使う (チャット応答パスの現行呼出はベクトルしか持たない)。
            query_vec: クエリの埋め込み。
            top_k: 返す最大件数。
            per_package_k: 各パッケージから引く件数 (既定 ``top_k``)。
            threshold: cosine のゲート閾値。0 で素通し (上流の品質判定 /
                floor がゲートを担っている間はこちらは開けておく)。
            now: 現在時刻の epoch 秒。全パッケージで同じ値を使い、鮮度が
                パッケージごとにずれないようにする。
            include_private: private セッションでのみ ``True``。

        Returns:
            :class:`CorpusHit` を ``score`` 降順で最大 ``top_k`` 件。
        """
        if top_k <= 0:
            return []
        candidates = self.centroid_gate(query_vec, self.loaded_ids)
        if not candidates:
            return []
        per_package = per_package_k if per_package_k is not None else top_k
        now_epoch = utc_now_dt().timestamp() if now is None else float(now)

        hits: list[CorpusHit] = []
        for package_id in candidates:
            package = self._packages.get(package_id)
            if package is None:
                continue
            prior = self.store_prior_for(package_id)
            rows = package.store.search(
                query_text, query_vec, per_package,
                threshold=threshold,
                now=now_epoch,
                include_private=include_private,
                store_prior=prior,
            )
            snapshot = package.store.snapshot
            if query_text and package.pseudo_queries is not None:
                self._seed_lexical_pq_targets(
                    package_id, package, query_text, now_epoch, include_private,
                )
            for row, cosine, score in rows:
                raw = snapshot.raw_at(row) if snapshot is not None else None
                attrs = (raw or {}).get("attrs") or {}
                hits.append(
                    CorpusHit(
                        package_id=package_id,
                        evidence_id=str(
                            (raw or {}).get("id")
                            or (snapshot.id_at(row) if snapshot is not None else ""),
                        ),
                        cosine=float(cosine),
                        score=float(score),
                        text=str((raw or {}).get("text") or ""),
                        heading=str(attrs.get("heading") or ""),
                        doc_id=str(attrs.get("doc_id") or ""),
                    ),
                )
            self._packages.move_to_end(package_id)

        hits.sort(key=lambda hit: -hit.score)
        return hits[:top_k]

    def search_pseudo(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        *,
        now: float | None = None,
        include_private: bool = False,
    ) -> list[CorpusHit]:
        """疑似クエリ索引を引き、対象チャンク本体の値で :class:`CorpusHit` にする。

        並びは **問い↔問いの cosine 順** (呼出側が疑似クエリ側先頭で interleave
        する)。``cosine`` は **問い↔問いの cosine** (ゲート / floor 用)、
        ``score`` は対象チャンク本体の順位式の値。較正済みの棒はクエリ↔発話
        (問い同士に近い) の分布から作られるので、問い同士の cosine の方が
        本体 cosine より棒のスケールに合う (f_01 §6.3 の実測)。本体の行が無い
        対象 (再構築で id が変わった等) は読み飛ばす。
        """
        if top_k <= 0:
            return []
        candidates = self.centroid_gate(query_vec, self.loaded_ids)
        if not candidates:
            return []
        now_epoch = utc_now_dt().timestamp() if now is None else float(now)
        ranked: list[tuple[float, CorpusHit]] = []
        for package_id in candidates:
            package = self._packages.get(package_id)
            if package is None or package.pseudo_queries is None:
                continue
            index = package.pseudo_queries
            if len(index) == 0:
                continue
            pairs = index.search(query_vec, top_k)
            if not pairs:
                continue
            snapshot = package.store.snapshot
            if snapshot is None:
                continue
            mask = package.store.active_mask(now_epoch, include_private)
            rows: list[int] = []
            kept: list[tuple[str, float, bool]] = []
            for target_id, pq_cos, hinted in pairs:
                row = snapshot.row_of(target_id)
                if row is None or not bool(mask[row]):
                    continue
                rows.append(row)
                kept.append((target_id, pq_cos, hinted))
            if not rows:
                continue
            row_array = np.asarray(rows, dtype=np.int64)
            cosines = package.store.cosines_for_rows(query_vec, row_array)
            scores = score_rows(
                np.nan_to_num(cosines, nan=0.0), row_array,
                RankColumns.from_columns(snapshot.columns),
                now_epoch, self.store_prior_for(package_id),
            )
            for (target_id, pq_cos, hinted), row, cosine, score in zip(
                kept, rows, cosines, scores, strict=True,
            ):
                if not np.isfinite(cosine):
                    continue
                raw = snapshot.raw_at(row) or {}
                attrs = raw.get("attrs") or {}
                ranked.append((
                    float(pq_cos),
                    CorpusHit(
                        package_id=package_id,
                        evidence_id=target_id,
                        cosine=float(pq_cos),
                        score=float(score),
                        hinted=hinted,
                        text=str(raw.get("text") or ""),
                        heading=str(attrs.get("heading") or ""),
                        doc_id=str(attrs.get("doc_id") or ""),
                    ),
                ))
        ranked.sort(key=lambda item: -item[0])
        return [hit for _, hit in ranked[:top_k]]

    def lexical_seat(
        self,
        query_text: str,
        query_vec: np.ndarray,
        exclude_ids: Iterable[str],
        *,
        seats: int = 1,
        now: float | None = None,
        include_private: bool = False,
    ) -> list[CorpusHit]:
        """転置索引の上位 ``seats`` 件で、``exclude_ids`` に無い行を返す (f_01 §8.1 の 4.3)。

        ``cosine`` / ``score`` は本体の値 (:meth:`search` と同じスケール)。順位式
        にもスコアにも触らず、呼出側が棒を掛けて位置で置く。転置索引は
        パッケージごとに引き、語彙順のまま返す。
        """
        if seats <= 0 or not query_text:
            return []
        excluded = set(exclude_ids)
        now_epoch = utc_now_dt().timestamp() if now is None else float(now)
        out: list[CorpusHit] = []
        for package_id in self.centroid_gate(query_vec, self.loaded_ids):
            package = self._packages.get(package_id)
            if package is None:
                continue
            store = package.store
            snapshot = store.snapshot
            if snapshot is None:
                continue
            try:
                mask = store.active_mask(now_epoch, include_private)
                rows, _scores = store.lexical_candidates(
                    query_text, seats + len(excluded), row_mask=mask,
                )
            except Exception as e:  # noqa: BLE001 — 席の探索で検索を止めない
                logger.debug("lexical seat skipped for %s: %s", package_id, e)
                continue
            picked: list[int] = []
            for row in rows:
                if f"{package_id}:{snapshot.id_at(int(row))}" in excluded:
                    continue
                picked.append(int(row))
                if len(picked) >= seats:
                    break
            if not picked:
                continue
            row_array = np.asarray(picked, dtype=np.int64)
            cosines = store.cosines_for_rows(query_vec, row_array)
            scores = score_rows(
                np.nan_to_num(cosines, nan=0.0), row_array,
                RankColumns.from_columns(snapshot.columns),
                now_epoch, self.store_prior_for(package_id),
            )
            for row, cosine, score in zip(picked, cosines, scores, strict=True):
                if not np.isfinite(cosine):
                    continue
                raw = snapshot.raw_at(row) or {}
                attrs = raw.get("attrs") or {}
                out.append(CorpusHit(
                    package_id=package_id,
                    evidence_id=str(raw.get("id") or snapshot.id_at(row)),
                    cosine=float(cosine),
                    score=float(score),
                    text=str(raw.get("text") or ""),
                    heading=str(attrs.get("heading") or ""),
                    doc_id=str(attrs.get("doc_id") or ""),
                ))
        return out

    def heading_path(self, package_id: str, evidence_id: str) -> list[str]:
        """チャンクの見出し階層 (親 → 自分) を原文から引く (f_01 §6.4)。

        パッケージ同梱の原文の見出し行 (`#` の深さ) をたどり、チャンクの
        ``attrs.heading`` に至る祖先を返す。原文が無い / 見出しが見つからない
        ときは ``attrs.heading`` だけ (無ければ空)。
        """
        described = self.describe_chunk(package_id, evidence_id)
        if described is None:
            return []
        heading = described["heading"]
        doc_id = described["doc_id"]
        if not heading or not doc_id:
            return [heading] if heading else []
        package = self._packages.get(package_id)
        if package is None:
            return [heading]
        key = (package_id, doc_id)
        outline = self._outlines.get(key)
        if outline is None:
            outline = _document_outline(package.directory / DOCS_DIR / doc_id)
            self._outlines[key] = outline
        stack: list[tuple[int, str]] = []
        for level, title in outline:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            if title == heading:
                return [t for _, t in stack]
        return [heading]

    def describe_chunk(self, package_id: str, evidence_id: str) -> dict[str, Any] | None:
        """出典表示用にチャンクの所在 (文書名 / 見出し) を引く。無ければ ``None``。"""
        package = self._packages.get(package_id)
        if package is None:
            return None
        snapshot = package.store.snapshot
        if snapshot is None:
            return None
        row = snapshot.row_of(evidence_id)
        if row is None:
            return None
        attrs = ((snapshot.raw_at(row) or {}).get("attrs")) or {}
        return {
            "package_id": package_id,
            "package_name": package.meta.name or package_id,
            "doc_id": str(attrs.get("doc_id") or ""),
            "heading": str(attrs.get("heading") or ""),
        }

    def _seed_lexical_pq_targets(
        self, package_id: str, package, query_text: str, now_epoch: float,
        include_private: bool,
    ) -> None:
        """転置索引の上位候補を疑似クエリの lazy 生成対象に積む (f_01 §6.4)。

        順位は cosine 1 本なので、固有語で当たったチャンクは cosine が低いと
        採用されず lazy の対象にも入らない。語彙一致は「ユーザーがその語で
        訊く」証拠なので、採用に関わらず Step 5.9 の対象にする (2026-09-12 (b))。
        """
        try:
            store = package.store
            mask = store.active_mask(now_epoch, include_private)
            rows, _scores = store.lexical_candidates(
                query_text, LEXICAL_SEED_K, row_mask=mask,
            )
        except Exception as e:  # noqa: BLE001 — 種まきで検索を止めない
            logger.debug("lexical pq seeding skipped for %s: %s", package_id, e)
            return
        snapshot = store.snapshot
        if snapshot is None:
            return
        ids = [f"{package_id}:{snapshot.id_at(int(r))}" for r in rows[:LEXICAL_SEED_K]]
        if ids:
            self.pq_hits.add_many(ids)

    def record_pq_misses(self, chunk_ids: Sequence[str], question: str) -> int:
        """取りこぼした問いの語彙候補を misses に積み、問いをヒントとして添える (f_01 §6.4)。

        Returns:
            積んだチャンク数。
        """
        ids = [cid for cid in chunk_ids if cid]
        q = (question or "").strip()
        if not ids or not q:
            return 0
        # 同じ節の兄弟へ広げる: 答えのチャンクは候補の隣にいることが多い
        # (語が節内に散る問い、f_01 §6.4)。
        expanded: list[str] = []
        seen: set[str] = set()
        for cid in ids:
            for sib in [cid, *self._section_siblings(cid, PQ_MISS_SIBLINGS)]:
                if sib not in seen:
                    seen.add(sib)
                    expanded.append(sib)
        ids = expanded
        self.pq_misses.add_many(ids)
        with self._pq_hints_lock:
            for cid in ids:
                hints = self._pq_hints.setdefault(cid, [])
                if q not in hints and len(hints) < PQ_HINTS_PER_CHUNK:
                    hints.append(q)
            while len(self._pq_hints) > PQ_HINTS_MAX_CHUNKS:
                self._pq_hints.pop(next(iter(self._pq_hints)))
        return len(ids)

    def _section_siblings(self, chunk_id: str, cap: int) -> list[str]:
        """同じ文書・同じ大節のチャンク id (自分を除く、snapshot 行順、最大 ``cap``)。"""
        package_id, sep, evidence_id = chunk_id.partition(":")
        package = self._packages.get(package_id) if sep else None
        if package is None or cap <= 0:
            return []
        snapshot = package.store.snapshot
        if snapshot is None:
            return []
        row = snapshot.row_of(evidence_id)
        if row is None:
            return []
        attrs = (snapshot.raw_at(row) or {}).get("attrs") or {}
        doc_id = str(attrs.get("doc_id") or "")
        heading = str(attrs.get("heading") or "")
        m = _HEADING_NUMBER_RE.match(heading)
        key = m.group(1) if m else heading
        out: list[str] = []
        for r in range(len(snapshot)):
            if r == row:
                continue
            a = (snapshot.raw_at(r) or {}).get("attrs") or {}
            if str(a.get("doc_id") or "") != doc_id:
                continue
            h = str(a.get("heading") or "")
            hm = _HEADING_NUMBER_RE.match(h)
            if (hm.group(1) if hm else h) != key:
                continue
            out.append(f"{package_id}:{snapshot.id_at(r)}")
            if len(out) >= cap:
                break
        return out

    def previous_chunk_context(
        self, chunk_id: str, tail_chars: int,
    ) -> tuple[str, str] | None:
        """同じ文書・同じ大節の直前チャンクの末尾 ``tail_chars`` 文字 (f_01 §8.1 の 7.65)。

        Returns:
            ``("<pkg>:<ev>", text)``。直前が無い / 別文書 / 別の大節なら ``None``。
        """
        if tail_chars <= 0:
            return None
        package_id, sep, evidence_id = chunk_id.partition(":")
        package = self._packages.get(package_id) if sep else None
        if package is None:
            return None
        snapshot = package.store.snapshot
        if snapshot is None:
            return None
        row = snapshot.row_of(evidence_id)
        if row is None or row == 0:
            return None
        here = (snapshot.raw_at(row) or {}).get("attrs") or {}
        prev = snapshot.raw_at(row - 1) or {}
        before = prev.get("attrs") or {}
        if str(before.get("doc_id") or "") != str(here.get("doc_id") or ""):
            return None

        def _key(h: str) -> str:
            hm = _HEADING_NUMBER_RE.match(h or "")
            return hm.group(1) if hm else (h or "")

        if _key(str(before.get("heading") or "")) != _key(str(here.get("heading") or "")):
            return None
        text = str(prev.get("text") or "").strip()
        if not text:
            return None
        return f"{package_id}:{snapshot.id_at(row - 1)}", text[-tail_chars:]

    def outdated_package_ids(self) -> list[str]:
        """chunker 版が現行より古いパッケージ id (f_01 §3.3 の 6)。"""
        out: list[str] = []
        for package_id, package in self._packages.items():
            try:
                version = int(package.store.manifest.chunker_version)
            except (AttributeError, TypeError, ValueError):
                continue
            if version < CHUNKER_VERSION:
                out.append(package_id)
        return out

    def take_pq_hints(self, chunk_id: str) -> list[str]:
        """チャンクに添えられた問いを取り出して消す (Step 5.9 が消費する)。"""
        with self._pq_hints_lock:
            return list(self._pq_hints.pop(chunk_id, ()))

    def record_pq_hits(self, chunk_ids: Sequence[str]) -> None:
        """応答パスで採用した corpus チャンク id を lazy 生成の対象へ溜める。"""
        self.pq_hits.add_many([cid for cid in chunk_ids if cid])

    # ── corpus 較正 (f_01 §6.6) ──

    def pq_coverage(self) -> float:
        """ロード済みパッケージ全体の疑似クエリ充足率 (問いを持つチャンク / 全チャンク)。"""
        total = 0
        covered = 0
        for package_id in self.loaded_ids:
            package = self._packages.get(package_id)
            if package is None:
                continue
            total += package.chunk_count
            if package.pseudo_queries is not None:
                covered += len(package.pseudo_queries.covered_target_ids())
        return (covered / total) if total else 0.0

    def calibration(self) -> dict[str, Any] | None:
        """較正済み閾値 (``relevance_threshold`` … ``pq_gate``)。未較正なら ``None``。"""
        if self._calibration is None:
            return None
        return dict(self._calibration.get("thresholds") or {})

    def _pooled_vectors(self) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
        """ロード済みパッケージのチャンク本体 / 疑似クエリのベクトルを 1 つに束ねる。"""
        chunk_blocks: list[np.ndarray] = []
        pq_blocks: list[np.ndarray] = []
        targets: list[int] = []
        offset = 0
        for package_id in self.loaded_ids:
            package = self._packages.get(package_id)
            if package is None:
                continue
            store = package.store.vector_store()
            if store is None or store.vectors_q8 is None or store.scales is None:
                continue
            chunk_vecs = dequantize_int8(np.asarray(store.vectors_q8), np.asarray(store.scales))
            if chunk_vecs.ndim != 2 or chunk_vecs.shape[0] == 0:
                continue
            row_of = {str(m.get("id")): i for i, m in enumerate(store.metadata)}
            chunk_blocks.append(chunk_vecs)
            index = package.pseudo_queries
            if index is not None and len(index) > 0:
                pq_store = index.store.vector_store()
                snapshot = index.store.snapshot
                if (
                    pq_store is not None and pq_store.vectors_q8 is not None
                    and pq_store.scales is not None and snapshot is not None
                ):
                    pq_vecs = dequantize_int8(
                        np.asarray(pq_store.vectors_q8), np.asarray(pq_store.scales),
                    )
                    keep: list[int] = []
                    for i, meta in enumerate(pq_store.metadata):
                        row = snapshot.row_of(str(meta.get("id")))
                        raw = snapshot.raw_at(row) if row is not None else None
                        target = ((raw or {}).get("attrs") or {}).get("target_id")
                        chunk_row = row_of.get(str(target)) if target else None
                        if chunk_row is None or i >= pq_vecs.shape[0]:
                            continue
                        keep.append(i)
                        targets.append(offset + chunk_row)
                    if keep:
                        pq_blocks.append(pq_vecs[keep])
            offset += chunk_vecs.shape[0]
        if not chunk_blocks:
            return None
        chunks = np.concatenate(chunk_blocks, axis=0)
        pqs = (
            np.concatenate(pq_blocks, axis=0)
            if pq_blocks else np.zeros((0, chunks.shape[1]), dtype=np.float32)
        )
        return chunks, pqs, targets

    def _calibration_signature(self, n_chunks: int, n_pq: int) -> str:
        return calibration_signature(self.embedding_model_id, n_chunks, n_pq)

    def _load_calibration_cache(self) -> None:
        pooled = self._pooled_vectors()
        if pooled is None:
            return
        chunks, pqs, _targets = pooled
        cached = load_corpus_calibration(
            self.corpus_dir, self._calibration_signature(chunks.shape[0], pqs.shape[0]),
        )
        if cached:
            self._calibration = cached
            logger.info("Corpus calibration loaded from cache: %s", cached.get("thresholds"))

    async def recalibrate(self, *, force: bool = False) -> dict[str, Any] | None:
        """疑似クエリとカナリア発話から corpus の棒を導き直す (sleep-time / 起動時)。

        署名 (埋め込み指紋 + チャンク数 + 疑似クエリ数) が一致するキャッシュが
        あれば再計算しない。``force`` で強制。
        """
        backend = self._embedding_backend
        pooled = self._pooled_vectors()
        if backend is None or pooled is None:
            return self._calibration
        chunks, pqs, targets = pooled
        signature = self._calibration_signature(chunks.shape[0], pqs.shape[0])
        if not force:
            cached = load_corpus_calibration(self.corpus_dir, signature)
            if cached:
                self._calibration = cached
                return cached
        if pqs.shape[0] == 0:
            return self._calibration
        try:
            canary = np.asarray(
                await backend.embed(list(CANARY_UTTERANCES), is_query=True, mode="chat"),
                dtype=np.float32,
            )
        except Exception as e:  # noqa: BLE001 — 較正の失敗で sleep-time を止めない
            logger.warning("Corpus calibration skipped (canary embedding failed): %s", e)
            return self._calibration
        result = compute_corpus_calibration(chunks, pqs, targets, canary)
        if not result.get("ok"):
            logger.info("Corpus calibration not applied: %s", result.get("reason"))
            return self._calibration
        payload = {
            "signature": signature,
            "n_chunks": result["n_chunks"],
            "n_pq": result["n_pq"],
            "distribution": result["distribution"],
            "thresholds": result["thresholds"],
        }
        self._calibration = payload
        save_corpus_calibration(self.corpus_dir, signature, result)
        return payload

    # ── 次元検査 ──

    def check_dimension_consistency(self, embedder_dim: int) -> list[str]:
        """全パッケージの埋め込み次元を現在の埋め込みモデルと突き合わせる。

        不一致のパッケージは **検索対象から外す** (unload)。載せたままにすると
        ``VectorStore.search`` が次元不一致で例外を投げ、1 パッケージのせいで
        検索パイプライン全体が落ちる (2026-09-02 監査 S-A1)。復旧は
        ``rebuild`` (``evoref reindex``)。

        Returns:
            不一致だったパッケージ id。
        """
        mismatched: list[str] = []
        changed = False
        for package_id, package in self._packages.items():
            stored = package.embedding_dim
            if not stored or stored == embedder_dim:
                continue
            mismatched.append(package_id)
            logger.warning(
                "Corpus package '%s' embedding_dim=%d != current embedder dim=%d. "
                "Run 'evoref reindex --cartridge %s' to rebuild.",
                package_id, stored, embedder_dim, package_id,
            )
            if package.loaded:
                package.loaded = False
                changed = True
                logger.warning(
                    "Corpus package '%s' excluded from search until rebuilt "
                    "(embedding dim mismatch)", package_id,
                )
        if changed:
            self._persist_loaded()
        return mismatched


# ── ディスクの小物 ──────────────────────────────────────────────────────


def _load_centroid(directory: Path) -> np.ndarray | None:
    """``centroid.npy`` を読む (無い / 壊れていれば ``None``)。"""
    path = directory / CENTROID_FILE
    if not path.exists():
        return None
    try:
        array = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError) as e:
        logger.warning("Failed to load centroid for %s: %s", directory.name, e)
        return None
    if array.ndim != 1 or array.size == 0:
        return None
    return array.astype(np.float32, copy=False)


def _save_centroid(directory: Path, store: EvidenceStore) -> np.ndarray | None:
    """正規化ベクトルの平均を ``centroid.npy`` に書く。"""
    vector_store = store.vector_store()
    centroid = None if vector_store is None else vector_store.compute_centroid()
    path = directory / CENTROID_FILE
    if centroid is None:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        return None
    with AtomicWriter(path, mode="wb") as f:
        np.save(f, centroid)
    return centroid


def _count_documents(docs_dir: Path) -> int:
    if not docs_dir.is_dir():
        return 0
    return sum(1 for p in docs_dir.rglob("*") if p.is_file())


def _dir_size_mb(directory: Path) -> float:
    if not directory.is_dir():
        return 0.0
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 3)


def _installed_at(directory: Path) -> str:
    """版ディレクトリの作成時刻 (ISO 8601 UTC)。"""
    try:
        stamp = directory.stat().st_mtime
    except OSError:
        return ""
    return format_utc(datetime.fromtimestamp(stamp, tz=UTC))


def _release_store(store: EvidenceStore) -> None:
    """memmap を握った VectorStore を手放す (Windows で削除できなくなるため)。"""
    store.close()


_OUTLINE_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(```|~~~)")


def _document_outline(path: Path) -> list[tuple[int, str]]:
    """原文の見出し行を ``[(深さ, 題)]`` で出現順に返す (読めなければ空)。

    コードフェンスの中は読まない (yaml の ``# ---`` コメントが H1 に化ける)。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _OUTLINE_HEADING_RE.match(line)
        if m and m.group(2).strip():
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


def _release_package(package: CorpusPackage) -> None:
    """本体と疑似クエリ索引の両方を手放す。"""
    _release_store(package.store)
    if package.pseudo_queries is not None:
        package.pseudo_queries.close()


def _clean_derived(directory: Path) -> None:
    """``package.json`` と ``docs/`` 以外の派生物を消す (rebuild 用)。"""
    for name in ("snapshot", EMBEDDINGS_DIR, "events", PREBUILT_DIR):
        target = directory / name
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)
    for name in ("manifest.json", CENTROID_FILE):
        target = directory / name
        if target.exists():
            try:
                target.unlink()
            except OSError as e:
                logger.warning("failed to remove %s: %s", target, e)


__all__ = [
    "CENTROID_FILE",
    "CORPUS_DIR_NAME",
    "CORPUS_MANIFEST_FILE",
    "DEFAULT_CARTRIDGE_GATE_THRESHOLD",
    "DEFAULT_CORPUS_STORE_PRIOR",
    "DEFAULT_VERSIONS_KEEP",
    "PACKAGES_DIR",
    "CorpusHit",
    "CorpusInstallCancelled",
    "CorpusManifest",
    "CorpusPackage",
    "CorpusStore",
    "InstallResult",
    "adapt_embedding_backend",
    "embedding_model_id_of",
    "merge_rag_evidence_config",
    "resolve_cartridge_gate_threshold",
    "version_sort_key",
]
