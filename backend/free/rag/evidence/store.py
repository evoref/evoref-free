"""`EvidenceStore` — 事象ログ + 版付き snapshot + ベクトル索引の基盤 (c_16 §2.1 / §5 / §6.1)

読み手は **active な snapshot と、それ以降の事象だけ** を見る。書き手
(sleep-time) は事象を追記するだけで、稼働中の snapshot / 索引には一切触れない。
索引の作り直しは :meth:`EvidenceStore.create_snapshot` の中でだけ起きる。

```
<store_dir>/
├── manifest.json              §5.1 (fsync)
├── events/<yyyy-mm>.jsonl     §5.2 追記のみ
├── snapshot/v<N>/             §5.3 records.jsonl / offsets.npy / columns.npz
│   ├── lexical.npz + shards.json  §6.2 転置索引 (全シャードを 1 ファイルに pack)
│   └── COMPLETE               §5.6 版の確定の印 (これが無い版は無効)
└── embeddings/<model_id>/v<N>/  §6.1 int8 + memmap + cluster index + 行の列 + stamp.json
```

## 事象の可視範囲 (phase 1a の割り切り)

``put`` / ``patch`` / ``retract`` / ``touch`` は追記した瞬間から
:meth:`get` / :meth:`iter_active` に反映される (プロセス内オーバーレイ)。
一方 **ベクトル検索は snapshot の行しか見ない** — 埋め込みと int8 索引は
snapshot 生成時にまとめて作るためで、新しく ``put`` したレコードが
:meth:`vector_candidates` に出るのは次の :meth:`create_snapshot` の後。
c_16 §2.1「稼働中の索引は書き換えず版を積んで指す」の帰結で、意図した挙動。

## 物理 GC は畳み込み時のフィルタ (c_16 §5.4)

``retract`` は状態遷移で、レコードを消さない (監査可能性)。行が実際に消えるのは
**版を作るときだけ** — :meth:`EvidenceStore.create_snapshot` が畳んだ結果に
``gc_filter`` を掛け、``False`` を返した id を新しい版へ書かない。落とした id は
``<store_dir>/gc.jsonl`` に 1 版 1 行で残し、manifest の ``_extra["gc"]`` にも
直近の件数を書く (消えた事実そのものは監査できるようにする)。フィルタは
``put`` / ``patch`` のオーバーレイ経路には **掛からない** — 掛けると「書いた
直後に読めない」レコードができる。

## 埋め込みは増分 (c_16 §6.1)

snapshot を作るたびに全行を埋め込み直すと、5 万件のストアでは 1 サイクルあたり
5 万回の埋め込み呼び出しになる。前の版の ``embeddings/<model_id>/`` に残っている
``text_hash`` (再利用鍵。**埋め込み側 + 本文** の sha256 先頭 16 桁) と突き合わせ、
**鍵が変わっていない id は int8 の行とスケールをそのまま流用** する (復元→
再量子化はしない。往復で値がずれる)。埋め込むのは新規 id と鍵が変わった id
だけ。クラスタ索引は numpy だけで安いので毎回作り直す。埋め込みモデルが
変わったときは全件やり直す。

## 埋め込み側はレコードが宣言する (c_16 §3.5 / §6.1)

Qwen3 系の instruction-aware な埋め込みは query 側と document 側で prefix が
変わり、**同一テキストの自己類似度が 0.78 程度まで落ちる**。既定は document 側
だが、内部索引 (``idx.command.*``) のように「過去の質問文を溜めて、今の質問
(``embed_query``) で引く」レコードは書く側も query 側で揃えないと閾値が本来の
尺度で意味を持たない。レコードは :data:`EMBED_AS_QUERY_ATTR` /
:data:`EMBED_MODE_ATTR` を ``attrs`` に立てて側を宣言し、
:meth:`EvidenceStore.embed_and_index_snapshot` が側ごとにバッチを分けて埋め込む
(行順は保つ)。側は再利用鍵にも入るので、宣言を反転させれば次の版で埋め直る。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import shutil
import threading
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.embed_priority import P1_FRESHNESS, embed_priority
from backend.free.rag.evidence.columns import active_mask as columns_active_mask
from backend.free.rag.evidence.columns import build_columns
from backend.free.rag.evidence.events import EventPosition, EvidenceEventLog
from backend.free.rag.evidence.lexical_index import (
    LEXICAL_INDEX_VERSION,
    SHARD_MAP_FILE,
    LexicalIndexBuilder,
    LexicalPack,
    LexicalParams,
    save_lexical_pack,
)
from backend.free.rag.evidence.manifest import EvidenceManifest
from backend.free.rag.embedding_backend import embedding_store_id
from backend.free.rag.evidence.tokenize import TOKENIZER_VERSION
from backend.free.rag.evidence.ranking import (
    RankColumns,
    collapse,
    gate_by_cosine,
    merge_candidates,
    score_rows,
)
from backend.free.rag.evidence.snapshot import (
    COMPLETE_FILE,
    SNAPSHOT_DIR,
    CompleteEmbedding,
    SnapshotReader,
    SnapshotWriter,
    apply_patch,
    list_incomplete_versions,
    list_versions,
    new_complete,
    prune_snapshots,
    read_complete,
    read_snapshot,
    snapshot_dir,
    trash_version,
    version_seq,
    write_complete,
)
from backend.free.rag.evidence.snapshot_build import (
    EmbedPrep,
    EmbedPrepSpec,
    IndexBuildJob,
    SnapshotBuildJob,
    run_index_build,
    run_snapshot_build,
    wait_until_idle,
)
from backend.free.rag.evidence.types import (
    EVIDENCE_RECORD_FORMAT,
    RECORD_VERSION,
    Evidence,
    EvidenceVersionError,
    check_writable,
    plain_value,
)
from backend.free.rag.evidence.vectors import (
    STAMP_FILE,
    EvidenceVectorStore,
    ids_digest,
)
from backend.free.rag.vector_store import (
    DEFAULT_MEMMAP_THRESHOLD,
    content_hash,
    dequantize_int8,
    quantize_int8,
)
from backend.io import AtomicWriter, JSONLAppendStore, format_health
from backend.io.readonly import is_readonly
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import READONLY_STATUSES, VersionedPayloadFile
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context
from backend.utils import parse_utc, utc_now, utc_now_dt

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("rag.evidence.store")

#: 埋め込みを 1 回の ``embed()`` に渡す件数。
EMBED_BATCH_SIZE = 64

#: 埋め込みモデル名が取れないときのディレクトリ名。
_UNKNOWN_MODEL = "unknown"

#: レコードが「query 側で埋め込め」と宣言する ``attrs`` キー (c_16 §3.5)。
#: 既定 (未設定 / False) は document 側。
EMBED_AS_QUERY_ATTR = "embed_as_query"

#: query 側で使う ``mode`` (``embedding.instructions`` の鍵) の ``attrs`` キー。
#: document 側では無視される (Qwen3 はドキュメント側に prefix を付けない)。
EMBED_MODE_ATTR = "embed_mode"

#: :data:`EMBED_MODE_ATTR` 未設定時の mode。
DEFAULT_EMBED_MODE = "chat"

#: tail 索引のサイドカー (c_16 §6.4)。derived — 壊れていれば捨てて埋め直す。
TAIL_DIR = "tail"
TAIL_INDEX_FILE = "index.json"
TAIL_VECTORS_FILE = "vectors.npy"
#: 封筒を持つのは索引 (JSON) だけ。ベクトル (npy) は索引の件数と突き合わせる。
TAIL_INDEX_FORMAT = register_format(FormatSpec(
    format_id="evidence.tail_index",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="store/memory/<store>/tail/index.json",
    retention="rows still in the tail; rewritten as the tail is embedded",
))

TAIL_VECTORS_FORMAT = register_format(FormatSpec(
    format_id="evidence.tail_vectors",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key=f"store/memory/<store>/{TAIL_DIR}/{TAIL_VECTORS_FILE}",
    retention="rows still in the tail; rewritten as the tail is embedded",
    encodings=("npy",),
))

#: snapshot 版の records 以外 (offsets / 列 / 語彙索引)。records から作り直せる。
SNAPSHOT_INDEX_FORMAT = register_format(FormatSpec(
    format_id="evidence.snapshot_index",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key=f"store/memory/<store>/{SNAPSHOT_DIR}/<ver>/**",
    retention="one per snapshot version (snapshots_keep)",
    encodings=("dir",),
))

#: 埋め込みの版 (``embeddings/<model_id>/<ver>/``、ids / 量子化ベクトル / クラスタ索引 / stamp)。
EMBEDDINGS_FORMAT = register_format(FormatSpec(
    format_id="evidence.embeddings",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="store/memory/<store>/embeddings/<model>/<ver>/**",
    retention="latest 2 versions per embedding model",
    encodings=("dir",),
))

#: 物理 GC の監査ログ (1 版 1 行の追記、c_16 §5.4)。
GC_LOG_FILE = "gc.jsonl"

GC_LOG_FORMAT = register_format(FormatSpec(
    format_id="evidence.gc_log",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key=f"store/memory/<store>/{GC_LOG_FILE}",
    retention="append-only, never pruned (the only trace of what was dropped)",
    export=True,
    encodings=("jsonl",),
))

#: :data:`GC_LOG_FILE` の行の版 (``_v``)。行の形を変えるときに上げる。
GC_LOG_ROW_VERSION = 1

#: ``shard_key_for`` を渡さないときの唯一のシャード名。
DEFAULT_SHARD = "all"

#: 語彙索引の既定 (c_16 §9 ``memory.evidence.lexical``)。
LEXICAL_DEFAULTS: dict[str, Any] = {
    "q_terms": 32,
    "m_postings": 2000,
    "max_df_ratio": 0.10,
    "budget_ms": 20.0,
}

#: :meth:`EvidenceStore.search` が各チャネルから引く候補数の倍率。
#: アクティブマスクと ``claim_key`` 畳み込みが top-k の **前** に候補を削るため、
#: 素の top-k だけ引くと最終件数が top-k に届かない。
CANDIDATE_MULTIPLIER = 4

#: 候補数の下限 (top_k が極端に小さいときに刈りすぎない)。
MIN_CANDIDATES = 20

#: ``memory.evidence.ranking.store_prior`` の店名 → 設定キー (c_16 §7.2)。
#: semantic は 1 ストアに 2 つの規則 (``mem.*`` / ``know.*``) が同居するため、
#: ここでは ``semantic_mem`` を既定にし、行ごとの上書きは
#: :meth:`SemanticStore.store_prior_per_row` が渡す。
STORE_PRIOR_KEYS: dict[str, str] = {
    "episodic": "episodic",
    "semantic": "semantic_mem",
    "corpus": "corpus",
}

#: ``store_prior`` が読めないときの既定 (c_16 §9)。
DEFAULT_STORE_PRIORS: dict[str, float] = {
    "episodic": 0.9, "semantic": 1.0, "corpus": 1.0,
}

#: 埋め込み索引の版ディレクトリ名 (``embeddings/<model_id>/v0003/``、c_16 §6.1)。
_EMBED_VERSION_FORMAT = "v{:04d}"

#: 埋め込み索引の版ディレクトリを読む正規表現。
_EMBED_VERSION_RE = re.compile(r"v(\d+)")

#: 埋め込み索引を何版残すか (snapshot と同じ理由 — 稼働中の読み手が memmap を
#: 掴んでいる版を消さない)。
EMBEDDING_VERSIONS_KEEP = 2


class EvidenceStoreReadonlyError(RuntimeError):
    """ストアが readonly (書けば壊す状態) なのに書き込みが試みられた。

    c_05 §0.5.1 の「新しい版のファイルは読まず書き戻しも拒否」を、
    ``manifest.json`` が読めない / 版が新しい / レコードの ``_version`` が
    新しい場合へ広げたもの (c_16 §5.1)。既定値のまま「空のストア」として
    動き出すと、``create_snapshot`` が v0001 を上書きし、``prune_snapshots``
    が本物の最新版を消す。
    """


class EvidenceIdExistsError(ValueError):
    """``create`` の id が既に在る (追加と全置換を区別する、c_16 §5.2)。"""


def embedding_version_name(seq: int) -> str:
    """埋め込み索引の版番号 → ディレクトリ名 (``v0003``)。"""
    return _EMBED_VERSION_FORMAT.format(int(seq))


def embedding_version_seq(name: str) -> int | None:
    """埋め込み索引のディレクトリ名 → 版番号。読めなければ ``None``。"""
    match = _EMBED_VERSION_RE.fullmatch(str(name).strip())
    return int(match.group(1)) if match else None


class UsageBuffer:
    """チャット経路が「使った」id を溜めるプロセス内バッファ (c_16 §2.1)。

    ``last_used_at`` をチャット中にディスクへ書くと、応答パスが書き手に
    なってしまう (f_02 §5 の不変則)。ここに溜めて sleep-time が
    :meth:`EvidenceStore.flush_touch` で **1 事象** に畳んで書く。

    複数のリクエストハンドラから同時に呼ばれるのでスレッド安全。
    """

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._lock = threading.Lock()

    def add(self, record_id: str) -> None:
        if not record_id:
            return
        with self._lock:
            self._ids.add(record_id)

    def add_many(self, record_ids: Sequence[str]) -> None:
        ids = {rid for rid in record_ids if rid}
        if not ids:
            return
        with self._lock:
            self._ids |= ids

    def drain(self) -> list[str]:
        """溜まった id を取り出してバッファを空にする。"""
        with self._lock:
            ids = sorted(self._ids)
            self._ids.clear()
        return ids

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)



def embed_side_of(attrs: Any) -> tuple[bool, str]:
    """``attrs`` からそのレコードの埋め込み側を読む (c_16 §6.1)。

    Returns:
        ``(is_query, mode)``。既定は document 側 (``(False, "chat")``)。
        document 側では ``mode`` に意味が無いので既定へ寄せる — 寄せないと
        同じ document 側の行が ``mode`` 違いで別バッチに割れる。
    """
    if not isinstance(attrs, dict):
        return False, DEFAULT_EMBED_MODE
    if not bool(attrs.get(EMBED_AS_QUERY_ATTR, False)):
        return False, DEFAULT_EMBED_MODE
    mode = attrs.get(EMBED_MODE_ATTR)
    return True, str(mode) if isinstance(mode, str) and mode else DEFAULT_EMBED_MODE


def embed_side_key(is_query: bool, mode: str) -> str:
    """埋め込み側を再利用鍵・metadata に載せる短い文字列にする。"""
    return f"q:{mode}" if is_query else "d"


def embed_reuse_key(text: str, is_query: bool, mode: str) -> str:
    """増分埋め込みの再利用鍵 (``text_hash``)。

    **側を鍵に含める** — 含めないと ``embed_as_query`` を後から立てても本文が
    同じである限り旧い側のベクトルが流用され続け、書く側と読む側の勘定が
    ずれたまま固定される (2026-09-02 監査 M19 と同じ形)。
    """
    return content_hash(f"{embed_side_key(is_query, mode)}\x00{text}")


def _backend_value(backend: object, name: str, default: object) -> object:
    """EmbeddingBackend の ``model_name`` / ``dim`` / ``backend_type`` を読む。

    Protocol (``embedding_backend.py``) ではメソッド、テスト用の簡易実装では
    属性のことがあるので、callable なら呼んで値にそろえる。
    """
    value = getattr(backend, name, default)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return default
    return value if value not in (None, "") else default

class EvidenceStore:
    """1 ストア (episodic / semantic / corpus) の永続基盤。

    Args:
        store_dir: ストアのルート (``<data_root>/g1/store/memory/<store>``)。
        store_name: ``episodic`` / ``semantic`` / ``corpus``。
        embedding_backend: 埋め込みバックエンド。``None`` なら snapshot 生成
            時にベクトル索引を作らない (縮退動作。検索は語彙索引のみになる)。
        rag_config: ``config.yaml`` の ``rag`` セクション (``quantization`` /
            ``memmap_threshold`` / ``cluster_index`` を読む)。``lexical`` 下位
            セクション (``q_terms`` / ``m_postings`` / ``max_df_ratio`` /
            ``budget_ms``、c_16 §9) があれば語彙索引に使う。``None`` で既定値。
        by: 事象ログの ``by`` (書き手コンポーネント名)。
        shard_key_for: レコード → 転置索引のシャード名 (c_16 §6.2)。既定は
            全件 1 シャード (:data:`DEFAULT_SHARD`)。episodic は ``tier × 月``、
            semantic は namespace、corpus はパッケージを返す関数を渡す。
        gc_filter: 物理 GC の判定 (c_16 §5.4)。``False`` を返したレコードは
            **新しい版に書かれない** = ディスクから消える。``None`` (既定) なら
            何も落とさない。呼ばれるのは :meth:`create_snapshot` の畳み込み後
            だけで、``put`` / ``patch`` のオーバーレイ経路には掛からない。
            レコードを見て決めるので版の組み立てが子プロセスへ出せなくなる —
            id で決まるなら ``gc_drop_ids`` を使う。
        gc_drop_ids: 物理 GC で落とす id 集合を返す関数。版の生成の直前に
            ループ上で 1 回呼び、集合を組み立て側 (子プロセス) へ渡す。
    """

    def __init__(
        self,
        store_dir: Path | str,
        store_name: str = "episodic",
        embedding_backend: "EmbeddingBackend | None" = None,
        rag_config: Any = None,
        *,
        by: str = "sleep_time",
        shard_key_for: Callable[[Evidence], str] | None = None,
        gc_filter: Callable[[Evidence], bool] | None = None,
        gc_drop_ids: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.store_name = store_name
        self.embedding_backend = embedding_backend
        self.rag_config = rag_config
        self.manifest = EvidenceManifest(self.store_dir, store_name)
        self.events = EvidenceEventLog(self.store_dir / "events", by=by)
        self.usage = UsageBuffer()
        self.shard_key_for: Callable[[Evidence], str] = (
            shard_key_for if shard_key_for is not None else _single_shard
        )
        self.gc_filter: Callable[[Evidence], bool] | None = gc_filter
        self.gc_drop_ids: Callable[[], Iterable[str]] | None = gc_drop_ids

        self._snapshot: SnapshotReader | None = None
        #: 事象で作られた / 上書きされたレコード (snapshot より優先)。
        self._overlay: dict[str, Evidence] = {}
        #: 版の組み立て (``await``) の間に適用した事象。差し替え後のオーバーレイへ
        #: 戻す (組み立て中でなければ ``None``)。
        self._events_during_build: list[dict[str, Any]] | None = None
        #: snapshot に無い = 次の snapshot まで **ベクトル検索に出ない** id。
        self._tail_ids: list[str] = []
        #: tail 索引 (c_16 §6.4): tail の id → (再利用鍵, 正規化済みベクトル)。
        #: derived でメモリだけに持つ — 再起動後は最初の :meth:`refresh_tail` が
        #: 埋め直す。snapshot 索引と違い、稼働中に書き換えてよいのはここだけ。
        self._tail_vectors: dict[str, tuple[str, np.ndarray]] = {}
        self._tail_model_id: str = ""
        self._tail_refreshing = False
        #: サイドカーをまだ読んでいない。埋め込みバックエンドは ``load`` の後に
        #: 付く (``_memory_init.attach_*_embedder``) ので、モデルを照合できる
        #: 最初の tail の利用時に読む。
        self._tail_sidecar_pending = False
        self._vector_store: EvidenceVectorStore | None = None
        #: snapshot の行 → VectorStore の行 (無い行は -1)。
        self._vector_rows: np.ndarray | None = None
        #: 埋め込み版の刻印が active snapshot と合わなかった (c_16 §5.6)。
        #: 検索には使わず、次の :meth:`refresh_tail` / 版の生成で作り直す。
        self._embedding_stale = False
        #: 埋め込みモデルの変更を利用者が確認した (:attr:`embedding_model_pending`)。
        self._reembed_confirmed = False
        #: active snapshot の転置索引 (lazy load。配列はシャード単位で読む)。
        self._lexical: LexicalPack | None = None
        #: :meth:`create_snapshot` の排他。版番号の発番 (``take_next_version``)
        #: と overlay / tail のリセットを 2 本が交互に踏むと、書きかけの版を
        #: 別内容で上書きしたり prune の保護対象が食い違ったりする。
        self._snapshot_lock = asyncio.Lock()
        #: 同期の書き込み API の再入検出。単一書き手 (sleep-time) という
        #: 不変則 (c_16 §2.1) が破れたときに **沈黙の破損ではなく例外** に
        #: するための番人で、直列化のためのロックではない (非ブロッキング)。
        self._writer_guard = threading.Lock()
        #: readonly に落ちた理由 (``None`` なら書ける)。
        self._readonly_reason: str | None = None
        #: readonly の警告を 1 度だけ出すためのフラグ。
        self._readonly_warned: bool = False
        #: 語彙索引パラメータの食い違い警告を 1 度だけ出すためのフラグ。
        self._lexical_drift_warned: bool = False

    # ── readonly (c_05 §0.5.1 / c_16 §5.1) ──

    @property
    def readonly(self) -> bool:
        """書き込みを拒否する状態か。"""
        return self._readonly_reason is not None

    @property
    def readonly_reason(self) -> str:
        """readonly に落ちた理由 (書けるなら空文字)。"""
        return self._readonly_reason or ""

    def _enter_readonly(self, reason: str) -> None:
        """ストアを readonly に落とす (WARNING は 1 度だけ)。"""
        if self._readonly_reason is not None:
            return
        self._readonly_reason = reason
        # manifest 側も封じる — 旧いコードで書き戻すとフィールドが落ちる。
        self.manifest.readonly = True
        format_health.report(
            EVIDENCE_RECORD_FORMAT.format_id, str(self.store_dir), "readonly",
            f"{self.store_name}: {reason}",
        )
        logger.warning(
            "Evidence store %s is read-only: %s. Writes (put / patch / "
            "retract / create_snapshot / prune) are refused.",
            self.store_name, reason,
        )

    def _refuse_write(self, operation: str) -> None:
        """書き込み系 API の入口。readonly なら例外で止める。"""
        if self._readonly_reason is None:
            return
        raise EvidenceStoreReadonlyError(
            f"{self.store_name} store is read-only ({self._readonly_reason}); "
            f"refusing {operation}",
        )

    @contextlib.contextmanager
    def _exclusive_write(self, operation: str) -> Iterator[None]:
        """同期の書き込み API を再入検出付きで囲む。

        3 ストアの書き手は sleep-time 1 本 (c_16 §2.1) という不変則が破れると、
        事象の追記とオーバーレイ更新の間に別の書き手が入り、``manifest`` の
        計数と in-memory の状態が食い違う。ここでは **直列化しない** —
        取れなければ即座に例外にして、沈黙の破損ではなく落ちるようにする。

        Raises:
            RuntimeError: 別の書き手が同じストアを書いている最中のとき。
        """
        if not self._writer_guard.acquire(blocking=False):
            raise RuntimeError(
                f"concurrent writer on the {self.store_name} evidence store "
                f"({operation}); writes are single-writer (sleep-time only)",
            )
        try:
            yield
        finally:
            self._writer_guard.release()

    def _warn_readonly_skip(self, operation: str) -> None:
        """壊れない no-op (touch 系) を飛ばしたことを 1 度だけ記録する。"""
        if self._readonly_warned:
            return
        self._readonly_warned = True
        logger.warning(
            "Skipping %s on the read-only %s store (%s)",
            operation, self.store_name, self._readonly_reason,
        )

    # ── ロード ──

    def load(self) -> None:
        """manifest → active snapshot → 未畳み込み事象の順に読む。

        manifest が読めなかった場合は **既定値のまま動き出さない** —
        ``active_snapshot=""`` / ``next_snapshot_seq=1`` のまま進むと、空の
        ストアとして起動したうえで次の ``create_snapshot`` が v0001 を
        上書きし、``prune_snapshots`` が本物の最新版を消す。版ディレクトリ
        から復元できるならそこから復元し、できなければ readonly に落とす
        (c_05 §0.5.1)。
        """
        self._readonly_reason = None
        self._readonly_warned = False
        format_health.clear(str(self.store_dir))
        # 前回のロードで立った封印は持ち越さない (直したファイルを読み直せる)。
        self.manifest.readonly = False
        manifest_existed = self.manifest.path.exists()
        if not self.manifest.load():
            self._recover_manifest(manifest_existed)
        elif self.manifest.record_version > RECORD_VERSION:
            self._enter_readonly(
                f"the manifest declares record_version "
                f"{self.manifest.record_version}, newer than the supported "
                f"{RECORD_VERSION}",
            )
        self._apply_config_retention()
        version = self.manifest.active_snapshot
        self._snapshot = (
            read_snapshot(self.store_dir, version) if version else None
        )
        if self._snapshot is not None and not self._snapshot.valid:
            if self._snapshot.complete_status in READONLY_STATUSES:
                self._enter_readonly(
                    f"snapshot {version} has a {COMPLETE_FILE} this version cannot read "
                    f"({self._snapshot.complete_status})",
                )
            else:
                self._fall_back_from(version)
        self._pad_event_log()
        self._overlay = {}
        self._tail_ids = []
        self._replay_events()
        self._tail_vectors = {}
        self._tail_sidecar_pending = True
        self._vector_store = None
        self._vector_rows = None
        self._embedding_stale = False
        self._lexical = None

    def _recover_manifest(self, manifest_existed: bool) -> None:
        """manifest が読めなかったときに版ディレクトリから状態を復元する (c_16 §5.1)。

        有効な版は ``COMPLETE`` のある版だけ (c_16 §5.6)。

        - 新しい版 / G1 の封筒でない (``VersionedJsonFile`` が拒否した) → readonly。
          読むだけは続けられるよう active 版は最新の有効な版を指す
        - 破損 / 消失 → 最新の有効な版を active、``next_snapshot_seq`` は版名の
          最大 +1、``folded_through`` は **その版の COMPLETE の値** (事象ログの先頭へ
          戻すと、GC / forget で版から落とした行が事象の replay で復活する)。
          COMPLETE の無い版は trash へ送る
        - 版が 1 つも無く manifest も無い → 新規ストア (通常起動)
        - 有効な版が無い → readonly
        """
        versions = list_versions(self.store_dir)
        incomplete = list_incomplete_versions(self.store_dir)
        if self.manifest.readonly:
            # 先に readonly へ落とす (版の選択中に無効な版を trash へ送らない)
            self._enter_readonly(
                f"{self.manifest.path.name} cannot be written back "
                f"({self.manifest.last_status})",
            )
            chosen = self._newest_valid_version(set())
            if chosen is not None:
                self._adopt_version(*chosen)
            return
        if not manifest_existed and not versions and not incomplete:
            return  # 新規ストア
        chosen = self._newest_valid_version(set())
        if chosen is None:
            self._enter_readonly(
                f"{self.manifest.path.name} is unreadable and there is no "
                "complete snapshot version to recover from",
            )
            return
        self._adopt_version(*chosen)
        self._bump_next_snapshot_seq()
        self._trash_versions(incomplete)
        logger.warning(
            "Recovered the %s manifest from snapshot versions: active=%s, "
            "next_snapshot_seq=%d, folded through %s:%d",
            self.store_name, self.manifest.active_snapshot,
            self.manifest.next_snapshot_seq,
            self.manifest.folded_through.month or "-", self.manifest.folded_through.line,
        )

    def _fall_back_from(self, invalid: str) -> None:
        """active 版が無効 (COMPLETE が無い / 行数が合わない) なら前の有効な版へ落ちる。

        回復と同じく、落ちた先の COMPLETE の ``folded_through`` から事象を畳み直す。
        落ちる先が無ければ readonly (読むだけは無効な版のまま続ける)。
        """
        chosen = self._newest_valid_version({invalid})
        if chosen is None:
            self._enter_readonly(
                f"snapshot {invalid} is not a complete version and there is no "
                "complete version to fall back to",
            )
            return
        self._adopt_version(*chosen)
        self._bump_next_snapshot_seq()
        self._trash_versions([invalid])
        logger.warning(
            "Snapshot %s of %s is not complete; fell back to %s (folded through %s:%d)",
            invalid, self.store_name, self.manifest.active_snapshot,
            self.manifest.folded_through.month or "-", self.manifest.folded_through.line,
        )

    def _newest_valid_version(
        self, skip: set[str],
    ) -> tuple[str, SnapshotReader] | None:
        """COMPLETE があり行数も合う最新の版 (``skip`` を除く)。

        途中で見つけた無効な版 (COMPLETE の行数と offsets が合わない) は、有効な版が
        見つかったときだけ trash へ送る。
        """
        invalid: list[str] = []
        for version in reversed(list_versions(self.store_dir)):
            if version in skip:
                continue
            reader = read_snapshot(self.store_dir, version)
            if reader.valid:
                self._trash_versions(invalid)
                return version, reader
            if reader.complete_status in READONLY_STATUSES:
                # 新しい版 / G1 でない印の版は trash へ送らない (読めないが壊さない)
                self._enter_readonly(
                    f"snapshot {version} has a {COMPLETE_FILE} this version cannot read "
                    f"({reader.complete_status})",
                )
                continue
            invalid.append(version)
        return None

    def _adopt_version(self, version: str, reader: SnapshotReader) -> None:
        """有効な版を active にし、畳み込み位置と埋め込み版をその COMPLETE から取る。"""
        self.manifest.active_snapshot = version
        self.manifest.folded_through = reader.folded_through
        embedding = reader.complete.embedding_version if reader.complete is not None else None
        if embedding is not None and embedding.version:
            self.manifest.embedding_version = embedding.version
            self.manifest.embedding_model_id = embedding.model_id
        self._snapshot = reader

    def _bump_next_snapshot_seq(self) -> None:
        """発番済みの版名 (無効な版を含む) を再利用しないよう番号を進める。"""
        seqs = [
            seq for seq in (
                version_seq(v) for v in
                list_versions(self.store_dir) + list_incomplete_versions(self.store_dir)
            )
            if seq is not None
        ]
        if seqs:
            self.manifest.next_snapshot_seq = max(
                self.manifest.next_snapshot_seq, max(seqs) + 1,
            )

    def _trash_versions(self, versions: Iterable[str]) -> None:
        """無効な版を trash へ改名する (消さない。readonly 中は触らない)。"""
        if self.readonly or is_readonly():
            return
        for version in versions:
            trash_version(self.store_dir, version)

    def _pad_event_log(self) -> None:
        """``folded_through`` が月ファイルの実際の行数を超えていたら空行で埋める。"""
        if self.readonly or is_readonly():
            return
        try:
            self.events.pad_to(self.manifest.folded_through)
        except OSError as e:
            logger.warning("Failed to pad the %s event log: %s", self.store_name, e)

    def _apply_config_retention(self) -> None:
        """``memory.evidence.retention`` を manifest の保持方針へ重ねる。

        保持方針は manifest に宣言するのが c_16 §5.4 だが、**設定を変えても
        効く先が無い** と config のキーが飾りになる (corpus で実際にそう
        なっていた)。設定側を後勝ちにし、次の ``save`` で manifest へも降ろす。
        """
        retention = _section_value(self.rag_config, "retention")
        if not isinstance(retention, dict) or not retention:
            return
        changed = {
            key: value for key, value in retention.items()
            if self.manifest.retention.get(key) != value
        }
        if not changed:
            return
        self.manifest.retention.update(retention)
        logger.info(
            "Retention for %s overridden by config: %s",
            self.store_name,
            ", ".join(f"{k}={v}" for k, v in sorted(changed.items())),
        )

    def _replay_events(self) -> None:
        """snapshot 以降の事象をオーバーレイへ適用する。"""
        applied = 0
        try:
            for event in self.events.iter_since(self.manifest.folded_through):
                self._apply_event(event)
                applied += 1
        except EvidenceVersionError as e:
            self._enter_readonly(f"the event log holds a newer record version: {e}")
        if applied:
            logger.info(
                "Replayed %d event(s) on top of snapshot %s (%s)",
                applied, self.manifest.active_snapshot or "(none)", self.store_name,
            )
        self.manifest.events_since_snapshot = applied
        snapshot = self._snapshot
        if snapshot is not None and snapshot.unsupported_version:
            self._enter_readonly(
                f"snapshot {self.manifest.active_snapshot} holds a newer "
                "record version",
            )

    def _apply_event(self, event: dict[str, Any]) -> None:
        """1 事象をオーバーレイへ適用する (畳み込みと同じ意味論)。"""
        if self._events_during_build is not None:
            self._events_during_build.append(event)
        folded = SnapshotWriter.fold(self._existing_for(event), [event])
        for record in folded:
            known = self._snapshot is not None and self._snapshot.row_of(record.id) is not None
            if not known and record.id not in self._overlay:
                self._tail_ids.append(record.id)
            self._overlay[record.id] = record

    def _existing_for(self, event: dict[str, Any]) -> list[Evidence]:
        """事象が参照するレコードの現在値だけを集める (全件展開しない)。"""
        payload = event.get("payload") or {}
        ids = [str(event.get("id") or "")]
        if event.get("op") == "touch":
            ids = [str(v) for v in (payload.get("ids") or [])]
        records: list[Evidence] = []
        for record_id in ids:
            if not record_id:
                continue
            current = self.get(record_id)
            if current is not None:
                records.append(current)
        return records

    # ── 読み出し ──

    def pending_ids(self) -> list[str]:
        """active snapshot 以後に ``put`` された (次の snapshot までベクトル検索外の) id。"""
        return list(self._tail_ids)

    def iter_pending(self) -> Iterator[Evidence]:
        """snapshot 以後に追加・更新されたレコード (オーバーレイ) を列挙する。"""
        return iter(list(self._overlay.values()))

    def get(self, record_id: str) -> Evidence | None:
        """id でレコードを引く (オーバーレイ優先)。"""
        record = self._overlay.get(record_id)
        if record is not None:
            return record
        if self._snapshot is None:
            return None
        return self._snapshot.get(record_id)

    def __len__(self) -> int:
        base = len(self._snapshot) if self._snapshot is not None else 0
        return base + len(self._tail_ids)

    def iter_records(self) -> Iterator[Evidence]:
        """snapshot + オーバーレイの全レコード (畳み込み済みの現在状態)。"""
        snapshot = self._snapshot
        if snapshot is not None:
            rows = len(snapshot)
            for row, raw in snapshot.iter_raw():
                if row >= rows:
                    break
                overlaid = self._overlay.get(snapshot.id_at(row))
                if overlaid is not None:
                    yield overlaid
                    continue
                record = snapshot.record_from_raw(raw, row)
                if record is not None:
                    yield record
        for record_id in self._tail_ids:
            record = self._overlay.get(record_id)
            if record is not None:
                yield record

    def snapshot_age_seconds(self) -> float:
        """active 版を作ってからの秒数 (版が無ければ ``inf``)。

        版ディレクトリは書いたら変えない (不変則 #11) ので、``records.jsonl`` の
        更新時刻がそのまま版を作った時刻になる。manifest に時刻の欄を足すと
        永続形式が変わるので、ここではファイルの時刻を読む。
        """
        snapshot = self._snapshot
        if snapshot is None:
            return float("inf")
        try:
            written = snapshot.records_path.stat().st_mtime
        except OSError:
            return float("inf")
        return max(0.0, utc_now_dt().timestamp() - written)

    # ── tail 索引 (c_16 §6.4) ──

    async def refresh_tail(self) -> int:
        """版に入っていない行 (tail) のうち、まだ埋め込んでいないものを埋め込む。

        応答の後と Full の後に呼ぶ (``SleepTimeWorker.refresh_tail``)。次の
        snapshot を待たずに新しい記憶が検索に出る — G0 は想起欠落が p50 7.6〜10
        分あった。埋め込むのは本文か埋め込み側が変わった行だけ (再利用鍵が同じ
        行は埋め直さない)。版に畳まれて tail から外れた行は捨てる。

        Returns:
            新しく埋め込んだ行数。
        """
        backend = self.embedding_backend
        if backend is None or self._tail_refreshing:
            return 0
        self._tail_refreshing = True
        try:
            if self._embedding_stale:
                await self._rebuild_stale_embeddings()
            self._ensure_tail_sidecar()
            model_id = self.embedding_model_id
            if model_id != self._tail_model_id:
                self._tail_vectors = {}
                self._tail_model_id = model_id
            self._prune_tail_vectors()
            todo: list[tuple[str, str, str, tuple[bool, str]]] = []
            for record_id in self._tail_ids:
                record = self._overlay.get(record_id)
                if record is None:
                    continue
                side = embed_side_of(record.attrs)
                text = str(record.text or "")
                key = embed_reuse_key(text, *side)
                held = self._tail_vectors.get(record_id)
                if held is not None and held[0] == key:
                    continue
                todo.append((record_id, key, text, side))
            if not todo:
                return 0
            with embed_priority(P1_FRESHNESS, override=False):
                vectors = np.asarray(
                    await self._embed_all([(text, *side) for _, _, text, side in todo]),
                    dtype=np.float32,
                )
            # 埋め込みの await の間に版が差し替わると、畳まれた行は tail から外れる
            live = set(self._tail_ids)
            added = 0
            for (record_id, key, _text, _side), vector in zip(todo, vectors, strict=True):
                if record_id not in live:
                    continue
                norm = float(np.linalg.norm(vector))
                self._tail_vectors[record_id] = (key, vector / norm if norm else vector)
                added += 1
            if added:
                self._prune_tail_vectors()
                await self._save_tail_sidecar()
            return added
        finally:
            self._tail_refreshing = False

    async def _rebuild_stale_embeddings(self) -> None:
        """刻印が active snapshot と合わない埋め込み版を作り直す (c_16 §5.6)。

        manifest の回復で前の版へ落ちた直後など、指している埋め込み版が別の
        snapshot のものだった場合。流用は id と再利用鍵で決まるので、合わない版の
        行もそのまま流用できる (埋め込むのは本文か側が変わった行だけ)。
        """
        if self.readonly or is_readonly() or self._snapshot is None:
            return
        logger.info(
            "Rebuilding the embedding index of %s for snapshot %s",
            self.store_name, self._snapshot.directory.name,
        )
        await self.embed_and_index_snapshot()

    def _tail_sidecar_dir(self) -> Path:
        return self.store_dir / TAIL_DIR

    async def _save_tail_sidecar(self) -> None:
        """tail のベクトルを ``tail/`` へ書く (再起動後に埋め直さずに戻すため)。

        索引 (id と再利用鍵) を後に書く — 読み手は索引の件数とベクトルの行数が
        合わなければ全部捨てる。書き込みはループの外で (512 行で約 2MB)。
        """
        if not self._tail_vectors:
            return
        ids = list(self._tail_vectors)
        keys = [self._tail_vectors[record_id][0] for record_id in ids]
        matrix = np.stack([self._tail_vectors[record_id][1] for record_id in ids]).astype(
            np.float32, copy=False,
        )
        directory = self._tail_sidecar_dir()
        index = self._tail_index_file(directory)
        index.payload = {
            "model_id": self._tail_model_id,
            "ids": ids,
            "keys": keys,
            "dim": int(matrix.shape[1]),
        }

        def write() -> None:
            directory.mkdir(parents=True, exist_ok=True)
            with AtomicWriter(directory / TAIL_VECTORS_FILE, mode="wb") as f:
                np.save(f, matrix)
            index.save()

        try:
            await run_in_executor_with_context(asyncio.get_running_loop(), None, write)
        except OSError as e:
            logger.warning("Failed to save the tail sidecar for %s: %s", self.store_name, e)

    def _ensure_tail_sidecar(self) -> None:
        """まだならサイドカーを読む (埋め込みバックエンドが付いてから 1 回だけ)。"""
        if self._tail_sidecar_pending and self.embedding_backend is not None:
            self._tail_sidecar_pending = False
            self._load_tail_sidecar()

    def _tail_index_file(self, directory: Path) -> VersionedPayloadFile:
        return VersionedPayloadFile(
            TAIL_INDEX_FORMAT, directory / TAIL_INDEX_FILE,
            component=f"{type(self).__name__}:{self.store_name}", state_logger=logger,
        )

    def _load_tail_sidecar(self) -> None:
        """``tail/`` から tail のベクトルを戻す (読めなければ捨てて埋め直しに任せる)。

        戻すのは今も tail にある行で、同じ埋め込みモデルのものだけ。本文が
        変わった行は次の :meth:`refresh_tail` が再利用鍵で見分けて埋め直す。
        """
        self._tail_vectors = {}
        directory = self._tail_sidecar_dir()
        if not self._tail_ids:
            return
        index = self._tail_index_file(directory)
        # 読めない / 別形式 (G0 の封筒を含む) / 新しい版はここで捨てられる (derived)
        if not index.load():
            return
        try:
            payload = index.payload
            if str(payload.get("model_id") or "") != self.embedding_model_id:
                return
            ids = [str(v) for v in payload.get("ids") or []]
            keys = [str(v) for v in payload.get("keys") or []]
            matrix = np.load(str(directory / TAIL_VECTORS_FILE), allow_pickle=False)
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Ignoring an unreadable tail sidecar for %s: %s", self.store_name, e)
            return
        if matrix.ndim != 2 or len(ids) != len(keys) or matrix.shape[0] != len(ids):
            logger.warning("Ignoring a misaligned tail sidecar for %s", self.store_name)
            return
        live = set(self._tail_ids)
        for record_id, key, vector in zip(ids, keys, matrix, strict=True):
            if record_id in live:
                self._tail_vectors[record_id] = (key, np.asarray(vector, dtype=np.float32))
        self._tail_model_id = self.embedding_model_id
        if self._tail_vectors:
            logger.info(
                "Restored %d tail vector(s) for %s from the sidecar",
                len(self._tail_vectors), self.store_name,
            )

    def _prune_tail_vectors(self) -> None:
        """tail から外れた (版に畳まれた) 行のベクトルを捨てる。"""
        if not self._tail_vectors:
            return
        live = set(self._tail_ids)
        for record_id in [rid for rid in self._tail_vectors if rid not in live]:
            del self._tail_vectors[record_id]

    def _tail_candidates(self) -> tuple[list[Evidence], np.ndarray]:
        """埋め込み済みの tail 行 (現在のレコードと正規化ベクトル)。"""
        self._ensure_tail_sidecar()
        if not self._tail_vectors:
            return [], np.zeros((0, 0), dtype=np.float32)
        records: list[Evidence] = []
        vectors: list[np.ndarray] = []
        for record_id in self._tail_ids:
            held = self._tail_vectors.get(record_id)
            record = self._overlay.get(record_id)
            if held is None or record is None:
                continue
            records.append(record)
            vectors.append(held[1])
        if not records:
            return [], np.zeros((0, 0), dtype=np.float32)
        return records, np.stack(vectors)

    @staticmethod
    def _unit(query_vec: np.ndarray) -> np.ndarray | None:
        query = np.asarray(query_vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        return query / norm if norm else None

    def tail_cosines(self, query_vec: np.ndarray) -> dict[str, float]:
        """埋め込み済みの tail 行の **素の cosine** (``id → cosine``、ゲート・順位なし)。"""
        records, matrix = self._tail_candidates()
        query = self._unit(query_vec) if records else None
        if query is None or matrix.shape[1] != query.shape[0]:
            return {}
        cosines = matrix @ query
        return {record.id: float(cos) for record, cos in zip(records, cosines, strict=True)}

    def search_tail(
        self,
        query_vec: np.ndarray,
        *,
        threshold: float = 0.0,
        now: float | None = None,
        include_private: bool = False,
        store_prior: float | Callable[[Evidence], float] | None = None,
    ) -> list[tuple[Evidence, float, float]]:
        """tail 行を総当たりで引く (c_16 §6.4)。

        :meth:`search` と同じ手順 — アクティブな行だけ、ゲートは素の cosine、順位は
        ``cos × freshness × confidence × store_prior``、``claim_key`` 畳み込み —
        を tail 行に掛ける。呼出側は :meth:`search` の結果と ``score`` で混ぜる。

        Args:
            store_prior: スカラ、レコードごとの値を返す関数、または ``None``
                (:attr:`default_store_prior`)。

        Returns:
            ``(レコード, cosine, score)`` をスコア降順で。
        """
        records, matrix = self._tail_candidates()
        query = self._unit(query_vec) if records else None
        if query is None or matrix.shape[1] != query.shape[0]:
            return []
        now_epoch = utc_now_dt().timestamp() if now is None else float(now)
        active = np.fromiter(
            (is_active(record, now_epoch, include_private) for record in records),
            dtype=bool, count=len(records),
        )
        cosines = (matrix @ query).astype(np.float64)
        keep = gate_by_cosine(cosines, threshold) & active
        if not np.any(keep):
            return []
        kept = [record for record, ok in zip(records, keep, strict=True) if ok]
        kept_cos = cosines[keep]
        columns = RankColumns.from_columns(build_columns(kept))
        rows = np.arange(len(kept), dtype=np.int64)
        if store_prior is None:
            prior: float | np.ndarray = self.default_store_prior
        elif callable(store_prior):
            prior = np.array([float(store_prior(record)) for record in kept], dtype=np.float64)
        else:
            prior = float(store_prior)
        scores = score_rows(kept_cos, rows, columns, now_epoch, prior)
        final_rows, final_scores = collapse(
            rows, scores, columns,
            allow_assistant_origin=self.allow_assistant_origin_injection,
        )
        out = [
            (kept[int(row)], float(kept_cos[int(row)]), float(score))
            for row, score in zip(final_rows, final_scores, strict=True)
        ]
        out.sort(key=lambda hit: hit[2], reverse=True)
        return out

    def iter_active(
        self, now: float | None = None, include_private: bool = False,
    ) -> Iterator[Evidence]:
        """注入候補になりうるレコードだけを返す (c_16 §7.3-1)。"""
        now_epoch = utc_now_dt().timestamp() if now is None else now
        for record in self.iter_records():
            if is_active(record, now_epoch, include_private):
                yield record

    def active_mask(
        self, now: float | None = None, include_private: bool = False,
    ) -> np.ndarray:
        """snapshot 行のアクティブマスク (オーバーレイの変更を反映済み)。

        カラム由来の bool 配列に対し、**事象で変わった行だけ** を上書きする
        (オーバーレイは高々数百件なのでループしてよい。N 件側は numpy)。
        """
        if self._snapshot is None:
            return np.zeros(0, dtype=bool)
        now_epoch = utc_now_dt().timestamp() if now is None else now
        mask = columns_active_mask(self._snapshot.columns, now_epoch, include_private)
        for record_id, record in self._overlay.items():
            row = self._snapshot.row_of(record_id)
            if row is None:
                continue
            mask[row] = is_active(record, now_epoch, include_private)
        return mask

    @property
    def snapshot(self) -> SnapshotReader | None:
        """現在の active snapshot (未生成なら ``None``)。"""
        return self._snapshot

    def text_at(self, row: int) -> str:
        """snapshot 行の本文を lazy に読む。"""
        return "" if self._snapshot is None else self._snapshot.text_at(row)

    # ── 書き込み API (sleep-time 専用。事象を追記するだけ) ──

    def create(self, record: Evidence, *, by: str | None = None) -> Evidence:
        """新しい id のレコードを ``create`` する (c_16 §5.2 の G1 の事象)。

        既存の id (retract 済み・未知の列挙値で使っていない行を含む) なら拒否する —
        既存レコードの更新は :meth:`patch`、明示の全置換は :meth:`put`。

        Raises:
            EvidenceIdExistsError: 同じ id のレコードが既に在る。
            EvidenceStoreReadonlyError: ストアが readonly のとき (c_05 §0.5.1)。
            RuntimeError: 別の書き手が同時に書いているとき。
        """
        self._refuse_write(f"create({record.id})")
        check_writable(record)
        if self.get(record.id) is not None:
            raise EvidenceIdExistsError(f"evidence id already exists: {record.id}")
        with self._exclusive_write(f"create({record.id})"):
            event = self.events.append_create(record.to_record(), by=by)
            self._apply_event(event)
            self.manifest.events_since_snapshot += 1
            return self._overlay.get(record.id, record)

    def put(self, record: Evidence, *, by: str | None = None) -> Evidence:
        """レコードを ``put`` する (追加 / 明示の全置換)。

        既存レコードの一部の更新には使わない (:meth:`patch`)。新しい id の追加は
        :meth:`create` (既存の id を黙って置き換えない)。

        Raises:
            EvidenceStoreReadonlyError: ストアが readonly のとき (c_05 §0.5.1)。
            RuntimeError: 別の書き手が同時に書いているとき。
        """
        self._refuse_write(f"put({record.id})")
        check_writable(record)
        with self._exclusive_write(f"put({record.id})"):
            event = self.events.append_put(record.to_record(), by=by)
            self._apply_event(event)
            self.manifest.events_since_snapshot += 1
            return self._overlay.get(record.id, record)

    def patch(
        self,
        record_id: str,
        *,
        by: str | None = None,
        unset: Sequence[str] = (),
        **fields: Any,
    ) -> Evidence | None:
        """変更フィールドだけを ``patch`` する (c_16 §5.2)。

        値を消すときは null ではなく ``unset`` (``"valid_until"`` / ``"attrs.<key>"``)。
        適用後のレコードを書き手の検査 (:func:`check_writable`) に通してから追記する。
        """
        self._refuse_write(f"patch({record_id})")
        current = self.get(record_id)
        if current is None:
            logger.warning("patch on unknown evidence id: %s", record_id)
            return None
        # 常駐レコードから写した型付きの階層 (provenance の要素等) も素の形で書く
        fields = {key: plain_value(value) for key, value in fields.items()}
        check_writable(apply_patch(current, fields, unset=unset))
        with self._exclusive_write(f"patch({record_id})"):
            event = self.events.append_patch(record_id, fields, unset=tuple(unset), by=by)
            self._apply_event(event)
            self.manifest.events_since_snapshot += 1
            return self._overlay.get(record_id)

    def retract(
        self, record_id: str, reason: str, *, by: str | None = None,
    ) -> Evidence | None:
        """``veracity=retracted`` にする。物理削除はしない (監査可能性)。"""
        self._refuse_write(f"retract({record_id})")
        if self.get(record_id) is None:
            logger.warning("retract on unknown evidence id: %s", record_id)
            return None
        with self._exclusive_write(f"retract({record_id})"):
            event = self.events.append_retract(record_id, reason, by=by)
            self._apply_event(event)
            self.manifest.events_since_snapshot += 1
            return self._overlay.get(record_id)

    def touch(self, record_ids: Sequence[str], *, by: str | None = None) -> int:
        """複数 id の ``last_used_at`` を 1 事象で更新する。

        readonly では **例外にせず 0 を返す** — ``last_used_at`` は落としても
        失われるのは順位のヒントだけで、飛ばしても壊れない (呼び手は
        sleep-time の ``flush_touch`` だけ)。
        """
        if self.readonly:
            self._warn_readonly_skip("touch")
            return 0
        known = [rid for rid in dict.fromkeys(record_ids) if self.get(rid) is not None]
        if not known:
            return 0
        with self._exclusive_write("touch"):
            event = self.events.append_touch(known, by=by)
            if event is None:
                return 0
            self._apply_event(event)
            self.manifest.events_since_snapshot += 1
            return len(known)

    def flush_touch(self, *, by: str | None = None) -> int:
        """:class:`UsageBuffer` を drain して 1 つの ``touch`` 事象にする。"""
        ids = self.usage.drain()
        return self.touch(ids, by=by) if ids else 0

    def save_manifest(self) -> None:
        """manifest を書き出す (``events_since_snapshot`` の永続化)。"""
        self.manifest.save()

    # ── ベクトル索引 (c_16 §6.1) ──

    @property
    def embedding_model_id(self) -> str:
        backend = self.embedding_backend
        if backend is None:
            return self.manifest.embedding_model_id or _UNKNOWN_MODEL
        return embedding_store_id(backend) or _UNKNOWN_MODEL

    @property
    def embedding_model_pending(self) -> bool:
        """埋め込みモデルが索引を作ったモデルと違い、再埋め込みの確認待ち (c_05 §0.5.7)。

        全件の埋め込み直しは数時間かかりうるので自動ではやらない。確認
        (:meth:`confirm_reembed`) までは版の索引を読まない (旧モデルのベクトルを
        新しいモデルのクエリで引かない)。
        """
        declared = self.manifest.embedding_model_id
        return bool(
            declared and self.embedding_backend is not None
            and not self._reembed_confirmed and declared != self.embedding_model_id
        )

    def confirm_reembed(self) -> None:
        """利用者の確認: 次の索引作りで現在の埋め込みモデルで全件を埋め直す。"""
        self._reembed_confirmed = True

    def embeddings_model_dir(self, model_id: str | None = None) -> Path:
        """``embeddings/<model_id>/`` (版ディレクトリの親)。"""
        return self.store_dir / "embeddings" / (model_id or self.embedding_model_id)

    def embeddings_dir(
        self, model_id: str | None = None, version: str | None = None,
    ) -> Path:
        """埋め込み索引の **版ディレクトリ** (``embeddings/<model_id>/v0003/``)。

        版を切らずに 1 つのディレクトリへ ``np.save`` × 2 + ``metadata.json``
        を書くと、(a) 3 ファイルの間に非原子な窓ができ、(b) 稼働中の読み手が
        memmap で掴んでいるファイルを上書きすることになる (c_16 §2.1
        「稼働中の索引は書き換えず版を積んで指す」/ c_05 §0.5)。書き手は必ず
        新しい版へ書き、manifest の ``embedding_version`` を snapshot の
        切り替えと **同時に** 進める。

        Args:
            model_id: 埋め込みモデル id。``None`` で現在のもの。
            version: 版名。``None`` なら manifest が指す版 → 無ければ
                ディスク上の最大版 → それも無ければ発番前の ``v0001``。
        """
        root = self.embeddings_model_dir(model_id)
        return root / (version or self._resolve_embedding_version(root))

    def _resolve_embedding_version(self, root: Path) -> str:
        """読み出しに使う埋め込み版を決める (manifest → ディスク → 既定)。"""
        declared = self.manifest.embedding_version
        if declared and (root / declared).is_dir():
            return declared
        latest = self._latest_embedding_version(root, stamped=True)
        if latest:
            if declared:
                logger.warning(
                    "Embedding version %s declared by the %s manifest is "
                    "missing; falling back to %s",
                    declared, self.store_name, latest,
                )
            return latest
        return declared or embedding_version_name(1)

    @staticmethod
    def _latest_embedding_version(root: Path, *, stamped: bool = False) -> str:
        """``embeddings/<model_id>/`` 配下の最大版名 (無ければ空文字)。

        ``stamped=True`` なら ``stamp.json`` のある (書き終えた) 版だけから選ぶ。
        発番は書きかけの版も数える (版名を再利用しない)。
        """
        if not root.is_dir():
            return ""
        names = [
            path.name for path in root.iterdir()
            if path.is_dir() and embedding_version_seq(path.name) is not None
            and (not stamped or (path / STAMP_FILE).exists())
        ]
        if not names:
            return ""
        return max(names, key=lambda name: embedding_version_seq(name) or 0)

    def _take_next_embedding_version(self, root: Path) -> str:
        """次に書く埋め込み版を発番する (ディスク上の最大版 + 1)。

        番号はディスクから引く — manifest だけを鍵にすると、manifest を
        復元した直後に既存の版へ書き戻してしまう。
        """
        latest = self._latest_embedding_version(root)
        seq = (embedding_version_seq(latest) or 0) if latest else 0
        declared = embedding_version_seq(self.manifest.embedding_version or "") or 0
        return embedding_version_name(max(seq, declared) + 1)

    def prune_embedding_versions(
        self, model_id: str | None = None, *, protect: Sequence[str] = (),
    ) -> list[str]:
        """古い埋め込み版を刈る (:data:`EMBEDDING_VERSIONS_KEEP` 版を残す)。

        snapshot の GC と同じ保護を掛ける — ``protect`` (旧 active と新版) は
        件数に関わらず残す。Windows では memmap を掴んだままの版が消せない
        ので、消せなかったものは黙って次回へ回す。

        Returns:
            消せた版名。
        """
        root = self.embeddings_model_dir(model_id)
        if not root.is_dir():
            return []
        protected = {v for v in protect if v}
        versions = sorted(
            (
                path.name for path in root.iterdir()
                if path.is_dir() and embedding_version_seq(path.name) is not None
            ),
            key=lambda name: embedding_version_seq(name) or 0,
        )
        surplus = len(versions) - EMBEDDING_VERSIONS_KEEP
        if surplus <= 0:
            return []
        removed: list[str] = []
        for version in [v for v in versions if v not in protected][:surplus]:
            target = root / version
            shutil.rmtree(target, ignore_errors=True)
            if target.exists():
                logger.warning(
                    "failed to prune embedding version %s (%s); retrying next time",
                    version, self.store_name,
                )
                continue
            removed.append(version)
        if removed:
            logger.info(
                "Pruned %d old embedding version(s) for %s: %s",
                len(removed), self.store_name, ", ".join(removed),
            )
        return removed

    async def embed_and_index_snapshot(self, *, defer_manifest: bool = False) -> int:
        """active snapshot の埋め込みを **増分で** 更新し、クラスタ索引を作る。

        c_16 §6.1: **全ストアで snapshot 生成時に構築**する (現行のカートリッジ
        のように 5,000 件以上でしか作らない、にはしない)。追記のたびの再構築は
        しない。

        前の版の ``embeddings/<model_id>/`` に ``text_hash`` (再利用鍵) が
        残っていれば、**本文も埋め込み側も変わっていない id は int8 の行と
        スケールをそのまま複写** する。復元 (``dequantize``) して量子化し直すと
        往復のたびに値がずれるので、量子化済みの行をそのまま運ぶ。埋め込みを
        呼ぶのは新規 id と鍵が変わった id だけ。

        埋め込む行は ``attrs`` が宣言する側 (:func:`embed_side_of`) ごとに
        バッチへ分け、``embed(..., is_query=..., mode=...)`` を側の数だけ呼ぶ。
        結果は元の行位置へ戻すので、行順は宣言に依らず snapshot と一致する。

        埋め込みモデルが manifest の宣言と違う場合は流用せず全件やり直す。

        書き先は **新しい版ディレクトリ** (``embeddings/<model_id>/v<N>/``)。
        稼働中の索引を上書きしないための版付けで、manifest の
        ``embedding_version`` が指す版だけが読まれる (c_16 §2.1 / §6.1)。

        Args:
            defer_manifest: ``True`` なら manifest を **保存しない** (版の
                指し替えは呼出側が snapshot の切り替えと同時に行う)。
                :meth:`create_snapshot` からの呼び出しだけが ``True``。

        Returns:
            索引に入れた件数 (埋め込みバックエンドが無ければ 0)。
        """
        if self.embedding_backend is None or self._snapshot is None or self.embedding_model_pending:
            return 0
        ids, texts, sides = self._snapshot_id_texts()
        if not ids:
            return 0

        model_id = self.embedding_model_id
        hashes = [
            embed_reuse_key(text, *side) for text, side in zip(texts, sides)
        ]

        model_root = self.embeddings_model_dir(model_id)
        previous_version = (
            self._resolve_embedding_version(model_root) if model_root.is_dir() else ""
        )
        previous: EvidenceVectorStore | None = None
        if previous_version and (model_root / previous_version).is_dir():
            previous = self._open_vector_store(model_root / previous_version)
        reusable = (
            self._reusable_rows(previous, model_id) if previous is not None else {}
        )
        reuse_dst, reuse_src, pending = plan_reuse(ids, hashes, reusable)

        vectors: np.ndarray | None = None
        if pending:
            vectors = await self._embed_all(
                [(texts[row], *sides[row]) for row in pending],
            )
        q8, scales = assemble_embeddings(
            previous, reuse_dst, reuse_src, pending, vectors, len(ids),
        )

        version = self._take_next_embedding_version(model_root)
        logger.info(
            "Embedded %d row(s), reused %d of %d for %s (%s, %s)",
            len(pending), len(reuse_dst), len(ids), self.store_name,
            model_id, version,
        )
        store = EvidenceVectorStore(
            model_root / version,
            memmap_threshold=self._rag_int("memmap_threshold", DEFAULT_MEMMAP_THRESHOLD),
            quantization=self._rag_str("quantization", "int8"),
        )
        self._write_vector_store(store, ids, hashes, sides, q8, scales)
        self._vector_store = store
        self._vector_rows = None
        self._embedding_stale = False
        self.manifest.embedding_version = version
        self.manifest.embedding_model_id = model_id
        if not defer_manifest:
            # 単発呼び出し (reindex / reembed) は自分で指し替えを永続化する。
            # ``create_snapshot`` からの呼び出しでは snapshot の切り替えと
            # **同じ save** で切り替わる。
            self.manifest.save()
        self.prune_embedding_versions(
            model_id, protect=(previous_version, version),
        )
        return len(ids)

    def _embed_prep_spec(self) -> tuple[EmbedPrepSpec | None, str]:
        """版の組み立てと一緒に埋め込みの準備をするための値と、前の埋め込み版名。

        埋め込みバックエンドが無ければ ``(None, "")`` (索引を作らない)。
        """
        if self.embedding_backend is None or self.embedding_model_pending:
            return None, ""
        model_id = self.embedding_model_id
        model_root = self.embeddings_model_dir(model_id)
        previous = self._resolve_embedding_version(model_root) if model_root.is_dir() else ""
        spec = EmbedPrepSpec(
            previous_dir=str(model_root / previous) if previous else "",
            declared_model_id=self.manifest.embedding_model_id,
            model_id=model_id,
            expected_dim=int(_backend_value(self.embedding_backend, "dim", 0) or 0),
            memmap_threshold=self._rag_int("memmap_threshold", DEFAULT_MEMMAP_THRESHOLD),
            quantization=self._rag_str("quantization", "int8"),
        )
        return spec, previous

    async def _build_prepared_index(
        self, prep: EmbedPrep, previous_version: str, *, snapshot_version: str, id_hash: str,
    ) -> str:
        """組み立て側が決めた計画で埋め込み、索引の新しい版を子プロセスで書く。

        :meth:`embed_and_index_snapshot` と同じ結果になる (同じ関数で組む)。違いは
        ループ上でやるのが **埋め込む行の HTTP だけ** になること — 本文の全件読み
        (50k で 0.6 秒)・int8 行列の組み立て・書き出し・k-means (同 2.2 秒) は
        子プロセス。読み手への切り替えは :meth:`_switch_index` (差し替えと同時)。

        Returns:
            書いた埋め込みの版名。
        """
        model_id = self.embedding_model_id
        model_root = self.embeddings_model_dir(model_id)
        vectors: np.ndarray | None = None
        if prep.pending:
            with embed_priority(P1_FRESHNESS, override=False):
                vectors = await self._embed_all([
                    (text, *prep.sides[row])
                    for row, text in zip(prep.pending, prep.pending_texts)
                ])
        version = self._take_next_embedding_version(model_root)
        logger.info(
            "Embedded %d row(s), reused %d of %d for %s (%s, %s)",
            len(prep.pending), len(prep.reuse_dst), len(prep.ids), self.store_name,
            model_id, version,
        )
        await run_index_build(IndexBuildJob(
            directory=str(model_root / version),
            previous_dir=str(model_root / previous_version) if previous_version else "",
            memmap_threshold=self._rag_int("memmap_threshold", DEFAULT_MEMMAP_THRESHOLD),
            quantization=self._rag_str("quantization", "int8"),
            ids=prep.ids,
            hashes=prep.hashes,
            sides=prep.sides,
            reuse_dst=prep.reuse_dst,
            reuse_src=prep.reuse_src,
            pending=prep.pending,
            vectors=vectors,
            source=snapshot_version,
            snapshot_rows=len(prep.ids),
            id_hash=id_hash,
            model_id=model_id,
            backend_type=str(_backend_value(self.embedding_backend, "backend_type", "")),
            cluster=self._cluster_enabled(),
            n_probe_ratio=self._cluster_n_probe_ratio(),
        ))
        return version

    def _switch_index(self, version: str, previous_version: str, rows: int) -> int:
        """書いた埋め込みの版へ読み手を切り替える (manifest の保存は呼出側)。"""
        model_id = self.embedding_model_id
        self.manifest.embedding_version = version
        self.manifest.embedding_model_id = model_id
        # 新しい版をここで読み込む (差し替えと同じ応答の合間)。遅延させると最初の
        # 検索がチャットの応答中に読み込みを払う。
        self._vector_store = None
        self._vector_rows = None
        self._embedding_stale = False
        self.vector_store()
        self.prune_embedding_versions(
            model_id, protect=(previous_version, version),
        )
        return rows

    def _snapshot_id_texts(
        self,
    ) -> tuple[list[str], list[str], list[tuple[bool, str]]]:
        """active snapshot の (id, 本文, 埋め込み側) を行順で読む。"""
        snapshot = self._snapshot
        if snapshot is None:
            return [], [], []
        ids: list[str] = []
        texts: list[str] = []
        sides: list[tuple[bool, str]] = []
        for row, raw in snapshot.iter_raw():
            if raw is None:
                continue
            ids.append(str(raw.get("id") or snapshot.id_at(row)))
            texts.append(str(raw.get("text") or ""))
            sides.append(embed_side_of(raw.get("attrs")))
        return ids, texts, sides

    def _reusable_rows(
        self, store: EvidenceVectorStore, model_id: str,
    ) -> dict[str, tuple[int, str]]:
        """前の版の VectorStore から ``id → (行, text_hash)`` を作る (:func:`reusable_rows`)。"""
        return reusable_rows(
            store,
            declared_model_id=self.manifest.embedding_model_id,
            model_id=model_id,
            expected_dim=int(_backend_value(self.embedding_backend, "dim", 0) or 0),
            store_name=self.store_name,
        )

    async def _embed_all(
        self, items: Sequence[tuple[str, bool, str]],
    ) -> np.ndarray:
        """``(本文, is_query, mode)`` を **側ごとに** バッチで埋め込む。

        ``embed()`` は 1 回の呼び出しで 1 つの側しか扱えない (instruction は
        バッチ単位で決まる) ので、側で分けてから :data:`EMBED_BATCH_SIZE` ずつ
        投げる。返す配列は **入力と同じ行順** — 呼出側は snapshot の行位置で
        書き戻すので、側でまとめた並びをそのまま返すと行が入れ替わる。
        """
        backend = self.embedding_backend
        assert backend is not None
        groups: dict[tuple[bool, str], list[int]] = {}
        for position, (_text, is_query, mode) in enumerate(items):
            groups.setdefault((is_query, mode), []).append(position)

        out: np.ndarray | None = None
        for (is_query, mode), positions in groups.items():
            for start in range(0, len(positions), EMBED_BATCH_SIZE):
                window = positions[start:start + EMBED_BATCH_SIZE]
                vectors = np.asarray(
                    await backend.embed(
                        [items[p][0] for p in window],
                        is_query=is_query,
                        mode=mode,
                    ),
                    dtype=np.float32,
                )
                if out is None:
                    out = np.zeros(
                        (len(items), int(vectors.shape[1])), dtype=np.float32,
                    )
                out[np.asarray(window, dtype=np.int64)] = vectors
        if out is None:
            return np.zeros((0, 0), dtype=np.float32)
        if len(groups) > 1:
            logger.info(
                "Embedded %d row(s) across %d embedding side(s) for %s: %s",
                len(items), len(groups), self.store_name,
                ", ".join(
                    f"{embed_side_key(q, m)}={len(rows)}"
                    for (q, m), rows in groups.items()
                ),
            )
        return out

    def _open_vector_store(self, directory: Path) -> EvidenceVectorStore:
        """埋め込みの版ディレクトリを開く (無い / 書きかけなら空で返る)。"""
        store = EvidenceVectorStore(
            directory,
            memmap_threshold=self._rag_int("memmap_threshold", DEFAULT_MEMMAP_THRESHOLD),
            quantization=self._rag_str("quantization", "int8"),
        )
        store.load()
        return store

    def _write_vector_store(
        self,
        store: EvidenceVectorStore,
        ids: list[str],
        hashes: list[str],
        sides: list[tuple[bool, str]],
        q8: np.ndarray,
        scales: np.ndarray,
    ) -> EvidenceVectorStore:
        """組み上がった int8 行列を埋め込みの版へ書く (:func:`write_vector_store`)。

        刻印の snapshot は **いま読んでいる版** (版の生成中は manifest がまだ旧版を
        指しているので、manifest からは取らない)。
        """
        snapshot = self._snapshot
        return write_vector_store(
            store, ids, hashes, sides, q8, scales,
            source=snapshot.directory.name if snapshot is not None else "",
            snapshot_rows=len(snapshot) if snapshot is not None else len(ids),
            id_hash=self._snapshot_id_hash(),
            model_id=self.embedding_model_id,
            backend_type=str(_backend_value(self.embedding_backend, "backend_type", "")),
            cluster=self._cluster_enabled(),
            n_probe_ratio=self._cluster_n_probe_ratio(),
        )

    def close(self) -> None:
        """memmap を握った索引と snapshot を手放す (Windows で削除できるように)。"""
        vector_store = self._vector_store
        if vector_store is not None:
            vector_store.vectors_q8 = None
            vector_store.scales = None
        self._vector_store = None
        self._vector_rows = None
        self._snapshot = None
        self._lexical = None

    def vector_store(self) -> EvidenceVectorStore | None:
        """ベクトル索引 (無ければ ``None``)。初回アクセスで lazy load。

        読むのは manifest の ``embedding_version`` が指す版だけ。指し先が
        無ければディスク上の最大版へ落とす (manifest を復元した直後)。

        埋め込み版の刻印 (snapshot 版・行数・id ハッシュ) が active snapshot と
        合わなければ使わない (c_16 §5.6) — :attr:`embedding_stale` を立て、
        :meth:`refresh_tail` か次の版の生成で作り直す。
        """
        if self.embedding_model_pending:
            return None
        if self._vector_store is not None:
            return self._vector_store
        if self._embedding_stale:
            return None
        directory = self.embeddings_dir(
            self.manifest.embedding_model_id or self.embedding_model_id,
        )
        if not directory.is_dir():
            return None
        store = self._open_vector_store(directory)
        if self._snapshot is not None and not self._stamp_matches(store):
            store.vectors_q8 = None
            store.scales = None
            self._embedding_stale = True
            return None
        self._vector_store = store
        return self._vector_store

    @property
    def embedding_stale(self) -> bool:
        """埋め込み版の刻印が active snapshot と合わなかった (作り直し待ち)。"""
        return self._embedding_stale

    def _snapshot_id_hash(self) -> str:
        """active snapshot の行 id のハッシュ (埋め込み版の刻印と照合する値)。

        版を作ったときに COMPLETE へ記録した値を使う (読み直さない)。無ければ
        columns の id から求める。
        """
        snapshot = self._snapshot
        if snapshot is None:
            return ""
        embedding = snapshot.complete.embedding_version if snapshot.complete is not None else None
        if embedding is not None and embedding.id_hash:
            return embedding.id_hash
        return ids_digest(value.decode("ascii") for value in snapshot.columns.ids.tolist())

    def _stamp_matches(self, store: EvidenceVectorStore) -> bool:
        """埋め込み版の刻印が active snapshot (版名・行数・id ハッシュ) と合うか。"""
        snapshot = self._snapshot
        assert snapshot is not None
        stamp = store.stamp
        expected = {
            "snapshot": snapshot.directory.name,
            "rows": len(snapshot),
            "id_hash": self._snapshot_id_hash(),
        }
        actual = {key: stamp.get(key) for key in expected}
        if actual == expected:
            return True
        logger.warning(
            "Embedding version %s of %s does not match snapshot %s "
            "(stamp=%s, expected=%s); it will be rebuilt",
            store.vectors_dir.name, self.store_name, snapshot.directory.name,
            actual, expected,
        )
        return False

    def vector_candidates(
        self, query_vec: np.ndarray, top_k: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ベクトル検索の候補を **snapshot の行番号** と cosine で返す。

        順位・ゲートは呼び出し側が素の cosine から計算する (c_16 §6.3 / §7.1)。
        snapshot に入っていない ``put`` 直後のレコードは対象外。
        """
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
        store = self.vector_store()
        if store is None or self._snapshot is None:
            return empty
        hits = store.search(query_vec, top_k=top_k)
        if not hits:
            return empty
        rows: list[int] = []
        scores: list[float] = []
        for record_id, score, _text in hits:
            row = self._snapshot.row_of(record_id)
            if row is None:
                continue
            rows.append(row)
            scores.append(float(score))
        if not rows:
            return empty
        return np.array(rows, dtype=np.int64), np.array(scores, dtype=np.float32)

    def _vector_row_map(self) -> np.ndarray | None:
        """snapshot 行 → VectorStore 行 (無い行は -1)。

        ``_write_vector_store`` は snapshot の行順で書くので普段は恒等写像に
        なるが、埋め込みが古い版のまま残っている / 一部の行が読めなかった場合
        にずれる。id で引き直して取り違えを防ぐ。
        """
        if self._vector_rows is not None:
            return self._vector_rows
        store = self.vector_store()
        snapshot = self._snapshot
        if store is None or snapshot is None:
            return None
        positions = {record_id: row for row, record_id in enumerate(store.row_ids)}
        mapping = np.fromiter(
            (positions.get(snapshot.id_at(row), -1) for row in range(len(snapshot))),
            dtype=np.int64,
            count=len(snapshot),
        )
        self._vector_rows = mapping
        return mapping

    def cosines_for_rows(
        self, query_vec: np.ndarray, rows: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """指定 snapshot 行の **素の cosine** を返す (``rows`` と同じ並び)。

        語彙索引だけで拾われた行に順位を付けるための入口 (c_16 §6.3)。
        ベクトルを持たない行は ``NaN`` を返す — 0.0 を返すと閾値が 0 以下の
        ときにゲートを素通りしてしまう。:func:`ranking.gate_by_cosine` は NaN を
        落とすので、そのまま合流させてよい。
        """
        idx = np.asarray(rows, dtype=np.int64)
        out = np.full(idx.shape[0], np.nan, dtype=np.float64)
        if idx.shape[0] == 0:
            return out
        store = self.vector_store()
        mapping = self._vector_row_map()
        if store is None or mapping is None or store.vectors_q8 is None:
            return out
        if store.scales is None or len(store.vectors_q8) == 0:
            return out

        in_range = (idx >= 0) & (idx < mapping.shape[0])
        vector_rows = np.where(in_range, mapping[np.clip(idx, 0, mapping.shape[0] - 1)], -1)
        known = in_range & (vector_rows >= 0)
        if not np.any(known):
            return out

        query = np.asarray(query_vec, dtype=np.float32).ravel()
        selected = vector_rows[known]
        restored = dequantize_int8(
            np.asarray(store.vectors_q8)[selected], np.asarray(store.scales)[selected],
        )
        if restored.ndim != 2 or restored.shape[1] != query.shape[0]:
            logger.warning(
                "cosines_for_rows: query dim %d != stored dim %s (%s)",
                query.shape[0],
                restored.shape[1] if restored.ndim == 2 else "?",
                self.store_name,
            )
            return out
        norms = np.linalg.norm(restored, axis=1).clip(min=1e-9)
        query_norm = float(np.linalg.norm(query).clip(min=1e-9))
        out[known] = restored @ query / (norms * query_norm)
        return out

    # ── 転置索引 (c_16 §6.2) ──

    def lexical_pack(self) -> LexicalPack | None:
        """active snapshot の転置索引を lazy に開く (無ければ ``None``)。

        開いた時点で読むのは ``shards.json`` だけ。配列はゲートを通ったシャードの
        分だけ :meth:`lexical_candidates` が読む (c_16 §6.2)。
        """
        if self._lexical is not None:
            return self._lexical
        if self._snapshot is None:
            return None
        directory = self._snapshot.directory
        if not (directory / SHARD_MAP_FILE).exists():
            return None
        try:
            pack = LexicalPack(directory)
        except (OSError, ValueError) as e:
            logger.warning("unreadable lexical shard map in %s: %s", directory, e)
            return None
        if not pack.names:
            logger.warning("lexical shard map in %s has no shards", directory)
            return None
        for name in pack.names:
            self._warn_lexical_param_drift(name, pack.params(name))
        self._lexical = pack
        return self._lexical

    def lexical_shard_names(self) -> list[str] | None:
        """active snapshot の語彙シャード名 (索引が無ければ ``None``。配列は読まない)。"""
        pack = self.lexical_pack()
        return None if pack is None else pack.names

    def lexical_candidates(
        self,
        query: str,
        top_k: int = 20,
        *,
        shards: Iterable[str] | None = None,
        row_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """転置索引の候補を **snapshot の行番号** で返す (c_16 §6.2 / §6.3)。

        Args:
            query: 生のクエリ文字列。
            top_k: 返す最大件数。
            shards: 読むシャード名。``None`` で全シャード。
            row_mask: グローバル行で引ける bool 配列。``False`` の行を除外する。

        Returns:
            ``(rows int64, scores float32)``。スコアは **候補の並べ替え専用**
            で、ゲート・順位には使わない (§6.3)。
        """
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        pack = self.lexical_pack()
        if pack is None or top_k <= 0:
            return empty
        names = list(shards) if shards is not None else pack.names
        if not names:
            return empty
        return pack.shard_set(names).candidates_in(
            names, query, top_k,
            budget_ms=self._lexical_budget_ms(),
            row_mask=row_mask,
        )

    def _write_lexical_index(
        self, directory: Path, records: Sequence[Evidence],
    ) -> int:
        """版ディレクトリへ転置索引の pack を書く (:func:`write_lexical_index`)。"""
        return write_lexical_index(
            directory, records,
            shard_key_for=self.shard_key_for,
            builder=self._lexical_builder(),
            lexical_version=int(self.manifest.lexical_version),
        )

    # ── 検索 (c_16 §6.3 + §7.1〜7.3) ──

    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int = 10,
        *,
        threshold: float = 0.0,
        now: float | None = None,
        include_private: bool = False,
        shards: Iterable[str] | None = None,
        store_prior: float | np.ndarray | None = None,
    ) -> list[tuple[int, float, float]]:
        """候補生成 → ゲート → 順位 → 畳み込みの 1 本 (c_16 §6.3 / §7.1〜7.3)。

        手順:

        1. アクティブマスク (``retracted`` / ``superseded`` / ``secret`` / 失効 /
           private) を先に作り、語彙索引の走査から除外する
        2. 候補 = ベクトル top-k ∪ 転置索引 top-k。語彙だけで拾われた行は
           :meth:`cosines_for_rows` で **実 cosine** を計算する (lexical スコアは
           持ち込まない)
        3. ゲートは素の cosine のみ (§7.1)
        4. ``score = cos × freshness × confidence × store_prior`` (§7.2)
        5. ``claim_key`` 畳み込み + origin 優先 + assistant 除外 (§7.3)

        Args:
            query_text: 語彙索引に渡す生のクエリ。
            query_vec: クエリの埋め込み。
            top_k: 返す最大件数。
            threshold: cosine のゲート閾値 (モデルプロファイル同期の値)。
            now: 現在時刻の epoch 秒。``None`` で ``utc_now_dt()``。同じ候補を
                何度評価しても同じ順位になるよう、内部で 2 回時刻を取らない。
            include_private: private セッションでのみ ``True``。
            shards: 読む語彙シャード。``None`` で全シャード。
            store_prior: §7.2 の ``store_prior`` (スカラ / 行揃え配列)。
                ``None`` で ``memory.evidence.ranking.store_prior`` の
                ストア別の値 (:attr:`default_store_prior`)。

        Returns:
            ``(snapshot 行, cosine, score)`` をスコア降順で最大 ``top_k`` 件。
        """
        snapshot = self._snapshot
        if snapshot is None or len(snapshot) == 0 or top_k <= 0:
            return []
        now_epoch = utc_now_dt().timestamp() if now is None else float(now)
        candidate_k = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)

        mask = self.active_mask(now_epoch, include_private)
        vector_rows, vector_cos = self.vector_candidates(query_vec, top_k=candidate_k)
        lexical_rows, _lexical_scores = self.lexical_candidates(
            query_text, candidate_k, shards=shards, row_mask=mask,
        )
        rows, cosines = merge_candidates(
            vector_rows, vector_cos, lexical_rows,
            lambda target: self.cosines_for_rows(query_vec, target),
        )
        if rows.shape[0] == 0:
            return []

        keep = gate_by_cosine(cosines, threshold) & mask[rows]
        rows = rows[keep]
        cosines = cosines[keep]
        if rows.shape[0] == 0:
            return []

        columns = RankColumns.from_columns(snapshot.columns)
        prior = (
            self.default_store_prior if store_prior is None else store_prior
        )
        scores = score_rows(cosines, rows, columns, now_epoch, prior)
        final_rows, final_scores = collapse(
            rows, scores, columns,
            allow_assistant_origin=self.allow_assistant_origin_injection,
        )
        if final_rows.shape[0] == 0:
            return []

        # rows は merge_candidates が行番号昇順で返し、以降のマスクも順序を
        # 保つので、二分探索で cosine を引き戻せる。
        positions = np.searchsorted(rows, final_rows)
        final_cos = cosines[positions]
        limit = min(top_k, final_rows.shape[0])
        return [
            (int(final_rows[i]), float(final_cos[i]), float(final_scores[i]))
            for i in range(limit)
        ]

    # ── snapshot 生成 (c_16 §5.3 / §5.6) ──

    async def create_snapshot(self, now: str | None = None) -> str:
        """事象を畳んで新しい版を作り、``COMPLETE`` を書いてから manifest を切り替える。

        手順 (c_16 §5.6。順序が意味を持つ):

        1. 事象ログの末尾位置を **先に** 確定し (畳み込み中の追記を「畳み済み」と
           誤記録しないため)、版番号を発番して manifest を保存する
        2. 畳む事象の月ファイルを fsync する (事象の prune 後は版の records が唯一の原本)
        3. 前 snapshot + その位置までの事象を畳み、物理 GC (``gc_filter`` /
           ``gc_drop_ids``、c_16 §5.4) を掛けて、新しい版ディレクトリへ
           ``records.jsonl`` (fsync) / ``offsets.npy`` / ``columns.npz`` /
           ``lexical.npz`` + ``shards.json`` を書く
        4. 埋め込み (増分) + クラスタ索引の新しい版を書く (snapshot 版・行数・
           id ハッシュを刻む)
        5. ``COMPLETE`` を fsync して書く — ここで版が確定する
        6. manifest を切り替える (``active_snapshot`` と埋め込み版を同じ ``save()`` で)
        7. 古い版 (COMPLETE の無い版を含む) を刈り、事象は保持中で最も古い
           COMPLETE 版の ``folded_through`` より前の月だけ刈る

        途中で落ちても manifest は生きた版を指したままで、書きかけの版は
        COMPLETE が無いので読まれない (次の GC が trash へ送る)。

        Returns:
            新しい版名 (``v0007``)。

        Raises:
            EvidenceStoreReadonlyError: ストアが readonly のとき。
        """
        if self._snapshot_lock.locked():
            # 並走は不変則違反 (書き手は sleep-time 1 本) だが、ここで落とすと
            # 版が飛ぶ。直列化して 1 本ずつ通し、痕跡をログに残す。
            logger.warning(
                "create_snapshot on %s is waiting for an in-progress snapshot "
                "(the store expects a single writer)", self.store_name,
            )
        async with self._snapshot_lock:
            return await self._create_snapshot_locked(now)

    async def _create_snapshot_locked(self, now: str | None = None) -> str:
        """:meth:`create_snapshot` の本体 (スナップショットロック保持中に呼ぶ)。"""
        self._refuse_write("create_snapshot")
        end = self.events.current_position()
        prev_version = (
            self._snapshot.directory.name if self._snapshot is not None else ""
        )
        drop_ids = (
            frozenset(self.gc_drop_ids()) if self.gc_drop_ids is not None else frozenset()
        )
        embed_spec, previous_embedding = self._embed_prep_spec()

        old_active = self.manifest.active_snapshot
        version = self.manifest.take_next_version()
        # 版番号は **書き始める前に** 永続化する。ここで落ちると (taskkill /F)
        # 次の起動が同じ番号を再発番し、書きかけの版を別内容で上書きする。
        # active_snapshot はまだ旧版のままなので、途中で落ちても読み手は
        # 生きた版を指したままになる。
        self.manifest.save()
        loop = asyncio.get_running_loop()
        await run_in_executor_with_context(
            loop, None, self.events.fsync_range, self.manifest.folded_through, end,
        )
        # 畳み込み・物理 GC・版ファイル・転置索引は子プロセスで組み立てる
        # (G1 設計 §11-2 の T1)。入力はファイルとここで固めた値だけで、
        # manifest・オーバーレイ・読み手には触らない。
        # 差し替え (L2) までの await の間に適用した事象を取っておく。
        self._events_during_build = []
        try:
            build = await run_snapshot_build(SnapshotBuildJob(
                store_dir=str(self.store_dir),
                prev_version=prev_version,
                folded_through=self.manifest.folded_through,
                end=end,
                version=version,
                drop_ids=drop_ids,
                shard_key_for=self.shard_key_for,
                builder=self._lexical_builder(),
                lexical_version=int(self.manifest.lexical_version),
                keep=self.gc_filter,
                embed=embed_spec,
            ))
            # 埋め込みと索引の新しい版も差し替えの前に作る (旧版の読み手は
            # 旧版の索引を id で引くので、その間の検索は整合したまま)。
            index_version = ""
            if build.embed_prep is not None and build.embed_prep.ids:
                index_version = await self._build_prepared_index(
                    build.embed_prep, previous_embedding,
                    snapshot_version=version, id_hash=build.id_hash,
                )
            await run_in_executor_with_context(
                loop, None, self._write_complete, version, end, build, index_version,
            )
            # L2 (読み手の差し替え・事象の戻し・索引の読み込み・manifest) は
            # ループ上で数十〜百数十 ms かかるので、応答の生成中と生成前の
            # リクエストがある間は待つ (最長 60 秒、G1 設計 §17.3)。
            await wait_until_idle()
        finally:
            late_events, self._events_during_build = self._events_during_build, None
        folded = build.folded
        shard_count = build.shard_count
        if build.dropped_ids:
            self._write_gc_log(version, build.dropped_ids)

        # 新しい版を読み直してから索引を作る (行番号は新 snapshot 基準)。
        self._snapshot = read_snapshot(self.store_dir, version)
        self._overlay = {}
        self._tail_ids = []
        # 組み立ての await の間に追記された事象 (``end`` より後) をオーバーレイへ
        # 戻す。捨てると次の版まで get / 検索から見えなくなる。事象ログから
        # 読み直すと月ファイルを先頭から数えることになる (50k 行で 0.24 秒
        # ループが止まる) ので、適用した事象をメモリに取っておく。
        for event in late_events:
            self._apply_event(event)
        self._prune_tail_vectors()
        self._vector_store = None
        self._vector_rows = None
        self._lexical = None
        if build.embed_prep is None:
            indexed = await self.embed_and_index_snapshot(defer_manifest=True)
        elif index_version:
            indexed = self._switch_index(
                index_version, previous_embedding, len(build.embed_prep.ids),
            )
        else:
            indexed = 0

        self.manifest.active_snapshot = version
        self.manifest.folded_through = end
        # ``= 0`` にすると、埋め込みの await 窓で追記された事象 (次の版が
        # 畳む分) まで「畳み済み」に数えてしまい、そのぶん次の snapshot が
        # 積まれなくなる。畳んだ数だけ引く。
        self.manifest.events_since_snapshot = max(
            0, self.manifest.events_since_snapshot - folded,
        )
        if self.embedding_backend is not None and indexed:
            self.manifest.embedding_model_id = self.embedding_model_id
            self.manifest.embedding_dim = int(
                _backend_value(self.embedding_backend, "dim", 0) or 0,
            )
        self.manifest.save()
        prune_snapshots(
            self.store_dir,
            int(self.manifest.retention_value("snapshots_keep")),
            protect={old_active, version},
        )
        self._prune_events()
        logger.info(
            "Created snapshot %s for %s: %d record(s), %d indexed, %d lexical shard(s), "
            "folded through %s:%d (at %s)",
            version, self.store_name, build.records, indexed, shard_count,
            end.month or "-", end.line, now or utc_now(),
        )
        return version

    def _write_complete(
        self, version: str, end: EventPosition, build: Any, index_version: str,
    ) -> None:
        """版を確定する ``COMPLETE`` を書く (c_16 §5.6 の手順 4、fsync)。"""
        embedding: CompleteEmbedding | None = None
        if index_version and build.embed_prep is not None:
            embedding = CompleteEmbedding(
                model_id=self.embedding_model_id,
                version=index_version,
                rows=len(build.embed_prep.ids),
                id_hash=build.id_hash,
            )
        write_complete(
            snapshot_dir(self.store_dir, version),
            new_complete(folded_through=end, rows=build.records, embedding_version=embedding),
        )

    def _prune_events(self) -> None:
        """保持中で最も古い COMPLETE 版の ``folded_through`` より前の月だけ事象を刈る。

        回復や無効な版からの後退は、残っている COMPLETE 版のどれからでも事象を
        畳み直せなければならない (c_16 §5.2)。新しい版の畳み込み位置で刈ると、
        古い版へ落ちたときに畳む事象が無くなる。
        """
        versions = list_versions(self.store_dir)
        if not versions:
            return
        oldest = read_complete(snapshot_dir(self.store_dir, versions[0]))
        position = oldest.position if oldest is not None else EventPosition()
        if not position.month:
            return
        self.events.prune(
            int(self.manifest.retention_value("events_keep_months")),
            folded_through=position,
        )

    def _write_gc_log(self, version: str, dropped_ids: list[str]) -> None:
        """落とした id を ``gc.jsonl`` と manifest の ``_extra`` に残す。

        レコードそのものは消えるので、**消えた事実だけは追記で残す** のが
        c_16 §5.4 の「物理削除は保持方針の GC のみ」に対する監査の担保。
        1 版 1 行 (行の版は ``_v``、c_05 §0.5.1)。
        """
        entry = {
            "_v": GC_LOG_ROW_VERSION,
            "written_at": utc_now(),
            "producer": {"component": type(self).__name__, "store": self.store_name},
            "payload": {
                "store": self.store_name,
                "snapshot_version": version,
                "dropped_count": len(dropped_ids),
                "dropped_ids": list(dropped_ids),
            },
        }
        store: JSONLAppendStore[dict[str, Any]] = JSONLAppendStore(
            self.store_dir / GC_LOG_FILE,
            serialize=lambda item: json.dumps(item, ensure_ascii=False),
            deserialize=json.loads,
            key_of=lambda item: str(item["payload"]["snapshot_version"]),
            row_version=GC_LOG_ROW_VERSION,
        )
        store.append(entry)
        logger.info(
            "Physical GC dropped %d record(s) from %s snapshot %s",
            len(dropped_ids), self.store_name, version,
        )
        self.manifest.note_extra(
            "gc", {"snapshot": version, "dropped": len(dropped_ids)},
        )

    # ── rag_config の読み出し (未設定でも動く) ──

    def _rag_int(self, key: str, default: int) -> int:
        value = _section_value(self.rag_config, key)
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _rag_str(self, key: str, default: str) -> str:
        value = _section_value(self.rag_config, key)
        return str(value) if value else default

    def _cluster_enabled(self) -> bool:
        cluster = _section_value(self.rag_config, "cluster_index")
        enabled = _section_value(cluster, "enabled")
        return True if enabled is None else bool(enabled)

    def _cluster_n_probe_ratio(self) -> float:
        cluster = _section_value(self.rag_config, "cluster_index")
        try:
            return float(_section_value(cluster, "n_probe_ratio") or 0.125)
        except (TypeError, ValueError):
            return 0.125

    # ── 語彙索引の設定 (c_16 §9 ``memory.evidence.lexical``) ──

    @property
    def default_store_prior(self) -> float:
        """``memory.evidence.ranking.store_prior.<store>`` (c_16 §7.2 / §9)。"""
        key = STORE_PRIOR_KEYS.get(self.store_name)
        default = DEFAULT_STORE_PRIORS.get(self.store_name, 1.0)
        if key is None:
            return default
        # 設定の形は ``ranking.store_prior.<store>`` — ``ranking.<store>`` を
        # 見ていたので、値を書いても既定のままだった (2026-09-08 監査)。
        priors = _section_value(self._ranking_section(), "store_prior")
        value = _section_value(priors, key)
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    @property
    def allow_assistant_origin_injection(self) -> bool:
        """``origin=assistant`` を注入候補に残すか (c_16 §7.3-4 / §9)。

        既定 ``False``。自分の出力が「過去の記録」として恒久再注入される事故
        (2026-08-15 ライブ監査) を origin 規則で塞ぐのが c_16 §1 の眼目なので、
        真にするのは検証用途に限る。
        """
        value = _section_value(
            self._ranking_section(), "allow_assistant_origin_injection",
        )
        return bool(value) if value is not None else False

    def _ranking_section(self) -> Any:
        """``rag_config`` に重ねられた ``memory.evidence.ranking`` (無ければ None)。"""
        return _section_value(self.rag_config, "ranking")

    def _lexical_setting(self, key: str) -> Any:
        """``rag_config.lexical.<key>`` を引く (dict / オブジェクトどちらでも)。"""
        default = LEXICAL_DEFAULTS[key]
        config = self.rag_config
        if config is None:
            return default
        section = (
            config.get("lexical") if isinstance(config, dict)
            else getattr(config, "lexical", None)
        )
        if section is None:
            return default
        value = (
            section.get(key) if isinstance(section, dict)
            else getattr(section, key, None)
        )
        return default if value is None else value

    def _warn_lexical_param_drift(self, name: str, stored: LexicalParams) -> None:
        """索引に焼き付いたパラメータと現在の設定の食い違いを 1 度だけ警告する。

        ``q_terms`` / ``m_postings`` / ``max_df_ratio`` は **索引を作った時点で
        凍る** (c_16 §6.2)。``max_df_ratio`` は剪定そのもの、``m_postings`` は
        posting の切り詰め位置に効くので、設定を変えても次の snapshot まで
        検索の挙動は変わらない。「変えたのに効かない」を黙らせない。
        """
        if self._lexical_drift_warned:
            return
        current = {
            "q_terms": int(self._lexical_setting("q_terms")),
            "m_postings": int(self._lexical_setting("m_postings")),
            "max_df_ratio": float(self._lexical_setting("max_df_ratio")),
            # トークナイザの切り方が変わった索引は再構築まで旧い語彙のまま
            # (新しい unigram は乗らない)。偽陽性は出ないが、黙らせない。
            "tokenizer_version": TOKENIZER_VERSION,
        }
        drift = {
            key: (getattr(stored, key), value)
            for key, value in current.items()
            if getattr(stored, key) != value
        }
        if not drift:
            return
        self._lexical_drift_warned = True
        logger.warning(
            "Lexical index params for %s (shard %s) were frozen at build time "
            "and differ from the config: %s. They take effect on the next "
            "snapshot (c_16 6.2).",
            self.store_name, name,
            ", ".join(
                f"{key}: index={was!r} config={now!r}"
                for key, (was, now) in sorted(drift.items())
            ),
        )

    def _lexical_builder(self) -> LexicalIndexBuilder:
        """設定 (未設定なら c_16 §9 の既定) から索引ビルダを作る。"""
        try:
            return LexicalIndexBuilder(
                q_terms=int(self._lexical_setting("q_terms")),
                m_postings=int(self._lexical_setting("m_postings")),
                max_df_ratio=float(self._lexical_setting("max_df_ratio")),
            )
        except (TypeError, ValueError) as e:
            logger.warning("invalid lexical config (%s); using defaults", e)
            return LexicalIndexBuilder(
                q_terms=int(LEXICAL_DEFAULTS["q_terms"]),
                m_postings=int(LEXICAL_DEFAULTS["m_postings"]),
                max_df_ratio=float(LEXICAL_DEFAULTS["max_df_ratio"]),
            )

    def _lexical_budget_ms(self) -> float:
        """走査予算 (ms、c_16 §6.2)。超過時は読めた分だけで続行する。"""
        try:
            return float(self._lexical_setting("budget_ms"))
        except (TypeError, ValueError):
            return float(LEXICAL_DEFAULTS["budget_ms"])


def _section_value(section: Any, key: str) -> Any:
    """dict / 属性のどちらでも読める安全なアクセサ (``None`` は未設定)。"""
    if section is None:
        return None
    value = (
        section.get(key) if isinstance(section, dict)
        else getattr(section, key, None)
    )
    return value


def reusable_rows(
    store: EvidenceVectorStore,
    *,
    declared_model_id: str,
    model_id: str,
    expected_dim: int,
    store_name: str,
) -> dict[str, tuple[int, str]]:
    """前の版の VectorStore から ``id → (行, text_hash)`` を作る。

    以下のいずれかに当たれば空を返す = 全件埋め込み直し:

    - manifest が別の埋め込みモデルを宣言している (モデル切替)
    - ベクトルが無い / 次元がバックエンドの宣言と食い違う
    """
    declared = declared_model_id
    if declared and declared != model_id:
        logger.info(
            "Embedding model changed (%s -> %s): full re-embed for %s",
            declared, model_id, store_name,
        )
        return {}
    if store.vectors_q8 is None or store.scales is None:
        return {}
    if len(store.vectors_q8) == 0 or store.vectors_q8.ndim != 2:
        return {}
    stored_dim = int(store.vectors_q8.shape[1])
    if expected_dim and stored_dim != expected_dim:
        logger.warning(
            "Embedding dim changed (%d -> %d) for %s: full re-embed",
            stored_dim, expected_dim, store_name,
        )
        return {}
    usable = min(len(store.row_ids), len(store.text_hashes), len(store.vectors_q8))
    reusable: dict[str, tuple[int, str]] = {}
    for row in range(usable):
        record_id = store.row_ids[row]
        digest = store.text_hashes[row]
        if record_id and digest:
            reusable[record_id] = (row, digest)
    return reusable


def plan_reuse(
    ids: Sequence[str],
    hashes: Sequence[str],
    reusable: dict[str, tuple[int, str]],
) -> tuple[list[int], list[int], list[int]]:
    """``(流用先の行, 流用元の行, 埋め込む行)`` を決める (本文も側も同じ id だけ流用)。"""
    reuse_dst: list[int] = []
    reuse_src: list[int] = []
    pending: list[int] = []
    for row, (record_id, digest) in enumerate(zip(ids, hashes)):
        prior = reusable.get(record_id)
        if prior is not None and prior[1] == digest:
            reuse_dst.append(row)
            reuse_src.append(prior[0])
        else:
            pending.append(row)
    return reuse_dst, reuse_src, pending


def assemble_embeddings(
    previous: EvidenceVectorStore | None,
    reuse_dst: Sequence[int],
    reuse_src: Sequence[int],
    pending: Sequence[int],
    vectors: np.ndarray | None,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """流用する int8 行と新しく埋め込んだ行から ``(q8, scales)`` を組み立てる。

    流用する行は量子化済みのまま複写する (復元して量子化し直すと往復のたびに
    値がずれる)。前の版の memmap は複写し終えたら手放す (Windows は掴んだ
    ままだと消せない)。
    """
    reused_q8: np.ndarray | None = None
    reused_scales: np.ndarray | None = None
    prev_dim = 0
    if (
        reuse_src
        and previous is not None
        and previous.vectors_q8 is not None
        and previous.scales is not None
    ):
        src = np.asarray(reuse_src, dtype=np.int64)
        reused_q8 = np.array(np.asarray(previous.vectors_q8)[src], dtype=np.int8)
        reused_scales = np.array(
            np.asarray(previous.scales)[src], dtype=np.float32,
        ).reshape(-1, 1)
        prev_dim = int(reused_q8.shape[1])
    if previous is not None:
        previous.vectors_q8 = None
        previous.scales = None

    new_q8: np.ndarray | None = None
    new_scales: np.ndarray | None = None
    if pending and vectors is not None:
        new_q8, new_scales = quantize_int8(np.asarray(vectors, dtype=np.float32))
        new_scales = new_scales.reshape(-1, 1)

    dim = int(new_q8.shape[1]) if new_q8 is not None else prev_dim
    q8 = np.zeros((rows, dim), dtype=np.int8)
    scales = np.zeros((rows, 1), dtype=np.float32)
    if reused_q8 is not None:
        dst = np.asarray(reuse_dst, dtype=np.int64)
        q8[dst] = reused_q8
        scales[dst] = reused_scales
    if new_q8 is not None:
        dst = np.asarray(pending, dtype=np.int64)
        q8[dst] = new_q8
        scales[dst] = new_scales
    return q8, scales


def write_vector_store(
    store: EvidenceVectorStore,
    ids: Sequence[str],
    hashes: Sequence[str],
    sides: Sequence[tuple[bool, str]],
    q8: np.ndarray,
    scales: np.ndarray,
    *,
    source: str,
    snapshot_rows: int,
    id_hash: str,
    model_id: str,
    backend_type: str,
    cluster: bool,
    n_probe_ratio: float,
) -> EvidenceVectorStore:
    """組み上がった int8 行列を埋め込みの版ディレクトリへ書く。

    ``VectorStore.add_vectors`` は使わない — あちらは位置採番の chunk id を
    振り、本文を ``chunks/<id>.txt`` に複製する。ここでは行 id を
    **evidence id** にし、本文は ``records.jsonl`` から lazy に読むので
    複製しない。``text_hash`` は次回の増分埋め込みが「本文か側が変わったか」
    を判定する唯一の根拠なので必ず持たせる (``embed_side`` はその内訳を
    後から読めるようにする観測用)。

    刻印 (``stamp.json``) には対応する snapshot の版名・行数・id ハッシュを書き
    (c_16 §5.6)、クラスタ索引の後の **最後に** 書く — 刻印の無い版は書きかけ。
    """
    dim = int(q8.shape[1]) if q8.ndim == 2 else 0
    store.vectors_q8 = q8
    store.scales = scales
    store.row_ids = list(ids)
    store.text_hashes = list(hashes)
    store.embed_sides = [embed_side_key(*side) for side in sides]
    store.stamp = {
        "snapshot": source,
        "rows": int(snapshot_rows),
        "id_hash": id_hash,
        "model_id": model_id,
        "backend_type": backend_type,
        "dim": dim,
    }
    store.store_info = {
        "embedding_model": model_id,
        "embedding_backend": backend_type,
        "embedding_dim": dim,
    }
    store.vectors_dir.mkdir(parents=True, exist_ok=True)
    if cluster:
        # threshold=1: 件数に関わらず必ず構築する (c_16 §6.1)。numpy だけの
        # K-means なので、増分埋め込みでも毎回作り直してよい。
        store.build_cluster_index(threshold=1, n_probe_ratio=n_probe_ratio)
    store.save()
    return store


def _shard_digest(param_key: str, texts: Iterable[str]) -> str:
    """シャードの中身 (構築パラメータ + 行の本文の並び) のハッシュ。"""
    digest = hashlib.sha256(param_key.encode("utf-8"))
    for text in texts:
        data = (text or "").encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()[:16]


def write_lexical_index(
    directory: Path,
    records: Sequence[Evidence],
    *,
    shard_key_for: Callable[[Evidence], str],
    builder: LexicalIndexBuilder,
    lexical_version: int,
    previous_dir: Path | None = None,
) -> int:
    """版ディレクトリへ全シャードの転置索引を 1 つの pack で書く (c_16 §6.2)。

    ``records`` の並びは ``records.jsonl`` の行順と一対一 (同じ列を
    ``write_snapshot`` に渡している)。シャードごとの索引はローカル行で
    引かれるので、``rows`` に **ローカル行 → グローバル行** を持たせて
    :class:`LexicalShardSet` を復元できるようにする。版の組み立て
    (``snapshot_build``、子プロセス) からも呼ぶのでストアの状態に触らない。

    ``previous_dir`` (前の版) に同じ中身 (構築パラメータと本文の並び) のシャードが
    あれば、索引を作り直さずに配列を写す (行の写像だけ新しい版の行で書く)。

    Returns:
        書いたシャード数。
    """
    groups: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        try:
            key = str(shard_key_for(record) or DEFAULT_SHARD)
        except Exception as e:  # noqa: BLE001 — 索引作成で snapshot を落とさない
            logger.warning("shard_key_for failed for %s: %s", record.id, e)
            key = DEFAULT_SHARD
        groups.setdefault(key, []).append(row)

    param_key = json.dumps(
        {"index_version": LEXICAL_INDEX_VERSION, **builder.params.to_json()},
        sort_keys=True,
    )
    digests = {
        name: _shard_digest(param_key, (records[row].text for row in member_rows))
        for name, member_rows in groups.items()
    }
    previous = _open_previous_pack(previous_dir)
    copied: dict[str, dict[str, np.ndarray]] = {}
    if previous is not None:
        same = [
            name for name in groups
            if name in previous.entries and previous.entries[name].get("digest") == digests[name]
        ]
        try:
            copied = previous.read_arrays(same)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            logger.warning("could not reuse lexical shards from %s: %s", previous_dir, e)

    shards: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for name, member_rows in groups.items():
        rows = np.asarray(member_rows, dtype=np.int32)
        arrays = copied.get(name)
        if arrays is not None and previous is not None:
            meta = {k: v for k, v in previous.entries[name].items() if k != "key"}
            shards[name] = ({**arrays, "rows": rows}, meta)
            continue
        index = builder.build([records[row].text for row in member_rows])
        shards[name] = (
            {**index.to_arrays(), "rows": rows},
            {**index.meta(), "digest": digests[name]},
        )

    save_lexical_pack(directory, shards, lexical_version=lexical_version)
    logger.info(
        "Wrote lexical index for %s: %d shard(s) (%d reused), %d record(s)",
        directory.name, len(shards), len(copied), len(records),
    )
    return len(shards)


def _open_previous_pack(previous_dir: Path | None) -> LexicalPack | None:
    """前の版の pack (無い / 読めなければ ``None``)。"""
    if previous_dir is None or not (previous_dir / SHARD_MAP_FILE).exists():
        return None
    try:
        return LexicalPack(previous_dir)
    except (OSError, ValueError) as e:
        logger.warning("ignoring the lexical pack of %s: %s", previous_dir, e)
        return None


def _single_shard(_record: Evidence) -> str:
    """既定のシャード鍵 — 全レコードを 1 シャードに入れる。"""
    return DEFAULT_SHARD


def is_active(record: Evidence, now_epoch: float, include_private: bool = False) -> bool:
    """1 レコードのアクティブ判定 (カラム版と同じ規則、c_16 §7.3-1)。

    未知の列挙値の行は使わない (c_05 §0.4.5 の読み手の規則。関門はここと
    ``active_mask`` だけ)。
    """
    if record.ignored:
        return False
    if record.veracity == "retracted" or record.superseded_by:
        return False
    if record.confidentiality == "secret":
        return False
    if record.private and not include_private:
        return False
    valid_until = parse_utc(record.valid_until)
    return not (valid_until is not None and valid_until.timestamp() <= now_epoch)


__all__ = [
    "CANDIDATE_MULTIPLIER",
    "EMBEDDING_VERSIONS_KEEP",
    "DEFAULT_EMBED_MODE",
    "DEFAULT_SHARD",
    "EMBED_AS_QUERY_ATTR",
    "EMBED_BATCH_SIZE",
    "EMBED_MODE_ATTR",
    "LEXICAL_DEFAULTS",
    "MIN_CANDIDATES",
    "EvidenceStore",
    "EvidenceStoreReadonlyError",
    "UsageBuffer",
    "embedding_version_name",
    "embedding_version_seq",
    "embed_reuse_key",
    "embed_side_key",
    "embed_side_of",
    "is_active",
]
