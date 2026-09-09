"""`EvidenceStore` — 事象ログ + 版付き snapshot + ベクトル索引の基盤 (c_16 §2.1 / §5 / §6.1)

読み手は **active な snapshot と、それ以降の事象だけ** を見る。書き手
(sleep-time) は事象を追記するだけで、稼働中の snapshot / 索引には一切触れない。
索引の作り直しは :meth:`EvidenceStore.create_snapshot` の中でだけ起きる。

```
<store_dir>/
├── manifest.json              §5.1 (fsync)
├── events/<yyyy-mm>.jsonl     §5.2 追記のみ
├── snapshot/v<N>/             §5.3 records.jsonl / offsets.npy / columns.npz
│   └── lexical/               §6.2 転置索引 (シャード別) + shards.json
└── embeddings/<model_id>/     §6.1 VectorStore (int8 + memmap + cluster index)
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
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.rag.evidence.columns import active_mask as columns_active_mask
from backend.free.rag.evidence.events import EventPosition, EvidenceEventLog
from backend.free.rag.evidence.lexical_index import (
    LexicalIndex,
    LexicalIndexBuilder,
    LexicalShardSet,
)
from backend.free.rag.evidence.manifest import EvidenceManifest
from backend.free.rag.evidence.ranking import (
    RankColumns,
    collapse,
    gate_by_cosine,
    merge_candidates,
    score_rows,
)
from backend.free.rag.evidence.snapshot import (
    SnapshotReader,
    SnapshotWriter,
    list_versions,
    prune_snapshots,
    read_snapshot,
    snapshot_dir,
    version_seq,
)
from backend.free.rag.evidence.types import (
    RECORD_VERSION,
    Evidence,
    EvidenceVersionError,
)
from backend.free.rag.vector_store import (
    DEFAULT_MEMMAP_THRESHOLD,
    VectorStore,
    content_hash,
    dequantize_int8,
    quantize_int8,
)
from backend.io import AtomicWriter, JSONLAppendStore, atomic_write_text
from backend.log_config import get_logger
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

#: 転置索引を置く snapshot 版配下のディレクトリ (c_16 §6.2)。
LEXICAL_DIR = "lexical"

#: シャード名 → ディレクトリ名 / 件数の対応表。
SHARDS_FILE = "shards.json"

#: 物理 GC の監査ログ (1 版 1 行の追記、c_16 §5.4)。
GC_LOG_FILE = "gc.jsonl"

#: :data:`GC_LOG_FILE` の行の版。行の形を変えるときに上げる。
GC_LOG_SCHEMA_VERSION = 1

#: 各シャードの「ローカル行 → snapshot のグローバル行」写像 (int32)。
SHARD_ROWS_FILE = "rows.npy"

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

#: シャード名をディレクトリ名へ落とすときに残す文字。
_SAFE_SHARD_RE = re.compile(r"[^A-Za-z0-9._-]+")


class EvidenceStoreReadonlyError(RuntimeError):
    """ストアが readonly (書けば壊す状態) なのに書き込みが試みられた。

    c_05 §0.5.1 の「新しい版のファイルは読まず書き戻しも拒否」を、
    ``manifest.json`` が読めない / 版が新しい / レコードの ``_version`` が
    新しい場合へ広げたもの (c_16 §5.1)。既定値のまま「空のストア」として
    動き出すと、``create_snapshot`` が v0001 を上書きし、``prune_snapshots``
    が本物の最新版を消す。
    """


def embedding_version_name(seq: int) -> str:
    """埋め込み索引の版番号 → ディレクトリ名 (``v0003``)。"""
    return _EMBED_VERSION_FORMAT.format(int(seq))


def embedding_version_seq(name: str) -> int | None:
    """埋め込み索引のディレクトリ名 → 版番号。読めなければ ``None``。"""
    match = _EMBED_VERSION_RE.fullmatch(str(name).strip())
    return int(match.group(1)) if match else None


def shard_dirname(name: str) -> str:
    """シャード名 → ディレクトリ名 (パス安全化)。

    シャード鍵は呼出側 (corpus / episodic / semantic のラッパ) が決める任意の
    文字列 (``short×2026-09`` / ``project:x`` / ``mem.personal``) なので、その
    ままディレクトリ名にはできない。安全な文字だけ残したうえで、**元の名前の
    sha256 先頭 8 桁を必ず付ける** — 記号を潰した結果が衝突しても別ディレクトリ
    になるようにするため。元の名前は :data:`SHARDS_FILE` 側が保持する。
    """
    slug = _SAFE_SHARD_RE.sub("_", name).strip("_")[:32] or "shard"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


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
        store_dir: ストアのルート (``local/memory/<store>``)。
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

        self._snapshot: SnapshotReader | None = None
        #: 事象で作られた / 上書きされたレコード (snapshot より優先)。
        self._overlay: dict[str, Evidence] = {}
        #: snapshot に無い = 次の snapshot まで **ベクトル検索に出ない** id。
        self._tail_ids: list[str] = []
        self._vector_store: VectorStore | None = None
        #: snapshot の行 → VectorStore の行 (無い行は -1)。
        self._vector_rows: np.ndarray | None = None
        #: active snapshot の転置索引 (lazy load)。
        self._lexical: LexicalShardSet | None = None
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
        self._overlay = {}
        self._tail_ids = []
        self._replay_events()
        self._vector_store = None
        self._vector_rows = None
        self._lexical = None

    def _recover_manifest(self, manifest_existed: bool) -> None:
        """manifest が読めなかったときに版ディレクトリから状態を復元する。

        - 新しい ``schema_version`` (``JsonStateFile`` が拒否した) → readonly。
          読むだけは続けられるよう active 版は最新の版ディレクトリを指す
        - 破損 / 消失 → 版ディレクトリの最大版を active、``next_snapshot_seq``
          はその +1、``folded_through`` は未定なので先頭へ戻す (事象は冪等に
          畳み直せる)
        - 版が 1 つも無く manifest も無い → 新規ストア (通常起動)
        - 版はあるが版名が読めない → readonly
        """
        versions = list_versions(self.store_dir)
        if self.manifest.readonly:
            if versions:
                self.manifest.active_snapshot = versions[-1]
            self._enter_readonly(
                f"{self.manifest.path.name} has a newer schema_version",
            )
            return
        if not manifest_existed and not versions:
            return  # 新規ストア
        if not versions:
            self._enter_readonly(
                f"{self.manifest.path.name} is unreadable and there is no "
                "snapshot version to recover from",
            )
            return
        seqs = [seq for seq in (version_seq(v) for v in versions) if seq is not None]
        if not seqs:
            self._enter_readonly(
                f"{self.manifest.path.name} is unreadable and the snapshot "
                f"version names cannot be parsed: {', '.join(versions)}",
            )
            return
        self.manifest.active_snapshot = versions[-1]
        self.manifest.next_snapshot_seq = max(seqs) + 1
        self.manifest.folded_through = EventPosition()
        logger.warning(
            "Recovered the %s manifest from snapshot versions: active=%s, "
            "next_snapshot_seq=%d (events are replayed from the beginning)",
            self.store_name, self.manifest.active_snapshot,
            self.manifest.next_snapshot_seq,
        )

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
        if self._snapshot is not None:
            for row in range(len(self._snapshot)):
                record_id = self._snapshot.id_at(row)
                overlaid = self._overlay.get(record_id)
                if overlaid is not None:
                    yield overlaid
                    continue
                record = self._snapshot.record_at(row)
                if record is not None:
                    yield record
        for record_id in self._tail_ids:
            record = self._overlay.get(record_id)
            if record is not None:
                yield record

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

    def put(self, record: Evidence, *, by: str | None = None) -> Evidence:
        """レコードを ``put`` する (追加 / 全置換)。

        Raises:
            EvidenceStoreReadonlyError: ストアが readonly のとき (c_05 §0.5.1)。
            RuntimeError: 別の書き手が同時に書いているとき。
        """
        self._refuse_write(f"put({record.id})")
        with self._exclusive_write(f"put({record.id})"):
            event = self.events.append_put(record.to_record(), by=by)
            self._apply_event(event)
            self.manifest.events_since_snapshot += 1
            return self._overlay.get(record.id, record)

    def patch(
        self, record_id: str, *, by: str | None = None, **fields: Any,
    ) -> Evidence | None:
        """変更フィールドだけを ``patch`` する (c_16 §5.2)。"""
        self._refuse_write(f"patch({record_id})")
        if self.get(record_id) is None:
            logger.warning("patch on unknown evidence id: %s", record_id)
            return None
        with self._exclusive_write(f"patch({record_id})"):
            event = self.events.append_patch(record_id, fields, by=by)
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
        return str(_backend_value(backend, "model_name", _UNKNOWN_MODEL))

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
        latest = self._latest_embedding_version(root)
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
    def _latest_embedding_version(root: Path) -> str:
        """``embeddings/<model_id>/`` 配下の最大版名 (無ければ空文字)。"""
        if not root.is_dir():
            return ""
        names = [
            path.name for path in root.iterdir()
            if path.is_dir() and embedding_version_seq(path.name) is not None
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
        if self.embedding_backend is None or self._snapshot is None:
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
        previous: VectorStore | None = None
        if previous_version and (model_root / previous_version).is_dir():
            previous = self._open_vector_store(model_root / previous_version)
        reusable = (
            self._reusable_rows(previous, model_id) if previous is not None else {}
        )

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

        # 流用する行は **先に** 実配列へ複写する。memmap のまま抱えたまま同じ
        # ファイルへ np.save すると Windows でロックに当たる。
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
        # 前の版の memmap はここで手放す (Windows は掴んだままだと消せない)。
        if previous is not None:
            previous.vectors_q8 = None
            previous.scales = None

        new_q8: np.ndarray | None = None
        new_scales: np.ndarray | None = None
        if pending:
            vectors = await self._embed_all(
                [(texts[row], *sides[row]) for row in pending],
            )
            new_q8, new_scales = quantize_int8(np.asarray(vectors, dtype=np.float32))
            new_scales = new_scales.reshape(-1, 1)

        dim = int(new_q8.shape[1]) if new_q8 is not None else prev_dim
        q8 = np.zeros((len(ids), dim), dtype=np.int8)
        scales = np.zeros((len(ids), 1), dtype=np.float32)
        if reused_q8 is not None:
            dst = np.asarray(reuse_dst, dtype=np.int64)
            q8[dst] = reused_q8
            scales[dst] = reused_scales
        if new_q8 is not None:
            dst = np.asarray(pending, dtype=np.int64)
            q8[dst] = new_q8
            scales[dst] = new_scales

        version = self._take_next_embedding_version(model_root)
        logger.info(
            "Embedded %d row(s), reused %d of %d for %s (%s, %s)",
            len(pending), len(reuse_dst), len(ids), self.store_name,
            model_id, version,
        )
        store = VectorStore(
            model_root / version,
            memmap_threshold=self._rag_int("memmap_threshold", DEFAULT_MEMMAP_THRESHOLD),
            quantization=self._rag_str("quantization", "int8"),
        )
        self._write_vector_store(store, ids, hashes, sides, q8, scales)
        self._vector_store = store
        self._vector_rows = None
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
        for row in range(len(snapshot)):
            raw = snapshot.raw_at(row)
            if raw is None:
                continue
            ids.append(str(raw.get("id") or snapshot.id_at(row)))
            texts.append(str(raw.get("text") or ""))
            sides.append(embed_side_of(raw.get("attrs")))
        return ids, texts, sides

    def _reusable_rows(
        self, store: VectorStore, model_id: str,
    ) -> dict[str, tuple[int, str]]:
        """前の版の VectorStore から ``id → (行, text_hash)`` を作る。

        以下のいずれかに当たれば空を返す = 全件埋め込み直し:

        - manifest が別の埋め込みモデルを宣言している (モデル切替)
        - ベクトルが無い / 次元がバックエンドの宣言と食い違う
        """
        declared = self.manifest.embedding_model_id
        if declared and declared != model_id:
            logger.info(
                "Embedding model changed (%s -> %s): full re-embed for %s",
                declared, model_id, self.store_name,
            )
            return {}
        if store.vectors_q8 is None or store.scales is None:
            return {}
        if len(store.vectors_q8) == 0 or store.vectors_q8.ndim != 2:
            return {}
        expected_dim = int(_backend_value(self.embedding_backend, "dim", 0) or 0)
        stored_dim = int(store.vectors_q8.shape[1])
        if expected_dim and stored_dim != expected_dim:
            logger.warning(
                "Embedding dim changed (%d -> %d) for %s: full re-embed",
                stored_dim, expected_dim, self.store_name,
            )
            return {}
        usable = min(len(store.metadata), len(store.vectors_q8))
        reusable: dict[str, tuple[int, str]] = {}
        for row in range(usable):
            meta = store.metadata[row]
            record_id = str(meta.get("id") or "")
            digest = meta.get("text_hash")
            if record_id and isinstance(digest, str) and digest:
                reusable[record_id] = (row, digest)
        return reusable

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

    def _open_vector_store(self, directory: Path) -> VectorStore:
        """``embeddings/<model_id>/`` を開く (無ければ空で返る)。"""
        store = VectorStore(
            directory,
            memmap_threshold=self._rag_int("memmap_threshold", DEFAULT_MEMMAP_THRESHOLD),
            quantization=self._rag_str("quantization", "int8"),
        )
        store.load()
        return store

    def _write_vector_store(
        self,
        store: VectorStore,
        ids: list[str],
        hashes: list[str],
        sides: list[tuple[bool, str]],
        q8: np.ndarray,
        scales: np.ndarray,
    ) -> VectorStore:
        """組み上がった int8 行列を ``embeddings/<model_id>/`` へ書く。

        ``VectorStore.add_vectors`` は使わない — あちらは位置採番の chunk id を
        振り、本文を ``chunks/<id>.txt`` に複製する。ここでは行 id を
        **evidence id** にし、本文は ``records.jsonl`` から lazy に読むので
        複製しない。``text_hash`` は次回の増分埋め込みが「本文か側が変わったか」
        を判定する唯一の根拠なので必ず持たせる (``embed_side`` はその内訳を
        後から読めるようにする観測用)。
        """
        now = utc_now()
        store.vectors_q8 = q8
        store.scales = scales
        store.metadata = [
            {
                "id": record_id,
                "source": self.manifest.active_snapshot,
                "chunk_index": row,
                "created_at": now,
                "category": "evidence",
                "has_context": True,
                "text_hash": hashes[row],
                "embed_side": embed_side_key(*sides[row]),
            }
            for row, record_id in enumerate(ids)
        ]
        backend = self.embedding_backend
        store.mark_reindexed(
            self.embedding_model_id,
            str(_backend_value(backend, "backend_type", "")),
            int(q8.shape[1]) if q8.ndim == 2 else 0,
        )
        store.save()
        if self._cluster_enabled():
            # threshold=1: 件数に関わらず必ず構築する (c_16 §6.1)。numpy だけの
            # K-means なので、増分埋め込みでも毎回作り直してよい。
            store.build_cluster_index(
                threshold=1,
                n_probe_ratio=self._cluster_n_probe_ratio(),
            )
        return store

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

    def vector_store(self) -> VectorStore | None:
        """ベクトル索引 (無ければ ``None``)。初回アクセスで lazy load。

        読むのは manifest の ``embedding_version`` が指す版だけ。指し先が
        無ければディスク上の最大版へ落とす (manifest を復元した直後)。
        """
        if self._vector_store is not None:
            return self._vector_store
        directory = self.embeddings_dir(
            self.manifest.embedding_model_id or self.embedding_model_id,
        )
        if not directory.is_dir():
            return None
        self._vector_store = self._open_vector_store(directory)
        return self._vector_store

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
        positions = {
            str(meta.get("id") or ""): row for row, meta in enumerate(store.metadata)
        }
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

    def lexical_shards(self) -> LexicalShardSet | None:
        """active snapshot の転置索引を lazy load する (無ければ ``None``)。"""
        if self._lexical is not None:
            return self._lexical
        if self._snapshot is None:
            return None
        root = self._snapshot.directory / LEXICAL_DIR
        map_path = root / SHARDS_FILE
        if not map_path.exists():
            return None
        try:
            raw: Any = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("unreadable lexical shard map %s: %s", map_path, e)
            return None
        entries = raw.get("shards") if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            logger.warning("lexical shard map %s has no shards", map_path)
            return None

        shards: dict[str, LexicalIndex] = {}
        rows: dict[str, np.ndarray] = {}
        skipped = 0
        for name, entry in entries.items():
            directory = str((entry or {}).get("dir") or "") if isinstance(entry, dict) else ""
            shard_dir = root / directory
            rows_path = shard_dir / SHARD_ROWS_FILE
            if not directory or not rows_path.exists():
                skipped += 1
                continue
            try:
                index = LexicalIndex.load(shard_dir)
                mapping = np.load(str(rows_path), allow_pickle=False)
            except (OSError, ValueError, KeyError) as e:
                logger.warning("failed to load lexical shard %s: %s", name, e)
                skipped += 1
                continue
            self._warn_lexical_param_drift(str(name), index)
            shards[str(name)] = index
            rows[str(name)] = mapping.astype(np.int64, copy=False)
        if skipped:
            logger.warning("%d lexical shard(s) skipped under %s", skipped, root)
        if not shards:
            return None
        try:
            self._lexical = LexicalShardSet(shards, rows)
        except ValueError as e:
            logger.error("lexical shard set inconsistent at %s: %s", root, e)
            return None
        return self._lexical

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
        shard_set = self.lexical_shards()
        if shard_set is None or top_k <= 0:
            return empty
        names = list(shards) if shards is not None else list(shard_set.shards)
        if not names:
            return empty
        return shard_set.candidates_in(
            names, query, top_k,
            budget_ms=self._lexical_budget_ms(),
            row_mask=row_mask,
        )

    def _write_lexical_index(
        self, directory: Path, records: Sequence[Evidence],
    ) -> int:
        """版ディレクトリ配下へシャード別の転置索引を書く (c_16 §6.2)。

        ``records`` の並びは ``records.jsonl`` の行順と一対一 (同じ列を
        ``write_snapshot`` に渡している)。シャードごとの索引はローカル行で
        引かれるので、``rows.npy`` に **ローカル行 → グローバル行** を持たせて
        :class:`LexicalShardSet` を復元できるようにする。

        Returns:
            書いたシャード数。
        """
        groups: dict[str, list[int]] = {}
        for row, record in enumerate(records):
            try:
                key = str(self.shard_key_for(record) or DEFAULT_SHARD)
            except Exception as e:  # noqa: BLE001 — 索引作成で snapshot を落とさない
                logger.warning("shard_key_for failed for %s: %s", record.id, e)
                key = DEFAULT_SHARD
            groups.setdefault(key, []).append(row)

        root = directory / LEXICAL_DIR
        root.mkdir(parents=True, exist_ok=True)
        builder = self._lexical_builder()
        entries: dict[str, dict[str, Any]] = {}
        for name, member_rows in groups.items():
            index = builder.build([records[row].text for row in member_rows])
            dirname = shard_dirname(name)
            shard_dir = root / dirname
            index.save(shard_dir)
            with AtomicWriter(shard_dir / SHARD_ROWS_FILE, mode="wb") as f:
                np.save(f, np.asarray(member_rows, dtype=np.int32))
            entries[name] = {"dir": dirname, "n_docs": len(member_rows)}

        atomic_write_text(
            root / SHARDS_FILE,
            json.dumps(
                {
                    "lexical_version": int(self.manifest.lexical_version),
                    "shards": entries,
                },
                ensure_ascii=False, indent=2,
            ),
        )
        logger.info(
            "Wrote lexical index for %s: %d shard(s), %d record(s)",
            directory.name, len(entries), len(records),
        )
        return len(entries)

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

    # ── snapshot 生成 (c_16 §5.3) ──

    async def create_snapshot(self, now: str | None = None) -> str:
        """事象を畳んで新しい版を作り、最後に manifest を切り替える。

        手順 (順序が意味を持つ):

        1. 事象ログの末尾位置を **先に** 確定する (畳み込み中の追記を
           「畳み済み」と誤記録しないため)
        2. 前 snapshot + その位置までの事象を畳み、``gc_filter`` が ``False``
           を返した行を落とす (物理 GC。c_16 §5.4 / :meth:`_apply_gc_filter`)
        3. 新しい版ディレクトリへ ``records.jsonl`` / ``offsets.npy`` /
           ``columns.npz`` を書く
        4. 同じ版ディレクトリへ転置索引 (``lexical/<shard>/``) を書く
        5. 埋め込み (増分) + クラスタ索引を作る
        6. 古い版 / 古い事象ファイルを刈る (旧 active と新版は保護)
        7. **最後に** manifest を書き替えて active を切り替える

        7 を最後にするのは、途中でクラッシュしても manifest が生きている版を
        指したままにするため。6 で旧 active を消さないのも同じ理由。

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
        events = list(self.events.iter_since(self.manifest.folded_through, until=end))
        prev = list(self._snapshot.iter_records()) if self._snapshot is not None else []
        records = SnapshotWriter.fold(prev, events)
        records, dropped_ids = self._apply_gc_filter(records)

        old_active = self.manifest.active_snapshot
        folded = len(events)
        version = self.manifest.take_next_version()
        # 版番号は **書き始める前に** 永続化する。ここで落ちると (taskkill /F)
        # 次の起動が同じ番号を再発番し、書きかけの版を別内容で上書きする。
        # active_snapshot はまだ旧版のままなので、途中で落ちても読み手は
        # 生きた版を指したままになる。
        self.manifest.save()
        SnapshotWriter.write_snapshot(self.store_dir, version, records)
        if dropped_ids:
            self._write_gc_log(version, dropped_ids)
        # 転置索引は畳み込み済みの ``records`` から作る。``write_snapshot`` へ
        # 渡したのと同じ列なので、行番号は records.jsonl と一対一。
        shard_count = self._write_lexical_index(
            snapshot_dir(self.store_dir, version), records,
        )

        # 新しい版を読み直してから索引を作る (行番号は新 snapshot 基準)。
        self._snapshot = read_snapshot(self.store_dir, version)
        self._overlay = {}
        self._tail_ids = []
        self._vector_store = None
        self._vector_rows = None
        self._lexical = None
        indexed = await self.embed_and_index_snapshot(defer_manifest=True)

        prune_snapshots(
            self.store_dir,
            int(self.manifest.retention_value("snapshots_keep")),
            protect={old_active, version},
        )
        self.events.prune(
            int(self.manifest.retention_value("events_keep_months")),
            folded_through=end,
        )

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
        logger.info(
            "Created snapshot %s for %s: %d record(s), %d indexed, %d lexical shard(s), "
            "folded through %s:%d (at %s)",
            version, self.store_name, len(records), indexed, shard_count,
            end.month or "-", end.line, now or utc_now(),
        )
        return version

    def _apply_gc_filter(
        self, records: list[Evidence],
    ) -> tuple[list[Evidence], list[str]]:
        """畳み込み結果に物理 GC を掛ける (c_16 §5.4)。

        フィルタが ``False`` を返したレコードは新しい版に書かれない = ディスク
        から消える。判定中の例外はレコードを **残す** 側に倒す (GC の誤りで
        生きた記録を落とすより、消し損ねるほうが安い)。

        Returns:
            ``(残すレコード, 落とした id)``。
        """
        gc_filter = self.gc_filter
        if gc_filter is None:
            return records, []
        kept: list[Evidence] = []
        dropped: list[str] = []
        for record in records:
            try:
                keep = bool(gc_filter(record))
            except Exception as e:
                logger.warning(
                    "gc_filter raised on %s (%s), keeping the record: %s",
                    record.id, self.store_name, e,
                )
                keep = True
            if keep:
                kept.append(record)
            else:
                dropped.append(record.id)
        return kept, dropped

    def _write_gc_log(self, version: str, dropped_ids: list[str]) -> None:
        """落とした id を ``gc.jsonl`` と manifest の ``_extra`` に残す。

        レコードそのものは消えるので、**消えた事実だけは追記で残す** のが
        c_16 §5.4 の「物理削除は保持方針の GC のみ」に対する監査の担保。
        1 版 1 行 (エンベロープは c_05 §0.5)。
        """
        entry = {
            "schema_version": GC_LOG_SCHEMA_VERSION,
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

    def _warn_lexical_param_drift(self, name: str, index: LexicalIndex) -> None:
        """索引に焼き付いたパラメータと現在の設定の食い違いを 1 度だけ警告する。

        ``q_terms`` / ``m_postings`` / ``max_df_ratio`` は **索引を作った時点で
        凍る** (c_16 §6.2)。``max_df_ratio`` は剪定そのもの、``m_postings`` は
        posting の切り詰め位置に効くので、設定を変えても次の snapshot まで
        検索の挙動は変わらない。「変えたのに効かない」を黙らせない。
        """
        if self._lexical_drift_warned:
            return
        stored = index.params
        current = {
            "q_terms": int(self._lexical_setting("q_terms")),
            "m_postings": int(self._lexical_setting("m_postings")),
            "max_df_ratio": float(self._lexical_setting("max_df_ratio")),
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


def _single_shard(_record: Evidence) -> str:
    """既定のシャード鍵 — 全レコードを 1 シャードに入れる。"""
    return DEFAULT_SHARD


def is_active(record: Evidence, now_epoch: float, include_private: bool = False) -> bool:
    """1 レコードのアクティブ判定 (カラム版と同じ規則、c_16 §7.3-1)。"""
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
    "LEXICAL_DIR",
    "MIN_CANDIDATES",
    "SHARDS_FILE",
    "SHARD_ROWS_FILE",
    "EvidenceStore",
    "EvidenceStoreReadonlyError",
    "UsageBuffer",
    "embedding_version_name",
    "embedding_version_seq",
    "embed_reuse_key",
    "embed_side_key",
    "embed_side_of",
    "is_active",
    "shard_dirname",
]
