"""`SemanticStore` — 構造化事実の唯一の永続層 (c_16 §4.2)。

``local/memory/semantic/`` に :class:`~backend.free.rag.evidence.EvidenceStore`
を **1 つだけ** 持ち、旧 ``SemanticFactStore`` のスコープ別ディレクトリ
(``global/`` / ``projects/<id>/``) を ``Evidence.scope`` フィールドへ畳む。

```
<memory_dir>/semantic/
├── manifest.json                 EvidenceManifest (§5.1)
├── events/<yyyy-mm>.jsonl        追記のみ (§5.2)
├── snapshot/v<N>/                records / offsets / columns / lexical (§5.3)
├── embeddings/<model_id>/        VectorStore (§6.1)
├── sources.jsonl                 know.* の取得元 (ks_、§4.2)
└── items.jsonl                   know.* の取得単位 (ki_、§4.2)
```

シャードは **namespace** (``mem`` / ``know`` / ``idx`` / ``loop`` / ``learn``、
c_16 §6.2)。競合の勝ち方・減衰・注入先も namespace で決まる
(:mod:`~backend.free.memory.semantic.namespaces`)。

## 誰が書くか

書き手は sleep-time (``SleepTimeWorker``) だけ (CLAUDE.md §6 #2)。例外は
``artifact`` ファクト (ラルフループの即時書込) の 1 つで、
:meth:`add_fact` がそのまま ``put`` 事象を追記する — snapshot は作らないので、
次の sleep-time までは :meth:`get_fact` / :meth:`all_facts` からは見えるが
ベクトル検索には出ない (c_16 §2.1 の「稼働中の索引は書き換えない」の帰結)。

## 在メモリの写し

``all_facts`` / ``search_by_type`` / 競合検出 / GC は **全件を列挙する**
消費者なので、事象と snapshot から畳んだ現在状態を
:class:`SemanticFact` の dict として在メモリに持つ (旧ストアと同じ形)。
ノート (数万件) と違ってファクトは数百〜数千件で、しかも消費者が
「全部を見る」前提で書かれている。行を選んでから本文を読む episodic の
やり方 (c_16 §5.3) はここでは費用が合わない。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.memory.semantic.fact import (
    FactRecordError,
    evidence_to_fact,
    fact_to_evidence,
)
from backend.free.memory.semantic.namespaces import (
    know_half_life_days,
    namespace_of,
    policy_for,
)
from backend.free.memory.semantic.subject_key import is_generic_subject
from backend.free.memory.semantic.sources import (
    ITEMS_FILENAME,
    SOURCES_FILENAME,
    ItemRegistry,
    SourceRegistry,
)
from backend.free.memory.types import FactType, SemanticFact
from backend.free.rag.evidence import (
    Evidence,
    EvidenceStore,
    UsageBuffer,
    list_versions,
    read_snapshot,
)
from backend.free.rag.evidence.ranking import RankColumns, score_rows
from backend.free.rag.vector_store import dequantize_int8
from backend.log_config import get_logger
from backend.utils import parse_utc, utc_now_dt

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.semantic.store")

#: ``store_prior`` を ``semantic_know`` で引く namespace (c_16 §7.2)。
KNOW_NAMESPACE = "know"

#: ``memory.evidence.ranking.store_prior`` が読めないときの既定 (c_16 §9)。
DEFAULT_SEMANTIC_MEM_PRIOR = 1.0
DEFAULT_SEMANTIC_KNOW_PRIOR = 0.9


def _resolve_store_priors(rag_config: Any) -> tuple[float, float]:
    """``(semantic_mem, semantic_know)`` を設定から解決する。

    ``rag_config`` は ``merge_rag_evidence_config`` が ``rag`` に
    ``memory.evidence.ranking`` を重ねた dict (または属性アクセスできる
    オブジェクト)。読めなければ c_16 §9 の既定へ落ちる。
    """
    def _get(section: Any, key: str) -> Any:
        if section is None:
            return None
        return (
            section.get(key) if isinstance(section, dict)
            else getattr(section, key, None)
        )

    priors = _get(_get(rag_config, "ranking"), "store_prior")
    out: list[float] = []
    for key, default in (
        ("semantic_mem", DEFAULT_SEMANTIC_MEM_PRIOR),
        ("semantic_know", DEFAULT_SEMANTIC_KNOW_PRIOR),
    ):
        value = _get(priors, key)
        try:
            out.append(float(value) if value is not None else default)
        except (TypeError, ValueError):
            out.append(default)
    return out[0], out[1]

#: ストアのディレクトリ名 (``local/memory/semantic``)。
STORE_DIRNAME = "semantic"

#: 事象ログの ``by`` (書き手コンポーネント)。
WRITER = "sleep_time.semantic"

#: 3 pillar namespace の subject 前方一致索引対象 prefix。
PILLAR_SUBJECT_PREFIXES: tuple[str, ...] = ("loop.", "learn.", "mem.")

#: ``know.*`` の失効判定に使う鮮度の下限 (c_16 §5.4)。
KNOW_FRESHNESS_FLOOR = 0.05

#: superseded / retracted を物理 GC してよくなるまでの版数 (c_16 §5.4)。
SUPERSEDED_GC_SNAPSHOTS = 3


def semantic_shard_key(record: Evidence) -> str:
    """``Evidence`` → 転置索引のシャード名 = namespace (c_16 §6.2)。"""
    structured = record.structured if isinstance(record.structured, dict) else {}
    return namespace_of(str(structured.get("subject") or ""))


class SemanticHit:
    """検索 1 件 (ファクト / 素の cosine / 順位式スコア)。

    ``cosine`` はゲート用 (c_16 §7.1 「ゲートは素の cosine のみ」)、``score``
    は順位用 (``cos × freshness × confidence × store_prior``)。2 つを分けて
    持つのは、閾値が cosine スケール前提で決まっているため。
    """

    __slots__ = ("cosine", "fact", "score")

    def __init__(self, fact: SemanticFact, cosine: float, score: float) -> None:
        self.fact = fact
        self.cosine = float(cosine)
        self.score = float(score)

    @property
    def id(self) -> str:
        return self.fact.id


class SemanticStore:
    """構造化事実 (kind=``fact`` / ``claim``) のストア。

    Args:
        memory_dir: ``local_paths.memory_dir``。実体は ``<memory_dir>/semantic``。
        embedding_backend: 埋め込みバックエンド。``None`` なら snapshot 生成時に
            ベクトル索引を作らない (語彙索引だけの縮退動作)。
        rag_config: ``config.yaml`` の ``rag`` セクション (量子化 / memmap /
            クラスタ索引 / ``lexical``)。
        retention: ``memory.evidence.retention`` (c_16 §9)。manifest へ宣言する。
        know_half_life: ``memory.evidence.know_half_life_days`` (c_16 §9)。
        debug_logger: JSONL 観測用 (memory カテゴリ)。
    """

    def __init__(
        self,
        memory_dir: Path | str,
        embedding_backend: "EmbeddingBackend | None" = None,
        rag_config: Any = None,
        *,
        retention: dict[str, Any] | None = None,
        know_half_life: dict[str, Any] | None = None,
        debug_logger: Any = None,
    ) -> None:
        self.store_dir = Path(memory_dir) / STORE_DIRNAME
        self.evidence = EvidenceStore(
            self.store_dir,
            store_name="semantic",
            embedding_backend=embedding_backend,
            rag_config=rag_config,
            by=WRITER,
            shard_key_for=semantic_shard_key,
            gc_filter=self._keep_in_snapshot,
        )
        self.sources = SourceRegistry(self.store_dir / SOURCES_FILENAME)
        self.items = ItemRegistry(self.store_dir / ITEMS_FILENAME)
        self._debug_logger = debug_logger
        self._retention_override = dict(retention or {})
        self._know_half_life = dict(know_half_life or {})
        # 順位式の store_prior (c_16 §7.2)。namespace で mem / know を分ける
        # ため、スカラではなく **行ごとの配列** を作って渡す (§7.2 の
        # ``semantic_mem`` / ``semantic_know``)。
        self._prior_mem, self._prior_know = _resolve_store_priors(rag_config)
        #: ``(active_snapshot, len)`` → 行ごとの store_prior。
        self._prior_cache_key: tuple[str, int] | None = None
        self._prior_cache: np.ndarray | None = None

        #: 畳み込み済みの現在状態 (id → ファクト)。
        self._facts: dict[str, SemanticFact] = {}
        self._by_subject: dict[str, set[str]] = {}
        self._by_type: dict[str, set[str]] = {}
        self._by_namespace: dict[str, set[str]] = {}
        self._pinned: set[str] = set()
        #: 書込のたびに +1 する世代番号。呼出側が派生物 (live リスト /
        #: ベクトル行列) をキャッシュしてよいかの判定に使う。
        self._revision: int = 0
        #: cosine 行列のキャッシュ鍵 (active snapshot 版, 未畳み込み事象数)。
        self._vector_cache_key: tuple[str, int] | None = None
        self._vector_cache: dict[str, np.ndarray] | None = None
        #: 物理 GC の対象 id (版の生成 1 回につき 1 度だけ計算する)。
        self._gc_ids: set[str] | None = None
        self._gc_ids_key: str | None = None

    # ── ライフサイクル ──────────────────────────────────────────────

    def load(self) -> None:
        """manifest / snapshot / 事象 / sources / items を読む。"""
        self.evidence.load()
        if self._retention_override:
            self.evidence.manifest.retention.update(self._retention_override)
        self.sources.load()
        self.items.load()
        self._rebuild_from_evidence()
        logger.info(
            "Semantic store loaded: %d fact(s), snapshot=%s, %d event(s) pending",
            len(self._facts),
            self.evidence.manifest.active_snapshot or "(none)",
            self.evidence.manifest.events_since_snapshot,
        )

    def _rebuild_from_evidence(self) -> None:
        """``EvidenceStore`` の現在状態から在メモリの写しを作り直す。

        読めないレコード (``attrs.fact_type`` 欠損等) は **そのレコードだけ**
        飛ばして件数を WARNING に出す (c_05 §0.5.2)。
        """
        self._facts = {}
        self._by_subject = {}
        self._by_type = {}
        self._by_namespace = {}
        self._pinned = set()
        skipped = 0
        for record in self.evidence.iter_records():
            if record.kind not in ("fact", "claim"):
                continue
            if record.veracity == "retracted":
                continue
            try:
                fact = evidence_to_fact(record)
            except FactRecordError as e:
                skipped += 1
                logger.debug("skipping unreadable semantic record: %s", e)
                continue
            self._facts[fact.id] = fact
            self._add_to_indexes(fact)
        if skipped:
            logger.warning(
                "Semantic store: skipped %d unreadable record(s)", skipped,
            )
        self._break_supersede_cycles()
        self._invalidate_vectors()
        self._invalidate_gc_cache()

    def _break_supersede_cycles(self) -> int:
        """``superseded_by`` の閉路を解いて最新世代を live へ戻す。

        :meth:`supersede` が閉路を作らせないので、ここに来るのは事象ログが
        壊れたときだけ。それでも見張るのは、閉路の壊れ方が **値が消えるのでは
        なく「そのスロットの live が 0 件になる」** 形だからで、想起は静かに
        1 世代前の生テキストへ落ちる (2026-08-30 監査で ``mem.personal.birthday``
        の 4 件が 2-閉路を含む鎖で全滅した)。

        解き方は「閉路の中で ``created_at`` が最新のものの ``superseded_by`` を
        外す」— 現在値として最も妥当な 1 件を live に戻す。修復は ``patch``
        事象として書くので、次の版に畳まれて永続化される。
        """
        nexts = {
            f.id: f.superseded_by for f in self._facts.values() if f.superseded_by
        }
        # **閉路ごとに** 1 件ずつ live へ戻す。全部を 1 つの集合にまとめて
        # ``max`` を取ると、独立した 2 つの閉路のうち片方しか直らない。
        cycles: list[list[str]] = []
        visited: set[str] = set()
        for start in list(nexts):
            if start in visited:
                continue
            seen: list[str] = []
            cur: str | None = start
            while cur is not None and cur in nexts and cur not in seen:
                seen.append(cur)
                cur = nexts[cur]
            visited.update(seen)
            if cur is not None and cur in seen:
                cycles.append(seen[seen.index(cur):])
        if not cycles:
            return 0
        repaired = 0
        for cycle in cycles:
            members = [self._facts[fid] for fid in cycle if fid in self._facts]
            if not members:
                continue
            newest = max(members, key=lambda f: (f.created_at, f.id))
            self.update_fact(newest.id, touch=False, superseded_by=None)
            repaired += 1
            logger.warning(
                "Semantic store: broke a supersede cycle of %d record(s); "
                "restored %s (%s) as live",
                len(cycle), newest.id, newest.subject,
            )
        return repaired

    def close(self) -> None:
        """memmap を握った索引を手放す (Windows で削除できるように)。"""
        self.evidence.close()

    @property
    def usage(self) -> UsageBuffer:
        """チャット経路が「使った」id を溜めるバッファ (c_16 §2.1)。"""
        return self.evidence.usage

    @property
    def revision(self) -> int:
        """書込世代番号。add / update / supersede / retract のたびに増える。

        値そのものに意味は無く、**変わったかどうか** だけを見る。
        """
        return self._revision

    def __len__(self) -> int:
        return len(self._facts)

    # ── スコープ束縛 ────────────────────────────────────────────────

    def scoped(self, scope: str = "global") -> "ScopedSemanticStore":
        """1 スコープに束縛したビューを返す (``global`` / ``project:<id>``)。

        スコープは **フィールド** であってディレクトリではない (c_16 §4.2)。
        呼出側 (``AppState.get_semantic_store`` / sleep-time の
        ``store_provider``) は従来どおり「1 スコープ 1 ストア」の面で使える。
        """
        if scope != "global" and not scope.startswith("project:"):
            raise ValueError(
                f"unknown scope: {scope!r} (expected 'global' or 'project:<id>')",
            )
        if scope.startswith("project:") and not scope.split(":", 1)[1]:
            raise ValueError(f"invalid scope: {scope}")
        return ScopedSemanticStore(self, scope)

    # ── 索引 ────────────────────────────────────────────────────────

    def _add_to_indexes(self, fact: SemanticFact) -> None:
        self._by_subject.setdefault(fact.subject, set()).add(fact.id)
        self._by_type.setdefault(str(fact.type), set()).add(fact.id)
        self._by_namespace.setdefault(namespace_of(fact.subject), set()).add(fact.id)
        if fact.pinned:
            self._pinned.add(fact.id)

    def _remove_from_indexes(self, fact: SemanticFact) -> None:
        _discard(self._by_subject, fact.subject, fact.id)
        _discard(self._by_type, str(fact.type), fact.id)
        _discard(self._by_namespace, namespace_of(fact.subject), fact.id)
        self._pinned.discard(fact.id)

    def _invalidate_vectors(self) -> None:
        self._vector_cache = None
        self._vector_cache_key = None

    # ── 書き込み ────────────────────────────────────────────────────

    def add_fact(self, fact: SemanticFact) -> SemanticFact:
        """新規ファクトを ``put`` する。

        - ``id`` が未設定なら発番する (``ev_`` + hex12)
        - 既存 ID と衝突したら ``ValueError``
        - ``created_at`` / ``accessed_at`` が 0 なら現在時刻で埋める
        - ``know.<domain>`` は namespace 既定の半減期を載せる (c_16 §4.2)
        """
        if not fact.id:
            fact.id = SemanticFact.new_id()
        if fact.id in self._facts:
            raise ValueError(f"fact id already exists: {fact.id}")
        now = utc_now_dt().timestamp()
        if fact.created_at == 0.0:
            fact.created_at = now
        if fact.accessed_at == 0.0:
            fact.accessed_at = fact.created_at
        record = fact_to_evidence(
            fact,
            half_life_days=know_half_life_days(fact.subject, self._know_half_life),
        )
        self.evidence.put(record)
        self._facts[fact.id] = fact
        self._add_to_indexes(fact)
        self._revision += 1
        self._invalidate_vectors()
        logger.debug(
            "add_fact: id=%s subject=%s type=%s scope=%s",
            fact.id, fact.subject, fact.type, fact.scope,
        )
        return fact

    def add_fact_bulk(self, facts: Sequence[SemanticFact]) -> list[SemanticFact]:
        """複数ファクトを ``put`` する (アトミックではない)。"""
        return [self.add_fact(fact) for fact in facts]

    def put_record(self, record: Evidence) -> SemanticFact:
        """組み立て済みの ``Evidence`` をそのまま ``put`` する。

        ``know.*`` の claim (:class:`~backend.free.memory.semantic.sources.KnowledgeIngest`)
        のように、``SemanticFact`` では表せないコアフィールド (``valid_until`` /
        ``origin=web``) を持つレコード用の入口。
        """
        self.evidence.put(record)
        fact = evidence_to_fact(record)
        self._facts[fact.id] = fact
        self._add_to_indexes(fact)
        self._revision += 1
        self._invalidate_vectors()
        return fact

    @property
    def know_half_life(self) -> dict[str, Any]:
        """``memory.evidence.know_half_life_days`` の上書き表 (c_16 §9)。"""
        return dict(self._know_half_life)

    def get_fact(self, fact_id: str) -> SemanticFact | None:
        """ID でファクトを取得する。存在しなければ ``None``。"""
        return self._facts.get(fact_id)

    def update_fact(
        self,
        fact_id: str,
        *,
        touch: bool = True,
        flush_embedding: bool = True,  # noqa: ARG002 — 呼出側互換 (版で一括保存)
        **changes: Any,
    ) -> SemanticFact:
        """既存ファクトのフィールドを差分更新する (``patch`` 事象)。

        Args:
            touch: ``accessed_at`` を現在時刻へ更新するか。既定 True。
                埋め込みの遡及生成のような **保守処理はアクセスではない** ため、
                False を渡して保持順を歪めないようにする。
            flush_embedding: 旧 API 互換のため受け取るだけ。ベクトルは snapshot
                生成時に増分で作られる (c_16 §6.1)。
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
        if touch:
            fact.accessed_at = utc_now_dt().timestamp()
        self._add_to_indexes(fact)
        # ``patch`` 事象は差分だけを書く。``subject`` / ``object`` /
        # ``fact_type`` のようなコア側の変更はレコード全体を組み直す必要が
        # あるので ``put`` で置き換える (同じ id への put は全置換)。
        record = fact_to_evidence(
            fact,
            half_life_days=know_half_life_days(fact.subject, self._know_half_life),
        )
        self.evidence.put(record)
        self._revision += 1
        self._invalidate_vectors()
        logger.debug("update_fact: id=%s changes=%s", fact_id, sorted(changes.keys()))
        return fact

    def retract_fact(self, fact_id: str, reason: str) -> bool:
        """``veracity=retracted`` にする (物理削除はしない、c_16 §3)。

        取り下げたファクトは読み出し系から消えるが、事象ログと snapshot には
        残るので監査で追える。物理削除は snapshot 3 版後の GC が担う
        (:meth:`pending_physical_gc`)。
        """
        fact = self._facts.pop(fact_id, None)
        if fact is None:
            return False
        self._remove_from_indexes(fact)
        self.evidence.retract(fact_id, reason)
        self._clear_dangling_supersession({fact_id})
        self._revision += 1
        self._invalidate_vectors()
        return True

    def delete_fact(self, fact_id: str) -> bool:
        """ファクトを取り下げる (旧 API 名。実体は :meth:`retract_fact`)。"""
        return self.retract_fact(fact_id, "deleted")

    def delete_facts(self, fact_ids: Iterable[str]) -> int:
        """複数ファクトをまとめて取り下げる。

        Returns:
            実際に取り下げた件数 (未存在の id は数えない)。
        """
        return sum(1 for fid in fact_ids if self.retract_fact(fid, "deleted"))

    def _clear_dangling_supersession(self, removed_ids: set[str]) -> None:
        """取り下げた ``removed_ids`` を指す ``superseded_by`` を外す。

        そのままだと **存在しない世代に置き換えられた** 状態で残り、読み出し系
        は既定で supersede 済を除外するのでスロットの live が 0 件になる。値が
        消えるのではなく「現在値が引けない」形で壊れる (2026-08-30 監査)。
        """
        for other in list(self._facts.values()):
            if other.superseded_by in removed_ids:
                logger.info(
                    "retract: restored %s as live (its superseder %s is gone)",
                    other.id, other.superseded_by,
                )
                self.update_fact(other.id, touch=False, superseded_by=None)

    # ── supersession / 競合 ─────────────────────────────────────────

    def supersedes_of(self, fact_id: str) -> list[str]:
        """``fact_id`` が置き換えたファクトの id (``superseded_by`` の逆写像)。"""
        return sorted(
            other.id for other in self._facts.values()
            if other.superseded_by == fact_id
        )

    def _supersede_chain_reaches(self, start_id: str, target_id: str) -> bool:
        """``start_id`` から ``superseded_by`` を辿って ``target_id`` に届くか。"""
        seen: set[str] = set()
        cur: str | None = start_id
        while cur and cur not in seen:
            if cur == target_id:
                return True
            seen.add(cur)
            fact = self._facts.get(cur)
            cur = fact.superseded_by if fact else None
        return False

    def supersede(self, old_id: str, new_id: str) -> None:
        """``old_id`` を ``new_id`` で置き換える (敗者に ``superseded_by``)。

        **閉路は作らせない。** ``A -> B`` の後に ``B -> A`` を通すと両方の
        ``superseded_by`` が埋まり、読み出し系が既定で supersede 済を除外する
        ため **そのスロットの live が 0 件になる**。値が消えるのではなく
        「現在値が引けない」形で壊れるので、想起は静かに 1 世代前の生テキスト
        へ落ちる。

        実データ (2026-08-30 ライブ監査): ``mem.personal.birthday`` の 4 件が
        2-閉路を含む鎖で全滅し、「私の誕生日はいつですか？」に **訂正前の
        3月14日** が返った。閉路はバッチを跨いで作られるため、抽出側のバッチ内
        ガードでは防ぎ切れない。ここが supersession の SSOT なので、不変則は
        ここで守る。
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
        if self._supersede_chain_reaches(new_id, old_id):
            raise ValueError(
                f"superseding {old_id} by {new_id} would create a cycle "
                f"({new_id} is already superseded by {old_id} transitively)",
            )
        # 競合が解けたので ``disputed`` は畳む (c_16 §4.2)。
        #
        # 敗者は ``touch=True`` — ``accessed_at`` が「置き換えられた時刻」に
        # なる。supersede 済みの保持期間 (``superseded_retention_days``) は
        # この時刻から数えるので、ここを ``touch=False`` にすると発話時刻から
        # 数えることになり、古い会話の訂正が即座に GC 対象になる。
        self.update_fact(
            old_id,
            superseded_by=new_id,
            veracity="retracted" if old.veracity == "disputed" else old.veracity,
        )
        # 敗者の元発話ノート id を勝者へ継承する (敗者は 3 版後に物理 GC され、
        # provenance から「現在値でない発話」を引く経路が切れるため)。
        inherited = set(new.retired_note_ids or ())
        inherited.update(old.retired_note_ids or ())
        inherited.update(p.note_id for p in (old.provenances or []) if p.note_id)
        changes: dict[str, Any] = {"retired_note_ids": sorted(inherited)}
        if new.veracity == "disputed":
            changes.update(veracity="stated", contradicts=[])
        self.update_fact(new_id, touch=False, **changes)
        logger.info("supersede: %s -> %s", old_id, new_id)
        self._supersede_generic_shadows(old_id, new_id)

    def _supersede_generic_shadows(self, old_id: str, new_id: str) -> int:
        """``old_id`` と同じノート由来の **汎用スロット** ファクトも一緒に畳む。

        1 つの発話は複数の tag で抽出されるため、属性を解決できた tag は
        ``mem.preference.tooling`` のような固有スロットへ、解決できなかった tag は
        ``mem.<kind>.user`` の **汎用フォールバック** へ入る。訂正は固有スロットに
        しか届かないので、汎用側の値は **supersede されないまま live に残る**。
        属性を持たないぶん ``asked_attrs`` の免除にも掛からず、埋め込み類似だけで
        いつまでも注入され続ける。

        実データ (2026-08-30 ライブ監査の追試): 「私の使っているエディタは Vim
        です。」がノート ``44cdd772ec6f`` から 2 件になり、
        ``mem.preference.tooling`` は Neovim の訂正で supersede されたのに
        ``mem.personal.user`` の Vim は live のまま残った。

        **汎用スロットは属性としての身元を持たない = その発話の影**なので、影の
        元が畳まれたら影も畳む。同一ノートでも **属性を持つ** 兄弟には手を出さない
        — 「私の名前は小川で、猫はミケです。」のような 1 発話 2 主張を巻き込まない
        ため (name を訂正しても pet は別の主張)。

        Returns:
            追加で supersede した件数。
        """
        old = self._facts.get(old_id)
        if old is None:
            return 0
        note_ids = {p.note_id for p in (old.provenances or []) if p.note_id}
        if not note_ids:
            # セッション要約 (mem.decision.history.*) は note_id を持たない。
            # ここを通すと「note_id なし」同士が 1 つの主張として束ねられる。
            return 0
        done = 0
        for fact in list(self._facts.values()):
            if fact.id in (old_id, new_id) or fact.superseded_by:
                continue
            if fact.scope != old.scope:
                continue
            if not is_generic_subject(fact.subject):
                continue
            if not note_ids.intersection(
                p.note_id for p in (fact.provenances or []) if p.note_id
            ):
                continue
            try:
                self.update_fact(fact.id, superseded_by=new_id)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Failed to retire generic shadow %s: %s", fact.id, exc,
                )
                continue
            done += 1
            logger.info(
                "supersede: retired generic shadow %s (%s) with %s",
                fact.id, fact.subject, old_id,
            )
        return done

    def mark_disputed(self, fact_ids: Sequence[str]) -> int:
        """グループを ``veracity=disputed`` + 相互 ``contradicts`` にする。

        競合が未解決のあいだの状態 (c_16 §4.2)。``superseded_by`` はまだ立て
        ない — 勝者が決まっていないので、どちらも live のまま残す。
        """
        live = [fid for fid in fact_ids if fid in self._facts]
        if len(live) < 2:
            return 0
        for fact_id in live:
            others = [fid for fid in live if fid != fact_id]
            self.update_fact(
                fact_id, touch=False, veracity="disputed", contradicts=others,
            )
        return len(live)

    def clear_dispute(self, fact_id: str) -> None:
        """``disputed`` を解いて ``stated`` に戻す (競合解決後)。"""
        fact = self._facts.get(fact_id)
        if fact is None or fact.veracity != "disputed":
            return
        self.update_fact(fact_id, touch=False, veracity="stated", contradicts=[])

    def disputed_facts(self, scope: str | None = None) -> list[SemanticFact]:
        """``veracity=disputed`` の live ファクト。"""
        return [
            f for f in self._facts.values()
            if f.veracity == "disputed" and not f.superseded_by
            and (scope is None or f.scope == scope)
        ]

    # ── 検索 ────────────────────────────────────────────────────────

    def search_by_subject(
        self,
        subject: str,
        *,
        include_superseded: bool = False,
        scope: str | None = None,
    ) -> list[SemanticFact]:
        """``subject`` に完全一致するファクトを返す。"""
        return self._collect(
            self._by_subject.get(subject, set()), include_superseded, scope,
        )

    def search_by_type(
        self,
        fact_type: FactType,
        *,
        include_superseded: bool = False,
        scope: str | None = None,
    ) -> list[SemanticFact]:
        """``type`` に完全一致するファクトを返す。"""
        return self._collect(
            self._by_type.get(str(fact_type), set()), include_superseded, scope,
        )

    def search_by_pillar_prefix(
        self,
        prefix: str,
        *,
        include_superseded: bool = False,
        scope: str | None = None,
    ) -> list[SemanticFact]:
        """3 pillar namespace の subject 前方一致でファクトを返す。

        ``prefix`` は ``loop.`` / ``learn.`` / ``mem.`` のいずれかで始まる
        完全な前方一致パターンを期待する (例: ``learn.policy.create.``)。
        """
        if not prefix.startswith(PILLAR_SUBJECT_PREFIXES):
            raise ValueError(
                "pillar prefix must start with one of "
                f"{PILLAR_SUBJECT_PREFIXES!r}, got {prefix!r}",
            )
        return self._collect(
            self._ids_by_subject_prefix(prefix, include_superseded=include_superseded),
            include_superseded,
            scope,
        )

    def _ids_by_subject_prefix(
        self,
        prefix: str | tuple[str, ...],
        *,
        include_superseded: bool,
    ) -> set[str]:
        """``subject`` が ``prefix`` で始まる fact id 集合 (namespace 索引経由)。

        候補を **cosine の前に** 絞るので、ストアが育っても走査は該当
        namespace の大きさにしか比例しない (2026-09-02 監査 H3)。
        """
        prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
        namespaces = {namespace_of(pf) for pf in prefixes}
        candidates: set[str] = set()
        for ns in namespaces:
            candidates |= self._by_namespace.get(ns, set())
        out: set[str] = set()
        for fid in candidates:
            fact = self._facts.get(fid)
            if fact is None or not fact.subject.startswith(prefixes):
                continue
            if not include_superseded and fact.superseded_by:
                continue
            out.add(fid)
        return out

    def count_by_subject_prefix(
        self,
        prefix: str | tuple[str, ...],
        *,
        include_superseded: bool = False,
    ) -> int:
        """``subject`` が ``prefix`` で始まるファクト数を返す。"""
        return len(
            self._ids_by_subject_prefix(prefix, include_superseded=include_superseded),
        )

    def count_by_type(
        self, fact_type: FactType, *, include_superseded: bool = False,
    ) -> int:
        """``type`` に該当するファクト数を返す。"""
        ids = self._by_type.get(str(fact_type), set())
        if include_superseded:
            return len(ids)
        return sum(
            1 for fid in ids
            if (f := self._facts.get(fid)) is not None and not f.superseded_by
        )

    def all_facts(
        self, *, include_superseded: bool = True, scope: str | None = None,
    ) -> list[SemanticFact]:
        """全ファクト (取り下げ済みは含まない)。"""
        return [
            f for f in self._facts.values()
            if (include_superseded or not f.superseded_by)
            and (scope is None or f.scope == scope)
        ]

    def know_facts(
        self, *, include_superseded: bool = False, scope: str | None = None,
    ) -> list[SemanticFact]:
        """``know.*`` の claim を namespace 索引経由で返す (c_16 §4.2)。

        取得器が「期限切れになった取得単位を指す claim」を引くのに使う。
        全件走査に落とすと、``mem.*`` が育つほど 1 サイクルが重くなる。
        """
        return self._collect(
            self._by_namespace.get("know", set()), include_superseded, scope,
        )

    def pinned_facts(self, scope: str | None = None) -> list[SemanticFact]:
        """pinned ファクトを返す。"""
        return [
            f for fid in self._pinned
            if (f := self._facts.get(fid)) is not None
            and (scope is None or f.scope == scope)
        ]

    def iter_facts(self) -> Iterator[SemanticFact]:
        return iter(list(self._facts.values()))

    def _collect(
        self,
        ids: Iterable[str],
        include_superseded: bool,
        scope: str | None = None,
    ) -> list[SemanticFact]:
        out: list[SemanticFact] = []
        for fid in ids:
            fact = self._facts.get(fid)
            if fact is None:
                continue
            if not include_superseded and fact.superseded_by:
                continue
            if scope is not None and fact.scope != scope:
                continue
            out.append(fact)
        return out

    # ── ベクトル (c_16 §6.1) ────────────────────────────────────────

    def vectors_for(self, fact_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """snapshot の埋め込みから float32 ベクトルを復元する。

        まだ snapshot に載っていないファクトは返らない (呼出側が落とす) —
        「版に載るまで密ベクトル検索の対象にならない」契約は episodic と同じ。
        """
        wanted = set(fact_ids)
        if not wanted:
            return {}
        store = self.evidence.vector_store()
        if store is None or store.vectors_q8 is None or store.scales is None:
            return {}
        rows: list[int] = []
        ids: list[str] = []
        for row, meta in enumerate(store.metadata):
            record_id = str(meta.get("id") or "")
            if record_id in wanted:
                rows.append(row)
                ids.append(record_id)
        if not rows:
            return {}
        index = np.asarray(rows, dtype=np.int64)
        restored = dequantize_int8(
            np.asarray(store.vectors_q8)[index], np.asarray(store.scales)[index],
        )
        return {
            record_id: np.asarray(restored[i], dtype=np.float32)
            for i, record_id in enumerate(ids)
        }

    def hydrate_embeddings(self, facts: Sequence[SemanticFact]) -> int:
        """``fact.embedding`` を snapshot のベクトルで埋める (作業用)。"""
        vectors = self.vectors_for([f.id for f in facts])
        filled = 0
        for fact in facts:
            vector = vectors.get(fact.id)
            if vector is not None:
                fact.embedding = vector
                filled += 1
        return filled

    def embedding_scores(self, query: np.ndarray) -> dict[str, float]:
        """クエリに対する全ファクトの **素の cosine** を返す。

        注入の関連度ゲート用。ゲートは順位ではなく閾値しか見ないので並べ替え
        は要らない。行列は snapshot の ``embeddings/`` に常駐しているため、
        候補ごとに正規化し直すより桁で速い。

        埋め込みを持たないファクトはキーに現れない (呼出側は従来どおり
        「判定不能なので通す」の分岐へ落ちる)。
        """
        snapshot = self.evidence.snapshot
        if snapshot is None or len(snapshot) == 0:
            return {}
        rows = np.arange(len(snapshot), dtype=np.int64)
        cosines = self.evidence.cosines_for_rows(query, rows)
        out: dict[str, float] = {}
        for row in range(len(snapshot)):
            value = cosines[row]
            if not np.isfinite(value):
                continue
            record_id = snapshot.id_at(row)
            if record_id in self._facts:
                out[record_id] = float(value)
        return out

    def store_prior_per_row(self) -> np.ndarray:
        """snapshot の行ごとの ``store_prior`` (c_16 §7.2)。

        ``namespace_id`` (columns.npz) と ``columns.namespaces`` の対応表から
        ``know`` の行だけ ``semantic_know`` に、それ以外を ``semantic_mem``
        にする。namespace を持たない行 (``UNKNOWN_I16``) は ``semantic_mem``。

        snapshot 版と行数が変わらない限り値は変わらないのでキャッシュする
        (1 ターンに ``search`` と ``ranking_scores`` の両方から呼ばれる)。
        """
        snapshot = self.evidence.snapshot
        if snapshot is None or len(snapshot) == 0:
            return np.empty(0, dtype=np.float64)
        key = (self.evidence.manifest.active_snapshot or "", len(snapshot))
        if key == self._prior_cache_key and self._prior_cache is not None:
            return self._prior_cache
        columns = snapshot.columns
        table = np.array(
            [
                self._prior_know if str(ns) == KNOW_NAMESPACE else self._prior_mem
                for ns in columns.namespaces
            ],
            dtype=np.float64,
        )
        ns_id = np.asarray(columns.namespace_id, dtype=np.int64)
        known = (ns_id >= 0) & (ns_id < table.shape[0])
        prior = np.full(ns_id.shape[0], self._prior_mem, dtype=np.float64)
        if table.shape[0] and np.any(known):
            prior[known] = table[ns_id[known]]
        self._prior_cache_key = key
        self._prior_cache = prior
        return prior

    def ranking_scores(self, query: np.ndarray) -> dict[str, float]:
        """全ファクトの **順位式スコア** を返す (c_16 §7.2)。

        ``score = cos × freshness × confidence × store_prior``。
        :meth:`embedding_scores` (素の cosine、ゲート用) と対になる出力で、
        ``MemoryInjector`` が ``[関連する記憶]`` の Tier 内の並びに使う。
        ゲートと順位で **同じ値を使わない** のが c_16 §7.1 の要点なので、
        2 つを別メソッドに分けている。

        埋め込みを持たないファクトはキーに現れない。
        """
        snapshot = self.evidence.snapshot
        if snapshot is None or len(snapshot) == 0:
            return {}
        rows = np.arange(len(snapshot), dtype=np.int64)
        cosines = self.evidence.cosines_for_rows(query, rows)
        finite = np.isfinite(cosines)
        if not np.any(finite):
            return {}
        columns = RankColumns.from_columns(snapshot.columns)
        now_epoch = utc_now_dt().timestamp()
        target = rows[finite]
        scores = score_rows(
            np.nan_to_num(cosines[finite]), target, columns, now_epoch,
            self.store_prior_per_row(),
        )
        out: dict[str, float] = {}
        for i, row in enumerate(target):
            record_id = snapshot.id_at(int(row))
            if record_id in self._facts:
                out[record_id] = float(scores[i])
        return out

    def search_by_embedding(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        include_superseded: bool = False,
        subject_prefix: str | tuple[str, ...] | None = None,
        scope: str | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        """埋め込みベクトルで cosine similarity 検索する。

        Args:
            subject_prefix: 与えると ``subject`` がこの接頭辞で始まるファクト
                だけを候補にして順位付けする。索引リコール (``idx.url.`` /
                ``idx.command.``) は必ずこれを渡すこと — グローバル top-k を
                引いてから接頭辞で絞ると、ストアが育った時点で索引行が top-k に
                入らなくなる (2026-09-02 監査 H3)。

        Returns:
            ``(fact, cosine)`` のリスト (cosine 降順、最大 ``top_k`` 件)。
        """
        if top_k <= 0:
            return []
        if subject_prefix is not None:
            ids = self._ids_by_subject_prefix(
                subject_prefix, include_superseded=include_superseded,
            )
        else:
            ids = {
                fid for fid, fact in self._facts.items()
                if include_superseded or not fact.superseded_by
            }
        if scope is not None:
            ids = {
                fid for fid in ids
                if (f := self._facts.get(fid)) is not None and f.scope == scope
            }
        if not ids:
            return []
        scores = self._cosines_for_ids(query, ids)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            (fact, score) for fid, score in ranked
            if (fact := self._facts.get(fid)) is not None
        ]

    def _cosines_for_ids(
        self, query: np.ndarray, ids: set[str],
    ) -> dict[str, float]:
        """``ids`` に限った素の cosine (snapshot 行から引く)。"""
        snapshot = self.evidence.snapshot
        if snapshot is None:
            return {}
        rows: list[int] = []
        row_ids: list[str] = []
        for fact_id in ids:
            row = snapshot.row_of(fact_id)
            if row is None:
                continue
            rows.append(row)
            row_ids.append(fact_id)
        if not rows:
            return {}
        cosines = self.evidence.cosines_for_rows(query, np.asarray(rows, dtype=np.int64))
        return {
            row_ids[i]: float(cosines[i])
            for i in range(len(row_ids))
            if np.isfinite(cosines[i])
        }

    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int = 10,
        *,
        threshold: float = 0.0,
        namespaces: Iterable[str] | None = None,
        scope: str | None = None,
        include_private: bool = False,
        now: float | None = None,
        store_prior: float | np.ndarray | None = None,
    ) -> list[SemanticHit]:
        """候補生成 → ゲート → 順位 → 畳み込みの 1 本 (c_16 §6.3 / §7.1〜7.3)。

        ゲートは素の cosine (``threshold``)、順位は
        ``cos × freshness × confidence × store_prior``。``namespaces`` は
        転置索引のシャード名そのもの (``mem`` / ``know`` / ``idx`` …) で、
        ``idx.*`` を注入に出さない側の門はここではなく
        :mod:`~backend.free.memory.semantic.namespaces` の ``injectable``。

        ``store_prior`` を省略すると **行ごとの値** を使う
        (:meth:`store_prior_per_row`): ``mem.*`` は
        ``memory.evidence.ranking.store_prior.semantic_mem``、``know.*`` は
        ``semantic_know``。1 つのストアに 2 つの規則が同居するので、
        c_16 §7.2 の係数はスカラでは表せない。
        """
        if top_k <= 0:
            return []
        prior: float | np.ndarray = (
            self.store_prior_per_row() if store_prior is None else store_prior
        )
        raw = self.evidence.search(
            query_text, query_vec, top_k,
            threshold=threshold,
            now=now,
            include_private=include_private,
            shards=namespaces,
            store_prior=prior,
        )
        snapshot = self.evidence.snapshot
        if snapshot is None:
            return []
        hits: list[SemanticHit] = []
        for row, cosine, score in raw:
            fact = self._facts.get(snapshot.id_at(row))
            if fact is None or fact.superseded_by:
                continue
            if scope is not None and fact.scope != scope:
                continue
            hits.append(SemanticHit(fact, cosine, score))
        return hits

    # ── 保持方針 (c_16 §5.4) ────────────────────────────────────────

    def enforce_retention(self, *, now: float | None = None) -> dict[str, int]:
        """``know.*`` の失効と ``idx.*`` の件数上限を適用する。

        - ``know.*``: ``valid_until`` 到来、または
          ``0.5 ** (age / half_life) < 0.05`` で ``retract("expired")``
        - ``idx.*``: ``idx_max_records`` を超えたぶんを ``last_used_at`` の
          古い順に ``retract("retention:idx_max_records")``
        - ``mem.*``: 上限なし (superseded は snapshot 3 版後に物理 GC)

        Returns:
            ``{"know_expired", "idx_evicted"}``。
        """
        reference = utc_now_dt().timestamp() if now is None else float(now)
        result = {"know_expired": 0, "idx_evicted": 0}

        for fact in list(self._facts.values()):
            if namespace_of(fact.subject) != "know":
                continue
            if self._is_expired(fact, reference):
                if self.retract_fact(fact.id, "expired"):
                    result["know_expired"] += 1

        limit = int(self.evidence.manifest.retention_value("idx_max_records") or 0)
        if limit > 0:
            idx_facts = [
                f for f in self._facts.values()
                if namespace_of(f.subject) == "idx" and not f.pinned
            ]
            overflow = len(idx_facts) - limit
            if overflow > 0:
                idx_facts.sort(key=lambda f: (f.accessed_at or f.created_at, f.id))
                for fact in idx_facts[:overflow]:
                    if self.retract_fact(fact.id, "retention:idx_max_records"):
                        result["idx_evicted"] += 1

        if result["know_expired"] or result["idx_evicted"]:
            logger.info(
                "Semantic retention: %d know fact(s) expired, %d idx fact(s) evicted",
                result["know_expired"], result["idx_evicted"],
            )
        return result

    def _is_expired(self, fact: SemanticFact, now: float) -> bool:
        """``know.*`` の失効判定 (c_16 §5.4)。"""
        record = self.evidence.get(fact.id)
        if record is None:
            return False
        valid_until = parse_utc(record.valid_until) if record.valid_until else None
        if valid_until is not None and valid_until.timestamp() <= now:
            return True
        half_life = record.half_life_days
        if half_life is None or half_life <= 0:
            return False
        age_days = max(0.0, (now - fact.created_at) / 86400.0)
        freshness = 0.5 ** (age_days / float(half_life))
        return freshness < KNOW_FRESHNESS_FLOOR

    def pending_physical_gc(self) -> list[str]:
        """物理 GC の対象になっている (superseded / retracted で 3 版経過) id。

        c_16 §5.4 は「superseded は snapshot 3 版後に物理 GC」と定める。
        判定に別の状態ファイルは要らない — 保持している版は
        ``retention.snapshots_keep`` (既定 3) 本なので、**いちばん古い保持版で
        既に死んでいたレコード** がちょうど「3 版経過した死者」になる。

        これは **公開の問い合わせ** (「今この瞬間、何件が対象か」)。実際に
        落とすのは :meth:`_keep_in_snapshot` — :meth:`create_snapshot` が
        ``EvidenceStore`` の ``gc_filter`` として渡すので、ここが返した id は
        次の版の ``records.jsonl`` に書かれず、ディスクから消える。

        pinned は落とさない (他の GC 経路と同じ扱い。
        :mod:`~backend.free.memory.semantic.gc`)。
        """
        versions = list_versions(self.store_dir)
        keep = int(
            self.evidence.manifest.retention_value("snapshots_keep")
            or SUPERSEDED_GC_SNAPSHOTS,
        )
        if len(versions) < keep:
            return []
        try:
            oldest = read_snapshot(self.store_dir, versions[0])
        except (OSError, ValueError) as e:
            logger.debug("pending_physical_gc: cannot read %s: %s", versions[0], e)
            return []
        out: list[str] = []
        for record in oldest.iter_records():
            if record.kind not in ("fact", "claim"):
                continue
            if record.veracity != "retracted" and not record.superseded_by:
                continue
            current = self.evidence.get(record.id)
            if current is None or current.pinned:
                continue
            if current.veracity == "retracted" or current.superseded_by:
                out.append(record.id)
        return out

    def _physical_gc_ids(self) -> set[str]:
        """物理 GC の対象 id (版の生成 1 回につき 1 度だけ計算する)。

        ``gc_filter`` はレコードごとに呼ばれるので、そのたびに最古版を読み直す
        と 1 版の生成で N 回 ``records.jsonl`` をなめることになる。鍵は active
        版名 — 版が切り替わるまで対象集合は動かない (書き手は sleep-time だけ)。
        """
        key = self.evidence.manifest.active_snapshot
        if self._gc_ids is None or self._gc_ids_key != key:
            self._gc_ids = set(self.pending_physical_gc())
            self._gc_ids_key = key
        return self._gc_ids

    def _keep_in_snapshot(self, record: Evidence) -> bool:
        """``EvidenceStore`` の ``gc_filter``。``False`` で新しい版から落とす。"""
        return record.id not in self._physical_gc_ids()

    def _invalidate_gc_cache(self) -> None:
        """物理 GC 対象のキャッシュを落とす。"""
        self._gc_ids = None
        self._gc_ids_key = None

    # ── snapshot ────────────────────────────────────────────────────

    def flush_touch(self) -> int:
        """:attr:`usage` を 1 つの ``touch`` 事象へ畳む (c_16 §2.1)。"""
        return self.evidence.flush_touch()

    async def create_snapshot(self) -> str | None:
        """未畳み込みの事象があるときだけ版を作る。

        畳み込みの結果には :meth:`_keep_in_snapshot` が掛かるので、この版で
        「3 版経過した死者」(:meth:`pending_physical_gc`) は物理的に消える。

        Returns:
            新しい版名。事象が無ければ ``None`` (無駄な版を積まない)。
        """
        if self.evidence.manifest.events_since_snapshot <= 0:
            logger.debug("Semantic: no events since the last snapshot, skipping")
            return None
        # 対象集合は畳み込みの直前に取り直す (前の版で数えたものを持ち越さない)。
        self._invalidate_gc_cache()
        version = await self.evidence.create_snapshot()
        # 版から行が消えたぶんを在メモリの写し / 索引へ反映する。
        self._rebuild_from_evidence()
        return version

    def save_manifest(self) -> None:
        """manifest を書き出す (``events_since_snapshot`` の永続化)。"""
        self.evidence.save_manifest()


class ScopedSemanticStore:
    """1 スコープに束縛した :class:`SemanticStore` のビュー。

    旧 ``SemanticFactStore`` (1 インスタンス = 1 スコープ) の面をそのまま保つ
    ためのもので、実体は共有の 1 ストア (c_16 §4.2)。読み出しは
    ``scope`` で絞り、書き込みは ``fact.scope`` を束縛値へ揃える。

    ``[global_store, project_store]`` の 2 本を渡す既存の呼出側 (Fact View /
    sleep-time) がそのまま動き、しかも同じレコードを 2 度数えない。
    """

    __slots__ = ("scope", "store")

    def __init__(self, store: SemanticStore, scope: str) -> None:
        self.store = store
        self.scope = scope

    # ── 観測 ──

    @property
    def revision(self) -> int:
        return self.store.revision

    @property
    def root_dir(self) -> Path:
        """スコープ別ファイル (競合ログ等) の置き場。"""
        if self.scope == "global":
            return self.store.store_dir / "global"
        return self.store.store_dir / "projects" / self.scope.split(":", 1)[1]

    def __len__(self) -> int:
        return len(self.store.all_facts(scope=self.scope))

    # ── CRUD ──

    def add_fact(self, fact: SemanticFact) -> SemanticFact:
        fact.scope = self.scope
        return self.store.add_fact(fact)

    def get_fact(self, fact_id: str) -> SemanticFact | None:
        # id 引きはスコープで絞らない — supersede / 競合解決は global と
        # project にまたがった id を辿る。
        return self.store.get_fact(fact_id)

    def update_fact(self, fact_id: str, **changes: Any) -> SemanticFact:
        return self.store.update_fact(fact_id, **changes)

    def delete_fact(self, fact_id: str) -> bool:
        return self.store.delete_fact(fact_id)

    def delete_facts(self, fact_ids: Iterable[str]) -> int:
        return self.store.delete_facts(fact_ids)

    def supersede(self, old_id: str, new_id: str) -> None:
        self.store.supersede(old_id, new_id)

    def mark_disputed(self, fact_ids: Sequence[str]) -> int:
        return self.store.mark_disputed(fact_ids)

    def clear_dispute(self, fact_id: str) -> None:
        self.store.clear_dispute(fact_id)

    def disputed_facts(self) -> list[SemanticFact]:
        return self.store.disputed_facts(scope=self.scope)

    def supersedes_of(self, fact_id: str) -> list[str]:
        return self.store.supersedes_of(fact_id)

    # ── 検索 ──

    def search_by_subject(
        self, subject: str, *, include_superseded: bool = False,
    ) -> list[SemanticFact]:
        return self.store.search_by_subject(
            subject, include_superseded=include_superseded, scope=self.scope,
        )

    def search_by_type(
        self, fact_type: FactType, *, include_superseded: bool = False,
    ) -> list[SemanticFact]:
        return self.store.search_by_type(
            fact_type, include_superseded=include_superseded, scope=self.scope,
        )

    def search_by_pillar_prefix(
        self, prefix: str, *, include_superseded: bool = False,
    ) -> list[SemanticFact]:
        return self.store.search_by_pillar_prefix(
            prefix, include_superseded=include_superseded, scope=self.scope,
        )

    def search_by_embedding(
        self,
        query: np.ndarray,
        top_k: int = 10,
        *,
        include_superseded: bool = False,
        subject_prefix: str | tuple[str, ...] | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        return self.store.search_by_embedding(
            query, top_k=top_k, include_superseded=include_superseded,
            subject_prefix=subject_prefix, scope=self.scope,
        )

    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[SemanticHit]:
        return self.store.search(
            query_text, query_vec, top_k, scope=self.scope, **kwargs,
        )

    def count_by_subject_prefix(
        self,
        prefix: str | tuple[str, ...],
        *,
        include_superseded: bool = False,
    ) -> int:
        return self.store.count_by_subject_prefix(
            prefix, include_superseded=include_superseded,
        )

    def count_by_type(
        self, fact_type: FactType, *, include_superseded: bool = False,
    ) -> int:
        return len(
            self.search_by_type(fact_type, include_superseded=include_superseded),
        )

    def all_facts(self, *, include_superseded: bool = True) -> list[SemanticFact]:
        return self.store.all_facts(
            include_superseded=include_superseded, scope=self.scope,
        )

    def pinned_facts(self) -> list[SemanticFact]:
        return self.store.pinned_facts(scope=self.scope)

    def embedding_scores(self, query: np.ndarray) -> dict[str, float]:
        return self.store.embedding_scores(query)

    def ranking_scores(self, query: np.ndarray) -> dict[str, float]:
        return self.store.ranking_scores(query)

    def vectors_for(self, fact_ids: Sequence[str]) -> dict[str, np.ndarray]:
        return self.store.vectors_for(fact_ids)

    def hydrate_embeddings(self, facts: Sequence[SemanticFact]) -> int:
        return self.store.hydrate_embeddings(facts)


def _discard(index: dict[str, set[str]], key: str, fact_id: str) -> None:
    """索引バケットから id を外し、空になったバケットを畳む。"""
    bucket = index.get(key)
    if bucket is None:
        return
    bucket.discard(fact_id)
    if not bucket:
        index.pop(key, None)


__all__ = [
    "DEFAULT_SEMANTIC_KNOW_PRIOR",
    "DEFAULT_SEMANTIC_MEM_PRIOR",
    "KNOW_FRESHNESS_FLOOR",
    "KNOW_NAMESPACE",
    "PILLAR_SUBJECT_PREFIXES",
    "STORE_DIRNAME",
    "SUPERSEDED_GC_SNAPSHOTS",
    "WRITER",
    "ScopedSemanticStore",
    "SemanticHit",
    "SemanticStore",
    "semantic_shard_key",
]
