"""`EpisodicStore` — 会話由来ノートの唯一の永続層 (c_16 §4.1)。

``local/memory/episodic/`` に :class:`~backend.free.rag.evidence.EvidenceStore`
を 1 つだけ持ち、WM / STM / LTM の 3 ストアを ``attrs.tier`` へ畳む。

```
working  プロセス内のみ (WorkingMemory。ここには入らない)
short    sleep-time が put する。会話ターンから起こしたノート
long     sleep-time が patch(tier="long") する。データは動かない
```

シャードは **tier × 月** (``short:2026-09`` / ``long:2026-08``、c_16 §6.2)。
検索は ``short:*`` を先に引き、``long:*`` は直近
:data:`LONG_SHARD_WINDOW_MONTHS` か月のシャードだけを見る (c_16 §7.1)。

書き手は sleep-time だけ。チャット応答パスは :meth:`search` で読み、
注入した id を :attr:`usage` へ入れる。``last_used_at`` の書き込みは
sleep-time の :meth:`flush_touch` が 1 事象に畳んで行う (c_16 §2.1)。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.memory.episodic.note import (
    MemoryNote,
    evidence_to_note,
    note_to_evidence,
)
from backend.free.memory.episodic.progress import PROGRESS_FILE, EpisodicProgress
from backend.free.rag.evidence import (
    Evidence,
    EvidenceStore,
    UsageBuffer,
    is_active,
    list_versions,
    new_evidence_id,
    read_snapshot,
)
from backend.free.rag.evidence.types import KIND_IDS, TIER_IDS, UNKNOWN_I16
from backend.free.rag.vector_store import dequantize_int8
from backend.log_config import get_logger
from backend.utils import parse_utc, utc_now_dt

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.episodic.store")

#: ストアのディレクトリ名 (``local/memory/episodic``)。
STORE_DIRNAME = "episodic"

#: ``long`` tier で検索するシャードの窓 (月数)。これより古い月のシャードは
#: 語彙索引を読まない (c_16 §7.1 「long は月次シャード重心で刈る」の第一段)。
LONG_SHARD_WINDOW_MONTHS = 12

#: 事象ログの ``by`` (書き手コンポーネント)。
WRITER = "sleep_time.episodic"

#: tier 別検索で ``EvidenceStore`` から引く候補の倍率。ベクトル索引は
#: ストア全体で 1 つなので、素の ``top_k`` だと片方の tier に食われる。
TIER_OVERSAMPLE = 4

#: ``retracted`` を物理 GC してよくなるまでの版数 (c_16 §5.4)。
#: manifest の ``retention.snapshots_keep`` が無いときの既定。
RETRACTED_GC_SNAPSHOTS = 3


def episodic_shard_key(record: Evidence) -> str:
    """``tier:yyyy-mm`` (c_16 §6.2)。tier / 月が読めなければ既定へ倒す。"""
    tier = record.tier or "short"
    month = (record.observed_at or "")[:7]
    return f"{tier}:{month or 'unknown'}"


class EpisodicHit:
    """検索 1 件 (``id`` / 素の cosine / 順位式スコア / 本文 / レコード)。

    ``cosine`` はゲート用 (c_16 §7.1 「ゲートは素の cosine のみ」)、``score``
    は順位用 (``cos × freshness × confidence × store_prior``)。2 つを分けて
    持つのは、閾値が cosine スケール前提で決まっているため。
    """

    __slots__ = ("cosine", "record", "score", "text")

    def __init__(self, record: Evidence, cosine: float, score: float, text: str) -> None:
        self.record = record
        self.cosine = float(cosine)
        self.score = float(score)
        self.text = text

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def tier(self) -> str:
        return self.record.tier or "short"


class EpisodicStore:
    """エピソード記憶 (kind=``note``) のストア。

    Args:
        memory_dir: ``local_paths.memory_dir``。実体は ``<memory_dir>/episodic``。
        embedding_backend: 埋め込みバックエンド。``None`` なら snapshot 生成時に
            ベクトル索引を作らない (語彙索引だけの縮退動作)。
        rag_config: ``config.yaml`` の ``rag`` セクション (量子化 / memmap /
            クラスタ索引 / ``lexical``)。
        retention: ``memory.evidence.retention`` (c_16 §9)。manifest へ宣言する。
        debug_logger: JSONL 観測用 (memory カテゴリ)。
    """

    def __init__(
        self,
        memory_dir: Path | str,
        embedding_backend: "EmbeddingBackend | None" = None,
        rag_config: Any = None,
        *,
        retention: dict[str, Any] | None = None,
        debug_logger: Any = None,
    ) -> None:
        self.store_dir = Path(memory_dir) / STORE_DIRNAME
        self.evidence = EvidenceStore(
            self.store_dir,
            store_name="episodic",
            embedding_backend=embedding_backend,
            rag_config=rag_config,
            by=WRITER,
            shard_key_for=episodic_shard_key,
            gc_filter=self._keep_in_snapshot,
        )
        self.progress = EpisodicProgress(self.store_dir / PROGRESS_FILE)
        self._debug_logger = debug_logger
        self._retention_override = dict(retention or {})
        #: :meth:`short_notes` のキャッシュ (snapshot 版 + 事象数で無効化)。
        self._short_cache: list[MemoryNote] | None = None
        self._short_cache_key: tuple[str, int] | None = None
        #: active snapshot に **まだ載っていない** レコードの id。
        #: カラムで行を選ぶ :meth:`iter_notes` が、版の外にあるレコードを
        #: 取りこぼさないために持つ (全件走査の代わり)。
        self._pending_ids: list[str] = []
        #: 物理 GC の対象 id (版の生成 1 回につき 1 度だけ計算する)。
        self._gc_ids: set[str] | None = None
        self._gc_ids_key: str | None = None

    # ── ライフサイクル ──

    def load(self) -> None:
        """manifest / snapshot / 事象 / 進捗を読む。"""
        self.evidence.load()
        self.progress.load()
        if self._retention_override:
            self.evidence.manifest.retention.update(self._retention_override)
        self._invalidate_cache()
        self._invalidate_gc_cache()
        self._pending_ids = []
        if self.evidence.manifest.events_since_snapshot > 0:
            # 版を作る前に落ちた分。ここだけは全件を見る (起動時 1 回)。
            snapshot = self.evidence.snapshot
            self._pending_ids = [
                record.id for record in self.evidence.iter_records()
                if snapshot is None or snapshot.row_of(record.id) is None
            ]
        logger.info(
            "Episodic store loaded: %d record(s), snapshot=%s, %d event(s) pending",
            len(self.evidence),
            self.evidence.manifest.active_snapshot or "(none)",
            self.evidence.manifest.events_since_snapshot,
        )

    @property
    def usage(self) -> UsageBuffer:
        """チャット経路が「使った」id を溜めるバッファ (c_16 §2.1)。"""
        return self.evidence.usage

    def __len__(self) -> int:
        return len(self.evidence)

    # ── 読み出し ──

    def get(self, record_id: str) -> Evidence | None:
        return self.evidence.get(record_id)

    def note(self, record_id: str) -> MemoryNote | None:
        """id から作業用 :class:`MemoryNote` を作る (無ければ ``None``)。"""
        record = self.evidence.get(record_id)
        return None if record is None else evidence_to_note(record)

    def iter_notes(
        self, *, tier: str | None = None, include_private: bool = False,
    ) -> list[MemoryNote]:
        """アクティブなノートを列挙する (``retracted`` / 要約済みは除く)。

        ``tier`` を指定したときは **カラム (``tier`` / アクティブマスク) で行を
        選んでから本文を読む**。``records.jsonl`` の行は lazy 読みなので、
        50,000 件の ``long`` を抱えた状態で ``short`` を引くたびに全件を
        JSON パースしないため (走査量を件数に依存させない、c_16 §5.3 / §6)。
        """
        snapshot = self.evidence.snapshot
        if tier is None or snapshot is None:
            out: list[MemoryNote] = []
            for record in self.evidence.iter_active(include_private=include_private):
                if record.kind != "note":
                    continue
                if tier is not None and (record.tier or "short") != tier:
                    continue
                out.append(evidence_to_note(record))
            return out

        now_epoch = utc_now_dt().timestamp()
        mask = self.evidence.active_mask(now_epoch, include_private)
        columns = snapshot.columns
        wanted = TIER_IDS.get(tier, UNKNOWN_I16)
        rows = np.flatnonzero(
            mask
            & (columns.tier == wanted)
            & (columns.kind == KIND_IDS["note"]),
        )
        notes: list[MemoryNote] = []
        for row in rows:
            record_id = snapshot.id_at(int(row))
            # 事象で書き換わった行はオーバーレイ側が現在値。カラムの tier は
            # 版を作るまで古いままなので、tier だけは実レコードで見直す
            # (``patch(tier="long")`` した直後の行がここに混ざる)。
            record = self.evidence.get(record_id)
            if record is None or (record.tier or "short") != tier:
                continue
            notes.append(evidence_to_note(record))
        # カラムに載っていない / カラムの tier が古い行を拾い直す。
        # ``_pending_ids`` は前回の版以降に触った id なので、走査量は
        # レコード総数ではなくサイクル内の書き込み件数に比例する。
        seen = {note.id for note in notes}
        for record_id in self._pending_ids:
            if record_id in seen:
                continue
            record = self.evidence.get(record_id)
            if record is None or record.kind != "note":
                continue
            if (record.tier or "short") != tier:
                continue
            if not is_active(record, now_epoch, include_private):
                continue
            notes.append(evidence_to_note(record))
        return notes

    def short_notes(self, *, include_private: bool = False) -> list[MemoryNote]:
        """``short`` tier のアクティブノート (**ベクトル込み**、snapshot 単位でキャッシュ)。

        チャット応答パスの 3 つの読み手が使う:

        - ``MemoryInjector`` の Tier 2 (``[関連する記憶]``) — 関連度ゲートが
          ``require_embedding=True`` なので、ベクトルを載せないと **pin 以外が
          全部落ちる**
        - ``search_pipeline.attach_superseding_corrections`` (訂正の随伴注入)
        - 「前にも同じことを聞いたか」の判定

        ベクトルの復元は snapshot の ``embeddings/`` を 1 度なめる。キャッシュ鍵は
        (active snapshot 版, 未畳み込み事象数) — 書き手は sleep-time だけなので、
        これが変わらない限り内容も変わらない = 走査は 1 サイクル 1 回で、
        ターン数に比例しない。
        """
        key = (
            self.evidence.manifest.active_snapshot,
            int(self.evidence.manifest.events_since_snapshot),
        )
        if self._short_cache is not None and self._short_cache_key == key:
            return self._short_cache
        notes = self.notes_with_vectors(
            tier="short", include_private=include_private,
        )
        self._short_cache = notes
        self._short_cache_key = key
        return notes

    def notes_with_vectors(
        self, *, tier: str | None = "short", include_private: bool = False,
    ) -> list[MemoryNote]:
        """ベクトルを載せたノート (閾値較正 / 進化の入力)。

        ベクトルは snapshot の ``embeddings/`` から復元する。まだ snapshot に
        載っていないノートは ``embedding=None`` のまま返る (呼出側が落とす)。
        """
        notes = self.iter_notes(tier=tier, include_private=include_private)
        vectors = self.vectors_for([note.id for note in notes])
        for note in notes:
            note.embedding = vectors.get(note.id)
        return notes

    def _invalidate_cache(self, record_id: str | None = None) -> None:
        """検索キャッシュを落とす (``record_id`` があれば版外レコードとして覚える)。"""
        self._short_cache = None
        self._short_cache_key = None
        if record_id and record_id not in self._pending_ids:
            self._pending_ids.append(record_id)

    # ── 検索 (チャット応答パス。読むだけ) ──

    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int = 5,
        *,
        threshold: float = 0.0,
        include_private: bool = False,
        now: float | None = None,
    ) -> list[EpisodicHit]:
        """``short`` → ``long`` の順に引いて上位 ``top_k`` 件を返す。

        ゲートは素の cosine (``threshold``)、順位は
        ``EvidenceStore.search`` の順位式 (c_16 §7.1 / §7.2)。``short`` を
        先に埋めるのは c_16 §7.1 のとおり — どちらも同じ cosine の棒を越えて
        いるので、同点帯では新しい会話ノートを優先する。
        """
        if top_k <= 0:
            return []
        now_epoch = utc_now_dt().timestamp() if now is None else float(now)
        short = self._search_tier(
            "short", query_text, query_vec, top_k,
            threshold=threshold, include_private=include_private, now=now_epoch,
        )
        hits = list(short)
        if len(hits) < top_k:
            seen = {hit.id for hit in hits}
            for hit in self._search_tier(
                "long", query_text, query_vec, top_k,
                threshold=threshold, include_private=include_private,
                now=now_epoch,
            ):
                if hit.id in seen:
                    continue
                hits.append(hit)
                if len(hits) >= top_k:
                    break
        logger.debug(
            "Episodic search: %d hit(s) (short=%d, top_k=%d, threshold=%.3f)",
            len(hits), len(short), top_k, threshold,
        )
        return hits[:top_k]

    def _search_tier(
        self,
        tier: str,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int,
        *,
        threshold: float,
        include_private: bool,
        now: float,
    ) -> list[EpisodicHit]:
        """1 つの tier のシャードだけを引く。"""
        shards = self._shards_for(tier, now)
        if shards is not None and not shards:
            return []
        # ベクトル候補は tier を区別しない (索引はストア全体で 1 つ)。素の
        # ``top_k`` だけ引くと、上位が全部もう片方の tier だったときにこの
        # tier の該当が 1 件も残らない。多めに引いてから tier で絞る。
        raw = self.evidence.search(
            query_text, query_vec, top_k * TIER_OVERSAMPLE,
            threshold=threshold,
            now=now,
            include_private=include_private,
            shards=shards,
        )
        snapshot = self.evidence.snapshot
        if snapshot is None:
            return []
        hits: list[EpisodicHit] = []
        for row, cosine, score in raw:
            record_id = snapshot.id_at(row)
            record = self.evidence.get(record_id)
            if record is None or record.kind != "note":
                continue
            if (record.tier or "short") != tier:
                continue
            hits.append(
                EpisodicHit(record, cosine, score, record.text or self.evidence.text_at(row)),
            )
            if len(hits) >= top_k:
                break
        return hits

    def _shards_for(self, tier: str, now: float) -> list[str] | None:
        """``tier`` のシャード名。索引が無ければ ``None`` (= 全シャード)。"""
        shard_set = self.evidence.lexical_shards()
        if shard_set is None:
            return None
        prefix = f"{tier}:"
        names = [name for name in shard_set.shards if name.startswith(prefix)]
        if tier != "long":
            return names
        cutoff = _month_cutoff(now, LONG_SHARD_WINDOW_MONTHS)
        return [name for name in names if name[len(prefix):] >= cutoff]

    # ── 書き込み (sleep-time 専用) ──

    def put_note(self, note: MemoryNote, *, tier: str = "short") -> Evidence:
        """ノートを ``put`` する (新規 id はここで発番)。"""
        if not note.id:
            note.id = new_evidence_id()
        note.tier = tier
        record = note_to_evidence(note, tier=tier)
        self._invalidate_cache(record.id)
        return self.evidence.put(record)

    def patch_note(self, record_id: str, **fields: Any) -> Evidence | None:
        """レコードの一部を ``patch`` する。"""
        self._invalidate_cache(record_id)
        return self.evidence.patch(record_id, **fields)

    def retract_note(self, record_id: str, reason: str) -> Evidence | None:
        """``veracity=retracted`` にする (物理削除はしない)。"""
        self._invalidate_cache(record_id)
        return self.evidence.retract(record_id, reason)

    def promote_aged_notes(self, *, short_days: float, now: float | None = None) -> int:
        """``short_days`` を超えた ``short`` ノートを ``long`` へ ``patch`` する。

        **データは動かさないし再チャンクもしない** (c_16 §4.1)。次の snapshot
        で ``long:<月>`` シャードへ入る。
        """
        reference = utc_now_dt().timestamp() if now is None else float(now)
        cutoff = reference - float(short_days) * 86400.0
        promoted = 0
        for note in self.iter_notes(tier="short", include_private=True):
            if note.created_at <= 0 or note.created_at > cutoff:
                continue
            if self.patch_note(note.id, tier="long") is not None:
                promoted += 1
        if promoted:
            self._invalidate_cache()
            logger.info(
                "Episodic: promoted %d note(s) to the long tier (short_days=%.1f)",
                promoted, short_days,
            )
        return promoted

    def write_summary_note(
        self,
        *,
        session_id: str,
        text: str,
        note_ids: list[str],
        mode: str = "chat",
        lang: str = "",
        project_id: str | None = None,
    ) -> str | None:
        """要約ノートを 1 件書き、元ノートに ``superseded_by`` を付ける。

        元ノートは **消さない** (c_16 §4.1)。要約が指しているので注入からは
        外れるが、監査では追える。
        """
        live = [nid for nid in note_ids if self.evidence.get(nid) is not None]
        if not text.strip() or len(live) < 2:
            return None
        summary = MemoryNote(
            id=new_evidence_id(),
            content=text.strip(),
            session_id=session_id,
            source="system",
            mode=mode,  # type: ignore[arg-type]
            lang=lang,
            project_id=project_id,
            created_at=utc_now_dt().timestamp(),
            summary_of=list(live),
            tier="long",
        )
        record = self.put_note(summary, tier="long")
        for note_id in live:
            self.patch_note(note_id, superseded_by=record.id)
        logger.info(
            "Episodic: wrote summary note %s over %d note(s) of session %s",
            record.id, len(live), session_id,
        )
        self._invalidate_cache()
        return record.id

    def enforce_retention(self, *, long_max_records: int | None = None) -> int:
        """``long`` が上限を超えたぶんを ``retract`` する (c_16 §5.4)。

        除外は ``pinned``、順は ``last_used_at`` の最も古いものから
        (未使用は ``observed_at`` で代用する — 一度も使われていないノートは
        「最後に使ったのが作成時点」と同じ扱いでよい)。
        """
        limit = int(
            long_max_records
            if long_max_records is not None
            else self.evidence.manifest.retention_value("long_max_records"),
        )
        if limit <= 0:
            return 0
        candidates: list[tuple[float, str]] = []
        total = 0
        for note in self.iter_notes(tier="long", include_private=True):
            total += 1
            if note.pin_flag:
                continue
            candidates.append((note.accessed_at or note.created_at, note.id))
        # 上限は ``long`` 全体に対する件数。落とせるのは pinned でないものだけ
        # なので、pinned が上限を埋めていると溢れたぶんを落としきれない
        # (それでよい — pin は「消すな」の明示)。
        overflow = min(total - limit, len(candidates))
        if overflow <= 0:
            return 0
        candidates.sort()
        for _, record_id in candidates[:overflow]:
            self.retract_note(record_id, "retention:long_max_records")
        logger.info(
            "Episodic: retracted %d long note(s) over the %d record cap",
            overflow, limit,
        )
        return overflow

    # ── 物理 GC (c_16 §5.4) ──

    def pending_physical_gc(self) -> list[str]:
        """物理 GC の対象になっている (``retracted`` で 3 版経過) ノートの id。

        c_16 §5.4 は episodic の evict を ``retract`` と定め、物理削除は
        「保持方針の GC のみ」とする。判定に別の状態ファイルは要らない —
        保持している版は ``retention.snapshots_keep`` (既定 3) 本なので、
        **いちばん古い保持版で既に ``retracted`` だったノート** がちょうど
        「``snapshots_keep`` 版を死んだまま過ごした」ものになる。

        ``pinned`` は落とさない。pin は「消すな」の明示なので、上限超過で
        ``retract`` されたあとでも物理削除の対象にはしない。

        これは公開の問い合わせで、実際に落とすのは :meth:`_keep_in_snapshot`
        (:class:`~backend.free.rag.evidence.EvidenceStore` の ``gc_filter``)。
        """
        versions = list_versions(self.store_dir)
        keep = int(
            self.evidence.manifest.retention_value("snapshots_keep")
            or RETRACTED_GC_SNAPSHOTS,
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
            if record.kind != "note" or record.veracity != "retracted":
                continue
            current = self.evidence.get(record.id)
            if current is None or current.pinned:
                continue
            if current.veracity == "retracted":
                out.append(record.id)
        return out

    def _physical_gc_ids(self) -> set[str]:
        """物理 GC の対象 id (版の生成 1 回につき 1 度だけ計算する)。

        ``gc_filter`` はレコードごとに呼ばれる。毎回いちばん古い版を読み直すと
        1 版の生成で N 回 ``records.jsonl`` をなめることになるので、active 版名
        を鍵にして 1 度だけ計算する (書き手は sleep-time だけなので、版が
        切り替わるまで対象集合は動かない)。
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

    def flush_touch(self) -> int:
        """:attr:`usage` を 1 つの ``touch`` 事象へ畳む (c_16 §2.1)。"""
        return self.evidence.flush_touch()

    async def create_snapshot(self) -> str | None:
        """未畳み込みの事象があるときだけ版を作る。

        畳み込みの結果には :meth:`_keep_in_snapshot` が掛かるので、この版で
        「``snapshots_keep`` 版を ``retracted`` のまま過ごしたノート」
        (:meth:`pending_physical_gc`) は物理的に消える。

        Returns:
            新しい版名。事象が無ければ ``None`` (無駄な版を積まない)。
        """
        if self.evidence.manifest.events_since_snapshot <= 0:
            logger.debug("Episodic: no events since the last snapshot, skipping")
            return None
        # 対象集合は畳み込みの直前に取り直す (前の版で数えたものを持ち越さない)。
        self._invalidate_gc_cache()
        version = await self.evidence.create_snapshot()
        self.progress.save()
        # 版から行が消えたぶんも含め、派生キャッシュを落とす。
        self._invalidate_cache()
        self._invalidate_gc_cache()
        # 版に畳まれたので「版の外」は無くなる。
        self._pending_ids = []
        return version

    def save_progress(self) -> None:
        """ノート化の進捗だけを書き出す (snapshot を作らないサイクル用)。"""
        self.progress.save()

    def close(self) -> None:
        """memmap を握った索引を手放す (Windows で削除できるように)。

        Windows は memmap で開いたままのファイルを削除できず、旧版の GC が
        黙って失敗する (CLAUDE.md §10)。SemMem と同じく shutdown で必ず呼ぶ。
        """
        self.evidence.close()
        self._invalidate_cache()

    # ── ベクトル (sleep-time の競合検出 / ノート進化が使う) ──

    def vectors_for(self, record_ids: list[str]) -> dict[str, np.ndarray]:
        """snapshot の埋め込みから float32 ベクトルを復元する。

        Step 6 (競合検出) / Step 7 (ノート進化) はノート同士の類似度を要る。
        ここで拾えなかった id だけを埋め込み直せば、毎サイクル全件を埋め直さ
        なくて済む (埋め込みそのものは snapshot 生成時に増分で作られる)。
        """
        wanted = set(record_ids)
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

    # ── 作業領域 ──

    def open_workspace(self, *, include_private: bool = False):
        """sleep-time 用の作業領域を開く (旧 ``ShortTermMemory`` の面)。

        循環 import を避けるため関数内で import する
        (workspace は store を型注釈で参照する)。
        """
        from backend.free.memory.episodic.workspace import EpisodicWorkspace

        return EpisodicWorkspace.load(self, include_private=include_private)


def _month_cutoff(now: float, months: int) -> str:
    """``now`` から ``months`` か月前の ``yyyy-mm``。"""
    from datetime import UTC, datetime

    reference = datetime.fromtimestamp(now, tz=UTC)
    total = reference.year * 12 + (reference.month - 1) - max(0, int(months))
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


__all__ = [
    "LONG_SHARD_WINDOW_MONTHS",
    "STORE_DIRNAME",
    "TIER_OVERSAMPLE",
    "WRITER",
    "EpisodicHit",
    "EpisodicStore",
    "episodic_shard_key",
]
