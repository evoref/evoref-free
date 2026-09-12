"""corpus パッケージの疑似クエリ索引 (f_01 §6 / c_16 §4.3)。

チャンクごとに補助タスクが作った「このチャンクが答える問い」を、パッケージ版
ディレクトリ配下の **独立した** :class:`EvidenceStore` (``pseudo_queries/``) に
``kind="doc_pseudo_query"`` で持つ。埋め込みは query 側 (instruction 付き) —
検索時のクエリと同じ側に揃えるため。

パッケージ本体の snapshot / centroid / chunk_count には混ぜない。順位付けも
順位式 1 本の外側で位置の interleave をする (``search_pipeline``)。ここでは
「問い↔問いの cosine 順に対象チャンク id を返す」ことだけを担う。
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.rag.evidence.events import EventPosition
from backend.free.rag.evidence.store import EvidenceStore
from backend.free.rag.evidence.types import Evidence, derive_confidence
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("rag.corpus.pseudo_queries")

#: パッケージ版ディレクトリ配下の索引ディレクトリ名。
PSEUDO_QUERIES_DIR = "pseudo_queries"
#: 疑似クエリレコードの kind。
PSEUDO_QUERY_KIND = "doc_pseudo_query"
#: 検索時のクエリと同じ instruction を使う embed mode。
PSEUDO_QUERY_EMBED_MODE = "chat"


def pseudo_query_id(target_id: str, position: int, text: str) -> str:
    """``(対象チャンク id, 位置, 問い)`` から内容由来の決定論 id を導く。"""
    payload = f"{target_id}\x00{position}\x00{text}"
    return "pq_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class PseudoQueryIndex:
    """1 パッケージ版の疑似クエリ索引。

    Args:
        version_dir: ``packages/<id>/<version>/``。
        package_id: ``attrs.package_id`` に刻む。
        embedding_backend: 埋め込みバックエンド (検索だけなら ``None`` 可)。
        rag_config: パッケージ本体と同じ ``rag`` セクション。
    """

    def __init__(
        self,
        version_dir: Path | str,
        package_id: str,
        embedding_backend: "EmbeddingBackend | None" = None,
        rag_config: Any = None,
    ) -> None:
        self.directory = Path(version_dir) / PSEUDO_QUERIES_DIR
        self.package_id = package_id
        self._store = EvidenceStore(
            self.directory,
            store_name="corpus",
            embedding_backend=embedding_backend,
            rag_config=rag_config,
            by="pseudo_query_gen",
            shard_key_for=lambda _record, key=package_id: key,
        )
        self._loaded = False
        #: put 済みで未 commit の対象チャンク id (同じサイクル内の二重生成を防ぐ)。
        self._pending_targets: set[str] = set()
        #: 対象チャンクが本体 snapshot に生きているか (f_01 §6.2 の孤児 GC)。
        self._is_live: Callable[[str], bool] = lambda _target: True

    def bind_targets(self, is_live: Callable[[str], bool]) -> None:
        """対象チャンクの生存判定を束ねる。孤児は検索 / 充足率から外し、commit で物理 GC する。"""
        self._is_live = is_live
        self._store.gc_filter = lambda record: bool(
            self._is_live(str((record.attrs or {}).get("target_id") or "")),
        )

    # ── ライフサイクル ──

    def exists(self) -> bool:
        """索引ディレクトリがあるか (manifest は最初の commit で書かれる)。"""
        return self.directory.is_dir()

    def load(self) -> None:
        """索引を開く (無ければ空のまま。ディレクトリは最初の書き込みで作る)。

        事象ログが無いのに畳み込みの位置が先頭でなければ、位置を戻す
        (修正前の :meth:`commit` が残した状態。そのままだと次の commit で
        書いた問いが全部落ちる、2026-09-12 (b))。
        """
        if self.exists():
            self._store.load()
            manifest = self._store.manifest
            if (
                not (self.directory / "events").exists()
                and manifest.folded_through != EventPosition()
                and not manifest.readonly
            ):
                logger.info(
                    "pseudo_queries[%s]: resetting a stale fold cursor left by an "
                    "earlier commit (events/ is gone)", self.package_id,
                )
                manifest.folded_through = EventPosition()
                manifest.events_since_snapshot = 0
                self._store.save_manifest()
        self._loaded = True

    def close(self) -> None:
        self._store.close()

    def set_embedding_backend(self, backend: "EmbeddingBackend | None") -> None:
        self._store.embedding_backend = backend

    @property
    def store(self) -> EvidenceStore:
        return self._store

    def __len__(self) -> int:
        return len(self._store) if self.exists() else 0

    def _snapshot_target_ids(self, *, live_only: bool = True) -> set[str]:
        snapshot = self._store.snapshot if self.exists() else None
        out: set[str] = set()
        if snapshot is None:
            return out
        for row in range(len(snapshot)):
            raw = snapshot.raw_at(row) or {}
            target = (raw.get("attrs") or {}).get("target_id")
            if isinstance(target, str) and target and (not live_only or self._is_live(target)):
                out.add(target)
        return out

    def orphan_count(self) -> int:
        """本体 snapshot に対応チャンクの無い問いの対象数 (commit で消える)。"""
        all_targets = self._snapshot_target_ids(live_only=False)
        return len(all_targets) - len(self._snapshot_target_ids())

    # ── 参照 ──

    def covered_target_ids(self) -> set[str]:
        """問いを持つ対象チャンク id の集合。"""
        return self._snapshot_target_ids() | self._pending_targets

    # ── 書き込み (sleep-time だけ) ──

    def add(
        self, target_id: str, questions: Iterable[str], *, hinted_from: int | None = None,
    ) -> int:
        """対象チャンクの問いを put する。``commit()`` で版を積むまで検索には出ない。

        ``hinted_from`` 以降の位置は取りこぼした問いの言い換え (``attrs.from_hint``、
        f_01 §6.3 / §6.4)。未充足でも先頭に置いてよい唯一の問い。
        """
        if not self.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            self._store.load()
        elif not self._loaded:
            self.load()
        now = utc_now()
        confidence = derive_confidence("document", 0)
        count = 0
        for position, question in enumerate(questions):
            text = (question or "").strip()
            if not text:
                continue
            attrs: dict[str, Any] = {
                "target_id": target_id,
                "package_id": self.package_id,
                "embed_as_query": True,
                "embed_mode": PSEUDO_QUERY_EMBED_MODE,
            }
            if hinted_from is not None and position >= hinted_from:
                attrs["from_hint"] = True
            record = Evidence(
                id=pseudo_query_id(target_id, position, text),
                kind=PSEUDO_QUERY_KIND,
                store="corpus",
                text=text,
                origin="document",
                observed_at=now,
                created_at=now,
                confidence=confidence,
                attrs=attrs,
            )
            self._store.put(record, by="pseudo_query_gen")
            count += 1
        if count:
            self._pending_targets.add(target_id)
        return count

    async def commit(self) -> int:
        """put した問いを snapshot に畳み、新規行だけ埋め込む。孤児は新しい版に書かない。"""
        if not self.exists():
            return 0
        if not self._loaded:
            self.load()
        orphans = self.orphan_count()
        if self._store.manifest.events_since_snapshot <= 0 and orphans <= 0:
            return 0
        if orphans:
            logger.info(
                "pseudo_queries[%s]: dropping %d orphaned target(s) whose chunk is gone",
                self.package_id, orphans,
            )
        await self._store.create_snapshot()
        # corpus 本体と同じく事象ログは畳んだら捨てる (版が履歴そのもの、
        # c_16 §2.1)。**畳み込みの位置も先頭へ戻す** — 位置を残すと次の
        # commit は作り直された事象ファイルの「前回の行数」以降しか読まず、
        # 書いた問いが全部落ちる (2026-09-12 (b): 3 サイクル分の Step 5.9 が
        # 同じ 20 件の複製 snapshot になり、充足率が一度も増えなかった)。
        shutil.rmtree(str(self.directory / "events"), ignore_errors=True)
        self._store.manifest.folded_through = EventPosition()
        self._store.manifest.events_since_snapshot = 0
        self._store.save_manifest()
        self._pending_targets.clear()
        return len(self._store)

    # ── 検索 ──

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float, bool]]:
        """問い↔問いの cosine 順に ``(対象チャンク id, cosine, from_hint)`` を返す。

        同じ対象チャンクは最良の問いだけ残す。順位式のスコアは使わない —
        ここで返す cosine は問い同士のもので、順位付けは呼出側が対象チャンク
        本体の値で行う (f_01 §6.3)。``from_hint`` は最良の問いが取りこぼした
        問いの言い換えかどうか。
        """
        if not self.exists() or top_k <= 0 or len(self._store) == 0:
            return []
        rows = self._store.search("", query_vec, top_k=top_k * 2)
        snapshot = self._store.snapshot
        if snapshot is None:
            return []
        best: dict[str, tuple[float, bool]] = {}
        for row, cosine, _score in rows:
            raw = snapshot.raw_at(row) or {}
            attrs = raw.get("attrs") or {}
            target = attrs.get("target_id")
            if not isinstance(target, str) or not target or not self._is_live(target):
                continue
            if target not in best or cosine > best[target][0]:
                best[target] = (float(cosine), bool(attrs.get("from_hint")))
        ranked = sorted(best.items(), key=lambda item: -item[1][0])
        return [(target, cosine, hinted) for target, (cosine, hinted) in ranked[:top_k]]


__all__ = [
    "PSEUDO_QUERIES_DIR",
    "PSEUDO_QUERY_EMBED_MODE",
    "PSEUDO_QUERY_KIND",
    "PseudoQueryIndex",
    "pseudo_query_id",
]
