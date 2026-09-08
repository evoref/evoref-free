"""sleep-time の作業領域 (``short`` tier のノートをメモリ上で扱う)。

sleep-time の一部の工程 — 競合解決 (Step 6) / ノート進化 (Step 7) /
ファクト抽出 (Step 8) / キュレーター (Step 8.4-8.6) — は、ノートを **その場で
書き換える** 前提で書かれている (``notes`` dict を触り、処理済みマーカーを
立てる)。事象ログは追記のみなので、その間の変更をここで受け止め、サイクルの
最後に :meth:`flush` が ``put`` / ``retract`` へ落とす。

- 読み込むのは ``short`` tier のアクティブノートだけ (``long`` は畳んだ後の
  ものなので工程の入力にしない)
- 変わったノートは ``put`` (全置換)、消えたノートは ``retract``
  (物理削除しない、c_16 §3 の状態遷移)
- private ノートは既定で読み込まない — 会話履歴に private ターンは残らない
  ので、そもそもストアにも入らない

phase 4 で SemMem 側の工程を Evidence 直読みに書き換えたら、この層は消える。
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.memory.episodic.note import MemoryNote
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.episodic.store import EpisodicStore

logger = get_logger("memory.episodic.workspace")

#: 差分判定から外すフィールド。``embedding`` は永続化しない作業値、
#: ``accessed_at`` は ``touch`` 事象 (``last_used_at``) が持つ。
_VOLATILE_FIELDS: frozenset[str] = frozenset({"embedding", "accessed_at"})


def _fingerprint(note: MemoryNote) -> tuple:
    """ノートの内容指紋 (差分検出用)。ndarray は含めない。"""
    return tuple(
        repr(getattr(note, f.name, None))
        for f in fields(MemoryNote)
        if f.name not in _VOLATILE_FIELDS
    )


class EpisodicWorkspace:
    """``short`` tier のノートを載せた可変の作業領域。

    旧 ``ShortTermMemory`` が持っていた面のうち、sleep-time の工程が実際に
    呼ぶものだけを持つ (``notes`` / ``mark_dirty`` / ``dirty`` /
    ``retrieve_top_k`` / ``max_notes``)。
    """

    def __init__(
        self,
        store: "EpisodicStore",
        notes: dict[str, MemoryNote],
        *,
        max_notes: int = 100,
    ) -> None:
        self.store = store
        self.notes: dict[str, MemoryNote] = notes
        self.max_notes = max_notes
        self._baseline: dict[str, tuple] = {
            note_id: _fingerprint(note) for note_id, note in notes.items()
        }
        self._dirty = False

    @classmethod
    def load(
        cls, store: "EpisodicStore", *, include_private: bool = False,
    ) -> "EpisodicWorkspace":
        """ストアの ``short`` ノートを読み込んで作業領域を作る。"""
        notes = {
            note.id: note
            for note in store.iter_notes(tier="short", include_private=include_private)
        }
        vectors = store.vectors_for(list(notes))
        for note_id, vector in vectors.items():
            notes[note_id].embedding = vector
        logger.info(
            "Episodic workspace: %d short note(s) loaded (%d with vectors)",
            len(notes), len(vectors),
        )
        return cls(store, notes)

    # ── 旧 ShortTermMemory の面 ──

    def mark_dirty(self) -> None:
        """ノート集合が変わったことを記録する。"""
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def add(self, note: MemoryNote) -> MemoryNote:
        """作業領域へノートを足す (``flush`` で ``put`` になる)。"""
        self.notes[note.id] = note
        self.mark_dirty()
        return note

    def retrieve_top_k(
        self, query_vec: np.ndarray, k: int = 3, *, include_private: bool = False,
    ) -> list[tuple[MemoryNote, float]]:
        """作業領域内の素の cosine 上位 k 件。

        合成スコア (旧 ``0.6cos + 0.4lightmem`` + pin 加点) は廃止した
        (c_16 §7.1: ゲートも順位も素の cosine)。ここはノート進化が「近い
        ノート」を集めるためだけに使う。
        """
        if query_vec is None or not self.notes:
            return []
        query = np.asarray(query_vec, dtype=np.float32).ravel()
        scored: list[tuple[MemoryNote, float]] = []
        for note in self.notes.values():
            if note.embedding is None:
                continue
            if note.private and not include_private:
                continue
            vector = np.asarray(note.embedding, dtype=np.float32).ravel()
            if vector.shape != query.shape:
                continue
            denom = float(np.linalg.norm(vector) * np.linalg.norm(query))
            if denom <= 0:
                continue
            scored.append((note, float(vector @ query / denom)))
        scored.sort(key=lambda item: -item[1])
        return scored[:k]

    # ── 書き戻し ──

    def flush(self) -> dict[str, int]:
        """変更を事象へ落とす。``{"put": n, "retracted": m}`` を返す。"""
        put = 0
        for note_id, note in self.notes.items():
            fingerprint = _fingerprint(note)
            if self._baseline.get(note_id) == fingerprint:
                continue
            self.store.put_note(note, tier=note.tier or "short")
            self._baseline[note_id] = fingerprint
            put += 1
        retracted = 0
        for note_id in list(self._baseline):
            if note_id in self.notes:
                continue
            self.store.retract_note(note_id, "workspace:removed")
            self._baseline.pop(note_id, None)
            retracted += 1
        self._dirty = False
        if put or retracted:
            logger.info(
                "Episodic workspace flushed: %d put, %d retracted", put, retracted,
            )
        return {"put": put, "retracted": retracted}

    def unembedded(self) -> list[MemoryNote]:
        """まだベクトルを持たないノート (競合検出 / 進化の対象外になる)。"""
        return [note for note in self.notes.values() if note.embedding is None]

    def __len__(self) -> int:
        return len(self.notes)

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"<EpisodicWorkspace notes={len(self.notes)} dirty={self._dirty}>"


def note_dicts(workspace: EpisodicWorkspace) -> list[dict[str, Any]]:
    """観測用にノートを dict 化する (テスト / 統計)。"""
    return [
        {f.name: getattr(note, f.name) for f in fields(MemoryNote)
         if f.name != "embedding"}
        for note in workspace.notes.values()
    ]


__all__ = ["EpisodicWorkspace", "note_dicts"]
