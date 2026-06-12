"""ShortTermMemory の JSON 永続化

`backend.free.memory.stores.short_term.ShortTermMemory` からドメインロジックを
分離するための infra 層。`ShortTermMemoryStore` は MemoryNote の
シリアライズ / デシリアライズと JSON ファイル I/O のみを担い、
ドメインルール (スコアリング・ノート結合・キャッシュ等) は持たない。

レイヤー責務:
- `ShortTermMemory`        — ドメイン (絶対参照のスコアリング、キャッシュ更新)
- `ShortTermMemoryStore`   — インフラ (JSON 永続化、ファイル I/O)

このため `ShortTermMemoryStore` は import 時に `ShortTermMemory` を参照せず、
`MemoryNote` のみに依存する (循環依存防止 + 単体テスト可能性確保)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.free.memory.stores.short_term import MemoryNote
from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("memory.short_term_store")


class ShortTermMemoryStore:
    """ShortTermMemory の純粋な永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    @staticmethod
    def serialize(notes: dict[str, MemoryNote]) -> list[dict]:
        """`MemoryNote` 辞書を JSON-serializable な list[dict] に変換する。

        埋め込みベクトルは `tolist()` で list 化し、None ノートはそのまま保持。
        """
        return [_note_to_dict(note) for note in notes.values()]

    @staticmethod
    def deserialize(data: list[dict]) -> dict[str, MemoryNote]:
        """list[dict] から `MemoryNote` 辞書を再構築する。"""
        notes: dict[str, MemoryNote] = {}
        for d in data:
            note = _note_from_dict(d)
            notes[note.id] = note
        return notes

    @staticmethod
    def save(notes: dict[str, MemoryNote], path: str | Path) -> None:
        """`notes` を JSON ファイルに書き出す。親ディレクトリは自動作成。"""
        path = Path(path)
        data = ShortTermMemoryStore.serialize(notes)
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved %d notes to %s", len(data), path)

    @staticmethod
    def load(path: str | Path) -> dict[str, MemoryNote] | None:
        """JSON ファイルから `MemoryNote` 辞書を読み込む。

        ファイルが存在しない場合は `None` を返す (空辞書とは区別する)。
        呼び出し側は `None` を「ファイル未存在 = 既存状態を保持」と解釈できる。
        """
        path = Path(path)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        notes = ShortTermMemoryStore.deserialize(data)
        logger.info("Loaded %d notes from %s", len(notes), path)
        return notes


# ──────────────────────────────────────────────────────────────────────────
# private serialize / deserialize helpers (一行ずつ純粋関数として保つ)
# ──────────────────────────────────────────────────────────────────────────


def _note_to_dict(note: MemoryNote) -> dict:
    return {
        "id": note.id,
        "content": note.content,
        "keywords": note.keywords,
        "tags": note.tags,
        "embedding": note.embedding.tolist() if note.embedding is not None else None,
        "lightmem_score": note.lightmem_score,
        "created_at": note.created_at,
        "accessed_at": note.accessed_at,
        "access_count": note.access_count,
        "session_id": note.session_id,
        "context_description": note.context_description,
        "evolution_pending": note.evolution_pending,
        "conflict_candidate": note.conflict_candidate,
        "conflict_partner_id": note.conflict_partner_id,
        # EvorefMem 拡張
        "source": note.source,
        "confidence": note.confidence,
        "pin_flag": note.pin_flag,
        "pin_reason": note.pin_reason,
        "extracted_fact_ids": list(note.extracted_fact_ids),
        "private": note.private,
        "mode": note.mode,
        "project_id": note.project_id,
        "is_tool_output": note.is_tool_output,
        "is_code_block": note.is_code_block,
        "extraction_skipped": note.extraction_skipped,
        "extraction_skip_reason": note.extraction_skip_reason,
        # executable command 学習用
        "tool_command": note.tool_command,
        "tool_command_name": note.tool_command_name,
        "tool_command_success": note.tool_command_success,
        # 統合追加
        "task_status": note.task_status,
        "task_id": note.task_id,
        "depends_on": list(note.depends_on),
        "failure_signature": note.failure_signature,
        "trace_id": note.trace_id,
        "links": list(note.links),
        "cluster_id": note.cluster_id,
        "url_curated_at": note.url_curated_at,
        "command_curated_at": note.command_curated_at,
    }


def _note_from_dict(d: dict) -> MemoryNote:
    emb = d.get("embedding")
    return MemoryNote(
        id=d["id"],
        content=d["content"],
        keywords=d.get("keywords", []),
        tags=d.get("tags", []),
        embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
        lightmem_score=d.get("lightmem_score", 0.5),
        created_at=d.get("created_at", 0.0),
        accessed_at=d.get("accessed_at", 0.0),
        access_count=d.get("access_count", 0),
        session_id=d.get("session_id", ""),
        context_description=d.get("context_description", ""),
        evolution_pending=d.get("evolution_pending", True),
        conflict_candidate=d.get("conflict_candidate", False),
        conflict_partner_id=d.get("conflict_partner_id"),
        # EvorefMem 拡張
        source=d.get("source", "user"),
        confidence=d.get("confidence", 1.0),
        pin_flag=d.get("pin_flag", False),
        pin_reason=d.get("pin_reason"),
        extracted_fact_ids=list(d.get("extracted_fact_ids", [])),
        private=d.get("private", False),
        mode=d.get("mode", "chat"),
        project_id=d.get("project_id"),
        is_tool_output=d.get("is_tool_output", False),
        is_code_block=d.get("is_code_block", False),
        extraction_skipped=d.get("extraction_skipped", False),
        extraction_skip_reason=d.get("extraction_skip_reason"),
        # executable command 学習用
        tool_command=d.get("tool_command"),
        tool_command_name=d.get("tool_command_name"),
        tool_command_success=d.get("tool_command_success"),
        # 統合追加
        task_status=d.get("task_status"),
        task_id=d.get("task_id"),
        depends_on=list(d.get("depends_on", [])),
        failure_signature=d.get("failure_signature"),
        trace_id=d.get("trace_id"),
        # 他フィールドと同じく防御的に読む。links / cluster_id だけ直アクセスだと
        # 1 ノートで両キーを欠くスナップショットが KeyError で deserialize 全体を
        # 失敗させ、起動時ロードが STM を空で開始して前回の全ノートを失う。
        links=list(d.get("links", [])),
        cluster_id=d.get("cluster_id"),
        url_curated_at=d.get("url_curated_at"),
        command_curated_at=d.get("command_curated_at"),
    )
