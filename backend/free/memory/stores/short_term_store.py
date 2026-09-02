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

        ``private=True`` のノートは **書き出さない**。プライベートセッションの
        ターンは WM/STM に留めてセッション終了で揮発する契約
        (``PrivateSessionConfig``) で、平文でディスクに残してはいけない。

        ``notes.values()`` は一度 list に写す — 保存は executor スレッドで走る
        ため、チャット側の ``absorb`` と重なると dict の走査中変更で落ちる。
        """
        return [
            _note_to_dict(note) for note in list(notes.values())
            if not note.private
        ]

    @staticmethod
    def deserialize(data: list[dict]) -> dict[str, MemoryNote]:
        """list[dict] から `MemoryNote` 辞書を再構築する。"""
        notes: dict[str, MemoryNote] = {}
        for d in data:
            note = _note_from_dict(d)
            notes[note.id] = note
        return notes

    @staticmethod
    def save(
        notes: dict[str, MemoryNote],
        path: str | Path,
        *,
        allow_empty: bool = False,
    ) -> bool:
        """`notes` を JSON ファイルに書き出す。親ディレクトリは自動作成。

        書き出したら ``True``、空上書きガードで拒否したら ``False``。

        **空の STM で既存スナップショットを上書きしない** (``allow_empty`` で解除)。

        ロードが何らかの理由で 0 件になった直後、SleepTimeWorker とシャットダウン
        フックは無条件に保存する。結果、一度でも空で起動すると次の保存で
        ディスク上の全ノートが消え、以後どれだけ再起動しても空のまま復帰しない
        (自己増殖する取りこぼし)。

        2026-08-16 ライブ監査での実際の進行:

            23:38 Saved 0 notes    ← 空の STM がスナップショットを潰した
            00:34 Loaded 0 notes   ← 当然 0 件で起動
            00:34 Memory threshold calibration skipped: only 0 notes with embeddings
            00:43〜 Step 2 STM: 0 hits (14 ターン連続)
            → MemoryInjector budget=800 に対し used=0 が 40 ターン中 37 ターン

        ``_note_from_dict`` の防御的読取 (links / cluster_id の KeyError 対策) は
        「読めなくなる」側の再発を塞いだが、**読めなかった後に書き潰す**側は
        素通しだった。保存側にもガードを置く。
        """
        path = Path(path)
        data = ShortTermMemoryStore.serialize(notes)
        if not data and not allow_empty:
            existing = _peek_note_count(path)
            if existing:
                logger.warning(
                    "Refusing to overwrite %s: in-memory STM is empty but the "
                    "snapshot holds %d notes. Pass allow_empty=True to clear "
                    "intentionally.", path, existing,
                )
                return False
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved %d notes to %s", len(data), path)
        return True

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


def _peek_note_count(path: Path) -> int:
    """スナップショットの件数だけを数える (空上書きガード用)。

    ノート本体の 95% は埋め込みベクトルで実測 1.5MB になるが、この読取は
    「保存しようとしている STM が空」のときにしか走らないので稀。読めない
    ファイルは 0 扱いにして保存を通す (壊れたスナップショットに縛られない)。
    """
    try:
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception as e:  # pragma: no cover - 壊れたファイルは上書きを許す
        logger.warning("Could not read existing STM snapshot %s: %s", path, e)
        return 0


def _note_to_dict(note: MemoryNote) -> dict:
    return {
        "id": note.id,
        "content": note.content,
        "keywords": note.keywords,
        "tags": note.tags,
        "embedding": note.embedding.tolist() if note.embedding is not None else None,
        "embed_failures": note.embed_failures,
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
        "is_correction": note.is_correction,
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
        "tool_command_source": note.tool_command_source,
        "tool_command_query": note.tool_command_query,
        "extraction_deferred": note.extraction_deferred,
        "trace_id": note.trace_id,
        "links": list(note.links),
        "cluster_id": note.cluster_id,
        "url_curated_at": note.url_curated_at,
        "command_curated_at": note.command_curated_at,
        "assertion_curated_at": note.assertion_curated_at,
        "assertion_slug": note.assertion_slug,
        "conflict_fail_count": note.conflict_fail_count,
        "conflict_cooldown_until": note.conflict_cooldown_until,
    }


def _sanitize_tags(tags: list[str], source: str) -> list[str]:
    """保存済みノートのタグを現行ルールへ揃える (読込時の遡及修復)。

    ``NoteBuilder.auto_tag`` は assistant 発話に ``fact`` を付けなくなったが、
    **既に保存されたノートには残っている**。``fact`` ノートは
    ``MemoryInjector`` がプロンプトへ「(過去の記録)」として注入し、システム
    プロンプトが「[関連する記憶] は自分の記憶より優先」と規定しているため、
    過去の誤答が以後のターンでモデル自身の知識を上書きし続ける
    (2026-08-15 ライブ監査で三名園の誤答が別セッションへ注入されていた)。

    生成側の修正だけでは既存データが直らないので、読込時にも同じルールを
    適用する。冪等で、次回保存時にファイルからも消える。
    """
    if source != "assistant" or not tags:
        return tags
    from backend.free.memory.notes.note_builder import NoteBuilder

    return [t for t in tags if t not in NoteBuilder.ASSISTANT_EXCLUDED_TAGS]


def _note_from_dict(d: dict) -> MemoryNote:
    emb = d.get("embedding")
    source = d.get("source", "user")
    return MemoryNote(
        id=d["id"],
        content=d["content"],
        keywords=d.get("keywords", []),
        tags=_sanitize_tags(d.get("tags", []), source),
        embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
        embed_failures=int(d.get("embed_failures", 0) or 0),
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
        source=source,
        confidence=d.get("confidence", 1.0),
        pin_flag=d.get("pin_flag", False),
        pin_reason=d.get("pin_reason"),
        is_correction=d.get("is_correction", False),
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
        tool_command_source=d.get("tool_command_source"),
        tool_command_query=d.get("tool_command_query"),
        extraction_deferred=bool(d.get("extraction_deferred", False)),
        trace_id=d.get("trace_id"),
        # 他フィールドと同じく防御的に読む。links / cluster_id だけ直アクセスだと
        # 1 ノートで両キーを欠くスナップショットが KeyError で deserialize 全体を
        # 失敗させ、起動時ロードが STM を空で開始して前回の全ノートを失う。
        links=list(d.get("links", [])),
        cluster_id=d.get("cluster_id"),
        url_curated_at=d.get("url_curated_at"),
        command_curated_at=d.get("command_curated_at"),
        assertion_curated_at=d.get("assertion_curated_at"),
        assertion_slug=d.get("assertion_slug"),
        conflict_fail_count=d.get("conflict_fail_count", 0),
        conflict_cooldown_until=d.get("conflict_cooldown_until"),
    )
