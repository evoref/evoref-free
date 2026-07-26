"""会話履歴の自動保存・アーカイブ・検索・圧縮

セッション終了時に自動保存し、sleep-time で要約・圧縮を行う。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from backend.free.history.utils import parse_iso, snippet_around
from backend.free.rag.bm25_retriever import tokenize_ja
from backend.io import atomic_write_text
from backend.log_config import get_logger
from backend.utils import utc_now_dt

logger = get_logger("history.manager")

# トークン重なり判定で「一致」とみなす、クエリの unique トークンに対する
# 最小重なり割合 (端数切上げ)。完全部分文字列一致に失敗した自然文クエリの
# 救済用フォールバックだが、日本語 bi-gram は「の天」「日の」のような助詞
# 絡みの連結詞的トークンが無関係なテキスト間でも偶然重なりやすいため、
# 過剰マッチを避ける程度には厳しめに取る (レビューで実際に false positive
# の可能性を指摘された)。
_TOKEN_OVERLAP_RATIO = 0.7
# クエリの unique トークン数がこれ未満ならトークン重なりフォールバックを
# 適用しない (= マッチ扱いにしない)。トークン数が少なすぎると、たまたま
# 1〜2 個の助詞絡み bi-gram が重なっただけで誤マッチしうるため。単語程度の
# 短いクエリは完全部分文字列一致 (高速パス) で既に救済されている前提。
_TOKEN_OVERLAP_MIN_QUERY_TOKENS = 5


@lru_cache(maxsize=64)
def _tokenize_cached(text: str) -> frozenset[str]:
    """``tokenize_ja`` の結果をキャッシュする (空白区切りは区間ごとにトークン化)。

    1 回の検索 (search_sessions/list_sessions) で同一のクエリ文字列に対して
    最大で「対象エントリ数 × 3 (summary/topics/search_text)」回呼ばれうる
    ため、同一入力の再トークン化を避ける。

    ``tokenize_ja`` は空白を除去してから日本語 bi-gram を切るため、語を空白で
    並べたキーワードクエリ (``search_history`` に渡る実引数はこの形が既定) では
    境界をまたぐ bi-gram が混入する。これは実在テキストに出現しえないトークン
    なので ``_TOKEN_OVERLAP_RATIO`` の分母だけを押し上げ、重なり率を構造的に
    達成不能にする (2026-07-26 実測: 「伝 ツール 名前 用途 正確 教」は 10
    トークン中 5 個が境界またぎの「伝ツ」「ル名」「前用」「途正」「確教」で、
    到達しうる重なりの上限 0.5 < 必要値 0.7 = どんな履歴にもマッチしない)。
    空白区間ごとにトークン化して境界またぎを作らない。
    """
    tokens: set[str] = set()
    for segment in text.split():
        tokens.update(tokenize_ja(segment))
    return frozenset(tokens)


def _text_matches_query(text: str, query_lower: str) -> bool:
    """テキストがクエリにマッチするか (完全部分文字列一致 → トークン重なりの順で判定)。

    まず従来通りの完全部分文字列一致を試す (高速・単語クエリでの後方互換)。
    失敗した場合は ``tokenize_ja`` (ASCII 語 + 日本語文字 bi-gram) で
    クエリ・対象テキスト双方をトークン化し、クエリの unique トークンの
    ``_TOKEN_OVERLAP_RATIO`` 以上が対象テキストに含まれるかで判定する
    (クエリのトークン数が ``_TOKEN_OVERLAP_MIN_QUERY_TOKENS`` 未満なら
    フォールバック自体を適用しない)。

    自然文クエリ (助詞・句読点を含む) は過去ログ原文と一字一句連続一致
    することがほぼ無く、完全部分文字列一致のみだと長期記憶検索のヒット率が
    著しく低くなる (実インシデント: 「私の好きなプログラミング言語は？」が
    ヒットせず、「Rust」という単語だけならヒットする現象を確認済み)。
    """
    if not text:
        return False
    text_lower = text.lower()
    if query_lower in text_lower:
        return True
    query_tokens = _tokenize_cached(query_lower)
    if len(query_tokens) < _TOKEN_OVERLAP_MIN_QUERY_TOKENS:
        return False
    target_tokens = _tokenize_cached(text_lower)
    if not target_tokens:
        return False
    overlap = len(query_tokens & target_tokens)
    required = math.ceil(len(query_tokens) * _TOKEN_OVERLAP_RATIO)
    return overlap >= required


@dataclass
class SessionData:
    """セッションデータ"""
    session_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_sec: int = 0
    mode: str = "chat"
    modes_used: list[str] = field(default_factory=list)
    instance_name: str = "evoref"
    base_model: str = ""
    source: str = "auto"  # "auto" | "manual"
    turns: list[dict] = field(default_factory=list)
    turn_count: int = 0
    context_files: list[str] = field(default_factory=list)
    cartridge_ids: list[str] = field(default_factory=list)
    token_info: dict = field(default_factory=dict)
    summary: str | None = None
    summary_embedding: list[float] | None = None
    #: ``summary`` を生成した時点の ``turn_count``。会話が進んで turn_count が
    #: これを上回ったら sleep-time が要約を作り直す (会話途中で要約が固定され、
    #: 後半の訂正が要約に反映されないのを防ぐ)。
    summary_turn_count: int = 0
    topics: list[str] = field(default_factory=list)
    archived_at: str = ""
    # SemMem への昇格済フラグ
    promoted_to_semmem: bool = False
    project_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> SessionData:
        """dict から SessionData を復元"""
        return cls(
            session_id=data.get("session_id", ""),
            started_at=data.get("started_at", ""),
            ended_at=data.get("ended_at", ""),
            duration_sec=data.get("duration_sec", 0),
            mode=data.get("mode", "chat"),
            modes_used=data.get("modes_used", []),
            instance_name=data.get("instance_name", "evoref"),
            base_model=data.get("base_model") or "",
            source=data.get("source", "auto"),
            turns=data.get("turns", []),
            turn_count=data.get("turn_count", 0),
            context_files=data.get("context_files", []),
            cartridge_ids=data.get("cartridge_ids", []),
            token_info=data.get("token_info", {}),
            summary=data.get("summary"),
            summary_embedding=data.get("summary_embedding"),
            summary_turn_count=int(data.get("summary_turn_count", 0) or 0),
            topics=data.get("topics", []),
            archived_at=data.get("archived_at", ""),
            promoted_to_semmem=bool(data.get("promoted_to_semmem", False)),
            project_id=data.get("project_id"),
        )


@dataclass
class IndexEntry:
    """インデックスエントリ"""
    session_id: str
    file: str
    started_at: str
    duration_sec: int
    mode: str
    turn_count: int
    summary: str | None = None
    #: ``summary`` を生成した時点の ``turn_count`` (0 = 未要約)。
    summary_turn_count: int = 0
    topics: list[str] = field(default_factory=list)
    size_bytes: int = 0
    search_text: str = ""
    # SemMem への昇格済フラグ
    promoted_to_semmem: bool = False
    project_id: str | None = None


@dataclass
class HistoryIndex:
    """アーカイブインデックス"""
    updated_at: str = ""
    total_sessions: int = 0
    total_turns: int = 0
    total_size_mb: float = 0.0
    sessions: list[IndexEntry] = field(default_factory=list)


_SEARCH_TEXT_MAX = 5000  # インデックスに保存する検索テキストの最大文字数


def _atomic_write_json(filepath: Path, data: dict) -> None:
    """JSON を atomic に書き込む (:func:`backend.io.atomic_write_text` 経由)。

    書き込み中のクラッシュでもデータが失われないことを保証する。Windows の
    書き込み競合時リトライは ``backend.io._retry`` が担う。
    """
    atomic_write_text(filepath, json.dumps(data, ensure_ascii=False, indent=2))


def _build_search_text(
    session: SessionData,
    max_len: int = _SEARCH_TEXT_MAX,
) -> str:
    """summary・topics・全ターンの content を結合して検索用テキストを生成"""
    parts: list[str] = []
    total = 0

    if session.summary:
        parts.append(session.summary)
        total += len(session.summary)
    if session.topics:
        topic_str = " ".join(session.topics)
        parts.append(topic_str)
        total += len(topic_str)

    for t in session.turns:
        content = t.get("content", "")
        if not content:
            continue
        remaining = max_len - total
        if remaining <= 0:
            break
        parts.append(content[:remaining])
        total += len(content[:remaining])
    return "\n".join(parts)


def _should_skip_session(session: SessionData, auto_save: bool) -> str | None:
    """保存スキップ判定（バリデーション）

    Returns:
        スキップ理由の文字列。保存すべき場合は None。
    """
    if not auto_save:
        return f"Auto-save disabled, skipping session {session.session_id}"
    if not session.turns or session.turn_count < 1:
        return f"Skipping empty session {session.session_id}"
    user_turns = [t for t in session.turns if t.get("role") == "user"]
    if len(user_turns) < 1:
        return f"Skipping non-conversation session {session.session_id}"
    return None


def _complete_session_metadata(session: SessionData) -> None:
    """セッション終了情報を補完"""
    if not session.ended_at:
        session.ended_at = _now_iso()
    if not session.archived_at:
        session.archived_at = _now_iso()
    if not session.turn_count:
        session.turn_count = len(session.turns)


def _score_entry(entry: IndexEntry, query_lower: str) -> float:
    """エントリの検索スコア計算（純粋関数）"""
    score = 0.0
    if _text_matches_query(entry.summary, query_lower):
        score += 1.0
    if any(_text_matches_query(t, query_lower) for t in entry.topics):
        score += 0.5
    if _text_matches_query(entry.search_text, query_lower):
        score += 0.5
    return score


def _find_matched_turns(session: SessionData, query_lower: str) -> list[dict]:
    """ターンマッチ抽出（純粋関数）"""
    matched: list[dict] = []
    for i, turn in enumerate(session.turns):
        content = turn.get("content", "")
        if not _text_matches_query(content, query_lower):
            continue
        # snippet_around はクエリの完全部分文字列一致を前提とするため、
        # トークン重なりのみで一致したケース (完全一致しない) は先頭からの
        # プレビューにフォールバックする。
        preview = (
            snippet_around(content, query_lower, context=50)
            if query_lower in content.lower()
            else (content[:100] + "…" if len(content) > 100 else content)
        )
        matched.append({
            "index": i,
            "role": turn.get("role", ""),
            "content_preview": preview,
        })
    return matched


def _summarize_session_file(filepath: Path) -> float:
    """要約化（ターン削除）、解放 MB を返す"""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("turns"):
        return 0.0
    old_size = filepath.stat().st_size
    data["turns"] = []
    _atomic_write_json(filepath, data)
    new_size = filepath.stat().st_size
    return (old_size - new_size) / (1024 * 1024)


def _compress_session_file(
    filepath: Path,
    preview_chars: int = 100,
) -> float:
    """圧縮保持（アシスタントターン切り詰め）、解放 MB を返す

    Args:
        filepath: セッションファイルのパス
        preview_chars: 圧縮時に保持する先頭文字数
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    compressed = False
    for turn in data.get("turns", []):
        if turn.get("role") == "assistant" and not turn.get("compressed"):
            content = turn.get("content", "")
            if len(content) > preview_chars * 2:
                turn["content"] = content[:preview_chars] + "…（圧縮済み）"
                turn["compressed"] = True
                turn["original_length"] = len(content)
                compressed = True
    if not compressed:
        return 0.0
    old_size = filepath.stat().st_size
    _atomic_write_json(filepath, data)
    new_size = filepath.stat().st_size
    return (old_size - new_size) / (1024 * 1024)


class HistoryManager:
    """会話履歴のアーカイブ管理"""

    def __init__(self, history_dir: Path, config: dict | None = None):
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        history_cfg = (config or {}).get("history", {})
        self.auto_save: bool = history_cfg.get("auto_save", True)
        self.checkpoint_interval: int = history_cfg.get("checkpoint_interval", 10)
        self.retention_full_days: int = history_cfg.get("retention_full_days", 90)
        self.retention_compressed_days: int = history_cfg.get("retention_compressed_days", 365)
        self.max_storage_mb: float = history_cfg.get("max_storage_mb", 200)
        self.compress_preview_chars: int = history_cfg.get("compress_preview_chars", 100)

        self._checkpoint_dir = history_dir / ".checkpoint"
        self._index: HistoryIndex | None = None

    # ── 保存 ──

    def save_session(self, session: SessionData) -> Path | None:
        """セッションをアーカイブに保存

        Returns:
            保存先パス（保存しなかった場合は None）
        """
        skip_reason = _should_skip_session(session, self.auto_save)
        if skip_reason:
            logger.debug(skip_reason)
            return None

        _complete_session_metadata(session)
        filepath = self._resolve_session_path(session)
        _atomic_write_json(filepath, asdict(session))

        entry = self._build_index_entry(session, filepath)
        self._update_index(entry)
        self._clear_checkpoint(session.session_id)

        logger.info("Session saved: %s (%d turns, %d bytes)",
                     session.session_id, session.turn_count, entry.size_bytes)
        return filepath

    def _resolve_session_path(self, session: SessionData) -> Path:
        """保存先パスの決定"""
        started = parse_iso(session.started_at) or utc_now_dt()
        month_dir = self.history_dir / started.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{started.strftime('%Y%m%d_%H%M%S')}_{session.session_id[:8]}.json"
        return month_dir / filename

    def _build_index_entry(self, session: SessionData, filepath: Path) -> IndexEntry:
        """インデックスエントリ構築"""
        size_bytes = filepath.stat().st_size
        rel_path = filepath.relative_to(self.history_dir).as_posix()
        return IndexEntry(
            session_id=session.session_id,
            file=rel_path,
            started_at=session.started_at,
            duration_sec=session.duration_sec,
            mode=session.mode,
            turn_count=session.turn_count,
            summary=session.summary,
            summary_turn_count=session.summary_turn_count,
            topics=session.topics,
            size_bytes=size_bytes,
            search_text=_build_search_text(session),
            promoted_to_semmem=session.promoted_to_semmem,
            project_id=session.project_id,
        )

    def mark_promoted_to_semmem(self, session_id: str) -> bool:
        """セッションを SemMem 昇格済としてマーク

        index 上のフラグを更新するとともに、セッション本体ファイルも
        書き戻す (再起動後に再昇格しないため)。

        Returns:
            マークに成功したら ``True``、未存在なら ``False``。
        """
        index = self._load_index()
        entry = next(
            (e for e in index.sessions if e.session_id == session_id),
            None,
        )
        if entry is None:
            return False
        entry.promoted_to_semmem = True
        self._save_index(index)
        # セッションファイル本体にも反映 (再起動耐性)
        try:
            session = self.get_session(session_id)
            if session is not None:
                session.promoted_to_semmem = True
                filepath = self.history_dir / entry.file
                if filepath.exists():
                    _atomic_write_json(filepath, asdict(session))
        except Exception as e:
            logger.warning(
                "mark_promoted_to_semmem: failed to update session file %s: %s",
                session_id, e,
            )
        return True

    # ── チェックポイント ──

    def save_checkpoint(self, session: SessionData) -> Path:
        """チェックポイントを保存"""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._checkpoint_dir / f"{session.session_id}.json"

        data = asdict(session)
        _atomic_write_json(filepath, data)

        logger.debug("Checkpoint saved: %s", session.session_id)
        return filepath

    def promote_checkpoints(self) -> int:
        """残存チェックポイントを正式アーカイブに昇格（起動時に呼ばれる）"""
        if not self._checkpoint_dir.exists():
            return 0

        promoted = 0
        for cp_file in self._checkpoint_dir.glob("*.json"):
            try:
                with open(cp_file, encoding="utf-8") as f:
                    data = json.load(f)
                session = SessionData.from_dict(data)
                if session.turns:
                    self.save_session(session)
                    promoted += 1
                cp_file.unlink()
            except Exception as e:
                logger.warning("Failed to promote checkpoint %s: %s", cp_file.name, e)

        if promoted:
            logger.info("Promoted %d checkpoints to archive", promoted)
        return promoted

    def _clear_checkpoint(self, session_id: str) -> None:
        """チェックポイントを削除"""
        cp = self._checkpoint_dir / f"{session_id}.json"
        if cp.exists():
            cp.unlink()

    # ── 取得 ──

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        mode: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        query: str | None = None,
    ) -> tuple[list[IndexEntry], int]:
        """セッション一覧

        Returns:
            (エントリリスト, 総件数)
        """
        index = self._load_index()
        entries = index.sessions

        # フィルタ
        if mode:
            entries = [e for e in entries if e.mode == mode]
        if date_from:
            entries = [e for e in entries if e.started_at >= date_from]
        if date_to:
            entries = [e for e in entries if e.started_at <= date_to]
        if query:
            q_lower = query.lower()
            matched: list[IndexEntry] = []
            for e in entries:
                if e.search_text:
                    if _text_matches_query(e.search_text, q_lower):
                        matched.append(e)
                else:
                    # search_text 未設定エントリは検索対象外（起動時に補完済みのはず）
                    logger.warning(
                        "Skipping entry without search_text: %s",
                        e.session_id,
                    )
            entries = matched

        # 新しい順
        entries.sort(key=lambda e: e.started_at, reverse=True)
        total = len(entries)

        return entries[offset:offset + limit], total

    def get_session(self, session_id: str) -> SessionData | None:
        """セッション詳細を取得"""
        index = self._load_index()
        entry = next((e for e in index.sessions if e.session_id == session_id), None)
        if entry is None:
            return None

        filepath = self.history_dir / entry.file
        if not filepath.exists():
            return None

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return SessionData.from_dict(data)

    def search_sessions(
        self,
        query: str,
        mode: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
        search_turns: bool = False,
        session_id: str | None = None,
    ) -> list[dict]:
        """セッション検索（全文検索）

        list_sessions でフィルタ済みのエントリをスコアリングし、
        search_turns=True 時はスコア上位 limit 件のみにターンマッチを適用する。

        ``session_id`` 指定時は該当セッションのみに絞り込む。「この会話で」
        等のセッション自己参照質問を他セッションの内容と混同しないための
        スコープ限定 (呼出元は ``ToolCallJudge._maybe_scope_session_search``)。
        """
        if session_id:
            # session_id 指定時はクエリ絞り込みを **かけずに** 対象セッションを
            # 取り、本体の turns を直接走査する。list_sessions(query=...) は
            # index の search_text (先頭 _SEARCH_TEXT_MAX 文字) しか見ないため、
            # 長いセッションでは後半の発言が索引に載らず、当のセッション自身が
            # session_id 判定に到達する前に脱落していた
            # (2026-07-25 実測: 100 ターン / 48,368 字のうち索引は 5,014 字 =
            #  先頭 14 ターンのみ。「訂正」の初出は 10,602 字目で拾えず、
            #  前日の別セッションが score 1.4 で 1 位になった)。
            entries, _ = self.list_sessions(
                limit=1000, mode=mode, date_from=date_from, date_to=date_to,
            )
            entries = [e for e in entries if e.session_id == session_id]
        else:
            entries, _ = self.list_sessions(
                limit=1000, mode=mode, date_from=date_from, date_to=date_to,
                query=query,
            )

        q_lower = query.lower()

        # list_sessions がクエリでフィルタ済みなのでスコアリングのみ実施
        scored: list[tuple[IndexEntry, float]] = []
        for entry in entries:
            score = _score_entry(entry, q_lower)
            scored.append((entry, max(score, 0.1)))

        # スコア降順でソートし、上位 limit 件に絞る
        scored.sort(key=lambda x: x[1], reverse=True)
        top_entries = scored[:limit]

        # ターンマッチは上位 N 件のみに適用（N+1 解消）
        results: list[dict] = []
        for entry, score in top_entries:
            matched_turns: list[dict] = []
            # session_id 指定時は索引を迂回してセッション本体を走査するため、
            # ターンマッチは必須 (これが唯一のヒット源になる)。
            if search_turns or session_id:
                session = self.get_session(entry.session_id)
                if session:
                    matched_turns = _find_matched_turns(session, q_lower)
                    score += 0.3 * min(len(matched_turns), 3)

            results.append({
                "session_id": entry.session_id,
                "started_at": entry.started_at,
                "mode": entry.mode,
                "summary": entry.summary,
                "relevance_score": score,
                "matched_turns": matched_turns,
            })

        # ターンマッチでスコアが変動した場合に再ソート
        if search_turns:
            results.sort(key=lambda r: r["relevance_score"], reverse=True)
        return results

    def delete_session(self, session_id: str) -> bool:
        """セッションを削除"""
        index = self._load_index()
        entry = next((e for e in index.sessions if e.session_id == session_id), None)
        if entry is None:
            return False

        filepath = self.history_dir / entry.file
        if filepath.exists():
            filepath.unlink()

        index.sessions = [e for e in index.sessions if e.session_id != session_id]
        self._save_index(index)

        logger.info("Session deleted: %s", session_id)
        return True

    def delete_sessions_batch(self, session_ids: list[str]) -> int:
        """複数セッションを一括削除

        Returns:
            削除したセッション数
        """
        index = self._load_index()
        target_ids = set(session_ids)
        to_delete = [e for e in index.sessions if e.session_id in target_ids]

        if not to_delete:
            return 0

        # インデックスから先に除去
        delete_ids = {e.session_id for e in to_delete}
        index.sessions = [
            e for e in index.sessions if e.session_id not in delete_ids
        ]
        self._save_index(index)

        # ファイル削除
        for entry in to_delete:
            filepath = self.history_dir / entry.file
            try:
                if filepath.exists():
                    filepath.unlink()
            except OSError as e:
                logger.warning("Failed to delete %s: %s", filepath, e)

        logger.info("Batch deleted %d sessions", len(to_delete))
        return len(to_delete)

    # ── 圧縮 ──

    def compact_sessions(self) -> dict:
        """保持ポリシーに基づく圧縮処理

        Returns:
            {"compressed": int, "summarized": int, "deleted": int, "freed_mb": float}
        """
        result = {"compressed": 0, "summarized": 0, "deleted": 0, "freed_mb": 0.0}
        now = utc_now_dt()
        compress_cutoff = now - timedelta(days=self.retention_full_days)
        summary_cutoff = now - timedelta(days=self.retention_compressed_days)
        index = self._load_index()

        for entry in index.sessions:
            started = parse_iso(entry.started_at)
            if started is None:
                continue
            filepath = self.history_dir / entry.file
            if not filepath.exists():
                continue

            if started < summary_cutoff:
                freed = _summarize_session_file(filepath)
                if freed > 0:
                    result["summarized"] += 1
                    result["freed_mb"] += freed
            elif started < compress_cutoff:
                freed = _compress_session_file(filepath, self.compress_preview_chars)
                if freed > 0:
                    result["compressed"] += 1
                    result["freed_mb"] += freed

        result["deleted"] += self._enforce_storage_limit(index)
        logger.info("Compact completed: %s", result)
        return result

    def get_stats(self) -> dict:
        """統計情報を取得"""
        index = self._load_index()
        mode_counts: dict[str, int] = {}
        for e in index.sessions:
            mode_counts[e.mode] = mode_counts.get(e.mode, 0) + 1

        return {
            "total_sessions": index.total_sessions,
            "total_turns": index.total_turns,
            "total_size_mb": round(index.total_size_mb, 2),
            "max_storage_mb": self.max_storage_mb,
            "mode_counts": mode_counts,
            "summary_generated": sum(
                1 for e in index.sessions if e.summary is not None
            ),
        }

    # ── インデックス管理 ──

    def _load_index(self) -> HistoryIndex:
        """インデックスを読込み（キャッシュ）"""
        if self._index is not None:
            return self._index

        index_path = self.history_dir / "index.json"
        if not index_path.exists():
            self._index = HistoryIndex()
            return self._index

        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)

        sessions = [
            IndexEntry(
                session_id=s["session_id"],
                file=s["file"],
                started_at=s.get("started_at", ""),
                duration_sec=s.get("duration_sec", 0),
                mode=s.get("mode", "chat"),
                turn_count=s.get("turn_count", 0),
                summary=s.get("summary"),
                summary_turn_count=int(s.get("summary_turn_count", 0) or 0),
                topics=s.get("topics", []),
                size_bytes=s.get("size_bytes", 0),
                search_text=s.get("search_text", ""),
                promoted_to_semmem=bool(s.get("promoted_to_semmem", False)),
                project_id=s.get("project_id"),
            )
            for s in data.get("sessions", [])
        ]

        self._index = HistoryIndex(
            updated_at=data.get("updated_at", ""),
            total_sessions=data.get("total_sessions", len(sessions)),
            total_turns=data.get("total_turns", 0),
            total_size_mb=data.get("total_size_mb", 0.0),
            sessions=sessions,
        )
        return self._index

    def _update_index(self, entry: IndexEntry) -> None:
        """インデックスにエントリを追加"""
        index = self._load_index()

        # 重複チェック
        index.sessions = [
            e for e in index.sessions if e.session_id != entry.session_id
        ]
        index.sessions.append(entry)

        # 統計更新
        index.total_sessions = len(index.sessions)
        index.total_turns = sum(e.turn_count for e in index.sessions)
        index.total_size_mb = sum(e.size_bytes for e in index.sessions) / (1024 * 1024)

        self._save_index(index)

    def _save_index(self, index: HistoryIndex) -> None:
        """インデックスを保存"""
        index.updated_at = _now_iso()

        data = {
            "updated_at": index.updated_at,
            "total_sessions": index.total_sessions,
            "total_turns": index.total_turns,
            "total_size_mb": round(index.total_size_mb, 4),
            "sessions": [asdict(s) for s in index.sessions],
        }

        index_path = self.history_dir / "index.json"
        _atomic_write_json(index_path, data)

        self._index = index

    def ensure_search_text(self) -> int:
        """search_text が未設定のエントリにターン本文を補完

        起動時やインデックス読み込み時に呼び出す。
        Returns:
            補完したエントリ数
        """
        index = self._load_index()
        updated = 0
        for entry in index.sessions:
            if entry.search_text:
                continue
            filepath = self.history_dir / entry.file
            if not filepath.exists():
                continue
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                entry.search_text = _build_search_text(
                    SessionData.from_dict(data),
                )
                updated += 1
            except Exception as e:
                logger.warning("Failed to build search_text for %s: %s",
                               entry.session_id, e)

        if updated:
            self._save_index(index)
            logger.info("Backfilled search_text for %d sessions", updated)
        return updated

    def rebuild_index(self) -> HistoryIndex:
        """ファイルスキャンでインデックスを再構築"""
        index = HistoryIndex()

        for month_dir in sorted(self.history_dir.iterdir()):
            if not month_dir.is_dir() or month_dir.name.startswith("."):
                continue

            for session_file in sorted(month_dir.glob("*.json")):
                try:
                    with open(session_file, encoding="utf-8") as f:
                        data = json.load(f)
                    sd = SessionData.from_dict(data)
                    entry = IndexEntry(
                        session_id=sd.session_id,
                        file=f"{month_dir.name}/{session_file.name}",
                        started_at=sd.started_at,
                        duration_sec=sd.duration_sec,
                        mode=sd.mode,
                        turn_count=sd.turn_count,
                        summary=sd.summary,
                        summary_turn_count=sd.summary_turn_count,
                        topics=sd.topics,
                        size_bytes=session_file.stat().st_size,
                        search_text=_build_search_text(sd),
                        promoted_to_semmem=sd.promoted_to_semmem,
                        project_id=sd.project_id,
                    )
                    index.sessions.append(entry)
                except Exception as e:
                    logger.warning("Failed to index %s: %s", session_file, e)

        index.total_sessions = len(index.sessions)
        index.total_turns = sum(e.turn_count for e in index.sessions)
        index.total_size_mb = sum(e.size_bytes for e in index.sessions) / (1024 * 1024)

        self._save_index(index)
        logger.info("Index rebuilt: %d sessions", index.total_sessions)
        return index

    # ── ユーティリティ ──

    def _enforce_storage_limit(self, index: HistoryIndex) -> int:
        """ストレージ上限チェック＋古い順削除"""
        total_mb = self._calc_total_size_mb()
        if total_mb <= self.max_storage_mb:
            return 0
        deleted = self._delete_oldest(index, total_mb - self.max_storage_mb)
        self._save_index(index)
        return deleted

    def _calc_total_size_mb(self) -> float:
        """history_dir 配下の総サイズ (MB)"""
        total = 0
        for root, _, files in os.walk(self.history_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)

    def _delete_oldest(self, index: HistoryIndex, target_mb: float) -> int:
        """ストレージ上限超過分を最古から削除

        安全な順序: 削除対象を特定 → インデックス更新 → ファイル削除
        """
        # 古い順にソートして削除対象を特定
        sorted_entries = sorted(index.sessions, key=lambda e: e.started_at)
        to_delete: list[IndexEntry] = []
        freed = 0.0

        for entry in sorted_entries:
            if freed >= target_mb:
                break
            filepath = self.history_dir / entry.file
            if filepath.exists():
                freed += filepath.stat().st_size / (1024 * 1024)
                to_delete.append(entry)

        if not to_delete:
            return 0

        # インデックスから先に除去
        delete_ids = {e.session_id for e in to_delete}
        index.sessions = [
            e for e in index.sessions if e.session_id not in delete_ids
        ]

        # ファイル削除（インデックス更新後なので失敗しても不整合にならない）
        for entry in to_delete:
            filepath = self.history_dir / entry.file
            try:
                filepath.unlink()
            except OSError as e:
                logger.warning("Failed to delete %s: %s", filepath, e)

        return len(to_delete)


def _now_iso() -> str:
    return utc_now_dt().isoformat()


# ── シングルトンファクトリ ──

_manager_cache: HistoryManager | None = None


def get_history_manager() -> HistoryManager:
    """シングルトンで HistoryManager を返す（インデックスキャッシュ有効化）

    API・CLI・sleep-time update など全レイヤーから共有で使用する。
    同一インスタンスを共有することで、セッション保存後のインデックス
    キャッシュが即座に一覧取得に反映される。
    """
    global _manager_cache
    if _manager_cache is not None:
        return _manager_cache
    from backend.config import get_config, get_path_resolver
    resolver = get_path_resolver()
    cfg = get_config()
    history_dir = resolver.resolve_local("history_dir")
    mgr = HistoryManager(history_dir, cfg)
    mgr.promote_checkpoints()
    mgr.ensure_search_text()
    _manager_cache = mgr
    return mgr




