"""会話履歴の自動保存・アーカイブ・検索・圧縮

アクティブなセッションはターンごとの追記ログに書き、閉じたときにセッション JSON へ
畳む (c_05 §2.1 / §0.5.9)。sleep-time で要約・圧縮を行う。
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.history.utils import parse_iso, snippet_around
from backend.free.rag.evidence.tokenize import tokenize_ja
from backend.io import jsoncodec
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.readonly import DataReadonlyError, is_readonly
from backend.io.versioned import ReadResult, build_envelope, read_versioned
from backend.io.writer_thread import ChatWriter, default_writer
from backend.log_config import get_logger
from backend.utils import utc_now, utc_now_dt

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

# 空白区切りのキーワード列 (判定側の抽出器が渡す形) に使う最小重なり割合。
#
# bi-gram 側の 0.7 をそのまま流用していたが、あの厳しさは「助詞絡みの連結詞的
# bi-gram が偶然重なる」ことへの対策であって、2 文字以上の内容語には当てはまら
# ない。しかも ceil の効き方で 2 語なら 2 件、3 語なら 3 件が必要になり、**実質
# AND** になっていた。抽出器は語を落とす方向に不完全 (漢字ランだけを拾うため
# 「食べ物」→「食」「物」、「昨日見た映画」→「昨日見」「映画」) なので、全語一致を
# 要求すると 1 個の壊れた語が検索全体を殺す。
#
# 実インシデント (2026-08-16 ライブ監査): search_history の発火 4 件が全て 0 件。
# 「昨日見た映画が…」→ ``昨日見 映画`` は ceil(2*0.7)=2 で ``昨日見`` との AND に
# なり、そんな語は履歴に存在しないため構造的に当たらなかった。
#
# 精度はここではなく ``_score_entry`` のランキングと呼出側の limit で担保する。
_KEYWORD_OVERLAP_RATIO = 0.5


@lru_cache(maxsize=64)
def _tokenize_cached(text: str) -> frozenset[str]:
    """``tokenize_ja`` の結果をキャッシュする (空白区切りは区間ごとにトークン化)。

    1 回の検索 (search_sessions/list_sessions) で同一のクエリ文字列に対して
    最大で「対象エントリ数 × 2 (summary/search_text)」回呼ばれうる
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
    # 空白区切りのキーワード列は「語の集合」として扱う。判定側の抽出器は
    # ``設計 言 答`` のような列を渡すので、連続部分文字列一致は構造的に当たらず、
    # トークン重なりも ``tokenize_ja`` が 1 文字語を落とすため語数不足で無効化され、
    # **どちらの経路にも乗らない** (実測 2026-08-10: tokens={'設計'} で len<5 →
    # fallback 無効 → 不一致。``設計`` 単独なら一致する)。結果 score は最低値
    # 0.1 のまま matched_turns も空になり、「この会話の最初に何を設計すると
    # 言ったか」に「記述はありません」と答えていた。
    terms = [t for t in query_lower.split() if len(t) >= 2]
    if terms and len(query_lower.split()) > 1:
        hit = sum(1 for t in terms if t in text_lower)
        # 抽出器が語を落とす方向に不完全なので、全語一致は要求しない
        # (:data:`_KEYWORD_OVERLAP_RATIO` の説明を参照)。
        required = max(1, math.ceil(len(terms) * _KEYWORD_OVERLAP_RATIO))
        if hit >= required:
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


@persisted()
@dataclass
class SessionData:
    """セッションデータ (``history.session`` の payload。コーデックの表、c_05 §0.5.2)"""
    session_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_sec: int = 0
    mode: str = "chat"
    modes_used: list[str] = field(default_factory=list)
    instance_name: str = "evoref"
    #: 表示用のモデル名 (GGUF のファイル名)。
    base_model: str = ""
    #: base モデルの ``model_key`` (c_05 §0.5.7)。モデルの照合はこちらで行う。
    model_key: str | None = None
    source: str = "auto"  # "auto" | "manual"
    turns: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    context_files: list[str] = field(default_factory=list)
    token_info: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    #: ``summary`` を生成した時点の ``turn_count``。会話が進んで turn_count が
    #: これを上回ったら sleep-time が要約を作り直す (会話途中で要約が固定され、
    #: 後半の訂正が要約に反映されないのを防ぐ)。
    summary_turn_count: int = 0
    archived_at: str = ""
    # SemMem への昇格済フラグ
    promoted_to_semmem: bool = False
    project_id: str | None = None
    #: 会話の主言語 (``ja`` / ``en`` / 未判定は空)。横断検索の重み付け用。
    lang: str = ""
    # 要約の埋め込みはセッションに持たない (G0 の履歴ファイルの 81% がこの JSON
    # 文字列だった)。埋め込みモデルごとの束 ``embeddings/<model>.npy`` + id 表
    # (``history.summary_embeddings``) に置く (c_05 §2.1)。版は封筒が持つ。
    #: この版が知らないキー (同じ版で足された任意フィールド)。書き戻しでトップへ戻す。
    _extra: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> SessionData:
        """payload の dict から復元する (null / 欠損は既定値、未知キーは ``_extra``)。

        型の違う値は :class:`~backend.io.codec.CodecError`。
        """
        return _SESSION_CODEC.decode(data)

    def to_dict(self) -> dict[str, Any]:
        """payload の dict (``_extra`` はトップへ戻す)。"""
        return _SESSION_CODEC.encode(self)


_SESSION_CODEC = codec_for(SessionData)


@persisted()
@dataclass
class TurnRow:
    """ターンの追記ログ (``history.turns``) の ``op: turn`` の行 (``_v`` / ``op`` を除く)。

    ``seq`` はセッション全体での位置。畳み込みは ``seq < len(セッション JSON のターン)``
    の行を飛ばすので、JSON を書いた後・ログを消す前に落ちても二重に畳まない。
    """

    seq: int
    turn: dict[str, Any]
    #: この版が知らない行のキー。畳むときはそのターンへ移す (ターン側のキーが勝つ)。
    _extra: dict[str, Any] | None = None

    def folded_turn(self) -> dict[str, Any]:
        """セッション JSON の ``turns`` に入れるターン (行の未知キーを含む)。"""
        return {**self._extra, **self.turn} if self._extra else self.turn


_TURN_ROW_CODEC = codec_for(TurnRow)
#: 行の封筒のキー (行のレコードには含めない)。
_ROW_KEYS = ("_v", "op")


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
    #: 最初のユーザ発話の先頭 (一覧表示のフォールバック見出し)。要約は
    #: sleep-time でしか付かないため、未要約セッションの見出しをこれで賄う。
    first_user_preview: str = ""
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
_FIRST_USER_PREVIEW_MAX = 100  # 一覧見出しフォールバックに使う先頭文字数

#: アクティブなセッションのターン追記ログの置き場 (``<history_dir>/active/``)。
ACTIVE_DIR = "active"
TURNS_SUFFIX = ".turns.jsonl"
#: 要約の埋め込みの束の置き場 (``<history_dir>/embeddings/``)。
EMBEDDINGS_DIR = "embeddings"
INDEX_FILE = "index.json"
#: 索引の遅延書き出し: 最後の変更からこの秒数静かなら書く。
_INDEX_IDLE_SECONDS = 5.0
#: 索引の遅延書き出し: 最初の変更からこの秒数で必ず書く。
_INDEX_MAX_DELAY_SECONDS = 30.0

SESSION_FORMAT = register_format(FormatSpec(
    format_id="history.session",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/history/<yyyy-mm>/<yyyymmdd_hhmmss>_<session_id>.json",
    retention="retention_full_days -> compressed -> summarized -> max_storage_mb",
    export=True,
    records=(SessionData,),
))
TURNS_FORMAT = register_format(FormatSpec(
    format_id="history.turns",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/history/active/<sid>.turns.jsonl",
    retention="folded into history.session on close / startup / shutdown / idle",
    # 停止中の export では畳まれていない最後のターンがここにだけある (取り込み先の起動時に畳む)
    export=True,
    encodings=("jsonl",),
    # メタ行はセッションの表 (SessionData) の部分集合を patch する素の dict
    records=(TurnRow,),
))
INDEX_FORMAT = register_format(FormatSpec(
    format_id="history.index",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="store/history/index.json",
    retention="rebuilt from history.session when built_from mismatches",
))
SUMMARY_EMBEDDINGS_FORMAT = register_format(FormatSpec(
    format_id="history.summary_embeddings",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="store/history/embeddings/<model>.npy",
    retention="one bundle (vectors .npy + id table .ids.json) per embedding model",
    encodings=("npy", "json"),
    keep_on_reset=True,
))


def _build_first_user_preview(
    session: SessionData,
    max_len: int = _FIRST_USER_PREVIEW_MAX,
) -> str:
    """最初のユーザ発話から一覧見出し用のプレビューを作る (決定論)。

    要約 (``summary``) は sleep-time の Full サイクルでしか付かず、アイドルが
    取れない日は 1 件も生成されない。未要約セッションの見出しが全部同じ
    「要約なし」になるのを避けるためのフォールバック。保存のたびに作り直す
    ので、要約器のゲート (``summary is None``) には影響しない。
    """
    for t in session.turns:
        if t.get("role") != "user":
            continue
        content = " ".join((t.get("content") or "").split())
        if not content:
            continue
        if len(content) > max_len:
            return content[:max_len] + "…"
        return content
    return ""


def _build_search_text(
    session: SessionData,
    max_len: int = _SEARCH_TEXT_MAX,
) -> str:
    """summary・全ターンの content を結合して検索用テキストを生成"""
    parts: list[str] = []
    total = 0

    if session.summary:
        parts.append(session.summary)
        total += len(session.summary)

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


def _summarize_session_data(data: dict) -> bool:
    """要約化 (ターン削除)。変えたら ``True``。"""
    if not data.get("turns"):
        return False
    data["turns"] = []
    return True


def _compress_session_data(data: dict, preview_chars: int = 100) -> bool:
    """圧縮保持 (アシスタントターンの切り詰め)。変えたら ``True``。

    Args:
        data: セッションの dict (``SessionData`` の永続形)
        preview_chars: 圧縮時に保持する先頭文字数
    """
    compressed = False
    for turn in data.get("turns", []):
        if turn.get("role") == "assistant" and not turn.get("compressed"):
            content = turn.get("content", "")
            if len(content) > preview_chars * 2:
                turn["content"] = content[:preview_chars] + "…（圧縮済み）"
                turn["compressed"] = True
                turn["original_length"] = len(content)
                compressed = True
    return compressed


# ── 永続形 (封筒・ターンの追記ログ) ──


def session_bytes(session: SessionData) -> bytes:
    """セッションを ``history.session`` の封筒にしたバイト列。"""
    envelope = build_envelope(
        format_id=SESSION_FORMAT.format_id, format_version=SESSION_FORMAT.version,
        payload=session.to_dict(), component="HistoryManager",
    )
    return jsoncodec.dumps_bytes(envelope)


def read_session_file(path: Path) -> ReadResult:
    """セッションファイルを読んで分類する (``read_versioned``)。

    payload がセッションの表で読めない (型の違う値) なら ``corrupt``。読めたときの
    ``payload`` は素の dict のまま (型付きにするのは :meth:`SessionData.from_dict`)。
    """
    result = read_versioned(
        path, format_id=SESSION_FORMAT.format_id, format_version=SESSION_FORMAT.version,
    )
    if result.ok:
        try:
            SessionData.from_dict(result.payload)
        except CodecError as e:
            return ReadResult("corrupt", version=result.version, detail=f"payload: {e}")
    return result


def turn_log_lines(meta: dict, rows: Iterable[TurnRow]) -> list[str]:
    """ターンの追記ログに足す行 (メタ 1 行 + ターンごとに 1 行)。"""
    version = TURNS_FORMAT.version
    lines = [jsoncodec.dumps({"_v": version, "op": "meta", **meta})]
    for row in rows:
        lines.append(jsoncodec.dumps({"_v": version, "op": "turn", **_TURN_ROW_CODEC.encode(row)}))
    return lines


@dataclass
class TurnLog:
    """ターンの追記ログを畳んだもの。"""

    meta: dict = field(default_factory=dict)
    modes: list[str] = field(default_factory=list)
    #: ターンの行 (書いた順)。
    turns: list[TurnRow] = field(default_factory=list)
    bad_lines: int = 0


def fold_turn_log(path: Path) -> TurnLog:
    """``<sid>.turns.jsonl`` を畳む。途中で切れた行・壊れた行・知らない版は飛ばす。

    メタ行の値の型はセッションの表 (:class:`SessionData`) で検査する。
    """
    log = TurnLog()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return log
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        if b"\x00" in line:
            log.bad_lines += 1
            continue
        try:
            obj = jsoncodec.loads(line)
        except (ValueError, UnicodeDecodeError):
            log.bad_lines += 1
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("_v"), int) or obj["_v"] > TURNS_FORMAT.version:
            log.bad_lines += 1
            continue
        op = obj.get("op")
        body = {k: v for k, v in obj.items() if k not in _ROW_KEYS}
        try:
            if op == "meta":
                meta = _SESSION_CODEC.check_mapping(body)
                mode = meta.get("mode")
                if isinstance(mode, str) and mode and mode not in log.modes:
                    log.modes.append(mode)
                log.meta.update(meta)
            elif op == "turn":
                log.turns.append(_TURN_ROW_CODEC.decode(body))
            else:
                log.bad_lines += 1
        except CodecError:
            log.bad_lines += 1
    if log.bad_lines:
        logger.warning("Skipped %d unreadable line(s) in %s", log.bad_lines, path)
    return log


def merge_turn_log(base: SessionData | None, log: TurnLog) -> SessionData:
    """セッション JSON (無ければ新規) にターンの追記ログを重ねる。

    ``seq`` が既存のターン数より小さい行は畳み込み済みなので飛ばす。
    開始時刻は既存の JSON が勝つ (保存先ファイル名を決める)。メタ行のうちセッションの
    表に無いキーはセッションの ``_extra`` へ、ターンの行の未知キーはそのターンへ移す
    (同じ版で足された任意フィールドを畳み込みで落とさない、c_05 §0.4.4)。
    """
    session = base if base is not None else SessionData()
    meta = log.meta
    session.session_id = meta.get("session_id") or session.session_id
    if not session.started_at:
        session.started_at = meta.get("started_at") or ""
    session.ended_at = meta.get("ended_at") or session.ended_at
    session.mode = meta.get("mode") or session.mode
    modes = list(session.modes_used or [])
    for mode in log.modes:
        if mode not in modes:
            modes.append(mode)
    session.modes_used = modes
    session.instance_name = meta.get("instance_name") or session.instance_name
    session.base_model = meta.get("base_model") or session.base_model
    session.model_key = meta.get("model_key") or session.model_key
    if meta.get("project_id") and not session.project_id:
        session.project_id = meta["project_id"]
    extra = {k: v for k, v in meta.items() if k not in _SESSION_CODEC.known}
    if extra:
        session._extra = {**(session._extra or {}), **extra}
    turns = list(session.turns)
    for row in log.turns:
        if row.seq < len(turns):
            continue
        turns.append(row.folded_turn())
    session.turns = turns
    session.turn_count = len(turns)
    return session


def _model_file_key(model: str) -> str:
    """埋め込みモデル名をファイル名の鍵にする。

    G1 の埋め込み ``model_key`` (c_05 §0.5.7) が入るまでの暫定: 埋め込みモデル名を
    ファイル名に使える文字へ寄せる。
    """
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", model or "").strip("._")
    return key or "unknown"


def _files_digest(files: dict[str, tuple[int, int]]) -> str:
    """原本 (セッション JSON) の一覧の指紋 (``built_from``)。"""
    h = hashlib.sha256()
    for rel in sorted(files):
        size, mtime = files[rel]
        h.update(f"{rel}\t{size}\t{mtime}\n".encode())
    return h.hexdigest()[:32]


class HistoryManager:
    """会話履歴のアーカイブ管理

    書き込みは全てチャット経路の書き手スレッド (c_05 §0.5.9、
    :mod:`backend.io.writer_thread`) の上で行う:

    - アクティブなセッションはターンごとの追記ログ ``active/<sid>.turns.jsonl``
      (``history.turns``) に書き、閉じたとき (:meth:`close_session`)・起動時・停止時・
      sleep-time (しばらく追記の無いもの) にセッション JSON (``history.session``) へ
      畳む。
    - 索引 ``index.json`` (``history.index``、derived) はメモリ上の不変スナップショットで、
      読み手はロックを取らずに読む (書き換えは書き手スレッドだけが新しいスナップショットを
      差し込む)。ファイルへの書き出しは遅延 (最後の変更から 5 秒 / 最初の変更から最長
      30 秒 / 停止時)。``built_from`` に原本の一覧の指紋を持ち、起動時に合わなければ
      原本から作り直す。書き順は常に原本が先。
    """

    def __init__(
        self, history_dir: Path, config: dict | None = None, *,
        writer: ChatWriter | None = None,
    ):
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        history_cfg = (config or {}).get("history", {})
        self.auto_save: bool = history_cfg.get("auto_save", True)
        self.retention_full_days: int = history_cfg.get("retention_full_days", 90)
        self.retention_compressed_days: int = history_cfg.get("retention_compressed_days", 365)
        self.max_storage_mb: float = history_cfg.get("max_storage_mb", 200)
        self.compress_preview_chars: int = history_cfg.get("compress_preview_chars", 100)

        self._writer = writer if writer is not None else default_writer()
        self._active_dir = history_dir / ACTIVE_DIR
        self._embeddings_dir = history_dir / EMBEDDINGS_DIR
        #: 公開中の索引 (差し替えるだけで書き換えない)。``None`` なら次の読みで読む。
        self._index: HistoryIndex | None = None
        #: 原本の相対パス → (サイズ, mtime_ns)。``built_from`` の元。
        self._files: dict[str, tuple[int, int]] = {}
        #: 索引の書き出し待ち: (最初の変更, 最後の変更) の monotonic 時刻。
        self._index_dirty: tuple[float, float] | None = None
        #: 埋め込みのサイドカー (鍵 → (id 列, 行列))。書き手が丸ごと差し替える。
        self._embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
        self._writer.add_tick(self._tick)

    # ── 書き手スレッドで実行する ──

    def _run(
        self, fn: Callable[[], Any], *, format_id: str = "history.session",
        paths: tuple[Path, ...] = (), wait: bool = True,
    ) -> Any:
        """``fn`` を書き手スレッドで実行する (``wait`` なら結果を待つ)。"""
        if self._writer.is_writer_thread():
            return fn()
        future = self._writer.call(fn, format_id=format_id, paths=paths)
        return future.result() if wait else None

    def _turns_path(self, session_id: str) -> Path:
        return self._active_dir / f"{session_id}{TURNS_SUFFIX}"

    # ── 保存 ──

    def save_session(self, session: SessionData) -> Path | None:
        """セッション全体をアーカイブに保存 (渡された内容が正)。

        Returns:
            保存先パス（保存しなかった場合は None）

        データ根が readonly なら書かない (チャットは動かし続ける、c_05 §0.4.2)。
        """
        if is_readonly():
            logger.debug("History not saved: data root is read-only")
            return None
        skip_reason = _should_skip_session(session, self.auto_save)
        if skip_reason:
            logger.debug(skip_reason)
            return None
        _complete_session_metadata(session)
        return self._run(lambda: self._write_session(session, None))

    def append_turns(
        self, session_id: str, turns: list[dict], meta: dict, *,
        first_seq: int, contents: tuple[tuple[str, str], ...],
    ) -> None:
        """アクティブなセッションのターンを追記ログへ出す (イベントループから)。

        ループで行うのは行の直列化と enqueue だけ。索引の項目は書き手スレッドが
        追記の後に差し替える (``contents`` は一覧の見出し・検索テキスト用の
        ``(role, content)`` の列 — セッション全体)。

        readonly なら :class:`~backend.io.readonly.DataReadonlyError`。
        """
        lines = turn_log_lines(meta, (TurnRow(first_seq + i, turn) for i, turn in enumerate(turns)))
        self._writer.append(self._turns_path(session_id), lines, format_id=TURNS_FORMAT.format_id)
        self._writer.call(
            lambda: self._refresh_active_entry(session_id, meta, contents),
            format_id=INDEX_FORMAT.format_id,
        )

    def close_session(self, session_id: str, *, wait: bool = False) -> None:
        """セッションを閉じる: 追記ログをセッション JSON へ畳む。"""
        if is_readonly():
            return
        # ログの有無は書き手スレッドで見る (ここでは先に出した追記がまだ書かれて
        # いないことがある)。
        self._run(lambda: self._fold(session_id), paths=(self._turns_path(session_id),), wait=wait)

    def fold_active_sessions(self, *, idle_seconds: float | None = None) -> int:
        """追記ログを畳む (起動時の取り残し・停止時・sleep-time の放置分)。

        ``idle_seconds`` を指定したら、最後の追記からその秒数を過ぎたログだけ。
        畳んだ件数を返す。readonly なら畳まない (読み手はログを重ねて読む)。
        """
        if is_readonly() or not self._active_dir.is_dir():
            return 0
        now = time.time()
        targets: list[str] = []
        for path in sorted(self._active_dir.glob(f"*{TURNS_SUFFIX}")):
            if idle_seconds is not None:
                try:
                    if now - path.stat().st_mtime < idle_seconds:
                        continue
                except OSError:
                    continue
            targets.append(path.name[: -len(TURNS_SUFFIX)])
        if not targets:
            return 0

        def run() -> int:
            self._index_or_load()
            return sum(1 for sid in targets if self._fold(sid) is not None)

        folded = self._run(run, paths=tuple(self._turns_path(s) for s in targets))
        if folded:
            logger.info("Folded %d active session log(s) into history", folded)
        return folded

    def update_session_fields(self, session_id: str, **fields: Any) -> bool:
        """セッションのフィールドを書き換える (要約・言語・昇格印)。

        追記ログがあれば一緒に畳む (書き手スレッドの上で追記と直列になるので、
        読んでから書くまでの間に届いたターンを落とさない)。
        """
        if is_readonly():
            return False
        return bool(self._run(
            lambda: self._fold(session_id, updates=fields, require_log=False) is not None,
            paths=(self._turns_path(session_id),),
        ))

    def mark_promoted_to_semmem(self, session_id: str) -> bool:
        """セッションを SemMem 昇格済としてマーク

        索引とセッション本体の両方に反映する (再起動後に再昇格しないため)。

        Returns:
            マークに成功したら ``True``、未存在なら ``False``。
        """
        if self._find_entry(session_id) is None:
            return False
        if self.update_session_fields(session_id, promoted_to_semmem=True):
            return True

        # 本体が無い (索引だけの) セッションは索引の印だけ付ける。
        def mark_index() -> bool:
            entry = self._find_entry(session_id)
            if entry is None:
                return False
            self._put_entry(replace(entry, promoted_to_semmem=True))
            return True

        return bool(self._run(mark_index, format_id=INDEX_FORMAT.format_id))

    # ── 書き手スレッドの中身 ──

    def _write_session(self, session: SessionData, path: Path | None) -> Path:
        target = path or self._resolve_session_path(session)
        self._writer.write_now(target, session_bytes(session), fsync=True)
        self._note_file(target)
        entry = self._build_index_entry(session, target)
        self._put_entry(entry)
        logger.info("Session saved: %s (%d turns, %d bytes)",
                    session.session_id, session.turn_count, entry.size_bytes)
        return target

    def _read_existing(self, session_id: str) -> tuple[SessionData | None, Path | None, bool]:
        """索引の指すセッション JSON を読む。``(session, path, writable)``。

        読めない (新しい版・別形式・壊れている) ファイルは上書きしない (``writable=False``)。
        """
        entry = self._find_entry(session_id)
        if entry is None:
            return None, None, True
        path = self.history_dir / entry.file
        result = read_session_file(path)
        if result.status == "absent":
            return None, None, True
        if not result.ok:
            logger.warning(
                "History session %s is not readable (%s: %s); not writing to it",
                path, result.status, result.detail,
            )
            return None, path, False
        return SessionData.from_dict(result.payload), path, True

    def _fold(
        self, session_id: str, *, updates: dict | None = None, require_log: bool = True,
    ) -> Path | None:
        log_path = self._turns_path(session_id)
        has_log = log_path.exists()
        if require_log and not has_log:
            return None
        base, path, writable = self._read_existing(session_id)
        if not writable:
            return None
        log = fold_turn_log(log_path) if has_log else TurnLog()
        if base is None and not log.turns:
            if has_log:
                self._writer.remove_now(log_path)
            return None
        session = merge_turn_log(base, log)
        for key, value in (updates or {}).items():
            setattr(session, key, value)
        skip_reason = _should_skip_session(session, self.auto_save)
        if skip_reason and base is None:
            logger.debug(skip_reason)
            self._writer.remove_now(log_path)
            return None
        _complete_session_metadata(session)
        target = self._write_session(session, path)
        if has_log:
            # 原本を書いた後で消す (間で落ちても seq で二重に畳まない)。
            self._writer.remove_now(log_path)
        return target

    def _refresh_active_entry(
        self, session_id: str, meta: dict, contents: tuple[tuple[str, str], ...],
    ) -> None:
        """追記の後に索引の項目を差し替える (書き手スレッド)。"""
        entry = self._find_entry(session_id)
        started_at = (entry.started_at if entry is not None and entry.started_at
                      else meta.get("started_at") or "")
        rel = entry.file if entry is not None else self._session_rel_path(started_at, session_id)
        size = 0
        for p in (self.history_dir / rel, self._turns_path(session_id)):
            try:
                size += p.stat().st_size
            except OSError:
                pass
        view = SessionData(
            summary=entry.summary if entry is not None else None,
            turns=[{"role": r, "content": c} for r, c in contents],
        )
        fresh = IndexEntry(
            session_id=session_id,
            file=rel,
            started_at=started_at,
            duration_sec=entry.duration_sec if entry is not None else 0,
            mode=meta.get("mode") or (entry.mode if entry is not None else "chat"),
            turn_count=len(contents),
            summary=view.summary,
            summary_turn_count=entry.summary_turn_count if entry is not None else 0,
            first_user_preview=_build_first_user_preview(view),
            size_bytes=size,
            search_text=_build_search_text(view),
            promoted_to_semmem=entry.promoted_to_semmem if entry is not None else False,
            project_id=(entry.project_id if entry is not None and entry.project_id
                        else meta.get("project_id")),
        )
        self._put_entry(fresh)

    def _session_rel_path(self, started_at: str, session_id: str) -> str:
        """``<yyyy-mm>/<stamp>_<session_id>.json`` (session_id は全体、c_05 §0.5.5)。

        先頭 8 文字に切ると、uuid4 の先頭が一致するセッション同士が同じファイル名になる。
        session_id は API で ``[a-z0-9-]`` に限っているのでファイル名に使える。
        """
        started = parse_iso(started_at) or utc_now_dt()
        return (f"{started.strftime('%Y-%m')}/"
                f"{started.strftime('%Y%m%d_%H%M%S')}_{session_id}.json")

    def _resolve_session_path(self, session: SessionData) -> Path:
        """保存先パスの決定"""
        path = self.history_dir / self._session_rel_path(session.started_at, session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

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
            first_user_preview=_build_first_user_preview(session),
            size_bytes=size_bytes,
            search_text=_build_search_text(session),
            promoted_to_semmem=session.promoted_to_semmem,
            project_id=session.project_id,
        )

    def _refresh_index_entry(self, entry: IndexEntry, filepath: Path) -> IndexEntry:
        """圧縮 / 要約化でファイルを書き換えた後、索引側を実ファイルへ揃える。

        揃えないと索引が「削除済みの本文」を持ち続け、検索がヒットするのに
        開くと無い状態になる。さらに保持ポリシーで消したはずの発話が
        ``index.json`` に原文のまま残る (2026-09-05 監査)。
        """
        result = read_session_file(filepath)
        if not result.ok:
            logger.warning("Failed to refresh index entry for %s: %s", filepath, result.detail)
            return entry
        session = SessionData.from_dict(result.payload)
        try:
            size = filepath.stat().st_size
        except OSError:
            size = entry.size_bytes
        return replace(
            entry,
            turn_count=session.turn_count or len(session.turns),
            search_text=_build_search_text(session),
            first_user_preview=_build_first_user_preview(session),
            summary=session.summary,
            summary_turn_count=session.summary_turn_count,
            size_bytes=size,
        )

    def _note_file(self, path: Path) -> None:
        rel = path.relative_to(self.history_dir).as_posix()
        try:
            st = path.stat()
        except OSError:
            self._files.pop(rel, None)
            return
        self._files[rel] = (st.st_size, st.st_mtime_ns)

    def _forget_file(self, rel: str) -> None:
        self._files.pop(rel, None)

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
        entries = list(index.sessions)

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

    def get_summary(self, session_id: str) -> str | None:
        """索引上の要約を返す (未登録 / 未要約なら ``None``)。"""
        entry = self._find_entry(session_id)
        return entry.summary if entry is not None else None

    def get_session_started_at(self, session_id: str) -> str | None:
        """索引上の開始時刻 (ISO 8601) を返す。未登録なら ``None``。

        保存先ファイル名は開始時刻から決まる。プロセス再起動後に同じ
        セッションが続くと、in-process の開始時刻が失われて別ファイルが
        できる — 保存側はまずこれで既存の開始時刻を引き継ぐ。
        """
        entry = self._find_entry(session_id)
        return entry.started_at if entry is not None and entry.started_at else None

    def _find_entry(self, session_id: str) -> IndexEntry | None:
        index = self._load_index()
        return next(
            (e for e in index.sessions if e.session_id == session_id), None,
        )

    def get_session(self, session_id: str) -> SessionData | None:
        """セッション詳細を取得 (アクティブなら追記ログを重ねて返す)。"""
        entry = self._find_entry(session_id)
        base: SessionData | None = None
        if entry is not None:
            result = read_session_file(self.history_dir / entry.file)
            if result.ok:
                base = SessionData.from_dict(result.payload)
            elif result.status != "absent":
                logger.warning(
                    "History session %s is not readable (%s: %s)",
                    entry.file, result.status, result.detail,
                )
        # 原本の後にログを読む (畳み込みと入れ違っても seq で二重にならない)。
        log_path = self._turns_path(session_id)
        if log_path.exists():
            log = fold_turn_log(log_path)
            if log.turns or base is not None:
                base = merge_turn_log(base, log)
        return base

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

    # ── 削除 ──

    def delete_session(self, session_id: str) -> bool:
        """セッションを削除"""
        return self.delete_sessions_batch([session_id]) > 0

    def delete_sessions_batch(self, session_ids: list[str]) -> int:
        """複数セッションを一括削除 (本体・追記ログ・要約の埋め込み)

        Returns:
            削除したセッション数
        """
        target_ids = set(session_ids)
        to_delete = [e for e in self._load_index().sessions if e.session_id in target_ids]
        if not to_delete:
            return 0
        logs = tuple(self._turns_path(e.session_id) for e in to_delete)

        def run() -> int:
            delete_ids = {e.session_id for e in to_delete}
            # 索引から先に除去 (公開は差し替え 1 回)
            index = self._index_or_load()
            self._publish(replace(
                index, sessions=[e for e in index.sessions if e.session_id not in delete_ids],
            ))
            for entry in to_delete:
                try:
                    self._writer.remove_now(self.history_dir / entry.file)
                    self._forget_file(entry.file)
                    self._writer.remove_now(self._turns_path(entry.session_id))
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", entry.file, e)
            self._drop_embeddings(delete_ids)
            # 利用者の削除操作は索引もすぐに書く (消した発話を index.json に残さない)。
            self._write_index()
            return len(to_delete)

        deleted = self._run(run, paths=logs)
        logger.info("Deleted %d session(s): %s", deleted, sorted(target_ids)[:5])
        return deleted

    # ── 圧縮 ──

    def compact_sessions(self) -> dict:
        """保持ポリシーに基づく圧縮処理

        Returns:
            {"compressed": int, "summarized": int, "deleted": int, "freed_mb": float}
        """
        return self._run(self._compact_sessions)

    def _compact_sessions(self) -> dict:
        result = {"compressed": 0, "summarized": 0, "deleted": 0, "freed_mb": 0.0}
        now = utc_now_dt()
        compress_cutoff = now - timedelta(days=self.retention_full_days)
        summary_cutoff = now - timedelta(days=self.retention_compressed_days)
        index = self._index_or_load()
        sessions = list(index.sessions)

        for i, entry in enumerate(sessions):
            started = parse_iso(entry.started_at)
            if started is None:
                continue
            filepath = self.history_dir / entry.file
            if not filepath.exists():
                continue
            if started < summary_cutoff:
                kind, change = "summarized", _summarize_session_data
            elif started < compress_cutoff:
                kind = "compressed"

                def change(data: dict) -> bool:
                    return _compress_session_data(data, self.compress_preview_chars)
            else:
                continue
            freed = self._rewrite_session_file(filepath, change)
            if freed > 0:
                result[kind] += 1
                result["freed_mb"] += freed
                sessions[i] = self._refresh_index_entry(entry, filepath)

        index = replace(index, sessions=sessions)
        index, deleted = self._enforce_storage_limit(index)
        result["deleted"] += deleted
        self._publish(index)
        self._write_index()
        logger.info("Compact completed: %s", result)
        return result

    def _rewrite_session_file(self, filepath: Path, change: Callable[[dict], bool]) -> float:
        """セッション JSON を ``change`` で書き換える。解放 MB を返す (変更なしは 0)。"""
        result = read_session_file(filepath)
        if not result.ok or not isinstance(result.payload, dict):
            return 0.0
        data = result.payload
        if not change(data):
            return 0.0
        old_size = filepath.stat().st_size
        self._writer.write_now(
            filepath, session_bytes(SessionData.from_dict(data)), fsync=True,
        )
        self._note_file(filepath)
        new_size = filepath.stat().st_size
        return (old_size - new_size) / (1024 * 1024)

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

    # ── 要約の埋め込み (history/embeddings/<model>.npy + id 表の 1 束) ──

    def put_summary_embedding(
        self, session_id: str, vector: list[float], model: str,
    ) -> None:
        """要約の埋め込みを束へ入れる (書き手スレッドで書く。待たない)。"""
        if is_readonly():
            return
        key = _model_file_key(model)
        vec = np.asarray(vector, dtype=np.float32)
        self._run(
            lambda: self._put_embedding(key, model, session_id, vec),
            format_id=SUMMARY_EMBEDDINGS_FORMAT.format_id, wait=False,
        )

    def get_summary_embedding(self, session_id: str, model: str) -> list[float] | None:
        """要約の埋め込み (無ければ ``None``)。"""
        ids, matrix = self._embedding_bundle(_model_file_key(model))
        try:
            row = ids.index(session_id)
        except ValueError:
            return None
        return matrix[row].tolist()

    def _embedding_paths(self, key: str) -> tuple[Path, Path]:
        return (self._embeddings_dir / f"{key}.npy",
                self._embeddings_dir / f"{key}.ids.json")

    def _embedding_bundle(self, key: str) -> tuple[list[str], np.ndarray]:
        cached = self._embeddings.get(key)
        if cached is not None:
            return cached
        npy, ids_path = self._embedding_paths(key)
        bundle: tuple[list[str], np.ndarray] = ([], np.zeros((0, 0), dtype=np.float32))
        result = read_versioned(
            ids_path, format_id=SUMMARY_EMBEDDINGS_FORMAT.format_id,
            format_version=SUMMARY_EMBEDDINGS_FORMAT.version,
        )
        if result.ok and isinstance(result.payload, dict):
            ids = [str(i) for i in result.payload.get("ids") or []]
            try:
                matrix = np.load(npy, allow_pickle=False)
            except (OSError, ValueError) as exc:
                logger.warning("Summary embeddings %s unreadable; dropping them: %s", npy, exc)
            else:
                # 束が揃っていなければ捨てる (derived。id 表が最後に書かれる)。
                if matrix.ndim == 2 and matrix.shape[0] == len(ids):
                    bundle = (ids, matrix)
                else:
                    logger.warning(
                        "Summary embeddings %s do not match their id table (%s rows vs %d ids); "
                        "dropping them", npy, matrix.shape, len(ids),
                    )
        self._embeddings[key] = bundle
        return bundle

    def _put_embedding(self, key: str, model: str, session_id: str, vec: np.ndarray) -> None:
        ids, matrix = self._embedding_bundle(key)
        ids = list(ids)
        if matrix.size and matrix.shape[1] != vec.shape[0]:
            logger.warning(
                "Summary embedding dimension changed for %s (%d -> %d); starting a new bundle",
                key, matrix.shape[1], vec.shape[0],
            )
            ids, matrix = [], np.zeros((0, vec.shape[0]), dtype=np.float32)
        if not matrix.size:
            matrix = np.zeros((0, vec.shape[0]), dtype=np.float32)
        if session_id in ids:
            matrix = matrix.copy()
            matrix[ids.index(session_id)] = vec
        else:
            ids.append(session_id)
            matrix = np.vstack([matrix, vec[None, :]])
        self._write_embeddings(key, model, ids, matrix)

    def _drop_embeddings(self, session_ids: set[str]) -> None:
        if not self._embeddings_dir.is_dir():
            return
        for ids_path in self._embeddings_dir.glob("*.ids.json"):
            key = ids_path.name[: -len(".ids.json")]
            ids, matrix = self._embedding_bundle(key)
            keep = [i for i, sid in enumerate(ids) if sid not in session_ids]
            if len(keep) == len(ids):
                continue
            model = ""
            result = read_versioned(
                ids_path, format_id=SUMMARY_EMBEDDINGS_FORMAT.format_id,
                format_version=SUMMARY_EMBEDDINGS_FORMAT.version,
            )
            if result.ok and isinstance(result.payload, dict):
                model = str(result.payload.get("model") or "")
            self._write_embeddings(key, model, [ids[i] for i in keep], matrix[keep])

    def _write_embeddings(self, key: str, model: str, ids: list[str], matrix: np.ndarray) -> None:
        npy, ids_path = self._embedding_paths(key)
        buf = io.BytesIO()
        np.save(buf, matrix.astype(np.float32, copy=False), allow_pickle=False)
        # 行列を先に、id 表を後に書く (id 表が束の確定。行数が合わなければ読み手が捨てる)。
        self._writer.write_now(npy, buf.getvalue(), fsync=False)
        envelope = build_envelope(
            format_id=SUMMARY_EMBEDDINGS_FORMAT.format_id,
            format_version=SUMMARY_EMBEDDINGS_FORMAT.version,
            payload={"model": model, "rows": len(ids), "ids": ids},
            component="HistoryManager",
        )
        self._writer.write_now(ids_path, jsoncodec.dumps_bytes(envelope), fsync=False)
        self._embeddings[key] = (ids, matrix)

    # ── インデックス管理 ──

    def _load_index(self) -> HistoryIndex:
        """公開中の索引を返す (ロックを取らない)。未読なら書き手スレッドで読む。"""
        index = self._index
        if index is not None:
            return index
        return self._run(self._index_or_load, format_id=INDEX_FORMAT.format_id)

    def _index_or_load(self) -> HistoryIndex:
        """(書き手スレッド) 索引を読む。``built_from`` が原本と合わなければ作り直す。"""
        if self._index is not None:
            return self._index
        files = self._scan_originals()
        digest = _files_digest(files)
        self._files = files
        index_path = self.history_dir / INDEX_FILE
        result = read_versioned(
            index_path, format_id=INDEX_FORMAT.format_id, format_version=INDEX_FORMAT.version,
        )
        payload = result.payload if result.ok and isinstance(result.payload, dict) else None
        if payload is not None and payload.get("built_from") == digest:
            self._index = _index_from_payload(payload)
            return self._index
        reason = (result.status if payload is None else "built_from mismatch")
        index = self._build_index_from_files(files)
        self._index = index
        if result.status != "absent" or files:
            logger.info("History index rebuilt from %d session file(s) (%s)", len(files), reason)
            try:
                self._write_index()
            except DataReadonlyError:
                logger.debug("History index not written: data root is read-only")
        return index

    def _scan_originals(self) -> dict[str, tuple[int, int]]:
        files: dict[str, tuple[int, int]] = {}
        if not self.history_dir.is_dir():
            return files
        for month_dir in self.history_dir.iterdir():
            if not month_dir.is_dir() or month_dir.name in (ACTIVE_DIR, EMBEDDINGS_DIR) \
                    or month_dir.name.startswith("."):
                continue
            for path in month_dir.glob("*.json"):
                try:
                    st = path.stat()
                except OSError:
                    continue
                files[f"{month_dir.name}/{path.name}"] = (st.st_size, st.st_mtime_ns)
        return files

    def _build_index_from_files(self, files: dict[str, tuple[int, int]]) -> HistoryIndex:
        index = HistoryIndex()
        for rel in sorted(files):
            path = self.history_dir / rel
            result = read_session_file(path)
            if not result.ok or not isinstance(result.payload, dict):
                logger.warning("Failed to index %s: %s %s", path, result.status, result.detail)
                continue
            sd = SessionData.from_dict(result.payload)
            index.sessions.append(IndexEntry(
                session_id=sd.session_id,
                file=rel,
                started_at=sd.started_at,
                duration_sec=sd.duration_sec,
                mode=sd.mode,
                turn_count=sd.turn_count,
                summary=sd.summary,
                summary_turn_count=sd.summary_turn_count,
                first_user_preview=_build_first_user_preview(sd),
                size_bytes=files[rel][0],
                search_text=_build_search_text(sd),
                promoted_to_semmem=sd.promoted_to_semmem,
                project_id=sd.project_id,
            ))
        index.sessions = _dedupe_by_session_id(index.sessions)
        _recount(index)
        return index

    def _put_entry(self, entry: IndexEntry) -> None:
        """(書き手スレッド) 1 件を差し替えた新しい索引を公開する。"""
        index = self._index_or_load()
        sessions = [e for e in index.sessions if e.session_id != entry.session_id]
        sessions.append(entry)
        self._publish(replace(index, sessions=sessions))

    def _publish(self, index: HistoryIndex) -> None:
        """(書き手スレッド) 集計を引き直して差し込み、書き出しを予約する。

        集計は **公開のたびに引き直す** (削除の経路が総数・総容量を陳腐化させて
        いた、2026-09-05 監査)。
        """
        _recount(index)
        self._index = index
        if not self._writer.running:
            # 書き手スレッドが無い (テスト・CLI) なら遅延させずにすぐ書く。
            self._write_index()
            return
        now = time.monotonic()
        first = self._index_dirty[0] if self._index_dirty is not None else now
        self._index_dirty = (first, now)

    def _tick(self, final: bool) -> None:
        """(書き手スレッド) 索引の遅延書き出し。"""
        dirty = self._index_dirty
        if dirty is None:
            return
        now = time.monotonic()
        first, last = dirty
        if final or now - last >= _INDEX_IDLE_SECONDS or now - first >= _INDEX_MAX_DELAY_SECONDS:
            self._write_index()

    def _write_index(self) -> None:
        """(書き手スレッド) 公開中の索引を ``index.json`` へ書く (原本の後)。"""
        index = self._index
        self._index_dirty = None
        if index is None:
            return
        index.updated_at = _now_iso()
        payload = {
            "built_from": _files_digest(self._files),
            "updated_at": index.updated_at,
            "total_sessions": index.total_sessions,
            "total_turns": index.total_turns,
            "total_size_mb": round(index.total_size_mb, 4),
            "sessions": [asdict(s) for s in index.sessions],
        }
        envelope = build_envelope(
            format_id=INDEX_FORMAT.format_id, format_version=INDEX_FORMAT.version,
            payload=payload, component="HistoryManager",
        )
        self._writer.write_now(
            self.history_dir / INDEX_FILE, jsoncodec.dumps_bytes(envelope), fsync=False,
        )

    def _save_index(self, index: HistoryIndex) -> None:
        """索引を公開して書く (保守・テスト用)。"""

        def run() -> None:
            self._index_or_load()  # 原本の一覧 (built_from) を揃えてから差し替える
            self._publish(index)
            self._write_index()

        self._run(run, format_id=INDEX_FORMAT.format_id)

    def ensure_search_text(self) -> int:
        """search_text が未設定のエントリにターン本文を補完

        起動時やインデックス読み込み時に呼び出す。
        Returns:
            補完したエントリ数
        """
        return self._backfill_entries(
            lambda e: not e.search_text,
            lambda e, s: replace(e, search_text=_build_search_text(s)),
            "search_text",
        )

    def ensure_first_user_preview(self) -> int:
        """``first_user_preview`` が未設定のエントリをセッション本体から補完

        既存インデックス (フィールド追加前に書かれたもの) の見出しを一覧で
        表示できるようにする。``ensure_search_text`` と同じく起動時に 1 回。

        Returns:
            補完したエントリ数
        """
        def fill(entry: IndexEntry, session: SessionData) -> IndexEntry | None:
            preview = _build_first_user_preview(session)
            return replace(entry, first_user_preview=preview) if preview else None

        return self._backfill_entries(lambda e: not e.first_user_preview, fill, "first_user_preview")

    def _backfill_entries(
        self,
        wanted: Callable[[IndexEntry], bool],
        fill: Callable[[IndexEntry, SessionData], IndexEntry | None],
        label: str,
    ) -> int:
        if not any(wanted(e) for e in self._load_index().sessions):
            return 0

        def run() -> int:
            index = self._index_or_load()
            sessions = list(index.sessions)
            updated = 0
            for i, entry in enumerate(sessions):
                if not wanted(entry):
                    continue
                filepath = self.history_dir / entry.file
                if not filepath.exists():
                    continue
                result = read_session_file(filepath)
                if not result.ok:
                    logger.warning("Failed to build %s for %s: %s",
                                   label, entry.session_id, result.detail)
                    continue
                new = fill(entry, SessionData.from_dict(result.payload))
                if new is not None:
                    sessions[i] = new
                    updated += 1
            if updated:
                self._publish(replace(index, sessions=sessions))
                logger.info("Backfilled %s for %d sessions", label, updated)
            return updated

        return self._run(run, format_id=INDEX_FORMAT.format_id)

    def rebuild_index(self) -> HistoryIndex:
        """ファイルスキャンでインデックスを再構築"""

        def run() -> HistoryIndex:
            files = self._scan_originals()
            self._files = files
            index = self._build_index_from_files(files)
            self._publish(index)
            self._write_index()
            logger.info("Index rebuilt: %d sessions", index.total_sessions)
            return index

        return self._run(run, format_id=INDEX_FORMAT.format_id)

    # ── ユーティリティ ──

    def _enforce_storage_limit(self, index: HistoryIndex) -> tuple[HistoryIndex, int]:
        """(書き手スレッド) ストレージ上限チェック＋古い順削除"""
        total_mb = self._calc_total_size_mb()
        if total_mb <= self.max_storage_mb:
            return index, 0
        return self._delete_oldest(index, total_mb - self.max_storage_mb)

    def _calc_total_size_mb(self) -> float:
        """history_dir 配下の総サイズ (MB)"""
        total = 0
        for root, _, files in os.walk(self.history_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)

    def _delete_oldest(self, index: HistoryIndex, target_mb: float) -> tuple[HistoryIndex, int]:
        """ストレージ上限超過分を最古から削除

        安全な順序: 削除対象を特定 → インデックス更新 → ファイル削除
        """
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
            return index, 0

        # インデックスから先に除去
        delete_ids = {e.session_id for e in to_delete}
        index = replace(index, sessions=[
            e for e in index.sessions if e.session_id not in delete_ids
        ])
        self._publish(index)

        # ファイル削除（インデックス更新後なので失敗しても不整合にならない）
        for entry in to_delete:
            try:
                self._writer.remove_now(self.history_dir / entry.file)
                self._forget_file(entry.file)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", entry.file, e)
        self._drop_embeddings(delete_ids)

        return index, len(to_delete)


def _index_from_payload(data: dict) -> HistoryIndex:
    known = {f.name for f in fields(IndexEntry)}
    sessions = []
    for s in data.get("sessions", []):
        if not isinstance(s, dict) or "session_id" not in s or "file" not in s:
            continue
        sessions.append(IndexEntry(
            session_id=s["session_id"],
            file=s["file"],
            started_at=s.get("started_at", ""),
            duration_sec=s.get("duration_sec", 0),
            mode=s.get("mode", "chat"),
            turn_count=s.get("turn_count", 0),
            **{k: v for k, v in s.items() if k in known and k not in (
                "session_id", "file", "started_at", "duration_sec", "mode", "turn_count",
            )},
        ))
    index = HistoryIndex(updated_at=data.get("updated_at", ""), sessions=sessions)
    _recount(index)
    return index


def _recount(index: HistoryIndex) -> None:
    index.total_sessions = len(index.sessions)
    index.total_turns = sum(e.turn_count for e in index.sessions)
    index.total_size_mb = sum(e.size_bytes for e in index.sessions) / (1024 * 1024)


def _dedupe_by_session_id(entries: list[IndexEntry]) -> list[IndexEntry]:
    """同じ ``session_id`` のエントリは 1 件 (最新) に畳む。

    再起動を跨いで続いたセッションが 2 ファイルに割れていた履歴に対し、
    索引が両方を載せると一覧で同じ会話が 2 回並ぶ。ターン数が多い方 →
    開始時刻が新しい方を残す。
    """
    best: dict[str, IndexEntry] = {}
    for e in entries:
        cur = best.get(e.session_id)
        if cur is None or (e.turn_count, e.started_at) > (
            cur.turn_count, cur.started_at,
        ):
            best[e.session_id] = e
    return [e for e in entries if best.get(e.session_id) is e]


def _now_iso() -> str:
    return utc_now()


# ── シングルトンファクトリ ──

_manager_cache: HistoryManager | None = None


def get_history_manager() -> HistoryManager:
    """シングルトンで HistoryManager を返す（インデックスキャッシュ有効化）

    API・CLI・sleep-time update など全レイヤーから共有で使用する。
    同一インスタンスを共有することで、セッション保存後のインデックス
    キャッシュが即座に一覧取得に反映される。

    初回構築で前回の取り残しの追記ログ (落ちたプロセスのアクティブなセッション) を
    畳む (G0 のチェックポイントの昇格の置き換え)。
    """
    global _manager_cache
    if _manager_cache is not None:
        return _manager_cache
    from backend.config import get_config, get_path_resolver
    resolver = get_path_resolver()
    cfg = get_config()
    history_dir = resolver.resolve_local("history_dir")
    mgr = HistoryManager(history_dir, cfg)
    mgr.fold_active_sessions()
    mgr.ensure_search_text()
    mgr.ensure_first_user_preview()
    _manager_cache = mgr
    return mgr


def active_base_model_name(config: dict | None) -> str:
    """アーカイブに刻むベースモデル名 (GGUF ファイル名) を解決する (純粋関数)。

    値の形式は ``ModelState.current_filename`` および
    ``ExperienceEntry.base_model`` と揃える (どちらも GGUF ファイル名)。同じ会話を
    経験バッファ側と突き合わせるとき、片方がフルパスだと照合できない。

    ``SessionData.base_model`` は API レスポンス (``SessionDetailResponse``) と UI
    (``SessionDetail.svelte`` の「モデル」表示) まで配線済みだったが、**書き込み側が
    どの経路にも無く**、Web 経路は未指定で既定の空文字列、CLI 経路は空文字列を
    ハードコードしていた。結果、UI の `{#if detail.base_model}` が常に偽で、
    「どのモデルが生成した会話か」を後から追えなかった (2026-08-03 に判明。
    ベースモデル差し替えによる日本語品質の劣化を調査した際、アーカイブからは
    モデルを特定できず backend.log を grep する必要があった)。

    セッション途中でのモデル差し替えは llama-server の再起動を伴うため実質的に
    起こらない。保存のたびに現行 config から解決すれば十分で、セッション開始時刻を
    別途保持する必要はない。
    """
    raw = ((config or {}).get("model_paths", {}) or {}).get("base_model") or ""
    return Path(raw).name


def active_base_model_key() -> str | None:
    """アーカイブに刻む base モデルの ``model_key`` (学習パーティションの active key)。

    ``ExperienceEntry.model_key`` と同じ値。解決できない (config 未ロード等) ときは
    ``None``。
    """
    try:
        from backend.config import get_path_resolver

        return get_path_resolver().active_model_key
    except Exception as exc:  # noqa: BLE001 — 保存自体は止めない
        logger.debug("model_key unavailable for the session: %s", exc)
        return None
