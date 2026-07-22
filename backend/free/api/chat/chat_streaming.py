"""ストリーミング関数 — SSEFrameBuilder / StreamPipeline 統合"""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, TYPE_CHECKING

from pathlib import Path

from backend.app_state import AppState
from backend.free.api.chat.chat_constants import (
    DEFAULT_KEEPALIVE_INTERVAL_SEC,
    MAX_STEP_QUEUE_SIZE,
)
from backend.free.api.chat._long_form_intent import (
    LongFormMode,
    detect_long_form_mode,
)
from backend.free.api.chat.chat_recorder import (
    record_meta_cognitive_response,
    record_response,
    record_long_form_response,
)
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import ChatMessage, GenerationParams, StepCallback
from backend.free.api.schemas import ChatResponse, TokenInfo
from backend.free.agent.deliberative import DeliberativeAgent
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.agent.output_format import infer_output_extension
from backend.free.core.sse import SSEFrameBuilder
from backend.free.core.stream_filter import (
    HeadBufferFilter, StreamThinkingFilter,
)
from backend.free.core.stream_pipeline import StreamPipeline
from backend.free.llm.local_client import LocalClient
from backend.free.llm.editor_filename import derive_editor_filename_stem
from backend.free.generation.document_gate import is_document_format
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.free.generation.validators import remove_code_fences
from backend.utils import estimate_tokens as _estimate_tokens, utc_compact_stamp
from backend.log_config import get_logger

from fastapi import HTTPException

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer

logger = get_logger("api.chat.streaming")

# SSE フレームビルダー（モジュールレベルの共有インスタンス）
sse = SSEFrameBuilder()


def meta_tool_routing_success(resp) -> bool:
    """meta-cognitive 応答でツールが 1 件以上実行成功したか (tool_routing 正例)。

    deliberative の ``tool_command_success is True`` と同義 (ツールが呼ばれ成功)。
    meta は複数ツールを呼ぶため any (1 件でも成功なら誘導は妥当)。tool_calls 空
    (= ツール未使用) は False。phase7 がクエリ単位の弱い正例として消費する。
    """
    if resp is None:
        return False
    return any(tc.get("success") for tc in (getattr(resp, "tool_calls", None) or []))


def meta_tool_routing_false_positive(resp) -> bool:
    """meta-cognitive 応答でツールを呼んだが全て失敗したか (tool_routing 誤検出)。

    deliberative の ``tool_command is not None and tool_command_success is False``
    と同義 (ルーティングしたが結果が伴わなかった同一ターンの明確な失敗)。tool_calls
    空 (未使用) は False。phase7 + パターン decay がクエリ単位で消費する。
    """
    if resp is None:
        return False
    tool_calls = getattr(resp, "tool_calls", None) or []
    return bool(tool_calls) and not any(tc.get("success") for tc in tool_calls)


def rag_signals_from_chunks(
    scored: list[tuple[str, float, str]] | None,
) -> tuple[bool, float | None]:
    """``scored_chunks`` から Level 0 経験記録用の ``(rag_used, rag_top1_score)`` を導出。

    ``scored_chunks`` は ``(chunk_id, score, content)`` の salience 降順リスト。
    空 / None なら ``(False, None)`` (RAG 未使用)。
    """
    if not scored:
        return False, None
    return True, scored[0][1]


def _emit_timing(
    state: AppState, timer: StageTimer | None,
    agent_layer: str, tokens_generated: int, mode: str = "",
) -> None:
    """StageTimer の計測結果をデバッグログに出力し、直近メトリクスを更新する"""
    if timer is None:
        return
    timing = timer.to_dict()

    # デバッグオーバーレイ用に直近メトリクスを AppState に保存
    from backend.app_state import LastRequestMetrics
    ttft_ms = timing.get("llm_first_token_ms")
    llm_total_ms = timing.get("llm_total_ms")
    tok_per_sec: float | None = None
    if llm_total_ms and llm_total_ms > 0 and tokens_generated > 0:
        tok_per_sec = round(tokens_generated / (llm_total_ms / 1000), 1)
    state.last_request_metrics = LastRequestMetrics(
        ttft_ms=ttft_ms,
        tok_per_sec=tok_per_sec,
        updated_at=time.monotonic(),
    )

    dl = state.debug_logger
    if dl is None:
        return
    if timing:
        dl.log_request_timing(
            timing, agent_layer=agent_layer,
            tokens_generated=tokens_generated, mode=mode,
        )

# ---------------------------------------------------------------------------
# セッション別キャンセルフラグ（chat.py の cancel エンドポイントからも参照）
# ---------------------------------------------------------------------------
_cancel_flags: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# cancel_scope — finally ブロックのクリーンアップを共通化
# ---------------------------------------------------------------------------

@asynccontextmanager
async def cancel_scope(session_id: str):
    """キャンセルフラグのスコープ管理

    ストリーミング関数の try/finally パターンを統一する。
    """
    _cancel_flags[session_id] = False
    try:
        yield
    finally:
        _cancel_flags.pop(session_id, None)


# ---------------------------------------------------------------------------
# 長文生成ファイル I/O ヘルパー
# ---------------------------------------------------------------------------

# ファイル出力判定パターン（出力/保存/書き出し/追記を含む指示）
_WRITE_HINT_RE = re.compile(
    r"(?:出力|保存|書[きく]込|書いて|書き出|追記|追加|作成|作って|生成|append|save|write|output|create)",
    re.IGNORECASE,
)

# 追記（append）意図の判定パターン
_APPEND_HINT_RE = re.compile(
    r"(?:追記|追加|append|続き.*(?:書|出力|保存|追加))", re.IGNORECASE,
)

# 既存ファイル参照が必要なパターン（追記・修正・確認+書く）
_NEEDS_EXISTING_RE = re.compile(
    r"(?:"
    r"追記|追加|append"  # 追記系
    r"|続き.*(?:書|出力|保存|追加)"  # 続編系
    r"|(?:確認|読[みむ]|見て|参照|チェック).*(?:追記|追加|続き|書|修正|直|改)"  # 確認+操作
    r"|(?:修正|リライト|書き直|加筆|編集).*(?:して|する)"  # 修正系
    r")",
    re.IGNORECASE,
)

# READ 参照意図: 既存ファイル参照に基づく **別物** の新規生成を示す表現。
# 既存 `_NEEDS_EXISTING_RE` は READ + 編集動詞 (追記/書/修正) を要求するため、
# 「X を参照して Y を作成」のように出力対象が分離するケースを拾えない。
# 本パターンは「参照源」だけを検知し、出力先シフトと既存内容の context 取り込み
# (read_existing_for_append) のトリガとして使う。
_READ_REF_RE = re.compile(
    r"(?:参照|参考|読[みむ]|見て|これを|これに|を基に|を元に|に基づ|"
    r"based on|refer(?: to)?|read)",
    re.IGNORECASE,
)

# LLM が出力しがちな見出し行を除去するパターン
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


async def read_existing_for_append(
    query: str, state: AppState,
) -> str:
    """既存ファイル内容を生成前に読み取る

    追記・修正・加筆など、既存ファイル内容を踏まえた生成が必要な場合に
    加え、「X を参照して Y を作成」のような READ 参照意図でもファイル内容を
    返し、long-form orchestrator の ``existing_content`` 引数として
    LLM context に渡せるようにする。

    Returns:
        既存ファイル内容。参照不要の場合や読み取り失敗時は空文字列。
    """
    if not (_NEEDS_EXISTING_RE.search(query) or _READ_REF_RE.search(query)):
        return ""

    file_path = _extract_file_path(query)
    if not file_path:
        return ""

    registry = state.tools_registry
    if registry is None or not registry.has("read_file"):
        return ""

    try:
        content = await registry.execute("read_file", file_path=file_path)
        if is_tool_error(content):
            return ""
        logger.info(
            "Read existing file for context: %d chars from %s",
            len(content), file_path,
        )
        return content
    except Exception as e:
        logger.warning("Failed to read existing file: %s", e)
        return ""


def clean_generated_text(text: str) -> str:
    """LLM 生成テキストからファイル出力に不要な要素を除去する"""
    # Markdown 見出し行（# 本文 等）を除去
    cleaned = _HEADING_LINE_RE.sub("", text)
    # 連続する空行を1つに圧縮
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


#: 逐語重複トリムの対象とする段落の最小文字数。短い定型句 (「敬具」「以上」
#: や結びの挨拶等) の正当な繰り返しを誤って落とさないための下限。日本語の
#: 1 段落は 40〜60 文字程度から成立するため 40 とする。
_DEDUP_MIN_PARAGRAPH_CHARS = 40


def dedup_verbatim_paragraphs(text: str) -> str:
    """既出段落の逐語再掲を除去する (長文ユニット結合の重複対策)。

    続きユニットが直前ユニットの段落を丸ごと再掲する退行 (2026-07-15:
    7 ファイルで冒頭段落の末尾再掲・セクション丸ごと重複) への決定論
    ガード。空白正規化後に完全一致する ``_DEDUP_MIN_PARAGRAPH_CHARS``
    文字以上の段落について、2 回目以降の出現を破棄する。
    """
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for para in paragraphs:
        norm = re.sub(r"\s+", " ", para).strip()
        if len(norm) >= _DEDUP_MIN_PARAGRAPH_CHARS:
            if norm in seen:
                removed += 1
                continue
            seen.add(norm)
        kept.append(para)
    if removed:
        logger.info(
            "Removed %d verbatim duplicated paragraph(s) from long-form output",
            removed,
        )
    return "\n\n".join(kept)


# エディタ出力用: 連続改行 (途中に whitespace のみの空行を含む) を単一 \n に圧縮する。
# file 経路の clean_generated_text とは異なり markdown 見出しは保持する。
_EDITOR_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def _normalize_editor_text(text: str) -> str:
    """エディタ出力用に空行を完全除去し前後空白を整える。

    LLM 生成本文は orchestrator がユニット間に ``\\n\\n`` を挿入する仕様 + LLM 自身が
    段落区切りで空行を入れるため、エディタへ流すと「1 行飛ばし」表示になる。
    エディタ経路では markdown 段落構造より「行を詰める」見え方を優先する。
    """
    return _EDITOR_BLANK_LINE_RE.sub("\n", text).strip()


# Markdown 出力意図を示すヒント。``md 形式`` / ``.md ファイル`` / ``markdown`` 等。
_MD_EXT_HINT_RE = re.compile(
    r"(?:"
    r"\.md(?:\b|ファイル|形式|で|に|を)"  # 「.md ファイル」「.md 形式」など
    r"|md[\s ]?(?:形式|ファイル|で出力|で保存|で書)"  # 「md 形式」「md ファイル」
    r"|markdown"
    r"|マークダウン"
    r")",
    re.IGNORECASE,
)

def _infer_output_extension(query: str, default: str = ".txt") -> str:
    """ユーザー指示文から出力ファイルの拡張子を推論する (agent 層に委譲)。

    long_form / SPLIT / CONTINUE 自動命名と meta_cognitive のディレクトリ解決で
    同一ロジックを共有するため ``backend.free.agent.output_format`` に集約している。
    """
    return infer_output_extension(query, default)


# エディタ出力時の言語識別子マップ (フロント側 Monaco の syntax highlight 用)。
# 長文生成 (仕様書 / 設計書 / 画面設計) はほぼ markdown 整形なので、
# 未知拡張子は markdown にフォールバックする。
_EDITOR_LANGUAGE_MAP = {
    ".md": "markdown",
    ".txt": "markdown",
    ".py": "python",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
}


def _editor_language_for_extension(ext: str) -> str:
    """エディタ出力用の言語識別子を拡張子から推論する。"""
    return _EDITOR_LANGUAGE_MAP.get(ext, "markdown")


def _resolve_editor_output_format(query: str, is_code: bool) -> tuple[str, str]:
    """エディタ出力の ``(拡張子, 言語)`` を解決する。

    ``query`` に markdown ヒントがあれば markdown を優先し、無ければ
    ``content_type == "code"`` のとき python (.py)、それ以外は従来通り
    markdown (.md) を返す。``_infer_output_extension`` は ``.py`` を判定
    できないため、コード生成が markdown 扱いになる問題をここで補正する。
    """
    if _MD_EXT_HINT_RE.search(query):
        return ".md", "markdown"
    if is_code:
        return ".py", "python"
    return ".md", "markdown"


def _resolve_long_form_target_path(file_path: str, query: str = "") -> str:
    """`long_form_write_file` 用の保存先パスを解決する。

    - 既存ディレクトリ → 配下に ``output_<UTC><ext>`` を自動付与。
      ``<ext>`` は ``query`` から推論 (``md 形式`` / ``markdown`` 等で ``.md``、
      他は既定 ``.txt``)。
    - 既存 *ファイル* + READ 参照意図 → ``{stem}_generated_<UTC>{suffix}``
      にシフトしてユーザ指定の参照ファイルを保護する
    - それ以外は原文のまま返し、``write_file`` 側のファイル作成挙動に委ねる

    ``Desktop\\test`` のように配下出力意図でディレクトリを指定された場合、
    従来は ``write_file`` の ``is_dir()`` チェックで失敗していた。
    また既存ファイルパスを ``参照して...作成`` 形式で指定されると、参照元の
    ファイルが上書きされてしまう問題があったため、READ 意図検知で保護する。
    """
    try:
        p = Path(file_path)
        if p.is_dir():
            ext = _infer_output_extension(query)
            auto_name = f"output_{utc_compact_stamp()}{ext}"
            resolved = str(p / auto_name)
            logger.info(
                "Long-form: '%s' is an existing directory; "
                "auto-deriving filename: %s",
                file_path, resolved,
            )
            return resolved
        if p.is_file() and query and _READ_REF_RE.search(query):
            stem = p.stem
            suffix = p.suffix or ".txt"
            new_name = f"{stem}_generated_{utc_compact_stamp()}{suffix}"
            resolved = str(p.parent / new_name)
            logger.warning(
                "Long-form: '%s' is an existing file referenced with READ "
                "intent; shifting output to '%s' to protect source",
                file_path, resolved,
            )
            return resolved
    except OSError as e:
        # ファイル名にできない文字 / 解決不能パス等。原文を返して
        # write_file 側で graceful error にする。
        logger.debug(
            "Long-form path resolution skipped (%s): %r", e, file_path,
        )
    return file_path


# SPLIT モード: file_name → 安全なファイル名 (英数字 + アンダースコア) への変換
_SPLIT_FILE_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug_for_split_file(file_name: str | None, idx: int) -> str:
    """SPLIT モードの ``file_name`` を安全な slug に変換する。

    LLM が CJK や記号を返したり ``None`` だったりするため、防御的に処理:
    - 英数字 + ``_`` + ``-`` のみ残し、他は ``_`` に置換
    - 32 char に切り詰め
    - 空 / 既定値 / None の場合は ``unit_<idx+1>`` フォールバック (連番)
    """
    if not file_name:
        return f"unit_{idx + 1:02d}"
    slug = _SPLIT_FILE_NAME_SAFE_RE.sub("_", file_name).strip("_-")
    slug = slug[:32].rstrip("_-")
    if not slug:
        return f"unit_{idx + 1:02d}"
    # 数字始まりは可読性のため接頭辞を付ける
    if slug[0].isdigit():
        slug = f"unit_{slug}"
    return slug


def _resolve_split_unit_path(
    base_path: str, idx: int, slug: str, used_paths: set[str],
    *, extension: str = ".txt",
) -> str:
    """SPLIT モードで 1 unit を書き出すパスを解決する。

    パターン: ``{source_stem}_{idx:02d}_{slug}{extension}``
    (ベースパスと同一ディレクトリ)。既に書き出した相対パスと衝突したら
    ``_2``, ``_3`` ... を付ける。

    ``extension`` はユーザー指示文から推論された拡張子 (``.txt`` / ``.md`` 等)。
    既定は後方互換のため ``.txt``。
    """
    p = Path(base_path)
    parent = p.parent
    # ベースパスがディレクトリの場合は配下に出力。ファイルの場合は stem を共有。
    if p.is_dir():
        stem = "output"
        parent = p
    elif p.is_file():
        stem = p.stem
    else:
        # 存在しないパス: ファイル扱いで stem 抽出
        stem = p.stem or "output"
    candidate = parent / f"{stem}_{idx + 1:02d}_{slug}{extension}"
    n = 2
    while str(candidate) in used_paths:
        candidate = parent / f"{stem}_{idx + 1:02d}_{slug}_{n}{extension}"
        n += 1
    return str(candidate)


async def split_write_single_unit(
    *,
    base_path: str,
    idx: int,
    total: int,
    heading: str,
    file_name: str | None,
    content: str,
    state: AppState,
    used_paths: set[str],
    extension: str = ".txt",
) -> str | None:
    """SPLIT モードで 1 unit を個別ファイルに書き込む。

    ``extension`` はユーザー指示文から推論された拡張子 (``.txt`` / ``.md`` 等)。
    既定は後方互換のため ``.txt``。

    成功時はそのファイルパスを返す。失敗時は ``None``。
    """
    registry = state.tools_registry
    if registry is None or not registry.has("write_file"):
        logger.warning("write_file tool not available for SPLIT unit %d", idx)
        return None

    slug = _slug_for_split_file(file_name, idx)
    out_path = _resolve_split_unit_path(
        base_path, idx, slug, used_paths, extension=extension,
    )
    cleaned = clean_generated_text(content)
    # ファイル先頭に機能名 (heading) を付与すると、ユーザーが内容を識別しやすい
    body = f"# {heading}\n\n{cleaned}\n" if heading else f"{cleaned}\n"
    try:
        await registry.execute("write_file", file_path=out_path, content=body)
        used_paths.add(out_path)
        logger.info(
            "Long-form SPLIT unit [%d/%d] written: %s", idx + 1, total, out_path,
        )
        return out_path
    except Exception as e:
        logger.error("Long-form SPLIT unit %d write failed: %s", idx, e)
        return None


async def split_write_index(
    *,
    base_path: str,
    written: list[dict],
    state: AppState,
) -> str | None:
    """SPLIT モード全 unit 書込み完了後、INDEX.md を生成する。

    ``written`` は ``[{"path": str, "heading": str, "idx": int}]``。
    """
    registry = state.tools_registry
    if registry is None or not registry.has("write_file") or not written:
        return None
    p = Path(base_path)
    parent = p.parent if not p.is_dir() else p
    stem = (p.stem if p.is_file() else "output") or "output"
    index_path = parent / f"{stem}_INDEX.md"
    lines: list[str] = [f"# {stem} — 機能別 詳細仕様書 索引", ""]
    for item in written:
        rel = Path(item["path"]).name
        lines.append(f"- [{item['heading']}](./{rel})")
    body = "\n".join(lines) + "\n"
    try:
        await registry.execute(
            "write_file", file_path=str(index_path), content=body,
        )
        logger.info("Long-form SPLIT index written: %s", index_path)
        return str(index_path)
    except Exception as e:
        logger.error("Long-form SPLIT index write failed: %s", e)
        return None


async def long_form_write_file(
    query: str, content: str, state: AppState,
) -> str | None:
    """長文生成後にファイル書き込みが必要なら実行する

    「追記」「続きを書いて」等の場合は既存内容の末尾に連結する。
    書き込み前に見出し行等の不要な要素を除去する。
    抽出パスが既存ディレクトリの場合は配下に ``output_<UTC>.txt`` を
    自動生成する (write_file の "is a directory" エラーを回避)。

    Returns:
        書き込み結果メッセージ。書き込み不要または失敗時は None。
    """
    if not _WRITE_HINT_RE.search(query):
        return None

    file_path = _extract_file_path(query)
    if not file_path:
        return None

    registry = state.tools_registry
    if registry is None or not registry.has("write_file"):
        logger.warning("write_file tool not available for long-form output")
        return None

    # ディレクトリ指定の場合は配下に自動ファイル名を付与。
    # 既存ファイル + READ 参照意図の場合は出力先をシフトして元ファイルを保護。
    file_path = _resolve_long_form_target_path(file_path, query)

    # コード拡張子 (.py / .js 等) の場合は LLM が付ける markdown コードフェンスを除去。
    # markdown / txt ではフェンスは本文の一部なので除去しない。
    suffix = Path(file_path).suffix
    if _editor_language_for_extension(suffix) != "markdown":
        content = remove_code_fences(content)

    # 生成テキストのクリーニング。ドキュメント形式 (docx/pptx/xlsx/odf/md) は
    # markdown 構造 (見出し / 表 / リスト) が export Writer の組版入力になるため、
    # 見出しを温存し空行圧縮のみ行う。clean_generated_text は見出し行を全削除する
    # ため、ここで適用すると pptx のスライド分割 (見出し level<=2 区切り) や docx の
    # 章節構造が壊れる。それ以外 (.txt 等) は従来どおり余計な見出し行を除去する。
    if is_document_format(suffix):
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
    else:
        content = clean_generated_text(content)

    # ユニット結合で生じた既出段落の逐語再掲を除去する (コード拡張子は
    # 正当な重複がありうるため文書系のみ)。
    if _editor_language_for_extension(suffix) == "markdown" or is_document_format(suffix):
        content = dedup_verbatim_paragraphs(content)

    try:
        # 追記モード: 既存ファイルの内容に連結
        if _APPEND_HINT_RE.search(query) and registry.has("read_file"):
            existing = await registry.execute("read_file", file_path=file_path)
            if not is_tool_error(existing):
                content = existing.rstrip() + "\n\n" + content
                logger.info("Long-form append mode: prepended %d chars from %s", len(existing), file_path)

        result = await registry.execute("write_file", file_path=file_path, content=content)
        logger.info("Long-form file output: %s", result)
        return str(result)
    except Exception as e:
        logger.error("Long-form file output failed: %s", e)
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Reactive ストリーミング
# ---------------------------------------------------------------------------

async def stream_reactive(
    content: str, instance_name: str, context_size: int,
) -> AsyncIterator[str]:
    """Reactive 層の応答を SSE ストリーミングで返す"""
    yield sse.agent_layer("reactive")
    yield sse.token(content)
    token_info = {"used": 0, "limit": context_size, "pct": 0, "instance_name": instance_name}
    yield sse.token_info(token_info)
    yield sse.done()


# ---------------------------------------------------------------------------
# Meta-Cognitive ストリーミング / 同期
# ---------------------------------------------------------------------------

def _build_meta_cognitive_agent_runner(
    agent: MetaCognitiveAgent,
    *,
    query: str,
    system_prompt: str,
    conversation: list[ChatMessage],
    client: LocalClient,
    state: AppState,
    session_id: str,
    mode: str,
    generation_params: GenerationParams | None,
    step_queue: "asyncio.Queue[dict | None]",
    result_holder: dict,
    output_target: str = "file",
):
    """MetaCognitive agent.process() をバックグラウンド実行するコルーチンを生成。

    ステップは step_queue に push され、完了・例外時に None で終端を通知する。
    結果または例外は result_holder に格納して呼び出し側に返す。
    """
    async def on_step(step_data: dict) -> None:
        await step_queue.put(step_data)

    async def _run_agent() -> None:
        try:
            resp = await agent.process(
                query=query,
                system_prompt=system_prompt,
                conversation=conversation,
                llm_client=client,
                tools_registry=state.tools_registry,
                on_step=on_step,
                generation_params=generation_params,
                session_id=session_id,
                mode=mode,
                output_target=output_target,
            )
            result_holder["resp"] = resp
        except Exception as e:
            result_holder["error"] = e
        finally:
            await step_queue.put(None)

    return _run_agent


async def _drain_meta_cognitive_steps(
    step_queue: "asyncio.Queue[dict | None]",
    session_id: str,
    keepalive_interval: float,
):
    """step_queue から step フレームを逐次 yield する（keepalive / cancel 対応）。"""
    while True:
        try:
            step_data = await asyncio.wait_for(
                step_queue.get(), timeout=keepalive_interval,
            )
        except asyncio.TimeoutError:
            yield sse.keepalive()
            continue

        if step_data is None:
            return

        if _cancel_flags.get(session_id):
            return

        yield sse.step(step_data)


async def _emit_meta_cognitive_result_frames(resp) -> AsyncIterator[str]:
    """MetaCognitive 応答から最終フレーム（task_result または token）を yield する。"""
    if resp is None:
        return

    # エディタ経路: 生成コードを専用チャネルで送出 (チャット本文には混ぜない)
    # editor_artifacts は dataclass の field(default_factory=list) で常にリスト
    if resp.editor_artifacts:
        for art in resp.editor_artifacts:
            yield sse.editor_code(art.content, language=art.language, filename=art.filename)

    if resp.tasks:
        logger.debug(
            "MetaCognitive final: sending %d task results as step events",
            len(resp.tasks),
        )
        for task in resp.tasks:
            detail = task.description
            if task.result:
                detail += f" {task.result[:500]}"
            logger.debug(
                "MetaCognitive task_result: status=%s, detail=%s",
                task.status, detail[:120],
            )
            yield sse.step({"type": "task_result", "detail": detail, "status": task.status})
    else:
        logger.debug(
            "MetaCognitive final: no tasks, sending content as token (%d chars)",
            len(resp.content),
        )
        yield sse.token(resp.content)


def _finalize_meta_cognitive_stream(
    resp,
    *,
    state: AppState,
    messages: list[ChatMessage],
    session_id: str,
    query: str,
    mode: str,
    instance_name: str,
    context_size: int,
    timer: "StageTimer | None",
    t_start: float,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> TokenInfo:
    """MetaCognitive ストリーム完了時の記録・タイミング計測を行い TokenInfo を返す。"""
    if timer:
        timer.stop("llm_total_ms")

    elapsed = time.monotonic() - t_start
    steps = resp.steps if resp else 0
    tool_calls_count = len(resp.tool_calls) if resp else 0
    logger.info(
        "MetaCognitive stream complete: steps=%d, tool_calls=%d, elapsed=%.2fs",
        steps, tool_calls_count, elapsed,
    )

    content = resp.content if resp else ""
    step_credits = resp.step_credits if resp else []
    estimated_tokens = max(1, _estimate_tokens(content))
    record_meta_cognitive_response(
        state, content, messages, session_id,
        query, mode, estimated_tokens, step_credits,
        private=private,
        agent_loops=steps,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
        tool_routing_success=meta_tool_routing_success(resp),
        tool_routing_false_positive=meta_tool_routing_false_positive(resp),
    )

    _emit_timing(state, timer, "meta_cognitive", estimated_tokens, mode=mode)
    return make_token_info(messages, estimated_tokens, context_size, instance_name)


async def stream_meta_cognitive(
    agent: MetaCognitiveAgent, query: str, system_prompt: str,
    conversation: list[ChatMessage], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[ChatMessage], mode: str,
    *, generation_params: GenerationParams | None = None,
    keepalive_interval: float = 15.0,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
):
    """Meta-Cognitive 層の SSE ストリーミング（ステップフレーム付き）

    agent.process() をバックグラウンドタスクで実行し、on_step コールバック
    からのステップ通知をリアルタイムで SSE フレームとして送信する。
    定期的に keepalive コメントを送信してクライアントのタイムアウトを防止する。
    """
    async with cancel_scope(session_id):
        t_start = time.monotonic()
        step_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        result_holder: dict = {"resp": None, "error": None}
        outcome_success = False

        run_agent = _build_meta_cognitive_agent_runner(
            agent,
            query=query, system_prompt=system_prompt,
            conversation=conversation, client=client, state=state,
            session_id=session_id, mode=mode,
            generation_params=generation_params,
            step_queue=step_queue, result_holder=result_holder,
            output_target=output_target,
        )

        try:
            yield sse.agent_layer("meta_cognitive")
            yield sse.step({"type": "plan", "detail": "Generating task plan...", "status": "running"})

            if timer:
                timer.start("llm_total_ms")
            agent_task = asyncio.create_task(run_agent())

            async for frame in _drain_meta_cognitive_steps(
                step_queue, session_id, keepalive_interval,
            ):
                yield frame

            await agent_task

            if result_holder["error"] is not None:
                raise result_holder["error"]

            resp = result_holder["resp"]

            if not _cancel_flags.get(session_id):
                async for frame in _emit_meta_cognitive_result_frames(resp):
                    yield frame

            ti = _finalize_meta_cognitive_stream(
                resp,
                state=state, messages=messages, session_id=session_id,
                query=query, mode=mode, instance_name=instance_name,
                context_size=context_size, timer=timer, t_start=t_start,
                private=private,
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
            )
            yield sse.token_info(ti)
            yield sse.done()
            outcome_success = True

        except Exception as e:
            logger.error("MetaCognitive stream error: %s", e)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "meta_cognitive", 0, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            dl = getattr(state, "debug_logger", None)
            if dl is not None:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                resp_obj = result_holder.get("resp")
                tokens_out = (
                    int(getattr(resp_obj, "tokens", 0) or 0)
                    if resp_obj is not None else 0
                )
                # SSE 完走 = success ではなくタスク成否を反映する。ファイル未作成の
                # 失敗ターンが success=True で記録され、負例が学習に伝播しない
                # 問題 (2026-07-15) への対策。
                quality_signals: dict = {"agent_layer": "meta_cognitive"}
                task_list = list(getattr(resp_obj, "tasks", None) or [])
                if task_list:
                    failed_tasks = sum(
                        1 for t in task_list
                        if getattr(t, "status", "") == "failed"
                    )
                    writes = sum(
                        1 for tc in (getattr(resp_obj, "tool_calls", None) or [])
                        if tc.get("tool") == "write_file" and tc.get("success")
                    )
                    quality_signals.update({
                        "tasks": len(task_list),
                        "failed_tasks": failed_tasks,
                        "writes": writes,
                    })
                    if failed_tasks:
                        outcome_success = False
                dl.log_outcome(
                    kind="chat_response",
                    success=outcome_success,
                    duration_ms=elapsed_ms,
                    tokens_out=tokens_out,
                    quality_signals=quality_signals,
                )


async def sync_meta_cognitive(
    agent: MetaCognitiveAgent, query: str, system_prompt: str,
    conversation: list[ChatMessage], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[ChatMessage], mode: str,
    *, generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> ChatResponse:
    """Meta-Cognitive 層の同期応答"""
    try:
        if timer:
            timer.start("llm_total_ms")
        tools_registry = state.tools_registry
        resp = await agent.process(
            query=query,
            system_prompt=system_prompt,
            conversation=conversation,
            llm_client=client,
            tools_registry=tools_registry,
            generation_params=generation_params,
            session_id=session_id,
            mode=mode,
            output_target=output_target,
        )

        if timer:
            timer.stop("llm_total_ms")

        # 非ストリームではエディタチャネルが無いため、エディタ経路の生成コードは
        # コードブロックとして応答本文に畳み込む (CLI 等で内容を失わない)。
        response_text = resp.content
        editor_artifacts = getattr(resp, "editor_artifacts", None)
        if editor_artifacts:
            blocks = "\n\n".join(
                f"```{art.language}\n{art.content}\n```" for art in editor_artifacts
            )
            response_text = blocks if not response_text else f"{response_text}\n\n{blocks}"

        estimated_tokens = max(1, _estimate_tokens(response_text))
        record_meta_cognitive_response(
            state, response_text, messages, session_id,
            query, mode, estimated_tokens, resp.step_credits,
            private=private,
            agent_loops=resp.steps,
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
            tool_routing_success=meta_tool_routing_success(resp),
            tool_routing_false_positive=meta_tool_routing_false_positive(resp),
        )

        _emit_timing(state, timer, "meta_cognitive", estimated_tokens, mode=mode)

        token_info_dict = make_token_info(messages, estimated_tokens,
                                          context_size, instance_name)
        return ChatResponse(
            response=response_text,
            token_info=TokenInfo(**token_info_dict),
            session_id=session_id,
            agent_layer="meta_cognitive",
        )
    except Exception as e:
        logger.error("MetaCognitive error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# Long-form ストリーミング / 同期
# ---------------------------------------------------------------------------

@dataclass
class _LongFormStreamState:
    """`stream_long_form` のループ mutable 状態を集約。"""

    tokens_generated: int = 0
    full_response: str = ""
    first_token_recorded: bool = False


async def _emit_long_form_init_steps(
    query: str, file_output_mode: bool,
) -> AsyncIterator[str]:
    """長文生成開始時の初期 SSE フレームを yield する。

    `agent_layer` → (file_output_mode 時) `long_form_file_mode` → `long_form_plan`。
    """
    yield sse.agent_layer("meta_cognitive")
    if file_output_mode:
        file_path = _extract_file_path(query)
        yield sse.step({
            "type": "long_form_file_mode",
            "detail": f"ファイル出力モード: {file_path}",
            "status": "running",
        })
    yield sse.step({
        "type": "long_form_plan",
        "detail": "Generating plan...",
        "status": "running",
    })


async def _flush_step_queue_to_sse(
    step_queue: list[dict],
) -> AsyncIterator[str]:
    """`step_queue` の蓄積フレームを順次 yield して空にする。"""
    while step_queue:
        step_data = step_queue.pop(0)
        yield sse.step(step_data)


async def _flush_step_queue_split_aware(
    step_queue: list[dict],
    *,
    long_form_mode: LongFormMode,
    base_path: str,
    used_paths: set[str],
    written: list[dict],
    state: AppState,
    extension: str = ".txt",
) -> AsyncIterator[str]:
    """SPLIT モード対応の step_queue flush。

    ``long_form_unit_file`` イベントを per-unit ファイル書込みに変換し、
    結果を ``written`` リストに追記する。それ以外のイベントは
    :func:`_flush_step_queue_to_sse` と同じく SSE フレームとして yield する。

    SPLIT 以外のモードでは ``_flush_step_queue_to_sse`` と完全等価に動作する。
    """
    while step_queue:
        step_data = step_queue.pop(0)
        if (
            long_form_mode == LongFormMode.SPLIT
            and step_data.get("type") == "long_form_unit_file"
            and base_path
        ):
            idx = int(step_data.get("idx", 0))
            total = int(step_data.get("total", 0))
            heading = str(step_data.get("heading", ""))
            file_name = step_data.get("file_name")
            content = str(step_data.get("content", ""))
            written_path = await split_write_single_unit(
                base_path=base_path,
                idx=idx,
                total=total,
                heading=heading,
                file_name=file_name,
                content=content,
                state=state,
                used_paths=used_paths,
                extension=extension,
            )
            if written_path:
                written.append({
                    "path": written_path, "heading": heading, "idx": idx,
                })
                yield sse.step({
                    "type": "long_form_file_written",
                    "detail": (
                        f"[{idx + 1}/{total}] {heading} → {Path(written_path).name}"
                    ),
                    "status": "done",
                })
            else:
                yield sse.step({
                    "type": "long_form_file_written",
                    "detail": f"[{idx + 1}/{total}] {heading} → write failed",
                    "status": "error",
                })
            continue
        yield sse.step(step_data)


def _emit_long_form_episode(
    sess_state: AppState,
    *,
    session_id: str,
    mode: str,
    query: str,
    delivered: str,
    metrics: dict,
    private: bool,
) -> None:
    """長文生成ターンを MDP episode として agent_trace へ記録する。

    cogwriter/recurrent 経路は AgentTracer を経由しないため、MDP ingest の
    decision/failure ファクトから長文ターンが欠落していた。単一ステップの
    episode として task / 成果メトリクス / 成否を残す。
    """
    tracer = getattr(sess_state, "agent_tracer", None)
    if tracer is None or private:
        return
    try:
        from backend.free.agent.agent_tracer import MDPStep

        units_completed = int(metrics.get("units_completed", 0) or 0)
        validation_errors = int(metrics.get("validation_errors", 0) or 0)
        success = units_completed > 0 and validation_errors == 0
        episode_id = tracer.begin_episode(session_id, mode)
        tracer.record_step(episode_id, MDPStep(
            step_index=0,
            state={
                "task": query[:200],
                "layer": "long_form",
                "strategy": str(metrics.get("strategy") or ""),
                "content_type": str(metrics.get("content_type") or ""),
                "units_total": int(metrics.get("units_total", 0) or 0),
                "units_completed": units_completed,
                "validation_errors": validation_errors,
            },
            action="long_form_generate",
            observation=(delivered or "")[:200],
            reward=1.0 if success else 0.0,
        ))
        tracer.end_episode(episode_id, "success" if success else "failure")
    except Exception as e:
        logger.debug("long_form episode emit skipped: %s", e)


async def _finalize_long_form_stream(
    state: _LongFormStreamState,
    sess_state: AppState,
    orchestrator: LongFormOrchestrator,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    mode: str,
    instance_name: str,
    context_size: int,
    file_output_mode: bool,
    timer: StageTimer | None,
    t_start: float,
    private: bool = False,
    long_form_mode: LongFormMode = LongFormMode.CONTINUE,
    split_written: list[dict] | None = None,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> AsyncIterator[str]:
    """長文生成終端処理: timer 停止 + record + write_file + token_info + done。

    SPLIT モード時は単一ファイル書込みをスキップし、INDEX.md を生成する。
    ``split_written`` は ``stream_long_form`` 内で `long_form_unit_file` を
    受けて書き込んだ各ファイルの ``[{"path", "heading", "idx"}]``。

    ``output_target == "editor"`` の場合はディスク書込みをスキップし、
    `sse.editor_code` で生成本文をエディタペインへ送出する (coding モードの
    ``_dispatch_meta_cognitive`` 経路の挙動に揃える)。
    """
    if timer:
        timer.stop("llm_total_ms")
    elapsed = time.monotonic() - t_start
    logger.info(
        "Long-form stream complete: strategy=%s, tokens=%d, elapsed=%.2fs, session=%s, file_mode=%s, lf_mode=%s, output_target=%s",
        orchestrator.strategy_name, state.tokens_generated, elapsed,
        session_id, file_output_mode, long_form_mode.value, output_target,
    )
    metrics = getattr(orchestrator, "last_metrics", {})
    # coding の editor/file 出力は orchestrator が検証・修正した assembled
    # (last_code_output) を配信する。生ストリーム (full_response) は review の
    # revise トークンが二重追記されるため、コード出力の確定本文には使わない。
    is_code = getattr(orchestrator, "last_content_type", None) == "code"
    code_output = getattr(orchestrator, "last_code_output", None)
    text_output = getattr(orchestrator, "last_text_output", None)
    if is_code and isinstance(code_output, str) and code_output:
        delivered = code_output
    elif (not is_code) and isinstance(text_output, str) and text_output:
        # document_quality モード: 改稿済みユニットから組んだ確定本文。生ストリーム
        # は revise トークンを二重追記するため file 出力には使わない (CODE と対称)。
        delivered = text_output
    else:
        delivered = state.full_response
    record_long_form_response(
        sess_state, delivered, messages, session_id,
        query, mode, state.tokens_generated, metrics,
        private=private,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
    )

    # 長文経路も MDP episode を残す (agent_trace 互換)。従来は agent_trace を
    # 経由しないため MDP ingest (decision/failure ファクト) から長文ターンが
    # 全欠落していた (2026-07-15: 最重要の問題経路 10 ターンが学習素材にならず)。
    _emit_long_form_episode(
        sess_state, session_id=session_id, mode=mode, query=query,
        delivered=delivered, metrics=metrics, private=private,
    )

    # 構文エラーを含むコードがエディタ/ディスクへそのまま出力されると、
    # ユーザは破損に気づけない (validate は事後計測でブロックしない)。
    # validation_errors > 0 のときは警告ステップを surface して認識させる。
    validation_errors = int(metrics.get("validation_errors", 0) or 0)
    if validation_errors > 0:
        logger.warning(
            "Long-form output has %d validation error(s) (session=%s); "
            "surfacing warning to user",
            validation_errors, session_id,
        )
        yield sse.step({
            "type": "task_result",
            "detail": (
                f"⚠ 生成コードに構文エラーが {validation_errors} 件検出されました。"
                "そのまま実行する前に内容を確認してください。"
            ),
            "status": "failed",
        })

    if getattr(orchestrator, "last_needed_clarification", False) is True:
        # 主題不明で確認質問を返しただけの応答。delivered は確認質問文なので、
        # そのまま file 書込み/editor 送出してしまうと確認質問がドキュメントに
        # なってしまう (2026-07-22 発見のトピック混入バグの対策で追加)。
        write_result = None
    elif output_target == "editor":
        # エディタ経路: ディスクには書込みせず生成本文を editor_code フレームで送出。
        # SPLIT モードでもエディタ送出を優先 (per-unit ファイルは on_step 側で完了済み)。
        ext, language = _resolve_editor_output_format(query, is_code)
        # コード生成時は確定本文 (delivered = 検証・修正済み assembled) を使い、
        # markdown コードフェンスを除去する。
        body = remove_code_fences(delivered) if is_code else delivered
        # 空行を含む連続改行を単一 \n に圧縮 (markdown 見出しは保持)。
        editor_text = _normalize_editor_text(body)
        # 生成内容からアシストモデルで ASCII snake_case のファイル名を導出する
        # (日本語見出しをそのまま流用するとタブ名が日本語化するため)。
        stem = await derive_editor_filename_stem(
            sess_state.assist_client, content=body, hint=query, language=language,
        )
        filename = f"{stem}{ext}"
        yield sse.editor_code(
            editor_text, language=language, filename=filename,
        )
        write_result = None
    elif long_form_mode == LongFormMode.SPLIT and split_written is not None:
        # SPLIT: per-unit 書込みは既に on_step で完了。INDEX.md だけ生成。
        base_path = _extract_file_path(query) or ""
        if base_path:
            index_path = await split_write_index(
                base_path=base_path,
                written=split_written,
                state=sess_state,
            )
            if index_path:
                yield sse.step({
                    "type": "task_result",
                    "detail": (
                        f"{len(split_written)} files written; index: {index_path}"
                    ),
                    "status": "done",
                })
            else:
                yield sse.step({
                    "type": "task_result",
                    "detail": f"{len(split_written)} files written (no index)",
                    "status": "done",
                })
        write_result = None
    else:
        write_result = await long_form_write_file(
            query, delivered, sess_state,
        )
    if write_result:
        logger.debug("Long-form file write result: %s", write_result[:120])
        yield sse.step({
            "type": "task_result", "detail": write_result, "status": "done",
        })
    _emit_timing(sess_state, timer, "meta_cognitive", state.tokens_generated, mode=mode)
    ti = make_token_info(
        messages, state.tokens_generated, context_size, instance_name,
    )
    yield sse.token_info(ti)
    yield sse.done()


async def stream_long_form(
    orchestrator: LongFormOrchestrator, query: str, session_id: str,
    mode: str, state: AppState,
    instance_name: str, context_size: int,
    messages: list[ChatMessage],
    existing_content: str = "",
    *,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    file_context_block: str | None = None,
):
    """長文生成の SSE ストリーミング（long_form_* ステップフレーム付き）

    ``output_target`` は coding モード時の出力先 (``"file"`` / ``"editor"`` /
    ``"chat"``)。``"editor"`` の場合はトークンの逐次送出を抑止して終端で
    `sse.editor_code` を送る (`_dispatch_meta_cognitive` 経路の挙動に揃える)。
    """
    async with cancel_scope(session_id):
        t_start = time.monotonic()
        stream_state = _LongFormStreamState()
        outcome_success = False

        # ファイル出力モード判定
        file_output_mode = bool(
            _WRITE_HINT_RE.search(query) and _extract_file_path(query)
        )
        # エディタ経路: トークンは chat へ流さず終端で editor_code 送出に切替える。
        editor_output_mode = (output_target == "editor")
        # token を sse.token としてチャットへ流さない条件を共通化。
        suppress_chat_token_stream = file_output_mode or editor_output_mode
        # 出力モード判定 (EXPAND / SPLIT / CONTINUE)。
        # SPLIT/EXPAND は P2/P3 で挙動分岐するが、P1 では CONTINUE と同じ動作。
        long_form_mode = detect_long_form_mode(
            query,
            has_existing_content=bool(existing_content),
            file_output_mode=file_output_mode,
        )
        if long_form_mode in (LongFormMode.EXPAND, LongFormMode.SPLIT):
            logger.info(
                "Long-form mode detected: %s (file_output_mode=%s, existing=%d chars)",
                long_form_mode.value, file_output_mode, len(existing_content),
            )

        step_queue: list[dict] = []
        on_step = _make_step_queue_callback(step_queue)

        # SPLIT モード用の per-unit 書込み状態
        split_base_path = _extract_file_path(query) or ""
        split_used_paths: set[str] = set()
        split_written: list[dict] = []
        # ユーザー指示文から拡張子を 1 度だけ推論し、SPLIT 全 unit に共通適用。
        split_extension = _infer_output_extension(query)

        def _flush(): return _flush_step_queue_split_aware(
            step_queue,
            long_form_mode=long_form_mode,
            base_path=split_base_path,
            used_paths=split_used_paths,
            written=split_written,
            state=state,
            extension=split_extension,
        )

        async def _flush_with_editor():
            """step をフラッシュしつつ、editor 経路ではユニット完了ごとに
            累積コードを ``editor_code(partial=True)`` で逐次送出する。

            フロント側は同一タブを上書き更新し、生成途中の経過を可視化する。
            終端の確定本文は ``_finalize_long_form_stream`` が partial=False で送る。
            """
            saw_unit_done = editor_output_mode and any(
                s.get("type") == "long_form_unit_done" for s in step_queue
            )
            async for frame in _flush():
                yield frame
            if saw_unit_done and stream_state.full_response.strip():
                # content_type は generate() 開始直後に確定するため、unit 完了が
                # 見えている時点では必ず set 済み。code ならフェンス除去 + python 表示。
                is_code = getattr(orchestrator, "last_content_type", None) == "code"
                _ext, language = _resolve_editor_output_format(query, is_code)
                body = (
                    remove_code_fences(stream_state.full_response)
                    if is_code else stream_state.full_response
                )
                yield sse.editor_code(
                    _normalize_editor_text(body),
                    language=language,
                    filename=None,
                    partial=True,
                )

        try:
            async for frame in _emit_long_form_init_steps(query, file_output_mode):
                yield frame

            if timer:
                timer.start("llm_total_ms")
                timer.start("llm_first_token_ms")

            # orchestrator.generate を async iterator として取り出し、
            # `asyncio.wait` でタイムアウトを挟むことで prefill が長引いて
            # トークンが届かない間も keepalive を送出する (S3)。
            # さらに file_output_mode 時はトークンをフロントに送出しないため、
            # `last_frame_at` で実 yield 時刻を追跡し、keepalive_interval 経過時に
            # keepalive を強制送出してフロントの chunk timeout を防ぐ。
            token_gen = orchestrator.generate(
                instruction=query,
                session_id=session_id,
                mode=mode,
                on_step=on_step,
                existing_content=existing_content,
                long_form_mode=long_form_mode,
                prefetched_rag=prefetched_rag,
                file_context_block=file_context_block,
                # ドキュメント品質ゲートは実ファイル出力時のみ意味を持つ。no-file の
                # チャット表示応答にゲート/本文差し替えを及ぼさないよう file 出力確定
                # 時だけ形式を渡す (非 file は "" → is_document_format=False で非適用)。
                target_format=(
                    _infer_output_extension(query) if file_output_mode else ""
                ),
            )
            aiter = token_gen.__aiter__()
            pending: asyncio.Task[str] | None = None
            last_frame_at = time.monotonic()
            while True:
                if _cancel_flags.get(session_id):
                    if pending is not None and not pending.done():
                        pending.cancel()
                    break
                if pending is None:
                    pending = asyncio.create_task(aiter.__anext__())
                done, _ = await asyncio.wait(
                    {pending}, timeout=DEFAULT_KEEPALIVE_INTERVAL_SEC,
                )
                if pending not in done:
                    # 同一 `pending` を維持したまま keepalive を送出。
                    # 蓄積中の step + editor 逐次更新も流して進行を可視化する。
                    async for frame in _flush_with_editor():
                        yield frame
                    yield sse.keepalive()
                    last_frame_at = time.monotonic()
                    continue
                try:
                    token = pending.result()
                except StopAsyncIteration:
                    pending = None
                    break
                pending = None

                step_frame_yielded = False
                async for frame in _flush_with_editor():
                    yield frame
                    step_frame_yielded = True
                if step_frame_yielded:
                    last_frame_at = time.monotonic()

                if not stream_state.first_token_recorded and timer:
                    timer.stop("llm_first_token_ms")
                    stream_state.first_token_recorded = True

                stream_state.full_response += token
                stream_state.tokens_generated += 1

                # needs_clarification 時は file_output_mode/editor_output_mode に
                # 関わらず必ずチャットへ送出する。抑制したままだと、write-hint 付き
                # query (例: 元バグの再現クエリ) で確認質問がユーザーに一切届かず
                # 無応答に見えてしまう (2026-07-22 発見のトピック混入バグ対策)。
                needs_clarification = (
                    getattr(orchestrator, "last_needed_clarification", False) is True
                )
                if not suppress_chat_token_stream or needs_clarification:
                    yield sse.token(token)
                    last_frame_at = time.monotonic()
                elif time.monotonic() - last_frame_at >= DEFAULT_KEEPALIVE_INTERVAL_SEC:
                    # file_output_mode / editor_output_mode はトークンを送出しないため、
                    # 長時間 unit でフロントの chunk timeout を防ぐ keepalive を送る。
                    yield sse.keepalive()
                    last_frame_at = time.monotonic()

            async for frame in _flush():
                yield frame

            _lf_rag_used, _lf_rag_top1 = rag_signals_from_chunks(prefetched_rag)
            async for frame in _finalize_long_form_stream(
                stream_state, state, orchestrator, query, messages,
                session_id, mode, instance_name, context_size,
                file_output_mode, timer, t_start,
                private=private,
                long_form_mode=long_form_mode,
                split_written=split_written,
                output_target=output_target,
                rag_used=_lf_rag_used,
                rag_top1_score=_lf_rag_top1,
            ):
                yield frame
            outcome_success = True

        except Exception as e:
            logger.error("Long-form stream error: %s", e, exc_info=True)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "meta_cognitive", stream_state.tokens_generated, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            dl = getattr(state, "debug_logger", None)
            if dl is not None:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                dl.log_outcome(
                    kind="chat_response",
                    success=outcome_success,
                    duration_ms=elapsed_ms,
                    tokens_out=stream_state.tokens_generated,
                    quality_signals={
                        "agent_layer": "long_form",
                        "file_output_mode": file_output_mode,
                        "output_target": output_target,
                    },
                )


async def sync_long_form(
    orchestrator: LongFormOrchestrator, query: str, session_id: str,
    mode: str, state: AppState,
    instance_name: str, context_size: int,
    messages: list[ChatMessage],
    existing_content: str = "",
    *,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    file_context_block: str | None = None,
) -> ChatResponse:
    """長文生成の同期応答

    ``output_target == "editor"`` の場合はディスク書込みをスキップし、
    生成本文をそのまま ``ChatResponse.response`` として返す (フロント側で
    エディタペインに流す前提)。
    """
    try:
        full_response = ""
        tokens_generated = 0

        # ストリーミングと同じ意図検出をかける (出力モード一貫性のため)
        file_output_mode = bool(
            _WRITE_HINT_RE.search(query) and _extract_file_path(query)
        )
        long_form_mode = detect_long_form_mode(
            query,
            has_existing_content=bool(existing_content),
            file_output_mode=file_output_mode,
        )

        if timer:
            timer.start("llm_total_ms")
            timer.start("llm_first_token_ms")
        first_token_recorded = False

        async for token in orchestrator.generate(
            instruction=query,
            session_id=session_id,
            mode=mode,
            existing_content=existing_content,
            long_form_mode=long_form_mode,
            prefetched_rag=prefetched_rag,
            file_context_block=file_context_block,
            # ドキュメント品質ゲートは実ファイル出力時のみ適用 (非 file は "")。
            target_format=(
                _infer_output_extension(query) if file_output_mode else ""
            ),
        ):
            if not first_token_recorded and timer:
                timer.stop("llm_first_token_ms")
                first_token_recorded = True
            full_response += token
            tokens_generated += 1

        if timer:
            timer.stop("llm_total_ms")

        metrics = getattr(orchestrator, "last_metrics", {})
        # coding の editor/file 出力は検証・修正済み assembled (last_code_output)
        # を配信する (生ストリームの revise 二重追記を解消)。
        is_code = getattr(orchestrator, "last_content_type", None) == "code"
        code_output = getattr(orchestrator, "last_code_output", None)
        text_output = getattr(orchestrator, "last_text_output", None)
        if is_code and isinstance(code_output, str) and code_output:
            full_response = code_output
        elif (not is_code) and isinstance(text_output, str) and text_output:
            # document_quality モード: 改稿済み確定本文 (revise 二重追記の解消)。
            full_response = text_output
        _lf_rag_used, _lf_rag_top1 = rag_signals_from_chunks(prefetched_rag)
        record_long_form_response(
            state, full_response, messages, session_id,
            query, mode, tokens_generated, metrics,
            private=private,
            rag_used=_lf_rag_used,
            rag_top1_score=_lf_rag_top1,
        )

        if getattr(orchestrator, "last_needed_clarification", False) is True:
            # 主題不明で確認質問を返しただけの応答。full_response は確認質問文
            # なので、そのまま file 書込みしてしまうと確認質問がドキュメントに
            # なってしまう (2026-07-22 発見のトピック混入バグの対策で追加)。
            write_result = None
        elif output_target == "editor":
            write_result = None
            # コード生成時は editor 表示前に markdown コードフェンスを除去。
            if is_code:
                full_response = remove_code_fences(full_response)
        else:
            write_result = await long_form_write_file(
                query, full_response, state,
            )

        _emit_timing(state, timer, "meta_cognitive", tokens_generated, mode=mode)

        response_text = write_result if write_result else full_response

        token_info_dict = make_token_info(
            messages, tokens_generated, context_size, instance_name,
        )
        return ChatResponse(
            response=response_text,
            token_info=TokenInfo(**token_info_dict),
            session_id=session_id,
            agent_layer="meta_cognitive",
        )
    except Exception as e:
        logger.error("Long-form error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# Deliberative ストリーミング / 同期
# ---------------------------------------------------------------------------


@dataclass
class _DeliberativeStreamState:
    """`stream_deliberative` のループ mutable 状態を集約。"""

    tokens_generated: int = 0
    full_response: str = ""
    first_token_recorded: bool = False
    # executable command 学習用 (run_command 実行ターンのみ非 None)
    tool_command: str | None = None
    tool_command_name: str | None = None
    tool_command_success: bool | None = None


def _make_step_queue_callback(
    step_queue: list[dict],
) -> StepCallback:
    """`step_queue` に要素を追加する on_step コールバックを構築する。

    `MAX_STEP_QUEUE_SIZE` を超えたら古い要素を破棄する (BUG-10 対策)。
    """

    def _on_step(step_data: dict) -> None:
        if len(step_queue) >= MAX_STEP_QUEUE_SIZE:
            logger.debug("Step queue overflow, discarding oldest event")
            step_queue.pop(0)
        step_queue.append(step_data)

    return _on_step


async def _drain_deliberative_step_queue(
    step_queue: list[dict],
) -> AsyncIterator[str]:
    """ツール実行で蓄積された step フレームを順次 yield する。"""
    if step_queue:
        logger.debug("Deliberative: sending %d step frames", len(step_queue))
    for step_data in step_queue:
        logger.debug(
            "Deliberative step: type=%s, status=%s, detail=%s",
            step_data.get("type"), step_data.get("status"),
            step_data.get("detail", "")[:120],
        )
        yield sse.step(step_data)
    step_queue.clear()


async def _stream_filtered_token_pipeline(
    token_stream: AsyncIterator[str],
    state: _DeliberativeStreamState,
    session_id: str,
    timer: StageTimer | None,
) -> AsyncIterator[str]:
    """フィルタパイプライン (思考ブロック除去 + 先頭ラベル除去) でトークンを yield する。

    LLM の prefill が長引いて keepalive_interval を超える間トークンが届かない場合は、
    SSE keepalive コメントを送出してフロントエンドの chunk timeout を防ぐ。
    """
    pipeline = StreamPipeline([
        StreamThinkingFilter(),
        HeadBufferFilter(),
    ])
    aiter = token_stream.__aiter__()
    pending: asyncio.Task[str] | None = None
    while True:
        if _cancel_flags.get(session_id):
            if pending is not None and not pending.done():
                pending.cancel()
            break
        if pending is None:
            pending = asyncio.create_task(aiter.__anext__())
        # `asyncio.wait` はタイムアウト時にタスクをキャンセルしないため、
        # keepalive 送出後も同じ `__anext__()` 呼び出しを継続できる。
        done, _ = await asyncio.wait(
            {pending}, timeout=DEFAULT_KEEPALIVE_INTERVAL_SEC,
        )
        if pending not in done:
            yield sse.keepalive()
            continue
        try:
            token = pending.result()
        except StopAsyncIteration:
            pending = None
            break
        pending = None
        state.full_response += token
        # raw トークン単位でカウント (tok/s 指標用)。
        # HeadBufferFilter によるバッファリングで SSE フレーム数と乖離するため、
        # フィルタ出力の有無にかかわらず受信トークンをそのまま数える。
        state.tokens_generated += 1
        filtered = pipeline.process(token)
        if filtered:
            if not state.first_token_recorded and timer:
                timer.stop("llm_first_token_ms")
                state.first_token_recorded = True
            yield sse.token(filtered)

    remaining = pipeline.flush()
    if remaining:
        yield sse.token(remaining)


async def _retry_zero_tokens_deliberative(
    state: _DeliberativeStreamState,
    messages: list[ChatMessage],
    client: LocalClient,
    max_tokens: int | None,
    session_id: str,
) -> AsyncIterator[str]:
    """tokens_generated==0 時に reasoning ループを回避するため再試行する。

    既にキャンセル済みか tokens_generated > 0 なら何も yield しない。
    リトライ後もゼロなら error フレームを yield。
    """
    if state.tokens_generated > 0 or _cancel_flags.get(session_id):
        return
    logger.warning(
        "No content tokens from llama-server, "
        "retrying with fresh request (reasoning-only or stale cache)",
    )
    retry_stream = await client.generate(
        messages, stream=True, id_slot=client.chat_slot,
        max_tokens=max_tokens,
    )
    async for token in retry_stream:
        if _cancel_flags.get(session_id):
            break
        state.full_response += token
        state.tokens_generated += 1
        yield sse.token(token)

    if state.tokens_generated > 0:
        logger.info("Retry succeeded: tokens=%d", state.tokens_generated)
        return
    logger.error("Retry also returned 0 content tokens")
    yield sse.error(
        "No content generated after retry. "
        "The model may be stuck in a reasoning loop."
    )


def _maybe_cache_reactive_response(
    sess_state: AppState,
    query: str,
    response: str,
    *,
    private: bool,
    tool_command: str | None,
    session_id: str,
) -> None:
    """deliberative / 軽量パス応答を ReactiveAgent キャッシュへ蓄積する。

    再訪クエリ (5 分以内・同一文) を reactive 層が即応答できるようにする。
    除外: private ターン / ツール使用応答 (時刻・環境依存で再利用不可) /
    キャンセル済み (部分テキスト) / 空応答。
    """
    agent = getattr(sess_state, "reactive_agent", None)
    if agent is None:
        return
    if private or tool_command is not None:
        return
    if _cancel_flags.get(session_id):
        return
    if not response or not response.strip():
        return
    agent.cache_response(query, response)


async def _finalize_deliberative_stream(
    state: _DeliberativeStreamState,
    sess_state: AppState,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    mode: str,
    instance_name: str,
    context_size: int,
    timer: StageTimer | None,
    t_start: float,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> AsyncIterator[str]:
    """Deliberative ストリーム終端処理: timer 停止 + record + token_info + done."""
    if timer:
        timer.stop("llm_total_ms")
    elapsed = time.monotonic() - t_start
    tok_per_sec = state.tokens_generated / elapsed if elapsed > 0 else 0
    logger.info(
        "Deliberative stream complete: tokens=%d, elapsed=%.2fs, tok/s=%.1f, session=%s",
        state.tokens_generated, elapsed, tok_per_sec, session_id,
    )
    record_response(
        sess_state, state.full_response, messages, session_id,
        query, mode, state.tokens_generated,
        private=private,
        tool_command=state.tool_command,
        tool_command_name=state.tool_command_name,
        tool_command_success=state.tool_command_success,
        tool_routing_success=state.tool_command_success is True,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
    )
    _maybe_cache_reactive_response(
        sess_state, query, state.full_response,
        private=private, tool_command=state.tool_command, session_id=session_id,
    )
    _emit_timing(sess_state, timer, "deliberative", state.tokens_generated, mode=mode)
    ti = make_token_info(
        messages, state.tokens_generated, context_size, instance_name,
    )
    yield sse.token_info(ti)
    yield sse.done()


async def stream_deliberative(
    agent: DeliberativeAgent, query: str, messages: list[ChatMessage],
    client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    *, mode: str = "chat", max_tokens: int | None = None,
    conversation: list[ChatMessage] | None = None,
    generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,
):
    """Deliberative 層の SSE ストリーミング

    DeliberativeAgent が返す生トークンストリームを SSE フレームに変換し、
    キャンセル・0トークンリトライ・token_info・record_response を処理する。
    ツール実行のステップフレームもリアルタイムで送信する。
    StreamPipeline でフィルタチェーン（思考ブロック除去 + 先頭ラベル除去）を適用。
    """
    async with cancel_scope(session_id):
        stream_state = _DeliberativeStreamState()
        t_start = time.monotonic()
        step_queue: list[dict] = []
        on_step = _make_step_queue_callback(step_queue)
        # (asyncio.CancelledError) 時も finally で確実に emit するため、
        # finalize 完了まで到達した場合のみ True にする。
        outcome_success = False
        # genuine error (except Exception) と client cancel
        # (CancelledError/GeneratorExit; except を素通り) を区別する。
        errored = False

        try:
            yield sse.agent_layer("deliberative")

            # Trigger A: LLM 生成開始直後に sleep-time Light を並列実行（§8.1）
            scheduler = state.sleep_scheduler
            if scheduler:
                scheduler.on_llm_start()

            if timer:
                timer.start("llm_total_ms")
                timer.start("llm_first_token_ms")

            # executable command 学習用に command/success を受け取る dict。
            # process() は iterator 返却前に _judge_and_execute_tool を完了
            # するため、await 完了時点で値が確定している。
            tool_capture: dict = {}
            token_stream = await agent.process(
                query=query,
                messages=list(messages),
                llm_client=client,
                mode=mode,
                stream=True,
                conversation=conversation,
                max_tokens=max_tokens,
                on_step=on_step,
                generation_params=generation_params,
                tool_capture=tool_capture,
                tool_judge_task=tool_judge_task,
                session_id=session_id,
            )
            stream_state.tool_command = tool_capture.get("command")
            stream_state.tool_command_name = tool_capture.get("command_name")
            stream_state.tool_command_success = tool_capture.get("success")

            async for frame in _drain_deliberative_step_queue(step_queue):
                yield frame

            async for frame in _stream_filtered_token_pipeline(
                token_stream, stream_state, session_id, timer,
            ):
                yield frame

            async for frame in _retry_zero_tokens_deliberative(
                stream_state, messages, client, max_tokens, session_id,
            ):
                yield frame

            async for frame in _finalize_deliberative_stream(
                stream_state, state, query, messages, session_id,
                mode, instance_name, context_size, timer, t_start,
                private=private,
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
            ):
                yield frame
            outcome_success = True

        except Exception as e:
            errored = True
            logger.error("Deliberative stream error: %s", e)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "deliberative", 0, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            # クライアント切断等でジェネレータが中断された場合、未完了の
            # precomputed tool 判定タスクが残らないよう cancel する (衛生)。
            # 正常経路では process() 内で既に await 済み (done) のため no-op。
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            # CancelledError 経路でも finally に入るため client cancel
            # 検知 (success=False) が確実に行える。
            dl = getattr(state, "debug_logger", None)
            if dl is not None:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                signals: dict = {"agent_layer": "deliberative"}
                if escalated_from:
                    signals["escalated_from"] = escalated_from
                # success=False かつ genuine error でない = client cancel。
                # evolve fitness がユーザーキャンセルを失敗計上しないよう区別する。
                if not outcome_success and not errored:
                    signals["cancelled"] = True
                dl.log_outcome(
                    kind="chat_response",
                    success=outcome_success,
                    duration_ms=elapsed_ms,
                    tokens_out=stream_state.tokens_generated,
                    quality_signals=signals,
                )


async def sync_deliberative(
    agent: DeliberativeAgent, query: str, messages: list[ChatMessage],
    client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    *, mode: str = "chat", max_tokens: int | None = None,
    conversation: list[ChatMessage] | None = None,
    generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,  # noqa: ARG001
) -> ChatResponse:
    """Deliberative 層の非ストリーミング応答 (escalated_from は API 一貫性用、未使用)"""
    logger.debug("Sync deliberative: session=%s, messages=%d", session_id, len(messages))
    try:
        # Trigger A: LLM 生成開始直後に sleep-time Light を並列実行（§8.1）
        scheduler = state.sleep_scheduler
        if scheduler:
            scheduler.on_llm_start()

        if timer:
            timer.start("llm_total_ms")
        resp = await agent.process(
            query=query,
            messages=list(messages),
            llm_client=client,
            mode=mode,
            stream=False,
            conversation=conversation,
            max_tokens=max_tokens,
            generation_params=generation_params,
            tool_judge_task=tool_judge_task,
            session_id=session_id,
        )

        if timer:
            timer.stop("llm_total_ms")

        estimated_tokens = max(1, _estimate_tokens(resp.content))
        record_response(
            state, resp.content, messages, session_id,
            query, mode, estimated_tokens,
            private=private,
            tool_command=resp.tool_command,
            tool_command_name=resp.tool_name if resp.tool_command else None,
            tool_command_success=resp.tool_command_success,
            tool_routing_success=resp.tool_command_success is True,
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
        )
        _maybe_cache_reactive_response(
            state, query, resp.content,
            private=private, tool_command=resp.tool_command, session_id=session_id,
        )

        _emit_timing(state, timer, "deliberative", estimated_tokens, mode=mode)

        token_info_dict = make_token_info(
            messages, estimated_tokens, context_size, instance_name,
        )

        return ChatResponse(
            response=resp.content,
            token_info=TokenInfo(**token_info_dict),
            session_id=session_id,
            agent_layer="deliberative",
        )
    except Exception as e:
        logger.error("Deliberative error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        # process() が判定タスクを await する前に例外で抜けた場合の衛生。
        # 正常経路では process() 内で await 済み (done) のため no-op。
        if tool_judge_task is not None and not tool_judge_task.done():
            tool_judge_task.cancel()


def _apply_generation_params(gen_kwargs: dict, generation_params: "GenerationParams | None") -> None:
    """モード別生成パラメータを client.generate の kwargs へ転写する。"""
    if not generation_params:
        return
    for k in ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty"):
        if k in generation_params:
            gen_kwargs[k] = generation_params[k]


async def stream_reactive_light(
    query: str,
    messages: list[ChatMessage],
    client: LocalClient,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
    *,
    mode: str = "chat",
    max_tokens: int | None = None,
    generation_params: "GenerationParams | None" = None,
    timer: "StageTimer | None" = None,
    private: bool = False,
):
    """Reactive 軽量パス: few-shot/RAG/semmem/tool なしの最小プロンプトで base 1 ターン。

    agent.process を介さず client.generate を直接叩き、deliberative のストリーミング
    ヘルパー (フィルタ / 0トークンリトライ / timing / cache) を再利用する。
    SSE 上の agent_layer は "reactive"。
    """
    async with cancel_scope(session_id):
        stream_state = _DeliberativeStreamState()
        t_start = time.monotonic()
        outcome_success = False
        errored = False
        try:
            yield sse.agent_layer("reactive")

            scheduler = state.sleep_scheduler
            if scheduler:
                scheduler.on_llm_start()

            if timer:
                timer.start("llm_total_ms")
                timer.start("llm_first_token_ms")

            gen_kwargs: dict = {"stream": True, "id_slot": client.chat_slot}
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            _apply_generation_params(gen_kwargs, generation_params)
            token_stream = await client.generate(list(messages), **gen_kwargs)

            async for frame in _stream_filtered_token_pipeline(
                token_stream, stream_state, session_id, timer,
            ):
                yield frame

            async for frame in _retry_zero_tokens_deliberative(
                stream_state, messages, client, max_tokens, session_id,
            ):
                yield frame

            if timer:
                timer.stop("llm_total_ms")
            record_response(
                state, stream_state.full_response, messages, session_id,
                query, mode, stream_state.tokens_generated,
                private=private,
                rag_used=False,
            )
            _maybe_cache_reactive_response(
                state, query, stream_state.full_response,
                private=private, tool_command=None, session_id=session_id,
            )
            _emit_timing(state, timer, "reactive", stream_state.tokens_generated, mode=mode)
            ti = make_token_info(
                messages, stream_state.tokens_generated, context_size, instance_name,
            )
            yield sse.token_info(ti)
            yield sse.done()
            outcome_success = True

        except Exception as e:
            errored = True
            logger.error("Reactive-light stream error: %s", e)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "reactive", 0, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            dl = getattr(state, "debug_logger", None)
            if dl is not None:
                elapsed_ms = (time.monotonic() - t_start) * 1000
                signals: dict = {"agent_layer": "reactive", "reactive_light": True}
                # success=False かつ genuine error でない = client cancel。
                if not outcome_success and not errored:
                    signals["cancelled"] = True
                dl.log_outcome(
                    kind="chat_response",
                    success=outcome_success,
                    duration_ms=elapsed_ms,
                    tokens_out=stream_state.tokens_generated,
                    quality_signals=signals,
                )


async def sync_reactive_light(
    query: str,
    messages: list[ChatMessage],
    client: LocalClient,
    state: AppState,
    session_id: str,
    instance_name: str,
    context_size: int,
    *,
    mode: str = "chat",
    max_tokens: int | None = None,
    generation_params: "GenerationParams | None" = None,
    timer: "StageTimer | None" = None,
    private: bool = False,
) -> ChatResponse:
    """Reactive 軽量パスの非ストリーミング応答。"""
    from backend.free.llm.utils import extract_content

    t_start = time.monotonic()
    try:
        scheduler = state.sleep_scheduler
        if scheduler:
            scheduler.on_llm_start()

        if timer:
            timer.start("llm_total_ms")
        gen_kwargs: dict = {"stream": False, "id_slot": client.chat_slot}
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        _apply_generation_params(gen_kwargs, generation_params)
        data = await client.generate(list(messages), **gen_kwargs)
        content = extract_content(data) if isinstance(data, dict) else str(data)
        if timer:
            timer.stop("llm_total_ms")

        estimated_tokens = max(1, _estimate_tokens(content))
        record_response(
            state, content, messages, session_id,
            query, mode, estimated_tokens,
            private=private,
            rag_used=False,
        )
        _maybe_cache_reactive_response(
            state, query, content,
            private=private, tool_command=None, session_id=session_id,
        )
        _emit_timing(state, timer, "reactive", estimated_tokens, mode=mode)

        dl = getattr(state, "debug_logger", None)
        if dl is not None:
            dl.log_outcome(
                kind="chat_response",
                success=True,
                duration_ms=(time.monotonic() - t_start) * 1000,
                tokens_out=estimated_tokens,
                quality_signals={"agent_layer": "reactive", "reactive_light": True},
            )

        token_info_dict = make_token_info(
            messages, estimated_tokens, context_size, instance_name,
        )
        return ChatResponse(
            response=content,
            token_info=TokenInfo(**token_info_dict),
            session_id=session_id,
            agent_layer="reactive",
        )
    except Exception as e:
        logger.error("Reactive-light sync error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# Staged コーディング (仕様書→コード→テスト) ストリーミング
# ---------------------------------------------------------------------------

_STAGE_LABELS = {"spec": "仕様書", "code": "コーディング", "test": "テスト"}


def _stage_label_for_task(task_id: str) -> str:
    if task_id.startswith("spec"):
        return _STAGE_LABELS["spec"]
    if task_id.startswith("code_"):
        return _STAGE_LABELS["code"]
    if task_id.startswith("test_"):
        return _STAGE_LABELS["test"]
    return "タスク"


def _translate_loop_event(
    evt, total_tasks: int = 0, task_indices: dict[str, int] | None = None,
) -> str | None:
    """LoopEvent を staged 進捗の SSE step フレームへ翻訳する (該当なしは None)。

    2 段階表示:
    - 上位 (工程タスク): ``task_picked`` → ``long_form_unit_start`` を
      ``[i/N] {工程}: {title}`` 形式で出し、フロントの ``parseLongFormProgress``
      が進捗バー化する。``iteration_ended`` → ``long_form_unit_done``。
    - 下位 (工程内サブステップ): ``stage_progress`` → ``task_progress`` step
      (フロントは折りたたみリスト、CLI は逐次表示)。

    ``task_indices`` (呼出側所有の可変 dict) を渡すと、ユニット番号を driver の
    iteration ではなく task_id の初出順で採番する。driver リトライで同一タスクが
    再 pick された場合は同じ番号を再利用し「(再試行)」を付ける (旧実装は
    iteration をそのまま使い ``[4/3]`` のように総数を超えて表示されていた)。
    """
    data = getattr(evt, "data", None) or {}
    tid = str(data.get("task_id", ""))
    label = _stage_label_for_task(tid)

    def _unit_index() -> tuple[int, bool]:
        """(表示番号, 再試行か)。task_indices 未指定時は従来の iteration。"""
        if task_indices is None or not tid:
            return getattr(evt, "iteration", 0) or 0, False
        if tid in task_indices:
            return task_indices[tid], True
        task_indices[tid] = len(task_indices) + 1
        return task_indices[tid], False

    if evt.event == "task_picked":
        title = str(data.get("title", ""))
        idx, is_retry = _unit_index()
        prefix = f"[{idx}/{total_tasks}] " if total_tasks else ""
        suffix = " (再試行)" if is_retry else ""
        return sse.step({
            "type": "long_form_unit_start",
            "detail": f"{prefix}{label}: {title}{suffix}".strip(),
            "status": "running",
        })
    if evt.event == "iteration_ended":
        outcome = data.get("last_outcome") or {}
        status = str(outcome.get("status", ""))
        ok = status == "success"
        if task_indices is not None and tid in task_indices:
            idx = task_indices[tid]
        else:
            idx = getattr(evt, "iteration", 0) or 0
        prefix = f"[{idx}/{total_tasks}] " if total_tasks else ""
        return sse.step({
            "type": "long_form_unit_done",
            "detail": f"{prefix}{label}: {'完了' if ok else (status or '終了')}",
            "status": "done" if ok else "failed",
        })
    if evt.event == "stage_progress":
        detail = str(data.get("detail", "")).strip()
        status = str(data.get("status", "running"))
        if not detail:
            return None
        return sse.step({
            "type": "task_progress",
            "detail": detail,
            "status": status,
        })
    if evt.event == "gate_result":
        ok = bool(data.get("ok"))
        # import スモークゲートは「import 成功＋エントリ静的整合＋OS 互換」までを
        # 静的に検証するもので、プログラムを実行したわけではない。「pass」と書くと
        # 実行検証済みと誤解されるため、起動可能性チェック (静的) と明示する。
        detail = (
            f"{label}: 起動可能性チェック合格 (import/エントリ/整合・静的検証/未実行)"
            if ok else f"{label}: 起動可能性チェック失敗 (起動不能の可能性)"
        )
        return sse.step({
            "type": "task_result",
            "detail": detail,
            "status": "done" if ok else "failed",
        })
    return None


# staged の task グラフは **リクエスト毎の隔離ストア** (workspace 内 .semmem) に持つ。
# 共有 project ストア (state.current_project_id) を使うと ①継続ターンで stale な
# done ファクトが新ターンの spec→code→test 依存ゲートを壊す ②自律ループ
# (state.loop_driver / RalphExecutor) が stage 付きタスクを誤実行する、という不具合に
# なるため、永続プロジェクトストアからは完全に切り離す。
_STAGED_PROJECT_ID = "staged"


async def _staged_write_file(
    state: AppState, logical_path: str, content: str,
) -> str | None:
    """output_target=="file" 時に生成ファイルを registry.write_file で書き出す。"""
    registry = state.tools_registry
    if registry is None or not registry.has("write_file"):
        return None
    try:
        # markdown (SPEC.md 等) は ```mermaid 等の正当なコードフェンスを含むため
        # 除去しない。コードファイルのみ LLM が付ける外側フェンスを剥がす。
        body = content if logical_path.endswith(".md") else remove_code_fences(content)
        return str(await registry.execute(
            "write_file", file_path=logical_path, content=body,
        ))
    except Exception as exc:
        logger.warning("staged write_file failed for %s: %s", logical_path, exc)
        return None


def _staged_postprocess(
    code_map: dict[str, str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """配信前に cross-file import を決定論的に配線し、静的整合 issue を集める。

    test 工程は wall-time で starve され得る (= 工程内スモークゲートが走らない) ため、
    予算非依存のこの終端で必ず検証する。

    - ``wire_imports`` / ``normalize_relative_imports`` は加算的 (不足 import を足し、
      flat 構成で解決不能な相対 import を除くだけ) で機能を削らない = ソースを劣化させない。
    - ``check_coherence`` は重複定義 / どのモジュールにも無い未定義名を検出する (advisory)。

    返り値は (配線済み code_map, issue リスト, 配線で変更したファイル一覧)。issue は
    配信を止めない (advisory)。配線変更一覧は long_form JSONL への可測化に使う。
    """
    from backend.free.generation.import_wirer import wire_imports
    from backend.free.generation.smoke_validator import (
        check_coherence,
        check_cross_module_imports,
        check_entrypoint,
        normalize_relative_imports,
    )

    out = dict(code_map)
    wired_paths: list[str] = []
    py_map = {p: c for p, c in out.items() if p.endswith(".py")}
    if len(py_map) > 1:
        try:
            wired = wire_imports(normalize_relative_imports(py_map))
            for p, c in wired.items():
                if c and c != out.get(p):
                    out[p] = c
                    wired_paths.append(p)
        except Exception as exc:
            logger.warning("staged finalize wire_imports failed: %s", exc)
    final_py = {p: c for p, c in out.items() if p.endswith(".py")}
    issues: list[str] = []
    # 重複定義/未定義名 (coherence) + 起動経路の未定義メソッド参照 (entrypoint) +
    # 生成物間 from-import の名前欠落 (cross_module_imports) を終端でも必ず検査する
    # (工程内スモークが starve された / 外部依存欠落で import スモークが盲目化した場合の保険)。
    for fn in (check_coherence, check_entrypoint, check_cross_module_imports):
        try:
            issues += list(fn(final_py))
        except Exception as exc:
            logger.warning("staged finalize %s failed: %s", fn.__name__, exc)
    return out, issues, sorted(wired_paths)


def _staged_internal_names(ws) -> frozenset[str]:
    """spec が宣言する内部契約名 (幻覚内部 import 判定用、読めなければ空)。

    smoke の「外部依存」分類に渡し、spec の Component / 正準モジュールに由来する
    import 失敗を環境要因 warning へ降格させない (2026-07-07 live: `from game
    import Game` が外部依存扱いになり起動不能コードが偽 success で配信された)。
    """
    try:
        from backend.free.loop.staged.spec_parts import internal_contract_names
        return internal_contract_names(ws.read_spec() or "")
    except Exception as exc:
        logger.debug("staged internal names unavailable: %s", exc)
        return frozenset()


async def _staged_import_smoke(
    code_map: dict[str, str], timeout_sec: float,
    internal_names: frozenset[str] = frozenset(),
) -> list[str]:
    """配信前の code_map を import スモークし error 文字列列を返す (失敗時は空)。

    静的検査 (check_coherence / check_entrypoint) では拾えない cross-file ImportError
    (``from game import GameConfig`` で GameConfig が実在しない等) を終端でも捕捉する。
    ``__main__`` は実行せず、外部依存 (pygame 等) の未インストールは warning に倒れる
    ため error には含まれない (内部契約名 ``internal_names`` に由来する幻覚 import
    は error 側に分類される)。
    """
    py_map = {p: c for p, c in code_map.items() if p.endswith(".py")}
    if not py_map:
        return []
    from backend.free.generation.smoke_validator import run_import_smoke
    try:
        res = await asyncio.to_thread(
            run_import_smoke, py_map, timeout_sec,
            internal_names=internal_names,
        )
    except Exception as exc:
        logger.warning("staged finalize import smoke failed: %s", exc)
        return []
    return [str(e) for e in (getattr(res, "errors", None) or [])]


async def stream_staged_coding(
    *,
    query: str,
    session_id: str,
    state: AppState,
    cfg: dict,
    instance_name: str,
    context_size: int,
    messages: list[ChatMessage],
    output_target: str,
    codegen,
    fallback_factory,
    part_codegen=None,
    timer: StageTimer | None = None,
    private: bool = False,
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL_SEC,
) -> AsyncIterator[str]:
    """専用 LoopDriver をインライン駆動し spec→code→test を実行してストリームする。

    タスクグラフ合成が空 (assist degraded 等) のときは ``fallback_factory`` が返す
    従来 longform ストリームへ委譲する。``part_codegen`` (部分ごと生成向けの別予算
    delegate) が渡されたときのみ部分生成→決定論結合経路を有効化する。
    """
    from uuid import uuid4

    from backend.config import get_path_resolver
    from backend.free.loop.artifact_writer import make_loop_artifact_hook
    from backend.free.loop.driver import LoopDriver, decode_task_fact
    from backend.free.loop.events import LoopEventBus
    from backend.free.loop.staged import (
        WorkspaceManager,
        synthesize_coding_task_graph,
    )
    from backend.free.loop.staged.executor import StagedCodingExecutor
    from backend.free.loop.staged.test_runner import StagedTestRunner
    from backend.free.generation.api_contract import check_api_contract
    from backend.free.generation.smoke_validator import (
        check_coherence,
        check_cross_module_imports,
        check_entrypoint,
        run_entry_smoke,
        run_import_smoke,
    )
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.views.loop import LoopFactView

    t_start = time.monotonic()
    staged_cfg = (cfg.get("coding", {}) or {}).get("staged", {}) or {}
    # editor_route は search_error_wrapper (chat.py) が冒頭で 1 度送出するため、
    # ここでは送らない (二重送出回避)。

    # リクエスト毎に隔離されたワークスペース + SemMem ストアを使う (継続ターンの
    # stale ファクト混入・自律ループとの干渉を構造的に排除する)。
    run_id = uuid4().hex[:12]
    workspace_root = get_path_resolver().resolve_local("coding_workspace_dir")
    ws = WorkspaceManager.open_or_create(
        workspace_root, workspace_id=run_id, session_id=session_id,
        project_id=_STAGED_PROJECT_ID, goal=query, debug_logger=state.debug_logger,
    )
    # 隔離 SemMem ストア (workspace 内 .semmem)。永続 project ストアには触れない。
    staged_store = SemanticFactStore.for_project(ws.root / ".semmem", _STAGED_PROJECT_ID)

    def _staged_view(_pid: str) -> LoopFactView:
        return LoopFactView(stores=[staged_store], writeback_store=staged_store)

    yield sse.step({
        "type": "long_form_plan",
        "detail": "タスクグラフ (仕様書/コード/テスト) を合成中…",
        "status": "running",
    })
    facts = await synthesize_coding_task_graph(
        request=query, project_id=_STAGED_PROJECT_ID,
        assist_client=state.assist_client,
        include_tests=(
            bool(staged_cfg.get("test_stage_enabled", True))
            or bool(staged_cfg.get("smoke_gate_enabled", True))
        ),
        debug_logger=state.debug_logger,
    )
    if not facts:
        logger.info("staged coding: empty task graph; falling back to longform")
        async for frame in fallback_factory():
            yield frame
        return

    for f in facts:
        try:
            staged_store.add_fact(f)
            tv = decode_task_fact(f)
            ws.upsert_task(
                task_id=tv.task_id, title=tv.title, stage=tv.stage or "code",
                status="open", depends_on=tv.depends_on,
            )
        except Exception as exc:
            logger.warning("staged coding: failed to register task: %s", exc)
    yield sse.step({
        "type": "long_form_plan",
        "detail": f"{len(facts)} タスクを生成 (仕様書→コード→テスト)",
        "status": "done",
    })

    test_runner = (
        StagedTestRunner(
            workspace=ws,
            test_timeout_sec=float(staged_cfg.get("test_timeout_sec", 120.0)),
        )
        if staged_cfg.get("test_stage_enabled", True) else None
    )
    event_bus = LoopEventBus()
    smoke_timeout = float(staged_cfg.get("test_timeout_sec", 120.0))
    entry_exec_enabled = bool(staged_cfg.get("entry_smoke_exec_enabled", True))
    entry_exec_timeout = float(staged_cfg.get("entry_smoke_timeout_sec", 10.0))

    def _smoke(files: dict[str, str]) -> object:
        # test 工程の決定論的ゲート。外部依存 (pygame 等) の ModuleNotFound は
        # run_import_smoke 内で warning 扱い (=合格)、ただし stdlib の OS 非互換
        # (Windows の curses 等) は error 化して有界リペア対象にする。import only では
        # 拾えない静的整合 (重複定義 / 未定義名) を check_coherence、起動不能 (エントリが
        # 未定義メソッドを呼ぶ) を check_entrypoint、生成物間 from-import の名前欠落
        # (外部依存欠落で import スモークが盲目化しても拾える) を
        # check_cross_module_imports で error に上乗せ。エントリ有界実行は advisory。
        result = run_import_smoke(
            files, timeout_sec=smoke_timeout,
            internal_names=_staged_internal_names(ws),
        )
        extra_errors: list[str] = []
        for fn in (check_coherence, check_entrypoint, check_cross_module_imports):
            try:
                extra_errors += list(fn(files))
            except Exception as exc:
                logger.debug("staged static gate %s failed: %s", fn.__name__, exc)
        if extra_errors:
            result.errors = list(result.errors) + extra_errors
        if entry_exec_enabled:
            try:
                ent = run_entry_smoke(files, timeout_sec=entry_exec_timeout)
                if getattr(ent, "warnings", None):
                    result.warnings = list(result.warnings) + list(ent.warnings)
            except Exception as exc:
                logger.debug("staged entry exec smoke failed: %s", exc)
        return result

    part_assembler = None
    if part_codegen is not None:
        # 部分結合 (EvorefGen 具象) は有効時のみ lazy import で注入する
        # (smoke_runner / contract_checker と同じ loop→gen 越境回避パターン)。
        from backend.free.generation.part_assembler import assemble_file_parts
        part_assembler = assemble_file_parts

    # spec 宣言契約と生成コードの照合 (EvorefGen 具象) も同パターンで注入。
    from backend.free.generation.spec_conformance import check_spec_conformance

    executor = StagedCodingExecutor(
        workspace=ws, assist_client=state.assist_client, codegen=codegen,
        smoke_runner=(_smoke if staged_cfg.get("smoke_gate_enabled", True) else None),
        test_runner=test_runner,
        contract_checker=check_api_contract,
        conformance_checker=check_spec_conformance,
        max_test_regen_rounds=int(staged_cfg.get("max_test_regen_rounds", 2)),
        max_repair_rounds=int(staged_cfg.get("max_repair_rounds", 2)),
        spec_max_tokens=int(staged_cfg.get("spec_max_tokens", 6144)),
        spec_timeout_sec=float(staged_cfg.get("spec_timeout_sec", 600.0)),
        flowchart_enabled=bool(staged_cfg.get("flowchart_enabled", True)),
        spec_deepen_enabled=bool(staged_cfg.get("spec_deepen_enabled", True)),
        spec_conformance_enabled=bool(
            staged_cfg.get("spec_conformance_enabled", True),
        ),
        max_spec_revision_rounds=int(staged_cfg.get("max_spec_revision_rounds", 1)),
        part_codegen=part_codegen,
        part_assembler=part_assembler,
        part_max_parts=int(staged_cfg.get("part_max_parts", 4)),
        event_bus=event_bus,
        debug_logger=state.debug_logger,
    )
    artifact_hook = make_loop_artifact_hook(_staged_view)
    max_iter = int(staged_cfg.get("max_iterations", 60))
    driver = LoopDriver(
        view_provider=_staged_view,
        executor=executor,
        max_iterations=max_iter,
        max_wall_time_sec=float(staged_cfg.get("total_timeout_sec", 2400.0)),
        # モジュールは互いに独立。1 モジュールの失敗で全体を打ち切らないよう
        # 連続失敗での abort を実質無効化する (max_iterations / 総時間で有界)。
        max_consecutive_failures=max_iter,
        artifact_hook=artifact_hook,
        event_bus=event_bus,
        debug_logger=state.debug_logger,
    )
    driver.start(_STAGED_PROJECT_ID)
    total_tasks = len(facts)
    task_indices: dict[str, int] = {}  # task_id 初出順の表示番号 (リトライで再利用)
    queue = event_bus.subscribe()
    run_task = asyncio.create_task(
        driver.run(_STAGED_PROJECT_ID), name="staged_coding.run",
    )
    last_ka = time.monotonic()
    try:
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if run_task.done() and queue.empty():
                    break
                if time.monotonic() - last_ka >= keepalive_interval:
                    yield sse.keepalive()
                    last_ka = time.monotonic()
                continue
            frame = _translate_loop_event(
                evt, total_tasks=total_tasks, task_indices=task_indices,
            )
            if frame:
                yield frame
                last_ka = time.monotonic()
    finally:
        event_bus.unsubscribe(queue)
        if not run_task.done():
            run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("staged coding run task failed: %s", exc)

    async for frame in _finalize_staged_stream(
        ws=ws, state=state, query=query, messages=messages,
        session_id=session_id, instance_name=instance_name,
        context_size=context_size, output_target=output_target,
        timer=timer, t_start=t_start, private=private,
        smoke_timeout=smoke_timeout,
    ):
        yield frame

    # 隔離ワークスペース (含 .semmem) のクリーンアップ (config で任意)。
    if staged_cfg.get("cleanup_workspace", False):
        ws.cleanup()


async def _finalize_staged_stream(
    *,
    ws,
    state: AppState,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    instance_name: str,
    context_size: int,
    output_target: str,
    timer: StageTimer | None,
    t_start: float,  # noqa: ARG001
    private: bool,
    smoke_timeout: float = 120.0,
) -> AsyncIterator[str]:
    """staged 終端: 生成物を集約し output_target 別に配信 + token_info/done。"""
    if timer:
        timer.stop("llm_total_ms")
    code_map: dict[str, str] = {}
    for wf in ws.list_files(kind="src"):
        c = ws.read_file(wf.logical_path, kind="src")
        if c:
            code_map[wf.logical_path] = c
    # 予算非依存の終端検証: cross-file import を決定論的に配線し (加算的 = 非劣化)、
    # 静的整合性 (重複定義 / 未定義名) を必ずチェックする。test 工程が wall-time で
    # starve されスモークゲートが走らなかった場合でも、配信前にここで担保される。
    code_map, coherence_issues, wired_paths = _staged_postprocess(code_map)

    # 終端の権威的な起動可能性判定: 静的整合 (coherence/entrypoint) に加え、配線後の
    # code_map へ import スモークを上乗せして cross-file ImportError も拾う。test 工程が
    # wall-time で starve された / test_stage_enabled=false でも、非起動コードを success
    # として学習記録しない (= ゲートをブロッキングにする) ための統合シグナル。
    import_errors = await _staged_import_smoke(
        code_map, smoke_timeout, _staged_internal_names(ws),
    )
    runnability_issues = coherence_issues + [
        e for e in import_errors if e not in coherence_issues
    ]

    manifest = ws.read_manifest() or {}
    progress = manifest.get("progress", {}) or {}
    tasks_failed = int(progress.get("tasks_failed", 0) or 0)
    # 生成テストが決定論的に赤のまま終わったモジュール数 (警告付き配信の明示用)。
    # `<task_id>.pytest` エントリは executor._run_advisory_pytest が永続化する。
    pytest_unpassed = sum(
        1 for key, rec in (manifest.get("test_results") or {}).items()
        if key.endswith(".pytest") and not (rec or {}).get("passed")
    )
    # 終端ゲート結果を long_form JSONL に記録し可測化する (develop=investigate/evolve
    # 時のみ出力)。SSE は表示専用で残らないため、配線件数/整合 issue を後から数値で追える。
    if state.debug_logger is not None:
        try:
            state.debug_logger.log_long_form_event({
                "phase": "staged_coherence",
                "strategy": "staged",
                "files": sum(1 for p in code_map if p.endswith(".py")),
                "wired_count": len(wired_paths),
                "wired_files": wired_paths,
                "coherence_issue_count": len(runnability_issues),
                "coherence_issues": runnability_issues[:20],
                "tasks_failed": tasks_failed,
                "pytest_unpassed_count": pytest_unpassed,
            })
        except Exception as exc:
            logger.debug("staged coherence long_form log failed: %s", exc)
    if tasks_failed:
        yield sse.step({
            "type": "task_result",
            "detail": f"⚠ {tasks_failed} 件のタスクが失敗しました (workspace: {ws.root})",
            "status": "failed",
        })
    if pytest_unpassed:
        yield sse.step({
            "type": "task_result",
            "detail": f"⚠ テスト未合格: {pytest_unpassed} モジュール — 生成テストが"
                      f"失敗しています (成果物は配信します)",
            "status": "failed",
        })
    if runnability_issues:
        head = "; ".join(runnability_issues[:5])
        more = (
            f" ほか{len(runnability_issues) - 5}件"
            if len(runnability_issues) > 5 else ""
        )
        yield sse.step({
            "type": "task_result",
            "detail": f"⚠ 起動可能性チェック: {len(runnability_issues)} 件の問題 "
                      f"({head}{more})",
            "status": "failed",
        })

    assembled = "\n\n".join(
        f"# === {p} ===\n{c}" for p, c in code_map.items()
    )
    # metrics は long_form router の success/false_positive 判定に使われる
    # (success = units_completed>0 ∧ validation_errors==0、false_positive = units==0)。
    # units_completed は生成ファイル数のまま (>0 → routing 自体は妥当で false_positive
    # にしない) だが、validation_errors に失敗タスク + 起動可能性 issue を畳み込み、
    # 非起動コードを long_form_success として学習記録しない (ゲートをブロッキング化)。
    staged_metrics = {
        "units_total": len(code_map),
        "units_completed": len(code_map),
        "validation_errors": tasks_failed + len(runnability_issues),
        "content_type": "code",
        "strategy": "staged",
    }
    try:
        record_long_form_response(
            state, assembled, messages, session_id, query, "coding",
            _estimate_tokens(assembled), staged_metrics, private=private,
        )
    except Exception as exc:
        logger.warning("staged: record_long_form_response failed: %s", exc)

    if not code_map:
        yield sse.step({
            "type": "task_result",
            "detail": "コードが生成されませんでした",
            "status": "failed",
        })
    elif output_target == "editor":
        for p, c in code_map.items():
            lang = _editor_language_for_extension(Path(p).suffix) or "python"
            yield sse.editor_code(
                remove_code_fences(c), language=lang, filename=p,
            )
    elif output_target == "chat":
        for p, c in code_map.items():
            lang = _editor_language_for_extension(Path(p).suffix) or ""
            yield sse.token(f"\n\n**{p}**\n```{lang}\n{c}\n```\n")
    else:  # file
        written = [
            res for p, c in code_map.items()
            if (res := await _staged_write_file(state, p, c))
        ]
        detail = (
            f"{len(written)} ファイルを書き込みました"
            if written else f"生成物は workspace にあります: {ws.root}"
        )
        yield sse.step({
            "type": "task_result", "detail": detail, "status": "done",
        })

    spec_md = ws.read_spec()
    if spec_md:
        # 設計フローチャートは UI 表示せず、ファイル成果物としてのみ出力する
        # (ユーザー要望)。チャットへの mermaid 描画フレームは送らない。
        flowchart = ws.read_flowchart()
        # SPEC.md (flowchart は含まない。flowchart.md は下で別ファイルとして届ける)
        # を output_target 別に成果物として届ける。
        if output_target == "editor":
            yield sse.editor_code(spec_md, language="markdown", filename="SPEC.md")
        elif output_target == "file":
            await _staged_write_file(state, "SPEC.md", spec_md)
        yield sse.step({
            "type": "task_result",
            "detail": f"設計仕様: {ws.path('spec.md')}",
            "status": "done",
        })

        # フローチャートを独立した成果物ファイルとしても届ける (ユーザー要望)。
        if flowchart and flowchart.strip():
            fc_doc = f"# 設計フローチャート\n\n```mermaid\n{flowchart.strip()}\n```\n"
            if output_target == "editor":
                yield sse.editor_code(fc_doc, language="markdown", filename="flowchart.md")
            elif output_target == "file":
                await _staged_write_file(state, "flowchart.md", fc_doc)
            yield sse.step({
                "type": "task_result",
                "detail": f"フローチャート: {ws.path('flowchart.md')}",
                "status": "done",
            })

    ti = make_token_info(
        messages, _estimate_tokens(assembled), context_size, instance_name,
    )
    yield sse.token_info(ti)
    yield sse.done()
