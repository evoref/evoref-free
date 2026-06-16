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
from backend.free.api.chat.chat_types import GenerationParams, StepCallback
from backend.free.api.schemas import ChatResponse, TokenInfo
from backend.free.agent.deliberative import DeliberativeAgent
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.core.sse import SSEFrameBuilder
from backend.free.core.stream_filter import (
    HeadBufferFilter, StreamThinkingFilter,
)
from backend.free.core.stream_pipeline import StreamPipeline
from backend.free.llm.local_client import LocalClient
from backend.free.llm.editor_filename import derive_editor_filename_stem
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
    """ユーザー指示文から出力ファイルの拡張子を推論する。

    現状は ``.md`` / ``.txt`` の 2 値のみ扱い、判定できなければ ``default`` を返す。
    SPLIT モードの個別 unit と CONTINUE モードの自動命名で共通利用する。

    Args:
        query: ユーザー指示文
        default: 推論できなかった場合に返す既定拡張子 (先頭ドット必須)

    Returns:
        ``.md`` / ``.txt`` 等の拡張子文字列 (先頭ドット付き)。
    """
    if _MD_EXT_HINT_RE.search(query):
        return ".md"
    return default


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
    if _editor_language_for_extension(Path(file_path).suffix) != "markdown":
        content = remove_code_fences(content)

    # 生成テキストのクリーニング（見出し行除去・空行圧縮）
    content = clean_generated_text(content)

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
    conversation: list[dict],
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
        except Exception as e:  # noqa: BLE001
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
    messages: list[dict],
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
    conversation: list[dict], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[dict], mode: str,
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
                dl.log_outcome(
                    kind="chat_response",
                    success=outcome_success,
                    duration_ms=elapsed_ms,
                    tokens_out=tokens_out,
                    quality_signals={"agent_layer": "meta_cognitive"},
                )


async def sync_meta_cognitive(
    agent: MetaCognitiveAgent, query: str, system_prompt: str,
    conversation: list[dict], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[dict], mode: str,
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


async def _finalize_long_form_stream(
    state: _LongFormStreamState,
    sess_state: AppState,
    orchestrator: LongFormOrchestrator,
    query: str,
    messages: list[dict],
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
    delivered = (
        code_output
        if (is_code and isinstance(code_output, str) and code_output)
        else state.full_response
    )
    record_long_form_response(
        sess_state, delivered, messages, session_id,
        query, mode, state.tokens_generated, metrics,
        private=private,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
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

    if output_target == "editor":
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
    messages: list[dict],
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

                if not suppress_chat_token_stream:
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
    messages: list[dict],
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
        if is_code and isinstance(code_output, str) and code_output:
            full_response = code_output
        _lf_rag_used, _lf_rag_top1 = rag_signals_from_chunks(prefetched_rag)
        record_long_form_response(
            state, full_response, messages, session_id,
            query, mode, tokens_generated, metrics,
            private=private,
            rag_used=_lf_rag_used,
            rag_top1_score=_lf_rag_top1,
        )

        if output_target == "editor":
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
    messages: list[dict],
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
    messages: list[dict],
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
    agent: DeliberativeAgent, query: str, messages: list[dict],
    client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    *, mode: str = "chat", max_tokens: int | None = None,
    conversation: list[dict] | None = None,
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
    agent: DeliberativeAgent, query: str, messages: list[dict],
    client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    *, mode: str = "chat", max_tokens: int | None = None,
    conversation: list[dict] | None = None,
    generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_judge_task: "asyncio.Task | None" = None,
    escalated_from: str | None = None,
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
    messages: list[dict],
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
    messages: list[dict],
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
