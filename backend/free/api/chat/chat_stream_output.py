"""生成物のファイル / エディタ出力

出力先パスの解決、エディタ表示用の整形、SPLIT モードのユニット書込みなど、
「生成したテキストをどこへどう書くか」だけを担う層。
"""

from __future__ import annotations

import re

from pathlib import Path
from backend.app_state import AppState
from backend.free.agent.meta_cognitive_utils import is_tool_error
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.agent.output_format import infer_output_extension
from backend.free.generation.document_gate import is_document_format
from backend.free.generation.validators import remove_code_fences
from backend.utils import utc_compact_stamp

from backend.free.api.chat.chat_stream_common import (
    logger,
)


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
