"""ファイル系ツール — 読み取り / 書き込み / 検索 / 一覧 / 差分適用

``write_file`` はリッチ文書 (OOXML 等) の生成と、書込み前のパス脱出検査まで
含む。プレーンテキストのまま .docx を書くと壊れたファイルになるため。
"""

from __future__ import annotations

import calendar
import os
import re
import subprocess

from pathlib import Path
from backend.log_config import get_logger
from backend.free.constants import READ_FILE_META_PREFIX

logger = get_logger("agent.tools.builtin")


# read_file / search_code が読み込むファイルの上限サイズ。models/ (GGUF, 数十GB) 等の
# 巨大バイナリを誤って text として全文読み込みすると、CPython の GIL が長時間手放されず
# 別スレッド実行下でもイベントループを事実上ブロックする (実インシデントで確認済み)。
# 通常のソースファイルはこれより十分小さい。
_TOOL_MAX_FILE_READ_BYTES = 2_000_000

#: read_file のデコード候補 (先頭から順に厳格デコードを試す)。BOM 付きは
#: utf-8-sig、Windows 既定のメモ帳 / 旧ツール出力は cp932。
_READ_FILE_ENCODINGS = ("utf-8", "utf-8-sig", "cp932")


def _decode_text_file(raw: bytes) -> tuple[str, str]:
    """ファイルのバイト列をデコードし ``(本文, 使ったエンコーディング)`` を返す。

    改行は ``Path.read_text`` (universal newlines) と同じく ``\\n`` へ正規化する。
    全候補で失敗した場合は utf-8 の置換デコードに落ちる。
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        candidates: tuple[str, ...] = ("utf-8-sig",)
    else:
        candidates = tuple(e for e in _READ_FILE_ENCODINGS if e != "utf-8-sig")
    text: str | None = None
    used = "utf-8"
    for enc in candidates:
        try:
            text = raw.decode(enc)
            used = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
        used = "utf-8 (replace)"
    return text.replace("\r\n", "\n").replace("\r", "\n"), used


def read_file(
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """ファイルの内容を読み込む

    先頭に ``[file: ... | lines: N | chars: M]`` のメタ行を付ける。行数・文字数は
    LLM が本文から数えても正確にならない (実測 2026-08-05: 文字数を問われて
    「確認できません」と回答放棄した) ため、決定論的に算出して渡す。

    ``start_line`` / ``end_line`` (1 始まり・両端含む) を指定すると該当行だけを
    返す。「最初の 3 行」のような範囲指定は本文全体を渡すとモデルが守らず
    ほぼ全文を出力してしまうため、ツール側で切り出す。
    """
    traversal_error = _check_path_traversal(file_path)
    if traversal_error:
        return traversal_error
    p = Path(file_path)
    if not p.exists():
        return f"Error: File not found: {file_path}"
    if not p.is_file():
        return f"Error: Not a file: {file_path}"
    try:
        size = p.stat().st_size
        if size > _TOOL_MAX_FILE_READ_BYTES:
            return (
                f"Error: file too large to read ({size} bytes, "
                f"limit {_TOOL_MAX_FILE_READ_BYTES}): {file_path}"
            )
        content, used_encoding = _decode_text_file(p.read_bytes())
        lines = content.splitlines()
        header = (
            f"{READ_FILE_META_PREFIX}{file_path}"
            f" | lines: {len(lines)} | chars: {len(content)}"
        )
        if used_encoding != "utf-8":
            header += f" | encoding: {used_encoding}"

        if start_line is not None or end_line is not None:
            start = max(1, int(start_line or 1))
            end = int(end_line) if end_line is not None else len(lines)
            end = min(len(lines), max(start, end))
            content = "\n".join(lines[start - 1:end])
            header += f" | showing lines {start}-{end}"

        header += "]"
        # 大きすぎるファイルは切り詰め (メタ行は残す)
        if len(content) > 50000:
            content = content[:50000] + "\n\n... (truncated, file too large)"
        return f"{header}\n{content}"
    except Exception as e:
        return f"Error: {e}"


def verify_syntax(file_path: str) -> str:
    """Python ファイルの構文を検証する（__pycache__ を生成しない）"""
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if not p.is_file():
        return f"Error: not a file: {file_path}"
    if p.suffix != ".py":
        return f"Error: not a Python file: {file_path}"
    try:
        import py_compile
        import os
        import tempfile
        # cfile を一時ファイルに指定し、__pycache__ の生成を防止
        fd, tmp_pyc = tempfile.mkstemp(suffix=".pyc")
        os.close(fd)
        try:
            py_compile.compile(str(p), cfile=tmp_pyc, doraise=True)
        finally:
            # 一時 .pyc ファイルを削除
            try:
                os.unlink(tmp_pyc)
            except OSError:
                pass
        return f"Syntax OK: {file_path}"
    except py_compile.PyCompileError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Error: {e}"


# リッチ(バイナリ)文書形式: プレーンテキスト書込みでは壊れたファイルになるため、
# backend.export レジストリ経由で実体 (OOXML / OPF 等) を生成する。対応ライブラリ
# 未導入で Writer が利用不可な場合に「リッチ形式の意図」を検知して明示エラーに
# するための最小定数。
_EXPORT_DOC_EXTS = frozenset(
    {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"},
)


def _block_has_renderable_content(block) -> bool:
    """ContentBlock が文書に描画される実体を持つか (空行/水平線は False)。"""
    if block.type == "table":
        return bool(block.rows)
    if block.type == "list":
        return bool(block.items)
    if block.type == "hr":
        return False
    return bool((block.content or "").strip())


def _reject_unrenderable_rich_content(ext: str, content: str, export_content) -> str | None:
    """空 / コード文字列だけのリッチ文書を success と誤報せず明示エラーにする。

    取得テーブルを使う決定論経路が外れて LLM 生成に落ちた時、モデルが空文字列や
    「文書を作るコード」をそのまま返すと、見た目は valid だが中身の無い .pptx/.docx を
    "Written N bytes" として成功扱いしてしまう (報告された .txt 不具合の .pptx 版)。
    描画可能なブロックが皆無、または table が無くコードに見える本文を弾く。
    問題なければ ``None`` を返す。
    """
    blocks = export_content.blocks
    if not any(_block_has_renderable_content(b) for b in blocks):
        return (
            f"Error: '{ext}' output produced no document content. "
            "Provide the document body as Markdown (headings, paragraphs, tables)."
        )
    has_table = any(b.type == "table" and b.rows for b in blocks)
    if not has_table:
        from backend.free.agent.meta_cognitive_utils import text_looks_like_code
        if text_looks_like_code(content):
            return (
                f"Error: '{ext}' output looks like program code, not document "
                "content. Provide the document body as Markdown (a heading and a "
                "Markdown table), not a script that generates the file."
            )
    return None


# 月ごとの実日数上限 (年に依存しない固定値)。2 月は閏年誤検知を避けるため
# 寛容側に 29 まで許容する (2026 年は平年で 28 日までだが、他年での誤検知を防ぐ)。
_MONTH_MAX_DAYS = {m: calendar.mdays[m] for m in range(1, 13)}
_MONTH_MAX_DAYS[2] = 29


def _check_calendar_table(export_content) -> str | None:
    """月別カレンダー表に実在しない日付が含まれていないか検証する。

    「1月から12月」等の月単位カレンダーは各行の先頭セルが月番号 (1-12)、
    後続セルが日番号という構造で生成されがちだが、ローカル小型モデルは
    「何月は何日まで」という長距離の正確な記憶・計算が不得手で、31日までしか
    無い月に32以上、2月に30日以上の値が混入することがある (実運用で発生:
    2026年2月の行に31日まで埋まっていた)。値の新規作成はせず、実在しない
    日付が見つかった場合のみ再生成を促すエラーを返す。
    """
    for block in export_content.blocks:
        if block.type != "table" or len(block.rows) < 3:
            continue
        month_rows: list[tuple[int, list[str]]] = []
        for row in block.rows[1:]:
            if not row:
                continue
            try:
                month = int(str(row[0]).strip())
            except (ValueError, AttributeError):
                continue
            if 1 <= month <= 12:
                month_rows.append((month, row))
        # 月番号が重複なく昇順に 2 行以上並ぶ表のみ「月別カレンダー」とみなす
        # (無関係な数値表の誤検知を避けるための構造的シグナル)。
        months = [m for m, _ in month_rows]
        if len(months) < 2 or months != sorted(set(months)):
            continue
        bad_months: list[str] = []
        for month, row in month_rows:
            max_day = _MONTH_MAX_DAYS[month]
            for cell in row[1:]:
                try:
                    day = int(str(cell).strip())
                except (ValueError, AttributeError):
                    continue
                # 1-31 の範囲内の値のみ「日番号」候補として扱う (無関係な数値
                # 表 (売上高等) の誤検知を避けるため)。
                if 1 <= day <= 31 and day > max_day:
                    bad_months.append(f"{month}月 ({day}日 > {max_day}日まで)")
                    break
        if bad_months:
            return (
                "Error: calendar table contains dates that do not exist: "
                f"{', '.join(bad_months)}. Each month's day column must stop at "
                "its actual last day (February has at most 29 days; April, June, "
                "September, November have at most 30). Rewrite the table with "
                "the correct number of days for each month."
            )
    return None


def _write_rich_document(p: Path, content: str) -> str:
    """``.docx`` 等のリッチ形式を export フレームワーク経由で実ファイル化する。

    文書 Writer (python-docx 等) がグローバルレジストリに登録・利用可能なら
    OOXML 等を生成する。ライブラリ未導入で利用不可の場合は壊れたファイルを
    書かず、必要パッケージを案内する明示エラーを返す (fail clearly)。
    """
    from backend.export import get_writer_registry
    from backend.export.content_converter import ContentConverter

    ext = p.suffix.lower()
    registry = get_writer_registry()
    writer = registry.get_writer(ext)
    if writer is None or not writer.is_available():
        requires = ", ".join(writer.requires) if writer else "python-docx"
        return (
            f"Error: '{ext}' output requires {requires}. "
            "Install the package, or use a text format such as .txt / .md."
        )
    try:
        # raw_markdown のみでは XLSX/CSV writer が表データを抽出できない
        # (no_table_data になる)。ContentConverter で markdown を構造化ブロック
        # (table 等) に変換して渡す。from_markdown は raw_markdown も保持するため
        # docx/odt 等のプローズ系 writer は従来どおり動作する。
        export_content = ContentConverter.from_markdown(content)
        guard_error = _reject_unrenderable_rich_content(ext, content, export_content)
        if guard_error:
            return guard_error
        calendar_error = _check_calendar_table(export_content)
        if calendar_error:
            return calendar_error
        result = registry.write(export_content, p)
        return f"Written {result.size_bytes} bytes to {p}"
    except Exception as e:
        return f"Error: {e}"


def _check_path_traversal(file_path: str) -> str | None:
    """``..`` セグメントを含む相対パス指定を拒否する。

    write_file / read_file 共通のガード。LLM が意図しない (幻覚・プロンプト
    インジェクション由来の) ``..`` を含むパスを発行し、呼び出し元が想定した
    範囲外のファイルを書き換え/読み出ししてしまう事故を防ぐ。本ツールには
    固定のワークスペースルート概念が無い (ユーザーが任意の絶対パスを指定して
    書き込む用途を意図的にサポートしている) ため、絶対パス自体は許可し
    ``..`` 相対脱出のみを検査対象とする。
    """
    if not file_path:
        return None
    try:
        normalized = file_path.replace("\\", "/")
        if ".." in normalized.split("/"):
            logger.warning("Path traversal detected in tool args: %s", file_path)
            return f"Error: path traversal not allowed: {file_path}"
    except (AttributeError, TypeError):
        pass
    return None


def write_file(file_path: str, content: str) -> str:
    """ファイルに書き込む

    ``file_path`` が既存ディレクトリを指している場合はエラーを返す。
    以前は配下に ``output_<UTC>.txt`` を自動付与していたが、read/verify 指示が
    ツール判定で write と誤認された際に、配下へ生成内容を捏造投棄する原因と
    なっていたため廃止した。長文生成経路は書込み前に
    ``_resolve_long_form_target_path`` でディレクトリ→ファイル名を解決済みのため
    影響しない。ディレクトリ配下の点検は list_directory / read_file を使う。

    ``.docx`` 等のリッチ文書形式 (``_EXPORT_DOC_EXTS``) は export フレームワーク
    経由で実体を生成する。それ以外は UTF-8 テキストとして書き込む。
    """
    traversal_error = _check_path_traversal(file_path)
    if traversal_error:
        return traversal_error
    p = Path(file_path)
    if p.is_dir():
        return (
            f"Error: '{file_path}' is a directory, not a file. "
            "Provide a file path, or use list_directory/read_file to inspect it."
        )
    if p.suffix.lower() in _EXPORT_DOC_EXTS:
        return _write_rich_document(p, content)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        # 書込み後のファイルサイズを stat で読む。len(content) は文字数なので
        # UTF-8 マルチバイト (日本語等) で乖離し、encode しても Windows の
        # 改行変換 (write_text は newline=None = LF -> CRLF) の分だけ足りない
        # (2026-07-26 実測: 報告 274 bytes に対し実ファイル 284 bytes、差は
        # 改行 10 個分)。"bytes" と明記する以上、実体と一致させる。
        return f"Written {p.stat().st_size} bytes to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def search_code(pattern: str, directory: str = ".", max_results: int = 20) -> str:
    """コードを正規表現パターンで検索する"""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex: {e}"

    results: list[str] = []
    base = Path(directory)
    if not base.exists():
        return f"Error: Directory not found: {directory}"

    for root, dirs, files in os.walk(base):
        # 隠しディレクトリ・一般的な除外 + モデル/ローカルデータ (数十GB級バイナリ/
        # 大量の実行時データを含み、ソースコードではない) をスキップ
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", ".git", "models", "local"}
        ]
        for fname in files:
            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size > _TOOL_MAX_FILE_READ_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{fpath}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (limited to {max_results} results)"
            except (OSError, UnicodeDecodeError):
                continue

    if not results:
        return f"No matches found for pattern: {pattern}"
    return "\n".join(results)


#: ツリー出力の行数上限。超過分は黙って捨てず、省略行数を明示する。
#: 黙って切ると、受け取ったモデルは手元の部分木を全体として提示する
#: (実インシデント 2026-08-01 ライブ監査 再検証: 6070 文字のツリーが途中で
#: 切れ、入れ子の項目が直下の項目として並べられた)。
_LIST_DIRECTORY_MAX_LINES = 200


def list_directory(directory: str = ".", max_depth: int = 3) -> str:
    """ディレクトリ構造を表示する

    ``max_depth=1`` で直下のみ。「直下を一覧して」型の依頼はこれで満たせる。
    """
    base = Path(directory)
    if not base.exists():
        return f"Error: Directory not found: {directory}"

    lines: list[str] = []
    _walk_tree(base, lines, prefix="", depth=0, max_depth=max_depth)

    if not lines:
        return "(empty directory)"
    if len(lines) > _LIST_DIRECTORY_MAX_LINES:
        omitted = len(lines) - _LIST_DIRECTORY_MAX_LINES
        lines = [
            *lines[:_LIST_DIRECTORY_MAX_LINES],
            f"... ({omitted} more entries omitted) ...",
        ]
    return "\n".join(lines)


def _walk_tree(path: Path, lines: list[str], prefix: str, depth: int, max_depth: int) -> None:
    """ディレクトリツリーを再帰的に構築

    ``max_depth`` は **表示する階層数**。``depth`` は 0 起点なので打ち切りは
    ``>=`` で行う。``>`` だと 1 階層余分に降り、``max_depth=1`` (直下だけ) が
    2 階層返る (実インシデント 2026-08-01 再検証: 直下一覧の依頼に
    ``max_depth=1`` が正しく渡ったのに孫階層まで出力され、受け取ったモデルが
    入れ子の項目を直下として並べた)。
    """
    if depth >= max_depth:
        return
    try:
        # **ファイルを先、ディレクトリを後** に並べる。逆にするとディレクトリを
        # 深さ優先で降りきってから自分自身のファイルへ戻るため、行数上限で
        # 打ち切られたときに真っ先に消えるのが「そのディレクトリ直下の
        # ファイル」になる (実インシデント 2026-09-03 ライブ監査 T05#4:
        # リポジトリ直下の ``list_directory(.)`` が 201 行中 **直下ファイル 0 件**
        # で打ち切られ、「README ファイルは存在しますか？」に
        # 「この範囲では確認できません」と答えた。README.md は実在する)。
        # この順序なら、各階層で「自分の持ち物」が部分木の展開より先に出る。
        entries = sorted(path.iterdir(), key=lambda p: (p.is_dir(), p.name))
    except PermissionError:
        lines.append(f"{prefix}└── (permission denied)")
        return
    skip_dirs = {".git", "node_modules", "__pycache__", ".svelte-kit", "dist"}
    for i, entry in enumerate(entries):
        if entry.name.startswith(".") and entry.is_dir():
            continue
        if entry.name in skip_dirs:
            continue
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _walk_tree(entry, lines, prefix + extension, depth + 1, max_depth)


def _check_diff_header_paths(p: Path, diff_text: str) -> str | None:
    """diff ヘッダ (``---`` / ``+++``) のパスが ``p`` と同じディレクトリを指すか検査する。

    ``patch -p0`` は ``cwd=p.parent`` でヘッダのパスをそのまま使うため、
    ``../etc/x`` や絶対パスを書いたヘッダは ``file_path`` の外を書き換える。
    """
    parent = p.resolve().parent
    for line in diff_text.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        target = line[4:].split("\t", 1)[0].strip()
        if not target or target == "/dev/null":
            continue
        if ".." in target.replace("\\", "/").split("/"):
            return f"Error: path traversal not allowed in diff header: {target}"
        try:
            resolved = (p.parent / target).resolve()
        except (OSError, ValueError):
            return f"Error: invalid path in diff header: {target}"
        if resolved.parent != parent:
            return (
                f"Error: diff header path is outside the target directory: {target}"
            )
    return None


def apply_diff(file_path: str, diff_text: str) -> str:
    """unified diff をファイルに適用する

    失敗時は原文を保持し、エラーメッセージを返す。
    """
    traversal_error = _check_path_traversal(file_path)
    if traversal_error:
        return traversal_error
    p = Path(file_path)
    if not p.exists():
        return f"Error: File not found: {file_path}"
    if not p.is_file():
        return f"Error: Not a file: {file_path}"
    header_error = _check_diff_header_paths(p, diff_text)
    if header_error:
        return header_error

    try:
        original = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return f"Error: {e}"

    try:
        result = subprocess.run(
            ["patch", "-p0", "--no-backup-if-mismatch"],
            input=diff_text,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=p.parent,
        )
        if result.returncode == 0:
            return f"Diff applied successfully to {file_path}"
        # 適用失敗: 原文を復元
        p.write_text(original, encoding="utf-8")
        return f"Error: Diff failed: {result.stderr.strip()}"
    except FileNotFoundError:
        return "Error: 'patch' command not found. Install GNU patch."
    except subprocess.TimeoutExpired:
        p.write_text(original, encoding="utf-8")
        return "Error: Diff application timed out"
