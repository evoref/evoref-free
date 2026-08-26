"""Meta-Cognitive のプロンプトと定数

``MetaCognitiveAgent`` 本体と各 mixin が共有する module レベルの定義。
mixin 側から本体を import すると循環するため、共有物はここに集約する。
"""

from __future__ import annotations

import re

from pathlib import Path
from backend.free.core.intent_vocab import EXPLICIT_WINDOWS_PATH_RE

#: パス区切り (ドライブ接頭辞 / スラッシュ / バックスラッシュ) を含むか。
#: 含まない = 裸のファイル名で、どのディレクトリか未確定。
_PATH_SEPARATOR_RE = re.compile(r"[\\/]")

# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """\
You are a task planning assistant. Given a user request, break it down into \
a list of concrete steps.

Output a JSON object with a single field "tasks", which is an array of strings — \
each entry being one task description.

IMPORTANT rules:
- Implement EXACTLY the program/feature the user requested, using the user's own terms. \
NEVER substitute a different or merely "similar" program. IGNORE any unrelated program \
names that appear in the examples below or in prior conversation context — they are \
illustrations of FORMAT only, not of WHAT to build.
- When the user asks to BUILD, CREATE, MAKE, or WRITE a program, script, app, a document, \
or a data file, the task MUST be an action that PRODUCES that deliverable. Do NOT reduce a \
build request to only a "Design/Analyze/Plan the structure" task — that yields just an \
explanation and no usable deliverable. In the examples below, <...> is a placeholder: \
replace it with the user's actual words. NEVER copy a <...> placeholder or an example \
sentence verbatim into your output.
  BAD  (user asked to build): {"tasks": ["Design the game structure and core logic"]}
  GOOD (user asked to build): {"tasks": ["Generate the full <deliverable the user asked for>"]}
- NEVER invent a file path. Only include a file path in a task when the USER explicitly \
gave one. If the user did not specify an output location, describe the task WITHOUT any path.
  BAD  (user gave no path): {"tasks": ["Create e:\\\\app\\\\solution.py with the full implementation"]}
  GOOD (user gave no path): {"tasks": ["Generate the full <deliverable the user asked for>"]}
  GOOD (user said save to e:\\\\app\\\\solution.py): {"tasks": ["Create e:\\\\app\\\\solution.py with the full implementation"]}
- When the user DID give an explicit output path, EVERY create/write task MUST repeat that \
exact path verbatim. Dropping the user's path from the task loses the write destination.
- Creating or rewriting a SINGLE file is always ONE task, not multiple tasks.
  BAD:  {"tasks": ["Create the core logic", "Add feature A", "Add input handling"]}
  GOOD: {"tasks": ["Generate the full <deliverable the user asked for> in a single file"]}
- When the user asks for a DOCUMENT or DATA FILE (Excel/spreadsheet/CSV, Word, \
PowerPoint, a calendar, a table, a report), the deliverable is the FILE CONTENT itself. \
Plan a SINGLE write task that writes the content to the file. Do NOT plan to "generate a \
Python/openpyxl/VBA script" and do NOT plan to "run/execute" a script — the system renders \
the content into the real .xlsx/.docx/.pptx automatically.
  BAD  (user asked for an Excel calendar): {"tasks": ["Generate a Python script that creates the Excel file", "Execute the generated script"]}
  GOOD (user asked for an Excel calendar): {"tasks": ["Write this month's calendar to the user's specified path"]}
- Only split into multiple tasks when genuinely different files or operations are needed.
- Each task should have a SINGLE action type. Do NOT combine "fetch/read" and "write/create" \
in one task.
  BAD:  {"tasks": ["Fetch URL and create script"]}
  GOOD: {"tasks": ["Fetch URL content", "Generate script from fetched content"]}
- For information-only requests (explain, summarize, list, show), use fetch_url or read_file \
as the task — do NOT plan to create files.
  BAD:  {"tasks": ["Create script to scrape website"]}
  GOOD: {"tasks": ["Fetch URL and summarize the content"]}
- Fetching a URL and saving its data to a file is exactly TWO tasks: fetch, then write. \
Do NOT add separate "extract", "generate the file", or "save" steps — extracting the data \
and creating the file both happen inside the single write task.
  BAD:  {"tasks": ["Fetch the URL", "Extract the results", "Generate the Excel file", "Save it to <path>"]}
  GOOD: {"tasks": ["Fetch the URL content", "Write the results to the user's specified path"]}
- Each task should be self-contained and produce a concrete result.

Example: {"tasks": ["Read foo.py", "Generate refactored code", "Run tests"]}
Output the JSON object and nothing else."""

EXECUTE_SYSTEM_PROMPT = """\
You are a coding assistant executing a specific task.
Available tools (call by outputting JSON):
{tool_descriptions}

To call a tool, output ONLY a JSON object: {{"tool": "tool_name", "args": {{...}}}}
Do NOT include any text before or after the JSON.
To provide a final answer (no tool call), output plain text.

Tool argument formats:
- write_file: {{"tool": "write_file", "args": {{"file_path": "path"}}}}
  Do NOT include "content" in args. The system will generate it separately.
  Parent directories are created automatically. Do NOT run mkdir before write_file.
- read_file:  {{"tool": "read_file", "args": {{"file_path": "path"}}}}
  Always include file_path.

Current task: {task}
Context from previous steps:
{context}"""

CONTENT_GENERATION_PROMPT = """\
Generate the requested content below. Output ONLY the content itself, \
no explanations, no markdown fences, no surrounding text. \
Do NOT include the file path as a comment at the top.
"""

# スプレッドシート/表形式の出力先では本文を GFM マークダウン表として生成させる。
# write_file → ContentConverter.from_markdown → XlsxWriter で実セルに展開される。
TABLE_CONTENT_INSTRUCTION = (
    "The target is a spreadsheet/table file. Output ONLY a GitHub-flavored "
    "Markdown table built from the data gathered in the previous steps: a header "
    "row, a `| --- |` separator row, then one row per record. Every row must "
    "start and end with a pipe `|`. No prose, no code fences, no extra text."
)

# .csv は export 変換を通らず raw テキストとして書き込まれるため、GFM 表ではなく
# CSV 行そのものを出力させる (散文/説明文の混入は書込み前検証で棄却される)。
CSV_CONTENT_INSTRUCTION = (
    "The target is a raw CSV file. Output ONLY comma-separated values: one "
    "header row, then one line per record. Use the exact columns the user "
    "asked for. No prose, no Markdown, no code fences, no extra text."
)

# Word/PowerPoint 等のリッチ文書で、取得済みテーブルが無くモデル生成に落ちる時の
# 保険。export Writer が変換できる GFM を出させ、python-pptx/VBScript 等の「文書を
# 作るコード」をテキスト出力する退行を明示的に禁じる。表は強制しない (散文文書も可)。
RICH_DOC_CONTENT_INSTRUCTION = (
    "The target is a Word/PowerPoint document. Output ONLY GitHub-flavored Markdown "
    "(headings with #, paragraphs, bullet lists, and Markdown tables as appropriate). "
    "Do NOT output python-pptx, python-docx, VBScript, openpyxl, or any program code. "
    "No code fences around the whole document."
)

# .md は Markdown そのものが本文フォーマット。既定の CONTENT_GENERATION_PROMPT は
# 「no explanations, no markdown fences」を無条件に指示するため、これを打ち消さないと
# 見出しもコードフェンスも説明文も書けず、拡張子だけ .md の裸テキストになる。
#
# 実インシデント (2026-08-14 ライブ監査 ターン13-14): 「デコレータの説明とコードを
# Markdown にまとめて E:\tmp\retry_decorator.md に保存して」で説明文・見出し・
# ```python フェンスがすべて落ちた生の Python コードが書かれ、「Markdown 形式
# (見出し・説明文・```python フェンス) で保存し直して」と明示し直しても
# 1962 → 1966 バイトの同じ生コードのままだった。
MARKDOWN_CONTENT_INSTRUCTION = (
    "The target is a Markdown document, so Markdown IS the content format: "
    "the 'no explanations / no markdown fences' rule above does NOT apply here. "
    "Write it as the user asked — use `#` headings, explanatory prose, bullet "
    "lists, and fenced code blocks (```python etc.) wherever they belong. "
    "Do not wrap the whole document in a single outer code fence, and do not "
    "add commentary about writing the file."
)

# ユーザークエリ/タスク記述中の明示的な絶対パス (Windows ドライブレター形式)。
# plan 後のパス脱落補完 (_normalize_planned_paths) で使用する。
# 定義は core.intent_vocab が SSOT (agent.feedback が同一定義を持っていた)。
_EXPLICIT_PATH_RE = EXPLICIT_WINDOWS_PATH_RE

# 書込み棄却の理由コード → ユーザー向けの短い説明。
#
# 理由を伏せると、次のターンで「なぜ失敗したのか」と聞かれたモデルが **事実と
# 異なる説明** を作る (実インシデント 2026-08-10 ライブ監査: 同一会話で 2 回
# 書き込みに成功しているのに「私はファイルを直接作成したり書き込んだりする権限を
# 持っていないため、保存に失敗しました」と答えた)。理由はこちらが付けたコードなので、
# 無関係なツール出力を露出させる心配なく添えられる。
_WRITE_REJECTION_REASON_JA: dict[str, str] = {
    "write_report_echo": "生成された本文が完了報告になっていたため",
    "task_log_echo": "生成された本文が進捗ノートになっていたため",
    "tool_call_syntax": "生成された本文がツールコール構文になっていたため",
    "write_script": "生成された本文がファイルを書くスクリプトになっていたため",
    "refusal_or_missing_info": "生成された本文が断り書きになっていたため",
    "prompt_echo": "生成された本文が内部プロンプトの写しになっていたため",
    "instruction_echo": "生成された本文が依頼文の写しになっていたため",
    "literal_wrapped": "生成された本文が依頼文の引用で包まれていたため",
    "path_only": "生成された本文がパスだけだったため",
    "csv_without_rows": "生成された CSV に行が無かったため",
    "edit_without_change": "内容が変わらなかったため",
    "task_restatement": "生成された本文が依頼の言い換えだったため",
}
_WRITE_REJECTION_RE = re.compile(r"invalid output \(([a-z_]+)\)")

# 「書くべき本文は会話にある」ことを示す参照表現。既存ファイルがある上書き
# 依頼でも、この語があるときは既存内容ではなく会話を素材にする
# (_generate_content 内の使用箇所のコメント参照)。
_PRIOR_CONTENT_REFERENCE_RE = re.compile(
    r"先(?:ほど|程)|さきほど|さっき|上記|直前|先の|前の"
    r"|提示した|示した|作成した|出力した"
    r"|\bearlier\b|\babove\b|\bprevious(?:ly)?\b",
    re.IGNORECASE,
)

# aux がパラメータ名をそのまま値として返したときに現れる「パスもどき」。
# ディレクトリ成分も拡張子も持たず、意味のあるファイル名ではない。
_PLACEHOLDER_WRITE_PATH_NAMES: frozenset[str] = frozenset({
    "file_path", "filepath", "file", "filename", "file_name", "fname",
    "path", "output", "output_file", "output_path", "outfile",
    "target", "target_file", "dest", "destination",
})


def _is_placeholder_write_path(file_path: str) -> bool:
    """``file_path`` が引数名プレースホルダそのものか判定する (純粋関数)。"""
    token = file_path.strip().strip("<>{}[]\"'`　 ").lower()
    return token in _PLACEHOLDER_WRITE_PATH_NAMES

# 実データを取得するツール。これらの生結果をタスク横断で蓄積し、後続の
# write タスクが取得済みデータを直接参照できるようにする (転記ハルシネーション防止)。
_DATA_BEARING_TOOLS: frozenset[str] = frozenset({
    "fetch_url", "read_file", "search_code", "search_history", "rag_search",
})

# 拡張子 → 言語識別子 (エディタ出力片のシンタックスハイライト用、best-effort)
_EXT_LANGUAGE_MAP: dict[str, str] = {
    "py": "python", "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "mts": "typescript", "cts": "typescript",
    "tsx": "typescript", "jsx": "javascript",
    "json": "json", "html": "html", "htm": "html", "css": "css",
    "xml": "xml", "yaml": "yaml", "yml": "yaml", "sql": "sql",
    "php": "php", "md": "markdown", "sh": "bash", "rb": "ruby",
    "go": "go", "rs": "rust", "java": "java", "c": "c", "cpp": "cpp", "cs": "csharp",
}

# 言語識別子 → 主要拡張子 (エディタ出力片のファイル名生成用、best-effort)。
# `_EXT_LANGUAGE_MAP` の逆引き。未知言語は呼出側で ``txt`` フォールバック。
_LANGUAGE_EXT_MAP: dict[str, str] = {
    "python": "py", "javascript": "js", "typescript": "ts",
    "json": "json", "html": "html", "css": "css", "xml": "xml",
    "yaml": "yaml", "sql": "sql", "php": "php", "markdown": "md",
    "bash": "sh", "ruby": "rb", "go": "go", "rust": "rs",
    "java": "java", "c": "c", "cpp": "cpp", "csharp": "cs",
}

# クエリ中の言語名キーワード → 言語識別子 (拡張子が無い場合のフォールバック、先頭優先)
_LANGUAGE_KEYWORDS: list[tuple[str, str]] = [
    ("typescript", "typescript"), ("javascript", "javascript"),
    ("python", "python"), ("html", "html"), ("css", "css"),
    ("rust", "rust"), ("golang", "go"), ("java", "java"),
    ("ruby", "ruby"), ("bash", "bash"), ("sql", "sql"),
]

# text_looks_like_code の code indicator が信頼できる言語。これら言語の生成物が
# コードに見えない場合は散文 (例: "設計します..." のみ) の false-success とみなす。
# markdown/html/css/json/yaml/xml/text は indicator 不在でも正当なため除外する。
_CODE_LANGUAGES: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "go", "rust",
    "java", "c", "cpp", "csharp", "ruby", "php", "bash",
})


def read_existing_file(file_path: str) -> str:
    """既存ファイルの内容を読み込む（存在しなければ空文字列）。

    ``_FastPathMixin`` の検証 (staticmethod で ``self`` を持たない) と
    ``_ContentGenerationMixin`` のプロンプト組み立ての双方から使うため、
    クラスの外に置く。
    """
    if not file_path:
        return ""
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def resolve_read_path(
    file_path: str, query: str, conversation: list[dict] | None,
) -> str:
    """read_file の裸のファイル名を、文脈で確定しているディレクトリへ解決する。

    書込み側には解決層が 2 つ (``_resolve_referenced_path`` = 会話に同じ
    basename のフルパスがある / ``_resolve_write_path`` = クエリのパスの
    **ディレクトリ**へ寄せる) あるのに、読取側は片方も配線されていなかった。
    そのため plannerが裸の名前を出すと ``read_file`` がプロセスの CWD を見て
    ``File not found`` になり、供給元を失った後続の書込みが本文を捏造する。

    実インシデント (2026-08-26 ライブ検証): 「E:\tmp に rs_a.txt を作り…」の
    次のターンで「rd_secret.txt の中身を rs_a.txt の末尾に追記してください。」
    と依頼すると ``read_file({'file_path': 'rd_secret.txt'})`` が失敗し、
    実ファイルの中身が ``SECRET-4417`` であるにもかかわらず
    ``ALPHA\nSECRET_CONTENT`` が書き込まれた。

    解決は **実在するときだけ** 行う。候補ディレクトリ配下に同名ファイルが
    無ければ元の値をそのまま返す (推測でパスを埋めない)。
    """
    if not file_path or _PATH_SEPARATOR_RE.search(file_path):
        return file_path

    name = Path(file_path).name

    # 1. 会話に同じ basename のフルパスがある (既存の参照解決と同じ強さ)
    from backend.free.agent.tool_call_judge import _resolve_referenced_path

    same = _resolve_referenced_path(file_path, conversation)
    if same:
        return same

    # 2. 文脈で確定しているディレクトリ配下に実在するか。
    #    現在のクエリを最優先し、次に会話を新しい順に見る。
    texts = [query or ""]
    for msg in reversed(list(conversation or [])):
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            texts.append(msg["content"])

    from backend.free.agent.tool_call_judge import _extract_file_path

    for text in texts:
        found = _extract_file_path(text)
        if not found or not _PATH_SEPARATOR_RE.search(found):
            continue
        base = Path(found)
        directory = base if base.is_dir() else base.parent
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return file_path
