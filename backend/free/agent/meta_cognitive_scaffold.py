"""プロンプト scaffold / タスクログの除去と検出

生成物へ混入した「エージェント自身の進行ノート」やプロンプトの骨組みを
落とす。few-shot に混ざると同じ形が再生産されるため、検出は生成側と
few-shot 採否の双方で使う。
"""

from __future__ import annotations

import re

from backend.free.core.prompt_blocks import CURRENT_DATETIME_LABEL
from backend.free.core.text_quality import (
    _TASK_LOG_FRAGMENT_RE as _CORE_TASK_LOG_FRAGMENT_RE,
    looks_like_task_log_residue as _looks_like_task_log_residue,
)

from backend.free.agent.meta_cognitive_write_rescue import (
    _LEAD_IN_LINE_RE,
)


# ---------------------------------------------------------------------------
# 生成コンテンツのエコー検出 (タスクログ / プロンプト scaffold)
# ---------------------------------------------------------------------------

#: エージェントの進捗ノート行 (最終応答フォーマット由来)。few-shot に混入した
#: 「- [done] ... / Written N bytes to ...」形式を小型モデルが成果物として
#: 復唱する退化 (#incident 2026-07-15: 29 ファイル中 10 件が本文なしのログ 1 行)
#: を書込み前に検出するためのパターン。
_TASK_LOG_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\[(?:done|failed|skipped)\]\s"
    r"|^\s*Written\s+\d+\s+bytes\s+to\s+\S"
    r"|^\s*Content of `[^`]+`\s*:?\s*$",
)

#: 生成プロンプトの内部 scaffold マーカー。成果物に現れたら「プロンプトの
#: エコー」であり本文生成に失敗している (例: relocation_notice.txt に
#: 「[現在日時 (UTC基準)] ...」とタスク英文がそのまま書き込まれた事例)。
#: 生成プロンプトが本文へ差し込むブロック見出し。**emit 側と検出側で同じ
#: 定数を使う**こと。片方だけに文字列を直書きすると、新しいブロックを足した
#: ときに検出漏れが生まれる (実インシデント 2026-07-29 ライブ監査: 翻訳結果の
#: 保存で ``## 取得済みデータ (前ステップで取得した実データ) / 以下のデータ
#: のみを根拠に…`` がそのままファイルへ書き込まれた。既存ファイル内容ブロック
#: だけがマーカー登録されており、取得済みデータブロックは素通りだった)。
EXISTING_CONTENT_BLOCK_HEADING = "## 既存ファイル内容"
FETCHED_DATA_BLOCK_HEADING = "## 取得済みデータ (前ステップで取得した実データ)"
FETCHED_DATA_BLOCK_NOTE = (
    "以下のデータのみを根拠に出力を生成し、データに無い情報は創作しないこと。"
)

_PROMPT_SCAFFOLD_MARKERS: tuple[str, ...] = (
    # ラベルは emit 側 (core.prompt_blocks) を唯一の出所とする。ここへ literal で
    # 書き写すと、出す側の文言を変えた瞬間に検出が黙って空振りする。
    CURRENT_DATETIME_LABEL,
    f"{EXISTING_CONTENT_BLOCK_HEADING} (",
    FETCHED_DATA_BLOCK_HEADING,
    FETCHED_DATA_BLOCK_NOTE,
    "【元のコード】",
    "【修正指示】",
)

#: user_prompt の内部構造「タスク: <英語タスク文>」のエコー検出。planner の
#: タスク文は英語動詞で始まるため、日本語文書中の一般語「タスク:」とは
#: 区別できる。
#: ラベルは日本語 (``タスク:``) だけでなく planner がそのまま出す英語
#: (``Task:``) の形もある (実インシデント 2026-07-29 ライブ監査: 発明された
#: ``document.txt`` に ``Task: Delete the 5th item from the document`` が
#: 本文として書き込まれた)。動詞を要求する制約は残し、実文書中の一般語
#: 「タスク:」との区別を保つ。
_TASK_SCAFFOLD_LINE_RE = re.compile(
    r"^(?:タスク|Task)\s*[:：]\s*"
    r"(?:Write|Generate|Create|List|Read|Fetch|Revise|Update|Delete|Remove"
    r"|Add|Append|Modify|Replace|Edit|Save|Output)\b",
    re.MULTILINE,
)


#: 生成プロンプト冒頭の角括弧ラベル (日付コンテキスト / 直近会話ブロック等)。
#: 小型モデルはこれを言い換えて本文冒頭に書き写すため、リテラル一致の
#: ``_PROMPT_SCAFFOLD_MARKERS`` では拾えない (実インシデント 2026-07-27:
#: 「[現在日時 (UTC基準)]」が「[現在の日付 (UTC基準)]」に化けて書き込まれた)。
_SCAFFOLD_LABEL_LINE_RE = re.compile(
    r"^\s*[\[【](?:現在[のな]?日[時付]|直近の会話|参考資料|参考情報"
    r"|既存ファイル|取得データ|Current\s+date|Recent\s+conversation)"
)


def strip_prompt_scaffold_lines(content: str) -> str:
    """先頭に混入した生成プロンプトのラベル行 (と続く注意書き) を除去する。

    角括弧ラベル行と、その直後に続く空行までを 1 ブロックとして落とす。
    ラベル行以外に当たった時点で打ち切るので、本文中の角括弧は残る (純粋関数)。
    """
    lines = content.split("\n")
    idx = 0
    stripped_any = False
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if not _SCAFFOLD_LABEL_LINE_RE.match(line):
            break
        # ラベル行から次の空行までを 1 ブロックとして捨てる
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            idx += 1
        stripped_any = True
    if not stripped_any:
        return content
    remainder = "\n".join(lines[idx:]).strip("\n")
    return remainder or content


#: 生成器に向けた指示文 (読者に向けた文書本文には現れない言い回し)。
#: リテラルの scaffold マーカーは小型モデルが **言い換えて** 再生成すると
#: 効かない (実インシデント 2026-07-29 ライブ監査: マーカー登録した
#: ``## 取得済みデータ (…) / 以下のデータのみを根拠に…`` を塞いだ直後、
#: 同じ書式を真似た ``## 保存済み翻訳データ (E:\\tmp\\audit_r6_ja.md) /
#: 以下の翻訳済み内容のみを根拠に…`` が書き込まれた)。語ではなく
#: 「生成器への指示である」という構造で拾う。
_GENERATOR_DIRECTIVE_RE = re.compile(
    r"(?:のみ|だけ)を根拠に.{0,24}(?:出力|生成)"
    r"|(?:データ|情報|内容)に(?:無|な)い.{0,12}(?:創作|捏造)"
    r"|(?:創作|捏造)しないこと",
)


def _heading_titles_output_path(line: str, file_path: str) -> bool:
    """見出し行が出力先パス自身を名乗っているか (純粋関数)。

    実文書が自分の保存先パスを見出しに書くことはまず無いので、これを
    scaffold の signal として使える。
    """
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return file_path in stripped or (bool(basename) and basename in stripped)


def strip_generator_scaffold_block(content: str, file_path: str) -> str:
    """先頭の「生成器向けブロック」(自己言及見出し + 指示文) を除去する。

    出力先パスを名乗る見出しと、生成器に向けた指示文が続く限り読み飛ばす。
    どちらでもない行に当たった時点で打ち切るので、実文書の見出しは残る。
    全部消える場合は原文を返す (純粋関数)。
    """
    if not file_path:
        return content
    lines = content.split("\n")
    idx = 0
    stripped_any = False
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if (
            _heading_titles_output_path(line, file_path)
            or _GENERATOR_DIRECTIVE_RE.search(line)
        ):
            idx += 1
            stripped_any = True
            continue
        break
    if not stripped_any:
        return content
    remainder = "\n".join(lines[idx:]).strip("\n")
    return remainder or content


def strip_output_lead_in(content: str, file_path: str) -> str:
    """先頭の「<パス> の内容は以下の通りです。」型の前置き 1 行を除去する。

    小型モデルは会話の癖でファイル本文に回答の前置きを付ける。既存の
    ``strip_answer_framing`` は全体が鉤括弧 1 組で閉じている形しか剥がせず、
    「前置き 1 行 + 空行 + 本文」の形は素通りしていた (実インシデント
    2026-07-29 ライブ監査: audit_r5.md の 1 行目に
    ``E:\\tmp\\audit_r5.md の内容は以下の通りです。`` が書き込まれた)。

    前置き行が出力先パス (またはそのファイル名) を含むことを共起条件にする
    ので、たまたま「以下の通りです。」で始まる正当な文書は巻き込まない。
    残りが空になる場合は原文を返す (純粋関数)。
    """
    if not file_path:
        return content
    lines = content.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not _LEAD_IN_LINE_RE.match(stripped):
            return content
        basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
        mentions = file_path in stripped or (
            bool(basename) and basename in stripped
        )
        if not mentions:
            return content
        remainder = "\n".join(lines[i + 1:]).strip("\n")
        return remainder or content
    return content


def strip_task_log_scaffold(content: str) -> str:
    """先頭に混入したタスク進捗ノート行を取り除いた残りを返す。

    「タスクログ + 本文」の連結出力 (部分症状) から本文だけを救済する。
    ログ行が無ければ原文をそのまま返し、全行がログなら空文字列を返す
    (呼出側でエコーとして棄却する)。
    """
    lines = content.split("\n")
    idx = 0
    for i, line in enumerate(lines):
        if not line.strip():
            idx = i + 1
            continue
        if _TASK_LOG_LINE_RE.match(line):
            idx = i + 1
            continue
        break
    if idx == 0:
        return content
    return "\n".join(lines[idx:]).strip("\n")


#: 進捗ノートの残骸判定は core.text_quality が SSOT (注入側 = EvorefMem からも
#: 使うため)。ここは後方互換のための再エクスポート (どちらも不変)。
_TASK_LOG_FRAGMENT_RE = _CORE_TASK_LOG_FRAGMENT_RE
looks_like_task_log_residue = _looks_like_task_log_residue


def fewshot_contains_task_log(fewshot_block: str) -> bool:
    """few-shot 例にタスク進捗ノート形式の応答が含まれるかを判定する。

    「- [done] ... Written N bytes」だけの応答例を参考例として注入すると
    「書いた事実の報告だけ出せば正解」というバイアスを与え、本文なしの
    極小ファイル生成を誘発する (#incident 2026-07-15)。該当例を含む
    few-shot ブロックは注入しない。
    """
    return any(
        _TASK_LOG_LINE_RE.match(ln) for ln in fewshot_block.split("\n")
    )


def looks_like_task_log_echo(content: str) -> bool:
    """content がタスク進捗ノートのエコー (本文なし) かを判定する。"""
    if not any(_TASK_LOG_LINE_RE.match(ln) for ln in content.split("\n")):
        return False
    remainder = strip_task_log_scaffold(content)
    if remainder == content:
        return False
    return len(remainder.strip()) < 40


#: 「書き込み完了の報告」を成果物として出力してしまう退化の検出用。
#: 本文の代わりに「議事録を保存しました。**ファイル**: `path` **保存内容**: ...」
#: が書き込まれた (実インシデント 2026-07-27)。既存の task_log_echo は
