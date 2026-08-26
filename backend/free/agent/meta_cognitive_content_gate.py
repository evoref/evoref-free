"""生成コンテンツの棄却判定

「モデルが返したものを本文として書いてよいか」だけを判定する層。エコー /
命令の言い換え / 拒否文 / パスだけ、といった **本文になっていない出力** を
決定論で弾く。整形は ``meta_cognitive_text``、救出は
``meta_cognitive_write_rescue`` の責務。
"""

from __future__ import annotations

import re

from difflib import SequenceMatcher

from backend.free.agent.meta_cognitive_write_rescue import (
    looks_like_literal_wrapper,
    looks_like_write_report,
    looks_like_write_script,
)

from backend.free.agent.meta_cognitive_scaffold import (
    _PROMPT_SCAFFOLD_MARKERS,
    _TASK_SCAFFOLD_LINE_RE,
    looks_like_task_log_echo,
)


# ---------------------------------------------------------------------------
# コンテンツ判定
# ---------------------------------------------------------------------------

_FEWSHOT_USER_LINE_RE = re.compile(r"^User:\s*(.+)$", re.MULTILINE)


def _char_bigrams(text: str) -> set[str]:
    """文字 bi-gram 集合を返す (テキスト間の粗い関連度判定用)。

    fewshot_pool.py の select_top_k と同じ手法だが、pillar 境界
    (EvorefLoop → EvorefLearn は import 不可) のためここで独立実装する。
    """
    t = text.strip()
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def fewshot_seems_relevant(
    task_text: str, fewshot_block: str, min_overlap: float = 0.03,
) -> bool:
    """few-shot ブロックが現在のタスクとどの程度関連しているかを粗く判定する。

    Level 1 が進化させた few-shot は query 類似度だけでなく fitness (過去の
    成功実績) も加味して選ばれるため、現在のタスクと無関係でも再利用され
    うる。無関係な few-shot 例をファイル内容生成タスクへそのまま注入すると、
    ローカル LLM がその例文をそのまま繰り返す退化を誘発しうる
    (#incident: 「テスト.docx」生成タスクに無関係な「映画ですか？」の
    few-shot が注入され、その例文の反復が延々と write_file されていた)。

    fewshot_block 中の ``User: ...`` 行 (例の質問文) だけを取り出して
    task_text と文字 bi-gram の Jaccard 重なりを見る。Assistant 側の応答文
    (定型の丁寧語などで一般的な bi-gram が多く、関連度判定のノイズになる)
    は比較対象から除く。``User:`` 行が見つからない/どちらかが極端に短い
    等で判定不能な場合は安全側 (注入する) に倒す。
    """
    queries = " ".join(_FEWSHOT_USER_LINE_RE.findall(fewshot_block))
    a = _char_bigrams(task_text)
    b = _char_bigrams(queries)
    if not a or not b:
        return True
    overlap = len(a & b) / len(a | b)
    return overlap >= min_overlap


#: 本文がパスそのものかを判定するパターン。区切り文字と拡張子を含む「パスの
#: 形」を要求する。以前は「/ か \\ を含み、最後のセグメントに . がある 1 行」
#: という緩い条件で、日付と小数を含む正当な本文まで棄却していた
#: (実インシデント 2026-07-27: 「チェーン注油 7/20、タイヤ空気圧 7.0気圧、
#: ブレーキパッドは残り半分」が path_only 判定で 4 回連続棄却され、
#: 上書き保存そのものが失敗した)。
_PATH_ONLY_RE = re.compile(
    r"^[\"'`]?"
    r"(?:[A-Za-z]:[\\/]|\.{0,2}[\\/]|~[\\/])"   # ドライブ / 相対 / ホーム起点
    r"[^\r\n\"'`]*"
    r"\.[A-Za-z0-9]{1,10}"                       # 拡張子で終わる
    r"[\"'`]?$",
)
#: パス判定から除外する文字 (日本語が含まれていれば散文と見なす)。
_CJK_RE = re.compile(r"[぀-ヿ一-鿿]")


def looks_like_path_not_content(content: str, file_path: str) -> bool:
    """content がファイルパスの誤出力かどうかを判定する"""
    if content == file_path:
        return True
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    if len(stripped) >= 260 or "\n" in stripped:
        return False
    if _CJK_RE.search(stripped):
        return False
    return bool(_PATH_ONLY_RE.match(stripped))

_TOOL_CALL_SYNTAX_RE = re.compile(
    r"<\|?\s*tool_call|tool_call\s*\|?>"
    r"|<\|?\s*(?:function|tool)_?(?:call|response)\s*\|?>"
    r"|\bcall\s*:\s*\w+\s*\{"
    r"|<\|(?:im_start|im_end|assistant|channel)\|>",
    re.IGNORECASE,
)


def looks_like_tool_call_syntax(content: str) -> bool:
    """content がツールコール構文/特殊トークンの吐き出しかを判定する (純粋関数)。"""
    return bool(_TOOL_CALL_SYNTAX_RE.search(content))


def looks_like_prompt_echo(content: str) -> bool:
    """content が生成プロンプトの scaffold を含む (= プロンプトエコー) かを判定する。"""
    if any(marker in content for marker in _PROMPT_SCAFFOLD_MARKERS):
        return True
    if looks_like_fewshot_echo(content):
        return True
    return bool(_TASK_SCAFFOLD_LINE_RE.search(content))


#: few-shot 例ブロック (``### Example N`` + ``User:`` / ``Assistant:``) の
#: 書式を成果物として複写した退化の検出用。
_FEWSHOT_EXAMPLE_HEADING_RE = re.compile(
    r"^#{1,4}\s*Example\s+\d+\s*$", re.MULTILINE | re.IGNORECASE,
)
_FEWSHOT_TURN_LINE_RE = re.compile(
    r"^(?:User|Assistant)\s*[:：]", re.MULTILINE,
)


def looks_like_fewshot_echo(content: str) -> bool:
    """content が few-shot 例ブロックの複写かを判定する。

    ``_generate_content`` の system には Level 1 が進化させた few-shot が
    ``### Example N / User: ... / Assistant: ...`` の書式で入る。書くべき本文が
    決まらないとき、小型モデルはこの書式ごと真似た架空の Q&A を「成果物」として
    出力する (実インシデント 2026-07-27: 直前に作った夏祭りの案内文を保存させたら、
    ファイルに ``## Example 1 / User: <保存先パス>を読んでください。 /
    Assistant: ...`` という架空の対話 3 件が書き込まれた)。
    見出しと発話行の両方が揃った場合のみ棄却する (Q&A 形式の正当な文書や、
    ``Example`` の語を含むだけの文書を巻き込まないため)。
    """
    if not _FEWSHOT_EXAMPLE_HEADING_RE.search(content):
        return False
    return len(_FEWSHOT_TURN_LINE_RE.findall(content)) >= 2


def csv_content_lacks_rows(content: str, file_path: str) -> bool:
    """.csv 出力先なのに区切り行が実質存在しないかを判定する。

    厳密な CSV 検証ではなく「散文/エコーを CSV として書く」事故の検出が目的。
    カンマ区切り行または GFM パイプ表行が 2 行以上 (ヘッダ+データ) あれば
    合格とする。``.csv`` 以外の出力先では常に False。
    """
    if not file_path.lower().endswith(".csv"):
        return False
    delimited = 0
    for line in content.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.count(",") >= 1 or (s.startswith("|") and s.endswith("|")):
            delimited += 1
        if delimited >= 2:
            return False
    return True


#: モデルが成果物の代わりに「アクセス権が無い」「内容を教えてほしい」等、
#: 自身の制約や情報不足を説明する断り書きを返す退化パターン (2026-07-22
#: 発見: 主題不明のまま write_file の内容生成をさせると、英語入力の
#: agentic write_file 経路でこの断り書き自体が本文として書き込まれた)。
_REFUSAL_MARKERS: tuple[str, ...] = (
    "i do not have direct file system access",
    "i don't have direct file system access",
    "i do not have access to the file system",
    "i don't have access to the file system",
    "as an ai model",
    "as an ai language model",
    "please provide the content",
    "please provide more details",
    "please specify what",
    "could you clarify",
    "could you please specify",
    "ファイルシステムに直接アクセス",
    "具体的な内容をご指定ください",
    "内容を教えてください",
    "どのような内容にすれば",
)


#: 日本語の断り書きは言い回しの幅が広く、逐語マーカーでは取りこぼす。謝辞と
#: 不能表現の **共起** で見る (実インシデント 2026-08-10 ライブ監査:
#: 「申し訳ありませんが、私はお客様のローカル環境にあるファイルシステム
#: （E:\\tmp\\ など）に直接ファイルを書き込む権限を持っていません。」が
#: _REFUSAL_MARKERS の「ファイルシステムに直接アクセス」に一致せず、
#: この断り書き自体が JSON Schema の代わりにファイルへ書き込まれ、UI は
#: ✓「書き込みました」と成功表示した)。
_JA_APOLOGY_RE = re.compile(r"申し訳(?:あり|ござい)ませ|恐れ入りますが|残念ながら")
_JA_INABILITY_RE = re.compile(
    r"権限[^。]{0,12}ませ"
    r"|(?:こと|の)は?できませ"
    r"|できかねま|いたしかねま"
    r"|アクセス(?:でき|は?できま)せ",
)


def looks_like_refusal_or_missing_info(content: str) -> bool:
    """content が本文ではなく、モデル自身の断り書き/情報不足の説明かを判定する。

    正当な長文コンテンツ中に類似語が偶然出現しても誤検出しないよう、短文
    (500 文字以下) の場合のみマーカーを見る。
    """
    if len(content) > 500:
        return False
    lowered = content.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return True
    return bool(_JA_APOLOGY_RE.search(content) and _JA_INABILITY_RE.search(content))


def looks_like_instruction_echo(content: str, instruction: str) -> bool:
    """content がユーザーの依頼文そのものの複写かを判定する。

    ``looks_like_prompt_echo`` は生成プロンプトの scaffold 語 (「タスク:」等) を
    見るため、依頼文を**そのまま**返した場合は素通りしていた
    (実インシデント 2026-07-27: 「E:\\tmp\\audit_r3.md というファイルを作って、
    中身は「監査テスト 1行目」だけにしてください。」→ 1 回目の生成は
    prompt_echo で棄却されたが、再生成が依頼文の逐語コピーを返し、それが
    そのままファイルに書き込まれた)。

    誤検出を避けるため判定は「ほぼ同一」に限る。依頼文に含まれる文言を
    書くよう頼まれる場面 (「『こんにちは』と書いて」→ 本文「こんにちは」) は
    長さが大きく異なるため、長さ比のガードで確実に除外される。
    """
    if not instruction:
        return False
    c = " ".join(content.split())
    q = " ".join(instruction.split())
    if len(c) < _INSTRUCTION_ECHO_MIN_CHARS or not q:
        return False
    ratio = len(c) / len(q)
    if not (_INSTRUCTION_ECHO_MIN_LEN_RATIO <= ratio <= _INSTRUCTION_ECHO_MAX_LEN_RATIO):
        return False
    return SequenceMatcher(None, c, q).ratio() >= _INSTRUCTION_ECHO_MIN_SIMILARITY


#: 依頼文エコー判定のガード。短文・長さ乖離・低類似は判定対象外にして
#: 正当な本文を巻き込まない。
_INSTRUCTION_ECHO_MIN_CHARS = 10
_INSTRUCTION_ECHO_MIN_LEN_RATIO = 0.8
_INSTRUCTION_ECHO_MAX_LEN_RATIO = 1.2
_INSTRUCTION_ECHO_MIN_SIMILARITY = 0.9


#: 「これから何をするか」を述べたメタラベルで始まる行 (成果物ではない)。
#: 既存の ``_TASK_SCAFFOLD_LINE_RE`` は英語動詞 (``タスク: Write ...``) の形しか
#: 見ておらず、日本語で依頼文を言い換えた形 (``タスクの内容: ファイル … に…``)
#: は素通りしていた (実インシデント 2026-07-29 ライブ監査)。
_TASK_RESTATEMENT_LABEL_RE = re.compile(
    r"^\s*[*_`#\s]*(?:タスク|指示|依頼|命令|要求|Task|Instruction|Request)"
    r"(?:の(?:内容|詳細))?\s*[:：]",
)


def looks_like_task_restatement(content: str, file_path: str) -> bool:
    """content が依頼文の言い換え (成果物ではない) かを判定する。

    冒頭がメタラベル (``タスク:`` / ``指示:``) で始まり、かつ本文が **出力先
    パスそのもの** を含む場合だけ真とする。自分の保存先を本文に書く文書は
    まず無いので、これを共起条件にすることで「タスク一覧」のような正当な
    文書 (先頭行が「タスク:」で始まりうる) を巻き込まない (純粋関数)。
    """
    if not file_path:
        return False
    head = next(
        (ln for ln in content.split("\n") if ln.strip()), "",
    )
    if not _TASK_RESTATEMENT_LABEL_RE.match(head):
        return False
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return file_path in content or (bool(basename) and basename in content)


#: 既存内容の変更を求める依頼の動詞。
#:
#: ``追記`` / ``書き足`` / ``末尾に`` は 2026-08-26 に追加した。``追加`` はあるが
#: ``追記`` はその部分文字列ではなく、英語側に ``append`` があるのに日本語側だけ
#: 欠けていた。そのため「B の中身を A の末尾に**追記**してください」で A 自身の
#: 内容がそのまま書き戻されても無変更と判定されず、"Written N bytes" と
#: 「完了しました」が返っていた (ライブ監査 T7-7)。
#:
#: これは ``intent_vocab.WRITE_VERB_RE`` で 2026-08-08 に直したのと **同じ
#: 語彙ドリフトが別の正規表現で再発** したもの。両者の同期は
#: ``test_audit_findings_20260826`` が検証する。
_EDIT_REQUEST_RE = re.compile(
    r"差し替え|差替え|置き換え|置換|入れ替え|変更|修正|直して|直す|更新"
    r"|書き換え|書き直|追加|追記|書き足|末尾に|足して|加えて"
    r"|削除|消して|除いて|外して"
    r"|replace|update|change|modif|edit|rewrite|append|remove|delete",
    re.IGNORECASE,
)


def _normalize_for_content_compare(text: str) -> str:
    """行末空白と改行コードの差を無視した比較用の正規形 (純粋関数)。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def edit_produced_no_change(
    content: str, existing_content: str, instruction: str,
) -> bool:
    """既存内容の変更を頼まれたのに生成結果が既存と同一かを判定する。

    小型モデルは既存内容ブロックを丸ごと写して編集指示を落とすことがある。
    書込み自体は成功するのでバイト数も返り、ユーザーには「完了しました」と
    表示されるが、ファイルは 1 文字も変わっていない (実インシデント
    2026-07-29 ライブ監査: 「そのリストの3番目を「ヘッドランプ」に差し替えて、
    同じファイルに保存し直してください。」で元のリストがそのまま書き戻され、
    "Written 126 bytes" と「タスクを完了しました。」が返った)。

    依頼文に変更を求める動詞があり、既存ファイルがあり、生成結果が既存と
    同一のときだけ真 (純粋関数)。新規作成や「同じ内容で別名保存」型の依頼は
    動詞が一致しないため巻き込まない。
    """
    if not existing_content or not instruction:
        return False
    if not _EDIT_REQUEST_RE.search(instruction):
        return False
    return (
        _normalize_for_content_compare(content)
        == _normalize_for_content_compare(existing_content)
    )


def generated_content_rejection(
    content: str, file_path: str, instruction: str = "",
    existing_content: str = "",
) -> str | None:
    """write_file 直前の生成コンテンツ検証。棄却理由を返す (適正なら None)。

    書込みパス (fast path / tool-loop / deliberative) から共通で呼び、
    タスクログエコー・プロンプトエコー・依頼文エコー・パス誤出力・
    CSV 構造欠落・断り書きを書込み前に弾く。

    Args:
        content: 書込み予定の生成コンテンツ。
        file_path: 書込み先パス (拡張子別の検証に使う)。
        instruction: 元のユーザー依頼文。空なら依頼文エコー検証を行わない。
        existing_content: 上書き対象の既存ファイル内容。空なら無変更検証を
            行わない (新規作成)。
    """
    if edit_produced_no_change(content, existing_content, instruction):
        return "edit_without_change"
    if looks_like_path_not_content(content, file_path):
        return "path_only"
    if looks_like_task_log_echo(content):
        return "task_log_echo"
    if looks_like_write_report(content, file_path):
        return "write_report_echo"
    if looks_like_task_restatement(content, file_path):
        return "task_restatement"
    # ツールコール構文はプロンプトエコーより先に見る。両方成立しうるが、
    # 「モデルがツールを呼ぼうとして本文を作らなかった」の方が具体的で、
    # 再生成の判断材料として正確 (2026-08-09 ライブ監査)。
    if looks_like_tool_call_syntax(content):
        return "tool_call_syntax"
    if looks_like_write_script(content, file_path):
        return "write_script"
    if looks_like_prompt_echo(content):
        return "prompt_echo"
    if looks_like_instruction_echo(content, instruction):
        return "instruction_echo"
    # instruction_echo (依頼文の逐語コピー) より後に置く。両方成立する場合は
    # より具体的な逐語コピー診断を優先し、報告される理由を安定させる。
    if looks_like_literal_wrapper(content, instruction, file_path):
        return "literal_wrapped"
    if csv_content_lacks_rows(content, file_path):
        return "csv_without_rows"
    if looks_like_refusal_or_missing_info(content):
        return "refusal_or_missing_info"
    return None


_CODE_INDICATORS: tuple[str, ...] = (
    "import ", "from ", "def ", "class ", "function ",
    "const ", "let ", "var ", "return ", "if __name__",
    "#include", "package ", "public class",
    "#!/", "# -*- create",
    "pygame", "print(", "console.log",
)


def contains_code_indicator(text: str) -> bool:
    """テキストにコードらしさを示すマーカーが1つ以上含まれるかを判定する。

    長さ・改行に依存しないため、1 行の短いコード片でも検出できる。
    散文 (例: "...を設計します。" のみ) は指標を含まず False になる。
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(ind.lower() in text_lower for ind in _CODE_INDICATORS)


def text_looks_like_code(text: str) -> bool:
    """テキストがプログラムコードに見えるかを判定する

    LLM がツールコール JSON ではなくコードをそのまま出力した場合に、
    それをファイルに書き込んでよいかを判定するために使用する。
    """
    if not text or len(text) < 20:
        return False
    if "\n" not in text:
        return False
    return contains_code_indicator(text)
