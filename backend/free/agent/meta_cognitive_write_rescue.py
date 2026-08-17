"""依頼文から書込み本文を救出する

モデルが本文を生成せず「保存しました」等の報告やスクリプトを返したとき、
依頼文そのもの (引用符で囲まれたリテラル / 直前の回答) から書込むべき本文を
取り出す層。救出できなければ空文字を返し、呼出側は従来どおり棄却する。
"""

from __future__ import annotations

import re

from pathlib import Path

from backend.free.agent.meta_cognitive_text import (
    _LITERAL_WRITE_EXTENSIONS,
    _LITERAL_WRITE_REJECT_RE,
)


_WRITE_REPORT_STEM = r"(?:保存|書き込み?|書込み?|作成|生成|出力|記録|更新)"
#: サ変動詞に付く助動詞。**活用形の全列挙ではなく助動詞の閉じた集合** として
#: 定義する。旧実装は能動の ``し|いたし`` だけを見ており、受身
#: 「作成されました」がすり抜けて実況文がファイル本文として書き込まれた
#: (実インシデント 2026-08-01 ライブ監査)。能動 / 受身 / 謙譲 / 可能 / 継続を
#: 網羅しておけば、以後は語形を足し続けなくてよい。
_SAHEN_AUX = (
    r"(?:し|され|なされ|いたし|致し|でき|出来"
    r"|しており|しておき|してあり|し終え|し終わり)"
)
#: 「書き込む」「書き出す」は五段動詞でサ変助動詞が付かないため別枝で持つ。
_GODAN_WRITE_REPORT = (
    r"(?:書き込み|書込み|書き出し|書出し)(?:ました|ます)"
    r"|(?:書き込|書込)まれ(?:ました|ます)"
    r"|(?:書き出|書出)され(?:ました|ます)"
)
_WRITE_REPORT_RE = re.compile(
    # 「保存しました」「作成されました」「作成いたしました」「出力できました」
    rf"{_WRITE_REPORT_STEM}(?:を)?{_SAHEN_AUX}(?:ました|ます|ています)"
    # 「書き込みました」「書き込まれました」(五段)
    rf"|{_GODAN_WRITE_REPORT}"
    # 「保存が完了しました」「書き込みは完了です」「作成済みです」(助詞挟み形)
    rf"|{_WRITE_REPORT_STEM}(?:[がはも])?\s*(?:完了|済み)"
    r"(?:し(?:ました|ます)|です|しています)?"
    r"|^\s*(?:Saved|Written|Created)\s+(?:to|the\s+file)\b",
    re.IGNORECASE | re.MULTILINE,
)
#: 「保存内容:」「ファイル:」のようなメタラベル (本文ではなく報告の構造)。
_WRITE_REPORT_LABEL_RE = re.compile(
    r"^\s*[*_`#\s]*(?:ファイル|パス|保存先|保存内容|出力先|File|Path|Content)"
    r"[*_`\s]*[:：]",
    re.MULTILINE,
)


#: 成果物そのものではなく「そのファイルを書くスクリプト」を返したときに現れる
#: 書込み命令 (実インシデント 2026-08-10 ライブ監査: member_schema.json に
#: ``$schema = @'…'@`` と ``$schema | Out-File -FilePath "E:\\tmp\\member_schema.json"``
#: が丸ごと書き込まれた。looks_like_tool_call_syntax と同じ
#: 「モデルが本文を作らず実行しようとした」退化だが、シェル構文で現れる)。
_WRITE_SCRIPT_RE = re.compile(
    r"Out-File|Set-Content|Add-Content|Export-Csv"
    r"|writeFileSync|WriteAllText|FileWriter"
    r"|open\s*\([^)]*['\"][wa]b?['\"]",
    re.IGNORECASE,
)

#: PowerShell の here-string 代入。成果物が .ps1 等のスクリプトなら正当なので、
#: 対象がスクリプトファイルのときは見ない。
_HERE_STRING_ASSIGN_RE = re.compile(r"^\s*\$\w+\s*=\s*@['\"]", re.MULTILINE)

#: 中身がスクリプトであることが期待される拡張子。
_SCRIPT_SUFFIXES: tuple[str, ...] = (
    ".ps1", ".psm1", ".sh", ".bash", ".bat", ".cmd", ".py", ".js", ".ts",
)


def looks_like_write_script(content: str, file_path: str) -> bool:
    """content が成果物ではなく「そのファイルを書くスクリプト」かを判定する。

    対象がスクリプトファイル自体の場合は、書込み命令もヒアストリングも正当な
    中身になりうるので判定しない。
    """
    if not content or not file_path:
        return False
    lowered_path = file_path.lower()
    if lowered_path.endswith(_SCRIPT_SUFFIXES):
        return False
    if _HERE_STRING_ASSIGN_RE.search(content):
        return True
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    mentions_target = file_path in content or (bool(basename) and basename in content)
    return bool(mentions_target and _WRITE_SCRIPT_RE.search(content))


def looks_like_write_report(content: str, file_path: str) -> bool:
    """content が「書き込みました」という完了報告かを判定する。

    冒頭で完了を宣言し、かつ保存先パス (またはそのファイル名) を含む場合のみ
    真とする。文書がたまたま「作成しました」を含むだけでは弾かない。
    """
    if not file_path:
        return False
    head = "\n".join(
        [ln for ln in content.split("\n") if ln.strip()][:3],
    )
    if not head:
        return False
    if not _WRITE_REPORT_RE.search(head):
        return False
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    mentions_target = (
        file_path in content
        or (bool(basename) and basename in content)
    )
    if not mentions_target:
        return False
    # 報告のメタラベルがあるか、全体が短い (本文が無い) 場合に限る
    return bool(_WRITE_REPORT_LABEL_RE.search(content)) or len(content) < 400


#: 依頼文中の引用スパン (本文候補)。
_QUOTED_SPAN_RE = re.compile(r"[「『]([^「」『』]{1,2000})[」』]")
#: 引用がパス / ファイル名を指しているケース (本文候補から外す)。
_QUOTE_IS_PATH_RE = re.compile(r"[/\\]|^\S+\.[A-Za-z][A-Za-z0-9]{0,9}$")
#: 実況文が引用本文を包んでいると判定する最小の余剰文字数。引用に句読点や
#: 改行が付いただけの整形は本文として通す。
_LITERAL_WRAPPER_MIN_EXTRA = 8
#: 「引用 + 前後の枠付け」に留まると見なす上限。引用を含みつつ本文を書き足した
#: 長い文書 (依頼文を冒頭で引用する議事録など) を巻き込まないための天井。
#: 引用長に比例させ、短い引用ほど枠付けの余地を許す。
_LITERAL_WRAPPER_MAX_RATIO = 3
_LITERAL_WRAPPER_MAX_FRAMING = 80


def quoted_write_literals(instruction: str) -> list[str]:
    """依頼文から本文候補となる引用スパンを取り出す (純粋関数)。

    パス / ファイル名を指す引用と、タイトル参照は候補から外す。
    """
    if not instruction or _LITERAL_WRITE_REJECT_RE.search(instruction):
        return []
    out: list[str] = []
    for m in _QUOTED_SPAN_RE.finditer(instruction):
        span = m.group(1).strip()
        if not span or _QUOTE_IS_PATH_RE.search(span):
            continue
        out.append(span)
    return out


def looks_like_literal_wrapper(
    content: str, instruction: str, file_path: str,
) -> bool:
    """ユーザーが引用符で確定させた本文を、実況文で包んだだけかを判定する。

    「…に『動作確認テスト』とだけ書いたファイルを作って」に対し、ファイルへ
    「ファイル `<path>` は「動作確認テスト」という内容で作成されました。」が
    書き込まれた (実インシデント 2026-08-01 ライブ監査)。

    依頼文の言い回し (書いて / 書いた / 書き直して …) に一切依存せず、**出力側の
    形** だけで判定するのが要点。完了報告の語形を追い続ける必要が無くなる。
    条件は「本文が確定している」「生成物がそれを枠付けしただけ」
    「生成物が自分の書込み先を名指している」の 3 つが揃うこと。依頼文を冒頭で
    引用しつつ本文を書き足した長い文書は、枠付けの天井を超えるので通る。
    """
    if not file_path or not content:
        return False
    literals = quoted_write_literals(instruction)
    if not literals:
        return False
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    if file_path not in content and not (basename and basename in content):
        return False
    body = content.strip()
    return any(
        literal in body
        and len(literal) + _LITERAL_WRAPPER_MIN_EXTRA <= len(body)
        <= len(literal) * _LITERAL_WRAPPER_MAX_RATIO + _LITERAL_WRAPPER_MAX_FRAMING
        for literal in literals
    )


def rescue_quoted_write_literal(instruction: str, file_path: str) -> str:
    """生成が棄却された後に限り使う、緩い本文引用の特定 (救済経路)。

    ``extract_literal_write_content`` は「引用の直後に書込み動詞」を要求する
    高精度マッチで、「『…』とだけ書いたファイルを作って」のような語形は
    取り逃がす。語形を足し続けるのは破綻するため、**生成が棄却された後** に
    限って動く低精度側を分離した。既に「生成は失敗」と分かっている状態で
    しか呼ばれないので、多少緩くても誤爆しない。

    候補が 1 つに定まる場合のみ返す (複数候補は本文が特定できないので空)。

    Returns:
        本文。特定できなければ空文字列 (純粋関数)。
    """
    if Path(file_path).suffix.lower() not in _LITERAL_WRITE_EXTENSIONS:
        return ""
    literals = quoted_write_literals(instruction)
    if len(literals) != 1:
        return ""
    return literals[0]


#: 「この案内文を保存して」のように直前の成果物を指す参照表現。
#:
#: 成果物の名詞は **散文以外も** 並べる。コード・スクリプト・定義・計算過程が
#: 抜けていたため「そのテストコードを保存して」で決定論経路が発火せず、LLM の
#: 再生成に回って `E:\tmp\test_mttr.py にテストコードを保存しました。` という
#: 完了報告を本文として出し、2 回とも棄却されて書込み自体が失敗した
#: (実インシデント 2026-08-10 ライブ監査)。直前の応答をそのまま書けばよい
#: ケースを生成に回さないのが最も確実。
_PREVIOUS_ANSWER_REF_RE = re.compile(
    r"(?:この|その|上記の?|先(?:ほど|程)の?|さっきの?|いまの|今の|先の|提示した)\s*"
    r"(?:案内文?|文章|文面|本文|内容|議事録|メモ|原稿|下書き|回答|答え|結果"
    r"|一覧|リスト|表"
    r"|コード|スクリプト|プログラム|関数|クラス|テスト|クエリ|設定|定義"
    r"|設計|手順|計算過程|計算|説明|要点)"
)
#: 直前の成果物に手を加える依頼 (そのまま書き写してはいけない)。
_TRANSFORM_VERB_RE = re.compile(
    r"翻訳|英訳|和訳|要約|短く|長く|整えて|直して|修正|変えて|変更|書き換え"
    r"|追記|付け加え|足して|加えて|敬語|丁寧に|箇条書きに|表にして|まとめ直"
)
#: 保存/書き出しの依頼であることのシグナル。
_WRITE_REQUEST_RE = re.compile(r"保存|書き出|書き込|出力|ファイルに|セーブ|save|write", re.IGNORECASE)

#: そのまま書き写す対象として採用する直前応答の最小文字数。
_PREVIOUS_ANSWER_MIN_CHARS = 40

#: 応答冒頭の前置き 1 文 (「〜を作成しました。」「以下の通りです。」)。
#: 会話では自然だがファイル本文としては不要なので、書き写す際に落とす。
_LEAD_IN_LINE_RE = re.compile(
    r"^.{0,60}?(?:作成しました|作りました|まとめました|用意しました"
    r"|以下の通りです|以下のとおりです|次のとおりです)[。．.]?$",
)


def _strip_lead_in(text: str) -> str:
    """先頭の前置き 1 文を落とす (残りが空なら原文を返す。純粋関数)。"""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not _LEAD_IN_LINE_RE.match(stripped):
            return text
        remainder = "\n".join(lines[i + 1:]).strip("\n")
        return remainder or text
    return text


#: 応答全体がひとつのコードフェンスで囲まれている形。
#:
#: ``strip_markdown_wrapper`` は「最長のフェンス内」を取り出すため、散文と
#: コードが混在する成果物 (.md 等) では地の文を落としてしまう。決定論経路は
#: 直前の応答を **そのまま** 使うのが原則なので、外してよいのは「応答が
#: コードブロックそのもの」のときだけに限る (実インシデント 2026-08-10
#: ライブ監査: 「そのコードを .py に保存して」で ```python 行ごと書き込まれた)。
_SOLE_CODE_FENCE_RE = re.compile(r"\A```[\w+-]*\s*\n(.*?)\n?```\s*\Z", re.DOTALL)


def unwrap_sole_code_fence(text: str) -> str:
    """全体が 1 つのコードフェンスなら中身だけ返す (純粋関数)。"""
    m = _SOLE_CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


#: 参照名詞が指す成果物の「見た目」。参照が種別を名指ししているのに直近の応答が
#: 別種別なら、1 つ飛ばして探す。
#:
#: 実インシデント 2026-08-10 ライブ監査: 表を出した次のターンで「その表は
#: どんな情報を持っていますか」と聞き、その次に「その表を保存して」と頼んだら、
#: **直前の 1 文 (41 文字の説明)** が保存された。参照は「表」と種別を明示して
#: いるのに、候補の中身と突き合わせていなかった。
_ARTIFACT_SHAPES: tuple[tuple[re.Pattern, re.Pattern], ...] = (
    # 表・リスト: GFM の行かタブ区切りの多列行
    (
        re.compile(r"表|テーブル|リスト|一覧"),
        re.compile(r"^\s*\|.*\|\s*$|^[^\t\n]+\t[^\t\n]+", re.MULTILINE),
    ),
    # コード類
    (
        re.compile(r"コード|スクリプト|プログラム|関数|クラス|テスト|クエリ"),
        re.compile(r"^\s*(?:def |class |import |from |SELECT |CREATE )", re.MULTILINE),
    ),
)


def _artifact_shape_for(query: str) -> "re.Pattern | None":
    """参照名詞から、候補が満たすべき見た目を返す (純粋関数)。"""
    for noun_re, shape_re in _ARTIFACT_SHAPES:
        if noun_re.search(query):
            return shape_re
    return None


def previous_answer_write_content(
    query: str, conversation: list[dict] | None,
) -> str:
    """「この文章を保存して」型の依頼に対し、直前の応答本文を決定論的に返す。

    書くべき本文が会話の中に既にあるのに毎回 LLM へ生成させ直すと、素材を
    見失って完了報告や架空の例文を本文として書き出す退化が起きる
    (実インシデント 2026-07-27: few-shot 書式の複写 / 「内容の保存が
    完了しました。」という報告文がファイルに書かれた)。参照表現があり、
    かつ加工指示 (翻訳・要約・修正等) が無い場合に限り、直前の assistant
    応答をそのまま採用する。該当しなければ空文字列。
    """
    if not conversation:
        return ""
    if not _WRITE_REQUEST_RE.search(query):
        return ""
    if not _PREVIOUS_ANSWER_REF_RE.search(query):
        return ""
    if _TRANSFORM_VERB_RE.search(query):
        return ""
    candidates: list[str] = []
    for msg in reversed(conversation):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        text = unwrap_sole_code_fence(_strip_lead_in(content.strip()))
        if len(text) >= _PREVIOUS_ANSWER_MIN_CHARS:
            candidates.append(text)
        # 直前が短い定型応答 (「タスクを完了しました。」等) ならさらに遡る
    if not candidates:
        return ""
    # 参照が種別を名指ししている (「その**表**を保存して」) なら、その見た目を
    # 持つ最も新しい応答を選ぶ。名指しが無い / 該当が無ければ従来どおり直近。
    shape = _artifact_shape_for(query)
    if shape is not None:
        for text in candidates:
            if shape.search(text):
                return text
    return candidates[0]


#: ベースモデルが本文の代わりに吐くツールコール構文。チャットテンプレートや
#: 学習データ由来の特殊トークンで、本文として書き込む価値は無い。
#:
#: 実インシデント 2026-08-09 (2 回目のライブ監査): 上書き依頼に対し base が
#: ``<|tool_call>call:write_file{file_path: "E:\tmp\club_plan.txt"}<tool_call|>``
#: を生成し、それが **チャット本文としてユーザーに露出** した。write_file は
#: 一度も走らずファイルは無変更だったが、UI には ✓ が付いた。
