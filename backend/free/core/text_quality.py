"""生成テキストの決定論的な品質チェック (pillar 非依存の共有基盤)

モデル差し替えで静かに劣化する「表記の崩れ」を、LLM 採点に頼らず決定論で検出する。
判定器をここに集約するのは、**同じ崩れを 2 箇所で別々に定義すると片方だけ直る**
ためで、実際に以下 2 系統が同一の判定を必要とする:

- :mod:`backend.free.learning.fewshot_pool` — 崩れた応答を手本に採用しない (入口ゲート)
- :mod:`backend.free.llm.quality_probe` — モデル切替時に崩れを検出する (事前ゲート)

補助タスク採点は使わない。小型モデルは自分と同種の崩れを問題と認識できず、実測で
空白混入例の quality 平均 0.80 に対し正常例 0.89 と 0.09 しか差が付かなかった
(混入例に 0.95 が 4 件)。決定論でのみ分離できる。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: 日本語の語間に混じった空白。
#:
#: 正常な日本語では和文文字が空白で分かたれることはない (実測: Qwen3.5-9B 時代の
#: 応答 96 件中 0 件)。一方 gemma-4-12b では 76〜83% に混入し、``temperature=0.0``
#: の貪欲法でも再現した — サンプリングではなく出力分布そのものの性質。
_JA_INTERWORD_SPACE_RE = re.compile(r"[ぁ-んァ-ヶ一-龥][ 　]+[ぁ-んァ-ヶ一-龥]")

#: コードブロック。整形済みコードやログ引用では半角空白が正常に現れる。
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

#: 和文文字。応答が日本語かどうかの判定に使う。
_JA_CHAR_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")

#: 日本語判定の下限。これ未満の和文文字しか無い応答は英語応答等とみなし、
#: 語間空白チェックの母数から外す (英文の空白を誤検出しないため)。
_JA_MIN_CHARS = 20


def has_broken_ja_spacing(text: str) -> bool:
    """日本語部分に不自然な語間空白が混じっているかを判定する (純粋関数)。

    コードブロック内は対象外。整形済みコードやログの引用で半角空白が現れるのは
    正常なため、fence で囲まれた領域を除いてから判定する。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    return bool(_JA_INTERWORD_SPACE_RE.search(outside))


def is_japanese_text(text: str) -> bool:
    """語間空白チェックの母数に含めてよい日本語応答かを判定する。

    和文文字が :data:`_JA_MIN_CHARS` 未満の応答 (英語応答 / 空応答 / コードのみ)
    は False。日本語で答えていないモデルを「空白混入 0%」と誤って合格させない
    ため、母数側でも呼び出し元が本関数で足切りする。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    return len(_JA_CHAR_RE.findall(outside)) >= _JA_MIN_CHARS


#: 日本語文に混じる簡体字。日本の常用漢字・人名用漢字には存在しない字形だけを
#: 挙げる (旧字体・異体字として日本語文に現れうる字は入れない)。
_SIMPLIFIED_ONLY_CHARS = "们这说认从个时么没很吗呢东车门问间语谁际现实发对开关书长风"

#: 中国語の繋辞「是」。日本語では ``是非`` / ``是正`` / ``是認`` / ``国是`` /
#: ``是々非々`` のような熟語でしか使われず、単独で名詞と名詞をつなぐ用法は無い。
#: 熟語の構成要素であるケースを除いてから検出する。
_JA_ZE_COMPOUND_RE = re.compile(r"是[非正認々]|[国是]是")
_BARE_COPULA_ZE_RE = re.compile(r"是")

_SIMPLIFIED_ONLY_RE = re.compile(f"[{_SIMPLIFIED_ONLY_CHARS}]")


def has_chinese_token_leak(text: str) -> bool:
    """日本語の応答に中国語の語彙が紛れ込んでいるかを判定する (純粋関数)。

    多言語モデルは日本語生成中に中国語トークンを混ぜることがある。
    2026-08-16 ライブ監査 (Qwen3.8-27B) では「私について知っていること」を
    2 度尋ねた両方で **「名前是小川さんです。」** と繋辞の ``是`` が出た。
    2 度目は 1 度目の出力が文脈に残っていたための複写で、この種の崩れは
    手本 (few-shot) に載ると再生産されて自己増幅する
    (``fewshot_pool._reject_reason`` の他ゲートと同じ構造)。

    誤検出を避けるため、判定はコードブロックの外に限り、かつ

    - 日本の漢字に存在しない **簡体字**
    - 熟語 (是非 / 是正 / 是認 / 国是) の構成要素でない **単独の「是」**

    という「日本語文には現れえない」形だけを見る。実測 (2026-08-16 監査の
    実データ): assistant 応答 40 件中 2 件 (上記の実インシデント) を検出し、
    user 発話 40 件 / STM ノート 50 件 / 是の熟語では 0 件だった。

    漢字は中国語と共有するため、中国語のみで書かれた文も True になる。手本の
    足切りという用途では望ましい側なのでそのままにする。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    if not _JA_CHAR_RE.search(outside):
        return False
    if _SIMPLIFIED_ONLY_RE.search(outside):
        return True
    return bool(_BARE_COPULA_ZE_RE.search(_JA_ZE_COMPOUND_RE.sub("", outside)))


#: 「<ラベル>は<数値>」型の言明。ラベルは記号・句読点で切れる 1 つながりの語で、
#: 数値は桁区切りと全角を許す。
#:
#: 例: 「年間売上は4,320,000円になります。」→ (年間売上, 4320000)
_LABELED_NUMBER_RE = re.compile(
    r"(?P<label>[0-9A-Za-z_ぁ-んァ-ヶーｦ-ﾟ一-龥]{2,24})"
    r"\s*(?:は|が|＝|=|:|：|\bis\b|\bwas\b|\bwere\b|\bare\b)\s*"
    r"(?P<num>[0-9０-９][0-9０-９,，.]*)",
    re.IGNORECASE,
)

#: 「<ラベル>はいくらですか」型の問い。値は**次の応答**に現れるので、ラベルと
#: 数値がメッセージをまたいで分かれる (実インシデントがまさにこの形だった)。
_LABEL_QUESTION_RE = re.compile(
    r"(?P<label>[0-9A-Za-z_ぁ-んァ-ヶーｦ-ﾟ一-龥]{2,24})"
    r"\s*(?:は|が)\s*(?:いくつ|いくら|何|どれ(?:くらい|ほど)?|どのくらい)",
)

#: 応答から拾う最初の数値。
_FIRST_NUMBER_RE = re.compile(r"[0-9０-９][0-9０-９,，.]*")

#: 全角数字 → 半角。
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９，", "0123456789,")


def _normalize_number(raw: str) -> str:
    """桁区切り・全角を落として数値文字列を正規化する。"""
    return raw.translate(_FULLWIDTH_DIGITS).replace(",", "").rstrip(".")


def _script_of(ch: str) -> str:
    """文字種。ラベルの語境界を推定するのに使う。"""
    if "ぁ" <= ch <= "ゟ":
        return "hira"
    if "゠" <= ch <= "ヿ" or "ｦ" <= ch <= "ﾟ":
        return "kata"
    if "一" <= ch <= "鿿":
        return "kanji"
    if ch.isdigit():
        return "digit"
    return "latin"


def _label_variants(run: str) -> set[str]:
    """ラベル候補の語を、文字種の切れ目で切り出した接尾辞ごと返す。

    正規表現が拾う「は」直前の 1 つながりの語には助詞や数量詞が前置される
    (「200人なら年間売上」)。一方、比較相手のテキストでは同じ語が裸で現れる
    (「年間売上」)。文字種が変わる位置を語境界とみなして接尾辞も候補に入れ、
    両者が同じ核 (「年間売上」) で一致できるようにする。
    """
    out = {run}
    for i in range(1, len(run)):
        if _script_of(run[i - 1]) != _script_of(run[i]) and len(run) - i >= 2:
            out.add(run[i:])
    return out


def _add_claim(claims: dict[str, set[str]], run: str, number: str) -> None:
    if run.translate(_FULLWIDTH_DIGITS).isdigit():
        return
    for label in _label_variants(run):
        claims.setdefault(label, set()).add(number)


def labeled_numeric_claims(text: str) -> dict[str, set[str]]:
    """テキストから「<ラベル>は<数値>」の言明を取り出す (純粋関数)。

    コードブロックは除外する (識別子と数値の羅列は言明ではない)。
    ラベルが数字だけのものは捨てる (「2026 は 8」のような偶発一致を拾わない)。
    """
    claims: dict[str, set[str]] = {}
    for m in _LABELED_NUMBER_RE.finditer(_CODE_FENCE_RE.sub("\n", text or "")):
        _add_claim(claims, m.group("label"), _normalize_number(m.group("num")))
    return claims


def conversational_numeric_claims(
    messages: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    """会話 ``(role, content)`` 列から確定済みの数値言明を集める (純粋関数)。

    1 メッセージ内で完結する「<ラベル>は<数値>」に加えて、**問いと答えが
    メッセージをまたいで分かれる形**も拾う:

        user      : 月額980円で有料ユーザーが200人なら年間売上はいくらですか？
        assistant : 2,352,000円です。

    実インシデント (2026-08-16 再測定) はまさにこの形で、ラベルは user 側、
    値は assistant 側にしか無かった。単文の抽出だけでは 1 件も拾えない。
    """
    claims: dict[str, set[str]] = {}
    pairs = list(messages or ())
    for role, content in pairs:
        if role != "system":
            for label, values in labeled_numeric_claims(content).items():
                claims.setdefault(label, set()).update(values)
    for (role, content), (next_role, next_content) in zip(pairs, pairs[1:]):
        if role != "user" or next_role != "assistant":
            continue
        answer = _FIRST_NUMBER_RE.search(_CODE_FENCE_RE.sub("\n", next_content or ""))
        if answer is None:
            continue
        number = _normalize_number(answer.group(0))
        for m in _LABEL_QUESTION_RE.finditer(content or ""):
            _add_claim(claims, m.group("label"), number)
    return claims


def find_superseded_claim(
    candidate: str, current_claims: dict[str, set[str]],
) -> tuple[str, str, set[str]] | None:
    """``candidate`` が現在の会話で既に確定した値と食い違うかを返す。

    返すのは ``(ラベル, 候補側の値, 現在の会話側の値集合)``。食い違いが無ければ
    ``None``。同じ値を再掲しているだけの候補は ``None`` (無害なので落とさない)。

    用途: 過去セッション由来の記憶 / RAG チャンクが、**今回の会話で算出・提示した
    値と同じラベルに別の値**を持ち込むのを止める。system プロンプトは
    「[関連する記憶]・[参考情報] は自分の記憶より優先して根拠にする」と規定して
    おり、例外は「ユーザー自身に関する事実」だけなので、**今回の会話で出した値は
    古い記録に負ける**。

    実インシデント (2026-08-16 再測定): 「月額980円×200人」で年間売上
    2,352,000 円を算出した直後に「さっき計算した年間売上をもう一度」と尋ねると、
    前セッションの ``年間売上は4,320,000円になります。`` が [関連する記憶] と
    [参考情報 1] の両方に載り、モデルは **4,320,000** を答えた。

    ラベル衝突 (別の話題で同じラベル語) が起きても、落とすのは常に古い側なので
    「今回の会話を優先する」という規定と同じ向きに倒れる。
    """
    if not current_claims:
        return None
    for label, values in labeled_numeric_claims(candidate).items():
        current = current_claims.get(label)
        if not current:
            continue
        conflicting = values - current
        if conflicting:
            return label, sorted(conflicting)[0], current
    return None


#: 「そのまま見せて」「先頭 N 行」型の、**逐語の抜粋**を求める依頼。
_VERBATIM_EXCERPT_RE = re.compile(
    r"そのまま(?:見せ|表示|出力|貼|書)"
    r"|(?:最初|先頭|冒頭|末尾|最後)\D{0,6}?[0-9０-９]{1,4}\s*(?:行|文字|lines?)"
    r"|全文\s*(?:を)?\s*(?:見せ|表示|出力|貼|読)"
    r"|(?:中身|内容)\s*(?:を)?\s*(?:そのまま|全部|すべて|丸ごと)"
    r"|verbatim|as[- ]is|\bfirst\s+\d+\s+lines?\b|\braw\s+content\b",
    re.IGNORECASE,
)


def asks_verbatim_excerpt(query: str) -> bool:
    """逐語の抜粋 (ファイル本文など) を求める依頼か (純粋関数)。

    この種の依頼への応答は **ツール出力の逐語コピー** であって、文体の手本では
    ない。few-shot に載せると「ペイロードを貼るのが正解」というバイアスを注入し、
    まったく別の質問にも本文の貼り付けを誘発する。

    実インシデント (2026-08-16 動作検証 T9): 「``README.md`` は存在しますか？」
    という yes/no の質問に対し、ツールは正しく 1 行だけ返していた
    (``read_file(..., start_line=1, end_line=1)``) のに、応答は README 本文の
    ダンプになった。プロンプトを見ると few-shot に

        User: 全文は長すぎます。そのファイルの先頭5行だけをそのまま見せてください。
        Assistant: ```\\n# evoref — 自己進化型ローカル LLM アシスタント\\n…

    が載っており、モデルはその形を写していた。**手本自体は当時正しい回答**
    だったが、別の問いの手本としては有害になる。
    """
    return bool(_VERBATIM_EXCERPT_RE.search(query or ""))


#: 本文に占めるコードフェンス内テキストの比率がこれ以上なら「ペイロードの
#: 貼り付け」とみなす。
#:
#: 実データでの分離 (2026-08-16 動作検証時の STM 75 件): 比率 0.5 以上は 2 件で
#: **どちらも README の全文ダンプ** (1.00 / 0.64)。次点は
#: ``datetime.utcnow()`` の解説 (コード例つきの正当な応答) で 0.25 と大きく離れる。
_PAYLOAD_FENCE_RATIO = 0.5


def is_payload_dump(text: str) -> bool:
    """本文の過半がコードフェンスの中身か (純粋関数)。

    こう判定されたテキストは「**いつでも取り直せるデータのコピー**」であって、
    覚えておくべき事実ではない。記憶として再注入すると、内容が古びるうえに
    「ペイロードを貼るのが正解」という手本として働く。

    実インシデント (2026-08-16 動作検証): モデルが README を全文ダンプした回答が
    assistant ノートとして STM に入り、``MemoryInjector`` が ``(過去の記録)`` として
    再注入していた。次のターンでモデルはそれを見てまたダンプする — **自分の出力が
    自分への指示になる自己増幅ループ**。ツール側 (PR #436/#439) と few-shot 側
    (PR #446) を塞いでも、この経路が残っていると再生産され続ける。
    """
    body = text or ""
    if not body:
        return False
    fenced = sum(len(m) for m in _CODE_FENCE_RE.findall(body))
    return fenced / len(body) >= _PAYLOAD_FENCE_RATIO


#: 逐語エコー判定に載せるユーザー発言の下限文字数。これ未満は「はい」「OK」
#: 等の短い相槌で、応答が偶然同じ語で始まっただけの誤検出になりうる。
_ECHO_MIN_QUERY_CHARS = 8


def strip_echoed_query(response: str, query: str) -> str:
    """応答冒頭に混じったユーザー発言の逐語コピーを取り除く (純粋関数)。

    ユーザーの問いをそのまま繰り返してから答える崩れが実在する。単発なら
    見た目が悪いだけだが、その応答が記憶ノートとして保存されると、同じ問いで
    想起されて再生産され、繰り返し回数が増えていく (実インシデント
    2026-08-04 ライブ監査:「今日は何曜日ですか。」に対し同文を 5 回返して
    答えが出ない状態まで悪化した。汚染ノートを除去したら 5/5 で解消)。

    ``query`` が :data:`_ECHO_MIN_QUERY_CHARS` 未満のときは何もしない。
    """
    q = (query or "").strip()
    if len(q) < _ECHO_MIN_QUERY_CHARS:
        return response
    out = response.lstrip()
    while out.startswith(q):
        out = out[len(q):].lstrip()
    return out


#: 復唱の前に許す前置きの長さ。呼びかけ (「小川さん、」) を挟んでから復唱する
#: 形が実在するため、位置 0 の一致だけでは取りこぼす (実測 2026-08-04:
#: 「私の名前を覚えていますか。」に対し「小川さん、私の名前を覚えていますか。」)。
#: 短い前置きに限るのは、問いを引用してから答える正常な応答を巻き込まないため。
_ECHO_MAX_LEAD_CHARS = 16


def is_query_echo(response: str, query: str) -> bool:
    """応答がユーザー発言の逐語繰り返しだけで中身を持たないかを判定する。"""
    q = (query or "").strip()
    if len(q) < _ECHO_MIN_QUERY_CHARS:
        return False
    if not strip_echoed_query(response, q).strip():
        return True
    idx = response.find(q)
    if 0 < idx <= _ECHO_MAX_LEAD_CHARS:
        return not strip_echoed_query(response[idx:], q).strip()
    return False


#: 本文の後ろに付く定型の締め文。system プロンプトの出力形式が明示的に禁止して
#: いる (「応答の末尾に自己紹介・挨拶・『他にご質問はありますか?』等の定型文を
#: 追加しない」) にもかかわらず実際に出る。禁止だけでは消えないのは、違反応答が
#: 手本として採用され再生産されるため (実インシデント 2026-08-04 ライブ監査:
#: fitness 0.889 の最上位帯で few-shot に載っていた)。
_BOILERPLATE_CLOSING_RE = re.compile(
    r"(?:"
    r"他に(?:も)?(?:ご質問|ご不明な点|お困りのこと|何か)[^。\n]{0,12}"
    r"|何か(?:他に)?お手伝いできること[^。\n]{0,8}"
    r"|(?:Is|Was) there anything else"
    r"|Let me know if you (?:need|have)"
    r")[^。\n]{0,12}[?？]\s*$",
)

#: 締め文を除いた本文がこれ未満なら、応答そのものが定型の挨拶とみなす。挨拶への
#: 反射応答 (``agent.reactive``) は締め文が本体なので、違反として扱わない。
_CLOSING_MIN_BODY_CHARS = 24


def has_boilerplate_closing(text: str) -> bool:
    """本文の末尾に禁止された定型の締め文が付いているかを判定する (純粋関数)。

    挨拶だけの短い応答は対象外。締め文が応答の本体である反射応答まで違反に
    数えると、正常な挨拶が手本から一律に落ちてしまう。
    """
    body = text.rstrip()
    m = _BOILERPLATE_CLOSING_RE.search(body)
    if m is None:
        return False
    return len(body[:m.start()].strip()) >= _CLOSING_MIN_BODY_CHARS


#: 終端記号を保ったまま文へ切る。
#:
#: 数字に挟まれた ``.`` は小数点なので文末にしない。素朴に ``.`` を終端に含めると
#: 「そこから決済手数料を 3.6% 引くと、手取りは年間いくらになりますか？」が
#: 「そこから決済手数料を 3.」と「6% …ますか？」の 2 文に割れ、前半が疑問文で
#: ないため :func:`carries_no_assertion` が **問いだけの発言を主張ありと誤判定** する
#: (2026-08-16 ライブ監査で実データから発覚。SemMem ファクト / STM ノートの
#: 問いゲートも同じ式を使っているため、そちらにも同じ穴があった)。
_SENTENCE_RE = re.compile(
    r"(?:[^。．.!！?？\n]|(?<=\d)[.．](?=\d))+"
    r"[。．.!！?？]?",
)

#: 疑問文の語尾。日本語は語尾が疑問を担い、疑問符が無いことが多い。
#: 過去形の丁寧疑問 (「変わりましたか。」「どうでしたか。」) が漏れていたため追加
#: (2026-08-16: 「Python 3.12 では何が変わりましたか。」が主張ありと判定されていた)。
_INTERROGATIVE_TAIL_RE = re.compile(
    r"(?:ですか|ますか|ましたか|でしたか|でしょうか|ありますか|いますか"
    r"|ませんか|ませんでしたか|だろうか)"
    r"[。．.]?$"
    r"|[?？]\s*$",
)


def strip_interrogative_sentences(text: str) -> str:
    """疑問文だけの文を落として、主張している部分を残す (純粋関数)。

    ファクトの ``object`` は発話原文をそのまま入れる設計なので、平叙文と疑問文が
    混じった発話は **問いごと** 記憶される。すると ``[関連する記憶]`` に
    「(emotion) mem.emotion.user feels: 夜更かしすると次の日つらいですよね。
    何かいい対策ありますか？」のように、答えではなく問いが根拠として並ぶ
    (2026-08-16 ライブ監査時点の実データ)。

    命題への完全な正規化には LLM が要るが、末尾の問いを落とすだけなら決定論で
    でき、観測された害はそれで消える。

    すべての文が疑問だった場合は **元のテキストを返す** (呼び出し側が
    ``carries_no_assertion`` で別途弾く前提。ここで空文字列を返すと
    ファクトそのものが消える)。
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text or "")]
    kept = [
        s for s in sentences
        if s and not _INTERROGATIVE_TAIL_RE.search(s)
    ]
    if not kept:
        return text
    return "".join(kept)


#: 文頭の談話標識。事実そのものではなく話の切り出し方。
_DISCOURSE_PREFIX_RE = re.compile(
    r"^\s*(?:ところで|そういえば|そう言えば|ちなみに|実は|じつは|あのー?|えっと"
    r"|なんか|ねえ|ねぇ|あっ|えっ|あ、|え、)\s*[、,]?\s*",
)

#: 文頭の一人称主題。「私は担々麺が好き」→「担々麺が好き」。
#: 主語を落として困るのは三人称の話をしているときだが、ここで扱うのは
#: 「ユーザー自身についてのファクト」なので主語は自明。
_FIRST_PERSON_TOPIC_RE = re.compile(
    r"^\s*(?:私|僕|俺|自分|わたし|ぼく|おれ)\s*(?:は|が|も|の場合は?)\s*"
    r"|^\s*(?:私|僕|俺|自分|わたし|ぼく|おれ)\s*[、,]\s*",
)


def strip_discourse_prefix(text: str) -> str:
    """文頭の談話標識を落とす (純粋関数)。"""
    return _DISCOURSE_PREFIX_RE.sub("", text or "", count=1)


def strip_first_person_topic(text: str) -> str:
    """文頭の一人称主題を落とす (純粋関数)。

    ファクトの主語は ``subject`` が持つので、``object`` 側に「私は」を残す
    意味は無い。むしろ ``[関連する記憶]`` に一人称の行が並ぶと、読み手が
    「誰の発言か」を取り違える材料になる。
    """
    return _FIRST_PERSON_TOPIC_RE.sub("", text or "", count=1)


def carries_no_assertion(text: str) -> bool:
    """本文が疑問だけで、知識としての主張を含まないかを判定する (純粋関数)。

    過去セッションのユーザー発言はそのまま記憶ノートになる。問いだけのノート
    (「今日は何曜日ですか。」) は答えを含まないのに ``(過去の記録)`` として
    想起され、モデルがそれを回答として出力してしまう (実インシデント
    2026-08-04 ライブ監査)。1 文でも断定・依頼が混じっていれば False を返す
    ため、「私の誕生日は 3 月 14 日です。あと何日ですか。」のような
    事実を含む発言は残る。
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text or "")]
    sentences = [s for s in sentences if s]
    if not sentences:
        return True
    return all(_INTERROGATIVE_TAIL_RE.search(s) for s in sentences)


__all__ = [
    "asks_verbatim_excerpt",
    "is_payload_dump",
    "carries_no_assertion",
    "conversational_numeric_claims",
    "find_superseded_claim",
    "has_boilerplate_closing",
    "has_broken_ja_spacing",
    "has_chinese_token_leak",
    "is_japanese_text",
    "labeled_numeric_claims",
    "is_query_echo",
    "strip_echoed_query",
    "strip_interrogative_sentences",
    "strip_discourse_prefix",
    "strip_first_person_topic",
]

#: 応答の途中で自分の結論を撤回する言い回し。1 つの応答に結論が 2 つ入る。
#:
#: 実インシデント (2026-08-07 ライブ監査):「2の10乗と10の3乗ではどちらが
#: 大きいですか？」に対し「10の3乗の方が大きいです。… 失礼しました、正しくは
#: 2の10乗（1,024）の方が大きいです。」と、誤った結論と訂正が同居した応答を
#: 返した。算術自体は正しいので ``find_arithmetic_contradictions`` では捕まらない。
#:
#: 消費側は 2 つ: few-shot の内容棄却ゲート (EvorefLearn) と、ターン成否の
#: 決定論判定 (``FeedbackCollector._derive_turn_outcome``、EvorefLoop)。
#: 「手本に採らない」だけでは学習の成否シグナルに届かないため、両方で見る。
_SELF_RETRACTION_RE = re.compile(
    r"(?:失礼しました|すみません|申し訳|訂正(?:します|いたします)|"
    r"間違えました|誤りでした)[、。,\s]*(?:正しくは|訂正)"
    r"|正しくは.{0,12}でした[。\s]*$",
)


def retracts_own_conclusion(text: str) -> bool:
    """応答が自分の結論を途中で撤回しているか (純粋関数)。"""
    return bool(_SELF_RETRACTION_RE.search(text or ""))
