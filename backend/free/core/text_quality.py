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
from collections.abc import Callable, Iterable, Sequence

#: 日本語の語間に混じった空白。
#:
#: 正常な日本語では和文文字が空白で分かたれることはない (実測: Qwen3.5-9B 時代の
#: 応答 96 件中 0 件)。一方 gemma-4-12b では 76〜83% に混入し、``temperature=0.0``
#: の貪欲法でも再現した — サンプリングではなく出力分布そのものの性質。
_JA_INTERWORD_SPACE_RE = re.compile(r"[ぁ-んァ-ヶ一-龥][ 　]+[ぁ-んァ-ヶ一-龥]")

#: 語間空白を数える単位 (文 / 行)。
#:
#: 「1 箇所でも見つかれば崩れ」としていたため、**レイアウトの空白**を崩れと
#: 誤判定していた。実インシデント (2026-09-04 ライブ監査、Qwen3.8-27B の実応答
#: 451 件を再走査): 検出 4 件は **すべて誤検出**で、真の崩れは 0 件だった::
#:
#:     承知しました。お名前はおがわ ひろゆきさんですね。   ← 姓名の分かち書き
#:     塩・こしょう 適量                                   ← 材料表の項目と分量
#:     …195 文字です。散乱 を使わない指定に対し…          ← 引用語と助詞の間
#:
#: 誤検出のたびに :class:`FeedbackCollector` が正しいターンを
#: ``Turn marked failed`` として Level 0 経験に刻み、``fewshot_pool`` は
#: その手本を捨てる。名前を復唱するたびに学習信号が汚れる。
#:
#: 崩れは **出力分布の性質** なので 1 文の中で連続して現れる
#: (「私 は 東京 に 住んで います」)。対してレイアウトの空白は 1 文 (1 行) に
#: 高々 1 つしか現れない — 姓と名の間、項目と分量の間、引用語と助詞の間の
#: いずれも文中で 1 回きり。よって **数ではなく密度**、それも文単位の密度で
#: 分離する。上記 451 件で誤検出 4→0、gemma 型の合成例は引き続き検出。
_JA_SEGMENT_SPLIT_RE = re.compile(r"[。！？!?\n]+")

#: 1 文の中でこれ以上語間空白が並んだら崩れとみなす。
_JA_INTERWORD_SPACE_MIN_PER_SENTENCE = 2

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

    判定単位は **1 文 (1 行)** で、その中に語間空白が
    :data:`_JA_INTERWORD_SPACE_MIN_PER_SENTENCE` 箇所以上並んだときだけ崩れと
    みなす。1 文に 1 箇所しかない空白はレイアウト
    (姓名の分かち書き / 項目と分量 / 引用語と助詞) であって崩れではない
    (:data:`_JA_SEGMENT_SPLIT_RE` の実測を参照)。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    for segment in _JA_SEGMENT_SPLIT_RE.split(outside):
        # 「あ い う」は 2 箇所と数える (マッチは 1 文字重なる)。
        found = 0
        pos = 0
        while (match := _JA_INTERWORD_SPACE_RE.search(segment, pos)) is not None:
            found += 1
            if found >= _JA_INTERWORD_SPACE_MIN_PER_SENTENCE:
                return True
            pos = match.start() + 1
    return False


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
#: ラベルと数値の間に挟まる **ぼかし**。「直線距離は、およそ15kmです。」の
#: ように読点とヘッジが入るのが日本語では普通で、これを許さないと
#: 「<ラベル>は<数値>」の抽出が実会話でほとんど当たらない (実測 2026-08-27:
#: 監査 T12 の距離の言明「東京駅と横浜駅の直線距離は、およそ15kmです。」が
#: 1 件も取れなかった)。
_HEDGE_BEFORE_NUMBER = r"(?:[、,]\s*)?(?:およそ|約|おおよそ|ほぼ|だいたい|概ね)?\s*"

_LABELED_NUMBER_RE = re.compile(
    r"(?P<label>[0-9A-Za-z_ぁ-んァ-ヶーｦ-ﾟ一-龥]{2,24})"
    r"\s*(?:は|が|＝|=|:|：|\bis\b|\bwas\b|\bwere\b|\bare\b)\s*"
    + _HEDGE_BEFORE_NUMBER
    + r"(?P<num>[0-9０-９][0-9０-９,，.]*)",
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


#: システムが後付けした開示注記 (文末の ``(注: …)``)。
#:
#: 開示そのものは必要 (制約を破ったことを黙るのは隠蔽) だが、**記憶に残す
#: 本文ではない**。注記込みで保存すると、次のターンでモデルがそれを自分が
#: 書いた文の一部として読む。
#:
#: 実インシデント (2026-08-27 ライブ監査 T09-2): 本文 45 文字 + 注記 34 文字を
#: 保存した結果、「いま書いた文章は何文字でしたか。」に **81 文字** と答えた。
#: 同 T10-3 では「※会話の前半は参照できないため…」という別の注記が継続出力の
#: 末尾に焼き付いた。
#:
#: 「制約を破った」という信号自体は issue 台帳 (agent.issue_ledger) が持つので、
#: 履歴から落としても失われない。
#:
#: 形は 2 種:
#:
#: - ``(注: …)`` / ``（注：…）`` — 開示注記。括弧は 1 段までの入れ子を許す
#:   (「(注: 上限 300 字 (空白込み) を超過)」のような形で ``[^）)]*`` が途中で
#:   閉じて注記が残った)。連続して複数付くこともある (文字数 + 禁止語)。
#: - ``※会話の前半は…`` — 可視範囲の断り (``chat_service._TRUNCATED_HISTORY_GUIDANCE``
#:   の指示でモデルが末尾に足す)。行末まで。
#:
#: いずれも **文末に連なっている分だけ** を落とす。本文中の丸括弧には触らない。
_SYSTEM_NOTE_UNIT = (
    r"[（(]注[:：](?:[^（）()]|[（(][^（）()]*[）)])*[）)]"
    r"|※会話の前半は[^\n]*"
)
SYSTEM_NOTE_TAIL_RE = re.compile(
    r"(?:\s*(?:" + _SYSTEM_NOTE_UNIT + r"))+\s*$", re.DOTALL,
)


def strip_system_notes(text: str) -> str:
    """システムが後付けした開示注記を落とす (純粋関数)。

    :data:`SYSTEM_NOTE_TAIL_RE` の説明を参照。本文中の丸括弧は落とさない —
    対象は文末に連なる ``(注: …)`` と ``※会話の前半は…`` だけ。
    """
    return SYSTEM_NOTE_TAIL_RE.sub("", text or "").rstrip()


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


#: 「100km ではありません」のように、値の直後で打ち消す言い回し。値に **言及した**
#: ことと **採用した** ことを分ける。実測 (2026-08-22): 訂正のターンで
#: 「東京と大阪の直線距離は約370kmです。約100kmという値は事実と異なります。」と
#: 即座に反論したため、訂正値 100 が本文に現れて「採用」と誤判定された。
VALUE_REJECTION_RE = re.compile(
    r"(?:では?あり?ま?せん|ではなく|では無く|は誤り|は間違|"
    r"正しくありません|事実と異な|正確ではあ|ではないです|ではない)",
)


def value_was_adopted(response: str, values: set[str]) -> bool:
    """応答が ``values`` のいずれかを **自分の答えとして採った** か (純粋関数)。

    単なる出現では判定しない。値の直後が打ち消しなら、言及はしていても採用は
    していない (:data:`VALUE_REJECTION_RE` の説明を参照)。

    「ユーザーが主張した値をアシスタントが採ったか」は 2 箇所で要る:

    - 学習層 (:class:`~backend.free.agent.feedback.FeedbackCollector`) —
      採らなかったなら ``user_correction`` を撤回する (自分の正答を失敗として
      学習しないため)。
    - 記憶層 (``sleep.assertion_curator``) — 採らなかった値を world_fact として
      永続化しない。実インシデント (2026-08-27 ライブ監査): ユーザーの
      「いや、それは間違いです。答えは 63800 ですよ。」(誤) が
      ``mem.world.assertion.correct_answer`` として live になった。アシスタントは
      同じ会話で 3 回とも 63802 を維持していたのに、記録側だけが誤りを残していた。

    2 箇所で書き写すと必ず食い違うのでここに置く (pillar をまたぐ純粋関数の
    正準置き場)。
    """
    text = response or ""
    for value in values:
        start = 0
        while True:
            idx = text.find(value, start)
            if idx < 0:
                break
            tail = text[idx + len(value): idx + len(value) + 14]
            if not VALUE_REJECTION_RE.search(tail):
                return True
            start = idx + len(value)
    return False


#: 「<話題> は <値> です」型の断定。``ですよね`` / ``ですね`` / ``でしょう`` も含む
#: (ユーザーの同意要求はこの形で来る)。値は読点・句点を跨がない短い名詞句に限る。
_TOPIC_VALUE_RE = re.compile(
    r"(?P<topic>[^、。！!？?]{1,24})は\s*(?P<value>[^、。！!？?]{1,24}?)\s*"
    r"(?:です|だ|である|でした)(?:よ)?ね?(?:う)?[。．!！]?\s*$",
)

#: 話題側から落とす飾り (「さっき」「実は」など、同じ話題の別表記を作る語)。
_TOPIC_NOISE_RE = re.compile(r"^(?:でも|ただ|実は|さっき|たしか|確か|やっぱり|でも、)")


def assertion_topic_value(sentence: str) -> tuple[str, str] | None:
    """1 文を ``(話題, 値)`` に割る (純粋関数)。断定形でなければ ``None``。

    「日本の首都は大阪です」→ ``("日本の首都", "大阪")``。
    「日本の首都は東京です」→ ``("日本の首都", "東京")``。

    用途は **同じ話題に別の値が対置されたか** の判定で、値の型を問わない。
    数値だけを見る判定 (:func:`value_was_adopted` を数値集合で使う経路) は
    「日本の首都は大阪ですよね。」のような **数値を含まない誤主張** を
    素通りさせるため (2026-08-28 ライブ監査 T11-3)。
    """
    m = _TOPIC_VALUE_RE.search((sentence or "").strip())
    if m is None:
        return None
    topic = _TOPIC_NOISE_RE.sub("", m.group("topic").strip()).strip()
    value = m.group("value").strip()
    if not topic or not value:
        return None
    return topic, value


def contradicts_asserted_value(claim: str, reply: str) -> bool:
    """``reply`` が ``claim`` と **同じ話題に別の値** を対置したか (純粋関数)。

    「日本の首都は大阪ですよね。」に「日本の首都は東京です。大阪は……首都では
    ありません。」が返れば ``True``。同じ値を言い直しただけなら ``False``。

    出現の有無では判定できない — 反論する側は打ち消しながら相手の値に言及する
    ので、``大阪`` が本文にあること自体は採用の証拠にならない
    (:func:`value_was_adopted` と同じ理由)。話題を揃えて値を比べる。
    """
    claims: dict[str, str] = {}
    for raw in _SENTENCE_SPLIT_FOR_CLAIMS.split(claim or ""):
        pair = assertion_topic_value(raw)
        if pair is not None:
            claims[pair[0]] = pair[1]
    if not claims:
        return False
    for raw in _SENTENCE_SPLIT_FOR_CLAIMS.split(reply or ""):
        pair = assertion_topic_value(raw)
        if pair is None:
            continue
        topic, value = pair
        other = claims.get(topic)
        if other is not None and other != value:
            return True
    return False


#: 文分割 (``extractors.chat._SENTENCE_SPLIT_RE`` と同じ規則を pillar 非依存で持つ)。
_SENTENCE_SPLIT_FOR_CLAIMS = re.compile(r"(?<=[。．!！?？\n])")


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


#: ``_append_self_output_measurement`` がプロンプトへ差し込む実測値行の目印。
#: 抽出側 (``extract_measured_values``) と注入側 (chat_service) で共有する。
SYSTEM_MEASUREMENT_MARKER = "[システム計測]"

#: 「86 文字」「12 行」「200 語」/ "86 characters" のような 数値 + 単位。
#: en ロケールの注記 (chat_service の ``_localized``) も同じ抽出器で読めるよう
#: 英語単位も受け、キーは日本語単位へ正規化する。
_MEASURED_VALUE_RE = re.compile(
    r"(\d+)\s*(文字|行|語|characters?|chars?|lines?|words?)(?![A-Za-z])",
    re.IGNORECASE,
)
_UNIT_CANON = {
    "文字": "文字", "character": "文字", "characters": "文字", "char": "文字", "chars": "文字",
    "行": "行", "line": "行", "lines": "行",
    "語": "語", "word": "語", "words": "語",
}


def _iter_measured(text: str):
    for num, unit in _MEASURED_VALUE_RE.findall(text):
        yield int(num), _UNIT_CANON[unit.lower()]


def extract_measured_values(text: str) -> dict[str, set[int]]:
    """``[システム計測]`` 行から ``{単位: 実測値集合}`` を取り出す (純粋関数)。

    実測値を注入したターンだけ、その数値と応答の食い違いを検出できるように
    するための入力。マーカー行が無ければ空 dict。
    """
    out: dict[str, set[int]] = {}
    for line in (text or "").splitlines():
        if SYSTEM_MEASUREMENT_MARKER not in line:
            continue
        for num, unit in _iter_measured(line):
            out.setdefault(unit, set()).add(num)
    return out


def contradicts_measured_values(
    response: str, measured: dict[str, set[int]],
) -> str | None:
    """応答が注入済みの実測値と食い違う数値を述べていれば理由を返す (純粋関数)。

    「数えるのはコード、モデルは読み上げるだけ」という前提 (``[システム計測]``
    の注入) が破られたターンを検出する。実インシデント 2026-08-22 ライブ監査:
    「ちょうど100文字で要約して」に 86 文字で答えた次のターン、実測値
    (86 / 84 文字) を注入済みだったにもかかわらず「100文字です」と断定した。
    要求した数をそのまま復唱しており、**検証可能な量の虚偽申告**になっている。

    同じ単位で、注入した値のどれとも一致しない数値を述べたときだけ失敗とする
    (単位ごとに独立。注入していない単位は判定しない)。
    """
    if not measured or not response:
        return None
    for num, unit in _iter_measured(response):
        known = measured.get(unit)
        if known and int(num) not in known:
            return (
                f"asserted {num} {unit} but the injected measurement was "
                f"{sorted(known)}"
            )
    return None


#: 状態変更の **完了** を述べる言い回し。丁寧過去 (「削除しました」) と
#: 「〜済み」だけを見る。否定形 (「削除できていません」「削除していません」) は
#: ``しました`` に一致しないので自然に外れ、疑問形 (「削除しましたか」) は
#: 直後の ``か`` で除外する。
_COMPLETION_CLAIM_RE = re.compile(
    r"(?:削除|消去|作成|生成|書き込み|書込み|追記|保存|上書き|更新|移動|コピー"
    r"|リネーム|改名|実行|適用)"
    r"(?:を)?(?:し|いたし|完了し)?(?:ました(?![かがのけ])|済みです|済みました)",
)


def claims_completed_state_change(text: str) -> str | None:
    """本文が状態変更の完了を述べていれば、その語を返す (純粋関数)。

    「撃てるツールが無かった」ことをシステムが既に知っているターンで使う。
    知っている側と述べている側が食い違うので、真偽の推定ではなく **矛盾の検出**
    になる。実インシデント 2026-08-22 ライブ監査: ``tool_call_judge`` が
    ``Action blocked: file deletion requested but no tool can delete`` を出して
    いたターンで、応答は「削除しました。」。ファイルは実際に残っていた。
    """
    m = _COMPLETION_CLAIM_RE.search(text or "")
    return m.group(0) if m else None


#: 進捗ノートの断片。行頭アンカーの ``_TASK_LOG_LINE_RE`` (agent 側) では
#: 落とし切れない、ノート行が他のテキストと 1 行に連結された形も拾う
#: (実インシデント 2026-07-29 ライブ監査: 改行を含む本文の書込み依頼で、応答本文が
#: ``行2 行3' to the file E:\tmp\audit_r9.txt / … Written 16 bytes to …``
#: という内部タスク文の断片になった)。
#:
#: EvorefMem (注入側) と EvorefLoop (生成側) の両方から使うため core に置く。
_TASK_LOG_FRAGMENT_RE = re.compile(
    r"Written\s+\d+\s+bytes\s+to\s+\S"
    r"|\[(?:done|failed|skipped)\]\s",
)


def looks_like_task_log_residue(text: str) -> bool:
    """テキストが進捗ノートの残骸 (ユーザー向け本文ではない) かを判定する (純粋関数)。

    生成側では ``strip_task_log_scaffold`` で落とし切れなかった断片の検出に、
    注入側では **既に記憶へ入ってしまった残骸** を再注入しないために使う。

    実インシデント 2026-08-22 ライブ監査: 記録側の浄化漏れで
    ``- [done] Confirm the file E:\tmp\bs_audit.py has been deleted`` が
    STM ノートになり、次の会話でも ``(過去の記録)`` として注入されていた。
    記録側を直しても、既存ノートは寿命が尽きるまで残る。
    """
    return bool(_TASK_LOG_FRAGMENT_RE.search(text or ""))


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


#: アシスタントへの **依頼** の文末。疑問形ではないが、ユーザー自身の事実の
#: 表明でもない。
#:
#: 実インシデント (2026-08-18 ライブ監査): 「データ分析で**よく使う**可視化
#: ライブラリを 3 つ挙げてください。」が preference トリガ ``よく使う`` に
#: 一致し、依頼文がまるごと ``mem.preference.user`` の object として保存された。
#: 疑問符も「〜ですか」も無いため ``_QUESTION_ENDING_RE`` では拾えない。
_REQUEST_ENDING_RE = re.compile(
    # ``ください`` は直前の助詞を問わず文末の依頼マーカー。``て/で`` を必須に
    # していたため「アドバイスを**ください**。」が依頼として認識されず、
    # ``mem.personal.user states: 私の好みを踏まえて、通勤についてアドバイスを
    # ください。`` が live なファクトとして残った (2026-08-28 ライブ監査、
    # 実ストアで確認)。語彙を 1 つ足すのではなく助詞依存を外す
    # (「語彙列挙は必ず漏れる」— 2026-08-23 / 08-25 でも同じ形で再発している)。
    # 文末に錨を張るので「〜と言ってくださいました。」のような非依頼は掛からない。
    # ``もう一度`` / ``再度`` は **依頼動詞が省略された依頼**
    # (「〜をもう一度 (教えてください)。」)。日本語では常態だが動詞が無いため
    # 上のどの語尾にも当たらない。実インシデント (2026-08-30 ライブ監査の検証、
    # 実ストアで確認): 「私の名前、住所、職業、ペットをもう一度。」
    # 「私の勤務地と居住地をもう一度。」が **そのままファクト化** し、
    # 単値スロットゆえに正しい値を supersede して消した::
    #
    #     LIVE  mem.personal.pet        | ペットをもう一度。
    #     SUPER mem.personal.pet        | 柴犬を1匹飼っています。
    #     LIVE  mem.personal.occupation | 職業
    #     SUPER mem.personal.occupation | …SREになりました。
    #
    # **その問いの答えを、問うた瞬間に破壊していた**。実機の回答は
    # 「あなたの職業は会社員です」「飼っているペットは、猫です」(いずれも捏造)。
    # 文末アンカーなので「もう一度確認しました。」のような言明は掛からない。
    # 依頼動詞の列挙は **また漏れた**。実インシデント (2026-08-31 ライブ監査、
    # 実ストアで確認): 「私の誕生日を当ててみて。」が
    # ``mem.personal.birthday states: 私の誕生日を当ててみて。`` として live に
    # なった (``当ててみて`` はどの語にも当たらない)。この docstring 自身が
    # 「語彙列挙は必ず漏れる」と 3 回書いている。
    #
    # 語を足す代わりに **形** で取る: 文が「て形の動詞」で終わっていれば依頼。
    # 日本語の平叙文はて形で終わらない (て形は必ず後続節へ繋がる)。
    # 条件を 2 つ課して言明を巻き込まないようにする:
    #
    # - 直前が **活用語尾のひらがな** であること。「すべて。」「全て。」
    #   「初めて。」のような名詞・副詞を除く。
    # - 終端の句読点に **読点を含めない**。「名古屋市中区に住んでいて、」は
    #   節の途中であって依頼ではない (根拠文の絞り込みが節を残す形と衝突する)。
    r"(?:ください|下さい"
    r"|(?:して|で)(?:ほしい|欲しい)"
    r"|お願いします|願います"
    r"|もう一度|もう1度|もういちど|再度"
    r"|(?:教え|挙げ|見せ|出し|作っ|書い|説明し|列挙し|示し)て)"
    r"[。．.、,！!\s\"'」』）)]*\s*$"
    # **動詞が落ちた依頼** の 2 つめの形。``もう一度`` を語彙で足したのと同じ
    # 穴がまた開いた。実インシデント (2026-09-04 ライブ監査、実ストアで確認):
    # 「私の名前と住んでいる場所を**一言で**。」が
    # ``mem.personal.location states`` として live になり、単値スロットゆえに
    # 直前の正しい値を supersede して消した::
    #
    #     LIVE  mem.personal.location | 私の名前と住んでいる場所を一言で。
    #     SUPER mem.personal.location | 実は先月引っ越して、今は川崎に住んでいます。
    #
    # またしても **問いがその答えを破壊した**。``一言で`` を語彙に足すのでは
    # なく形で取る: ``を`` で目的語を示したまま **述語を持たずに副詞句で終わる**
    # 文は日本語では依頼 (「〜を一言で (言ってください)」「〜を簡潔に」
    # 「〜を 3 行で」)。値の言明は ``を`` の後に必ず述語が続く
    # (「猫を 2 匹飼っています」)。
    #
    # ``を`` と副詞句の間に **ひらがなを許さない** のは、活用語を挟む接続
    # (「〜を読むことが多い**ので**。」「〜を撮るのが好き**で**。」) を
    # 巻き込まないため。依頼の副詞句は体言 (一言 / 簡潔 / 3 行 / 日本語) で、
    # 活用語を挟まない。
    r"|(?<=を)[一-龥ァ-ヶーA-Za-z0-9０-９々〆ヵヶ・ー]{1,12}[でに]"
    r"[。．.！!\s\"'」』）)]*\s*$"
    # ``べ`` は ``すべて`` (副詞) と衝突するので、その 1 語だけ手前で外す。
    r"|(?:[いえきぎしちにびみりっん]|(?<!す)べ)[てで]"
    r"[。．.！!\s\"'」』）)]*\s*$",
)

#: 一人称マーカー。依頼形でもこれを伴う文は本人の事実表明を含みうるため
#: (例:「私はダークテーマが好きなので、そう設定してください。」)、依頼を
#: 理由に捨てない。ただし **一人称があるだけでは免除しない** —
#: :func:`_asserts_before_request` を参照。
_SELF_REFERENCE_RE = re.compile(r"(?:私|僕|俺|自分|わたし|ぼく|うち)")

#: 従属節の切れ目 (接続助詞 + 読点)。依頼文の中で「言明の節」と「依頼の節」を
#: 分ける境界として使う。読点を必須にするのは、体言の並列 (「AとB、Cを…」) を
#: 節の切れ目と誤認しないため。
#:
#: 実インシデント (2026-08-19 ライブ監査): 「私の好きな飲み物をもう一度教えて
#: ください。」が ``mem.personal.beverage states`` / ``mem.preference.beverage
#: prefers`` の 2 件として保存され、さらに本人の実際の言明
#: (「私の好きな飲み物は緑茶です」) と同じ (subject, predicate) に並んだため
#: 競合の当事者になり pending に滞留した。依頼形ゲート自体は存在したが、
#: 一人称を含むだけで無条件に免除していたため機能していなかった。
#:
#: ``で`` / ``て`` は入れない。「私の好きな飲み物を調べて、教えてください。」の
#: ような**依頼の中の依頼**まで免除してしまい、直そうとしている誤りが戻る。
#: 取りこぼす側 (「私は東京在住で、近くの店を教えてください。」) の損失は
#: 候補 1 件であり、ゴミを入れる損失より小さい。
_CLAUSE_BREAK_RE = re.compile(
    r"(?:ので|のに|から|ため|けれども|けれど|けど|ですが|だが|ますが)[、,]",
)


def _asserts_before_request(sentence: str) -> bool:
    """依頼文が、依頼節より **前の節** に本人の言明を含むかを判定する。

    一人称の有無だけで判定すると、一人称が依頼の**目的語**でしかない文
    (「私の好きな飲み物をもう一度教えてください。」) まで本人の表明として
    通ってしまう。言明は依頼とは別の節に立つはずなので、従属節の切れ目
    (:data:`_CLAUSE_BREAK_RE`) より前に一人称があることを要求する。

    - ``私はダークテーマが好きなので、そう設定してください。`` → ``ので、``
      より前に「私」がある → True (本人の表明を含む)
    - ``私の好きな飲み物をもう一度教えてください。`` → 節の切れ目が無い
      → False (依頼でしかない)
    - ``明日の予定を、私の代わりに調べてください。`` → 読点はあるが接続助詞
      ではなく、そもそも「私」は読点より後 → False
    """
    last_break = -1
    for m in _CLAUSE_BREAK_RE.finditer(sentence):
        last_break = m.end()
    if last_break < 0:
        return False
    return bool(_SELF_REFERENCE_RE.search(sentence[:last_break]))



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


def states_no_user_value(text: str) -> bool:
    """本文が **ユーザーについての値** を述べていないかを判定する (純粋関数)。

    :func:`carries_no_assertion` (問いだけか) より広く、**純粋な依頼**も
    「値ではない」と扱う。依頼はノートとしては残す価値がある (「明日までに
    資料をまとめてください。」は後から引きたい) ので
    ``carries_no_assertion`` は変えず、**ファクトの値として扱ってよいか**を
    問う場面だけこちらを使う:

    - ``[関連する記憶]`` へ「(personal_fact) mem.personal.X states: …」と
      **断定形**で並べるとき (:class:`MemoryInjector`)
    - ``[記憶の競合]`` の当事者として「旧/新」を付けて並べるとき
      (:func:`collect_review_groups`)

    実インシデント (2026-08-19 ライブ監査): 「私の好きな飲み物をもう一度
    教えてください。」が ``mem.personal.beverage states`` /
    ``mem.preference.beverage prefers`` の 2 件として保存され、本人の実際の
    言明と同じスロットに並んで **競合の当事者**になり pending に滞留した。
    抽出側には依頼形ゲートがあるが (``extractors/chat.py``)、それ以前に
    保存された行は残り続けるため読み出し側にも同じ判定が要る。
    実測 (2026-08-21、実ストア): active 146 件中 21 件が依頼形で、うち
    **14 件が飲み物スロット**に滞留していた。

    依頼節より前の節に本人の言明がある複合文 (「私は〜なので、〜して
    ください。」) は事実表明を兼ねるので落とさない
    (:func:`_asserts_before_request`)。
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text or "")]
    sentences = [s for s in sentences if s]
    if not sentences:
        return True
    return all(_is_non_assertive_sentence(s) for s in sentences)


def _is_non_assertive_sentence(sentence: str) -> bool:
    """1 文が「値の表明ではない」か (疑問形 または 純粋な依頼形)。"""
    if _INTERROGATIVE_TAIL_RE.search(sentence):
        return True
    if not _REQUEST_ENDING_RE.search(sentence):
        return False
    return not _asserts_before_request(sentence)


__all__ = [
    "asks_verbatim_excerpt",
    "is_payload_dump",
    "carries_no_assertion",
    "states_no_user_value",
    "conversational_numeric_claims",
    "find_superseded_claim",
    "count_response_lines",
    "match_enumeration_count",
    "match_enumeration_count_strict",
    "match_line_count",
    "violates_line_count",
    "count_list_items",
    "violates_enumeration_count",
    "value_was_adopted",
    "VALUE_REJECTION_RE",
    "has_boilerplate_closing",
    "has_broken_ja_spacing",
    "has_chinese_token_leak",
    "is_japanese_text",
    "labeled_numeric_claims",
    "strip_system_notes",
    "SYSTEM_NOTE_TAIL_RE",
    "is_query_echo",
    "strip_echoed_query",
    "strip_interrogative_sentences",
    "strip_discourse_prefix",
    "strip_first_person_topic",
    "match_length_directive",
    "violates_length_constraint",
    "ANSWER_ONLY_RE",
    "BULLET_FORM_RE",
    "ITEM_COUNT_RE",
    "match_output_form_directive",
    "violates_output_form",
    "has_verifiable_output_constraint",
    "length_disclosure_note",
    "count_belongs_to_another_subject",
    "match_word_limit",
    "count_response_words",
    "violates_word_count",
    "claimed_written_files",
    "unwritten_file_claims",
    "unwritten_file_disclosure_note",
]

#: 「<パス> に書き込みました」型の主張。**実際に書けたか** は別途 file system で
#: 確かめる (下記 :func:`unwritten_file_claims`)。
#:
#: 拡張子を持つトークンだけを対象にする。「メモに保存しました」のような対象が
#: ファイルでない言い回しを拾わないための境界。
_WRITE_CLAIM_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/])?[^\s、。「」『』（）()\[\]]*"
    r"[A-Za-z0-9_\-぀-ヿ一-鿿][.][A-Za-z0-9]{1,8})"
    r"\s*(?:[へに]|に対して)\s*"
    r"(?:[^\s、。]{0,8})?"
    r"(?:書き込|書き出|書きだ|保存|出力|作成|生成|エクスポート|セーブ)"
    r"(?:み|し|きま)?(?:ました|ます|た|できました)",
)


def claimed_written_files(response: str) -> list[str]:
    """応答が「書き込んだ」と述べているファイルパスを出現順に返す (純粋関数)。"""
    out: list[str] = []
    for m in _WRITE_CLAIM_RE.finditer(response or ""):
        path = m.group("path").strip()
        if path and path not in out:
            out.append(path)
    return out


def unwritten_file_claims(response: str, exists: "Callable[[str], bool]") -> list[str]:
    """主張されたのに **存在しない** ファイルパスを返す (純粋関数)。

    ``exists`` はパス文字列の実在判定 (呼出側が file system を注入する)。

    実インシデント (2026-08-30 ライブ監査 T17-6): 「CSV形式で、名前と年齢の
    3件のサンプルデータをファイルに出してください。」に
    **「sample_data.csv に書き込みました。」** と答えたが、その턴では書込み
    ツールが 1 度も実行されておらず (backend.log に Executing tool の行が無い)、
    ファイルも存在しなかった。次のターンの read_file は
    ``File not found: sample_data.csv`` で失敗している。

    ツール実行の有無ではなく **実体の有無** で判定する。長文生成や
    meta_cognitive の計画経路など、書込みが成立する経路は複数あり、層ごとに
    「書いたか」を集めると取りこぼす。同ターンに実体があれば主張は正しい。
    """
    return [
        path for path in claimed_written_files(response)
        if not exists(path)
    ]


def unwritten_file_disclosure_note(paths: "Sequence[str]") -> str:
    """存在しないファイルを主張したときに末尾へ足す開示文 (純粋関数)。

    黙って出すと「保存できたつもり」でユーザーが離れる。長さ制約の開示
    (:func:`length_disclosure_note`) と同じ扱いで、本文の外ではなく末尾注記に
    する (``strip_system_notes`` が記憶へ積む前に落とすので履歴は汚れない)。
    """
    if not paths:
        return ""
    listed = "、".join(paths)
    return (
        f"\n\n(注: {listed} は実際には作成されていません。"
        f"書き込みは行われませんでした)"
    )

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


#: 「A と答えましたが、正しくは B」型の訂正フレーム。A と B を取り出す。
#:
#: 謝罪語を伴わない訂正 (``_SELF_RETRACTION_RE`` が要求する形) を拾うためでは
#: なく、**訂正として退化しているか** を見るために使う。訂正は「誤った値 A を
#: 正しい値 B に置き換える」構造なので、**A == B なら訂正が成立していない**。
#: これは語形ではなく構造から決まるので、言い回しを列挙する必要がない。
#:
#: 実インシデント (2026-08-23 ライブ監査): 「1234 × 5678 の答えを「7,006,652」と
#: 答えましたが、**正しくは 7,006,652 であり**、ここは正しくありません
#: （※実際には計算ミスではなく正解ですが…）」が few-shot の手本に載っていた。
#: 謝罪語が無いため ``_SELF_RETRACTION_RE`` は非マッチ。
_DEGENERATE_CORRECTION_RE = re.compile(
    # ``before`` は「と答え」の直前の数値。あいだに別の数値を挟ませない
    # (挟ませると文頭の被演算子 (「1234 × 5678 の答え」の 1234) を拾ってしまい、
    #  退化していない訂正まで別値と判定して見逃す)。
    r"[「『\s]?(?P<before>[\d,．.０-９]{2,})[」』\s]?"
    r"[^。．\n\d０-９]{0,40}?"
    r"(?:と(?:答え|述べ|言い|回答))[^。．\n]{0,20}?"
    r"[、,][^。．\n]{0,20}?正しくは\s*"
    r"[「『]?(?P<after>[\d,．.０-９]{2,})",
)


def retracts_own_conclusion(text: str) -> bool:
    """応答が自分の結論を途中で撤回しているか (純粋関数)。"""
    if _SELF_RETRACTION_RE.search(text or ""):
        return True
    return degenerate_correction(text) is not None


def degenerate_correction(text: str) -> str | None:
    """「A と答えたが正しくは A」型の **成立していない訂正** を検出する。

    Returns:
        退化を表す ``"<value>"`` 文字列。該当しなければ ``None`` (純粋関数)。
    """
    for m in _DEGENERATE_CORRECTION_RE.finditer(text or ""):
        before = _normalize_number(m.group("before"))
        after = _normalize_number(m.group("after"))
        if before and before == after:
            return before
    return None


# ── 明示された出力長の指定と、その遵守判定 ──
#
# 指定の抽出はプロンプト注記 (``core.inference._char_limit_note``) と遵守判定
# (``violates_length_constraint``) の **両方** が必要とする。別々に書くと
# 「注記では拾うのに検証では拾わない」形の食い違いが静かに残るため、正規表現も
# 優先順位もここを SSOT とする。

#: 「10 文字以内」「200字以下」型の上限指定。数値と単位が隣接する形だけを拾う。
_CHAR_LIMIT_RE = re.compile(
    r"(\d{1,5})\s*(?:文字|字)\s*(?:以内|以下|まで)"
    r"|(?:within|under|at\s+most)\s+(\d{1,5})\s*(?:characters?|chars?)",
    re.IGNORECASE,
)

#: 「300字ちょうど」「ちょうど300文字で」型の **厳密指定**。上限指定
#: (_CHAR_LIMIT_RE) とは守り方が違う (足りない側も直す必要がある) ため分ける。
_CHAR_EXACT_RE = re.compile(
    r"(?:ちょうど|丁度|きっかり|正確に)\s*(\d{1,5})\s*(?:文字|字)"
    r"|(\d{1,5})\s*(?:文字|字)\s*(?:ちょうど|丁度|きっかり|で書|で説明|で答)"
    r"|exactly\s+(\d{1,5})\s*(?:characters?|chars?)",
    re.IGNORECASE,
)

#: 「「あ」を50回」型の反復回数指定。
_REPEAT_COUNT_RE = re.compile(
    r"(\d{1,4})\s*回\s*(?:だけ)?\s*(?:続けて|繰り返|repeat)"
    r"|(?:続けて|繰り返して)\s*(\d{1,4})\s*回"
    r"|repeat(?:ed)?\s+(\d{1,4})\s+times",
    re.IGNORECASE,
)


#: 「各項目 20 文字以内」「1 行は 20 文字以内」型の **1 項目あたり** の文字数上限。
#: 応答全体の上限 (:data:`_CHAR_LIMIT_RE`) とは測る対象が違う。
#:
#: 実インシデント (2026-08-28 ライブ監査 T07-5):
#: 「箇条書きでちょうど7項目、各項目20文字以内で書いてください。」に対し
#: 5 項目 (各 4〜5 文字) で答えたのに、開示注記は
#: 「20 文字以内の指定に対し、上の回答は 34 文字です」だった。34 は応答全文の
#: 長さで、per-item 指定を全文長として測っていた。しかも偽の違反が先に立つため、
#: 本当の違反 (7 項目指定に対し 5 項目) は検証にも開示にも出なかった。
#:
#: 数量詞に続く **単位の名詞を必須** にする ("120文字以内" の先頭 1 を
#: 「1 項目」と読まないため)。
_PER_ITEM_CHAR_LIMIT_RE = re.compile(
    r"(?:各|1|１|一)\s*(?:項目|行|つ|個|文|箇条|要素)\s*"
    r"(?:は|あたり|当たり|につき|ずつ|に)?\s*"
    r"([0-9０-９]{1,4})\s*(?:文字|字)\s*(?:以内|以下|まで)"
    r"|それぞれ\s*([0-9０-９]{1,4})\s*(?:文字|字)\s*(?:以内|以下|まで)",
)


def match_per_item_char_limit(text: str) -> int:
    """1 項目あたりの文字数上限を返す (純粋関数)。無ければ ``0``。"""
    m = _PER_ITEM_CHAR_LIMIT_RE.search(text or "")
    if not m:
        return 0
    raw = next(g for g in m.groups() if g).translate(_FULLWIDTH_DIGITS)
    return int(raw) if raw.isdigit() else 0


def _per_item_limit_spans(text: str) -> list[tuple[int, int]]:
    """per-item 上限指定の出現範囲 (全文上限との取り違え防止に使う)。"""
    return [m.span() for m in _PER_ITEM_CHAR_LIMIT_RE.finditer(text or "")]


#: 「40 words 以内」「40語以内」型の **語数** 上限。
#:
#: 文字数・行数・項目数と同じく数えれば分かる制約なのに、検証も開示も無かった。
#: 実インシデント (2026-08-29 ライブ監査 T28#3): 「同じ内容を、今度は英語で
#: **40 words 以内**で。」に **42 words** で答え、日本語の文字数制約では必ず
#: 付く開示注記が **1 つも付かなかった** (検証が日本語の文字数/行数に限定
#: されていたため)。
_WORD_LIMIT_RE = re.compile(
    r"([0-9０-９]{1,4})\s*(?:words?|語|単語|ワード)\s*(?:以内|以下|まで)"
    r"|(?:within|under|at\s+most|no\s+more\s+than)\s+"
    r"([0-9０-９]{1,4})\s*(?:words?)",
    re.IGNORECASE,
)

#: 語のトークン。英数字とアポストロフィ・ハイフンで 1 語 (don't / e-mail)。
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def match_word_limit(text: str) -> int:
    """発話中の語数上限を返す (純粋関数)。無ければ ``0``。"""
    m = _WORD_LIMIT_RE.search(text or "")
    if not m:
        return 0
    raw = next(g for g in m.groups() if g).translate(_FULLWIDTH_DIGITS)
    return int(raw) if raw.isdigit() else 0


def count_response_words(response: str) -> int:
    """応答の語数を数える (純粋関数)。

    システムが後付けした開示注記は除く (自分の注記で違反を作らないため)。
    """
    body = SYSTEM_NOTE_TAIL_RE.sub("", response or "")
    return len(_WORD_TOKEN_RE.findall(body))


def violates_word_count(query: str, response: str) -> str | None:
    """語数の上限を超えていれば理由を返す (純粋関数)。

    上限だけを見る (文字数の ``limit`` と同じ)。「ちょうど N words」は
    実用上ほぼ現れないので扱わない。
    """
    limit = match_word_limit(query)
    if limit <= 0:
        return None
    got = count_response_words(response)
    if got <= limit:
        return None
    return f"asked for at most {limit} words but the answer has {got} words"


def match_length_directive(text: str) -> tuple[str, int] | None:
    """発話に含まれる **応答全体** の出力長指定を ``(種別, 数)`` で返す。

    種別は ``exact`` (ちょうど N 文字) / ``repeat`` (N 回) / ``limit``
    (N 文字以内)。厳密指定を先に見るのは「300字ちょうど」が上限指定の
    パターンにも部分一致しうるため。指定が無ければ ``None``。

    「各項目 20 文字以内」型は **1 項目あたり** の指定なので全体の指定として
    返さない (:func:`match_per_item_char_limit` が扱う)。純粋関数。
    """
    spans = _per_item_limit_spans(text)
    for kind, pattern in (
        ("exact", _CHAR_EXACT_RE),
        ("repeat", _REPEAT_COUNT_RE),
        ("limit", _CHAR_LIMIT_RE),
    ):
        for m in pattern.finditer(text or ""):
            if any(s <= m.start() and m.end() <= e for s, e in spans):
                continue
            return kind, int(next(g for g in m.groups() if g))
    return None


#: 「「AI」という単語を使わずに」型の **禁止語**。引用符か「という単語/語/言葉」を
#: 要求して、「AI を使わずに実装して」(道具としての AI) と切り分ける。
_BANNED_WORD_RE = re.compile(
    r"[「『\"']\s*([^「」『』\"']{1,20}?)\s*[」』\"']\s*(?:という)?\s*"
    r"(?:単語|語|言葉)?\s*(?:を|は)?\s*使わ(?:ず|ない)"
    r"|([^\s、。「」]{1,20}?)\s*という\s*(?:単語|語|言葉)\s*(?:を|は)?\s*使わ(?:ず|ない)",
)

#: 「カタカナを使わずに」型の **禁止文字種**。名詞は閉じた集合で、判定は
#: Unicode の範囲で行う (語彙の列挙ではない)。
_BANNED_SCRIPT_RE = re.compile(
    r"(カタカナ|片仮名|ひらがな|平仮名|漢字|英語|アルファベット)"
    r"\s*(?:を|は)?\s*使わ(?:ず|ない)",
)

#: 文字種名 → その文字種にマッチする正規表現。
_SCRIPT_PATTERNS: dict[str, "re.Pattern[str]"] = {
    "カタカナ": re.compile(r"[ァ-ヶー]"),
    "片仮名": re.compile(r"[ァ-ヶー]"),
    "ひらがな": re.compile(r"[ぁ-ゖ]"),
    "平仮名": re.compile(r"[ぁ-ゖ]"),
    "漢字": re.compile(r"[一-鿿]"),
    "英語": re.compile(r"[A-Za-z]"),
    "アルファベット": re.compile(r"[A-Za-z]"),
}


def match_banned_content(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """発話中の「使うな」指定を ``(禁止語, 禁止文字種)`` で返す (純粋関数)。"""
    words: list[str] = []
    for m in _BANNED_WORD_RE.finditer(query or ""):
        token = next((g for g in m.groups() if g), "").strip()
        if token:
            words.append(token)
    scripts = [m.group(1) for m in _BANNED_SCRIPT_RE.finditer(query or "")]
    return tuple(dict.fromkeys(words)), tuple(dict.fromkeys(scripts))


#: 引用符として扱う開き / 閉じの対。禁止語がこの内側にあるだけなら「言及」。
_MENTION_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ('"', '"'), ("'", "'"), ("“", "”"))


def _strip_quoted_mentions(body: str, words: tuple[str, ...] | list[str]) -> str:
    """禁止語の **引用された出現** を本文から除いたコピーを返す (純粋関数)。

    「『散乱』の代わりに」のように鉤括弧で括られた出現は、その語を *使って*
    いるのではなく *指して* いる。括弧の外に 1 つでも出ていれば違反のままなので、
    「引用しつつ本文でも使う」応答は従来どおり検出される。
    """
    out = body
    for w in words:
        if not w:
            continue
        for open_q, close_q in _MENTION_QUOTE_PAIRS:
            out = out.replace(f"{open_q}{w}{close_q}", "")
    return out


def violates_banned_content(query: str, response: str) -> str | None:
    """禁止語・禁止文字種の指定を破っていれば理由を返す (純粋関数)。

    文字数・行数・個数と同じく **数えれば分かる** 制約なのに、検証も開示も
    無かった。実インシデント (2026-08-28 ライブ監査 T18-3):
    「カタカナを使わずに、コンピュータを説明してください。」に
    「コンピュータは、電気信号を使って計算や情報処理を行う機械です。」と答え、
    同じ会話の最後の自己申告 (T18-10) でもこの違反は挙がらなかった
    (挙がったのは文字数の 1 件だけ)。

    システムが後付けした開示注記は数えない (自分の注記で違反を作らないため)。
    """
    words, scripts = match_banned_content(query)
    if not (words or scripts):
        return None
    body = SYSTEM_NOTE_TAIL_RE.sub("", response or "").strip()
    if not body:
        return None
    # **言及と使用を区別する。** 「『散乱』の代わりに『バウンド』と言えます」は
    # 禁止語を *使って* いない — 引用して「これを避ける」と述べている。素の
    # 部分文字列判定はこれを違反と数え、指示に正しく従った応答へ
    # ``(注: 散乱 を使わない指定に対し、上の回答は使っています)`` という
    # 偽の注記を付けていた (2026-09-03 ライブ監査 T9#2)。
    # 鉤括弧などで囲まれた出現はメタ的な言及とみなして除外する。
    body_used = _strip_quoted_mentions(body, words)
    hits: list[str] = []
    hits.extend(f"the banned word {w!r}" for w in words if w and w in body_used)
    for name in scripts:
        pattern = _SCRIPT_PATTERNS.get(name)
        if pattern is not None and pattern.search(body):
            hits.append(f"banned script {name}")
    if not hits:
        return None
    return "asked not to use " + ", ".join(hits) + " but the answer contains it"


def violates_per_item_length(query: str, response: str) -> str | None:
    """1 項目あたりの文字数上限を破っていれば理由を返す (純粋関数)。

    指定が無い / 応答をリストとして数えられない場合は ``None`` (後続の全体長
    判定へ委ねず、per-item 指定があるなら全体長では測らない — 測る対象が
    違うので、数えられないなら黙って通す)。
    """
    limit = match_per_item_char_limit(query)
    if limit <= 0:
        return None
    items = _list_item_texts(response)
    if not items:
        return None
    longest = max(items, key=len)
    if len(longest) <= limit:
        return None
    return (
        f"asked for at most {limit} chars per item but the longest item is "
        f"{len(longest)}"
    )


def _list_item_texts(response: str) -> list[str]:
    """応答をリストとして項目本文の一覧に分解する (純粋関数)。

    箇条書き記号を落として本文だけを返す。リストと確信できなければ空リスト
    (:func:`count_list_items` と同じ判定を項目本文にも使う)。
    """
    body = SYSTEM_NOTE_TAIL_RE.sub("", response or "").strip()
    if not body:
        return []
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if _BULLET_LINE_RE.match(ln)]
    picked = bullets or lines
    if not bullets and len(picked) < _ENUM_MIN_LINES:
        return []
    return [_BULLET_MARKER_RE.sub("", ln).strip() for ln in picked]


def violates_length_constraint(query: str, response: str) -> str | None:
    """応答が発話中の文字数指定を破っていれば理由を返す (純粋関数)。

    指定を注記としてプロンプトへ渡してはいた (``_char_limit_note``) が、
    **守れたかを誰も見ていなかった**。そのため実インシデント 2026-08-22
    ライブ監査の「ちょうど100文字で要約して」→ 86 文字は ``turn_outcome``
    上は success として学習に入り、few-shot の手本にもなり得た。
    指定は本文にあり長さは数えるだけなので、真偽の推定ではなく **矛盾**
    であり、``claims_completed_state_change`` / ``contradicts_measured_values``
    と同格に扱える。

    数え方の曖昧さ (改行・空白を数えるか) は **どちらかの数え方で満たして
    いれば違反としない** ことで回避する。プロンプトへ注入する実測値
    (``chat_service._measure_text``) も総文字数と空白・改行を除いた数の
    両方を出しており、モデルに要求している基準と揃う。

    反復回数指定 (``repeat``) は数える対象の同定が要るため見ない。

    「各項目 20 文字以内」型は 1 項目ずつ測る (:func:`violates_per_item_length`)。
    """
    per_item = violates_per_item_length(query, response)
    if per_item is not None:
        return per_item
    directive = match_length_directive(query)
    if directive is None:
        return None
    kind, expected = directive
    if kind == "repeat" or not (response or "").strip():
        return None
    total = len(response)
    stripped = len("".join(response.split()))
    if kind == "limit":
        if min(total, stripped) <= expected:
            return None
        return (
            f"asked for at most {expected} chars but the answer is "
            f"{total} ({stripped} without whitespace)"
        )
    if expected in (total, stripped):
        return None
    return (
        f"asked for exactly {expected} chars but the answer is "
        f"{total} ({stripped} without whitespace)"
    )


# ---------------------------------------------------------------------------
# 出力形式の指定と、その **検証**
#
# 文字数と同じ立て付け。指定は本文にあり、守れたかは数えれば分かる。プロンプト
# へ注記を足すだけで検証しないと、破られたことに誰も気づかない
# (``violates_length_constraint`` の docstring 参照)。
#
# 正規表現の実体はここに置く。以前は ``core.inference`` にあり、注記の生成に
# しか使われていなかった。検証側 (ストリーム終端) と注記側 (プロンプト構築) が
# **同じ定義**を見るようにする — 語彙が 2 箇所に分かれると片方だけ直る。
# ---------------------------------------------------------------------------

#: 「数値だけ」「一言で」型の **答えだけを求める** 指定。
#:
#: 実インシデント (2026-08-14 ライブ監査 ターン15): 「摂氏 23 度は華氏何度ですか？
#: 数値だけ答えてください。」に対し 300 字超の解説を返した。
#: 「だけ」の後に **応答を指す動詞** を要求する。これが無いと
#: 「この値だけを使って計算して」(= 使う値の限定) まで拾ってしまう。
ANSWER_ONLY_RE = re.compile(
    r"(?:数値|数字|値|結論|答え|回答)\s*だけ\s*(?:を|で)?\s*"
    r"(?:答え|回答|示し|教え|出力|返し|書い|述べ|お願い)"
    r"|(?:一言|ひとこと|一語|単語)\s*(?:で|だけ)\s*"
    r"(?:答え|回答|示し|教え|言っ|いっ|まとめ|表現|お願い)"
    r"|(?<![A-Za-z])(?:just|only)\s+the\s+(?:number|value|answer)(?![A-Za-z])"
    r"|(?<![A-Za-z])in\s+one\s+word(?![A-Za-z])"
    r"|(?<![A-Za-z])(?:number|value|answer)\s+only(?![A-Za-z])",
    re.IGNORECASE,
)

#: 「箇条書きで」型の **出力形式** 指定。
#:
#: 実インシデント (2026-08-14 ライブ監査 ターン39): 「利点と欠点を、各 3 つずつ
#: 箇条書きで。」に対し「利点：A、B、C」と読点区切りの 1 行で返した。
BULLET_FORM_RE = re.compile(
    r"箇条書き|リスト形式|(?:マークダウン|markdown)\s*の?\s*リスト"
    r"|(?<![A-Za-z])bullet(?:\s+points?|\s+list)?(?![A-Za-z])"
    r"|(?<![A-Za-z])as\s+a\s+list(?![A-Za-z])",
    re.IGNORECASE,
)

#: 「各 3 つずつ」「3 個挙げて」型の個数指定 (箇条書き指定と併用されたときだけ使う)。
ITEM_COUNT_RE = re.compile(
    r"(?:各)?\s*(\d{1,2})\s*(?:つ|個|点|項目)\s*"
    r"(?:ずつ|ずつで|挙げ|書|列挙|箇条書き|リスト)"
    r"|(\d{1,2})\s*(?:items?|points?|bullets?)(?![A-Za-z])",
    re.IGNORECASE,
)

#: 箇条書きの 1 項目として数える行頭。Markdown のリストと日本語の中黒・番号。
#:
#: 中黒 (``・``) だけは **空白を要求しない**。日本語の箇条書きは ``・項目`` と
#: 詰めて書くのが普通で、空白必須にすると正しい出力を違反と誤判定する。
#: 一方 ``-`` / ``*`` / 番号は空白を必須にする — ``-1度`` や ``3.14`` のような
#: 数値表現をリスト項目と数えないため。
_BULLET_LINE_RE = re.compile(r"^\s*(?:・\s*|(?:[-*+]|\d{1,2}[.)、])\s+)\S")

#: :data:`_BULLET_LINE_RE` の記号部だけ (本文 1 文字を巻き込まずに剥がす)。
_BULLET_MARKER_RE = re.compile(r"^\s*(?:・\s*|(?:[-*+]|\d{1,2}[.)、])\s+)")

#: 「47 都道府県を全部列挙して」型の **列挙個数** 指定。
#:
#: :data:`ITEM_COUNT_RE` は数値が ``つ`` / ``個`` / ``点`` / ``項目`` に付く形しか
#: 拾わないので、数値が対象の名詞に直結する日本語 (``47都道府県`` / ``5教科``) を
#: 取りこぼす。しかも個数チェックは箇条書き指定との AND でしか走らなかった。
#:
#: 実インシデント (2026-08-27 ライブ監査): 「日本の47都道府県を、県庁所在地と
#: ともに全部列挙してください。」に **46 件** で答え (沖縄県が欠落)、検証も
#: 修復も走らなかった。数値は本文にあり、返ってきた行は数えられる。
#:
#: 「数値 + 短い名詞 + を + 列挙動詞」の形だけを採る。列挙動詞を必須にするのは、
#: 「1387 かける 46 は」「45 日後は」のような数値を含む別種のクエリを
#: 列挙要求と誤認しないため。
ENUMERATION_COUNT_RE = re.compile(
    r"(\d{1,3})\s*[^\s、。0-9]{1,8}?\s*を"
    r"[^。]{0,24}?(?:列挙|挙げ|並べ|書き出|リストアップ)",
)

#: 列挙個数の検証で「リストとして数えてよい」とみなす最小行数。
_ENUM_MIN_LINES = 3
#: 同上、1 項目とみなす行の最大長。これを超える行が多い応答は散文なので数えない。
_ENUM_ITEM_MAX_CHARS = 40
#: 非空行のうち、短い行が占める最小割合。
_ENUM_SHORT_LINE_RATIO = 0.8


#: 数量の一致が **既出の出力・別の対象への参照** であることを示す後続語。
#: 「その3行**のうち**」「3行**目**」「4つ**目**」は今回の出力への指定ではない。
#: 行数と個数で同じ語が効くので **共有する** (行数側だけに入れていたため、
#: 個数側は同じ形で誤検知していた — 下記 :func:`_count_is_reference` の説明)。
_COUNT_REFERENCE_TAIL = ("のうち", "の中", "目")
#: 同じく直前語。「**その**3行」「**先ほどの**3点」。
_COUNT_REFERENCE_HEAD = (
    "その", "この", "あの", "先ほどの", "さっきの", "上の", "上記の", "先の",
)

#: **配分**を示す直前語。「結末を2案、**それぞれ**1行で」は *1 項目あたり* 1 行
#: であって、応答全体を 1 行にしろという指定ではない。ここを拾っていたため、
#: 2 案を正しく 2 行で返した応答に
#: ``(注: 1 行の指定に対し、上の回答は 2 行です)`` という **偽の違反注記**が付き、
#: さらに ``constraint_repair`` が修復再生成を 1 往復むだに回していた
#: (2026-09-03 ライブ監査 T4#9)。
#:
#: 文字数側には既に per-item 判定 (:func:`match_per_item_char_limit` /
#: :func:`violates_per_item_length`) がある。行数側だけ非対称に欠けていた。
_COUNT_DISTRIBUTIVE_HEAD = (
    "それぞれ", "各", "夫々", "1つずつ", "一つずつ", "ひとつずつ", "1件ずつ",
)
#: 上記の語が数量の **直前の短い窓** に現れるか。「2案、それぞれ 1行で」の
#: ように読点や空白を挟む形が普通なので、末尾一致では拾えない。
_COUNT_DISTRIBUTIVE_RE = re.compile(
    "(?:" + "|".join(re.escape(w) for w in _COUNT_DISTRIBUTIVE_HEAD) + r")[\s、,]{0,3}$",
)


def _count_is_reference(src: str, match: re.Match[str]) -> bool:
    """数量の一致が **今回の出力への指定ではない** 文脈か (純粋関数)。

    行数 (:func:`match_line_count`) と個数 (:func:`match_enumeration_count`) で
    共有する。序数・部分指示・配分はどちらの単位でも「指定ではない」形で、
    行数側だけがガードを持っていたため個数側が同じ形で誤検知していた。

    実測 (2026-09-04): 「4つ目の観点をもう少し詳しく説明してください。」に
    3 項目の箇条書きで答えると ``asked to enumerate 4 items but the answer
    lists 3`` が立ち、**偽の開示注記** が付いたうえ ``constraint_repair`` が
    修復生成を 1 往復むだに回していた (行数側で 2026-09-03 T4#9 として
    潰したのと同じ形)。
    """
    tail = src[match.end():match.end() + 12]
    if tail.startswith(_COUNT_REFERENCE_TAIL):
        return True
    head = src[: match.start()]
    if head.endswith(_COUNT_REFERENCE_HEAD):
        return True
    return bool(_COUNT_DISTRIBUTIVE_RE.search(head))


#: 個数の直後に来ると **ぴったり N を求める指定ではなくなる** 語。
#:
#: - 上限 (``以内`` / ``以下`` / ``まで``): 下回っても違反ではない
#: - 概数 (``程度`` / ``くらい`` / ``ぐらい`` / ``ほど`` / ``前後``): 同上
#:
#: 実測 (2026-09-04): 「持ち物リストを箇条書きで10項目以内でまとめて
#: ください。」に 9 項目で答えると ``asked to enumerate 10 items but the
#: answer lists 9`` が立っていた。緩い側 (:data:`_ITEM_COUNT_IN_REQUEST_RE`)
#: だけがこの除外を持ち、strict 側の :data:`_ITEM_COUNT_BARE_RE` が持って
#: いなかったため、**箇条書き指定が付くと上限指定でも違反が立つ**。
_ITEM_COUNT_NOT_EXACT_TAIL = r"(?!\s*(?:以内|以下|まで|程度|くらい|ぐらい|ほど|前後))"

#: 「7 項目」のように **個数だけ** の形。:data:`ITEM_COUNT_RE` は後続の動詞
#: (``挙げ`` / ``書`` / ``列挙`` …) を要求するため、個数の直後で文が区切れると
#: 拾えない。箇条書き指定が別に立っているときだけ使う (単独で使うと
#: 「5 個のりんごを持っています」のような数量表現まで個数指定に読む)。
#:
#: 実インシデント (2026-08-28 ライブ監査 T07-5):
#: 「箇条書きでちょうど7項目、各項目20文字以内で書いてください。」の ``7項目``
#: の直後が読点で、個数指定として認識されず 5 項目の回答が違反にならなかった。
_ITEM_COUNT_BARE_RE = re.compile(
    r"(?:ちょうど|丁度|きっかり|正確に)?\s*(\d{1,2})\s*(?:つ|個|点|項目)"
    + _ITEM_COUNT_NOT_EXACT_TAIL,
)

#: **依頼文の中の個数** を形で取る規則。
#:
#: :data:`ITEM_COUNT_RE` は個数と列挙動詞が **隣接** し、かつ動詞が固定リスト
#: (``挙げ`` / ``書`` / ``列挙`` / ``箇条書き`` / ``リスト``) にあることを要求する。
#: 日本語の依頼はどちらの条件も普通に破るので、この形は必ず漏れる:
#:
#: - 「あなたにできないことを 3 つ**、具体的に**挙げてください。」
#:   → 読点と副詞が挟まって不一致
#: - 「記事全体のタイトル案を 5 つ**出して**ください。」
#:   → ``出し`` が語彙表に無い
#:
#: 実測 (2026-09-04 ライブ監査、実発話 23 件): 現行の判定が拾えたのは **8 件**
#: だけだった。拾えないと ``violates_enumeration_count`` も
#: ``match_output_form_directive`` の ``items`` も動かず、個数の指定は
#: **検証の対象にすらならない**。
#:
#: 語彙表を伸ばす代わりに **形** で取る: 個数の助数詞を含む文が
#: :data:`_REQUEST_ENDING_RE` の依頼形なら個数指定と読む。動詞の語彙に
#: 依存しないので、上の 2 型はどちらも同じ規則で拾える (実測 22/23)。
#:
#: 判定は **文単位**。「3 つの案があります。要点を教えてください。」のように、
#: 個数と依頼が別の文にあるものを結び付けないため。
#:
#: 上限・概数の指定 (「10 項目**以内**で」「3 つ**ほど**」) は除く
#: (:data:`_ITEM_COUNT_NOT_EXACT_TAIL`)。序数・部分指示・配分
#: (「4 つ**目**」「3 つ**のうち**」「**それぞれ** 3 つ」) は
#: :func:`_count_is_reference` で落とす。
_ITEM_COUNT_IN_REQUEST_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:つ|個|点|項目)" + _ITEM_COUNT_NOT_EXACT_TAIL,
)


def match_enumeration_count_strict(text: str) -> int:
    """列挙個数指定のうち、**個数の指定だと確信できる形** だけを返す。

    ``ITEM_COUNT_RE`` 等、列挙動詞や箇条書き指定を伴う形に限る。
    :func:`has_verifiable_output_constraint` はこちらを使う — あの述語は
    「応答をバッファして検証するか」を決めており、真になったターンは
    **本文が 1 文字も流れない**。緩い判定で真を増やすと、検証の利得が無い
    ターンからストリーミングを奪う (下記の実測を参照)。
    """
    m = ENUMERATION_COUNT_RE.search(text or "")
    if m:
        return int(m.group(1))
    m2 = ITEM_COUNT_RE.search(text or "")
    if m2:
        return int(next(g for g in m2.groups() if g))
    # 箇条書き / リスト形式の指定が立っているなら、個数だけの形も個数指定と読む。
    src = text or ""
    if BULLET_FORM_RE.search(src):
        for m3 in _ITEM_COUNT_BARE_RE.finditer(src):
            if _count_is_reference(src, m3):
                continue
            return int(m3.group(1))
    return 0


def match_enumeration_count(text: str) -> int:
    """発話中の列挙個数指定を返す (純粋関数)。無ければ ``0``。

    :func:`match_enumeration_count_strict` に、依頼形の文の中の助数詞
    (:data:`_ITEM_COUNT_IN_REQUEST_RE`) を足した広い判定。**検証・開示** の
    側で使う。

    バッファ判定 (:func:`has_verifiable_output_constraint`) には使わない。
    実測 (2026-09-04 ライブ監査、実発話 359 件): こちらをバッファ判定に
    使うとバッファ対象が 29 → 44 件へ増える一方、増えた 15 件で実際に
    違反が立つものは **0 件** だった。得るものが無いまま 15 件から
    ストリーミングを奪う (本 PC の実測で 1 ターン 30〜160 秒の無表示)。
    """
    strict = match_enumeration_count_strict(text)
    if strict:
        return strict
    # 依頼形の文に助数詞付きの個数があれば個数指定
    # (_ITEM_COUNT_IN_REQUEST_RE 参照)。上の規則の置き換えではなく **足す** —
    # 「3 日分の候補を 1 つずつ。」のような依頼形に当たらない形は上で拾う。
    for sentence in _SENTENCE_RE.findall(text or ""):
        if not _REQUEST_ENDING_RE.search(sentence):
            continue
        for m4 in _ITEM_COUNT_IN_REQUEST_RE.finditer(sentence):
            # 序数・部分指示・配分は指定ではない (_count_is_reference 参照)。
            if _count_is_reference(sentence, m4):
                continue
            return int(m4.group(1))
    return 0


def count_list_items(response: str) -> int:
    """応答を「1 行 1 項目のリスト」として数える (純粋関数)。

    リストと **確信できる** 形のときだけ数え、そうでなければ ``0`` を返す。
    誤検知は無駄な再生成になるので、散文を項目数 0 として扱う側に倒す。

    箇条書き行があればそれを数え、無ければ「短い行が大半を占める複数行」を
    リストとみなす。実インシデントの回答は ``北海道: 札幌市`` のような
    **記号の無い 1 行 1 項目** だった。
    """
    body = (response or "").strip()
    if not body:
        return 0
    bullets = [ln for ln in body.splitlines() if _BULLET_LINE_RE.match(ln)]
    if bullets:
        return len(bullets)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < _ENUM_MIN_LINES:
        return 0
    short = [ln for ln in lines if len(ln) <= _ENUM_ITEM_MAX_CHARS]
    if len(short) / len(lines) < _ENUM_SHORT_LINE_RATIO:
        return 0
    return len(lines)


#: 「N 行で」型の行数指定。``1行は20文字以内`` のような **1 行あたりの制約**
#: とは別物なので、``1行は`` の形は拾わない。
#:
#: 実インシデント (2026-08-27 ライブ監査 T09-5): 「横浜を3行で紹介してください。
#: 1行は20文字以内にしてください。」に **1 行** で答え、しかも文字数制約には
#: 出る開示注記が行数には出なかった。#502 の列挙個数の検証は箇条書きには
#: 効いていたが (同 T09-3 は 7 個ちょうど成功)、「N 行」という単位に無かった。
LINE_COUNT_RE = re.compile(
    r"(?<![0-9０-９])([0-9０-９]{1,3})\s*行(?:で|に|以内で)?"
    r"(?![^。]{0,6}(?:以内|まで)\s*[^。]{0,4}(?:文字|字))",
)


#: 「<X>の行数を N 行以内」の <X>。ここが応答を指す語なら今回の出力への指定、
#: そうでなければ **別の対象についての規約** を述べているだけ。
_COUNT_SUBJECT_RE = re.compile(
    r"([^\s、。,.]{1,12})の(?:行数|行の数|文字数|字数|文字量|長さ|サイズ)を?$",
)
#: 応答そのものを指す語。ここに載る語が主語なら分量指定として扱う。
_REPLY_SUBJECT_WORDS = frozenset({
    "回答", "答え", "返答", "応答", "出力", "本文", "文章", "説明", "記事",
    "要約", "text", "レス", "返事", "それ", "これ",
})


def count_belongs_to_another_subject(head: str) -> bool:
    """数量指定の直前 (``head``) を見て、その数量が **応答以外** の属性か。

    「1ファイルの行数を500行以内にする」の ``500行`` はファイルについての規約で、
    今回の応答を 500 行にしろという依頼ではない。いっぽう「回答の行数を3行以内に」
    は応答への指定なので拾う。判定は帰属先の語だけで決まる。

    **この判定は 2 箇所が必要とする** — 制約検証 (:func:`match_line_count`) と
    router の分量判定 (``agent.router.requests_long_output`` /
    ``requests_short_output``)。同じ関係を 2 実装に分けると片方だけ直る
    (本リポジトリで繰り返し起きている「食い違った複製」)。ここを唯一の出所とする。

    実インシデント:

    - 2026-08-30 ライブ監査 T04#2: 「関数の行数を50行以内に収めることです。」への
      承諾 1 文に ``(注: 50 行の指定に対し、上の回答は 1 行です)`` が付いた
      (検証側で発生)。
    - 2026-08-31 ライブ監査 T05#2: 「1ファイルの行数を500行以内にすることです。」が
      **long_form へ振られ**、``どのような内容・主題の文書をご希望ですか？`` と
      返した (router 側で発生 — 検証側だけ直していたため残っていた)。
    """
    subject = _COUNT_SUBJECT_RE.search(head or "")
    return bool(subject) and subject.group(1).lower() not in _REPLY_SUBJECT_WORDS


def match_line_count(text: str) -> int:
    """発話中の行数指定を返す (純粋関数)。無ければ ``0``。

    **参照は指定ではない。** 「その 3 行のうち、2 行目だけを 10 文字に縮めて
    ください。」は *前ターンの出力* を指しているだけで、今回の応答に 3 行を
    求めてはいない。これを指定として拾っていたため、1 行で答えるのが正しい
    ターンに `(注: 3 行の指定に対し、上の回答は 1 行です)` という **偽の違反
    注記** が付いた (2026-08-29 ライブ監査 T04#4)。序数 (「2 行目」) も同様に
    行数の指定ではない。

    **他人の行数も指定ではない。** 「2つ目の基準は、関数の行数を50行以内に
    収めることです。」はコードレビュー規約の申告であって、応答を 50 行にしろ
    という依頼ではない。ここを拾ったため、承諾の 1 文に
    `(注: 上の回答は 30 文字です。50 行の指定に対し、上の回答は 1 行です)` と
    いう二重の偽注記が付いた (2026-08-30 ライブ監査 T04#2)。行数の帰属先が
    応答を指す語 (:data:`_REPLY_SUBJECT_WORDS`) でなければ指定ではない。
    """
    src = text or ""
    for m in LINE_COUNT_RE.finditer(src):
        # 「1行は20文字以内」は 1 行あたりの制約であって行数の指定ではない。
        tail = src[m.end():m.end() + 12]
        if tail.startswith(("は", "あたり", "当たり", "につき")):
            continue
        # 参照 (「その3行」「3行目」) と配分 (「2案、それぞれ 1行で」) は
        # 指定ではない。個数側と共有の判定 (_count_is_reference 参照)。
        if _count_is_reference(src, m):
            continue
        head = src[: m.start()]
        if count_belongs_to_another_subject(head):
            continue
        raw = m.group(1).translate(_FULLWIDTH_DIGITS)
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 0


def count_response_lines(response: str) -> int:
    """応答の行数を数える (純粋関数)。

    空行は数えない (段落の区切りであって行ではない)。システムが後付けした
    開示注記も除く — 数えた結果を開示に使うので、注記を数えると自分の注記で
    行数が増える。
    """
    body = SYSTEM_NOTE_TAIL_RE.sub("", response or "")
    return len([ln for ln in body.splitlines() if ln.strip()])


def violates_line_count(query: str, response: str) -> str | None:
    """行数の指定と食い違うときだけ理由を返す (純粋関数)。

    列挙個数 (``violates_enumeration_count``) は不足だけを見るが、行数は
    **過不足の両方** を見る。「3 行で」は文字数の ``exact`` と同じく
    ぴったりを求める指定で、4 行返すのも指定違反になる。
    """
    wanted = match_line_count(query)
    if wanted <= 0:
        return None
    got = count_response_lines(response)
    if got <= 0 or got == wanted:
        return None
    return f"asked for {wanted} lines but the answer has {got}"


def violates_enumeration_count(query: str, response: str) -> str | None:
    """列挙個数の指定に **足りない** ときだけ理由を返す (純粋関数)。

    超過は見ない — 「各 3 つずつ」のようにグループ化された正当な出力を
    違反と誤判定するため (:func:`violates_output_form` が同じ理由で剰余判定に
    している)。足りない側だけが実インシデント (46/47) の形。

    **箇条書き指定が個数を伴うターンは見ない。** その形は
    :func:`violates_output_form` の剰余判定が過不足の両方を見ており、こちらも
    立つと ``violation_reason`` が同じ違反を 2 回並べる (修復指示にもそのまま
    載る)。個数の抽出は共通化したので、どちらが立つかは指定の形だけで決まる。
    """
    directive = match_output_form_directive(query)
    if directive is not None and directive["bullet"] and directive["items"]:
        return None
    wanted = match_enumeration_count(query)
    if wanted <= 0:
        return None
    got = count_list_items(response)
    if got <= 0 or got >= wanted:
        return None
    return f"asked to enumerate {wanted} items but the answer lists {got}"


#: ``answer_only`` を破ったと **確信できる** 長さ。
#:
#: 「答えだけ」への正解は高々 1 文なので、これを大きく超えたら解説が付いている。
#: 閾値をきつくしないのは、**誤検知が無駄な再生成を生む**ため。実インシデントは
#: 300 字超で、この棒の 3 倍以上あった。
_ANSWER_ONLY_MAX_CHARS = 80


def match_output_form_directive(text: str) -> dict[str, int | bool] | None:
    """発話に含まれる出力形式の指定を返す (純粋関数)。

    Returns:
        ``{"answer_only": bool, "bullet": bool, "items": int}``。
        どの指定も無ければ ``None``。``items`` は箇条書き指定と併用された
        個数指定 (無ければ 0)。
    """
    t = text or ""
    answer_only = bool(ANSWER_ONLY_RE.search(t))
    bullet = bool(BULLET_FORM_RE.search(t))
    if not (answer_only or bullet):
        return None
    # 個数の抽出は :func:`match_enumeration_count_strict` に寄せる。ここが
    # ``ITEM_COUNT_RE`` を直接引いていたため、**同じ個数を 3 箇所が別々の
    # 規則で取っていた** (この関数 / strict / 緩い側)。実害は超過の取りこぼし:
    # 「箇条書きでちょうど7項目、各項目20文字以内で書いてください。」に
    # 9 項目で答えても ``items`` が 0 のまま剰余判定が走らず、
    # ``violates_enumeration_count`` は不足しか見ないので誰も報告しなかった
    # (2026-09-04 実測)。
    items = match_enumeration_count_strict(t) if bullet else 0
    return {"answer_only": answer_only, "bullet": bullet, "items": items}


def violates_output_form(query: str, response: str) -> str | None:
    """応答が発話中の **形式指定** を破っていれば理由を返す (純粋関数)。

    判定はすべて数えるだけで、真偽の推定を含まない。誤検知は無駄な再生成に
    なるので、どの規則も「破ったと確信できる」側に倒してある。
    """
    directive = match_output_form_directive(query)
    if directive is None:
        return None
    body = (response or "").strip()
    if not body:
        return None

    bullet_lines = [ln for ln in body.splitlines() if _BULLET_LINE_RE.match(ln)]

    if directive["bullet"] and not bullet_lines:
        return "asked for a bullet list but the answer has no list items"

    items = int(directive["items"] or 0)
    if directive["bullet"] and items and bullet_lines:
        # 「各 N つずつ」は複数グループに分かれることがあるので、総数が N の
        # 倍数なら満たしているとみなす (2 グループ × 3 項目 = 6 行)。
        if len(bullet_lines) % items != 0:
            return (
                f"asked for {items} items but the answer has "
                f"{len(bullet_lines)} list items"
            )

    if directive["answer_only"] and len(body) > _ANSWER_ONLY_MAX_CHARS:
        return (
            f"asked for the answer only but the reply is {len(body)} chars"
        )
    return None


def has_verifiable_output_constraint(query: str) -> bool:
    """発話に **決定論で検証できる** 出力制約が含まれるか (純粋関数)。

    ``True`` のターンだけ、応答をバッファして検証・修復する価値がある
    (:mod:`backend.free.api.chat.chat_stream_common` の repair 経路)。
    """
    q = query or ""
    return (
        match_length_directive(q) is not None
        or match_output_form_directive(q) is not None
        # 「47 都道府県を全部列挙して」型。箇条書き指定が無いので
        # match_output_form_directive では拾えないが、数値は本文にあり
        # 返ってきた行は数えられる (2026-08-27 ライブ監査: 46/47 を見逃した)。
        #
        # **strict 版を使う。** この述語が真のターンは本文が 1 文字も流れない
        # ので、緩い判定で真を増やすと検証の利得が無いターンから
        # ストリーミングを奪う (match_enumeration_count の説明の実測を参照)。
        or match_enumeration_count_strict(q) > 0
        or match_line_count(q) > 0
        # 「「AI」という単語を使わずに」「カタカナを使わずに」も数えれば分かる
        # (2026-08-28 ライブ監査 T18-3)。
        or any(match_banned_content(q))
        or match_per_item_char_limit(q) > 0
        # 「40 words 以内で」も数えれば分かる。日本語の文字数だけを見ていた
        # ため、英語の語数違反は検証も開示もされなかった
        # (2026-08-29 ライブ監査 T28#3: 40 words 指定に 42 words)。
        or match_word_limit(q) > 0
    )


def length_disclosure_note(query: str, response: str) -> str:
    """文字数指定を満たせなかったときに末尾へ足す開示文 (純粋関数)。

    黙って出すと「制約違反の隠蔽」になり、後続ターンの自己申告
    (「いま書いた説明は何文字でしたか？」) とも食い違う。ストリームの開示
    フィルタ (``LengthDisclosureFilter``) と修復経路
    (``api.chat.constraint_repair``) の両方がこれを使う — 文言を 2 箇所に
    書くと片方だけ直る。

    **指定値も併記する。** 実測値だけだと、ユーザーは自分が何文字と言ったかを
    覚えていないと過不足が判断できない (2026-08-25 ライブ監査: 「ちょうど50
    文字で」への 45 文字の回答に「上の回答は 45 文字です」とだけ出た)。
    """
    measured = len((response or "").strip())
    directive = match_length_directive(query or "")
    prefix = chr(10) * 2
    parts: list[str] = []
    # 「各項目 20 文字以内」型は 1 項目あたりの指定。全文の長さを開示すると
    # 守れている出力を破ったように見せる (2026-08-28 ライブ監査 T07-5:
    # 各項目 4〜5 文字の箇条書きに「20 文字以内の指定に対し 34 文字」と出た)。
    per_item_limit = match_per_item_char_limit(query or "")
    if per_item_limit > 0:
        items = _list_item_texts(response)
        longest = max((len(t) for t in items), default=0)
        if longest > per_item_limit:
            parts.append(
                f"1 項目 {per_item_limit} 文字以内の指定に対し、"
                f"最長の項目は {longest} 文字です",
            )
            _record_constraint_issue(
                f"1 項目 {per_item_limit} 文字以内の指定に対し最長 {longest} 文字",
            )
    elif directive is None and match_word_limit(query or "") <= 0:
        # 語数指定のターンで文字数を併記すると、訊かれていない単位の数字が
        # 先に立って読み手を混乱させる (「40 語以内」に「308 文字です」)。
        parts.append(f"上の回答は {measured} 文字です")
    if directive is not None:
        kind, expected = directive
        unit = "文字以内" if kind == "limit" else "文字ちょうど"
        parts.append(
            f"{expected} {unit}の指定に対し、上の回答は {measured} 文字です",
        )
        # 制約を満たせなかったことを不首尾の台帳へ落とす。この関数は開示文の
        # 唯一の生成点 (ストリームフィルタと修復経路の両方が使う) なので、
        # ここで記録すれば経路の取りこぼしが出ない。
        # 監査では「いままでの出力形式の指定に、全部従えましたか。」に
        # 「はい、すべて従いました。」と答えていた。
        _record_constraint_issue(f"{expected} {unit}の指定に対し {measured} 文字")
    # 行数の指定があれば併記する。文字数だけ開示して行数を黙っていると、
    # 「3 行で」に 1 行で答えた事実がユーザーにもモデルにも残らない
    # (2026-08-27 ライブ監査 T09-5)。
    wanted_lines = match_line_count(query or "")
    if wanted_lines > 0:
        got_lines = count_response_lines(response)
        if got_lines != wanted_lines:
            parts.append(
                f"{wanted_lines} 行の指定に対し、上の回答は {got_lines} 行です",
            )
            _record_constraint_issue(
                f"{wanted_lines} 行の指定に対し {got_lines} 行",
            )
    # 列挙個数も併記する。長さの側だけ開示すると、「ちょうど 7 項目」に 5 項目で
    # 答えた事実が残らない (2026-08-28 ライブ監査 T07-5)。
    wanted_items = match_enumeration_count(query or "")
    if wanted_items > 0:
        got_items = count_list_items(response)
        if 0 < got_items < wanted_items:
            parts.append(
                f"{wanted_items} 項目の指定に対し、上の回答は {got_items} 項目です",
            )
            _record_constraint_issue(
                f"{wanted_items} 項目の指定に対し {got_items} 項目",
            )
    # 語数の上限も併記する。日本語の文字数には必ず注記が付くのに、英語の語数は
    # 検証も開示も無く、40 words 指定に 42 words で答えても黙っていた
    # (2026-08-29 ライブ監査 T28#3)。
    word_limit = match_word_limit(query or "")
    if word_limit > 0:
        got_words = count_response_words(response)
        if got_words > word_limit:
            parts.append(
                f"{word_limit} 語以内の指定に対し、上の回答は {got_words} 語です",
            )
            _record_constraint_issue(
                f"{word_limit} 語以内の指定に対し {got_words} 語",
            )
    # 禁止語・禁止文字種も開示する。守れなかったことを黙っていると、後続ターンの
    # 自己申告 (「守れなかった制約を挙げて」) から丸ごと落ちる
    # (2026-08-28 ライブ監査 T18-10 は文字数の 1 件しか挙げなかった)。
    banned_words, banned_scripts = match_banned_content(query or "")
    if banned_words or banned_scripts:
        body = SYSTEM_NOTE_TAIL_RE.sub("", response or "").strip()
        broken = [w for w in banned_words if w and w in body]
        broken += [
            name for name in banned_scripts
            if (pat := _SCRIPT_PATTERNS.get(name)) is not None and pat.search(body)
        ]
        if broken:
            joined = "・".join(broken)
            parts.append(f"{joined} を使わない指定に対し、上の回答は使っています")
            _record_constraint_issue(f"{joined} を使わない指定に違反")
    if not parts:
        return ""
    return prefix + "(注: " + "。".join(parts) + ")"


def _record_constraint_issue(detail: str) -> None:
    """制約違反を不首尾の台帳へ記録する (失敗は握る)。

    ``text_quality`` は agent pillar に依存しない純粋関数の置き場なので、
    import は関数内に閉じる。台帳が無い経路 (単体テスト等) では no-op。
    """
    try:
        from backend.free.core.verifier_events import record_verifier_hit

        record_verifier_hit(classify_constraint_verifier(detail))
    except Exception:  # noqa: BLE001
        pass
    try:
        from backend.free.agent.issue_ledger import record_current_issue

        record_current_issue("constraint_violated", detail)
    except Exception:  # noqa: BLE001 - 記録の失敗で開示自体を止めない
        pass


def classify_constraint_verifier(detail: str) -> str:
    """開示注記の文面から検証器 id (``constraint.*``) を決める (純粋関数)。

    規則台帳の計数 (f_03 §3.5.1) 用。文面は ``length_disclosure_note`` /
    ``violation_reason`` が作るもので、種別ごとに固有の語を含む。
    """
    text = detail or ""
    if "使わない指定" in text or "banned" in text:
        return "constraint.banned"
    if "行" in text or "lines" in text:
        return "constraint.lines"
    if "項目" in text or "個" in text or "items" in text or "enumerate" in text:
        return "constraint.items"
    if "語" in text or "words" in text:
        return "constraint.words"
    if "箇条" in text or "形式" in text or "format" in text:
        return "constraint.form"
    return "constraint.length"
