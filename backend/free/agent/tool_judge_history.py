"""履歴リコールクエリの検出と縮約 (純粋関数)

``search_history`` へ渡す内容キーワードの抽出と、履歴参照語の種別判定
(近接リコールか長距離リコールか) を担う。HistoryManager の字句照合は疑問文
全文にマッチしないため、縮約はこの層の責務。
"""

from __future__ import annotations

import re

from backend.free.agent.tool_judge_args import quoted_spans
from backend.free.core.intent_vocab import (
    ASSISTANT_REFERENCE_STEMS,
    KA_QUESTION_END_RE,
    PAST_RECALL_TAIL_RE,
    QUESTION_END_RE,
    USER_REFERENCE_STEMS,
    has_history_recall_keyword,
    is_request_sentence,
    split_sentences,
)
from backend.free.core.locale_patterns import has_japanese_script
from backend.free.core.response_dates import literal_date_ranges, nearest_date
from backend.free.core.temporal_deixis import DAY_OFFSETS, alternation, kanji_terms
from backend.free.core.script_ranges import (
    HIRAGANA,
    KANJI,
    KANJI_MARKS,
    KATAKANA_BLOCK,
)

# 時系列順序指定を含む履歴クエリの検出 (「一番最初に」「最後に」等)。
# aux が合成する小さい limit (例: limit=1) は字句スコア最上位への
# 切り詰めであり時系列意味論を持たないため、順序指定クエリでは limit を
# ハンドラ既定値まで引き上げ、turn# 付きの全マッチターンを digest に渡す。
_ORDERED_HISTORY_QUERY_RE = re.compile(
    r"最初|最後|何番目|何回目|直近|first|last|earliest|latest",
    re.IGNORECASE,
)

# builtin._make_search_history の limit 既定と同期
_HISTORY_SEARCH_DEFAULT_LIMIT = 10

# 順序リコール質問から search_history 用の内容キーワードを抽出するための定義。
# 「この会話で一番最初に計算させた問題は何？」→「計算」。
# 除去対象の scaffolding フレーズ (self-reference / 複合順序語)。単純な文字
# クラス抽出では「一番最初」等が 1 つの漢字ランに連結するため、先にフレーズ
# 単位で除去してから内容ランを取り出す。
_ORDER_QUERY_SCAFFOLD_RE = re.compile(
    r"今までの(?:会話|やり取り)|今日の(?:追加分の)?会話|今回の(?:追加分の)?会話"
    r"|前回の会話|この会話|このやり取り|その会話"
    r"|過去の(?:会話|やり取り)|以前の会話|会話履歴"
    r"|一番最初|一番最後|何番目|何回目"
    # 「何 + 助数詞」は数量を問う骨組みであって話題語ではない。ラン抽出の前に
    # 落とさないと、``何`` だけがストップワードとして剥がれて助数詞が後続の
    # 動詞語幹と融合し、実在しない語が検索クエリに載る (実インシデント
    # 2026-09-03 ライブ監査 T01#6: 「猫は何匹飼っているか覚えていますか？」が
    # ``匹飼`` になり ``No results found for: 匹飼``。話題語の ``猫`` は
    # 1 文字のため落ちており、残ったのは融合語だけだった)。
    r"|何[匹個枚本台人回冊杯件歳番度種文字行]",
)
# 内容ラン (漢字 / カタカナ / ラテン / 数字。ひらがなの助詞・活用語尾は自然に
# 脱落する)。
#
# ひらがなを語の一部として取り込む案は採らない。送り仮名 (食べ物) と助詞・活用
# 語尾 (私が今日 / 話した / 見た映画) を辞書無しで区別できず、取り込むと
# 「が今日ハマってるって話した食べ物」のような **1 個の巨大な融合語** になる。
# 融合語は照合側の定足数を確実に落とすため、分割 (食べ物 → 食 / 物) より害が
# 大きい。語の分断は照合側の定足数を緩めることで受け止める
# (``history.history_manager._text_matches_query``)。
# 例外は **送り仮名 1 文字を挟んだ漢字ラン** (飲み物 / 食べ物 / 読み方) だけ。
# 分割すると 飲/物・食/物 のように 1 文字語しか残らず、
# :data:`_ORDER_QUERY_MIN_TERM_LEN` で全滅して **内容語ゼロ → 生クエリへ
# フォールバック** する。実インシデント (2026-08-29 ライブ監査 T17#1):
# 「以前の会話で、私の**飲み物**の好みについて話したはずです。探してください。」
# が ``飲``/``物``/``好`` を全て 1 文字で落とし、search_history に
# **発話文が丸ごと** 渡って ``success=False (empty result)``。T02 で
# 「好きな飲み物はコーヒー→紅茶」と明示的に会話しているのに空振りした。
# 助詞 (が/は/を/に/へ/と/で/の/も/や/か/ね/よ) と活用語尾 (た/て/だ/ん) は除外する。
# 含めると「私が今日」「見た映画」のような **別語の融合** が起き、上のコメントが
# 警告している害 (照合の定足数が落ちる) がそのまま出る。
_ORDER_QUERY_CONTENT_RE = re.compile(
    f"[{KANJI}{KATAKANA_BLOCK}{KANJI_MARKS}a-zA-Z0-9]+"
    f"(?:(?![がはをにへとでのもやかねよたてだん])[{HIRAGANA}]"
    f"[{KANJI}{KATAKANA_BLOCK}{KANJI_MARKS}]+)*",
)

#: 格助詞。1 文字ランの直後に来れば、そのランは動詞語幹ではなく名詞
#: (話題語) と見なす。``_reduce_ordered_history_query`` の 1 文字語
#: フォールバックで、どれを採るかの決定に使う。
_CASE_PARTICLES = frozenset("はがをにへとのも")

#: 1 文字の内容ランは検索語として意味を持たない (良 / 久 / 泣 / 人 / 勧)。
#: 照合側が 2 文字未満を捨てるため効きもしないのに、クエリ文字列だけを膨らませる
#: (実インシデント 2026-08-16 ライブ監査 ターン5:
#: ``昨日見 映画 良 久 泣 人 勧`` の 7 語のうち 5 語が 1 文字だった)。
_ORDER_QUERY_MIN_TERM_LEN = 2
# 内容ランのうち scaffolding とみなして落とす語 (質問・順序・自己参照の骨組み)。
_ORDER_QUERY_STOPWORD_RUNS = frozenset({
    "会話", "一番", "最初", "最後", "直近", "以前", "前回", "今日", "今回", "今",
    # 時点の scaffolding。「今日」だけが登録されていたため「昨日見た映画が…」が
    # ``昨日見`` という壊れた融合語になっていた (2026-08-16 ライブ監査 ターン5)。
    "昨日", "明日", "昨夜", "今朝", "先日", "最近", "先週", "先月",
    "問題", "質問", "内容", "話題", "話", "何", "誰", "私", "貴方", "君", "僕",
    "俺", "覚", "番目", "回目", "先",
    # 位置の骨組み。``先`` / ``前回`` / ``以前`` は登録済みだったが裸の ``前``
    # (「前の会話で車の話をしましたか？」) だけが漏れており、1 文字語の
    # フォールバックで話題語 ``車`` より先に選ばれてしまう。
    "前", "後",
    # 明示的な履歴検索依頼の骨組み (「過去の会話で〜を探して/調べて」)
    "過去", "履歴", "探", "検索", "調", "教", "知",
    # 「もう一度」「〜させた」等の依頼骨組み (2026-08-05 追加)。
    "一度", "度", "全部", "全て", "読",
    # 出力の指図 (「箇条書きで挙げて」「違いを説明して」「推測は入れないで」)。
    # 話題語ではないのに内容ランとして残り、検索語を薄めていた
    # (2026-08-30 ライブ監査: ``箇条書 推測`` / ``デプロイ 説明`` /
    # ``周辺 観光地 3つ挙`` の 3 例が揃って 0 件だった)。
    "説明", "箇条書", "推測", "挙", "列挙", "復唱",
    # 同じく出力の指図。「これまでの会話を要約してください。」が ``要約`` を
    # 検索語にして空振りしていた (2026-09-03 ライブ監査 T07#9)。``説明`` と
    # 同じ扱いで、話題語ではなく指図として落とす。
    "要約", "整理",
})

#: 「3つ」「5個」のような **個数だけ** の語。指図の一部であって話題語ではない。
#: ストップワード剥がしの後に残る (「3つ挙」→ ``挙`` を剥がして ``3つ``)。
_ORDER_QUERY_COUNT_ONLY_RE = re.compile(r"^[0-9０-９]+[つ個件点名章行字]?$")
#: 日本語ストップワードを長い順に固定した並び (最長一致 + 決定論のため)。
#: frozenset をそのまま走査すると反復順が実行ごとに変わり、剥がれ方が
#: 非決定になる。
_ORDER_QUERY_STOPWORDS_BY_LEN: tuple[str, ...] = tuple(
    sorted(_ORDER_QUERY_STOPWORD_RUNS, key=len, reverse=True),
)


def _strip_stopword_affixes(run: str) -> str:
    """内容ランの前後に貼り付いたストップワードを剥がす (純粋関数)。

    日本語側は「漢字・カタカナ・ラテンの連続」を 1 ランとして切り出すため、
    隣接したストップワード同士が 1 つのランに融合してしまう。ラン単位の
    ストップワード照合はこの融合語を素通しし、語中で切れた無意味なキーワードが
    検索クエリに載る (2026-08-05 ライブ監査: 「今日私が最初に読ませたファイルの
    フルパスをもう一度教えてください」→ ``今日私 読 ファイル フルパス 一度教``
    で 0 件。``今日``+``私``、``一度``+``教`` がそれぞれ融合していた)。

    剥がすのは **残りが 2 文字以上、残り自体がストップワード、または剥がした
    ストップワードが 2 文字以上** の場合だけにする。無条件に剥がすと「教育」→
    「育」のように内容語を壊す (``教`` がストップワード)。

    3 つ目の条件は「2 文字以上の時点語 + 1 文字の動詞」の融合を解くためのもの
    (実インシデント 2026-08-16 ライブ監査 ターン5: 「昨日見た映画が…」が
    ``昨日見`` という実在しない語になり、照合の定足数を確実に落としていた)。
    1 文字のストップワードでは発動しないので「教育」は壊れない。
    """
    changed = True
    while changed and run:
        changed = False
        for stopword in _ORDER_QUERY_STOPWORDS_BY_LEN:
            if len(stopword) >= len(run):
                continue
            for rest in (
                run[len(stopword):] if run.startswith(stopword) else None,
                run[: -len(stopword)] if run.endswith(stopword) else None,
            ):
                if rest is None:
                    continue
                if (
                    len(rest) >= 2
                    or rest in _ORDER_QUERY_STOPWORD_RUNS
                    or len(stopword) >= 2
                ):
                    run, changed = rest, True
                    break
            if changed:
                break
    return run

# _ORDER_QUERY_SCAFFOLD_RE/_ORDER_QUERY_CONTENT_RE/_ORDER_QUERY_STOPWORD_RUNS
# の英語版。日本語版の「文字クラスで内容語/機能語を分離」は英語 (全て
# Latin script) には構造上適用できないため、単語トークン化 + ストップ
# ワードセット方式に設計変更する (_reduce_ordered_history_query 側で分岐)。
_ORDER_QUERY_SCAFFOLD_RE_EN = re.compile(
    r"\bin\s+(?:this|our)\s+conversation\b"
    r"|\bthis\s+(?:chat|conversation|thread)\b"
    r"|\bwhat\s+we\s+(?:talked|discussed)\s+about\b"
    r"|\b(?:very\s+)?first\s+(?:thing|time|question|message)\b"
    r"|\b(?:very\s+)?last\s+(?:thing|time|question|message)\b",
    re.IGNORECASE,
)
_ORDER_QUERY_CONTENT_RE_EN = re.compile(r"[A-Za-z0-9']+")
_ORDER_QUERY_STOPWORD_RUNS_EN = frozenset({
    "the", "a", "an", "in", "on", "at", "of", "to", "is", "was", "were",
    "what", "when", "where", "who", "which", "did", "do", "does",
    "i", "you", "we", "me", "my", "our", "your",
    "conversation", "chat", "thread", "talk", "talked", "discussed",
    "first", "last", "earliest", "latest", "very", "thing", "things",
    "time", "question", "message", "asked", "ask", "about",
    # 明示的な履歴検索依頼の骨組み
    "past", "previous", "history", "search", "find", "look", "tell",
    "ever", "any", "topic", "topics",
})


#: ユーザーが明示的に括った検索語の長さ上限。``「猫」`` ``『Python』`` ``"foo"``
#: を拾う。括弧の中身は **ユーザー自身が「これを探せ」と指定した語** なので、
#: 内容語の抽出規則より優先する。括りの取り出しは ``tool_judge_args.quoted_spans``
#: (判定系で唯一の実装) を使う。
_QUOTED_TERM_MAX_LEN = 40


def quoted_search_terms(query: str) -> list[str]:
    """クエリ中で括られた検索語を出現順に返す (純粋関数)。

    実インシデント (2026-08-25 ライブ監査): 「過去の会話から**「猫」**について
    話した内容を検索してください。」で ``search_history`` に **生の文全体** が
    渡り 0 件。内容語抽出は 1 文字語を落とすため (``_ORDER_QUERY_MIN_TERM_LEN``、
    ``tokenize_ja`` が bi-gram なので 1 文字はトークンが作れないことに由来)、
    縮約に掛けても「猫」は残らず生クエリへフォールバックしていた。

    括られた語は長さを問わず採る。``HistoryManager._text_matches_query`` の
    **部分一致 (tier 0) は 1 文字でも当たる** ので、キーワード段・bi-gram 段の
    1 文字制限は問題にならない。
    """
    return quoted_spans(query, max_len=_QUOTED_TERM_MAX_LEN)


def _reduce_ordered_history_query(query: str) -> str:
    """履歴リコール質問から search_history 用の内容キーワードを抽出する。

    ユーザーが検索語を括っている場合 (:func:`quoted_search_terms`) はそれを
    そのまま返す。内容語抽出より優先するのは、括った語が **ユーザー自身の
    指定** だから。

    レイヤー5.5 の強制フォールバックが search_history に生クエリ全文を渡すと、
    HistoryManager の字句照合は長い疑問文を短い会話ターンにマッチできない
    (2026-07-21 ライブ検証: 「この会話で一番最初に計算させた問題は何？」が
    索引の search_text に「計算」を含むのに No results found。2026-07-27
    ライブ検証: 「過去の会話で、登山の話題をしたことはありますか？探して
    ください。」→「登山」)。self-reference / 順序語 / 検索依頼 /
    疑問 scaffolding を除去して内容キーワードを残す。
    抽出できなければ生クエリを返す (悪化させない安全側)。digest には別途 raw
    query が渡るため、順序解釈 (「一番最初」) はこの縮約で失われない。
    """
    quoted = quoted_search_terms(query)
    if quoted:
        return " ".join(quoted)
    # 縮約アルゴリズムは **クエリ自身の字種** で選ぶ (GUI locale ではない)。
    # 日本語版は「漢字/カタカナのランを切り出し、ひらがなの助詞・活用語尾を
    # 落とす」文字クラス方式、英語版は「空白トークン + ストップワード」方式で、
    # 前提が言語ごとに違うため union できない (両方走らせても意味を成さない)。
    # locale で選ぶと、既定 'ja' のまま英語で打ったクエリに日本語の文字クラス
    # 抽出が掛かり、"the" / "what" が内容語として search_history へ載る。
    en = not has_japanese_script(query)
    if en:
        scaffold_re, content_re, stopwords = (
            _ORDER_QUERY_SCAFFOLD_RE_EN, _ORDER_QUERY_CONTENT_RE_EN,
            _ORDER_QUERY_STOPWORD_RUNS_EN,
        )
    else:
        scaffold_re, content_re, stopwords = (
            _ORDER_QUERY_SCAFFOLD_RE, _ORDER_QUERY_CONTENT_RE,
            _ORDER_QUERY_STOPWORD_RUNS,
        )
    stripped = scaffold_re.sub(" ", query)
    terms: list[str] = []
    noun_singles: list[str] = []
    for m in content_re.finditer(stripped):
        run = m.group(0)
        # 日本語はランの融合を解いてから照合する (英語は空白で切れており不要)。
        term = run if en else _strip_stopword_affixes(run)
        if not term or term.lower() in stopwords:
            continue
        # 1 文字の内容語は照合側が捨てるので、ここで落としてクエリを汚さない
        # (:data:`_ORDER_QUERY_MIN_TERM_LEN`)。英語側は元から空白区切りで
        # 1 文字語がほぼ出ないため、日本語だけに掛ける。
        if not en and len(term) < _ORDER_QUERY_MIN_TERM_LEN:
            # 直後が格助詞なら名詞、活用語尾なら動詞語幹と見なす。辞書を使わずに
            # 話題語だけを拾うための識別 (「飼っている猫について」で ``飼`` では
            # なく ``猫``)。動詞語幹しか無いクエリ (「一番最後に聞いた質問は？」)
            # は話題語ゼロなので、従来どおり生クエリへフォールバックさせる。
            if stripped[m.end():m.end() + 1] in _CASE_PARTICLES:
                noun_singles.append(term)
            continue
        if not en and _ORDER_QUERY_COUNT_ONLY_RE.match(term):
            continue
        terms.append(term)
    # 2 文字以上の内容語が 1 つも残らなかった場合に限り、名詞と判定した
    # 1 文字語の **先頭 1 つ** を検索語に採る。日本語の話題語には 1 文字漢字が多く
    # (猫 / 犬 / 車 / 本 / 父 / 色 / 山)、これを落とすと内容語ゼロになり
    # **生クエリ全文へのフォールバック** が起きる — この関数が防ぐために
    # 存在する当の失敗である (実インシデント 2026-09-03 ライブ監査:
    # 「前の会話で車の話をしましたか？」が発話文まるごとで 0 件)。
    # 複数を並べず 1 つに絞るのは、照合側で 1 文字語が効くのが
    # ``query in text`` の完全部分文字列一致 (単一語) 経路だけだから
    # (``history.history_manager._text_matches_query``)。空白区切りの
    # 複数語にすると語彙重なり経路へ入り、そこは 2 文字未満を捨てるため
    # 全語が消えて必ず不一致になる。
    if not terms and noun_singles:
        terms = noun_singles[:1]
    reduced = " ".join(terms).strip()
    return reduced if reduced else query


#: 「過去の会話について尋ねている」ことの **構造的** な signal。
#:
#: :data:`HISTORY_KEYWORDS` は閉じた語彙リストなので必ず漏れる。実インシデント
#: (2026-08-29 ライブ監査 T17): 「私が出張の話をしたのはいつですか。」
#: 「過去に、私が猫について話したことはありますか。」がどちらも語彙に当たらず、
#: 履歴検索が撃たれないまま **「2026年6月17日に行われた会話で」と日付を捏造**
#: した (実際は同日 40 分前)。
#:
#: 語彙を足す代わりに「**過去形の言及動詞 + 問いかけ**」という形で採る。
#:
#: **層 5.5 の強制フォールバックにも配線する** (2026-08-30 に変更)。
#: 当初は注記のゲート専用にして「発火条件を広げると別の退行を招く」と
#: 保留していたが、撃たなかった場合の実害の方が大きいことが実測で出た —
#: 2026-08-30 ライブ監査 T06 では 10 ターンすべてが
#: ``no_match_in_any_layer`` で search_history が 1 度も撃たれず、
#: 「いつ、どんな話をしましたか。」に **「2025年6月15日（日）の午後4時20分頃に」**
#: と、実在しない日時を断定で返した (実際は同じ日の 20 分前)。
#: 履歴の時刻は履歴ストアにしか無いので、撃たない限り必ず捏造になる。
#:
#: 空振りしても ``No results found`` 経由で「見つからなかった」という正直な
#: 応答に倒れるだけで、無言のまま確信を持って幻覚するより悪化しない
#: (層 5.5 の既存コメントと同じ論法)。
#:
#: 3 つ目の枝は **「いつ」が先に来る形**。日本語では
#: 「いつ、どんな話をしましたか。」のように時を先に置くのが自然で、
#: 1 つ目の枝 (言及動詞 → 問いかけ) の語順では拾えない。
#:
#: 発話動詞 (各枝の ``v1`` / ``v2`` / ``v3``) に直に掛かる相手が対話の当事者以外
#: なら会話の外の出来事 (「今日先生に相談したら何て言われたと思う?」、実機
#: 2026-10-03 run10)。相手の判定は日付の判定と同じ :func:`_talks_with_someone_else`。
_PAST_CONVERSATION_ASK_RE = re.compile(
    r"(?P<v1>話|言|伝え|聞|教え|質問|相談|説明)(?:を?し|っ|い)?(?:た|ました)"
    r"[^。？?\n]{0,24}(?:いつ|あります|ありました|ましたか|でしたか|どこ|何)"
    r"|(?:過去|以前|前回|昔|これまで)[^。？?\n]{0,24}"
    r"(?P<v2>話|言っ|聞い|伝え|教え|質問|会話|やり取り|探し|検索)"
    r"|いつ[^。？?\n]{0,24}(?P<v3>話|言っ|聞い|伝え|教え|質問|会話|やり取り)",
)


#: 会話を指す発話動詞の語幹 (相対日 / 具体日付の「その日の会話」で共有する)。
_SPEECH_VERB_STEMS = r"相談|話|聞|言|質問|依頼|頼|尋ね|伝え|教え"

#: 「今日 / 昨日 / 一昨日 + (私が) + 発話動詞の過去形」— **その日の会話** を
#: 尋ねる形。日の語は単独では暦の語 (「今日は何日ですか」「昨日見た映画」) なので
#: 履歴参照語には入れず、発話動詞と組で構造として取る。
#:
#: 実インシデント (2026-09-09 ライブ監査 H-08): 「今日私が相談した技術的な
#: 話題を 3 つ挙げてください。」がどの層でも履歴検索にならず、episodic 想起は
#: 今日と昨日のノートを日付ラベル無しで混在注入したため、「今日の会話履歴には
#: 技術的な話題が記録されていません」と昨日の話題 3 つを答えた。
_DAY_SCOPE_RECALL_RE = re.compile(
    r"(?P<day>" + alternation(kanji_terms(DAY_OFFSETS, where=lambda off: off <= 0))
    + r")(?:は|に|も|の)?(?P<gap>[^。！？!?\n]{0,16}?)"
    rf"(?:(?:{_SPEECH_VERB_STEMS})(?:を?[しっいん]|ね|え)?"
    r"(?:た|て|ました|まし)|話題|会話|やり取り)",
)

#: 発話動詞 (会話の名詞) に **直に** 掛かる相手 — 共格「と」・与格「に」(「友達と話した」
#: 「上司に話したら」「友達との会話」)。相手のある会話はユーザーの外の出来事で、履歴に無い。
#: 相手を語で列挙せず、助詞の位置と直前の文字種で取る。相手は名詞なので助詞の直前は
#: 漢字・カタカナ・英字か、それらに続く敬称 (母さん / 田中くん / 花子ちゃん)。
#: ひらがなの後の と は引用 (「まとめたいと言った」)、に / と は副詞 (すぐに / ちゃんと /
#: きちんと)、々と / 的に も副詞 (色々と / 具体的に) なので相手にしない。
#: 助詞の直後の漢字 1 字は、発話動詞の語幹と合わさった
#: 漢語 (「母に電話した」の 電+話) で、相手はその漢語に掛かる。
_SPEECH_PARTNER_END_RE = re.compile(
    rf"[{KANJI}{KATAKANA_BLOCK}a-zA-Z](?<!的)(?:(?:さ|く|ちゃ)ん)?(?:と|に)(?:の|[{KANJI}])?$",
)
#: 漢字で書く **順序・時** の語 + に は相手ではなく副詞句 (「最後に話した」「2番目に
#: 聞いた」「午後に相談した」)。会話内の位置の想起 (「最後に話したのは何でしたか」)
#: を相手のある会話として落とさない。「先」は前が漢字なら名詞 (取引先 / 勤務先)。
_ORDER_TIME_ADVERB_END_RE = re.compile(
    rf"(?:最初|最後|(?<![{KANJI}])先|前|後|次|(?:番|つ|回)目|時|頃|朝|昼|夜|夕方)に"
    rf"(?:[{KANJI}])?$",
)
#: 相手が対話の当事者 (アシスタント / ユーザー自身) なら履歴の会話そのもの
#: (「昨日あなたと話した」「昨日私に教えてくれた」)。
_DIALOG_PARTNER_END_RE = re.compile(
    rf"(?:{ASSISTANT_REFERENCE_STEMS}|{USER_REFERENCE_STEMS})(?:と|に)(?:の|[{KANJI}])?$",
    re.IGNORECASE,
)


def _talks_with_someone_else(gap: str) -> bool:
    """日の語と発話動詞の間 ``gap`` が、対話の当事者以外の相手で終わるか (純粋関数)。"""
    return (
        bool(_SPEECH_PARTNER_END_RE.search(gap))
        and not _DIALOG_PARTNER_END_RE.search(gap)
        and not _ORDER_TIME_ADVERB_END_RE.search(gap)
    )


_DAY_SCOPE_OFFSETS: dict[str, int] = {
    "今日": 0, "本日": 0, "昨日": 1, "一昨日": 2,
}


def day_scope_recall(query: str) -> int | None:
    """その日の会話を尋ねているなら、今日から何日前か (0 / 1 / 2) を返す (純粋関数)。"""
    m = _DAY_SCOPE_RECALL_RE.search(query or "")
    if m is None:
        return None
    return _DAY_SCOPE_OFFSETS.get(m.group("day"))


def history_day_window(days_ago: int, now_local) -> tuple[str, str]:
    """ローカル日の [開始, 終了] を、履歴索引の ``started_at`` と同じ形
    (UTC、``+00:00``) で返す (純粋関数)。

    索引側は時刻 (``parse_utc``) で比べるので書式は問わない。
    """
    from datetime import timedelta

    day = (now_local - timedelta(days=days_ago)).date()
    return _local_days_window(day, day, now_local.tzinfo)


#: 具体日付の直後に続く **その日の会話** の形 (「10月3日の会話」「10月2日から
#: 10月3日までのやり取り」「10/3 に話したこと」)。日付は暦の語 (「10月3日は何曜日」
#: 「締切は10月3日」) なので、相対日と同じく会話の名詞か発話動詞の **過去形** と
#: 組で取る (て形は「10月3日までに相談して」の依頼に当たるので採らない)。
#: 「履歴」は複合語 (購入履歴 / 変更履歴) に埋もれるので、日付に直接続く形だけ。
#:
#: 日付の出来事はアシスタントの外 (会議・電話・友人) のことが多いので、相対日
#: (:data:`_DAY_SCOPE_RECALL_RE`) より狭く取る — 発話動詞は日付の助詞に **直に**
#: 続く形だけ (間に置けるのは疑問の「何を / どんな」だけ。「10月3日の会議で話した」
#: 「10月3日に友達と話した」「電話した」を採らない)、名詞の前に挟めるのは相手・場を表す と / で を含まない短い語だけ、
#: 名詞は複合語の頭 (会話劇 / チャットログ) を採らない。「話題」は「10月3日の話題の
#: ニュース」に当たるので名詞に入れない。さらに :func:`dated_conversation_dates` が
#: 日付以降に依頼か問いの文を要求する。
#:
#: 実機 2026-10-03: 「10月2日から10月3日までの会話を一覧にして」が kNN ゲートで
#: no_tool になり、「10月3日の会話履歴を検索して」は ``query='10月3日'`` の窓なし
#: 検索になった (日付の窓を付けるのは相対日だけだった)。
_DATED_CONVERSATION_TAIL_RE = re.compile(
    r"(?:まで)?(?:の間)?(?:は|に|も|の|で)?\s*"
    r"(?:(?:チャット)?履歴"
    r"|[^。！？!?\nとで]{0,8}?(?:会話|やり取り|チャット)(?:履歴)?"
    rf"(?![{KATAKANA_BLOCK}{KANJI}])"
    rf"|(?:何を?|どんな)?(?:{_SPEECH_VERB_STEMS})(?:を?[しっいん]|ね|え)?(?:た|ました))",
)

_YMD = tuple[int | None, int, int]


def dated_conversation_dates(query: str) -> tuple[_YMD, _YMD] | None:
    """具体日付 (または範囲) の会話を尋ねているなら ``(始まり, 終わり)`` を返す (純粋関数)。

    日付の綴りと範囲の読みは ``core.response_dates`` (日付の読み取りの SSOT)。
    年を書いていない日付は年を ``None`` のまま返す (補うのは :func:`history_dates_window`)。
    """
    text = query or ""
    for end, first, last in literal_date_ranges(text):
        if _DATED_CONVERSATION_TAIL_RE.match(text, end) and _asks_from(text, end):
            return first, last
    return None


def _asks_from(text: str, pos: int) -> bool:
    """``pos`` 以降の文のどれかが依頼か問いか (純粋関数)。

    平叙の報告 (「10月3日に話した」「昨日友達と話した」「10月3日に聞いた講演が
    よかった」) は履歴を尋ねていない。
    """
    return any(
        is_request_sentence(s) or QUESTION_END_RE.search(s) or KA_QUESTION_END_RE.search(s)
        for s in split_sentences(text[pos:])
    )


def asks_about_day_scoped_conversation(query: str) -> bool:
    """**特定の日** の会話を尋ねているか (相対日 / 具体日付、純粋関数)。

    日の語と発話動詞の組に加えて、その後の文が依頼か問いであることを要る
    (「昨日友達と話した」は報告で、履歴の検索に回すと別の会話が混ざる)。
    router の層振り分けと、ツール判定の強制発火・抑止ガードが共有する 1 つの判定
    (片方だけだと「判定は正しいのに経路が外れる」)。
    """
    text = query or ""
    m = _DAY_SCOPE_RECALL_RE.search(text)
    if (
        m is not None
        and not _talks_with_someone_else(m.group("gap"))
        and _asks_from(text, m.start())
    ):
        return True
    return dated_conversation_dates(text) is not None


def history_dates_window(first: _YMD, last: _YMD, now_local) -> tuple[str, str] | None:
    """具体日付の範囲 (両端を含む) をローカル日の窓にする (純粋関数)。

    年の無い終わりは今日以前で最も近い日、年の無い始まりは終わり以前で最も近い日
    (「12月30日〜1月2日」は年をまたぐ)。片方だけ年があればもう片方も同じ年。
    存在しない日付 (2月30日) は ``None``。
    """
    from datetime import date

    def _resolve(ymd: _YMD, anchor: date, year: int | None) -> date | None:
        y, m, d = ymd
        y = y if y is not None else year
        if y is None:
            return nearest_date(m, d, anchor, future_days=0)
        try:
            return date(y, m, d)
        except ValueError:
            return None

    end = _resolve(last, now_local.date(), first[0])
    if end is None:
        return None
    start = _resolve(first, end, last[0])
    if start is None or start > end:
        return None
    return _local_days_window(start, end, now_local.tzinfo)


def _local_days_window(first, last, tz) -> tuple[str, str]:
    """ローカル日 ``first`` の始まり〜 ``last`` の翌日の始まりを UTC (``+00:00``) で返す。"""
    from datetime import datetime, timedelta, timezone

    start = datetime(first.year, first.month, first.day, tzinfo=tz)
    end = datetime(last.year, last.month, last.day, tzinfo=tz) + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%S.%f+00:00"
    return (
        start.astimezone(timezone.utc).strftime(fmt),
        end.astimezone(timezone.utc).strftime(fmt),
    )


def asks_about_past_conversation(query: str) -> bool:
    """過去の会話そのものについて尋ねているか (純粋関数)。

    語彙リスト (:func:`_has_history_recall_keywords`) と構造パターン
    (:data:`_PAST_CONVERSATION_ASK_RE`) の **どちらか** に当たれば真。
    構造パターンは発話動詞の相手が対話の当事者以外の一致を数えない。
    """
    text = query or ""
    if not text:
        return False
    return _has_history_recall_keywords(text) or _asks_past_conversation_structurally(text)


def _asks_past_conversation_structurally(text: str) -> bool:
    """:data:`_PAST_CONVERSATION_ASK_RE` の一致のうち、発話動詞に他人の相手が
    掛かっていないものがあるか (純粋関数)。"""
    pos = 0
    while (m := _PAST_CONVERSATION_ASK_RE.search(text, pos)) is not None:
        if not _talks_with_someone_else(text[: m.start(m.lastgroup or "v1")]):
            return True
        pos = m.start() + 1
    return False


def _has_history_recall_keywords(query: str) -> bool:
    """明示的な履歴参照キーワード (``HISTORY_KEYWORDS``) を含むか。

    実体は ``core.intent_vocab.has_history_recall_keyword`` (router の層分類と
    同じ照合)。**router と同じく JA / EN 両方を見る** — 以前はこちらだけ locale
    で片寄っており、同じ語彙・同じ関数を通しながら層分類と強制発火の判定が
    食い違っていた。
    """
    return has_history_recall_keyword(query)
#: 会話に既出の対象を指す連体詞 + 名詞。「今日」「現在」のような直示語は
#: 含めない (それらは実測して答えるのが正しい)。
_ANAPHORIC_REFERENCE_RE = re.compile(
    r"(?:その|あの|例の|先ほどの|さきほどの|さっきの|前述の|上記の|くだんの)"
    r"\s*[^\s、。，．]{1,12}",
)
#: 過去に述べられた内容を尋ね直す文末形。語彙は core.intent_vocab が SSOT
#: (``tool_judge_commands`` の now-only 抑止と同じ形)。後方互換で旧名を残す。
_RETROSPECTIVE_QUESTION_RE = PAST_RECALL_TAIL_RE


def asks_about_prior_conversation_entity(query: str) -> bool:
    """会話に既出の対象について尋ね直しているか (純粋関数)。

    ``_INFER_TOOL_EXEC_QUERY_RE`` は「何曜日」「日付」等の語だけで実行可能
    クエリと判定するため、会話で決めた予定を尋ね直す文まで日時取得コマンドに
    乗ってしまう。ツール結果は「唯一の事実根拠」として base に渡るので、
    現在時刻が会話の文脈を押しのけて誤答になる (実インシデント 2026-07-29
    ライブ監査: 「来週の水曜日に東京で」→「大阪の木曜に訂正」と直した直後に
    「その打ち合わせは何曜日にどこでしたか？」と尋ねたところ、
    ``datetime.now()`` が発火し、訂正前の「来週の水曜日に東京で打ち合わせが
    あります。」がそのまま返った)。

    連体詞による既出参照と、過去を尋ね直す文末形の **両方** を要求する。
    「今日は何曜日でしたっけ?」は既出参照が無いので従来どおり実測へ回る。
    """
    if not query:
        return False
    return bool(
        _ANAPHORIC_REFERENCE_RE.search(query)
        and _RETROSPECTIVE_QUESTION_RE.search(query)
    )
