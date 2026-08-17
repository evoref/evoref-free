"""履歴リコールクエリの検出と縮約 (純粋関数)

``search_history`` へ渡す内容キーワードの抽出と、履歴参照語の種別判定
(近接リコールか長距離リコールか) を担う。HistoryManager の字句照合は疑問文
全文にマッチしないため、縮約はこの層の責務。
"""

from __future__ import annotations

import re

from backend.free.agent.router import (
    HISTORY_KEYWORDS,
    HISTORY_KEYWORDS_EN,
)
from backend.free.core.intent_vocab import PROXIMAL_RECALL_KEYWORDS
from backend.free.core.locale_patterns import is_en_locale, select_locale_variant

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
    r"|一番最初|一番最後|何番目|何回目",
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
_ORDER_QUERY_CONTENT_RE = re.compile(
    r"[一-鿿゠-ヿ々〆a-zA-Z0-9]+",
)

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
    # 明示的な履歴検索依頼の骨組み (「過去の会話で〜を探して/調べて」)
    "過去", "履歴", "探", "検索", "調", "教", "知",
    # 「もう一度」「〜させた」等の依頼骨組み (2026-08-05 追加)。
    "一度", "度", "全部", "全て", "読",
})
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


def _reduce_ordered_history_query(query: str) -> str:
    """履歴リコール質問から search_history 用の内容キーワードを抽出する。

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
    en = is_en_locale()
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
    for run in content_re.findall(stripped):
        # 日本語はランの融合を解いてから照合する (英語は空白で切れており不要)。
        term = run if en else _strip_stopword_affixes(run)
        if not term or term.lower() in stopwords:
            continue
        # 1 文字の内容語は照合側が捨てるので、ここで落としてクエリを汚さない
        # (:data:`_ORDER_QUERY_MIN_TERM_LEN`)。英語側は元から空白区切りで
        # 1 文字語がほぼ出ないため、日本語だけに掛ける。
        if not en and len(term) < _ORDER_QUERY_MIN_TERM_LEN:
            continue
        terms.append(term)
    reduced = " ".join(terms).strip()
    return reduced if len(reduced) >= 2 else query
#: 進行中の会話を指す近接リコール語。これらは「今のセッションの中」を指すので、
#: 現在セッションを除外した search_history では構造的に当たらない。
def _only_proximal_recall_keywords(query: str) -> bool:
    """履歴参照語が近接リコール語だけか (純粋関数)。

    長距離リコール語 (「以前」「最初に」「覚えて」等) が 1 つでもあれば False。
    """
    q_lower = query.lower()
    keywords = select_locale_variant(HISTORY_KEYWORDS, HISTORY_KEYWORDS_EN)
    matched = [kw for kw in keywords if kw in q_lower]
    if not matched:
        return False
    return all(kw in PROXIMAL_RECALL_KEYWORDS for kw in matched)


def _has_history_recall_keywords(query: str) -> bool:
    """明示的な履歴参照キーワード (router.HISTORY_KEYWORDS) を含むか。

    router.ComplexityClassifier._has_history_keywords と同じ判定 (小文字化
    後の部分文字列一致) を、layer 分類とは独立に tool 強制発火の判定に使う。
    """
    q_lower = query.lower()
    keywords = select_locale_variant(HISTORY_KEYWORDS, HISTORY_KEYWORDS_EN)
    return any(kw in q_lower for kw in keywords)
#: 会話に既出の対象を指す連体詞 + 名詞。「今日」「現在」のような直示語は
#: 含めない (それらは実測して答えるのが正しい)。
_ANAPHORIC_REFERENCE_RE = re.compile(
    r"(?:その|あの|例の|先ほどの|さきほどの|さっきの|前述の|上記の|くだんの)"
    r"\s*[^\s、。，．]{1,12}",
)
#: 過去に述べられた内容を尋ね直す文末形。
_RETROSPECTIVE_QUESTION_RE = re.compile(
    r"でした(?:か|っけ)|だった(?:か|っけ)|でしたよね|だっけ"
    r"|と言(?:い|っ)ました|と伝えました",
)


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
