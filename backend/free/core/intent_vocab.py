"""会話パイプラインのルールベース判定が共有する語彙・パターンの SSOT。

``backend/free/document_nouns.py`` (文書名詞) と同じ趣旨で、複数モジュールが
独立に持っていた判定語彙をここへ集約する。pillar 境界 (gen は他 pillar を一切
import できない) を越えずに済むよう、どの pillar にも属さない ``core/`` に置く
(``locale_patterns.py`` と同じ理由。``core/__init__.py`` は空なので連鎖 import の
コストも無い)。

**ここへ載せてよいのは「複数の消費側が同一の語彙を使う」ものだけ**。
判定語彙の重複には 2 種類あり、扱いが異なる:

- **同一の重複**: 定義が byte 一致で、片方だけ直すと不整合になる。→ ここへ集約。
- **意図的な分岐**: 消費側ごとに誤検出コストが違うため語彙をあえて変えている
  (例: セッション自己参照は tool_call_judge では検索範囲を絞るだけだが
  self_rag_judge では RAG を丸ごと skip するため、話題ポインタにもなる語を
  self_rag 側は採らない — 実測付きの根拠が self_rag_judge に残っている)。
  → **統合してはいけない**。共有するなら構造 (近接窓・否定先読み) だけにし、
  語彙差は名前付き定数として宣言する。
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────
# ファイルパス
# ─────────────────────────────────────────────────────────────────────

#: 明示的な Windows ドライブレターパス。空白と引用符・括弧で区切られた
#: 1 トークンを取る。``agent.meta_cognitive`` (plan 後のパス脱落補完) と
#: ``agent.feedback`` (経験記録のパス抽出) が byte 一致の定義を各々持っていた。
#:
#: 注意: 他モジュールのパス正規表現とは **目的が違うので統合しない**。
#: ``rag.self_rag_judge.FILE_PATH_PATTERN`` は拡張子を必須にし
#: (Unix パスも拾う)、``agent.router._LOCAL_PATH_RE`` はドライブ接頭辞の
#: 存在だけを見る。要求が異なるものを 1 本にすると、どちらかの誤検出率が上がる。
EXPLICIT_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'「」()（）]+")


# ─────────────────────────────────────────────────────────────────────
# 書込み意図
# ─────────────────────────────────────────────────────────────────────

#: 「書く・作る」系の動作動詞。``agent.meta_cognitive_tasks`` の
#: タスク種別判定と ``agent.meta_cognitive_tools`` の write_file ルーティングが
#: byte 一致の正規表現を各々持っていた。
#:
#: 名詞 (excel / docx 等) は含めない — 単独で発火させると「report.xlsx を
#: 読んで」のような read 文脈を誤って書込みと判定する。
WRITE_VERB_RE = re.compile(
    r"作成|追加|実装|修正|変更|書き込|生成|更新|書く|書いて"
    r"|create|write|add|implement|modify|update|generate|fix|refactor",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────
# 履歴参照キーワード
# ─────────────────────────────────────────────────────────────────────

#: 履歴参照キーワードと、それが指す時間的な距離。
#:
#: ``proximal`` (「さっき」「先ほど」) は **進行中の会話** を指す。会話履歴は既に
#: コンテキストへ載っているため、現在セッションを除外した検索を撃つと構造的に
#: 必ず空振りする。``long_range`` (「以前」「最初に」「覚えて」) は過去セッション
#: を指すので撃つ価値がある。
#:
#: 以前は近接語だけを別タプルで再掲していたため、``HISTORY_KEYWORDS`` 側へ
#: 近接語を足しても距離の分類に反映されない状態だった (実際に「さきほど」が
#: 近接語タプルにだけ存在し、キーワード側に無いため一度も効かない死んだ
#: エントリになっていた)。距離をキーワード表そのものに持たせて派生させる。
_HISTORY_KEYWORD_DISTANCE: tuple[tuple[str, str], ...] = (
    ("前に", "long_range"),
    ("以前", "long_range"),
    ("先週", "long_range"),
    ("先月", "long_range"),
    ("この間", "long_range"),
    ("前回", "long_range"),
    ("前の会話", "long_range"),
    ("さっき", "proximal"),
    ("昨日", "long_range"),
    ("今朝", "proximal"),
    ("先ほど", "proximal"),
    ("最初に", "long_range"),
    ("覚えて", "long_range"),
    ("覚えてる", "long_range"),
    ("覚えている", "long_range"),
    ("過去の会話", "long_range"),
    ("過去のやり取り", "long_range"),
    ("過去に話", "long_range"),
    ("以前の会話", "long_range"),
    ("会話履歴", "long_range"),
    ("earlier", "long_range"),
    ("previously", "long_range"),
    ("last time", "long_range"),
    ("yesterday", "long_range"),
    ("before", "long_range"),
    ("remember", "long_range"),
    ("recall", "long_range"),
)

#: ``_HISTORY_KEYWORD_DISTANCE`` の英語版 (locale='en' でのみ使う)。
#: 日本語版と 1 対 1 の逐語訳である必要はなく、各言語で自然な語彙を選ぶ。
_HISTORY_KEYWORD_DISTANCE_EN: tuple[tuple[str, str], ...] = (
    ("earlier", "long_range"),
    ("previously", "long_range"),
    ("last time", "long_range"),
    ("yesterday", "long_range"),
    ("before", "long_range"),
    ("remember", "long_range"),
    ("recall", "long_range"),
    ("this morning", "proximal"),
    ("just now", "proximal"),
    ("a moment ago", "proximal"),
    ("a while back", "long_range"),
    ("at first", "long_range"),
    ("in the beginning", "long_range"),
    ("past conversation", "long_range"),
    ("previous conversation", "long_range"),
    ("conversation history", "long_range"),
    ("chat history", "long_range"),
)

HISTORY_KEYWORDS: list[str] = [kw for kw, _ in _HISTORY_KEYWORD_DISTANCE]
HISTORY_KEYWORDS_EN: list[str] = [kw for kw, _ in _HISTORY_KEYWORD_DISTANCE_EN]

#: 進行中の会話を指すキーワード (日英まとめて保持する。照合側は
#: ``HISTORY_KEYWORDS`` / ``HISTORY_KEYWORDS_EN`` から取った語の距離を引くだけ)。
PROXIMAL_RECALL_KEYWORDS: frozenset[str] = frozenset(
    kw
    for kw, distance in (*_HISTORY_KEYWORD_DISTANCE, *_HISTORY_KEYWORD_DISTANCE_EN)
    if distance == "proximal"
)


# ─────────────────────────────────────────────────────────────────────
# セッション自己参照 (「この会話で〜」)
# ─────────────────────────────────────────────────────────────────────
#
# 消費側が 2 つあり、**構造は共有するが語彙は共有しない**:
#
# - ``agent.tool_call_judge``: マッチすると search_history を現在セッションへ
#   限定するだけ。誤検出コストは軽微なので語彙は広く取る (BROAD)。
# - ``rag.self_rag_judge``: マッチすると RAG 検索を丸ごと skip する。誤検出
#   すると外部知識が引けなくなるため語彙は狭く取る (NARROW)。
#
# 実測 (2026-07-27、自己参照 8 件 + 外部知識 9 件のプローブ):
#   全語同期  → 自己参照 8/8 捕捉、外部知識 4/9 を誤って skip
#   NARROW    → 自己参照 8/8 捕捉、外部知識 0/9 誤爆
# 「この会話で〈話した/聞いた/質問した/指摘された〉X について詳しく教えて」の
# ように、会話を X の指し示しに使いつつ欲しいのは外部知識、という形が自然に
# 成立するため、話題ポインタにもなる語 (順番/質問/指摘/言った/話した/聞いた)
# は NARROW では採らない。**この差は意図的であり、統合してはいけない。**
#
# 一方で下の 3 要素 (アンカー / 否定先読み / 近接窓) は byte 一致で共有されて
# おり、以前は両ファイルに書き写されていた。窓幅を 20→40 に広げた際は両方を
# 手で直す必要があり、片方だけ直すと挙動がずれる状態だった。構造はここから
# 派生させ、機械的に同期させる。

#: 会話そのものを指す前置き。
SESSION_ANCHOR_JA = (
    r"(?:この会話|このやり取り|今までの(?:会話|やり取り)"
    r"|今日の(?:追加分の)?会話|今回の(?:追加分の)?会話)"
)

#: 明示的な話題切断の前置きを弾く否定先読み。「この会話とは別に、相対性理論に
#: ついて教えて」のような外部知識質問を自己参照と誤判定しないためのガード。
SESSION_TOPIC_BREAK_LOOKAHEAD_JA = r"(?!とは別|とは関係|は関係な|じゃなく|ではなく)"

#: アンカーと反省語の近接窓。文境界 (句点・感嘆符・疑問符・改行) を跨がない
#: 文字クラスで 40 文字。任意文字 ``.{0,20}`` だった頃は「この会話で一番最初に
#: 私が計算させた問題は何だったか覚えてますか？」(間 21 文字) を 1 文字超過で
#: 取りこぼし、任意文字のまま 50 へ広げると外部知識質問まで拾ってしまった。
SESSION_PROXIMITY_WINDOW_JA = r"[^。．!！?？\n]{0,40}?"


def session_self_reference_pattern_ja(reflective_vocab: str) -> str:
    """セッション自己参照パターンの正規表現文字列を組み立てる (純粋関数)。

    Args:
        reflective_vocab: 会話自体を振り返る語の alternation。消費側ごとの
            誤検出コストに応じて BROAD / NARROW を渡し分ける (上のコメント参照)。
    """
    return (
        SESSION_ANCHOR_JA
        + SESSION_TOPIC_BREAK_LOOKAHEAD_JA
        + SESSION_PROXIMITY_WINDOW_JA
        + f"(?:{reflective_vocab})"
    )


#: 英語版のアンカー / 否定先読み / 近接窓。
SESSION_ANCHOR_EN = (
    r"(?:this\s+conversation|this\s+chat|our\s+conversation"
    r"|what\s+we\s+(?:talked|discussed|were\s+talking)\s+about"
    r"|earlier\s+in\s+this\s+(?:conversation|chat)"
    r"|so\s+far\s+in\s+this\s+conversation)"
)
SESSION_TOPIC_BREAK_LOOKAHEAD_EN = (
    r"(?!\s*(?:is|was|has)?\s*(?:not\s+related|unrelated|nothing\s+to\s+do))"
)
SESSION_PROXIMITY_WINDOW_EN = r"[^.!?\n]{0,40}?"


# ─────────────────────────────────────────────────────────────────────
# 進行中セッションの位置指定 (「この会話で最初に言ったこと」)
# ─────────────────────────────────────────────────────────────────────
#
# 位置で決まる事実は検索でもモデルの読解でもなく、並び順から機械的に確定する。
# 進行中の会話は全文がコンテキストに載っている一方、``search_history`` の索引に
# はまだ入っておらず、現在セッションを検索しても中身の無いヘッダしか返らない。

#: 「最初 / 最後」のどちらを指しているか。両方現れる曖昧な文は採らない。
_SESSION_POSITION_FIRST_RE = re.compile(
    r"(?:一番)?最初|最初に|first|earliest", re.IGNORECASE,
)
_SESSION_POSITION_LAST_RE = re.compile(
    r"(?:一番)?最後|最後に|直前|last|latest|most\s+recent", re.IGNORECASE,
)
#: 位置指定の対象がユーザー自身の発言であること。「最初に説明した内容」等の
#: 話題ポインタを巻き込まないよう、発言そのものを指す語を要求する。
_SESSION_POSITION_TARGET_RE = re.compile(
    r"(?:送|言|聞|尋|書|投げ|打)\S{0,4}?"
    r"(?:メッセージ|発言|質問|こと|内容|の)"
    r"|メッセージ|発言"
    r"|(?:message|question|thing)\s+(?:i|you)\s+(?:sent|said|asked)"
    r"|(?:sent|said|asked)",
    re.IGNORECASE,
)
#: アンカー + 話題切断の否定先読み。「この会話とは別に、最初に送るメッセージの
#: 例を教えて」のような外部依頼を自己参照と誤判定しないため、他の自己参照判定と
#: 同じ先読みを掛ける。
_SESSION_ANCHOR_ANY_RE = re.compile(
    f"(?:{SESSION_ANCHOR_JA}{SESSION_TOPIC_BREAK_LOOKAHEAD_JA})"
    f"|(?:{SESSION_ANCHOR_EN}{SESSION_TOPIC_BREAK_LOOKAHEAD_EN})",
    re.IGNORECASE,
)


def session_position_kind(query: str) -> str | None:
    """クエリが進行中セッションの「最初 / 最後の発言」を尋ねているか判定する。

    3 条件すべてを満たす場合だけ確定する: 会話そのものへのアンカーがある /
    位置語が片方だけ現れる / 対象が発言そのものである。曖昧な文は None を
    返して従来経路へ委ねる (純粋関数)。

    Returns:
        ``"first"`` / ``"last"`` / ``None``。
    """
    if not query or not _SESSION_ANCHOR_ANY_RE.search(query):
        return None
    if not _SESSION_POSITION_TARGET_RE.search(query):
        return None
    is_first = bool(_SESSION_POSITION_FIRST_RE.search(query))
    is_last = bool(_SESSION_POSITION_LAST_RE.search(query))
    if is_first == is_last:
        return None
    return "first" if is_first else "last"


def resolve_session_position_message(
    conversation: list[dict] | None, query: str, position: str,
) -> str:
    """会話履歴から最初 / 直近のユーザー発言を取り出す (純粋関数)。

    今まさに尋ねている質問自体は対象から外す。``conversation`` に現在ターンが
    既に積まれているかは呼出経路によって違うため、内容一致で除外する。

    Returns:
        発言本文。確定できなければ空文字列。
    """
    texts = [
        content.strip()
        for msg in conversation or []
        if msg.get("role") == "user"
        and isinstance(content := msg.get("content"), str)
        and content.strip()
        and content.strip() != query.strip()
    ]
    if not texts:
        return ""
    return texts[0] if position == "first" else texts[-1]


# ─────────────────────────────────────────────────────────────────────
# 挨拶
# ─────────────────────────────────────────────────────────────────────
#
# 消費側が 2 つあり、ここでも **構造は共有するが語彙は共有しない**:
#
# - ``agent.router.GREETING_PATTERNS``: reactive 層へ振り分けるための粗い判定。
# - ``agent.reactive.GREETING_RESPONSES``: 挨拶ごとの定型応答を選ぶための判定。
#   応答文と 1 対 1 で対応させる必要があるため分割が細かい。
#
# 語彙は実際に食い違っている (例: 「おはようございます」は reactive の
# ``おはよう(?:ございます)?`` は拾うが router の ``おはよう`` は拾わない)。
# 揃えると層の振り分けが変わるため、ここでは **囲いの体裁だけ** を共有する。

#: 挨拶マッチの体裁。クエリ全体が挨拶だけであることを要求する
#: (部分一致だと「こんにちは、ところで〜」のような本題付きまで反射応答に
#: 落ちてしまう)。末尾の句読点・感嘆符は任意。
def exact_greeting_pattern(alternation: str, *, punctuation: str) -> str:
    r"""``^(?:<alternation>)\s*[<punctuation>]?\s*$`` を組み立てる (純粋関数)。

    Args:
        alternation: 挨拶語の alternation (``|`` 区切り)。
        punctuation: 許容する末尾記号の文字クラス中身 (日本語は全角を含む)。
    """
    return rf"^(?:{alternation})\s*[{punctuation}]?\s*$"


#: 日本語クエリで許容する末尾記号 (全角の ！。 を含む)。
GREETING_PUNCTUATION_JA = r"!！。."
#: 英語クエリで許容する末尾記号。
GREETING_PUNCTUATION_EN = r"!."


# ─────────────────────────────────────────────────────────────────────
# ASCII 境界ガード
# ─────────────────────────────────────────────────────────────────────
#
# ``\b`` は日本語文字を ``\w`` とみなすため、日英混在クエリの英語-日本語境界で
# 期待どおりに働かない。短い ASCII 略語 (CPU / RAM / GPU / OS / env 等) を
# ``re.IGNORECASE`` で素のまま並べると、英単語の内部に部分マッチする
# (``program`` / ``diagram`` / ``telegram`` の 'ram' が典型)。
#
# この穴は 2026-07-22 監査で tool_call_judge 側だけ塞がれ、router 側の
# ``_EXECUTABLE_QUERY_PATTERNS`` には残っていた。境界の付け忘れが再発しないよう、
# 手書きの後読み/先読みではなくこのヘルパを通す。


def ascii_boundary(term: str) -> str:
    """``term`` の前後に ASCII 英字の境界を要求する正規表現片を返す。"""
    return rf"(?<![A-Za-z]){term}(?![A-Za-z])"


def ascii_boundary_alternation(*terms: str) -> str:
    """複数の短い ASCII 語をまとめて境界付き alternation にする。"""
    return "|".join(ascii_boundary(t) for t in terms)


# ─────────────────────────────────────────────────────────────────────
# 既存ファイルへの再保存 (参照表現)
# ─────────────────────────────────────────────────────────────────────

#: 保存先を直前の文脈に委ねる参照表現。「同じファイルに保存し直して」のような
#: 追記・修正依頼はパスを本文に持たないため、パス正規表現には掛からない。
#:
#: 消費側が 2 つあり、**同じ語彙を同じ意味で使う**:
#:
#: - ``agent.router``: 書込み意図の検出。掛からないと deliberative へ落ちて
#:   read_file だけが走り、書込みが一度も起きないまま「保存し直した」体の
#:   回答になる (実測 2026-07-27)。
#: - ``agent.feedback``: 訂正判定の除外。「3 番目の項目を直して、同じファイルに
#:   保存し直して」は **編集依頼** であってアシスタントの誤りへの訂正ではない。
#:   これを訂正として数えると、成功した書込みターンが失敗として学習される。
REFERENTIAL_WRITE_TARGET_RE = re.compile(
    r"(?:同じ|その|この|先ほどの?|さっきの?)\s*(?:ファイル|ところ|場所)"
    r"|保存し直|上書き|同じ場所に"
    r"|\b(?:same|that)\s+file\b|\boverwrite\b",
    re.IGNORECASE,
)
