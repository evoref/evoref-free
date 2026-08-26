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
#: 「追記」「書き足」は 2026-08-08 に追加。``追加`` はあるのに ``追記`` が無く、
#: 「同じファイルに 3 行追記して」が書込み意図として認識されずルータの
#: local_write_intent を外れていた。その結果 chat では実行できる書込みツールが
#: 無い経路に落ち、ツールを 1 つも撃たないまま「追記しました」と完了を捏造した
#: (実ファイルは無変更。ライブ監査 ターン6)。
#: 「作って」「作る」は 2026-08-09 に追加。``作成`` はあるのに口語の ``作って``
#: が無く、``meta_cognitive_tools._TOOL_PATTERNS`` は write を先に照合する設計
#: なのに書込み動詞として当たらず、後段の read パターン (``中身``/``内容``) に
#: 落ちていた。その結果「E:\tmp に 在庫メモ.txt も **作って** ください。
#: 中身は…」がファイル作成ではなく ``read_file`` として実行され、
#: ``File not found`` で失敗した (ファイルは未作成。2026-08-09 ライブ監査)。
#: 同じ依頼に「書いて」を足すと write 経路に乗って成功しており、差は動詞だけ。
#:
#: 「書き直」「書き換え」「上書き」「差し替え」「保存し」は 2026-08-09 の
#: 2 回目のライブ監査で追加。``書[きい]`` 系は ``書く`` / ``書いて`` /
#: ``書き足`` / ``書き込`` しか無く、**「書き直してください」が書込み期待と
#: 判定されなかった**。その結果 ``determine_task_status`` の
#: 「write を期待したのに write_file が走っていない → failed」ガードが効かず、
#: ツールを 1 度も撃たないまま ``status=done`` で ✓ 表示され、ベースモデルが
#: 吐いたツールコール構文 (``<|tool_call>call:write_file{...}``) がそのまま
#: チャット本文に出た (実ファイルは無変更)。
#: ``保存`` は「保存場所」「保存されている」のような状態の言及も拾うため、
#: 連用形の ``保存し`` に限定する。
WRITE_VERB_RE = re.compile(
    r"作成|作って|作る|追加|追記|書き足|書き直|書き換え|上書き|差し替え|保存し"
    r"|実装|修正|変更|書き込|生成|更新|書く|書いて"
    r"|create|write|append|add|implement|modify|update|generate|fix|refactor",
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
    # 「前に」単独は 2 文字の部分文字列で、履歴参照と無関係な語に必ず埋もれる
    # (締切直**前に** / 名**前に** / 事**前に** / 手**前に** / 目の**前に**)。
    # 照合は小文字化後の素の部分一致なので境界が無く、実インシデント
    # (2026-08-09 2 回目のライブ監査): 「大会エントリーは締切直前にアクセスが
    # 集中します…」という純粋な技術質問で search_history が強制発火し、
    # 新規インストールで索引が空のため 1 往復を空費した。
    # 履歴参照として意味を持つのは発話動詞を伴う形なので、そちらを列挙する。
    # 単独の「以前」「前回」「過去の会話」は別エントリで拾えている。
    ("前に言", "long_range"),
    ("前にも言", "long_range"),
    ("前に話", "long_range"),
    ("前に聞", "long_range"),
    ("前に伝え", "long_range"),
    ("前に教え", "long_range"),
    ("以前", "long_range"),
    ("先週", "long_range"),
    ("先月", "long_range"),
    ("この間", "long_range"),
    ("前回", "long_range"),
    ("前の会話", "long_range"),
    ("さっき", "proximal"),
    # 「昨日」「最初に」単独は「前に」「覚えて」と同じ部分文字列の罠。どちらも
    # **会話とは無関係な文脈** に日常的に現れる。実インシデント
    # (2026-08-16 ライブ監査):
    #   ターン5 「昨日見た映画がすごく良くてさ、久しぶりに泣いちゃった。」
    #   ターン11「…複数のコルーチンを同時に走らせて最初に終わったものだけ…」
    #   ターン21「新規事業の企画書を作るとき、最初に固めるべき項目を…」
    # の 3 件で search_history が強制発火し、いずれも 0 件で 1 往復を空費した
    # (発火 4 件中 3 件が誤発火)。履歴参照として意味を持つのは発話動詞を伴う形
    # なので、そちらを列挙する。「この会話で最初に送ったメッセージ」のような
    # 位置指定は ``session_position_kind`` /
    # ``is_whole_session_scope_query`` が別経路で拾うため、ここを絞っても
    # 取りこぼさない。
    ("昨日言", "long_range"),
    ("昨日話", "long_range"),
    ("昨日聞", "long_range"),
    ("昨日伝え", "long_range"),
    ("昨日教え", "long_range"),
    ("昨日質問", "long_range"),
    ("昨日の会話", "long_range"),
    ("昨日のやり取り", "long_range"),
    ("今朝", "proximal"),
    ("先ほど", "proximal"),
    # 「一番最初 / 一番最後」は会話の順序を指す複合語で、日常文には出にくい。
    # 単独の「最初に」と違って誤発火の余地がほぼ無いので、動詞を問わず採る。
    ("一番最初", "long_range"),
    ("一番最後", "long_range"),
    ("最初に言", "long_range"),
    ("最初に話", "long_range"),
    ("最初に聞", "long_range"),
    ("最初に送", "long_range"),
    ("最初に頼", "long_range"),
    ("最初に依頼", "long_range"),
    ("最初に質問", "long_range"),
    ("最初に読ま", "long_range"),
    ("最初に教え", "long_range"),
    # 「覚えて」単独は「前に」と同じ部分文字列の罠。**保存指示**の
    # 「覚えておいて(ください)」「覚えといて」にも当たってしまう。実インシデント
    # (2026-08-12 ライブ監査 ターン3): 「私の名前は小川博之です。覚えておいて
    # ください。」で search_history が強制発火した。しかも
    # ``_maybe_scope_session_search`` が現セッションを除外するため、たった今
    # 述べられた事実は **構造的に必ず空振り**する (1 往復を確実に空費する)。
    # 同じ曖昧性は EvorefMem 側で解決済み (notes/pin_detector.py の
    # ``_trigger_evidence_is_question_only``: 「覚えておいて」は保存指示、
    # 「覚えていますか？」は想起依頼) だが、この表へ伝播していなかった。
    # 想起を問う活用形だけを採る (次の文字が お / と なら保存指示側)。
    ("覚えてい", "long_range"),
    ("覚えてる", "long_range"),
    ("覚えてま", "long_range"),
    ("おぼえてい", "long_range"),
    ("おぼえてる", "long_range"),
    ("おぼえてま", "long_range"),
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
    # 「このセッション」「この対話」は「この会話」と同義のアンカー。同義語を
    # 落とすと同じ取りこぼしを言い換えのたびに繰り返す (2026-08-08 監査で
    # 「このセッションで最後に指示した内容は」がアンカー無し扱いになった)。
    # 「ここまでの」「これまでの」は「今までの」と同義。落としていたため
    # 「ここまでの会話を要約して」がアンカー無し扱いになり、reactive 軽量パス
    # (直近 6 メッセージ・STM/SemMem 注入なし) で直近 3 往復だけを要約した
    # (2026-08-12 ライブ監査 ターン21)。
    r"(?:この会話|このやり取り|このセッション|この対話"
    r"|(?:今|ここ|これ)までの(?:会話|やり取り|対話|セッション)"
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
    # 動詞に依頼系 (依頼|お願い|頼|指示) を含める。ユーザーは自分の発言を
    # 「質問」ではなく「依頼」と呼ぶことがあり、語彙が欠けていると同じ
    # 取りこぼしを繰り返す (実インシデント 2026-08-08 ライブ監査:
    # 「この会話で私が一番最初に依頼したことは何でしたか。」に対し 9 番目の
    # 発言「今日から100日後は…」と誤答した。2026-08-04 の「した質問」の
    # 取りこぼしと同型)。
    r"(?:送|言|聞|尋|書|投げ|打|依頼|お願い|頼|指示)\S{0,4}?"
    r"(?:メッセージ|発言|質問|依頼|指示|こと|内容|の)"
    # 名詞化しない疑問形も同じ対象を指す。「言った**こと**」は拾えるのに
    # 「言った**か**」が漏れており、search_history 経由へ落ちて答えられな
    # かった (実インシデント 2026-08-10 ライブ監査: 「この会話の最初に、
    # 私が何を決めると言ったか答えてください。」→ 抽出後のキーワードが
    # ``決 言 答`` と 1 文字語だけになり score 0.1 で「記述はありません」)。
    # このクエリには内容名詞が無く、字句検索では構造的に answered できない。
    # 位置参照として解くのが正しい経路。
    r"|(?:言|述べ|話|聞|尋|送|書)\S{0,3}?(?:た|ました)(?:か|っけ|でしょうか)"
    # 「質問」は動詞を伴わない形でも発言そのものを指す。動詞集合に無い
    # 「した質問」や、動詞を持たない「一番最初の質問」が素通りして
    # search_history 経由の誤答になった (実インシデント 2026-08-04 ライブ監査:
    # 「この会話で私が一番最初にした質問は何でしたか。」に対し 6 番目の発言
    # 「今日の日付と現在時刻を教えてください。」と誤答)。
    r"|メッセージ|発言|質問"
    # **過去形の発話動詞 + 発話を指す名詞**。上 2 つの列挙は動詞側と名詞側を
    # 別々に伸ばしてきたため、両方が同時に漏れると素通りする。実インシデント
    # (2026-08-26 ライブ監査 T10-1): 「この会話で私が最初に**伝えた情報**は
    # 何ですか？」が ``伝え`` (動詞集合に無い) と ``情報`` (名詞集合に無い) の
    # 二重の漏れで非マッチ。決定論の位置事実が付かないまま deliberative へ落ち、
    # 注入された **別セッションの記憶** (「神戸に住んでいる」) を「この会話で
    # 最初に伝えた情報」として提示した。会話の 1 ターン目で、この会話には
    # ユーザーの他の発話が存在しないターンだった。
    #
    # 過去形 (``た``) を必須にするのは、未来の依頼 (「最初に伝えたい情報を
    # 整理して」) を位置参照と取り違えないため。
    r"|(?:伝え|述べ|挙げ|示し|言っ|話し|送っ|書い|聞い|尋ね)た"
    r"(?:メッセージ|発言|質問|依頼|指示|こと|内容|情報|話)"
    r"|(?:message|question|thing|request)\s+(?:i|you)\s+"
    r"(?:sent|said|asked|requested)"
    r"|(?:sent|said|asked|requested)",
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


#: 会話全体を走査しないと答えられない質問の 3 系統。
#:
#: (a) 会話そのものへのアンカー (「この会話で」)
#: (b) 位置指定 (「最初に / 最後に 〜した」) — 位置は見えている範囲の端では
#:     なく会話全体の端で決まる
#: (c) 網羅指定 (「全部 / すべて 〜列挙して」)
#:
#: (b)/(c) の対象語は「ユーザー自身が会話の中で行ったこと」に限定する。
#: 「最初に何をすべき？」「全部教えて」のような一般依頼を巻き込むと、
#: 会話と無関係な質問にまで注記が付く。
#: 動詞は活用語尾まで含めて書く。裸の漢字 1 字 (``書`` ``言`` ``聞`` …) にすると
#: 無関係な名詞の内部にヒットする。
#:
#: 実インシデント (2026-08-16 ライブ監査 ターン21): 「新規事業の**企画書**を作る
#: とき、**最初に**固めるべき項目を優先度順に3つ。」が、裸の ``書`` が「企画書」に
#: 当たったことで会話走査質問と判定され、ビジネスの回答の末尾に
#: 「※会話の前半は参照できないため、これより前にも関連する議論があった可能性が
#: あります。」という無関係な断りが付いた。同じ穴は 報告書 / 仕様書 / 文書 /
#: 辞書 / 請求書、あるいは 言語 / 聞き手 / 打ち合わせ 等にもある。
_WHOLE_SESSION_ACTION_TARGET_RE = re.compile(
    r"(?:送(?:っ|り|ら|信)|言(?:っ|い|わ)|聞(?:い|き|か)|尋ね|書(?:い|き|か|け)"
    # 「打ち」は「打ち合わせ」に当たるので、動詞形は「打った」「打ち込」だけ拾う。
    r"|投げ|打(?:った|ち込)|頼(?:ん|み|ま)|命じ|訊(?:い|き|か)"
    r"|読ま|作ら|やらせ|させ)"
    r"|(?:メッセージ|発言|質問|依頼|指示|お願い|やり取り|ファイル操作|操作)"
    r"|(?:asked|said|sent|requested|told\s+you|instructed)",
    re.IGNORECASE,
)
_WHOLE_SESSION_EXHAUSTIVE_RE = re.compile(
    r"(?:全部|すべて|全て|漏れなく|残らず)"
    r"|(?:\ball\b|\bevery\b|\beverything\b)",
    re.IGNORECASE,
)
_WHOLE_SESSION_ENUMERATE_RE = re.compile(
    r"(?:リストアップ|列挙|挙げ|並べ|洗い出|書き出|まとめ)"
    r"|(?:list|enumerate)",
    re.IGNORECASE,
)
#: 文頭の談話標識。「最後に、〜」の「最後に」は位置指定ではなく「ついでに最後の
#: 質問だが」の意で、位置語として数えると「最初」と衝突して曖昧扱いになり、
#: 本来の位置指定 (「最初に読ませたファイル」) を取りこぼす (2026-08-05 ライブ
#: 監査 ターン40)。読点で区切られた文頭のものだけを落とす — 「この会話の最後に
#: 言ったこと」のような文中の位置指定は残す。
_LEADING_DISCOURSE_MARKER_RE = re.compile(
    r"^\s*(?:最後に|さいごに|ついでに|ちなみに|それでは|では|あと)\s*[、,]\s*"
    r"|^\s*(?:lastly|finally|by\s+the\s+way|also)\s*,\s*",
    re.IGNORECASE,
)


def is_whole_session_scope_query(query: str) -> bool:
    """会話全体を見ないと正しく答えられない質問か判定する (純粋関数)。

    「この会話で依頼したファイル操作を全部」「最初に読ませたファイルは」の
    ように、**答えが会話の全範囲に依存する**質問を拾う。会話履歴がワーキング
    メモリの上限で切り詰められていると、見えている範囲だけを根拠に「ありません」
    「〜です」と断定してしまう (2026-08-05 ライブ監査で 2 件発生。ターン19 の
    書き込みが窓外に落ちた状態で「この会話で依頼したファイル操作を全部」→
    「ありません」、ターン7 で読んだ README が窓外の状態で「最初に読ませた
    ファイルは」→ 窓内で最後に読んだ別ファイルを回答)。

    消費側は「実際に切り詰めが起きているか」と AND を取って使うこと。切り詰めが
    無ければ会話全体が見えているので注記は不要であり、そのぶん誤検出のコストは
    ほぼゼロになる。
    """
    if not query:
        return False
    if _SESSION_ANCHOR_ANY_RE.search(query):
        return True
    if not _WHOLE_SESSION_ACTION_TARGET_RE.search(query):
        return False
    body = _LEADING_DISCOURSE_MARKER_RE.sub("", query, count=1)
    is_first = bool(_SESSION_POSITION_FIRST_RE.search(body))
    is_last = bool(_SESSION_POSITION_LAST_RE.search(body))
    if is_first != is_last:
        return True
    return bool(
        _WHOLE_SESSION_EXHAUSTIVE_RE.search(query)
        and _WHOLE_SESSION_ENUMERATE_RE.search(query),
    )


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


# ─────────────────────────────────────────────────────────────────────
# 直前の出力への後方参照 (照応)
# ─────────────────────────────────────────────────────────────────────

#: 「直前に出力された内容」を指す後方参照。
#:
#: 動的ブロック (記憶注入 / RAG / few-shot) は最後の user メッセージの **先頭** に
#: 前置されるため、この種の照応を含むクエリでは参照先の候補として注入ブロックが
#: 生クエリのすぐ上に並ぶ。ラベルで「今回の会話で述べられた内容ではない」と
#: 否定していても、位置的な隣接が勝つ。
#:
#: 実インシデント (2026-08-03 ライブ監査): 「上の内容を箇条書き 5 行にまとめ直して
#: ください」に対し、直前ターンのリモートワークの話ではなく **注入された記憶ブロック**
#: (ユーザー名 / 過去のコマンド実行記録 / 別セッションの大阪府の天気予報) を要約した。
#: 次ターンの英訳依頼にもその汚染がそのまま伝播した。
#:
#: 誤検出のコストは低い — 該当ターンで注入先が同じ user メッセージ内で
#: 「生クエリの前」から「生クエリの後ろ」へ移るだけで、記憶そのものは失われず
#: prefix KV キャッシュも保たれる。ツール発火系の判定 (誤爆するとコマンドが走る)
#: とは非対称なので、取りこぼしを減らす側に倒してよい。
#:
#: 2026-08-16 まで、このケースでは注入先を **system** へ回していた。system は
#: prompt の先頭なので足しても外しても prefix が丸ごと動き、cache_prompt が全損する
#: (実測: last-user 配置 13.2s / system 配置 95.1s / 外して戻す次ターン 99.5s)。
#: 「誤検出のコストは低い」という前提が配置先の選択で崩れていたため、後置へ変更した。
#: 詳細は ``core.inference.build_messages`` の該当箇所を参照。
BACKREFERENCE_TO_OUTPUT_RE = re.compile(
    r"(?:上の|上記の?|前述の?|先(?:ほど|程)の?|さっきの?|直前の)"
    r"|(?:それ|これ|その内容|この内容|上記)を"
    r"|(?<![A-Za-z])(?:the\s+)?above(?![A-Za-z])"
    r"|(?<![A-Za-z])previous(?:ly)?(?![A-Za-z])",
    re.IGNORECASE,
)


def refers_to_previous_output(text: str) -> bool:
    """``text`` が直前に出力された内容を指し示しているか (純粋関数)。"""
    return bool(BACKREFERENCE_TO_OUTPUT_RE.search(text))


# ─────────────────────────────────────────────────────────────────────
# 直前の出力そのものの計量 (「今の回答は何文字?」)
# ─────────────────────────────────────────────────────────────────────
#
# 自分が今出力した文章の文字数・行数は、モデルに数えさせても当たらない。
# 実インシデント (2026-08-05 ライブ監査 ターン33):「今の回答は実際に何字あり
# ましたか？」に対し「488 文字」と回答したが実測は 633 文字。しかもクエリが
# 17 文字と短いため router が short_query → reactive に落とし、ツール判定も
# 検索も走らない経路だった (層をどう直しても「数える道具」が無い)。
#
# ファイルの文字数は read_file のメタ行 (lines / chars) で決定論化済み。
# 直前の出力も同じ扱いにする — 数えるのはコード、モデルは読み上げるだけ。

#: 直前の**自分の出力**を指す照応。BACKREFERENCE_TO_OUTPUT_RE は「上の」等の
#: 指示語だけで成立するが、ここでは計量対象を確定させる必要があるため
#: 「〜の回答 / 文章」まで含む形に限定する (「今の」は上記正規表現に無い)。
#:
#: 指示語と名詞の **隣接** だけを見ていたため、間に動詞が入る普通の言い方
#: (「いま**書いた**要約は何文字でしたか？」) が漏れていた。実インシデント
#: (2026-08-22 ライブ監査): 実測 86 文字の要約に対し「100文字です」と断定した
#: — 機構自体は動いていて、ここで外れて実測値が注入されなかっただけ。
#:
#: 挟めるのは **「あなたが出力した」ことを表す動詞の連体形** だけに限る
#: (`書いた` / `作成した` / `出力した` / `生成した` 等)。任意の語を許すと
#: 「さっき**私が言った**文」のようにユーザー自身の発言を指す照応まで拾い、
#: assistant の直前出力を計量して答えてしまう。
_SELF_OUTPUT_PRODUCED_VERB = (
    r"(?:書|作|作成|出|出力|答え|示|生成|まとめ|返|述べ)[^\s。、]{0,3}?た"
)
_SELF_OUTPUT_REFERENCE_RE = re.compile(
    r"(?:今|いま|先(?:ほど|程)|さっき|直前|上|上記|その|この)の?\s*"
    rf"(?:{_SELF_OUTPUT_PRODUCED_VERB}\s*)?"
    r"(?:回答|返答|答え|文章|文面|出力|説明|要約|本文|テキスト|メッセージ|文)"
    r"|(?:your\s+)?(?:previous|last|above)\s+"
    r"(?:answer|reply|response|text|message|summary|paragraph)",
    re.IGNORECASE,
)
#: 名詞リストに依存しない緩い照応。**名詞を列挙し続ける方式は語彙漏れが構造的に
#: 再発する** — 直前に「動詞の連体形を挟めるようにする」修正 (2026-08-22) を
#: 入れた直後の監査で、今度は名詞側が漏れた。実インシデント
#: (2026-08-22 ライブ監査 2 回目 ターン 74): 「さっきのキャッチコピーは何文字
#: ですか？」で ``キャッチコピー`` が名詞リストに無く、実測 18 文字に対し
#: 「20文字です」と断定した (指定は 20 文字ちょうど = 制約違反の隠蔽)。
#: 生成物の呼び名は無限にあるので、**照応語 + 計量語**だけで成立させ、
#: 誤りやすい側 (ユーザー自身の発話 / 外部対象) を除外条件で落とす。
#:
#: 緩い方では ``その`` / ``この`` を採らない — 「この本は何文字ありますか？」の
#: ような外部対象を巻き込むため。時間的な照応語だけが「直前の自分の出力」を
#: 一意に指す。
_SELF_OUTPUT_REFERENCE_LOOSE_RE = re.compile(
    r"(?:今|いま|先(?:ほど|程)|さっき|直前|上記|一つ前|ひとつ前)"
    r"|(?:your\s+)?(?:previous|last|above)(?![A-Za-z])",
    re.IGNORECASE,
)
#: 産出主体がユーザー自身であることを示す語。あれば assistant の出力ではない。
_MEASURE_USER_AUTHORED_RE = re.compile(
    r"(?:私|僕|俺|自分|わたし|ぼく|ユーザー)(?:が|の)"
    r"|(?<![A-Za-z])(?:i|my|mine)(?![A-Za-z])",
    re.IGNORECASE,
)
#: ファイル・URL を指しているクエリは対象外 (計量対象が直前の出力ではない)。
_MEASURE_EXTERNAL_TARGET_RE = re.compile(
    r"https?://|[A-Za-z]:\\|/\w+/|\.\w{1,5}(?:\s|$|は|を|の|が)",
)
_MEASURE_KIND_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "chars",
        re.compile(
            r"何文字|何字|文字数|字数|character\s*count|how\s+many\s+characters?",
            re.IGNORECASE,
        ),
    ),
    (
        "lines",
        re.compile(
            r"何行|行数|line\s*count|how\s+many\s+lines?", re.IGNORECASE,
        ),
    ),
    (
        "words",
        re.compile(
            r"何語|何単語|単語数|語数|word\s*count|how\s+many\s+words?",
            re.IGNORECASE,
        ),
    ),
)


#: 「あなたは実際に〜したか」— **自分自身の処理経路**についての問い。
#:
#: 会話履歴にはツール実行の痕跡が残らないため、窓を越えた自己申告は base の
#: 事前知識で埋められる。実インシデント (2026-08-22 ライブ監査 2 回目
#: ターン 40 / 100): 「これまでの計算のうち、ツールを使わず暗算したものは
#: どれですか？」→ 実際は ``calculate`` / ``run_command_readonly`` が繰り返し
#: 走っていたのに 17 件すべてを暗算と申告。「あなたが実際に文字数を数えた場面は
#: ありましたか？」→「いいえ、ありません」(ターン 64 で決定論の文字数注記が
#: 入り正答している)。``agent.tool_ledger`` の実記録を注入するための発火条件。
#:
#: 「実行してください」のような **依頼** と取り違えないよう、過去・完了の
#: 問い掛け形 (``〜たか`` / ``〜ましたか`` / ``〜ものはどれ``) を必須にする。
_OWN_PROCESS_SUBJECT_RE = re.compile(
    r"あなた|君(?![が-ん])|自分で|実際に|本当に"
    r"|(?<![A-Za-z])(?:you|did\s+you)(?![A-Za-z])",
    re.IGNORECASE,
)
#: これ自体が「assistant の実行」を一意に指す語。主語が書かれていなくても
#: 発火してよい (「ツールを使わず暗算したものはどれですか？」に「あなた」は
#: 現れないが、他の誰かの実行を指しようがない)。
#:
#: ``実測`` は「実際に測ったか」そのものを指す語で、他人の行為を表す用法が無い。
#: ターン 296「今答えた値のうち、実測できなかったものはどれですか？」が
#: どの条件にも掛からず、直前 6 ターンのシステム情報とは無関係な
#: 「富士山の標高です」を返した。
_OWN_PROCESS_TOOL_NOUN_RE = re.compile(
    r"ツール|(?<![A-Za-z])tool(?:s)?(?![A-Za-z])|コマンド|暗算|実測|実行結果",
)
#: 主語 (``あなた`` / ``実際に``) と組み合わせて初めて自己申告の問いになる動作語。
_OWN_PROCESS_ACTION_RE = re.compile(
    r"(?:実行|使用|起動|検証|計測|測定)し"
    r"|(?:使っ|走らせ|叩い|数え|測っ|調べ|確かめ)",
)
#: 過去・完了の問い掛け形。依頼形 (「〜を挙げてください」) も、対象が
#: **過去形の連体修飾** (``実行したツール`` / ``使ったコマンド``) なら自己申告の
#: 問いになる。実インシデント (2026-08-22 ライブ監査 2 回目 セット2 最終ターン):
#: 「ここまでのやり取りで、あなたが実際に実行したツールを全部挙げてください。」が
#: どの条件にも掛からず台帳が注入されないまま「実行したツールはありません」と
#: 回答した (実際は system_hardware_info / evoref_runtime_info が走っている)。
#: ``使えるツール`` は ``使え+る`` なのでこの形には当たらず、目録の問い
#: (``tool_inventory_question``) との棲み分けが保たれる。
#: 実行を表す動作語の **過去形**。
#:
#: 「実行した」「使った」という過去形そのものが「もう起きたことを訊いている」
#: 構造を表す。名詞化辞 (もの / 場面 / 箇所 / ケース / とき) を並べる書き方は
#: 必ず漏れる — 実インシデント (2026-08-23 ライブ監査セット 1 T4-10):
#: 「いま計算した中で、あなたが電卓ツールを **使ったのは** どれですか？」が
#: 最頻出の名詞化辞「の」を列挙に持たないため非マッチ。台帳が注入されないまま
#: 「電卓ツールは使っていません」と作話した (実際は ``calculate`` が 7 回実行済)。
#:
#: 誤爆の心配は :func:`own_process_question` 側の AND 条件が引き受ける
#: (ツール語があるか、主語 + 動作語が揃うか)。ここは「過去の実行を訊いている」
#: 構造だけを見る。
_OWN_PROCESS_ACTION_PAST_RE = re.compile(
    r"(?:実行|使用|起動|検証|計測|測定|計算)した"
    r"|(?:使っ|走らせ|叩い|数え|測っ|調べ|確かめ|やっ)た",
)
_OWN_PROCESS_PAST_ASK_RE = re.compile(
    r"(?:まし|でし|かっ)たか|したか(?![らるれ])|ましたっけ"
    r"|たものは(?:どれ|何)|た(?:場面|箇所|もの|ケース|とき)"
    r"|(?:実行し|使用し|走らせ|叩い|使っ)た(?:ツール|コマンド|操作|もの)"
    r"|(?<![A-Za-z])did\s+you(?![A-Za-z])|(?<![A-Za-z])have\s+you(?![A-Za-z])",
)


#: 「ファイルに保存して」型の永続化依頼。**保存先が書かれていなくても**
#: 拾う点が ``WRITE_VERB_RE`` との違い。
#:
#: 実インシデント (2026-08-22 ライブ監査 2 回目 ターン 252):
#: 「ファイルに保存しておいて。」(12 文字) が ``_is_local_write_intent`` の
#: パス必須条件を外れ、``short_query`` → reactive に落ちてツール判定に一度も
#: 到達せず、「ファイル保存機能は利用できないため、保存できません。」と回答した。
#: **同じ会話のターン 122 で ``write_file`` が成功している** ので、能力が無いと
#: いう説明そのものが誤りだった。保存先が無いなら聞き返すのが正しい応答で、
#: それは deliberative 側でしか出せない。
#:
#: ``WRITE_VERB_RE`` をそのまま使うと ``修正`` / ``変更`` / ``更新`` / ``実装``
#: まで拾い、ごく普通の依頼が軒並み deliberative へ上がる。**永続化の対象語**
#: (ファイル / 保存 / 書き出し) との共起に限定する。
_PERSIST_REQUEST_RE = re.compile(
    r"(?:ファイル|ふぁいる|(?<![A-Za-z])file(?![A-Za-z]))"
    r"[^。．\n]{0,12}?(?:保存|書[きい]出|書[きい]込|出力|セーブ|save|write)"
    r"|保存し(?:て|と[いこ]|ておい)"
    r"|書[きい]出し(?:て|と[いこ]|ておい)"
    r"|(?<![A-Za-z])save\s+(?:it|this|that)?\s*to\s+(?:a\s+)?file(?![A-Za-z])",
    re.IGNORECASE,
)


def persist_request(query: str) -> bool:
    """クエリが「ファイルへ保存する」操作を求めているか (純粋関数)。

    保存先パスの有無は問わない — パスが無いこと自体が「聞き返す」理由になる。
    """
    return bool(query) and bool(_PERSIST_REQUEST_RE.search(query))


#: ファイル名 / パスを名指ししている形。拡張子付きの名前か、ドライブレター付きの
#: 絶対パスを要求する (「メモ」のような裸の語では発火しない)。
_NAMES_FILE_TARGET_RE = re.compile(
    r"[A-Za-z]:[\\/][^\s\"']+"
    r"|[\w\-.]+\.(?:txt|md|json|csv|log|yaml|yml|py|ts|js|html|xml|ini|toml)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def names_file_target(query: str) -> bool:
    """クエリがファイル名 / パスを名指ししているか (純粋関数)。

    「保存して」のような動詞を伴わず **引数だけを与えるターン** を拾うための
    述語。単独では意味が決まらないので、呼出側が文脈 (直前の user 発話が保存
    依頼だったか) と組み合わせて使う。
    """
    return bool(query) and bool(_NAMES_FILE_TARGET_RE.search(query))


def own_process_question(query: str) -> bool:
    """クエリが「自分が実際に何を実行したか」を尋ねているか (純粋関数)。"""
    q = query or ""
    if not q:
        return False
    # 「過去の実行を訊いている」構造。名詞化辞の列挙 (旧実装) ではなく動作語の
    # 過去形で受ける (:data:`_OWN_PROCESS_ACTION_PAST_RE` のコメントを参照)。
    if not (
        _OWN_PROCESS_PAST_ASK_RE.search(q)
        or _OWN_PROCESS_ACTION_PAST_RE.search(q)
    ):
        return False
    if _OWN_PROCESS_TOOL_NOUN_RE.search(q):
        return True
    return bool(
        _OWN_PROCESS_SUBJECT_RE.search(q) and _OWN_PROCESS_ACTION_RE.search(q),
    )


def self_output_measure_kinds(query: str) -> tuple[str, ...]:
    """直前の自分の出力の計量を尋ねているか判定する (純粋関数)。

    Returns:
        ``("chars",)`` / ``("lines",)`` / ``("chars", "lines")`` 等。該当
        しなければ空タプル。
    """
    if not query:
        return ()
    if not _SELF_OUTPUT_REFERENCE_RE.search(query) and not (
        _SELF_OUTPUT_REFERENCE_LOOSE_RE.search(query)
        and not _MEASURE_USER_AUTHORED_RE.search(query)
    ):
        return ()
    if _MEASURE_EXTERNAL_TARGET_RE.search(query):
        return ()
    return tuple(
        kind for kind, pattern in _MEASURE_KIND_PATTERNS if pattern.search(query)
    )


# ─────────────────────────────────────────────────────────────────────
# 数値計算を求めるクエリ
# ─────────────────────────────────────────────────────────────────────

#: クエリ中の数値リテラル。
NUMBER_LITERAL_RE = re.compile(r"\d+(?:\.\d+)?")

#: 「値を尋ねている」手掛かり語。
#:
#: 単位を列挙すると必ず取りこぼす (実インシデント 2026-07-29 ライブ監査:
#: 「残りは何リットルですか？」「進む距離は何kmですか？」がどちらも手掛かり語
#: なしと判定され、base の暗算で誤答した)。「何<単位>+文末表現」という構造で
#: 受ける。数値を含まないクエリは後段の数値チェックで落ちるため、「これは何
#: ですか」のような非数量文で判定が緩むことはない。
CALCULATION_CUE_RE = re.compile(
    # ``どれだけ`` 系 (量を問う疑問詞) は ``いくつ`` / ``いくら`` と同じ役割
    # なのに欠けていた。「どれだけ増えましたか？」が手掛かり語なしと判定され、
    # 差分クエリが router で short_query → reactive へ落ちる原因の片方だった
    # (2026-08-26 ライブ監査。もう片方は ANAPHORIC_OPERAND_RE 側)。
    r"(?:いくつ|いくら|どれ(?:だけ|くらい|ほど)"
    r"|何[個円分秒時間日年枚人倍%％]|何キロ|何マイル|何時間"
    # 文末表現に「ます」を含める。「進みます」「増えます」「減ります」など
    # 動作動詞の丁寧形が最も普通の言い方なのに、``です|でしょ|になり|かかり|
    # ありま`` だけでは受からなかった (実インシデント 2026-08-08 ライブ監査:
    # 「時速240kmで2時間30分走ると何km進みますか。」が手掛かり語なしと判定
    # され、base の暗算で 540km と誤答。正解は 600km)。
    r"|何[ぁ-んァ-ヴーA-Za-z一-龥%％]{0,6}?(?:です|でしょ|になり|かかり|ありま|ます)"
    # 丁寧形を列挙しても辞書形が漏れる。「何<単位>...？」と疑問符で閉じる形は
    # 文末表現に依らず受ける (実インシデント 2026-08-12 ライブ監査:
    # 「時速72km で 45 分走ると何 km 進む？」が手掛かり語なしと判定され、
    # ツール判定に一度も到達せず base の暗算で 6km と誤答。正解は 54km)。
    # 数値ゼロのクエリ (「これは何？」) は後段の数値チェックで落ちる。
    r"|何[ぁ-んァ-ヴーA-Za-z一-龥%％\s]{0,8}?[?？]"
    r"|合計|総額|平均|割合|求め|計算"
    r"|(?<![A-Za-z])how\s+(?:much|many)(?![A-Za-z])"
    r"|(?<![A-Za-z])what\s+is(?![A-Za-z])|(?<![A-Za-z])total(?![A-Za-z]))",
    re.IGNORECASE,
)

#: 環境依存事実クエリ (実行可能コマンドで答える層が扱う) を巻き込まないための除外語。
CALCULATION_EXCLUDE_RE = re.compile(
    r"(?:何月|何日|何曜日|現在時刻|日付|バージョン|version"
    r"|ディスク|メモリ|使用量|(?<![A-Za-z])CPU(?![A-Za-z]))",
    re.IGNORECASE,
)


#: 前提への同意を求める確認形。「〜ですよね？」「〜で合っていますか」等。
#:
#: 過去形 (でした / だった / ました) と文末の句点を取りこぼしていた。実インシデント
#: 2026-08-10 ライブ監査:「平均のほうの毎分リクエスト数は 5,787 **でしたよね。**」が
#: 確認形と判定されず、検証されないまま追認された (自分が 4 ターン前に calculate で
#: 出した 3,472.2 と矛盾していた)。文末の ``。`` ``．`` も許す。
#:
#: router (誤前提を検索せず追認しないための分類) と deliberative (未検証の主張への
#: 注記) が同じ判定を使うため core に置く。
PREMISE_CONFIRMATION_RE = re.compile(
    r"(?:です|ます|でした|ました|だ|だった|でしょう)よね[?？]?[。．]?$"
    r"|(?:で|て)?(?:合って|あって|正しい|間違いな)(?:い)?(?:ます|です)?"
    r"(?:か|よね|ね)[?？]?[。．]?$"
    r"|(?:じゃない|ではない|ないです)(?:か|よね)[?？]?[。．]?$"
    r"|\b(?:right|correct)\?$"
    r"|\b(?:isn't|aren't|doesn't|don't)\s+(?:it|they|that)\?$",
)

#: 桁区切り入りの数字 (``5,787`` / ``1,234,567``)。
_GROUPED_NUMBER_LITERAL_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")

#: 「会話に出ていない」と判定する最小桁数。1〜2 桁は「3 倍」「1 日」のように
#: 文中で普通に現れるため、突き合わせても信号にならない。
_CLAIM_NUMBER_MIN_DIGITS = 3


def _normalized_numbers(text: str) -> set[str]:
    """桁区切りを外した数値リテラルの集合 (純粋関数)。"""
    found = set(NUMBER_LITERAL_RE.findall(text))
    found.update(m.replace(",", "") for m in _GROUPED_NUMBER_LITERAL_RE.findall(text))
    return found


def unverified_claim_numbers(query: str, context: str) -> tuple[str, ...]:
    """確認形のクエリが持ち込んだ「会話に無い数値」を返す (純粋関数)。

    ユーザーが確認を求める形で数値を挙げたとき、その値が会話に一度も現れて
    いなければ、同意してよい根拠が無い。実インシデント 2026-08-10 ライブ監査:
    自分が calculate で出した 3,472.2 があるのに「5,787 でしたよね」に
    「はい、5,787 です」と追認した。注記を添えると 3/3 で「その値は出ていません。
    …3,472.2 です」と訂正するようになる (実測)。

    確認形でないクエリ (ユーザーが新しい前提を述べているだけ) は対象外。
    """
    if not PREMISE_CONFIRMATION_RE.search(query.strip()):
        return ()
    known = _normalized_numbers(context)
    unverified: list[str] = []
    for raw in _GROUPED_NUMBER_LITERAL_RE.findall(query):
        plain = raw.replace(",", "")
        if plain not in known and raw not in unverified:
            unverified.append(raw)
    grouped_plain = {
        m.replace(",", "") for m in _GROUPED_NUMBER_LITERAL_RE.findall(query)
    }
    for raw in NUMBER_LITERAL_RE.findall(query):
        digits = raw.replace(".", "")
        if len(digits) < _CLAIM_NUMBER_MIN_DIGITS:
            continue
        # 桁区切り表記の一部 (5,787 の 787) は重複計上しない
        if any(raw in g for g in grouped_plain):
            continue
        if raw not in known and raw not in unverified:
            unverified.append(raw)
    return tuple(unverified)


#: 被演算子が直前ターンにしかないことを示す照応。「その差」「先ほどの合計」等。
#:
#: 数量を表す名詞まで含めて限定する。指示語だけ (「それはどう思う?」) を拾うと
#: 計算でないターンまで deliberative に回るため。なお呼び出し側は
#: ``CALCULATION_CUE_RE`` (「何分になりますか」等) と **context に数値があること**
#: も同時に要求するので、この正規表現単体で判定が緩むことはない。
#:
#: 指示語 + 数量名詞だけでは **差分クエリ** が漏れる。基準値の指し方が指示語
#: ではなく序数的 (「最初の」「当初の」「前回」) で、対象の名詞もドメイン語
#: (「在庫」「残高」) なので、閉じた数量名詞の一覧には永久に載らない。
#:
#: 実インシデント (2026-08-26 ライブ監査): 在庫の推移を数ターンやり取りした後の
#: 「最初の在庫からいくつ減りましたか？」(17 文字) が照応にも数量名詞にも
#: 掛からず short_query → reactive に落ち、差分ではなく **現在値** を答えた。
#:
#: そこで 2 つの形を足す。いずれも「被演算子が前ターンにある」ことを表す:
#:   (a) 基準時点への参照 + 比較の格助詞 (「最初の在庫**から**」「前回**より**」)
#:   (b) 量を問う疑問詞が増減動詞に直結する形 (「どれだけ**減**りましたか」)
#: 呼出側は ``CALCULATION_CUE_RE`` と **context に数値があること** を併せて
#: 要求するため、(b) 単体で判定が緩むことはない (「気分は変わりましたか」は
#: 手掛かり語が無いので通らない)。
ANAPHORIC_OPERAND_RE = re.compile(
    r"(?:その|それ|この|これ|先(?:ほど|程)の|さきほどの|さっきの|上記の|直前の)\s*"
    r"(?:差|値|数値|数|合計|総額|金額|件数|回数|結果|平均|割合|時間|人数)"
    r"|(?:最初|当初|もともと|元々|開始時|スタート時|初期|前回|以前)"
    r"[^。？?\n]{0,12}?(?:から|より|と比[べく]|に比[べく])"
    r"|(?:いくつ|いくら|どれ(?:だけ|くらい|ほど)|何[0-9]*(?:個|円|件|人|枚|台|冊|％|%))"
    r"\s*(?:も|ほど|くらい|ぐらい)?\s*"
    r"(?:減|増|変わ|伸び|縮ま|上が|下が|落ち)"
    r"|(?<![A-Za-z])(?:that|the)\s+"
    r"(?:difference|total|sum|result|number|amount|average|count)(?![A-Za-z])",
    re.IGNORECASE,
)


def looks_like_numeric_question(query: str, context: str = "") -> bool:
    """式は書かれていないが数値計算の答えを求めているクエリか (純粋関数)。

    2026-07-27 ライブ検証: 「1マイルは約1.609キロメートルです。42.195キロ
    メートルは何マイルですか？」が全層 no_tool となり、base の暗算で 26.195
    と誤答した (正解 26.224)。同じ計算を式で書くと層1 が calculate へ流して
    正答するため、差は「式が書かれているか」だけだった。

    値を尋ねる手掛かり語があり、環境依存事実クエリの語を含まず、クエリ自身に
    数値が 1 つ以上あり、かつクエリ + ``context`` (直近の会話) を合わせて数値が
    2 つ以上ある場合のみ True。

    ``context`` を数える理由 (2026-07-28 ライブ検証): 「1マイルは何キロ?」の
    直後に「では 26.2 マイルのフルマラソンは何キロですか。」と尋ねると、被演算子
    の片方 (換算率 1.61) は直前ターンにしか無いためクエリ単独では 1 数値となり、
    本フィルタで弾かれて base の暗算に倒れていた (42.19 と誤答。正解 42.16)。
    手掛かり語と「クエリ自身に数値が要る」制約は残すので、数値を含まない雑談で
    判定が緩むことはない。

    ``agent.tool_call_judge`` (calculate フォールバックの事前フィルタ) と
    ``agent.router`` (計算クエリを reactive へ落とさないための分類) が同じ判定を
    使うため core に置く。``tool_call_judge`` は ``router`` を import している
    ので、逆向きの参照は循環になり作れない。
    """
    if CALCULATION_EXCLUDE_RE.search(query):
        return False
    if not CALCULATION_CUE_RE.search(query):
        return False
    query_numbers = NUMBER_LITERAL_RE.findall(query)
    if not query_numbers:
        # 被演算子が **すべて** 直前ターンにある形。照応で数量を名指ししている
        # ときだけ許す (実インシデント 2026-08-10 ライブ監査: 525.6 分と 262.8 分
        # を出した直後の「その差を月あたりに直すと何分になりますか。」が
        # 数値ゼロで弾かれ、router で short_query → reactive に落ちてツール判定に
        # 一度も到達せず、base の暗算で 13.4 分と誤答した。正解 21.9 分)。
        if not ANAPHORIC_OPERAND_RE.search(query):
            return False
        return bool(NUMBER_LITERAL_RE.search(context))
    if len(query_numbers) >= 2:
        return True
    return bool(NUMBER_LITERAL_RE.search(context))


#: 「あなたは何のツールが使えるか」を尋ねる問い。ツール目録は決定論で答えられる
#: 事実 (ToolsRegistry が SSOT) なのに、チャット応答パスの system プロンプトには
#: 一覧が載っていない。ツール選択は別レイヤ (ToolCallJudge / grammar 分類器) が
#: 担うため、base はツールの存在自体を知らないまま答える。
#:
#: 実インシデント (2026-08-14 ライブ監査 ターン33): 「あなたが今使えるツールを
#: 全部列挙してください。推測せず、実際に利用可能なものだけで。」に対し
#: 「現在、私が直接利用できるツールはありません」と回答した。同じ会話で
#: calculate / run_command_readonly / search_history / write_file が実行済み。
_TOOL_INVENTORY_SUBJECT_RE = re.compile(
    r"(?:ツール|tool)"
    r"|(?:機能|できること|出来ること)",
    re.IGNORECASE,
)
#: 「一覧を出せ / 何が使えるか」に相当する問いかけ。
_TOOL_INVENTORY_ASK_RE = re.compile(
    r"(?:使え|使用でき|利用でき|呼べ|実行でき)"
    r"|(?:一覧|列挙|リスト|教えて|何がある|どんな|何ができ|何が出来)"
    r"|(?:what|which|list)\b",
    re.IGNORECASE,
)
#: ツール「を使って何かをしろ」という依頼を目録質問と取り違えないための除外。
#: 「このツールでファイルを読んで」等は目録ではなく実行依頼。
_TOOL_INVENTORY_EXCLUDE_RE = re.compile(
    r"(?:を使って|で実行|を実行して|して[くだ]さい\s*$)"
    r"|[A-Za-z]:[\/]",
)


#: evoref 自身の実行構成を尋ねるクエリ (`evoref_runtime_info` の発火条件)。
#:
#: 「あなたはどのモデルか」「コンテキストサイズは」「llama-server のポートは」に
#: 答える経路がどこにも無かった。実インシデント (2026-08-22 ライブ監査):
#: 「今動いているモデルの名前を教えてください。」→「私は「Alice」という名前で
#: 対応しています」(インスタンス名であってモデル名ではない)、「あなたが使っている
#: 埋め込みモデルは？」→「特定の埋め込みモデル名を保持したり開示したりする仕様では
#: ありません」(存在しない方針の捏造)、ポート / n_ctx →「確認できていません」。
#:
#: **裸の「モデル」は拾わない**。機械学習一般の話 (「このモデルの精度は？」
#: 「モデルの汎化性能」) に当たるため、自己を指す語との近接を要求する。
#: 逆に `ベースモデル` / `埋め込みモデル` / `n_ctx` / `llama-server` は
#: それ自体が evoref の構成語なので単独で受ける。
#:
#: バージョンも同じ理由で **自己を指す語との近接を要求する** (「Python の
#: バージョンは？」は evoref の構成ではない)。稼働時間 / uptime / エディションは
#: チャット相手以外の主語を取りにくいので単独で受ける。
#: 実インシデント (2026-08-22 ライブ監査 2 回目 ターン 6/7/9):
#: 「あなたのバージョン番号を教えてください。」→「バージョン番号は提供されて
#: いません」(``/api/status`` は 0.0.58 を返す)、「現在の稼働時間 (uptime) は
#: どれくらいですか？」→「確認できるツールが利用できないため確認できていません」、
#: 「あなたは Free 版ですか Pro 版ですか？」→「Free版やPro版といった商用
#: ライセンスのカテゴリには該当しません」(実際は edition=develop)。
#:
#: 定義をここに置くのは ``agent.tool_judge_signals`` (ツール発火) と
#: ``agent.router`` (層分類) の両方が同じ判定を使うため。router 側に無いと
#: 短い自己構成クエリが ``short_query`` で reactive_light に落ち、ツール結果も
#: 決定論の事実注記も無いまま base の作話に倒れる。
RUNTIME_INFO_QUERY_RE = re.compile(
    r"(?:ベースモデル|(?<![A-Za-z])base\s*model(?![A-Za-z])"
    r"|埋め込みモデル|(?<![A-Za-z])embedding\s*model(?![A-Za-z])"
    r"|(?<![A-Za-z_])n_ctx(?![A-Za-z_])"
    r"|コンテキスト\s*(?:サイズ|長|ウィンドウ)"
    r"|(?<![A-Za-z])context\s*(?:size|window|length)(?![A-Za-z])"
    r"|llama[-_]?server"
    r"|(?:あなた|君|自分|今\s*(?:動いて|使って|ロードされて)いる"
    r"|いま\s*(?:動いて|使って|ロードされて)いる|稼働中|使用中|動作中)"
    r"[^。．\n]{0,10}?モデル"
    r"|(?<![A-Za-z])(?:what|which)\s+model\s+(?:are\s+you|is\s+(?:this|running))"
    # 稼働時間 — チャット相手以外の主語を取りにくいので単独で受ける。
    r"|稼働時間|起動してから|いつから(?:動いて|起動して|稼働して)"
    r"|(?<![A-Za-z])uptime(?![A-Za-z])"
    # エディション (Free / Pro / Develop)。
    r"|エディション|(?<![A-Za-z])edition(?![A-Za-z])"
    r"|(?:Free|Pro|フリー|プロ)\s*版"
    # バージョン — 自己を指す語との近接を要求する。
    r"|(?:あなた|君|自分|evoref|アシスタント|このシステム|本システム)"
    r"[^。．\n]{0,12}?(?:バージョン|(?<![A-Za-z])version(?![A-Za-z]))"
    r")",
    re.IGNORECASE,
)


def runtime_info_question(query: str) -> bool:
    """クエリが evoref 自身の実行構成を尋ねているか (純粋関数)。"""
    return bool(query) and bool(RUNTIME_INFO_QUERY_RE.search(query))


#: クエリ **全体** が「続きを書いて」を意味しているか。文中一致は取らない。
#:
#: 定義をここに置くのは ``api.chat._continuation`` (継続生成の発火) と
#: ``agent.router`` (層分類) の両方が同じ判定を使うため。router 側に無いと
#: 「続けて」(3 文字) が ``short_query`` → reactive_light に落ち、直近数件の
#: 履歴だけを見たモデルが **直前の user 発話を逐語で復唱**する。実インシデント
#: (2026-08-25 ライブ監査 T6-5): 長文生成が正常終了したため継続待ちは武装
#: しておらず、「続けて」が通常ターンとして reactive_light へ落ちて
#: 「Pythonのデコレータについて2000文字程度で詳しく解説してください。」という
#: **前ターンの依頼文そのもの**を返した。
#:
#: 「続けて説明してください」のような *別の依頼* を継続に化けさせないため、
#: 部分一致は取らない (継続経路は元の依頼文を見ないので、誤発火すると質問
#: そのものが失われる)。
CONTINUATION_REQUEST_RE = re.compile(
    r"^\s*(?:"
    # 続き / つづき (+ を) (+ 依頼末尾)
    r"(?:続き|つづき)(?:を)?"
    r"(?:書いて|出力して|生成して|見せて|教えて)?"
    r"(?:ください|下さい|お願い(?:します|いたします)?|どうぞ)?"
    # 続けて / つづけて / 続行 / 再開
    r"|(?:続けて|つづけて|続行|再開)(?:して)?"
    r"(?:ください|下さい|お願い(?:します|いたします)?)?"
    # 英語
    r"|(?:please\s+)?(?:continue|go\s+on|keep\s+going|carry\s+on)(?:\s+please)?"
    r")\s*$",
    re.IGNORECASE,
)

#: 末尾の句読点・記号だけを落とす (「続けて。」「continue!」を同一視する)。
_CONTINUATION_TRAILING_PUNCT_RE = re.compile(r"[\s。．\.！!？\?、,]+$")


def continuation_request(query: str) -> bool:
    """クエリ全体が「続きを書いて」だけを意味しているか (純粋関数)。"""
    normalized = _CONTINUATION_TRAILING_PUNCT_RE.sub("", (query or "").strip())
    if not normalized:
        return False
    return CONTINUATION_REQUEST_RE.match(normalized) is not None


def tool_inventory_question(query: str) -> bool:
    """クエリが「使えるツール / 機能の一覧」を尋ねているか (純粋関数)。

    主語 (ツール / 機能 / できること) と問いかけ (使える / 一覧 / 何が…) の
    両方が現れ、実行依頼の語が無い場合だけ True。曖昧な文は False を返して
    従来経路へ委ねる。

    **過去の実行を訊く問い (:func:`own_process_question`) は目録ではない**。
    ``ASK`` 側に ``教えて`` / ``列挙`` が入っているため、「これまでにあなたが
    使ったツール名を教えてください」のような自己申告の問いも主語 (``ツール``)
    と噛み合って目録質問になってしまう。目録は
    ``deliberative._append_tool_inventory_fact`` が **判定経路ごと短絡**
    させるので、先に目録が立つと台帳 (``_append_tool_ledger_fact``) には
    一生到達しない。実インシデント (2026-08-25 ライブ監査 T4-10 / T10-4):
    同一セッションで ``write_file`` が 4 回成功しているのに「ファイル書き込みに
    使ったツールはありません」、``search_history`` 実行済みなのに「この会話では
    ツールを使用していません」と回答した (backend.log 側は
    ``Tool inventory fact pinned`` のみで ``Tool ledger fact pinned`` が無い)。
    """
    if not query:
        return False
    if own_process_question(query):
        return False
    if _TOOL_INVENTORY_EXCLUDE_RE.search(query):
        return False
    if not _TOOL_INVENTORY_SUBJECT_RE.search(query):
        return False
    return bool(_TOOL_INVENTORY_ASK_RE.search(query))


#: 現在日時 / 日付を尋ねるクエリ。``agent.tool_call_judge`` (datetime コマンド
#: 合成) と ``agent.router`` (executable_query 分類) が同じ判定を使う。
#:
#: 以前は両モジュールが個別に正規表現を持っており、``(?!間)`` ガードが
#: tool_call_judge 側にしか無い等の食い違いがあった。ここを SSOT にする。
#:
#: 英語側は以前 ``now`` / ``date`` / ``time`` を **裸で** 拾っていた。これらは
#: 談話副詞・一般名詞として頻出するため、時刻と無関係な文でツールが発火する。
#: 本パターンは aux の否定票を上書きする高特異度扱い (``_upgrade_command_via_aux``
#: の降格例外) なので、誤検出はそのまま無駄なツール実行になる。
#: 実インシデント (2026-08-14 ライブ監査 ターン28/29):
#: 「Please answer in English **from now** until I say otherwise. What are the
#: three main benefits of using type hints in Python?」と
#: 「**Now** give me a JSON object with keys ...」の 2 回、現在時刻の取得
#: コマンドが撃たれた。どちらも日時とは無関係。
#: 疑問構文・明示的な「current/today's」構文に限定して受ける。
#: ``時刻`` は「現在時刻」だけを拾っていた。「今の時刻を教えてください」
#: 「現在の時刻は？」のように **の** が挟まる普通の言い方が漏れており、
#: 直前のターンで同じコマンドが現在日時を返していたのに
#: 「現在の正確な時刻は確認できていません」と答えた
#: (実インシデント 2026-08-22 ライブ監査 ターン20)。
#: ``時刻表`` (時刻表アプリ / 列車の時刻表) だけ除外する。
DATETIME_QUERY_RE = re.compile(
    r"(?:何時(?!間)|何月|何日(?!間)|何曜日"
    r"|(?:日時|日付)(?!型|形式|フォーマット|カラム|列)|時刻(?!表)"
    # 「何日間」は ``何日(?!間)`` で意図的に外している (「有給は何日間？」の
    # ような、今日と無関係な期間の問いを拾わないため)。ただし **年つきの
    # 絶対日付が直前にある** 場合は 2 点間の日数を数える問いなので通す
    # (``tool_judge_commands._day_count_command`` が両端を Python に数えさせる)。
    # 年を必須にすることで、ツールが答えを出せない形は従来どおり素通りする。
    r"|(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}[-/]\d{1,2}[-/]\d{1,2})[^\n]{0,40}?何日間"
    r"|(?<![A-Za-z])what(?:'s|’s|\s+is|\s+are|\s+was)?"
    r"[^.?!\n]{0,15}?(?<![A-Za-z])(?:date|time|day)(?![A-Za-z])"
    r"|(?<![A-Za-z])current\s+(?:date|time)(?![A-Za-z])"
    r"|(?<![A-Za-z])today'?s?\s+date(?![A-Za-z])"
    r"|(?<![A-Za-z])day\s+of\s+the\s+week(?![A-Za-z])"
    r"|(?<![A-Za-z])(?:tell|give)\s+me[^.?!\n]{0,20}?"
    r"(?<![A-Za-z])(?:date|time)(?![A-Za-z]))",
    re.IGNORECASE,
)


#: 問い・依頼を表すマーカー。これが 1 つでもあれば平叙の自己申告ではない。
#: ``agent.router._ENV_FACT_ASK_RE`` より広く取る (「調べたい」「表示して」等の
#: 語幹も拾う) — こちらは **実行を止める** 側の判定なので、取りこぼしより
#: 過剰抑止の方が害が大きい。
_REQUEST_MARKER_RE = re.compile(
    r"[?？]"
    r"|(?:教えて|おしえて|ですか|でしょうか|ありますか|ますか"
    r"|知りたい|見たい|調べ|確認|表示|出力|見せ|示して|挙げ|列挙|一覧"
    r"|ください|下さい|くれる|ほしい|欲しい|頂け|いただけ|できますか)"
    r"|(?:what|which|how|why|when|where|who"
    r"|show|tell|list|print|check|give)",
    re.IGNORECASE,
)

#: 平叙文の文末 (「〜です。」「〜使っています。」「〜しました」)。
#: 体言止め (「PC のスペック」「hostname」) は含めない — 短い名詞句クエリは
#: 実際の問い合わせとして頻出するため。
_STATEMENT_TAIL_RE = re.compile(
    r"(?:です|ます|でした|ました|ている|ています|でいる|だった|かった"
    r"|しました|なりました|らしい|そうです|らしいです)"
    r"\s*[。．.!！]*\s*$",
)


def is_plain_statement(query: str) -> bool:
    """問い・依頼のマーカーが無く、平叙の文末で終わる **自己申告** か (純粋関数)。

    語彙一致だけで実行を決めるルール表は、ユーザーの自己紹介・状況説明にも
    当たる。実インシデント (2026-08-19 ライブ監査 ターン3):
    「愛用エディタは Neovim で、ターミナルは Windows Terminal を使っています。」
    という **単なる報告** に ``Windows`` が部分一致し、``run_command_readonly``
    で ``platform.platform()`` が撃たれた (17 秒消費し、回答も OS の話に流れた)。

    ``asks_environment_fact`` (router) の裏返しだが、こちらは「問いのマーカーが
    無い」だけでは False を返さない。「PC のスペック」「hostname」のような
    体言止めの問い合わせを止めないためで、**平叙の文末で終わる場合に限って**
    True を返す。
    """
    if not query:
        return False
    if _REQUEST_MARKER_RE.search(query):
        return False
    return bool(_STATEMENT_TAIL_RE.search(query.strip()))
