"""Self-RAG 判定 (ルールベース / LLM 呼び出しなし)

検索結果の品質判定 (:class:`RetrievalQualityJudge`) はベクトル閾値、検索
必要性判定 (:class:`RetrievalNecessityJudge`) は正規表現ルールで完結する。
判別不能な ``uncertain`` ケースは呼出側 (``search_pipeline``) が embedding
決定論的リコール (``rag_judge_recall``) で補い、それでも決まらなければ
安全側の ``retrieve`` に倒す。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.free.core.intent_vocab import session_self_reference_pattern_ja
from backend.free.core.locale_patterns import is_en_locale
from backend.free.document_nouns import (
    DOCUMENT_NOUNS_NEEDS_SUFFIX,
    DOCUMENT_NOUNS_NEEDS_SUFFIX_EN,
    DOCUMENT_NOUNS_STANDALONE,
    DOCUMENT_NOUNS_STANDALONE_EN,
)
from backend.log_config import get_logger

if TYPE_CHECKING:

    from backend.debug_logger import DebugLogger

logger = get_logger("rag.self_rag_judge")

VALID_QUALITIES = {"high", "medium", "low"}

# 検索スキップパターン
SKIP_PATTERNS = re.compile(
    r"(こんにちは|こんばんは|おはよう|ありがとう|了解|OK|はい|いいえ|さようなら)",
    re.IGNORECASE,
)

# SKIP_PATTERNS の英語版。短い英単語 ("no"/"yes") の部分一致誤爆
# ("know"/"yesterday" 等) を防ぐため \b で単語境界を必須にする。
SKIP_PATTERNS_EN = re.compile(
    r"\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening)"
    r"|thanks?(?:\s+you)?|thank\s+you"
    r"|ok(?:ay)?|yes|yeah|yep|no|nope"
    r"|bye|goodbye|see\s+you)\b",
    re.IGNORECASE,
)

# 外部 fetch 意図パターン (確実シグナルのみ)
# URL を含む / 明示的 fetch 動詞は 100% fetch 意図とみなし RAG をスキップ。
# リアルタイムキーワード (ニュース / 株価 / 天気 等) はLLM 判定に委譲し、
# 固定キーワードリストの陳腐化を避ける。旧 Phase 2 (learned_pattern 化) の
# 代替として embedding 決定論的リコール (backend.free.memory.pipeline.
# rag_judge_recall) を導入済み — 意味的類似性が支配的な判定には正規表現
# キーワードより embedding の方が適合するため。
FETCH_INTENT_PATTERNS = re.compile(
    r"(https?://"
    r"|フェッチ|fetch"
    r"|アクセスして|アクセスし"
    r"|取得して|取得しなおして|取り直して|再取得"
    r"|ブラウズ|browse|ダウンロード|download)",
    re.IGNORECASE,
)

# FETCH_INTENT_PATTERNS の英語版。
FETCH_INTENT_PATTERNS_EN = re.compile(
    r"(https?://"
    r"|\bfetch\b"
    r"|\baccess\s+(?:it|that|the\s+(?:page|site|url|link))\b"
    r"|\bretrieve\b"
    r"|\bget\s+it\s+again\b|\bre-?fetch\b|\bre-?download\b"
    r"|\bbrowse\b|\bdownload\b)",
    re.IGNORECASE,
)

# 質問マーカーパターン
# これらを含むクエリは多ターン会話でも常に retrieve とする。
# 多ターン会話の途中で新しい知識質問が来た場合に、コンテキスト
# スキップ規則によってカートリッジ / LTM 検索が完全にスキップされ、
# モデルが事前知識のみで回答してしまう問題を防ぐ。
QUESTION_PATTERNS = re.compile(
    r"(\?|？|ですか|ますか|でしょうか|は何|は誰|はいつ|はどこ|はどう|はなぜ"
    r"|を教え|を説明|を知|を教|の意味|の理由|の特徴|の違い|の使い方|の歴史"
    # 旧 FORCE_PATTERNS から移管: 知識質問の意図を表す動詞 / 願望表現
    r"|教えて|教えてくだ|教えて欲|知りたい|調べて|確認したい|について教)",
)

# QUESTION_PATTERNS の英語版。日本語版は「?」記号への依存が大きいため、
# 「?」を付けない英語依頼文 ("Tell me the difference between A and B") も
# 拾えるよう疑問語・依頼動詞を明示的に含める。
QUESTION_PATTERNS_EN = re.compile(
    r"(\?"
    r"|\bwhat(?:'s|\s+is|\s+are)\b|\bwho(?:'s|\s+is)\b|\bwhen(?:'s|\s+is)\b"
    r"|\bwhere(?:'s|\s+is)\b|\bwhy\s+(?:is|does|do)\b|\bhow\s+(?:is|does|do)\b"
    r"|\btell\s+me\b|\bexplain\b|\bdo\s+you\s+know\b"
    r"|\bi\s+want\s+to\s+know\b|\bcurious\s+about\b|\bcould\s+you\s+check\b)",
    re.IGNORECASE,
)

# 自明な質問パターン (RAG 不要の即 skip 確定)
# 時刻 / 日付 / 曜日 / 自己同一性 / 簡単な雑談 / セッション自己参照を捕捉する。
# `FORCE_PATTERNS` (教えて / 調べて等) より優先順位が低いため、「教えて、
# 今は何時？」のような知識要求を伴うクエリは retrieve に倒れる。
#
# 「この会話で」等のセッション自己参照は、会話履歴が既にコンテキストに
# 含まれておりカートリッジ/LTM 横断検索の対象ではないため skip する。
# パターンの **構造** (アンカー / 否定先読み / 近接窓) は
# backend/free/core/intent_vocab.py から派生させ、
# backend/free/agent/tool_call_judge.py の _SELF_SESSION_REFERENCE_PATTERNS と
# 機械的に同期する (以前は両ファイルへ書き写しており、窓幅を変えるたびに手で
# 両方を直す必要があった)。ただし **語彙は同一にしてはいけない** — 理由は直下参照。
#
# ── 語彙が同一でない理由 (2026-07-27 実測) ─────────────────────
# 2 つの消費側はマッチ時の効果も誤検出コストも異なる:
#   tool_call_judge: search_history を現在セッションへ限定するだけ → 誤検出は軽微
#   self_rag_judge : RAG 検索を丸ごと skip する → 誤検出で外部知識が引けなくなる
# そのため「会話への言及 + 語」で拾える語のうち、**話題ポインタとしても使われる
# 語** は self_rag 側では採らない。この会話で〈話した/聞いた/質問した/指摘された〉X
# について詳しく教えて、のように「会話は X の指し示しに使い、欲しいのは外部知識」
# という形が自然に成立するため。
#   採用: 順序 / 列挙 / 並べ / 訂正 / 直した / 思い出   (誤爆 0)
#   不採用: 順番 / 質問 / 指摘 / 言った / 話した / 聞いた (EXT 質問を誤って skip)
# 実測 (自己参照 8 件 + 外部知識 9 件のプローブ):
#   全語同期    → SELF 捕捉 8/8 だが EXT 誤爆 4/9
#   上記の採用のみ → SELF 捕捉 8/8 / EXT 誤爆 0/9
# 英語側も同じ理由で ``asked`` を採らない (採ると EXT 誤爆 1/4 → 3/4)。
# 実インシデント: 「この会話で一番面白かったやり取りは？」が31文字で
# QUESTION_PATTERNS の長文分岐(>=30char)にマッチし retrieve に倒れ、
# 無関係な過去セッションのチャンクがヒットして混同された。
# ただし「この会話」等の言及だけで無条件 skip にすると、「この会話の続き
# ですが、量子もつれについて詳しく教えて」のように自己参照を前置きにしつつ
# 外部知識を要する質問まで retrieve をスキップしてしまう (レビューで発見)。
# そのため、会話自体を振り返る反省的な語 (面白い/振り返る/まとめ/感想等) との
# 近接共起を要求し、外部知識質問への誤爆を防ぐ。
# 近接窓は「同一文内 (句点・疑問符・感嘆符を跨がない) の 40 文字以内」。
# 旧実装の任意文字 {0,20} は「この会話で一番最初に私が計算させた問題は
# 何だったか覚えてますか？」(間 21 文字) を 1 文字超過で取りこぼし retrieve
# に倒れていた (2026-07-20 ライブ再検証で確認)。一方、窓を任意文字のまま
# {0,50} へ広げるだけでは「この会話とは別に、相対性理論について詳しく
# 教えてください。とても面白いですよね？」のような外部知識質問まで誤って
# マッチする (過去レビューで判明) ため、(a) 文境界を跨がない文字クラスで
# 窓を絞り、(b) 「とは別/とは関係」等の明示的な話題切断の前置きを negative
# lookahead で弾く、の二重ガード付きで窓を 40 に広げる
# (tool_call_judge.py 側と同期)。
# 反省的な語には時系列順序語 (最初/最後/何番目/何回目) も含める
# (2026-07-21: 「この会話で一番最初に計算させた問題は?」が反省語を欠き
# 非マッチ → retrieve に倒れ cross-session チャンク混同のリスク)。順序語で
# マッチ面が広がる分「じゃなく/ではなく」の話題切断も lookahead へ追加。
#: 順序語 (最初/最後) が「会話の発話単位」を指していると判断するための共起語。
#: 『議論された物』(引数・リリース等) を修飾しているだけのケースと区別する。
_JA_SPEECH_UNIT = (
    r"言っ|聞い|話し|尋ね|質問|発言|やり取り|回答|答え|返答"
    r"|計算させ|書かせ|作らせ|指示|依頼|お願い"
)

#: 会話自体を振り返る語 (NARROW)。マッチすると RAG 検索を丸ごと skip するため
#: 誤検出コストが桁違いに高く、tool_call_judge 側の BROAD にある
#: 順番 / 質問 / 指摘 / 言った / 話した / 聞いた は **意図的に採らない**
#: (根拠と実測は下の「語彙が同一でない理由」コメント参照)。
#:
#: 順序語 (最初/最後) は発話単位を指す語との近接共起を要求する。単独で採ると
#: 「この会話で触れた関数の**最初の引数**の意味を詳しく教えて」「この会話で
#: 扱ったフレームワークの**最後のメジャーリリース**はいつ」のように、順序語が
#: 『議論された物』を修飾しているだけの外部知識質問まで RAG を skip する
#: (2026-07-27 実測: 誤爆 2/5)。共起要求で 0/5 になり、自己参照の捕捉は落ちない。
#: 何番目/何回目 は物を修飾しないので単独可。
_SESSION_REFLECTIVE_VOCAB_NARROW_JA = (
    r"面白|印象|振り返|まとめ|要約|感想|どう思|覚えて|何でした|どうでした"
    r"|順序|列挙|並べ|訂正|直した|思い出"
    rf"|(?:最初|最後)[^。．!！?？\n]{{0,20}}?(?:{_JA_SPEECH_UNIT})"
    rf"|(?:{_JA_SPEECH_UNIT})[^。．!！?？\n]{{0,20}}?(?:最初|最後)"
    r"|何番目|何回目"
)

#: 会話の前の方 / 過去のやりとりを指す**後方参照**の語。
#:
#: ルール 6 (``context_count >= 3`` → skip) は「質問マーカーが無く、会話が
#: 続いている = 継続フィラー」という推定で立っている。ところが依頼文の多くは
#: 質問マーカーを持たないため、**会話 3 通目以降は照応を含む依頼まで一律に
#: skip される**。「さっきのライブラリの件、まとめて」「前に貼ったログをもう一度
#: 整理して」はどれもエピソード記憶 (STM / LTM) を引かないと答えられない。
#:
#: 実測 (2026-08-18、chat 135 ターン): retrieval が走らなかったのは 33 ターン
#: (24%)。一方 retrieval の実コストは中央値 7.5ms しかなく、skip して得られる
#: 節約はほぼ無い (支配項は埋め込み往復で、これは skip でも払う)。
#:
#: したがって後方参照があるターンだけルール 6 を降ろす。誤って降ろしても代償は
#: 7.5ms + フロア (較正値) を越えたチャンクだけの注入で、取りこぼしの代償
#: (「さっきの件」に答えられない) より小さい非対称がある。
#:
#: 語彙は「会話の前を指す」ことがほぼ確定するものに絞る。指示語単独
#: (それ / これ) は「それでいいです」のような相槌に当たるので採らない。
_PAST_REFERENCE_RE = re.compile(
    # 時間の後方参照
    r"さっき|さきほど|先ほど|先程|前回|この前|以前|冒頭"
    # 指示語 + 事物名詞 (「その件」「例の話」)
    r"|(?:例|その|あの)の?(?:件|話|やつ|とき|時)"
    # 直前の発話・成果物への言及 (過去形に限る)
    r"|(?:言|話|伝え|教え|挙げ|出し|貼っ|渡し|送っ)(?:った|た|ました|てた|ていた)",
)

#: :data:`_PAST_REFERENCE_RE` の英語版。短い一般語 (before) が別語の内部へ
#: 当たらないよう単語境界を必須にする。
_PAST_REFERENCE_RE_EN = re.compile(
    r"\bearlier\b|\bpreviously\b|\bbefore\b|\blast\s+time\b"
    r"|\b(?:you|i|we)\s+(?:mentioned|said|gave|showed|pasted|sent)\b",
    re.IGNORECASE,
)


#: 「この会話で〜」「ここまでのやり取りで〜」型の**セッション自己参照**。
#:
#: :data:`TRIVIAL_QUESTION_PATTERNS` の一枝として使うが、単独でも参照できる形で
#: 切り出してある。この枝の skip は「答えは今の会話ウィンドウの中にある」という
#: 前提の上に立っており、WorkingMemory が 1 件でも押し出した後はその前提が
#: 成り立たない (:meth:`RetrievalNecessityJudge._judge_rule` の ``window_complete``
#: を参照)。
_SESSION_SELF_REFERENCE_JA_SRC = session_self_reference_pattern_ja(
    _SESSION_REFLECTIVE_VOCAB_NARROW_JA,
)
SESSION_SELF_REFERENCE_PATTERNS = re.compile(_SESSION_SELF_REFERENCE_JA_SRC)


TRIVIAL_QUESTION_PATTERNS = re.compile(
    r"(今.{0,3}(何時|何分|時刻|時間)"
    r"|今日.{0,3}(何月|何日|何曜)"
    r"|何曜日"
    r"|今.{0,2}(日付|曜日)"
    r"|現在.{0,2}(時刻|時間|日時)"
    r"|あなた(は|の)(名前|誰|何者)"
    r"|お前(は|の)(名前|誰)"
    r"|君の名前"
    # NOTE: 旧 ``名前は何`` は削除した。主語を要求しないため
    # 「**私の**名前は何でしたっけ？」「僕の名前は何だった？」という
    # **ユーザー自身についての長期記憶質問**まで skip し、記憶検索が
    # 到達しなくなっていた (2026-08-12 実測: backend.log の necessity
    # skip 率 86.1% のうち、個人想起質問がこの枝で落ちていた)。
    # アシスタントの同一性質問は上の ``あなた(は|の)(名前|誰|何者)`` /
    # ``お前(は|の)(名前|誰)`` / ``君の名前`` が既に捕捉しており、本枝は
    # 冗長だった (実測で 3 件すべて他枝が先に一致)。主語の無い
    # 「名前は何ですか？」は uncertain → retrieve に倒れるが、無駄な検索
    # 1 回のコストは想起失敗より安い。
    r"|あなたは誰"
    r"|あなたは何者"
    r"|元気ですか"
    r"|お元気"
    r"|調子はどう"
    r"|元気\?"
    r"|元気？"
    rf"|{_SESSION_SELF_REFERENCE_JA_SRC})",
)

# TRIVIAL_QUESTION_PATTERNS の英語版。
# (pillar境界のため backend/free/agent/tool_call_judge.py の
# _SELF_SESSION_REFERENCE_PATTERNS_EN と同義の定義を重複させている。
# 両ファイルを変更する際は同期させること)。
# 時刻/自己同一性/雑談の短いフレーズ群 (1 行目の alternation) は ^...$ で
# クエリ全体との一致を要求する。アンカー無しの re.search だと "How are you
# handling auth in this API?" のような RAG が必要な技術的質問が部分文字列
# 一致で誤って skip される (2026-07-22 監査で判明)。セッション自己参照の
# 複合パターン (2/3 行目) は元々 40 文字の近接窓 + 否定先読みで十分に
# 絞り込まれているため対象外。
#: ``_JA_SPEECH_UNIT`` の英語版と、それを使った順序語サブパターン。
#: 活用形の取りこぼしを避けるため動詞は語幹 + ``\w*`` で書く。過去形だけを
#: 並べると "what did we discuss first?" (原形) を落とす (テストで検出)。
_EN_SPEECH_UNIT = (
    r"thing|things|message|messages|question|questions|topic|topics"
    r"|point|points|say|said|says|ask\w*|tell|told|telling"
    r"|mention\w*|discuss\w*|talk\w*|repl(?:y|ies|ied|ying)"
    r"|answer\w*|correct\w*"
)
_EN_ORDINAL_WITH_SPEECH = (
    rf"(?:first|last|earliest|latest)[^.!?\n]{{0,24}}?(?:{_EN_SPEECH_UNIT})"
    rf"|(?:{_EN_SPEECH_UNIT})[^.!?\n]{{0,24}}?(?:first|last|earliest|latest)"
)

#: :data:`SESSION_SELF_REFERENCE_PATTERNS` の英語版 (切り出す理由は JA 側と同じ)。
_SESSION_SELF_REFERENCE_EN_SRC = (
    r"(?:this\s+conversation|this\s+chat|our\s+conversation"
    r"|what\s+we\s+(?:talked|discussed|were\s+talking)\s+about"
    r"|earlier\s+in\s+this\s+(?:conversation|chat)"
    r"|so\s+far\s+in\s+this\s+conversation)"
    r"(?!\s*(?:is|was|has)?\s*(?:not\s+related|unrelated|nothing\s+to\s+do))"
    r"[^.!?\n]{0,40}?"
    r"(?:interesting|memorable|impressive|funn(?:y|iest)"
    r"|summar\w*|recap\w*|think|thought|feel|felt|remember"
    # 2026-07-27 追加。tool_call_judge 側にある asked は **意図的に採らない**
    # (下のコメント「語彙が同一でない理由」参照)。
    r"|order|sequence|enumerate"
    # 順序語は発話単位を指す語との近接共起を要求する (JA 側と同じ理由)。
    # 単独で採ると "what is the **latest version**?" / "the **last stable
    # release**" / "what does the **first argument** do?" のように、順序語が
    # 『議論された物』を修飾しているだけの外部知識質問まで RAG を skip する
    # (2026-07-27 実測: 誤爆 4/6 → 共起要求で 0/6、自己参照の捕捉は不変)。
    rf"|{_EN_ORDINAL_WITH_SPEECH})"
    r"|(?:interesting|memorable|impressive|funn(?:y|iest)"
    r"|summar\w*|recap\w*|order|sequence|enumerate"
    rf"|{_EN_ORDINAL_WITH_SPEECH})"
    r"[^.!?\n]{0,40}?"
    r"(?:this\s+conversation|this\s+chat|our\s+conversation)"
)
SESSION_SELF_REFERENCE_PATTERNS_EN = re.compile(
    _SESSION_SELF_REFERENCE_EN_SRC, re.IGNORECASE,
)

TRIVIAL_QUESTION_PATTERNS_EN = re.compile(
    r"(^\s*(?:what\s+time\s+is\s+it"
    r"|what'?s?\s+(?:today'?s?\s+date|the\s+date)"
    r"|what\s+day\s+is\s+it"
    r"|current\s+time"
    r"|what'?s\s+your\s+name"
    r"|who\s+are\s+you"
    r"|what\s+are\s+you"
    r"|how\s+are\s+you"
    r"|how'?re\s+you\s+doing"
    r"|how'?s\s+it\s+going)\s*[?.!]*\s*$"
    rf"|{_SESSION_SELF_REFERENCE_EN_SRC})",
    re.IGNORECASE,
)

# ファイルパス検出 (Windows ドライブレター / Unix 拡張子付きパス)。
# `tool_call_judge._extract_file_path` をミラーした最小実装。
# pillar 境界 (EvorefGen `rag/` ⇄ EvorefLoop `agent/`) を越境しないよう
# 重複定義を許容する。
FILE_PATH_PATTERN = re.compile(
    r"[A-Za-z]:[\\/][^\s　\"']*\.[A-Za-z0-9]{1,10}"   # Windows
    r"|/[\w./_-]+\.[A-Za-z0-9]{1,10}",                # Unix
)

# コード / ドキュメント生成意図。router の LONG_FORM_PATTERNS と
# 一貫させた語彙 (docs/f_03_agent_engine.md §1.2 参照)。名詞側は
# backend/free/document_nouns.py の共有語彙を参照する (以前は独立ハード
# コードで router.py/content_detector.py と語彙が乖離していた3つ目の重複
# 箇所だった)。
# 保存/書き出し/テンプレート系の語彙も自己完結の生成タスクとして扱う
# (2026-07-15: 「〜を作って <path> に保存して」が skip されず 13 件の
# 無関連チャンク + 7〜10 秒の判定コストが乗った)。
CODE_DOC_GEN_INTENT_PATTERNS = re.compile(
    rf"(?:作成|作って|実装|生成|書いて|書く|出力|保存|書き出|エクスポート|"
    rf"{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX + DOCUMENT_NOUNS_STANDALONE)}|"
    r"create|implement|generate|write|save|export)",
    re.IGNORECASE,
)

# CODE_DOC_GEN_INTENT_PATTERNS の英語版。
CODE_DOC_GEN_INTENT_PATTERNS_EN = re.compile(
    rf"(?:create|write|implement|generate|save|export|build|draft|prepare|"
    rf"{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN + DOCUMENT_NOUNS_STANDALONE_EN)})",
    re.IGNORECASE,
)

# how-to / 教示質問マーカー。ファイルパスを含んでいても「方法を教えて」系の
# 知識質問は生成タスクではないため、write-intent 早期 skip から除外する。
_HOWTO_QUESTION_RE = re.compile(
    r"(?:教えて|おしえて|方法|どうやって|どうすれば|とは|って何|ですか|ますか)",
)

# _HOWTO_QUESTION_RE の英語版。
_HOWTO_QUESTION_RE_EN = re.compile(
    r"\b(?:how\s+(?:do|can|could|would)\s+i|how\s+to|what\s+is|what'?s"
    r"|tell\s+me\s+(?:how|about)|explain|describe)\b",
    re.IGNORECASE,
)

# uncertain 化の最大クエリ長。これより長い質問は QUESTION_PATTERNS
# にマッチした時点で retrieve に倒す (情報量があるため検索の便益が高い前提)。
_UNCERTAIN_QUERY_MAX_CHARS = 30

# デフォルト閾値
DEFAULT_RELEVANCE_THRESHOLD = 0.65
DEFAULT_SUPPORT_THRESHOLD = 0.50
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_HYSTERESIS_BAND = 0.02


@dataclass(frozen=True)
class QualityThresholds:
    """品質判定閾値（config.yaml の rag セクションから設定可能）"""

    relevance: float = DEFAULT_RELEVANCE_THRESHOLD
    support: float = DEFAULT_SUPPORT_THRESHOLD
    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    hysteresis_band: float = DEFAULT_HYSTERESIS_BAND

    @classmethod
    def from_config(
        cls, rag_cfg: dict, calibration: dict[str, float] | None = None,
    ) -> QualityThresholds:
        """config.yaml の rag セクションから閾値を読込み

        3 閾値は cosine スケールの絶対値だが、到達可能なスコア域は埋め込みモデル
        ごとに違う。``rag.self_rag.threshold_mode`` が ``auto`` (既定) で較正値が
        あればそちらを採用し、``manual`` なら config の静的値をそのまま使う。
        ``hysteresis_band`` は帯幅であってスケール依存の絶対閾値ではないため
        較正対象外 (常に config 値)。

        Args:
            rag_cfg: config.yaml の ``rag`` セクション。
            calibration: 較正済み閾値。``None`` の場合はプロセス共通の
                アクティブ較正値を参照する (テストからは明示注入できる)。
        """
        mode = str((rag_cfg.get("self_rag") or {}).get("threshold_mode", "auto"))
        if calibration is None and mode == "auto":
            from backend.free.rag.memory_threshold_calibration import (
                get_active_calibration,
            )
            calibration = get_active_calibration()

        if mode == "auto" and calibration:
            return cls(
                relevance=float(
                    calibration.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD),
                ),
                support=float(
                    calibration.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD),
                ),
                confidence=float(
                    calibration.get(
                        "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD,
                    ),
                ),
                hysteresis_band=rag_cfg.get(
                    "hysteresis_band", DEFAULT_HYSTERESIS_BAND,
                ),
            )
        return cls(
            relevance=rag_cfg.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD),
            support=rag_cfg.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD),
            confidence=rag_cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            hysteresis_band=rag_cfg.get("hysteresis_band", DEFAULT_HYSTERESIS_BAND),
        )


class RetrievalNecessityJudge:
    """検索必要性のルール判定

    `_judge_rule` がルールで `"retrieve"` / `"fetch"` / `"skip"` /
    `"uncertain"` を返す。外向きの後方互換 API `judge` は `uncertain` を
    安全側の `"retrieve"` に正規化する。
    """

    def _judge_rule(
        self, query: str, context_count: int = 0, *, window_complete: bool = True,
    ) -> str:
        """純ルール判定 (3 値 + uncertain).

        Returns: "retrieve" | "fetch" | "skip" | "uncertain"

        判定順 (上から先勝ち):
            0. ``window_complete=False`` かつセッション自己参照 → uncertain
               (下の「窓の完全性」を参照)
            1. クエリ < 3 文字 → skip
            2. URL 含む or 明示的 fetch 動詞 → fetch (確定)
            3. TRIVIAL (時刻/自己同一性/雑談) → skip
            4. QUESTION_PATTERNS マッチ:
                - 長文 (>= 30 char) → retrieve (情報量がある知識質問)
                - 短文 → uncertain (呼出側のリコール送り)
            5. SKIP_PATTERNS (挨拶/相槌) + 短文 → skip
                (QUESTION より後に置くのは「発売日はいつ」など SKIP の "はい"
                substring に誤マッチする知識質問を retrieve に倒すため)
            6. context_count >= 3 かつ ``window_complete`` → skip (会話継続フィラー)。
               ただし会話の前を指す後方参照 (さっき / 前回 / その件 …) が
               あれば uncertain へ降ろす (:data:`_PAST_REFERENCE_RE`)
            7. デフォルト → uncertain (呼出側のリコール送り)

        **窓の完全性 (``window_complete``)**

        ルール 3 のセッション自己参照枝とルール 6 は、どちらも「答えは今の会話
        ウィンドウの中にある」という前提で skip している。この前提が真なのは
        WorkingMemory が 1 件も押し出していない間だけで、押し出しが起きた
        瞬間から恒久的に偽になる — にもかかわらず ``context_count >= 3`` は
        押し出し後も常に真のままなので、**セッションの残り全部で記憶検索が
        skip され続ける**。

        実インシデント (2026-08-23 ライブ監査セット 1 ターン 91):
        「私が来月出張する都市を、確信度を付けて答えてください。」が質問マーカー
        を持たないためルール 6 に落ち、``skip (sufficient context: 37 turns)``。
        その 37 ターンは押し出し後の窓で、出張先を述べたターンは既に窓の外に
        あった。記憶検索が一度も走らないまま「確信度は 100% です。…東京です。」
        と作話した (正解は大阪)。同一セッションでルール 6 の skip は 35/94
        ターン (37%) を占めていた。

        窓が不完全なら前提が崩れているので skip せず uncertain へ降ろす。
        降ろす先の埋め込みリコールは query ベクトルを既に計算済みの経路で、
        追加コストは検索実体の中央値 7.5ms のみ (:data:`_PAST_REFERENCE_RE`
        の実測を参照)。
        """
        query_stripped = query.strip()
        en = is_en_locale()
        skip_pat = SKIP_PATTERNS_EN if en else SKIP_PATTERNS
        fetch_pat = FETCH_INTENT_PATTERNS_EN if en else FETCH_INTENT_PATTERNS
        trivial_pat = TRIVIAL_QUESTION_PATTERNS_EN if en else TRIVIAL_QUESTION_PATTERNS
        codegen_pat = CODE_DOC_GEN_INTENT_PATTERNS_EN if en else CODE_DOC_GEN_INTENT_PATTERNS
        howto_pat = _HOWTO_QUESTION_RE_EN if en else _HOWTO_QUESTION_RE
        question_pat = QUESTION_PATTERNS_EN if en else QUESTION_PATTERNS
        past_ref_pat = _PAST_REFERENCE_RE_EN if en else _PAST_REFERENCE_RE
        session_ref_pat = (
            SESSION_SELF_REFERENCE_PATTERNS_EN if en
            else SESSION_SELF_REFERENCE_PATTERNS
        )

        # 0. 窓が不完全なセッション自己参照は「窓の中で完結する」前提が崩れている
        #    (docstring の「窓の完全性」を参照)。TRIVIAL より先に評価する。
        if not window_complete and session_ref_pat.search(query_stripped):
            logger.debug(
                "Necessity: uncertain (session self-reference with an "
                "incomplete window: %r)", query_stripped[:50],
            )
            return "uncertain"

        # 1. 短すぎるクエリはスキップ
        if len(query_stripped) < 3:
            logger.debug("Necessity: skip (query too short: %d chars)", len(query_stripped))
            return "skip"

        # 2. 外部 fetch 意図 (確定): URL を含む or 明示的 fetch 動詞
        if fetch_pat.search(query_stripped):
            logger.debug(
                "Necessity: fetch (fetch intent pattern matched: %r)",
                query_stripped[:50],
            )
            return "fetch"

        # 3. 自明な質問 (時刻 / 日付 / 自己同一性 / 雑談): 即 skip 確定
        if trivial_pat.search(query_stripped):
            logger.debug(
                "Necessity: skip (trivial question pattern matched: %r)",
                query_stripped[:50],
            )
            return "skip"

        # 3.5. ファイル参照 + コード/ドキュメント生成意図 → skip
        # 例: "C:\path\spec.txt を参照してテトリスを Python で作成"
        #     "議事録テンプレートを作って C:\path\minutes.md に保存して"
        # ユーザが提示したファイルを文脈とする新規生成/書出しタスクで、
        # local KB (SemMem / 履歴) には引き当てるべき情報がない。
        # この時点で RAG 全工程 (embedding + LTM) を早期 skip する。
        # QUESTION より先に評価する: 生成依頼の付帯表現 (「〜が欲しいです」等)
        # が質問マーカーに食われて uncertain に流れるのを防ぐ。
        # ただし how-to 質問 (「作成する方法を教えて」) は除外する。
        if (
            FILE_PATH_PATTERN.search(query_stripped)
            and codegen_pat.search(query_stripped)
            and not howto_pat.search(query_stripped)
        ):
            logger.debug(
                "Necessity: skip (file path + code/doc-gen intent: %r)",
                query_stripped[:60],
            )
            return "skip"

        # 4. 質問マーカー (SKIP より優先)
        # 「発売日はいつ」が SKIP の "はい" substring にヒットして誤って
        # skip されないよう、QUESTION を SKIP より先に評価する。
        if question_pat.search(query_stripped):
            if len(query_stripped) < _UNCERTAIN_QUERY_MAX_CHARS:
                logger.debug(
                    "Necessity: uncertain (short question marker: %r)",
                    query_stripped[:50],
                )
                return "uncertain"
            logger.debug(
                "Necessity: retrieve (long question marker matched: %r)",
                query_stripped[:50],
            )
            return "retrieve"

        # 5. スキップパターン (挨拶/相槌) — 短文のみ
        if skip_pat.search(query_stripped) and len(query_stripped) < 20:
            logger.debug("Necessity: skip (greeting/simple pattern matched: %r)", query_stripped[:30])
            return "skip"

        # 6. コンテキストが十分ある場合はスキップ
        # (質問マーカーなし + 多ターン会話の場合は会話継続フィラーとみなす)
        # ただし会話の前を指す後方参照があるターンは降ろす
        # (:data:`_PAST_REFERENCE_RE` の説明を参照)。
        if context_count >= 3 and window_complete:
            if past_ref_pat.search(query_stripped):
                logger.debug(
                    "Necessity: uncertain (back-reference to an earlier turn "
                    "overrides the sufficient-context skip: %r)",
                    query_stripped[:50],
                )
                return "uncertain"
            logger.debug("Necessity: skip (sufficient context: %d turns)", context_count)
            return "skip"

        # 7. デフォルトは uncertain (呼出側のリコール送り)
        # 旧 FORCE_PATTERNS が拾っていたケースもここに落ちる。
        logger.debug("Necessity: uncertain (default, query=%r)", query_stripped[:50])
        return "uncertain"

    def judge(
        self, query: str, context_count: int = 0, *, window_complete: bool = True,
    ) -> str:
        """ルール判定の後方互換ラッパ (2 値返却).

        `_judge_rule` の 3 値 + uncertain を旧 API の 2 値
        (``"retrieve"`` / ``"skip"``) に正規化する:

        - ``"fetch"`` → ``"skip"`` (RAG 不要の意味では同義)
        - ``"uncertain"`` → ``"retrieve"`` (安全側、検索する)

        3 値のまま扱いたい呼出側は ``judge_rule_only`` を使うこと。
        """
        rule = self._judge_rule(
            query, context_count, window_complete=window_complete,
        )
        if rule == "uncertain":
            return "retrieve"
        if rule == "fetch":
            return "skip"
        return rule

    def judge_rule_only(
        self, query: str, context_count: int = 0, *, window_complete: bool = True,
    ) -> str:
        """ルール判定の 3 値 (``uncertain`` 含む) をそのまま返す。

        embedding 決定論的リコール (``rag_judge_recall``) が「ルールで確定
        できるか」を判定するための公開 API。``uncertain`` の場合のみ呼出側が
        リコールへフォールバックする。
        """
        return self._judge_rule(
            query, context_count, window_complete=window_complete,
        )


class RetrievalQualityJudge:
    """検索結果品質のベクトル閾値判定"""

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        debug_logger: "DebugLogger | None" = None,
    ):
        """
        Args:
            thresholds: 品質判定の閾値設定。
            debug_logger: marginal 判定の選択 (decision_point=
                ``self_rag_judge_path``) を ``decision.jsonl`` に記録する。
                ``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        self.thresholds = thresholds or QualityThresholds()
        self._debug_logger = debug_logger

    def judge(
        self,
        results: list[tuple[str, float, str]],
    ) -> str:
        """
        検索結果の品質を判定する。

        Args:
            results: [(chunk_id, score, text), ...]

        Returns: "high" | "medium" | "low"
        """
        if not results:
            logger.debug("Quality: low (no results)")
            return self._record("low", "no_results", results, 0.0, 0.0)

        th = self.thresholds
        top_score = results[0][1]
        scores = [s for _, s, _ in results]
        top_3_avg = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores)

        logger.debug(
            "Quality judge: %d results, top_score=%.3f, top3_avg=%.3f, "
            "thresholds=(confidence=%.2f, relevance=%.2f, support=%.2f)",
            len(results), top_score, float(top_3_avg),
            th.confidence, th.relevance, th.support,
        )

        # ヒステリシス帯: confidence ± hysteresis_band は medium
        # 境界付近での判定のぶれを防止する
        high_boundary = th.confidence + th.hysteresis_band
        low_boundary = th.confidence - th.hysteresis_band

        # 高信頼: トップスコアがヒステリシス上限以上
        if top_score >= high_boundary:
            logger.debug("Quality: high (top_score %.3f >= %.2f)", top_score, high_boundary)
            return self._record("high", "top_score_above_confidence_band",
                                results, top_score, top_3_avg)

        # ヒステリシス帯: 境界付近は安定して medium を返す
        if top_score >= low_boundary:
            logger.debug(
                "Quality: medium (hysteresis band: %.3f in [%.2f, %.2f))",
                top_score, low_boundary, high_boundary,
            )
            return self._record("medium", "hysteresis_band",
                                results, top_score, top_3_avg)

        # 中信頼: トップスコアが関連性閾値以上 かつ 上位3件の平均が支持閾値以上
        if top_score >= th.relevance and top_3_avg >= th.support:
            logger.debug("Quality: medium (top=%.3f, avg=%.3f)", top_score, float(top_3_avg))
            return self._record("medium", "relevance_and_support_met",
                                results, top_score, top_3_avg)

        logger.debug("Quality: low (top_score=%.3f below thresholds)", top_score)
        return self._record("low", "top_score_below_thresholds",
                            results, top_score, top_3_avg)

    def _record(
        self,
        quality: str,
        reason: str,
        results: list[tuple[str, float, str]],
        top_score: float,
        top_3_avg: float,
    ) -> str:
        """判定を ``decision.jsonl`` に残して ``quality`` をそのまま返す。

        ``__init__`` は 2026-07 から ``debug_logger`` を受け取り docstring も
        「``self_rag_judge_path`` を記録する」と書いていたが、``judge()`` は
        一度もそれを呼んでいなかった (保持するだけ)。実測 2026-08-18 の
        ``decision.jsonl`` 313 行に ``self_rag_judge_path`` は **0 行**で、
        「品質ラベルがどう決まったか」を後から追えない状態が続いていた。

        品質ラベルは relevance floor の適用結果 (``log_rag_selection``) と
        並べて初めて意味を持つので、同じ ``trace_id`` で join できるここに出す。
        """
        dl = self._debug_logger
        if dl is None:
            return quality
        th = self.thresholds
        dl.log_decision(
            decision_point="self_rag_judge_path",
            chosen=quality,
            candidates=["high", "medium", "low"],
            reason=reason,
            context={
                "n_results": len(results),
                "top_score": round(float(top_score), 4),
                "top3_avg": round(float(top_3_avg), 4),
                "confidence": th.confidence,
                "relevance": th.relevance,
                "support": th.support,
                "hysteresis_band": th.hysteresis_band,
            },
            scope="request",
        )
        return quality
