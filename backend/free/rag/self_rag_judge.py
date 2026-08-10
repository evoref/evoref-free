"""Self-RAG 品質判定（ルールベース、ベースモデル呼び出し禁止）

ルールベース判定が既定 (``RetrievalQualityJudge.judge``)。
``config.yaml`` の ``rag.self_rag.assist_judge`` (default enabled) で
制御され、セッション / クエリ単位の発火上限は ``AssistJudgeUsageTracker``
が担う。呼び出しの組立ては ``backend.free.memory.pipeline.search_pipeline``
``_maybe_assist_judge_quality`` に集約される。

検索必要性 (``RetrievalNecessityJudge``) はハイブリッド 3 値構成:
ルールで ``retrieve`` / ``fetch`` / ``skip`` が確定するケースはアシスト 0
呼び出しで即返し、判別不能な ``uncertain`` ケースのみ ``judge_with_assist``
がアシストモデルに 3 値 JSON
(``{"action": "retrieve" | "fetch" | "skip"}``) を問う。``fetch`` は外部
fetch_url ツールに委ねる意図のため、search_pipeline では ``skip`` 同等に
RAG をスキップする。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.free.core.intent_vocab import session_self_reference_pattern_ja
from backend.free.core.locale_patterns import is_en_locale
from backend.free.llm.assist_client import assist_ready
from backend.free.document_nouns import (
    DOCUMENT_NOUNS_NEEDS_SUFFIX,
    DOCUMENT_NOUNS_NEEDS_SUFFIX_EN,
    DOCUMENT_NOUNS_STANDALONE,
    DOCUMENT_NOUNS_STANDALONE_EN,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.debug_logger import DebugLogger
    from backend.free.rag.assist_judge_tracker import AssistJudgeUsageTracker

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
# リアルタイムキーワード (ニュース / 株価 / 天気 等) はアシスト判定に委譲し、
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

TRIVIAL_QUESTION_PATTERNS = re.compile(
    r"(今.{0,3}(何時|何分|時刻|時間)"
    r"|今日.{0,3}(何月|何日|何曜)"
    r"|何曜日"
    r"|今.{0,2}(日付|曜日)"
    r"|現在.{0,2}(時刻|時間|日時)"
    r"|あなた(は|の)(名前|誰|何者)"
    r"|お前(は|の)(名前|誰)"
    r"|君の名前"
    r"|名前は何"
    r"|あなたは誰"
    r"|あなたは何者"
    r"|元気ですか"
    r"|お元気"
    r"|調子はどう"
    r"|元気\?"
    r"|元気？"
    rf"|{session_self_reference_pattern_ja(_SESSION_REFLECTIVE_VOCAB_NARROW_JA)})",
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
    r"|(?:this\s+conversation|this\s+chat|our\s+conversation"
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
    r"(?:this\s+conversation|this\s+chat|our\s+conversation))",
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

# アシストモデルへの検索必要性判定プロンプトの指示部 (日本語、3 値 action 応答)。
# AssistPromptManager (task=rag_necessity) 未注入時のフォールバック既定値。
# 動的データ (直前文脈 / クエリ) は judge_with_assist が末尾に連結するため、
# ここにフォーマットスロットは含めない (str.format は使わない)。
_NECESSITY_INSTRUCTIONS = (
    "ユーザーの最新クエリを、3つの検索アクションのいずれかに分類してください。\n"
    "\n"
    "- retrieve: ローカルの知識ベース（アップロード文書・過去の会話・"
    "導入済みカートリッジ）から答えるのが最適なもの。使い方・定義・"
    "既知の内容への意見・以前の話題への言及など。\n"
    "- fetch: 静的な知識ベースでは提供できない最新／ライブの外部情報を"
    "要するもの。最新ニュース、現在の株価・天気・スポーツのスコア、"
    "本日の見出し、特定サイトのリアルタイムな状態など。システムは"
    "ローカル検索ではなく Web 取得ツールを使う。\n"
    "- skip: 検索も取得も不要な些末なもの。現在時刻・日付・曜日、"
    "簡単な挨拶、自己同一性、雑談のフィラーなど。\n"
    "\n"
    "直前のローカルな話題を指す短いフォローアップ質問は retrieve を"
    "優先する。外部の最新状態を尋ねるものは fetch を優先する。\n"
    "\n"
    'JSON形式で回答: {"action": "retrieve"} / {"action": "fetch"} / {"action": "skip"}'
)

# アシストプロンプトへ含める直前ターン数の既定値 (user/assistant 合計)
# 大きくしすぎるとレイテンシ増 + 関係ないトピックを引きずるため、
# 直前 1 ターン (user + assistant) = 2 メッセージを既定とする。
_DEFAULT_CONTEXT_TURNS = 2

# 1 ターンあたりに含める content の最大文字数。
# 長文 assistant 応答が context を埋めないようトリムする。
_CONTEXT_TURN_MAX_CHARS = 200


def _format_context_for_assist(
    recent_context: list[dict] | None,
    *,
    max_turns: int = _DEFAULT_CONTEXT_TURNS,
    max_chars: int = _CONTEXT_TURN_MAX_CHARS,
) -> str:
    """`judge_with_assist` のプロンプトに埋め込む直前ターン文字列を作る。

    `recent_context` から末尾 `max_turns` 件を取り出し、各ターンの
    content を `max_chars` 文字で切り詰めて role 付き 1 行にする。
    空 / None / `max_turns <= 0` のときは空文字列を返す。
    """
    if not recent_context or max_turns <= 0:
        return ""
    tail = recent_context[-max_turns:]
    if not tail:
        return ""
    lines = []
    for turn in tail:
        role = str(turn.get("role", "")).strip() or "user"
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[: max_chars - 3] + "..."
        if not content:
            continue
        lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "直近の会話:\n" + "\n".join(lines) + "\n"

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
    def from_config(cls, rag_cfg: dict) -> QualityThresholds:
        """config.yaml の rag セクションから閾値を読込み"""
        return cls(
            relevance=rag_cfg.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD),
            support=rag_cfg.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD),
            confidence=rag_cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            hysteresis_band=rag_cfg.get("hysteresis_band", DEFAULT_HYSTERESIS_BAND),
        )


class RetrievalNecessityJudge:
    """検索必要性のハイブリッド判定 (ルール → 任意でアシスト併用)

    `_judge_rule` がルールで `"retrieve"` / `"skip"` / `"uncertain"` を
    返し、外向きの後方互換 API `judge` は `uncertain` を `"retrieve"`
    に正規化する。`judge_with_assist` は `uncertain` のみアシスト
    モデルへ問い合わせ、失敗時は安全側の `"retrieve"` にフォールバック
    する (現状挙動と一致させ回帰防止)。
    """

    def __init__(self, necessity_instructions: str | None = None) -> None:
        """
        Args:
            necessity_instructions: アシスト必要性判定プロンプトの指示部。
                AssistPromptManager (task=rag_necessity) 由来の編集可能テキストを
                composition 層 (api/chat) から注入する。``None`` の場合は
                ``_NECESSITY_INSTRUCTIONS`` 既定値にフォールバックする
                (degraded-safe)。
        """
        self._necessity_instructions = (
            necessity_instructions or _NECESSITY_INSTRUCTIONS
        )

    def _judge_rule(self, query: str, context_count: int = 0) -> str:
        """純ルール判定 (3 値 + uncertain).

        Returns: "retrieve" | "fetch" | "skip" | "uncertain"

        判定順 (上から先勝ち):
            1. クエリ < 3 文字 → skip
            2. URL 含む or 明示的 fetch 動詞 → fetch (確定)
            3. TRIVIAL (時刻/自己同一性/雑談) → skip
            4. QUESTION_PATTERNS マッチ:
                - 長文 (>= 30 char) → retrieve (情報量がある知識質問)
                - 短文 → uncertain (アシスト判定送り)
            5. SKIP_PATTERNS (挨拶/相槌) + 短文 → skip
                (QUESTION より後に置くのは「発売日はいつ」など SKIP の "はい"
                substring に誤マッチする知識質問を retrieve に倒すため)
            6. context_count >= 3 → skip (会話継続フィラー)
            7. デフォルト → uncertain (アシスト判定送り)

        旧 ``FORCE_PATTERNS`` (教えて/情報/方法/...) は廃止。固定キーワード
        では「Yahoo の最新ニュース教えて」(fetch 意図) と「RAG の方法を教えて」
        (retrieve 意図) を区別できないため、アシスト 3 値判定に委ねる。
        """
        query_stripped = query.strip()
        en = is_en_locale()
        skip_pat = SKIP_PATTERNS_EN if en else SKIP_PATTERNS
        fetch_pat = FETCH_INTENT_PATTERNS_EN if en else FETCH_INTENT_PATTERNS
        trivial_pat = TRIVIAL_QUESTION_PATTERNS_EN if en else TRIVIAL_QUESTION_PATTERNS
        codegen_pat = CODE_DOC_GEN_INTENT_PATTERNS_EN if en else CODE_DOC_GEN_INTENT_PATTERNS
        howto_pat = _HOWTO_QUESTION_RE_EN if en else _HOWTO_QUESTION_RE
        question_pat = QUESTION_PATTERNS_EN if en else QUESTION_PATTERNS

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
        # この時点で RAG 全工程 (assist 判定 + embedding + LTM) を
        # 早期 skip して 10 秒以上のレイテンシを排除する。
        # QUESTION より先に評価する: 生成依頼の付帯表現 (「〜が欲しいです」等)
        # が質問マーカーに食われて uncertain → assist 判定 (5 秒) に流れるのを
        # 防ぐ。ただし how-to 質問 (「作成する方法を教えて」) は除外する。
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
        if context_count >= 3:
            logger.debug("Necessity: skip (sufficient context: %d turns)", context_count)
            return "skip"

        # 7. デフォルトは uncertain (アシスト判定送り)
        # 旧 FORCE_PATTERNS が拾っていたケースもここに落ちる。
        logger.debug("Necessity: uncertain (default, query=%r)", query_stripped[:50])
        return "uncertain"

    def judge(self, query: str, context_count: int = 0) -> str:
        """ルール判定の後方互換ラッパ (2 値返却).

        `_judge_rule` の 3 値 + uncertain を旧 API の 2 値
        (``"retrieve"`` / ``"skip"``) に正規化する:

        - ``"fetch"`` → ``"skip"`` (RAG 不要の意味では同義)
        - ``"uncertain"`` → ``"retrieve"`` (安全側、検索する)

        アシスト併用 + 3 値を使いたい呼出側は ``judge_with_assist`` を
        await すること。
        """
        rule = self._judge_rule(query, context_count)
        if rule == "uncertain":
            return "retrieve"
        if rule == "fetch":
            return "skip"
        return rule

    def judge_rule_only(self, query: str, context_count: int = 0) -> str:
        """ルール判定の 3 値 (``uncertain`` 含む) をそのまま返す。

        embedding 決定論的リコール (``rag_judge_recall``) が assist 呼出前に
        「ルールで確定できるか」を判定するための公開 API。``uncertain`` の
        場合のみ呼出側がリコール → assist の順にフォールバックする。
        """
        return self._judge_rule(query, context_count)

    async def judge_with_assist(
        self,
        query: str,
        context_count: int,
        assist_client,
        *,
        recent_context: list[dict] | None = None,
        session_id: str = "default",
        tracker: "AssistJudgeUsageTracker | None" = None,
        debug_logger: "DebugLogger | None" = None,
        config: dict | None = None,
        record_assist: "Callable[[str, str, str, float], None] | None" = None,
    ) -> str:
        """ルール判定 + uncertain 時のアシスト救済 (3 値返却).

        ルールで ``retrieve`` / ``fetch`` / ``skip`` が確定すればアシストを
        呼ばない。``uncertain`` のみアシストモデルに 3 値 JSON
        (``{"action": "retrieve" | "fetch" | "skip"}``) を問う。
        tracker / timeout / 例外でフォールバックする場合は安全側の
        ``"retrieve"`` を返す。

        Args:
            recent_context: 会話履歴 (role/content dict のリスト)。末尾の
                ``config["context_turns"]`` 件 (既定 2) をアシストプロンプトに
                埋め込み、フォローアップ質問の判定精度を上げる。
            session_id: 発火回数カウンタのキー (``WorkingMemory.session_id``)。
            tracker: ``AssistJudgeUsageTracker`` 互換のセッションカウンタ。
                ``None`` ならカウンタ評価をスキップする (テスト経路互換)。
            config: ``rag.self_rag.assist_necessity`` セクション。

        Returns: "retrieve" | "fetch" | "skip"
        """
        rule = self._judge_rule(query, context_count)
        if rule != "uncertain":
            return rule

        cfg = config or {}

        # assist が無い / 非常駐 (residency=on_demand のチャット中) なら
        # degraded mode → 安全側 retrieve
        if not assist_ready(assist_client, "retrieval_necessity_judge"):
            logger.debug("Necessity assist: skipped (assist not available)")
            return "retrieve"

        # tracker による発火上限チェック (本機能の quality キーは "uncertain")
        if tracker is not None:
            decision = tracker.check(
                session_id=session_id,
                namespace="necessity",
                quality="uncertain",
                query_count=0,
                config=cfg,
            )
            if not decision.allowed:
                logger.debug(
                    "Necessity assist: skipped (reason=%s, session=%d)",
                    decision.reason, decision.session_count,
                )
                if debug_logger is not None:
                    debug_logger.log_decision(
                        decision_point="self_rag_necessity_path",
                        chosen="retrieve",
                        candidates=["retrieve", "fetch", "skip"],
                        reason=f"tracker_skipped:{decision.reason}",
                        context={
                            "session_count": decision.session_count,
                            "query_count": decision.query_count,
                        },
                        scope="request",
                    )
                return "retrieve"

        # cfg["timeout_s"] (rag.self_rag.assist_necessity.timeout_s) は使わない:
        # schema既定値(5.0)が常に埋まるため「ユーザー明示設定」と「未設定」を
        # 区別できず、反応的タイムアウト較正 (_calibrated_timeouts) を永久に
        # 無効化してしまう。purpose別上書きは assist_model.timeouts.<purpose>
        # (全 purpose 共通の正規チャネル) に委ね、ここでは較正込みの実効値を使う。
        timeout_s = assist_client.resolve_effective_timeout("retrieval_necessity_judge")
        context_turns = int(cfg.get("context_turns", _DEFAULT_CONTEXT_TURNS))
        context_block = _format_context_for_assist(
            recent_context, max_turns=context_turns,
        )
        prompt = (
            f"{self._necessity_instructions}\n{context_block}最新のクエリ: {query}"
        )
        if context_block:
            logger.debug(
                "Necessity assist: context_block included (chars=%d, turns=%d)",
                len(context_block), min(context_turns, len(recent_context or [])),
            )
        else:
            logger.debug("Necessity assist: no context_block (empty/disabled)")
        try:
            # timeout は generate_json 側の purpose 別 (realtime) 総予算強制に
            # 一本化する。外側に別途 wait_for を掛けると、外側の締切が内側の
            # asyncio.timeout より先に (または同時に) 発火した場合、内側が
            # 自分の締切超過と認識できず CancelledError のまま抜けてしまい、
            # generate() の except (TimeoutError) に到達せず反応的タイムアウト
            # 較正 (_bump_calibrated_timeout) が発火しない不具合があった。
            result = await assist_client.generate_json(
                prompt,
                max_tokens=32,
                temperature=0.0,
                purpose="retrieval_necessity_judge",
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            # fail-closed: タイムアウト時は skip に倒す。fail-open (retrieve)
            # だとタイムアウト税 (timeout_s) + 無関連チャンク添付の二重コストに
            # なる (2026-07-15: timeout 2/3 が retrieve に倒れて 13 件の過去
            # 雑談ノートがコンテキストに注入された)。ここに来るのは rule が
            # uncertain と判定した短い質問のみで、検索スキップの損失は小さい。
            logger.warning(
                "Necessity assist: timeout after %.1fs (fail-closed to skip)",
                timeout_s,
            )
            if debug_logger is not None:
                debug_logger.log_decision(
                    decision_point="self_rag_necessity_path",
                    chosen="skip",
                    candidates=["retrieve", "fetch", "skip"],
                    reason="assist_timeout",
                    scope="request",
                )
            return "skip"
        except Exception as e:
            logger.warning("Necessity assist: failed (%s)", type(e).__name__)
            if debug_logger is not None:
                debug_logger.log_decision(
                    decision_point="self_rag_necessity_path",
                    chosen="retrieve",
                    candidates=["retrieve", "fetch", "skip"],
                    reason=f"assist_call_failed:{type(e).__name__}",
                    scope="request",
                )
            return "retrieve"

        # 新 3 値 schema: {"action": "retrieve" | "fetch" | "skip"}
        # 旧 2 値 schema: {"need_rag": bool}  ← 後方互換のため両方を受理
        action = result.get("action")
        if not isinstance(action, str) or action not in {"retrieve", "fetch", "skip"}:
            need_rag = result.get("need_rag")
            if isinstance(need_rag, bool):
                action = "retrieve" if need_rag else "skip"
            else:
                logger.warning(
                    "Necessity assist: invalid response %r, falling back to retrieve",
                    result,
                )
                if debug_logger is not None:
                    debug_logger.log_decision(
                        decision_point="self_rag_necessity_path",
                        chosen="retrieve",
                        candidates=["retrieve", "fetch", "skip"],
                        reason="invalid_assist_response",
                        scope="request",
                    )
                return "retrieve"

        if tracker is not None:
            tracker.record(session_id, namespace="necessity")
        logger.info(
            "Necessity assist: %s (query=%r)", action, query[:50],
        )
        if debug_logger is not None:
            debug_logger.log_decision(
                decision_point="self_rag_necessity_path",
                chosen=action,
                candidates=["retrieve", "fetch", "skip"],
                reason="assist_judge_used",
                context={"action": action},
                scope="request",
            )
        # assist が有効判定を返したケースのみ assist 経験に記録 (outcome=1.0)。
        # tracker_skipped / timeout / 例外 / 不正応答の fallback 経路では呼ばない。
        if record_assist is not None:
            record_assist("rag_necessity", query, action, 1.0)
        return action


# 検索結果品質判定プロンプトの指示部 (日本語、3 値 quality 応答)。
# AssistPromptManager (task=rag_quality) 未注入時のフォールバック既定値。
# クエリ / 検索結果は judge_with_assist が末尾に連結する。
_QUALITY_INSTRUCTIONS = (
    "以下のクエリに対する検索結果の関連性を判定してください。\n\n"
    "判定基準:\n"
    "- high: 検索結果がクエリに直接的に関連し、十分な情報を含む\n"
    "- medium: 部分的に関連するが、情報が不十分\n"
    "- low: 検索結果がクエリにほぼ関連しない\n\n"
    'JSON形式で回答: {"quality": "high" or "medium" or "low"}'
)


class RetrievalQualityJudge:
    """検索結果品質のベクトル閾値判定"""

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        debug_logger: "DebugLogger | None" = None,
        quality_instructions: str | None = None,
    ):
        """
        Args:
            thresholds: 品質判定の閾値設定。
                marginal 判定時の rule-based vs assist 救済の選択 (decision_point=
                ``self_rag_judge_path``) を ``decision.jsonl`` に記録する。
                ``evolve`` レベル限定で実発火、それ以外は no-op。
            quality_instructions: アシスト品質判定プロンプトの指示部。
                AssistPromptManager (task=rag_quality) 由来の編集可能テキストを
                composition 層 (api/chat) から注入する。``None`` の場合は
                ``_QUALITY_INSTRUCTIONS`` 既定値にフォールバックする
                (degraded-safe)。
        """
        self.thresholds = thresholds or QualityThresholds()
        self._debug_logger = debug_logger
        self._quality_instructions = (
            quality_instructions or _QUALITY_INSTRUCTIONS
        )

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
            return "low"

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
            return "high"

        # ヒステリシス帯: 境界付近は安定して medium を返す
        if top_score >= low_boundary:
            logger.debug(
                "Quality: medium (hysteresis band: %.3f in [%.2f, %.2f))",
                top_score, low_boundary, high_boundary,
            )
            return "medium"

        # 中信頼: トップスコアが関連性閾値以上 かつ 上位3件の平均が支持閾値以上
        if top_score >= th.relevance and top_3_avg >= th.support:
            logger.debug("Quality: medium (top=%.3f, avg=%.3f)", top_score, float(top_3_avg))
            return "medium"

        logger.debug("Quality: low (top_score=%.3f below thresholds)", top_score)
        return "low"

    async def judge_with_assist(
        self,
        query: str,
        results: list[tuple[str, float, str]],
        assist_client,
        rule_based_quality: str,
        record_assist: "Callable[[str, str, str, float], None] | None" = None,
    ) -> str:
        """アシストモデルで閾値境界の品質を再判定する。

        ルールベース判定が "medium"（閾値境界）の場合に呼び出し、
        アシストモデル LLM で関連性をより正確に判定する。
        エラー時はルールベース結果にフォールバックする。

        Args:
            query: ユーザークエリ
            results: [(chunk_id, score, text), ...]
            assist_client: AssistModelClient インスタンス
            rule_based_quality: ルールベース判定の結果（フォールバック用）

        Returns: "high" | "medium" | "low"
        """
        try:
            top_results = results[:3]
            formatted = "\n".join(
                f"- (スコア: {score:.2f}) {text[:150]}"
                for _, score, text in top_results
            )
            prompt = (
                f"{self._quality_instructions}\n\n"
                f"クエリ: {query}\n\n"
                f"検索結果:\n{formatted}"
            )

            result = await assist_client.generate_json(
                prompt, max_tokens=64, temperature=0.1,
                purpose="retrieval_quality_judge",
            )
            quality = result.get("quality", "")

            if quality not in VALID_QUALITIES:
                logger.warning(
                    "Assist judge returned invalid quality %r, "
                    "falling back to rule-based: %s",
                    quality, rule_based_quality,
                )
                if self._debug_logger is not None:
                    self._debug_logger.log_decision(
                        decision_point="self_rag_judge_path",
                        chosen="rule_based",
                        candidates=["rule_based", "assist_judge"],
                        reason="invalid_assist_response",
                        context={
                            "rule_based_quality": rule_based_quality,
                            "assist_quality": quality,
                        },
                        scope="request",
                    )
                return rule_based_quality

            logger.info(
                "Assist judge: rule_based=%s -> assist=%s",
                rule_based_quality, quality,
            )
            if self._debug_logger is not None:
                self._debug_logger.log_decision(
                    decision_point="self_rag_judge_path",
                    chosen="assist_judge",
                    candidates=["rule_based", "assist_judge"],
                    reason="marginal_quality_assist_used",
                    context={
                        "rule_based_quality": rule_based_quality,
                        "assist_quality": quality,
                    },
                    scope="request",
                )
            # assist が有効 quality を返したケースのみ記録。
            # outcome は high/medium/low を 1.0/0.5/0.0 にマップ。fallback は記録しない。
            if record_assist is not None:
                _q_outcome = {"high": 1.0, "medium": 0.5, "low": 0.0}.get(quality, 0.5)
                record_assist("rag_quality", query, quality, _q_outcome)
            return quality

        except Exception as e:
            logger.warning(
                "Assist judge failed (%s), falling back to rule-based: %s",
                e, rule_based_quality,
            )
            if self._debug_logger is not None:
                self._debug_logger.log_decision(
                    decision_point="self_rag_judge_path",
                    chosen="rule_based",
                    candidates=["rule_based", "assist_judge"],
                    reason=f"assist_call_failed:{type(e).__name__}",
                    context={"rule_based_quality": rule_based_quality},
                    scope="request",
                )
            return rule_based_quality
