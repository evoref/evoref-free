"""複雑度分類ルーター: クエリを3層エージェントに振り分ける"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.free.agent.context_budget import resolve_meta_cognitive_loop_budget
from backend.free.agent.safety_patterns import strip_command_literals
from backend.free.core.intent_vocab import (
    GREETING_PUNCTUATION_JA,
    HISTORY_KEYWORDS as _HISTORY_KEYWORDS,
    HISTORY_KEYWORDS_EN as _HISTORY_KEYWORDS_EN,
    REFERENTIAL_WRITE_TARGET_RE,
    ascii_boundary_alternation,
    exact_greeting_pattern,
)
from backend.free.core.session_mode import is_chat_mode, is_coding_mode
from backend.free.document_nouns import (
    DOCUMENT_NOUN_LEARNABLE_JA,
    DOCUMENT_NOUNS_NEEDS_SUFFIX,
    DOCUMENT_NOUNS_NEEDS_SUFFIX_EN,
    DOCUMENT_NOUNS_STANDALONE,
    DOCUMENT_NOUNS_STANDALONE_EN,
)
from backend.log_config import get_logger
from backend.policy_helpers import get_policy_value

if TYPE_CHECKING:
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("agent.router")

# 長文生成リクエスト検出パターン（設計書 f_03 §1.2）
# 名詞群・動詞群の語彙拡張時は f_03 §1.2 の表も同時更新すること。
# ハードコード regex に加え、`LearnedPatternStore.match(category="long_form")`
# の学習済みキーワード (FeedbackCollector で蓄積、LearningScheduler で進化)
# も判定対象 (`_detect_long_form_learned()` で OR 統合)。
LONG_FORM_PATTERNS = [
    re.compile(r"(\d{3,})\s*[字文行]"),                        # 「3000字の記事」「500行」
    # 文書系名詞 → 動作動詞 の組合せ。
    # 「仕様書を作成して」「ドキュメントを出力して」「計画書をまとめて」等
    # を長文生成として拾う。名詞語彙は backend/free/document_nouns.py で
    # content_detector.py (EvorefGen) の TEXT_PATTERNS と共有する。router は
    # 常にサフィックス必須で運用するため、名詞単体マッチ許容語彙
    # (DOCUMENT_NOUNS_STANDALONE) もここではサフィックス必須の安全側で使う。
    re.compile(
        rf"({'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX + DOCUMENT_NOUNS_STANDALONE)})"
        r".*(書|作成|生成|まとめ|出力)",
    ),
    # 「長文」「長編」の言及、または「長い」+ 創作文書系名詞 (物語/小説等、
    # pattern[1] の文書名詞リストには含まれない) の言及が、生成依頼のて形
    # 動詞と共起する場合に長文生成として拾う。
    # 「一番長い川」のような一般形容詞や、「長文が重複生成されている」といった
    # (受身形の) バグ報告文自体の誤爆を防ぐため、pattern[1] と異なりて形の
    # 完全一致を要求する (「生成」の bare stem だと「生成され」にも誤って一致する)。
    re.compile(
        r"(?:長(?:文|編)|長い(?:物語|小説|エッセイ|詩|お話))"
        r".*(?:書いて|作成して|生成して|まとめて|出力して)",
    ),
    re.compile(
        r"(ファイル|モジュール|クラス|プロジェクト)"
        r".*(一式|全体|まるごと|フル).*(作成|生成|実装)",
    ),
    re.compile(r"(実装|作成).*全体"),
    re.compile(r"(完全|網羅的|包括的).*(実装|ガイド|解説)"),
]

# LONG_FORM_PATTERNS の英語版。GUI locale に関わらず LONG_FORM_PATTERNS と
# 常に両方評価する (2026-07-22 発見: GUI locale が既定 'ja' のまま英語で
# チャットすると、以前は locale 排他選択のため英語の文書作成依頼が一切
# 検出されなかった。詳細は _detect_long_form 参照)。
LONG_FORM_PATTERNS_EN = [
    re.compile(r"\b(\d{3,})[\s-]?(?:words?|lines?)\b", re.IGNORECASE),  # "3000-word article" "500 lines"
    re.compile(
        rf"\b(?:write|draft|create|compose|generate|produce|prepare|put\s+together)\b"
        rf".*\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN + DOCUMENT_NOUNS_STANDALONE_EN)})\b"
        r"|"
        rf"\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN + DOCUMENT_NOUNS_STANDALONE_EN)})\b"
        rf".*\b(?:write|draft|create|compose|generate|produce|prepare|put\s+together)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:write|draft|create|compose)\b.*\b(?:long|lengthy|in-depth|comprehensive)\b"
        r".*\b(?:article|essay|story|novel|report|poem)\b"
        r"|"
        r"\b(?:long|lengthy|in-depth|comprehensive)\b.*\b(?:article|essay|story|novel|report|poem)\b"
        r".*\b(?:write|draft|create|compose)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:files?|modules?|classes?|the\s+(?:whole|entire)\s+project)\b"
        r".*\b(?:entire|whole|full|complete)\b.*\b(?:create|generate|implement|build)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:implement|create|build)\b.*\b(?:everything|the\s+whole\s+thing|the\s+entire\s+project)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:complete|comprehensive|exhaustive|full)\b"
        r".*\b(?:implementation|guide|explanation|walkthrough|write-?up)\b",
        re.IGNORECASE,
    ),
]

# URL を含み、かつファイル書込み/出力意図のあるクエリは「取得 → 書込み」の連鎖を
# 要するため meta_cognitive 層へ振る。deliberative は 1 ターン 1 ツールで連鎖できず、
# long_form は URL を取得せず散文を生成する (内容を捏造する) ため、いずれも不適。
_URL_HINT_RE = re.compile(r"https?://", re.IGNORECASE)
_FILE_WRITE_INTENT_RE = re.compile(
    r"(?:ファイル|file|csv|excel|エクセル|xlsx|スプレッドシート|ドキュメント|word|ワード"
    r"|powerpoint|パワーポイント|パワポ|pptx|プレゼンテーション)"
    r"|(?:出力|保存|書き出|書き込|エクスポート|export|セーブ"
    r"|(?<![A-Za-z])save(?![A-Za-z]))",
    re.IGNORECASE,
)

# URL を伴わずローカルパスへ書き出す意図のクエリ (例: 「C:\\...\\aa 配下に Excel で
# カレンダーを作成して出力」) も「生成 → 書込み」の連鎖を要するため meta_cognitive 層へ
# 振る。long_form (文書系名詞) にも _is_url_write_intent にも当たらないデータ成果物
# (カレンダー/一覧表等) を拾い、deliberative のディレクトリ書込み除外による拒否を防ぐ。
# CJK (ひらがな / カタカナ / 漢字) の検出。クエリ長判定の分岐に使う。
_CJK_CHAR_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
# ローカルファイルパス (Windows ドライブ / Unix) の存在検出。
_LOCAL_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/]"                  # Windows ドライブパス (C:\ / C:/)
    r"|(?:^|[\s　])(?:/[\w._-]+){2,}",  # Unix パス (/home/user/...)
)
# 書込み「動詞」のみ。名詞 (excel / docx 等) 単独では発火させない (「report.xlsx を
# 読んで」のような read 文脈で誤検出しないため)。
_WRITE_VERB_RE = re.compile(
    r"(?:作成|作って|生成|出力|保存|書[きい]|書込|エクスポート|export"
    r"|(?<![A-Za-z])save(?![A-Za-z])"
    r"|(?<![A-Za-z])write(?![A-Za-z])"
    r"|(?<![A-Za-z])create(?![A-Za-z])"
    r"|(?<![A-Za-z])output(?![A-Za-z]))",
    re.IGNORECASE,
)
# 既に書かれた成果物を **指し示す** 連体修飾節。「いま書いたそのファイルを読み込んで」
# の「書いた」は依頼された動作ではなく対象の説明であり、書込み意図ではない。
# これを書込み動詞として拾うと、純粋な読み取り依頼が meta_cognitive (書込み志向の
# プランナ) へ振られ、read → 「合計を計算」→「中身を表示」のような非ファイル
# サブタスクまでファイルツールで実行しようとして破綻する (実インシデント
# 2026-07-29 ライブ監査: 「いま書いたそのファイルを読み込んで、金額の合計を
# 計算してください」が 3 タスクに分解され、うち 2 つが実在しないパス
# (prices.txt / unknown) の read_file になって失敗し、質問は無回答のまま
# 「1 件のタスクを完了し、2 件が失敗しました。」だけが返った)。
_DESCRIPTIVE_WRITE_CLAUSE_RE = re.compile(
    r"(?:書[きい]た|作成した|作った|生成した|保存した|出力した|書き込んだ)"
    r"\s*(?:ばかりの?|ところの?)?"
    r"\s*(?:その|この|あの|先ほどの?|さっきの?)?"
    r"\s*(?:ファイル|もの|やつ|データ|内容|中身)"
    r"|(?:you\s+(?:just\s+)?(?:wrote|created|saved|generated))",
    re.IGNORECASE,
)
# 保存先を直前の文脈に委ねる参照表現。「同じファイルに保存し直して」のような
# 追記・修正依頼はパスを本文に持たないため _LOCAL_PATH_RE に掛からず、
# deliberative に落ちて read_file だけが走り、書込みが一度も起きないまま
# 「保存し直した」体の回答になっていた (実測 2026-07-27)。
# 定義は core.intent_vocab が SSOT (agent.feedback の訂正判定でも同じ意味で使う)。
_REFERENTIAL_WRITE_TARGET_RE = REFERENTIAL_WRITE_TARGET_RE
# 表形式データの出力先拡張子。long_form (散文ユニット分割) では表構造を
# 生成できないため、これらへの書出し意図は long_form 判定より優先して
# local_write_intent (write-fast 経路) に振る。
_TABULAR_TARGET_RE = re.compile(
    r"\.(?:csv|tsv|xlsx|ods)(?![A-Za-z])", re.IGNORECASE,
)

# 学習済み long_form パターン単独発火の抑止床。閾値以上の一致が 1 語のみの
# 場合、その重み合計がこの値以上でなければ long_form に分類しない。
_LEARNED_LONG_FORM_MIN_WEIGHT_SUM = 0.8

# how-to / 質問形マーカー。「作成する方法を教えて」のような書込み動詞を含む
# 知識質問を local_write_intent から除外する。``_is_knowledge_query`` は
# 「一覧を…」等のデータ語を広く拾い正当な書込みコマンドまで除外してしまうため、
# ここでは教示・疑問マーカーに限定する。locale で切替えず単一の正規表現に
# JA/EN 両方の教示・疑問フレーズを併記する (末尾の [?？] は locale 非依存の
# フォールバック)。英語側は疑問符依存のみだと "How do I create this report
# at C:\reports\" のように疑問符無しで how-to 意図を書くクエリを取りこぼす
# ため、明示的な how-to フレーズを追加した (2026-07-22 監査で判明)。
_HOWTO_QUERY_RE = re.compile(
    r"(?:教えて|おしえて|どうやって|どうすれば|どうやったら"
    r"|とは|って何|何ですか|ですか|ますか|でしょうか|ありますか|[?？]"
    r"|how\s+(?:do|can|should|would)\s+i\b|how\s+to\b"
    r"|(?:could|can|would)\s+you\s+(?:tell|show)\s+me\s+how\b"
    r"|tell\s+me\s+how\b|what'?s?\s+the\s+way\s+to\b)",
    re.IGNORECASE,
)

# 既出の成果物から一部だけを抜き出す / 並べ直す依頼。生成量を **減らす**
# 依頼なので、ユニット分割の長文生成は categorically 不適
# (実インシデント 2026-07-29 ライブ監査: 800 字の記事を書かせた直後に
# 「その記事の見出しだけを箇条書きで並べてください。」と頼んだところ、
# 見出しの一覧ではなく春の話題の散文生成が始まった)。
_EXTRACTION_REQUEST_RE = re.compile(
    r"(?:見出し|表題|タイトル|項目|要点|ポイント|キーワード|一覧|リスト)"
    r"\s*(?:だけ|のみ)"
    r"|(?:だけ|のみ)\s*を?\s*(?:並べ|挙げ|列挙|抜き出|抜粋|取り出)"
    r"|(?:headings?|titles?|key\s*points?|items?)\s+only"
    r"|(?:list|extract)\s+(?:just|only)\s+the",
    re.IGNORECASE,
)

# 複雑度を示すキーワードパターン（同義語・表記揺れ対応済み）
COMPLEX_KEYWORDS = [
    "比較", "なぜ", "どのように", "違い", "分析", "解析",
    "メリット", "デメリット", "利点", "欠点", "長所", "短所",
    "原因", "理由", "要因", "根本原因",
    "仕組み", "構造", "アーキテクチャ",
]

# COMPLEX_KEYWORDS の英語版。"how"/"cons"/"pros" 等の短い英単語は
# "however"/"prospect"/"considering" 等への substring 誤爆を防ぐため、
# 単純な文字列一致ではなく ASCII 境界ガード付き正規表現で構成する
# (META_KEYWORDS_EN_PATTERNS の慣用句を踏襲)。
COMPLEX_KEYWORDS_EN_PATTERNS = [
    re.compile(r"(?<![A-Za-z])explain(?:ed|ing|s)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])compar(?:e[ds]?|ing|ison)(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])analy[sz]e[ds]?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])analy[sz]ing(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])why(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])how(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])difference[s]?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])pros(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])cons(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])trade[\s-]?offs?(?![A-Za-z])", re.IGNORECASE),
]

# 履歴検索が必要なキーワード → Deliberative 層
# 「覚えて」「最初に」は当初リストに含まれておらず、20+ ターンの長い会話で
# 「この会話の最初に〜覚えてますか」型の recall 質問が reactive の既定分岐
# (step 12) まで素通りし、reactive_light 経路 (直近 REACTIVE_LIGHT_HISTORY_TURNS
# 件のみ・SemMem/STM 注入なし) に落ちて古いターンの内容を想起できない実インシデントが
# あった (2026-07-19)。deliberative に昇格させれば build_semmem_injection 経由で
# STM ノート (pin ブースト込み) が注入されるため、search_history ツール発火の有無に
# 関わらず recall 精度が回復する。
#
# 「過去の会話」「会話履歴」等の明示的な履歴検索語も対象。これらが無いと、
# ユーザーが明示的に検索を依頼しても知識質問プレフィルタで落ち、
# search_history が一度も発火しないまま「確認したが見当たらない」と
# 未確認のまま断言する (実インシデント 2026-07-27 ライブ検証:
# 「過去の会話で、登山の話題をしたことはありますか？探してください。」が
# tool_call_decision=no_tool / reason=no_match_in_any_layer で素通りし、
# 実際には 1 時間前に登山の会話があったのに「見当たりません」と回答)。
# 履歴参照キーワードは core.intent_vocab が SSOT (キーワードと「近接 / 長距離」の
# 距離分類を 1 つの表で持つ)。従来ここに定義されていたため、後方互換で再輸出する。
HISTORY_KEYWORDS = _HISTORY_KEYWORDS
HISTORY_KEYWORDS_EN = _HISTORY_KEYWORDS_EN

# Meta-Cognitive 層へのエスカレーションキーワード（同義語拡張済み）
# 注意: 命令形（〜して）のみ対象。「計画を教えて」等の知識質問は
# ここに含めない（RAG パイプラインで処理する）。
META_KEYWORDS = [
    "ステップ", "手順", "実装して", "作って", "作成して", "書いて",
    "リファクタ", "修正して", "デバッグ", "設計して", "組み立て",
    "構築", "ビルド", "テストして", "動かして",
    "implement", "refactor", "fix", "debug",
    "build", "design",
]

# META_KEYWORDS の英語版。短い英単語の substring 誤爆 ("prefix"/"contest"/
# "rebuild" 等) を防ぐため、単純な文字列一致ではなく ASCII 境界ガード付き
# 正規表現で構成する (_EXECUTABLE_QUERY_PATTERNS の慣用句を踏襲)。
META_KEYWORDS_EN_PATTERNS = [
    re.compile(r"(?<![A-Za-z])steps?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"step[\s-]by[\s-]step", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])implement(?:ed|ing|ation)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])write(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])create(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])build(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])refactor(?:ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])fix(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])debug(?:ging)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])design(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"set\s*up", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])construct(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])test(?:ed|ing)?(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"run\s+(?:it|this|that)\b", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])make(?![A-Za-z])", re.IGNORECASE),
]

# 知識質問パターン — ツール呼び出しではなく RAG で処理すべきクエリ
# これらにマッチするクエリはツール/Meta-Cognitive エスカレーションをスキップする
#
# 「明示的な疑問形式」(strict) と「名詞+助詞だけの緩いパターン」(loose) を
# 分けて定義し、全体はその合成として組み立てる。両者を 1 つのリストに混ぜて
# index で除外すると (旧実装は ``if i != 1``)、リストに要素を 1 つ挿入した
# 瞬間に除外対象が黙って別のパターンへすり替わる。
_KNOWLEDGE_QUERY_STRICT_PATTERNS = [
    re.compile(r"(?:教えて|おしえて|とは|って何|ですか|でしょうか|ありますか)", re.IGNORECASE),
    re.compile(r"(?:について|に関して|に関する)", re.IGNORECASE),
    re.compile(r"(?:知りたい|確認したい|調べたい|わかる|分かる)", re.IGNORECASE),
    re.compile(r"(?:what is|tell me|explain|describe|how does)\b", re.IGNORECASE),
]

# 名詞 + 助詞のみで判定するため緩く、コーディングモードの正規な生成依頼
# ("サンプルCSVデータを作成し" 等) の一部にも誤マッチする (2026-07-22 ライブ
# 検証で発覚)。strict 側 (教えて/について/知りたい 等) は明示的な疑問形式のみを
# 対象とするため誤爆が少ない。coding_meta_keywords の is_knowledge 上書き判定
# では、この緩いパターンを除いた strict のみを知識質問シグナルとして扱う。
_KNOWLEDGE_QUERY_LOOSE_PATTERN = re.compile(
    r"(?:資料|情報|データ|内容|概要|詳細|特徴|一覧).*(?:は|を|が|に)", re.IGNORECASE,
)

KNOWLEDGE_QUERY_PATTERNS = [
    *_KNOWLEDGE_QUERY_STRICT_PATTERNS,
    _KNOWLEDGE_QUERY_LOOSE_PATTERN,
]

# KNOWLEDGE_QUERY_PATTERNS の英語版。「about」は汎用前置詞のため直訳せず、
# 限定的な表現で構成する (誤爆抑制)。
KNOWLEDGE_QUERY_PATTERNS_EN = [
    re.compile(
        r"\b(?:what\s+is|what's|what\s+are|tell\s+me\s+(?:about|more)|explain"
        r"|describe|how\s+does|how\s+do|could\s+you\s+explain)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tell\s+me\s+about|what\s+about|information\s+(?:about|on)"
        r"|details?\s+(?:about|on)|overview\s+of|regarding)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:want\s+to\s+know|would\s+like\s+to\s+know|i'?d\s+like\s+to\s+know"
        r"|curious\s+about|want\s+to\s+understand|want\s+to\s+check)\b",
        re.IGNORECASE,
    ),
]

# ツール呼び出しを示すパターン（Meta-Cognitive 層へのエスカレーション対象）
# 注意: 単純なファイル読み書きは Deliberative 層の ToolCallJudge が処理する。
# ここには複数ツールの連携が必要なパターンのみ含める。
# 「検索」等の汎用語は知識質問にもマッチするため、コード/ファイル文脈を要求する。
TOOL_PATTERNS = [
    # ファイル操作: 読み取り+変更など複合操作のみ（単一操作は Deliberative で処理）
    re.compile(r"(?:ファイル|file).*(?:読|開).*(?:書|修正|変更|削除|追加)", re.IGNORECASE),
    re.compile(r"(?:コマンド|command).*(?:実行|run)", re.IGNORECASE),
    # コード/ファイル検索: 汎用「検索」ではなく、コード・ファイル文脈を要求
    re.compile(r"(?:コード|ファイル|ソース|関数|クラス|code|file|source|function|class).*(?:検索|search|grep|find)", re.IGNORECASE),
    re.compile(r"(?:検索|search|grep|find).*(?:コード|ファイル|ソース|code|file|source)", re.IGNORECASE),
    re.compile(r"(?:URL|url|https?://)", re.IGNORECASE),
    re.compile(r"(?:計算|calculate)\s", re.IGNORECASE),
]

# TOOL_PATTERNS の英語版。末尾4パターン (コード検索/URL/計算) は元々
# ASCII 語彙のみで日英両対応済みのためそのまま複製する。1つ目 (ファイル
# 読み書き複合操作) のみ、日本語版が活用語尾ゲート限定のため英語文で
# 発火しない (128行目相当) ので、英語動詞で作り直す。2つ目 (コマンド実行)
# も日本語活用語尾 (して/する) を要求せずに発火するよう緩和する。
TOOL_PATTERNS_EN = [
    re.compile(r"\bfile\b.*\b(?:read|open)\b.*\b(?:write|modify|change|update|delete|remove|edit|append)\b", re.IGNORECASE),
    re.compile(r"\b(?:run|execute|exec)\b.*\bcommand\b|\bcommand\b.*\b(?:run|execute|exec)\b", re.IGNORECASE),
    re.compile(r"(?:コード|ファイル|ソース|関数|クラス|code|file|source|function|class).*(?:検索|search|grep|find)", re.IGNORECASE),
    re.compile(r"(?:検索|search|grep|find).*(?:コード|ファイル|ソース|code|file|source)", re.IGNORECASE),
    re.compile(r"(?:URL|url|https?://)", re.IGNORECASE),
    re.compile(r"(?:計算|calculate)\s", re.IGNORECASE),
]

# Python 実行で正確に答えられるクエリのパターン
# 知識質問パターンにマッチしてもこれらを含む場合は deliberative に昇格して
# ToolCallJudge によるツール実行を誘導する
_EXECUTABLE_QUERY_PATTERNS = [
    # 注意: \b は日本語文字を \w とみなすため英語-日本語境界で機能しない。
    # 英語の短いキーワードは (?<![A-Za-z])...(?![A-Za-z]) で ASCII 境界を使用。
    # CPU/RAM/GPU/VRAM/spec は ASCII 境界必須。境界を付けないと IGNORECASE で
    # "program" / "diagram" / "telegram" の 'ram' に部分マッチし、無関係な
    # クエリが executable_query として deliberative へ強制昇格していた。
    # tool_call_judge 側は 2026-07-22 監査で塞がれたが、こちらは残っていた。
    re.compile(
        r"(?:スペック|メモリ|ディスク|容量|ストレージ|"
        + ascii_boundary_alternation("CPU", "RAM", "GPU", "VRAM", "spec")
        + r")",
        re.IGNORECASE,
    ),
    # 「何月|何日|何曜日」は明確な疑問語のみ追加 (「今日|明日|昨日」単独は
    # 「今日の予定」等の文脈で誤検出するため見送り)。
    re.compile(r"(?:何時|何月|何日|何曜日|日時|日付|現在時刻|(?<![A-Za-z])today(?![A-Za-z])|(?<![A-Za-z])now(?![A-Za-z])|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:IP\s*アドレス|ホスト名|(?<![A-Za-z])hostname(?![A-Za-z])|(?<![A-Za-z])ip\s*address)", re.IGNORECASE),
    re.compile(r"(?:(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:Python|python)\s*(?:バージョン|version)", re.IGNORECASE),
    re.compile(r"(?:環境変数|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:階乗|素数|フィボナッチ|素因数|進数変換|桁)", re.IGNORECASE),
    re.compile(r"(?:集計|合計|平均|中央値|標準偏差|ソート|統計)", re.IGNORECASE),
    re.compile(r"(?:変換|エンコード|デコード|Base64|ハッシュ|タイムスタンプ)", re.IGNORECASE),
]

# _EXECUTABLE_QUERY_PATTERNS の英語版。
_EXECUTABLE_QUERY_PATTERNS_EN = [
    re.compile(r"(?<![A-Za-z])(?:specs?|CPU|memory|RAM|GPU|VRAM|disk|capacity|storage)(?![A-Za-z])", re.IGNORECASE),
    re.compile(
        r"\b(?:what'?s?\s*(?:the\s*)?(?:time|date)|current\s*(?:time|date)"
        r"|today'?s?\s*date|what\s*day\s*(?:is\s*it|of\s*the\s*week))\b"
        r"|(?<![A-Za-z])date(?![A-Za-z])|(?<![A-Za-z])time(?![A-Za-z])",
        re.IGNORECASE,
    ),
    re.compile(r"(?:IP\s*address|hostname|(?<![A-Za-z])hostname(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:(?<![A-Za-z])OS(?![A-Za-z])|operating\s*system|(?<![A-Za-z])Windows(?![A-Za-z])|(?<![A-Za-z])Linux(?![A-Za-z])|(?<![A-Za-z])Mac(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"(?:Python|python)\s*version", re.IGNORECASE),
    re.compile(r"(?:environment\s*variable|(?<![A-Za-z])env(?![A-Za-z])|(?<![A-Za-z])PATH(?![A-Za-z]))", re.IGNORECASE),
    re.compile(r"\b(?:factorial|prime(?:\s*numbers?)?|fibonacci|prime\s*factorization|base\s*conversion|number\s*of\s*digits?|digits?)\b", re.IGNORECASE),
    # sum/average/mean/sort は日常会話で極めて頻出する多義語 ("What do you
    # mean?"/"I sort of agree"/"on average, this works fine") のため、
    # 単独では発火させず数値/データ文脈語との近接共起を要求する (2026-07-22
    # 監査で判明)。total/median/standard deviation/std dev/statistics/
    # aggregate は既存テスト (test_tool_call_judge.py の bare "What's the
    # total?") が単独発火を前提としており、日常会話での多義性も相対的に
    # 低いため単独発火のまま維持する。
    re.compile(r"\b(?:total|median|standard\s*deviation|std\s*dev|statistics|aggregate)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:sum|average|mean|sort(?:ed|ing)?)\b.{0,20}"
        r"\b(?:numbers?|data|list|array|values?|dataset|figures?)\b"
        r"|\b(?:numbers?|data|list|array|values?|dataset|figures?)\b.{0,20}"
        r"\b(?:sum|average|mean|sort(?:ed|ing)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:convert|encode|decode|encoding|decoding|Base64|hash(?:ing)?|timestamp)\b", re.IGNORECASE),
]

# 挨拶・簡単な定型パターン → Reactive 層で即応答
# 体裁 (クエリ全体が挨拶だけであることの要求) は core.intent_vocab から派生させる。
# 語彙は reactive.GREETING_RESPONSES とは **意図的に別** (あちらは応答文と 1 対 1 で
# 対応させるため分割が細かく、「おはようございます」等も個別に拾う)。
GREETING_PATTERNS = [
    re.compile(
        exact_greeting_pattern(
            r"こんにち[はわ]|おはよう|こんばんは|やあ|ども|hello|hi|hey",
            punctuation=GREETING_PUNCTUATION_JA,
        ),
        re.IGNORECASE,
    ),
    re.compile(
        exact_greeting_pattern(
            r"ありがと[うございます]*|thanks|thank you",
            punctuation=GREETING_PUNCTUATION_JA,
        ),
        re.IGNORECASE,
    ),
    re.compile(
        exact_greeting_pattern(
            r"おやすみ|さようなら|bye|goodbye",
            punctuation=GREETING_PUNCTUATION_JA,
        ),
        re.IGNORECASE,
    ),
]


def _matches_any(patterns: list[re.Pattern], query: str) -> bool:
    """パターンリストのいずれかが query にマッチするか判定する。

    JA/EN 両方のパターンリストを GUI locale に関わらず両方評価する用途で使う
    (2026-07-22 発見: 以前は locale で片方の言語のみ評価しており、GUI locale
    と実際の入力言語が食い違うと該当言語側の判定が一切効かなかった)。
    """
    return any(p.search(query) for p in patterns)


#: ルール表で meta_cognitive を表すセンチネル。実際の層はマッチ時に
#: ``_guard_meta_cognitive()`` で解決する (コンテキスト予算が足りなければ
#: deliberative へ降格するため、表に静的な層名は書けない)。
_META = "__meta_cognitive__"


class _ClassifyContext:
    """1 回の ``classify()`` 呼出で共有する導出値。

    ``is_knowledge`` / ``is_short`` 等は複数のルールが参照するため、都度
    計算せずここでキャッシュする。すべて純粋関数の結果なので、評価タイミングが
    変わっても判定は変わらない (早期マッチ時に評価されないよう遅延させている)。
    """

    __slots__ = ("_c", "_cache", "mode", "query", "rag_results")

    def __init__(
        self,
        classifier: "ComplexityClassifier",
        query: str,
        mode: str,
        rag_results: list,
    ) -> None:
        self._c = classifier
        self.query = query
        self.mode = mode
        self.rag_results = rag_results
        self._cache: dict[str, object] = {}

    def _memo(self, key: str, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def is_knowledge(self) -> bool:
        return self._memo("k", lambda: self._c._is_knowledge_query(self.query))

    @property
    def is_short(self) -> bool:
        return self._memo(
            "s", lambda: self._c._is_short_query(self.query, self.mode),
        )

    @property
    def is_long_form_candidate(self) -> bool:
        return self._memo("lf", lambda: self._c._detect_long_form(self.query))

    @property
    def rag_threshold(self) -> float:
        return self._memo(
            "rt",
            lambda: self._c._get_policy(
                "router", "rag_score_threshold", self.mode, 0.8,
            ),
        )


@dataclass(frozen=True)
class _ClassifyRule:
    """``classify()`` の 1 判定。表中の並び順がそのまま優先度。"""

    #: decision.jsonl / policy_adjuster へ渡る matched-rule 識別子。
    reason: str
    #: 遷移先。``_META`` はマッチ時にガード経由で解決する。
    layer: str
    predicate: Callable[["ComplexityClassifier", _ClassifyContext], bool]
    #: マッチ時に ``is_long_form`` へ設定する値。
    long_form: bool = False


#: 判定順序の SSOT。**並び順が優先度**であり、番号は振らない
#: (かつて 1.4 / 2.5 / 8.5 のような小数が後付け層の痕跡として残り、
#: 欠番 3 / 4 を docstring が参照する事故も起きていた)。
_CLASSIFY_RULES: tuple[_ClassifyRule, ...] = (
    _ClassifyRule(
        "greeting", "reactive",
        lambda c, x: c._is_greeting(x.query),
    ),
    # URL 取得 → ファイル書込みの連鎖は meta_cognitive でしか実行できない。
    # 学習語による long_form 誤振り分けより先に評価する。
    _ClassifyRule(
        "url_write_intent", _META,
        lambda c, x: c._is_url_write_intent(x.query, x.mode),
    ),
    # 表形式データ出力先 (.csv/.tsv/.xlsx/.ods) はユニット分割の散文生成が
    # 構造を壊すため long_form を抑止して write-fast 経路へ落とす
    # (2026-07-15: annual_events.csv が 26 unit の散文 CSV 化)。
    # long_form 候補の部分集合なので long_form ルールより前に置く。
    _ClassifyRule(
        "local_write_intent", _META,
        lambda c, x: (
            x.is_long_form_candidate and c._is_tabular_write_intent(x.query, x.mode)
        ),
    ),
    _ClassifyRule(
        "long_form", _META,
        lambda c, x: x.is_long_form_candidate,
        long_form=True,
    ),
    # long_form (文書系名詞) にも url_write_intent にも当たらないデータ成果物
    # (Excel カレンダー/一覧表等) のローカル出力を拾う。long_form 判定の後に
    # 置くことで「仕様書/ドキュメント + パス」は従来どおり long_form を維持する。
    _ClassifyRule(
        "local_write_intent", _META,
        lambda c, x: c._is_local_write_intent(x.query, x.mode),
    ),
    # coding の正規な生成依頼が executable_query へ短絡するのを防ぐため、
    # executable_query より先に評価する (2026-07-22: 「CSV を集計する
    # プログラムを作成して」が staged 生成へ一切到達しなかった)。
    _ClassifyRule(
        "coding_tools", _META,
        lambda c, x: (
            is_coding_mode(x.mode) and not x.is_knowledge and c._needs_tools(x.query)
        ),
    ),
    # 緩い名詞+助詞パターンは正規の生成依頼にも誤マッチするため、ここでは
    # strict 側 (明示的な疑問形式) だけを知識質問として扱う。
    _ClassifyRule(
        "coding_meta_keywords", _META,
        lambda c, x: (
            is_coding_mode(x.mode)
            and c._has_meta_keywords(x.query)
            and not c._is_strict_knowledge_query(x.query)
        ),
    ),
    # 「明日は何月何日?」のような ? 終わりの疑問文は知識質問パターンが要求する
    # 末尾形式を満たさないため、is_knowledge を見ずに常時評価する。
    _ClassifyRule(
        "executable_query", "deliberative",
        lambda c, x: c._contains_executable_query_keywords(x.query),
    ),
    _ClassifyRule(
        "learned_tool_pattern", "deliberative",
        lambda c, x: x.is_knowledge and c._contains_learned_tool_patterns(x.query),
    ),
    _ClassifyRule(
        "history_ref", "deliberative",
        lambda c, x: c._has_history_keywords(x.query),
    ),
    _ClassifyRule(
        "complex_keywords", "deliberative",
        lambda c, x: c._has_complex_keywords(x.query),
    ),
    _ClassifyRule(
        "short_high_rag", "reactive",
        lambda c, x: bool(
            x.is_short and x.rag_results and x.rag_results[0][1] > x.rag_threshold
        ),
    ),
    # reactive / reactive_light は検索パイプラインを一切走らせないため、短い
    # 知識質問を short_query で reactive に落とすとカートリッジを参照せず
    # 事前知識だけで答えてしまう。short_high_rag の後に置くことで、RAG ヒット
    # 済みの短文即応答は従来挙動を保つ。
    _ClassifyRule(
        "knowledge_query", "deliberative",
        lambda c, x: x.is_knowledge,
    ),
    _ClassifyRule(
        "short_query", "reactive",
        lambda c, x: x.is_short,
    ),
    _ClassifyRule(
        "no_rag_results", "deliberative",
        lambda c, x: not x.rag_results,
    ),
    _ClassifyRule(
        "tool_patterns", _META,
        lambda c, x: c._needs_tools(x.query),
    ),
    _ClassifyRule("default", "reactive", lambda c, x: True),
)


class ComplexityClassifier:
    """クエリの複雑度を分類し、適切なエージェント層を決定する

    分類はすべてルールベース。LLM 呼び出しゼロ。
    ランタイムガードにより、Meta-Cognitive 層が利用できない場合は
    Deliberative 層にフォールバックする。
    """

    def __init__(
        self,
        config: dict | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        policy: PolicyInterpreter | None = None,
    ):
        self._config = config or {}
        self._policy = policy
        self.is_long_form: bool = False
        self._learned_patterns = learned_patterns
        # ツールパターン学習の閾値
        learning_cfg = self._config.get("learning", {})
        self._tool_routing_threshold: float = learning_cfg.get(
            "tool_pattern_match_threshold", 0.4,
        )
        # 長文生成パターン学習の閾値 (ハードコード regex に加えて
        # `category="long_form"` の学習済みパターンも OR 判定する)
        self._long_form_threshold: float = learning_cfg.get(
            "long_form_pattern_match_threshold", 0.4,
        )

    def classify(
        self,
        query: str,
        mode: str = "chat",
        rag_results: list | None = None,
        context_turns: int = 0,  # noqa: ARG002
    ) -> str:
        """クエリの複雑度を分類する

        判定は ``_CLASSIFY_RULES`` の **並び順がそのまま優先度** で、最初に
        マッチしたルールの層を返す。順序と条件はすべて表側にあるので、本体は
        表を上から評価するだけ。ルールを足すときは表の適切な位置へ 1 エントリ
        加える (以前は 18 個の if が本体に並び、後付け層が 1.4 / 2.5 / 8.5 の
        ような小数コメントで表現されていた)。

        Returns:
            "reactive" | "deliberative" | "meta_cognitive"
        """
        rag_results = rag_results or []
        self.is_long_form = False
        self._classify_mode = mode
        self._last_classify_reason = "default"

        ctx = _ClassifyContext(self, query, mode, rag_results)
        for rule in _CLASSIFY_RULES:
            if not rule.predicate(self, ctx):
                continue
            self.is_long_form = rule.long_form
            layer = (
                self._guard_meta_cognitive()
                if rule.layer == _META
                else rule.layer
            )
            return self._record_classification(layer, rule.reason, query)

        # 表は必ず default ルールで終わるため到達しない。
        return self._record_classification("reactive", "default", query)

    def _record_classification(self, layer: str, reason: str, query: str) -> str:
        """分類結果をログし matched-rule 識別子を保持して layer を返す。

        ``self._last_classify_reason`` に matched-rule 識別子を残す。chat 側が
        primary routing を decision.jsonl に記録する際の ``reason`` に使う
        (context={"mode": ...} と併せて policy_adjuster の mode 別学習へ供給)。
        """
        self._last_classify_reason = reason
        logger.info("Classified as %s (%s): %s", layer, reason, query[:50])
        return layer

    def _guard_meta_cognitive(self) -> str:
        """Meta-Cognitive 層のランタイムガード

        設定で無効化されている場合や、コンテキスト予算が不足する場合は
        Deliberative 層にフォールバックする。
        """
        agent_cfg = self._config.get("agent", {})

        # 明示的無効化チェック
        if not agent_cfg.get("meta_cognitive_enabled", True):
            logger.info("Meta-Cognitive disabled by config, falling back to deliberative")
            return "deliberative"

        # コンテキスト予算チェック
        mode = getattr(self, "_classify_mode", "chat")
        if not _can_use_meta_cognitive(self._config, self._policy, mode):
            logger.warning(
                "Meta-Cognitive budget insufficient, falling back to deliberative"
            )
            return "deliberative"

        return "meta_cognitive"

    def _is_short_query(self, query: str, mode: str = "chat") -> bool:
        """クエリが短い（単純）かどうかを判定

        日本語テキストはスペースで分割されないため、文字数で判定する。
        分岐は **表記体系** で決める。空白トークン数で分岐すると、日本語文に
        ASCII 断片 (パス / コマンド / 識別子) が混ざっただけで英語扱いになり、
        長い日本語クエリが「短文」と誤判定される (実インシデント 2026-07-29
        ライブ監査: 55 文字の「コマンド `dir E:\\tmp\\x` を実行して、返ってきた
        出力をそのまま報告してください。」が空白 4 トークンなので short_query
        と判定され、検索もツール判定も走らない reactive に落ちて、ベースモデルが
        実行していないコマンドの出力を捏造した)。
        """
        min_tokens = self._get_policy("router", "short_query_min_tokens", mode, 3)
        max_tokens = self._get_policy("router", "short_query_max_tokens", mode, 10)
        max_chars = self._get_policy("router", "short_query_max_chars", mode, 20)

        # 英語テキスト: 単語数で判定 (CJK を含まない場合のみ)
        if not _CJK_CHAR_RE.search(query):
            tokens = len(query.split())
            if tokens >= min_tokens:
                return tokens < max_tokens
        # 日本語テキスト: 文字数で判定
        return len(query.strip()) < max_chars

    def _is_greeting(self, query: str) -> bool:
        """挨拶パターンを検出"""
        return any(p.match(query.strip()) for p in GREETING_PATTERNS)

    def _has_complex_keywords(self, query: str) -> bool:
        """複雑度を示すキーワードを検出"""
        q_lower = query.lower()
        if any(kw in q_lower for kw in COMPLEX_KEYWORDS):
            return True
        return _matches_any(COMPLEX_KEYWORDS_EN_PATTERNS, query)

    def _has_history_keywords(self, query: str) -> bool:
        """履歴参照キーワードを検出"""
        q_lower = query.lower()
        return any(kw in q_lower for kw in HISTORY_KEYWORDS) or any(
            kw in q_lower for kw in HISTORY_KEYWORDS_EN
        )

    def _has_meta_keywords(self, query: str) -> bool:
        """Meta-Cognitive 層へのエスカレーションキーワードを検出"""
        q_lower = query.lower()
        if any(kw in q_lower for kw in META_KEYWORDS):
            return True
        return _matches_any(META_KEYWORDS_EN_PATTERNS, query)

    def _is_url_write_intent(self, query: str, mode: str = "chat") -> bool:
        """URL からデータを取得してファイルに書き出す意図を検出する。

        chat モードのみ対象 (coding モードは step 2.5 の ``_needs_tools`` が同等の
        meta_cognitive 振り分けを行う)。URL とファイル書込み/出力意図の双方が
        揃った場合のみ True を返す。
        """
        if not is_chat_mode(mode):
            return False
        if not _URL_HINT_RE.search(query):
            return False
        return bool(_FILE_WRITE_INTENT_RE.search(query))

    def _is_local_write_intent(self, query: str, mode: str = "chat") -> bool:
        """ローカルパス + ファイル書込み意図 (URL 無し) を検出する。

        chat モードのみ対象 (coding モードは step 2.5 / 2.6 の ``_needs_tools`` /
        ``_has_meta_keywords`` が同等の振り分けを行う)。URL を含む場合は
        ``_is_url_write_intent`` / fetch 経路に委ね、知識質問 (「作成方法を
        教えて」等の how-to) は除外する。ローカルパスと書込み動詞の双方が
        揃った場合のみ True を返す。
        """
        if not is_chat_mode(mode):
            return False
        if _URL_HINT_RE.search(query):
            return False
        if _HOWTO_QUERY_RE.search(query):
            return False
        # コマンドリテラル (`dir E:\tmp\x`) 内のパスは実行対象の引数であって
        # 書込み先ではない。書込み動詞・パスの双方をこれを除いた本文で判定する。
        probe = strip_command_literals(query)
        # 「いま書いたそのファイル」のような既出成果物への言及は、依頼された
        # 動作ではなく対象の説明。書込み動詞判定から除外する。
        probe = _DESCRIPTIVE_WRITE_CLAUSE_RE.sub(" ", probe)
        if not _WRITE_VERB_RE.search(probe):
            return False
        if _LOCAL_PATH_RE.search(probe):
            return True
        # パスは直前ターンにしか無い参照依頼 (「同じファイルに保存し直して」)。
        # 保存先は write-fast 経路が会話から解決する。
        return bool(_REFERENTIAL_WRITE_TARGET_RE.search(probe))

    def _is_tabular_write_intent(self, query: str, mode: str = "chat") -> bool:
        """表形式データ (.csv/.tsv/.xlsx/.ods) のローカル書出し意図を検出する。

        long_form (散文ユニット分割) では表構造の成果物を生成できないため、
        long_form 判定に勝つ優先ゲートとして使う。ローカル書込み意図と
        表形式拡張子の双方が揃った場合のみ True。
        """
        if not self._is_local_write_intent(query, mode):
            return False
        return bool(_TABULAR_TARGET_RE.search(query))

    def _detect_long_form(self, query: str) -> bool:
        """長文生成リクエストかどうかを判定

        ハードコード regex (LONG_FORM_PATTERNS) と
        ``LearnedPatternStore`` の ``category="long_form"`` を OR 判定する。
        後者は FeedbackCollector / LearningScheduler により自己進化する。
        """
        # URL を含むクエリは取得 (fetch) を要するためツール経路で扱う。散文生成
        # (long_form) は URL 内容を取得できず捏造するため、学習語が一致しても除外する。
        if _URL_HINT_RE.search(query):
            return False
        # how-to / 質問形は「作成方法の説明」を求めているのであり、実際の生成を
        # 依頼しているわけではない。_is_local_write_intent と同じ _HOWTO_QUERY_RE
        # で除外しないと、「手順書を作成する方法を教えて」/ "How do I create a
        # status report at <path>" のような howto 質問が実際に長文ドキュメントを
        # 生成・書込みしてしまう (2026-07-22 ライブ検証で判明。JA/EN 両方で
        # 再現し、PR#298-300 の対象範囲より前から存在した既存バグ)。
        if _HOWTO_QUERY_RE.search(query):
            return False
        # 「見出しだけ並べて」型は既出成果物の抽出であって生成ではない。
        if _EXTRACTION_REQUEST_RE.search(query):
            return False
        if _matches_any(LONG_FORM_PATTERNS, query) or _matches_any(
            LONG_FORM_PATTERNS_EN, query,
        ):
            return True
        return self._contains_learned_long_form_patterns(query)

    def _contains_learned_long_form_patterns(self, query: str) -> bool:
        """学習済み long_form パターンにマッチするか判定

        単一キーワード hit での発火は一般語の自己強化ループを招く
        (2026-07-15: 「文書」「明日」等の学習語 1 hit で朝礼メモや CSV 依頼が
        ユニット分割パイプラインへ流出)。閾値以上の一致が 2 語以上、または
        重み合計が ``_LEARNED_LONG_FORM_MIN_WEIGHT_SUM`` 以上の場合のみ発火する。

        さらに 2026-07-25 から、学習語だけで発火させるには文書種別名詞
        (``DOCUMENT_NOUN_LEARNABLE_JA``) を 1 語以上含むことを必須にした。
        学習側の許容リスト (``is_long_form_learnable``) と二重防御の関係で、
        ストアに既存の汚染語が残っていても誤発火しない
        (実測: 「ありがとう。技術的な議論の部分はとても有益だった」が
        技術的 0.65 + 技術 0.5 の 2 語成立で 6,436 字の長文生成に化けた)。
        """
        if self._learned_patterns is None:
            return False
        matches = self._learned_patterns.match(query, category="long_form")
        if not matches:
            return False
        if not any(noun in query for noun in DOCUMENT_NOUN_LEARNABLE_JA):
            return False
        eligible = [w for _, w in matches if w >= self._long_form_threshold]
        if len(eligible) >= 2:
            return True
        return sum(eligible) >= _LEARNED_LONG_FORM_MIN_WEIGHT_SUM

    def _is_knowledge_query(self, query: str) -> bool:
        """知識質問パターンを検出（RAG で処理すべきクエリ）"""
        return _matches_any(KNOWLEDGE_QUERY_PATTERNS, query) or _matches_any(
            KNOWLEDGE_QUERY_PATTERNS_EN, query,
        )

    def _is_strict_knowledge_query(self, query: str) -> bool:
        """緩い名詞+助詞パターン (_KNOWLEDGE_QUERY_LOOSE_PATTERN) を除いた、
        明示的な疑問形式のみによる知識質問判定。coding_meta_keywords の
        is_knowledge 上書き判定用 (詳細は _KNOWLEDGE_QUERY_STRICT_PATTERNS 参照)。
        """
        return _matches_any(_KNOWLEDGE_QUERY_STRICT_PATTERNS, query) or _matches_any(
            KNOWLEDGE_QUERY_PATTERNS_EN, query,
        )

    def _contains_executable_query_keywords(self, query: str) -> bool:
        """Python 実行で正確に答えられるクエリのキーワードを検出"""
        return _matches_any(_EXECUTABLE_QUERY_PATTERNS, query) or _matches_any(
            _EXECUTABLE_QUERY_PATTERNS_EN, query,
        )

    def _contains_learned_tool_patterns(self, query: str) -> bool:
        """学習済み tool_routing パターンにマッチするか判定"""
        if self._learned_patterns is None:
            return False
        matches = self._learned_patterns.match(query, category="tool_routing")
        if not matches:
            return False
        return matches[0][1] >= self._tool_routing_threshold

    def _needs_tools(self, query: str) -> bool:
        """ツール呼び出しが必要なパターンを検出"""
        return _matches_any(TOOL_PATTERNS, query) or _matches_any(
            TOOL_PATTERNS_EN, query,
        )

    def _get_policy(
        self, domain: str, key: str, mode: str, default: int | float,
    ) -> int | float:
        """PolicyInterpreter からパラメータを取得（フォールバック付き）"""
        return get_policy_value(self._policy, domain, key, default, mode=mode)


def _can_use_meta_cognitive(
    config: dict,
    policy: PolicyInterpreter | None = None,
    mode: str = "chat",
) -> bool:
    """コンテキスト予算が Meta-Cognitive ループに十分か判定"""
    loop_budget = resolve_meta_cognitive_loop_budget(config, mode)

    cfg_default = config.get("agent", {}).get("meta_cognitive_min_budget", 512)
    min_budget = get_policy_value(
        policy, "agent", "meta_cognitive_min_budget", cfg_default, mode=mode,
    )

    return loop_budget >= min_budget
