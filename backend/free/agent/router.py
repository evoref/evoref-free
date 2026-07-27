"""複雑度分類ルーター: クエリを3層エージェントに振り分ける"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.free.agent.context_budget import resolve_meta_cognitive_loop_budget
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
HISTORY_KEYWORDS = [
    "前に", "以前", "先週", "先月", "この間", "前回", "前の会話",
    "さっき", "昨日", "今朝", "先ほど", "最初に", "覚えて", "覚えてる",
    "覚えている",
    "過去の会話", "過去のやり取り", "過去に話", "以前の会話", "会話履歴",
    "earlier", "previously", "last time", "yesterday", "before",
    "remember", "recall",
]

# HISTORY_KEYWORDS の英語版。
HISTORY_KEYWORDS_EN = [
    "earlier", "previously", "last time", "yesterday", "before",
    "remember", "recall", "this morning", "just now", "a moment ago",
    "a while back", "at first", "in the beginning",
    "past conversation", "previous conversation", "conversation history",
    "chat history",
]

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
KNOWLEDGE_QUERY_PATTERNS = [
    re.compile(r"(?:教えて|おしえて|とは|って何|ですか|でしょうか|ありますか)", re.IGNORECASE),
    re.compile(r"(?:資料|情報|データ|内容|概要|詳細|特徴|一覧).*(?:は|を|が|に)", re.IGNORECASE),
    re.compile(r"(?:について|に関して|に関する)", re.IGNORECASE),
    re.compile(r"(?:知りたい|確認したい|調べたい|わかる|分かる)", re.IGNORECASE),
    re.compile(r"(?:what is|tell me|explain|describe|how does)\b", re.IGNORECASE),
]

# KNOWLEDGE_QUERY_PATTERNS[1] (資料|情報|データ|... + は/を/が/に) は名詞+助詞のみ
# で判定するため緩く、コーディングモードの正規な生成依頼 ("サンプルCSVデータを
# 作成し" 等) の一部にも誤マッチする (2026-07-22 ライブ検証で発覚)。それ以外
# (教えて/について/知りたい 等) は明示的な疑問形式のみを対象とするため誤爆が
# 少ない。coding_meta_keywords の is_knowledge 上書き判定では、この緩い
# パターンを除いた「明示的な疑問形式」のみを知識質問シグナルとして扱う。
_KNOWLEDGE_QUERY_STRICT_PATTERNS = [
    p for i, p in enumerate(KNOWLEDGE_QUERY_PATTERNS) if i != 1
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
    re.compile(r"(?:スペック|CPU|メモリ|RAM|GPU|VRAM|ディスク|容量|ストレージ|(?<![A-Za-z])spec(?![A-Za-z]))", re.IGNORECASE),
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
GREETING_PATTERNS = [
    re.compile(r"^(?:こんにち[はわ]|おはよう|こんばんは|やあ|ども|hello|hi|hey)\s*[!！。.]?\s*$", re.IGNORECASE),
    re.compile(r"^(?:ありがと[うございます]*|thanks|thank you)\s*[!！。.]?\s*$", re.IGNORECASE),
    re.compile(r"^(?:おやすみ|さようなら|bye|goodbye)\s*[!！。.]?\s*$", re.IGNORECASE),
]


def _matches_any(patterns: list[re.Pattern], query: str) -> bool:
    """パターンリストのいずれかが query にマッチするか判定する。

    JA/EN 両方のパターンリストを GUI locale に関わらず両方評価する用途で使う
    (2026-07-22 発見: 以前は locale で片方の言語のみ評価しており、GUI locale
    と実際の入力言語が食い違うと該当言語側の判定が一切効かなかった)。
    """
    return any(p.search(query) for p in patterns)


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

        Returns:
            "reactive" | "deliberative" | "meta_cognitive"
        """
        rag_results = rag_results or []
        self.is_long_form = False
        self._classify_mode = mode
        self._last_classify_reason = "default"

        # 1. 挨拶・定型パターン → reactive
        if self._is_greeting(query):
            return self._record_classification("reactive", "greeting", query)

        # 1.4. URL + ファイル書込み意図 → meta_cognitive
        #   「URL を取得してファイルに出力」は取得 → 書込みの連鎖を要し、これを実行
        #   できるのは meta_cognitive 層のみ。学習語による long_form 誤振り分け
        #   (step 1.5) より優先して評価する。
        if self._is_url_write_intent(query, mode):
            self.is_long_form = False
            return self._record_classification(
                self._guard_meta_cognitive(), "url_write_intent", query,
            )

        # 1.5. 長文生成判定 → meta_cognitive（Orchestrator に委任）
        #   ただし表形式データ出力先 (.csv/.tsv/.xlsx/.ods) はユニット分割の
        #   散文生成が構造を壊すため (2026-07-15: annual_events.csv が 26 unit
        #   の散文 CSV 化)、long_form を抑止して write-fast 経路へ落とす。
        if self._detect_long_form(query):
            if self._is_tabular_write_intent(query, mode):
                self.is_long_form = False
                return self._record_classification(
                    self._guard_meta_cognitive(), "local_write_intent", query,
                )
            self.is_long_form = True
            return self._record_classification(
                self._guard_meta_cognitive(), "long_form", query,
            )

        # 1.6. ローカルパス + ファイル書込み意図 (URL 無し) → meta_cognitive
        #   long_form (文書系名詞) にも url_write_intent にも当たらないデータ成果物
        #   (Excel カレンダー/一覧表等) のローカル出力を拾う。is_long_form=False の
        #   まま planner + write-fast 経路 (_dispatch_meta_cognitive) で dir→file
        #   解決 + 生成 → write_file を成立させる。long_form 判定後に評価することで
        #   「仕様書/ドキュメント + パス」は従来どおり long_form を維持する。
        if self._is_local_write_intent(query, mode):
            self.is_long_form = False
            return self._record_classification(
                self._guard_meta_cognitive(), "local_write_intent", query,
            )

        # 2. 知識質問の検出 → ツール/Meta-Cognitive エスカレーションをスキップ
        # コーディングモードでもカートリッジ検索による知識質問は有効。
        # 知識質問はツール操作ではないため、ここでツール判定をバイパスし
        # 後続の通常分類（reactive/deliberative）に委ねる。
        is_knowledge = self._is_knowledge_query(query)

        # 2.5. コーディングモードでツール操作が必要 → meta_cognitive
        #   (2026-07-22 まではこの判定より旧 2.5 の executable_query 判定が
        #   先に評価されており、「CSV を集計する“プログラムを作成して”」の
        #   ようなコーディングモードの正規の生成依頼が、集計/フィボナッチ/
        #   ソート等の汎用実行可能キーワードにマッチしただけで deliberative の
        #   単発ツール実行に短絡し、staged 生成パイプラインへ一切到達しない
        #   実害があった。is_coding_mode ゲートを保ったまま先に評価すること
        #   で、非コーディングモードの挙動 (即 no-op) は変えずに解消する。
        if is_coding_mode(mode) and not is_knowledge and self._needs_tools(query):
            return self._record_classification(
                self._guard_meta_cognitive(), "coding_tools", query,
            )

        # 2.6. Meta-Cognitive キーワード検出（コーディングモード）
        #   (2026-07-23: is_knowledge ガードを _is_strict_knowledge_query に
        #   差し替え。KNOWLEDGE_QUERY_PATTERNS の緩い名詞+助詞パターン
        #   (資料|情報|データ|... + は/を/が/に) は、「サンプルCSVデータを
        #   作成し」のような正規の生成依頼の一部 (「データを」) にも誤マッチ
        #   して is_knowledge=True にしてしまう (2026-07-22 ライブ検証で発覚。
        #   CSV 集計依頼が meta_cognitive に到達せず deliberative の
        #   executable_query に短絡していた)。一方「ファイルを作って整理する
        #   方法について教えて」のような真正の howto 知識質問は「教えて」
        #   「について」等の明示的疑問形式で is_knowledge=True になっており、
        #   これは META_KEYWORDS ("作って") が同時に立っていても知識質問の
        #   まま扱うべき (Meta-Cognitive へ昇格させない)。_is_strict_knowledge_
        #   query は緩いパターンだけを除外するため、両ケースを正しく判別できる。
        if (
            is_coding_mode(mode)
            and self._has_meta_keywords(query)
            and not self._is_strict_knowledge_query(query)
        ):
            return self._record_classification(
                self._guard_meta_cognitive(), "coding_meta_keywords", query,
            )

        # 2.7. Python 実行で正確に答えられるクエリは例外的に deliberative に
        # 昇格してツール実行を誘導する。
        # 知識質問パターン (KNOWLEDGE_QUERY_PATTERNS) は「ですか」「教えて」等の
        # 末尾形式を要求するため、「明日は何月何日?」のような ? 終わりの疑問文
        # を取りこぼす。executable_query_keywords は明確なツール経路シグナル
        # なので is_knowledge 判定を外して常時評価する。
        if self._contains_executable_query_keywords(query):
            return self._record_classification(
                "deliberative", "executable_query", query,
            )

        # 2.8. 知識質問でも学習済み tool_routing パターンにマッチする場合は
        # deliberative に昇格してツール実行を誘導する
        if is_knowledge and self._contains_learned_tool_patterns(query):
            return self._record_classification(
                "deliberative", "learned_tool_pattern", query,
            )

        # 5. 履歴参照キーワード → deliberative（短いクエリより優先）
        if self._has_history_keywords(query):
            return self._record_classification("deliberative", "history_ref", query)

        # 6. 複雑キーワード → deliberative
        if self._has_complex_keywords(query):
            return self._record_classification(
                "deliberative", "complex_keywords", query,
            )

        # 7. クエリ長の判定（日本語はスペース分割できないため文字数も考慮）
        is_short = self._is_short_query(query, mode)

        # 8. 短いクエリ + 高スコア RAG ヒット → reactive
        rag_threshold = self._get_policy("router", "rag_score_threshold", mode, 0.8)
        if is_short and rag_results and rag_results[0][1] > rag_threshold:
            return self._record_classification("reactive", "short_high_rag", query)

        # 8.5. 知識質問 → deliberative（検索を走らせる）
        #   「Xについて教えて」等の知識質問は LTM / カートリッジ参照が要る。reactive /
        #   reactive_light は検索パイプラインを一切走らせない (chat 側 search_task は
        #   agent_layer != reactive のときのみ起動) ため、短い知識質問が step 9 の
        #   short_query で reactive に落ちると、カートリッジを参照せずベースモデルの
        #   事前知識だけで回答してしまう。知識質問は短くても検索経路 (deliberative)
        #   へ送る。step 8 (short_high_rag) の後に置くことで、RAG ヒット済みの短文
        #   即応答 (本番では rag_results 未注入のため発火しない) は従来挙動を保つ。
        if is_knowledge:
            return self._record_classification(
                "deliberative", "knowledge_query", query,
            )

        # 9. 短いクエリ（単純な質問）→ reactive
        if is_short:
            return self._record_classification("reactive", "short_query", query)

        # 10. RAG 結果がない → deliberative（検索が必要）
        if not rag_results:
            return self._record_classification(
                "deliberative", "no_rag_results", query,
            )

        # 11. ツール操作が必要 → meta_cognitive
        if self._needs_tools(query):
            return self._record_classification(
                self._guard_meta_cognitive(), "tool_patterns", query,
            )

        # 12. デフォルト: reactive
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

        日本語テキストはスペースで分割されないため、文字数も考慮する。
        """
        min_tokens = self._get_policy("router", "short_query_min_tokens", mode, 3)
        max_tokens = self._get_policy("router", "short_query_max_tokens", mode, 10)
        max_chars = self._get_policy("router", "short_query_max_chars", mode, 20)

        tokens = len(query.split())
        # 英語テキスト: 単語数で判定
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

        chat モードのみ対象 (coding モードは step 3 の ``_needs_tools`` が同等の
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

        chat モードのみ対象 (coding モードは step 3/4 の ``_needs_tools`` /
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
        if not _LOCAL_PATH_RE.search(query):
            return False
        return bool(_WRITE_VERB_RE.search(query))

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
        """緩い名詞+助詞パターン (KNOWLEDGE_QUERY_PATTERNS[1]) を除いた、
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
