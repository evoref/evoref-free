"""複雑度分類ルーター: クエリを3層エージェントに振り分ける"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.config import resolve_context_size_for_mode
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
    # を長文生成として拾うため、設計系文書 (仕様書/設計書/要件定義/計画書/
    # 手順書/README/読本) と出力動詞 (出力) を追加。
    re.compile(
        r"(記事|レポート|報告書|文書|論文|マニュアル|ドキュメント|論説|手引き|"
        r"仕様書|設計書|要件定義|計画書|手順書|README|読本)"
        r".*(書|作成|生成|まとめ|出力)",
    ),
    re.compile(r"長(い|文|編)"),
    re.compile(
        r"(ファイル|モジュール|クラス|プロジェクト)"
        r".*(一式|全体|まるごと|フル).*(作成|生成|実装)",
    ),
    re.compile(r"(実装|作成).*全体"),
    re.compile(r"(完全|網羅的|包括的).*(実装|ガイド|解説)"),
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
# how-to / 質問形マーカー。「作成する方法を教えて」のような書込み動詞を含む
# 知識質問を local_write_intent から除外する。``_is_knowledge_query`` は
# 「一覧を…」等のデータ語を広く拾い正当な書込みコマンドまで除外してしまうため、
# ここでは教示・疑問マーカーに限定する。
_HOWTO_QUERY_RE = re.compile(
    r"(?:教えて|おしえて|どうやって|どうすれば|どうやったら"
    r"|とは|って何|何ですか|ですか|ますか|でしょうか|ありますか|[?？])",
    re.IGNORECASE,
)

# 複雑度を示すキーワードパターン（同義語・表記揺れ対応済み）
COMPLEX_KEYWORDS = [
    "比較", "なぜ", "どのように", "違い", "分析", "解析",
    "メリット", "デメリット", "利点", "欠点", "長所", "短所",
    "原因", "理由", "要因", "根本原因",
    "仕組み", "構造", "アーキテクチャ",
    "explain", "compare", "analyze", "analyse", "why", "how",
    "difference", "pros", "cons", "trade-off", "tradeoff",
]

# 履歴検索が必要なキーワード → Deliberative 層
HISTORY_KEYWORDS = [
    "前に", "以前", "先週", "先月", "この間", "前回", "前の会話",
    "さっき", "昨日", "今朝", "先ほど",
    "earlier", "previously", "last time", "yesterday", "before",
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

# 知識質問パターン — ツール呼び出しではなく RAG で処理すべきクエリ
# これらにマッチするクエリはツール/Meta-Cognitive エスカレーションをスキップする
KNOWLEDGE_QUERY_PATTERNS = [
    re.compile(r"(?:教えて|おしえて|とは|って何|ですか|でしょうか|ありますか)", re.IGNORECASE),
    re.compile(r"(?:資料|情報|データ|内容|概要|詳細|特徴|一覧).*(?:は|を|が|に)", re.IGNORECASE),
    re.compile(r"(?:について|に関して|に関する)", re.IGNORECASE),
    re.compile(r"(?:知りたい|確認したい|調べたい|わかる|分かる)", re.IGNORECASE),
    re.compile(r"(?:what is|tell me|explain|describe|how does)\b", re.IGNORECASE),
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

# 挨拶・簡単な定型パターン → Reactive 層で即応答
GREETING_PATTERNS = [
    re.compile(r"^(?:こんにち[はわ]|おはよう|こんばんは|やあ|ども|hello|hi|hey)\s*[!！。.]?\s*$", re.IGNORECASE),
    re.compile(r"^(?:ありがと[うございます]*|thanks|thank you)\s*[!！。.]?\s*$", re.IGNORECASE),
    re.compile(r"^(?:おやすみ|さようなら|bye|goodbye)\s*[!！。.]?\s*$", re.IGNORECASE),
]


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
        context_turns: int = 0,
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
        if self._detect_long_form(query):
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

        # 2.5. Python 実行で正確に答えられるクエリは例外的に deliberative に
        # 昇格してツール実行を誘導する。
        # 知識質問パターン (KNOWLEDGE_QUERY_PATTERNS) は「ですか」「教えて」等の
        # 末尾形式を要求するため、「明日は何月何日?」のような ? 終わりの疑問文
        # を取りこぼす。executable_query_keywords は明確なツール経路シグナル
        # なので is_knowledge 判定を外して常時評価する。
        if self._contains_executable_query_keywords(query):
            return self._record_classification(
                "deliberative", "executable_query", query,
            )

        # 2.6. 知識質問でも学習済み tool_routing パターンにマッチする場合は
        # deliberative に昇格してツール実行を誘導する
        if is_knowledge and self._contains_learned_tool_patterns(query):
            return self._record_classification(
                "deliberative", "learned_tool_pattern", query,
            )

        # 3. コーディングモードでツール操作が必要 → meta_cognitive
        if mode == "coding" and not is_knowledge and self._needs_tools(query):
            return self._record_classification(
                self._guard_meta_cognitive(), "coding_tools", query,
            )

        # 4. Meta-Cognitive キーワード検出（コーディングモード）
        if mode == "coding" and not is_knowledge and self._has_meta_keywords(query):
            return self._record_classification(
                self._guard_meta_cognitive(), "coding_meta_keywords", query,
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
        return any(kw in q_lower for kw in COMPLEX_KEYWORDS)

    def _has_history_keywords(self, query: str) -> bool:
        """履歴参照キーワードを検出"""
        q_lower = query.lower()
        return any(kw in q_lower for kw in HISTORY_KEYWORDS)

    def _has_meta_keywords(self, query: str) -> bool:
        """Meta-Cognitive 層へのエスカレーションキーワードを検出"""
        q_lower = query.lower()
        return any(kw in q_lower for kw in META_KEYWORDS)

    def _is_url_write_intent(self, query: str, mode: str = "chat") -> bool:
        """URL からデータを取得してファイルに書き出す意図を検出する。

        chat モードのみ対象 (coding モードは step 3 の ``_needs_tools`` が同等の
        meta_cognitive 振り分けを行う)。URL とファイル書込み/出力意図の双方が
        揃った場合のみ True を返す。
        """
        if mode != "chat":
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
        if mode != "chat":
            return False
        if _URL_HINT_RE.search(query):
            return False
        if _HOWTO_QUERY_RE.search(query):
            return False
        if not _LOCAL_PATH_RE.search(query):
            return False
        return bool(_WRITE_VERB_RE.search(query))

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
        if any(p.search(query) for p in LONG_FORM_PATTERNS):
            return True
        return self._contains_learned_long_form_patterns(query)

    def _contains_learned_long_form_patterns(self, query: str) -> bool:
        """学習済み long_form パターンにマッチするか判定"""
        if self._learned_patterns is None:
            return False
        matches = self._learned_patterns.match(query, category="long_form")
        if not matches:
            return False
        return matches[0][1] >= self._long_form_threshold

    def _is_knowledge_query(self, query: str) -> bool:
        """知識質問パターンを検出（RAG で処理すべきクエリ）"""
        return any(p.search(query) for p in KNOWLEDGE_QUERY_PATTERNS)

    def _contains_executable_query_keywords(self, query: str) -> bool:
        """Python 実行で正確に答えられるクエリのキーワードを検出"""
        return any(p.search(query) for p in _EXECUTABLE_QUERY_PATTERNS)

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
        return any(p.search(query) for p in TOOL_PATTERNS)

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
    ctx_size = resolve_context_size_for_mode(config, mode)
    history_budget = config.get("memory", {}).get("working_max_tokens", 2048)
    loop_budget = ctx_size - 512 - 400 - history_budget

    cfg_default = config.get("agent", {}).get("meta_cognitive_min_budget", 512)
    min_budget = get_policy_value(
        policy, "agent", "meta_cognitive_min_budget", cfg_default, mode=mode,
    )

    return loop_budget >= min_budget
