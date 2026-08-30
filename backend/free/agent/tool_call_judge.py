"""補助タスクによるツール呼び出し判定（Free/Pro 共通）

ユーザークエリと利用可能なツール一覧を補助タスクに提示し、
ツール呼び出しの要否・ツール名・引数を判定する。
補助タスク未接続時はルールベースにフォールバックする。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from backend.config import get_project_root
from backend.free.agent.router import (
    asks_directory_listing,
    is_environment_fact_query,
)
from backend.free.core.intent_vocab import (
    asks_to_restate_prior_report,
    ANAPHORIC_OPERAND_RE,
    NUMBER_LITERAL_RE,
    is_plain_statement,
    looks_like_numeric_question,
)
from backend.free.core.locale_patterns import is_en_locale, select_locale_variant
from backend.free.core.session_mode import is_create_mode
from backend.free.agent.safety_patterns import (
    extract_command_literal,
    strip_command_literals,
)
from backend.free.agent.tools_registry import ToolDefinition, ToolsRegistry
from backend.free.agent.grammar_tool_classifier import (
    CLASSIFY_MAX_TOKENS,
    EXPRESSION_SCHEMA,
    EXPRESSION_SYSTEM,
    EXPRESSION_SYSTEM_EN,
    build_classifier_schema,
    build_tool_menu,
    parse_classifier_response,
    parse_expression_response,
)
from backend.free.llm.json_extract import extract_json_object
from backend.log_config import get_logger

# --- 責務別モジュール ---------------------------------------------------------
# 判定に使う正規表現・純粋関数は責務ごとに分割してある。本モジュールは判定フロー
# (``ToolCallJudge.judge`` の層構成) だけを持ち、以下は **再エクスポート** として
# 取り込む。既存の呼出元・テストが ``tool_call_judge.<名前>`` を直接参照しており、
# ``mock.patch("...tool_call_judge.Path.exists")`` のようなパッチ対象にもなって
# いるため、名前の見え方は分割前と一致させる。
from backend.free.agent.tool_judge_types import (
    ToolJudgement,
)
from backend.free.agent.tool_judge_dialogue import (
    _CALCULATE_CONTEXT_TURNS,
    _dialogue_text,
    _SYNTHESIS_CONTEXT_TURNS,
    _recent_dialogue_messages,
    _recent_dialogue_text,
    query_needs_dialogue,
)
from backend.free.agent.tool_judge_signals import (
    _ASSISTANT_PREFERENCE_PATTERNS,
    _CODE_IDENTIFIER_RE,
    _CODE_SEARCH_PATTERNS,
    _CODE_USAGE_LOCATION_RE,
    _CODE_USAGE_STOPWORDS,
    _CODE_USAGE_VERB_RE,
    _DELETE_FS_TARGET_RE,
    _DELETE_INTENT_RE,
    _EXPLICIT_EXEC_VERB_RE,
    _FIRST_PERSON_REFERENCE_RE,
    _HARDWARE_MEMORY_QUERY_RE,
    _RUNTIME_INFO_QUERY_RE,
    _IMMEDIATE_CHILDREN_RE,
    _INFER_TOOL_EXEC_QUERY_RE,
    _INFER_TOOL_EXEC_QUERY_RE_EN,
    _KNOWLEDGE_PATTERNS,
    _KNOWLEDGE_PATTERNS_EN,
    _LOCAL_FILE_REFERENCE_RE,
    _PATH_OR_URL_SIGNAL_RE,
    _READ_PATH_TOOLS,
    _RECURSIVE_LISTING_RE,
    _SELF_ACTION_PATTERNS,
    _SELF_SESSION_REFERENCE_PATTERNS,
    _SELF_SESSION_REFERENCE_PATTERNS_EN,
    _SESSION_REFLECTIVE_VOCAB_BROAD_EN,
    _SESSION_REFLECTIVE_VOCAB_BROAD_JA,
    _SESSION_REFLECTIVE_VOCAB_LEADING_EN,
    _SESSION_TOPIC_BREAK_LEAD_RE_EN,
    _TOOL_PATTERNS,
    _TOOL_PATTERNS_EN,
    _WEB_REFERENCE_RE,
    _code_usage_location_pattern,
    _is_code_usage_location_query,
    _query_has_tool_signal,
    _query_targets_local_file_only,
)
from backend.free.agent.tool_judge_grounding import (
    _DURATION_H_HALF_RE,
    _DURATION_H_RE,
    _DURATION_HM_RE,
    _DURATION_M_RE,
    _DURATION_MS_RE,
    _FULLWIDTH_DIGITS,
    _GROUPED_NUMBER_RE,
    _INTERVAL_M_RE,
    _NUMBER_LITERAL_RE,
    _NUMERIC_LITERAL_RE,
    _PERCENT_LITERAL_RE,
    _UNIT_SYSTEM_CONSTANTS,
    _WEEKDAY_ORDER,
    _WEEKDAY_RANGE_RE,
    _duration_derived_numbers,
    _known_numbers,
    _numeric_literals,
    _synthesized_expression_grounded,
    _ungrounded_numbers,
)
from backend.free.agent.tool_judge_commands import (
    _DATE_ARITHMETIC_RE,
    _DATETIME_NOW_COMMAND,
    _DATETIME_QUERY_RE,
    _DRIVE_LETTER_RE,
    _EXECUTABLE_QUERY_COMMANDS,
    _OFFSET_UNITS,
    _READONLY_INSPECT_COMMANDS,
    _REL_PREFIX,
    _REL_SUFFIX,
    _RELATIVE_OFFSET_RE,
    _build_datetime_command,
    _build_spec_command,
    _command_is_readonly_inspection,
    _infer_executable_command,
    _readonly_command_rejected,
    recalled_command_fits_query,
)
from backend.free.agent.tool_judge_args import (
    _ARITH_BARE_TAIL_RE,
    _ARITH_DATE_LIKE_RE,
    _ARITH_NORMALIZE,
    _ARITH_REQUEST_CUE_RE,
    _ARITH_RUN_RE,
    _ARITH_SAFE_NODES,
    _DIR_PATH_RE,
    _FILE_CONTENT_REQUEST_RE,
    _FILE_EXISTENCE_RE,
    _HEAD_LINES_RE,
    _NAMED_DIRECTORY_RE,
    _PROJECT_ROOT_REFERENCE_RE,
    _QUOTE_PAIRS,
    _URL_IN_QUERY_RE,
    _ZENKAKU_DIGITS,
    _coerce_positive_int,
    _extract_arithmetic_expression,
    _extract_file_path,
    _extract_head_line_count,
    _extract_quoted_filename,
    _extract_search_pattern,
    _is_numeric_expression,
    _normalize_path_separators,
    _normalize_path_text,
    _trim_nonexistent_path_tail,
    asks_file_existence_only,
    resolve_listing_directory,
)
from backend.free.agent.tool_judge_history import (
    _ANAPHORIC_REFERENCE_RE,
    _HISTORY_SEARCH_DEFAULT_LIMIT,
    _ORDER_QUERY_CONTENT_RE,
    _ORDER_QUERY_CONTENT_RE_EN,
    _ORDER_QUERY_MIN_TERM_LEN,
    _ORDER_QUERY_SCAFFOLD_RE,
    _ORDER_QUERY_SCAFFOLD_RE_EN,
    _ORDER_QUERY_STOPWORD_RUNS,
    _ORDER_QUERY_STOPWORD_RUNS_EN,
    _ORDER_QUERY_STOPWORDS_BY_LEN,
    _ORDERED_HISTORY_QUERY_RE,
    _RETROSPECTIVE_QUESTION_RE,
    _has_history_recall_keywords,
    _only_proximal_recall_keywords,
    _reduce_ordered_history_query,
    _strip_stopword_affixes,
    asks_about_past_conversation,
    asks_about_prior_conversation_entity,
)
from backend.free.agent import tool_judge_guards as guards
from backend.free.agent.tool_judge_guards import (
    _COMMAND_TOOL_NAMES,
    _MODE_CAPABILITY_SIBLINGS,
    _STATE_CHANGING_TOOL_NAMES,
    _TEXT_OPERAND_TOOLS,
    GuardContext,
    apply_guards,
)
from backend.free.agent.tool_judge_referential import (
    _FILE_CONTENT_DISPLAY_RE,
    _FILE_METRICS_RE,
    _FILE_NOUN_RE,
    _PATH_SEPARATOR_RE,
    _REFERENTIAL_TARGET_RE,
    _REWRITE_VERB_RE,
    _referential_read_judgement,
    _referential_rewrite_judgement,
    _resolve_referenced_path,
)

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.memory.views.mem import MemFactView
    from backend.free.rag.cartridge_manager import CartridgeManager
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("agent.tool_call_judge")

# executable command リコールの候補プールがこの件数未満のとき、類似度閾値を
# ``_RECALL_SMALL_POOL_MARGIN`` だけ嵩上げする。学習初期は top-K も success_avg も
# 選別として機能せず、類似度ゲート 1 本で決まってしまうため。
_RECALL_SMALL_POOL_SIZE = 3
_RECALL_SMALL_POOL_MARGIN = 0.1


#: 「これは文であってキーワードではない」ことを示す印。文末記号と依頼形は
#: 検索語には現れないので、分類器が生の質問文をそのまま渡したことの目印になる。
_SENTENCE_SHAPE_RE = re.compile(r"[。．？?！!]|ください|下さい|お願い")


def _looks_like_sentence(candidate: str, raw_query: str) -> bool:
    """``candidate`` がキーワードではなく **文** か (純粋関数)。"""
    c = (candidate or "").strip()
    if not c:
        return False
    if c == (raw_query or "").strip():
        return True
    return bool(_SENTENCE_SHAPE_RE.search(c))


def _executable_tool_for_mode(tools_registry: ToolsRegistry, mode: str) -> str:
    """現在の ``mode`` で使える executable コマンドツール名を解決する。

    create では従来の ``run_command``、chat では読み取り専用の
    ``run_command_readonly`` (modes=["chat"], hidden) を返す。どちらも使えない
    場合は ``""``。

    2026-07-18 の mode ゲート導入で chat の executable query (時刻 / OS /
    スペック等) が合成成功後に必ず no_tool へ格下げされる回帰が起きたため、
    executable 経路 (early-return / _infer_tool 実行可能クエリ分岐 / 層5
    fallback / SemMem recall) はこのヘルパでツール名を解決する。mode ゲート
    (_validate_tool_availability / deliberative 実行前ゲート) には例外を作らず、
    「chat で使える実行ツールは readonly 検証付き func しか登録されていない」
    を登録構造で保証する (docs/f_03_agent_engine.md §3.1)。
    """
    if tools_registry.is_available("run_command", mode):
        return "run_command"
    if tools_registry.is_available("run_command_readonly", mode):
        return "run_command_readonly"
    return ""


class ToolCallJudge:
    """補助タスクによるツール呼び出し判定

    補助タスクにクエリと利用可能ツール一覧を渡し、
    適切なツールの選択を判定させる。
    判定は決定論層 (ルール / カートリッジ / 学習済みパターン / リコール) を
    順に試し、決まらなければベースモデルの文法制約 JSON 分類へ落とす。
    """

    def __init__(
        self,
        config: dict | None = None,
        cartridge_manager: CartridgeManager | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        debug_logger: "DebugLogger | None" = None,
        mem_view: "MemFactView | None" = None,
        embedder: "EmbeddingBackend | None" = None,
        profile_id: str = "default",
        llm_client=None,
    ):
        """
        Args:
            config: config.yaml 全体の dict
            cartridge_manager: CartridgeManager インスタンス（None でカートリッジ hints 無効）
            learned_patterns: LearnedPatternStore インスタンス（None で学習済みパターン無効）
            mem_view: MemFactView インスタンス（None で URL リコール無効）。
                URL なしの fetch 意図クエリで過去 ``mem.world.url.*`` を
                引き当てるために使う。
            embedder: EmbeddingBackend インスタンス（None で URL リコール無効）。
                ユーザクエリの embedding を生成して類似 URL fact を引く。
            profile_id: URL fact の profile_id フィルタ。引き当て時に同一
                profile の fact のみ採用する。
                ツール呼出判定の多段フォールバック (decision_point=
                ``tool_call_decision``、chosen=``rule``/``cartridge``/
                ``learned``/``recall``/``no_tool``) を ``decision.jsonl`` に
                記録する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        self._llm_client = llm_client
        self._config = config or {}
        self._cartridge_manager = cartridge_manager
        self._learned_patterns = learned_patterns
        self._debug_logger = debug_logger
        self._mem_view = mem_view
        self._embedder = embedder
        self._profile_id = profile_id
        # ツールパターン学習の閾値（通常パターンより高め）
        learning_cfg = self._config.get("learning", {})
        self._tool_routing_threshold: float = learning_cfg.get(
            "tool_pattern_match_threshold", 0.4,
        )
        agent_cfg = self._config.get("agent", {})
        # ベースモデルの文法制約ツール分類 (docs/c_14 §1.3)。決定論層が
        # すべて外れたときの最終層。``response_format`` を受け付けない
        # llama-server build では初回失敗時に落として以後試さない
        # (毎ターン 4xx を踏まないため)。
        self._tool_classifier_enabled: bool = bool(
            agent_cfg.get("tool_classifier_enabled", True),
        )
        self._tool_classifier_supported: bool = True
        self._tool_classifier_max_tokens: int = int(
            agent_cfg.get("tool_classifier_max_tokens", CLASSIFY_MAX_TOKENS),
        )
        self._tool_classifier_timeout_sec: float = float(
            agent_cfg.get("tool_classifier_timeout_sec", 60.0),
        )
        # 分類器を撃つかどうかの門。正規表現 ``_query_has_tool_signal`` は
        # 実クエリ 137 件のベンチで recall 66.2% しかなく、ツールが要る
        # クエリの 3 分の 1 を落としていた。埋め込み近傍なら 98.5% (k=5)。
        # 埋め込み未準備 / 無効時は従来の正規表現へ縮退する。
        self._tool_gate: "ToolGateKNN | None" = None
        if embedder is not None and agent_cfg.get("tool_gate_knn_enabled", True):
            from backend.free.agent.tool_gate_knn import DEFAULT_K, ToolGateKNN

            self._tool_gate = ToolGateKNN(
                embedder, k=int(agent_cfg.get("tool_gate_knn_k", DEFAULT_K)),
            )
        # 直近の層0.5 リコールの診断値 (sim / min_sim / success_avg / 候補数)。
        # _log_tool_decision が decision.jsonl の context に載せる。
        self._last_recall_diag: dict[str, Any] = {}
        # 直近の judge() で「実測しようとしたが実行できなかった」か。
        # readonly 検証違反 (PowerShell 等) / mode 非対応での降格で立つ。
        # deliberative がこれを見て、測っていない値の捏造を禁じる注記を付ける。
        self._measurement_blocked: bool = False
        # 直近の judge() で「状態を変える操作を選んだが実行できなかった」か。
        # measurement_blocked (値を測れなかった) とは別物で、こちらは
        # 「やっていないことをやったと言わせない」ためのフラグ。
        self._action_blocked: bool = False
        # 直近の judge() のユーザークエリ。measurement_blocked の適用可否判定用。
        self._last_query: str = ""
        # 直近の judge() の会話履歴。コマンド合成の指示語解決に使う
        # conversation を持たないため、単一の入口である judge() で保持する)。
        self._last_conversation: list[dict] = []

    @property
    def measurement_blocked(self) -> bool:
        """直近の判定で実行可能コマンドが「棄却されて」ツールが立たなかったか。

        「そもそもツールが不要だった」(知識質問等) とは区別する。True のときは
        システムが実測を試みて失敗しているので、呼出側は測っていない値を
        断定させないためのガードを掛ける。
        """
        return self._measurement_blocked

    @property
    def action_blocked(self) -> bool:
        """直近の判定で「状態を変える操作」が選ばれたのに実行できなかったか。

        chat には書込みツールが無く (``write_file`` は create 限定)、
        ``run_command_readonly`` は書込みコマンドを正しく拒否する。その結果
        ツールが 1 つも立たないまま base に丸投げされると、**やっていない操作を
        やったと報告する** (2026-08-08 ライブ監査 ターン6)。呼出側はこれを見て
        完了報告を禁じる注記を付ける。
        """
        return self._action_blocked

    def _user_requested_measurement(self) -> bool:
        """直近クエリが実測 (環境事実の取得 / コマンド実行) を求めているか。

        層5 のコマンド合成は環境事実を尋ねていないクエリにも投機的に走るため、
        合成の棄却だけで「実測できなかった」と記録すると、測定を求めていない
        質問にまで断り書きが混入する (実インシデント 2026-08-01 ライブ監査:
        「あなたは何ができますか？」への回答に「PC の空き容量や具体的なスペックは
        測定ツールが利用できないため取得できていません」が混ざった)。

        ユーザー意図を確立していない投機的な経路だけがこれを見る。意図が
        呼出時点で確定している経路 (明示コマンド + 実行動詞 / 判定層が
        コマンドツールを選択済み) は無条件に記録してよい。

        判定材料が無い場合 (クエリ未設定の直接呼出) は従来どおり True。
        """
        if not self._last_query:
            return True
        return is_environment_fact_query(self._last_query) or bool(
            extract_command_literal(self._last_query),
        )

    def _reject_readonly(self, exec_tool: str, command: str) -> bool:
        """``_readonly_command_rejected`` に「実測が阻まれた」記録を足したもの。

        コマンドは層5 が投機的に合成したものでもあり得るため、ユーザーが実測を
        求めていないクエリでは記録しない (``_user_requested_measurement`` 参照)。
        """
        rejected = _readonly_command_rejected(exec_tool, command)
        if rejected and self._user_requested_measurement():
            self._measurement_blocked = True
        return rejected

    def _mark_blocked_if_unexecutable_command(
        self, query: str, tools_registry: ToolsRegistry, mode: str,
    ) -> None:
        """明示コマンドを撃てないまま no_tool に落ちる場合、実測失敗を記録する。

        ユーザーがバッククォートでコマンドを書いた依頼は「実行して結果を見せて」
        という明確な実測要求。chat モードの ``run_command_readonly`` は python の
        allow-list しか通さないため ``dir`` / ``git`` 等は実行できない。撃てないまま
        no_tool で base に丸投げすると、実行していないコマンドの出力を捏造する
        (実インシデント 2026-07-29 ライブ監査: ``dir E:\\tmp\\no_such_dir_zzz`` の
        実行依頼に対し「dir: ...: No such file or directory」という、Windows の
        ``dir`` が決して返さない Unix 形式のエラーを実行結果として提示した)。

        ``measurement_blocked`` が立つと ``deliberative`` が「測っていない値を
        断定しない」注記を base の文脈へ足す。
        """
        command = extract_command_literal(query)
        if not command:
            return
        if not _EXPLICIT_EXEC_VERB_RE.search(query):
            return
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if exec_tool and not _readonly_command_rejected(exec_tool, command):
            return
        logger.info(
            "Measurement blocked: explicit command %r cannot be executed in "
            "mode=%s (exec_tool=%s)", command[:80], mode, exec_tool or "none",
        )
        self._measurement_blocked = True

    def _mark_blocked_if_unsupported_mutation(self, query: str) -> None:
        """ファイル削除依頼のまま no_tool に落ちる場合、未実行を記録する。

        削除ツールはどのモードにも存在しない (``_DELETE_INTENT_RE`` のコメント
        参照)。``action_blocked`` を立てて ``deliberative`` に「何も実行して
        いない」注記を足させ、完了の捏造を塞ぐ。
        """
        if not _DELETE_INTENT_RE.search(query):
            return
        if not _DELETE_FS_TARGET_RE.search(query):
            return
        logger.info(
            "Action blocked: file deletion requested but no tool can delete: %s",
            query[:80],
        )
        self._action_blocked = True

    @property
    def enabled(self) -> bool:
        """ツール判定が有効かどうか (``agent.tool_judge_enabled``、既定 True)。

        False にすると最終層のベースモデル分類を撃たなくなる。決定論層は
        本フラグに関係なく常に動く。
        """
        return self._config.get("agent", {}).get("tool_judge_enabled", True)

    async def judge(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
        conversation: list[dict] | None = None,
        session_id: str = "",
        *,
        allow_classifier: bool = True,
    ) -> ToolJudgement:
        """ツール呼び出しの要否を判定し、**ターン固有の値を結果に載せて** 返す。

        ``action_blocked`` / ``measurement_blocked`` は判定の途中でインスタンス
        属性へ立つが、本インスタンスはプロセス唯一の共有オブジェクトで、
        呼出側は judge() 完了後の別タイミング (reactive-light ゲート / 経験記録 /
        deliberative の注記) で読む。チャットが 2 本重なると他方の judge() が
        先にリセットするため、ここで **await を挟まずに** 結果へ写し取る。
        以後の読み手は ``ToolJudgement`` 側を見ること。
        """
        result = await self._judge_inner(
            query, tools_registry, mode, conversation, session_id,
            allow_classifier=allow_classifier,
        )
        # ここに await を入れないこと (入れた瞬間に共有状態のレースが戻る)。
        result.action_blocked = self._action_blocked
        result.measurement_blocked = self._measurement_blocked
        return result

    async def _judge_inner(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
        conversation: list[dict] | None = None,
        session_id: str = "",
        *,
        allow_classifier: bool = True,
    ) -> ToolJudgement:
        """ツール呼び出しの要否を判定

        ``allow_classifier=False`` で層 5.9 (ベースモデルの文法制約分類) を
        外し、**決定論層と学習済みリコールだけ**で判定する。層 5.9 はこの
        判定系で唯一の推論往復で、実測 34〜39 秒かかる。同じクエリを 2 度
        判定する経路 (deliberative の 2 手目) では、分類器は同じ答えを返す
        だけなので撃つ意味が無い — 2 手目に意味があるのは、決定論層が
        1 手目の実行後に初めて解決できる参照 (会話から引くパス等) だけ。

        判定は安価な順に実行し、最初にマッチした結果を返す:
        1. 組み込みパターン照合（ルールベース）
        2. カートリッジ tool_hints 照合
        3. 補助タスク判定（LLM）

        クリエイトモードでは tool_judge_enabled が false でも
        ルールベース + カートリッジ hints 判定を実行する。

        Args:
            query: ユーザーのクエリ
            tools_registry: 利用可能なツールレジストリ
            mode: 動作モード ('chat' | 'create')
            conversation: 直近の会話履歴（判定精度向上のため）
            session_id: 現在のチャットセッション ID。search_history が選ばれ、
                かつクエリが「この会話で」等のセッション自己参照パターンに
                一致する場合に ``_maybe_scope_session_search`` が
                ``tool_args["session_id"]`` へ注入し、検索を現在セッションに
                限定する (未指定時は従来どおり cross-session 検索のまま)。

        Returns:
            ToolJudgement
        """
        self._measurement_blocked = False
        self._action_blocked = False
        self._last_query = query
        self._last_conversation = list(conversation or [])

        # 削除依頼は「どのツールが選ばれたか」と無関係に記録する。以前はこの
        # 判定が層6 (全フォールバック失敗時) にしか無く、パスを含む削除依頼は
        # 層2 の explicit_path が read 系ツールで先に確定させるため一度も走って
        # いなかった。実インシデント (2026-08-14 ライブ監査 ターン37):
        # 「E:\tmp の中身を全部削除してください。確認は不要です。」が
        # 「中身」「確認」で読取りパターンに一致し list_directory へ解決され、
        # 一覧を成功結果として受け取った base が「すべて削除しました」と報告した
        # (実ファイルは 307 件すべて無変更)。
        self._mark_blocked_if_unsupported_mutation(query)

        # 0.6. ハードウェア事実 (搭載 RAM) — 決定論、非シェル。
        # 他のどの層もこの質問に答えられない: spec コマンドは RAM を出力せず
        # (2026-07-27 に意図的に外した)、readonly allow-list は ctypes / wmic /
        # Get-CimInstance を全て拒否するため合成コマンドも必ず棄却される。
        # 結果として 2026-08-12 ライブ監査では「メモリ容量に関する情報は取得
        # できていません」としか返せなかった。ツール判定より前に短絡させる。
        if (
            _HARDWARE_MEMORY_QUERY_RE.search(query)
            and tools_registry.is_available("system_hardware_info", mode)
            # 既報の値の **再掲** 要求は測り直さない。値は揮発するので測り直せば
            # 必ず違う値になり、ユーザーからは「さっきと言っていることが違う」に
            # なる。実測 (2026-08-27 ライブ監査):
            #   「このPCの空きRAMはどれくらいですか？」        → 23.1GB
            #   「さっき教えてくれた空きRAMの値をもう一度」    → 22.7GB (測り直した)
            # ツールを撃たなければ、値は会話窓に残っているのでモデルが読み上げる。
            # 「もう一度測って」のような明示的な再計測要求は
            # ``asks_to_restate_prior_report`` 側が False を返すので通る。
            and not asks_to_restate_prior_report(query)
        ):
            hw_result = self._finalize(
                ToolJudgement(
                    tool_needed=True,
                    tool_name="system_hardware_info",
                    tool_args={},
                    source="rule",
                ),
                tools_registry, mode, query=query,
            )
            self._log_tool_decision(hw_result, "hardware_facts_query")
            return hw_result

        # 0.6b. evoref 自身の実行構成 — 決定論、非シェル。
        # ハードウェアと同じ理由でどの層も答えられない: readonly allow-list は
        # config も /props も読めず、モデル名・n_ctx・ポートは backend 内に
        # しか無い。実測 (2026-08-22 ライブ監査): 「今動いているモデルの名前を
        # 教えてください。」→「私は「Alice」という名前で対応しています」
        # (インスタンス名)、「埋め込みモデルは？」→「開示したりする仕様では
        # ありません」(存在しない方針の捏造)、ポート / n_ctx →
        # 「確認できていません」。
        # 自己申告の平叙文 (「llama-server は 8080 で動かしています。」) は
        # 実行要求ではないので落とす (_infer_executable_command と同じ扱い)。
        if (
            _RUNTIME_INFO_QUERY_RE.search(query)
            and not is_plain_statement(query)
            and tools_registry.is_available("evoref_runtime_info", mode)
        ):
            rt_result = self._finalize(
                ToolJudgement(
                    tool_needed=True,
                    tool_name="evoref_runtime_info",
                    tool_args={},
                    source="rule",
                ),
                tools_registry, mode, query=query,
            )
            self._log_tool_decision(rt_result, "runtime_info_query")
            return rt_result

        # 決定論ショートカット: 明示された算術式 / URL / パス / 実行可能
        # コマンドは「強い意図表明」なのでモデル判断を仰がずここで確定させる。
        # 決まらなければ以降の決定論層 (0.9 / 1 / 2 / 3 / 5.5) と、最後に
        # ベースモデルの文法制約分類 (5.9) へ落とす。
        # 式が書かれていてもツールが撃てないと base の暗算に落ちて誤答する
        # (実インシデント 2026-08-08 ライブ検証: 「1234 * 5678 はいくつ？」に
        # 7,006,552 と回答。正解は 7,006,652)。
        # ``_extract_arithmetic_expression`` は純粋関数で LLM を使わない。
        expression = _extract_arithmetic_expression(query)
        if expression and tools_registry.has("calculate"):
            logger.debug("Arithmetic expression detected: %s", expression)
            result = self._finalize(
                ToolJudgement(
                    tool_needed=True,
                    tool_name="calculate",
                    tool_args={"expression": expression},
                    source="rule",
                ),
                tools_registry, mode, query=query,
            )
            if result.tool_needed:
                self._log_tool_decision(result, "arithmetic_expression")
                return result

        # クエリに URL が明示的に含まれる場合は tool_judge_enabled に
        # 関係なく fetch_url を返す。ユーザが URL を書く = 「これを読んで」
        # の強い意図表明であり、LLM 判断を仰がず決定論的に拾う
        url_match = _URL_IN_QUERY_RE.search(query)
        if url_match and tools_registry.has("fetch_url"):
            logger.debug("Explicit URL detected: %s", query[:50])
            result = ToolJudgement(
                tool_needed=True,
                tool_name="fetch_url",
                tool_args={"url": url_match.group(1)},
                source="rule",
            )
            result = self._finalize(
                result, tools_registry, mode, query=query,
            )
            self._log_tool_decision(result, "explicit_url")
            return result

        # 明示パスも URL と同じ扱いにする。ユーザーがパスを書く =
        # 「これを見て」の強い意図表明で、LLM 判断を仰ぐ理由が無い。
        # ここが無いと chat では file 系が決定論で解決されず、同じ依頼が
        # ツール未発火に落ちて「存在しない」と誤答していた (実インシデント
        # 2026-08-04 ライブ監査: 「E:/tmp/a.txt の中身を見せて、あわせて
        # 文字数も教えてください。」でツールが 1 つも走らなかった)。
        # ``_infer_tool`` は読み書きの動詞が無ければ空を返すので、パスに
        # 言及しただけの文は従来どおり後続へ落ちる。
        if _PATH_OR_URL_SIGNAL_RE.search(query):
            path_tool, path_args = self._infer_tool(query, tools_registry, mode)
            if path_tool:
                logger.debug(
                    "Explicit path resolved to %s: %s", path_tool, query[:50],
                )
                result = self._finalize(
                    ToolJudgement(
                        tool_needed=True,
                        tool_name=path_tool,
                        tool_args=path_args,
                        source="rule",
                    ),
                    tools_registry, mode, query=query,
                )
                if result.tool_needed:
                    self._log_tool_decision(result, "explicit_path")
                    return result

        # 実行可能コマンドを解決する (ルール表 + 学習済みリコール)。
        # ツール名は mode から解決する (chat は run_command_readonly)。
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if exec_tool:
            command = await self._resolve_executable_command(
                query, readonly=exec_tool == "run_command_readonly",
            )
            if command and not self._reject_readonly(exec_tool, command):
                logger.debug("Executable query detected: %s", query[:50])
                result = self._finalize(
                    ToolJudgement(
                        tool_needed=True,
                        tool_name=exec_tool,
                        tool_args={"command": command},
                        source="rule",
                    ),
                    tools_registry, mode, query=query,
                )
                # 層1 と同じ理由で降格時は後続層へ落とす。ここだけ
                # ``_finalize`` の戻りを無条件に返しており、ガードが no_tool へ
                # 降格させると層0.9〜5.9 が丸ごとスキップされていた。
                # 併せて ``_log_tool_decision`` を追加する: judge() の全 exit の
                # うちこの層だけが decision.jsonl に何も残さず、ツールを実際に
                # 実行したターンが記録から消えていた (2026-08-18 ライブ監査:
                # 8 ターン中 2 ターンが run_command_readonly を実行したのに
                # tool_call_decision エントリ無し)。
                if result.tool_needed:
                    self._log_tool_decision(result, "executable_command_rule")
                    return result


        # 0.9. 「同じファイルに保存し直して」型: パスは直前ターンにしか無い。
        # ルール層はパス必須、aux 層は read_file を選びがちで、書込みが
        # 一度も走らないまま「直した内容」だけを返してしまう (実測 2026-07-27:
        # 「体重を4.5kgに直して、同じファイルに保存し直して」→ read_file のみ
        # 実行され、ファイルは旧内容のまま「保存し直した」体で回答された)。
        # 会話から直近のパスを引いて write_file に確定させる。
        rewrite = _referential_rewrite_judgement(query, conversation, tools_registry)
        if rewrite is not None:
            rewrite = self._finalize(rewrite, tools_registry, mode, query=query)
            if rewrite.tool_needed:
                self._log_tool_decision(rewrite, "referential_rewrite")
                return rewrite

        # 0.95. 「そのファイルの全文を見せて」型: 0.9 の読取版。read_file が
        # 撃たれないと、記憶から再構成した内容を「ファイルの中身」として提示
        # する (実測 2026-08-09: 追記直後の「全文をそのまま見せて」で 3 行とも
        # 実ファイルと不一致。明示的に「read_file で読み直して」と言うと正しく
        # 読み「先ほどの内容は記憶に基づくものでした」と自己訂正した = ゲートが
        # 開かないだけだった)。
        ref_read = _referential_read_judgement(query, conversation, tools_registry)
        if ref_read is not None:
            ref_read = self._finalize(ref_read, tools_registry, mode, query=query)
            if ref_read.tool_needed:
                self._log_tool_decision(ref_read, "referential_read")
                return ref_read

        # 1. 組み込みパターン照合（ルールベース）
        # ツール名まで確定したときだけ層1で打ち切る。``tool_needed=True`` かつ
        # ``tool_name=""`` (汎用ツール指示) で即 return すると、実行できる
        # ツールが 1 つも無いまま deliberative に落ちる。LLM は「ツールを使った
        # 建前」で文脈だけから答えるため、ツールで確かめるべき事実を捏造する
        # (2026-07-26 ライブ検証: 「保存したファイルを読み込んで、中身をそのまま
        # 見せてください。」— パスは直前ターンにあり本文には無い — が
        # tool_name="" で確定し read_file が発火せず、実ファイルと全く異なる
        # 内容を「ファイルの中身」として提示した。同じ依頼をパス明示で出すと
        # read_file が正しく発火し実内容を返す)。
        # ツール名が空のときは後続層 (カートリッジ / 学習済み / aux) に
        # 具体化を委ねる。aux 層は会話履歴を見るため、本文に無いパスを
        # 直前ターンから補える。
        result = self._judge_with_rules(query, tools_registry, mode)
        if result.tool_needed and result.tool_name:
            await self._maybe_recall_url(result, query, mode=mode)
            self._maybe_scope_session_search(result, query, session_id)
            result = self._finalize(result, tools_registry, mode, query=query)
            if result.tool_needed:
                self._log_tool_decision(result, "rule_pattern_matched")
                return result
            # 降格 (aux の not-executable 判定 / 引数欠落 / mode 不可) は
            # 「このツールは使えない」であって「ツールは不要」ではない。ここで
            # return すると層2〜5.2 が丸ごと死に、最も救済が要るクエリだけが
            # 素の base 暗算に落ちる (2026-07-28 ライブ検証: 「その距離を時速
            # 12キロで走ると何時間何分かかりますか。」が run_command_readonly
            # として rule 一致 → aux が not-executable と判定 → 即 no_tool
            # となり、層4 aux なら calculate を選べたのに base の暗算で
            # 「約3時間30分」と答えたうえ本文末尾で「正確には3時間31分」と
            # 自己矛盾した)。後続層へ落とす。
            logger.debug(
                "Rule layer match downgraded to no_tool; falling through to "
                "later layers: %s", query[:50],
            )

        # 2. カートリッジ tool_hints 照合
        result = self._judge_with_cartridge_hints(query, tools_registry)
        if result.tool_needed:
            self._maybe_scope_session_search(result, query, session_id)
            # ``_judge_with_cartridge_hints`` は常に ``tool_args={}`` を返す
            # (カートリッジはツール名しか宣言しない) ため、引数欠落ガードは
            # 特にこの層で効く。tool_hints はカートリッジのメタデータ由来で
            # ユーザーが書けるため、この経路は実データで到達可能。
            result = self._finalize(result, tools_registry, mode, query=query)
            self._log_tool_decision(result, "cartridge_hint_matched")
            return result

        # 3. 学習済みパターン照合
        result = self._judge_with_learned_patterns(query, tools_registry, mode)
        if result.tool_needed:
            await self._maybe_recall_url(result, query, mode=mode)
            self._maybe_scope_session_search(result, query, session_id)
            result = self._finalize(result, tools_registry, mode, query=query)
            if result.tool_needed:
                self._log_tool_decision(result, "learned_pattern_matched")
                return result
            # 層1 と同じ理由で降格時は後続層へ落とす。
            logger.debug(
                "Learned layer match downgraded to no_tool; falling through to "
                "later layers: %s", query[:50],
            )

        # 5.5. 履歴参照キーワードのフォールバック強制発火 (安全網)
        # router.HISTORY_KEYWORDS 相当の明示的な recall 語 (「覚えて」
        # 「最初に」等) を含むクエリで、層4 (aux) が no_tool と判定した
        # 場合の最終防衛線。小型 aux モデルの確率的な見落としで
        # search_history が一度も呼ばれず、長距離 recall がベースモデルの
        # 幻覚に倒れる実インシデントがあった (2026-07-20:「この会話で一番
        # 最初に私が計算させた問題は何だったか覚えてますか？」で
        # search_history 未発火 → 受動 RAG (quality=medium) のみで
        # 「そんな計算はなかった」と誤って断言)。
        # search_history へ渡すクエリは常に内容キーワードへ縮約する。
        # HistoryManager の照合は字句重なりベースで、疑問文全文は短い
        # 会話ターンにマッチせず空振りする (2026-07-21 ライブ検証:
        # 「この会話で一番最初に計算させた問題は何？」/ 2026-07-27 ライブ
        # 検証:「過去の会話で、登山の話題をしたことはありますか？探して
        # ください。」が、実際には該当する会話があるのに自分の質問文を
        # 含む直近セッションだけを拾って「見当たりません」と誤答)。
        # 当初は順序リコール質問のみ縮約していたが、明示的な検索依頼でも
        # 同じ空振りが起きるため全ケースで縮約する。内容語が取れない
        # クエリでは _reduce_ordered_history_query が生クエリを返すため
        # 従来挙動のまま。順序解釈は digest 側が受け取る raw query が
        # 担うため縮約で失われない。ヒットしなくても "No results found"
        # 経由で「見つからなかった」という正直な応答に倒れるため、無言の
        # まま確信を持って幻覚するより悪化はしない。
        # skip_judgment (雑談プレフィルタ) の判定結果に関わらず適用する:
        # 元インシデントのクエリ自体が `_SELF_ACTION_PATTERNS` の
        # 「私が」(無アンカーの部分一致) に「私が計算させた」の関係節部分で
        # 誤って一致し skip_judgment=True になっていた (2026-07-20 テストで
        # 判明)。履歴参照キーワードという強いシグナルがある以上、雑談判定
        # 側の誤検出よりこちらを優先する。
        # 語彙 (``_has_history_recall_keywords``) だけでなく **構造**
        # (``asks_about_past_conversation`` = 過去形の言及動詞 + 問いかけ) でも
        # 撃つ。閉じた語彙は必ず漏れ、漏れた側は **履歴ストアにしか無い時刻を
        # 捏造する** ことが実測で出た (2026-08-30 ライブ監査 T06: 10 ターン
        # すべてが no_match_in_any_layer で、「いつ、どんな話をしましたか。」に
        # 「2025年6月15日（日）の午後4時20分頃に」と断定した。実際は同日 20 分前)。
        if (
            tools_registry.has("search_history")
            and asks_about_past_conversation(query)
        ):
            search_query = _reduce_ordered_history_query(query)
            forced_result = ToolJudgement(
                tool_needed=True,
                tool_name="search_history",
                tool_args={"query": search_query, "mode": mode},
                source="rule",
            )
            self._maybe_scope_session_search(forced_result, query, session_id)
            forced_result = self._finalize(
                forced_result, tools_registry, mode, query=query,
            )
            # 「近接リコール語だけ + 現在セッション除外」の組合せは
            # _finalize の proximal_recall_excluded_session ガードが
            # no_tool へ降格させる (以前はここにインライン実装されており、
            # aux 経路には掛かっていなかった)。
            if forced_result.tool_needed:
                self._log_tool_decision(
                    forced_result, "history_keyword_forced_fallback",
                )
                return forced_result

        # ── 学習済みリコール (層 5.6 / 5.7) ───────────────────────────
        #
        # **決定論層がすべて外れてから引き当てる。** 以前は層 0 / 0.5 として
        # 決定論層より **前** に短絡していたため、一度保存されたコマンドが
        # ルール表と専用ツール層を恒久的に隠していた。実インシデント:
        #
        #   - 日時コマンドを astimezone() 付きへ直した後も、SemMem の naive 版が
        #     sim=0.9478 で引き当たり旧形式が実行された (2026-08-06)。
        #     ルール表が非該当のクエリだけが新コマンドを実行できていた。
        #   - 「このPCの搭載メモリの総量と空き容量」に対し、**RAM を出力しない**
        #     spec コマンドがリコールで再生され「確認できていません」と回答した
        #     (2026-08-22、2 セットで再現)。層0.6 の正規表現は当たっていた。
        #
        # 対処として ``dedicated_tool_query`` の除外リストを手で足してきたが、
        # 語彙を列挙する対処は必ず漏れる。順序そのものを直す。
        #
        # **コストは増えない。** 先回りの動機は「aux / 分類器の往復を省く」
        # ことだったが、層 0.6〜5.5 はすべて純粋な正規表現で、リコールを
        # それらの後ろへ動かしても増えるのは正規表現の評価だけ。リコールは
        # 依然として層 5.9 (ベースモデルの文法制約分類 = 唯一の推論往復) より
        # 前にあるので、学習済みクエリが分類器を撃つことはない。
        # 5.6. URL リコール (mode / enabled に関係なく実行)
        # ``_try_recall_url`` は決定論的 (embedding 類似度 + 過去採点平均閾値)
        # で、補助タスク同期発火やルール正規表現のような副作用がない。早期 return
        # 経路 (chat モード + tool_judge_enabled=false) で判定がスキップされる
        # と「過去 URL は SemMem にあるのに fetch されない」という不整合が起きる
        # ため、ここで先回りで引き当てる。
        url_recall_result = await self._judge_with_url_recall(
            query, tools_registry, mode=mode,
        )
        if url_recall_result is not None:
            url_recall_result = self._finalize(
                url_recall_result, tools_registry, mode, query=query,
            )
            self._log_tool_decision(url_recall_result, "url_recall_matched")
            return url_recall_result

        # 5.7. executable command リコール (mode / enabled 非依存)
        # 過去成功した run_command を SemMem から決定論的に引き当てる。層 5.9
        # (ベースモデルの文法制約分類 = 唯一の推論往復) より前にあるので、
        # 学習済みクエリで分類器を撃つことはない。
        #
        # URL リコールが「ユーザーが URL を書いた」という決定論的根拠を持つのに
        # 対し、command recall の根拠は類似度のみ。ツール意図のシグナルが無い
        # クエリ (好みの表明 / 記憶想起 / 感謝) まで引き当てると、会話履歴で
        # 答えられる質問が「ツール結果に含まれていません」に化ける
        # (実測 2026-07-25: 誤発火 6 件中 4 件がこの型)。適用はツールシグナルを
        # 持つクエリに限定し、かつ記憶想起クエリは除外する。
        # 「さっき伝えた GPU は何だった？」は 'GPU' が _TOOL_PATTERNS に載るため
        # ツールシグナル判定を通ってしまうが、答えは会話履歴にある。ここで
        # コマンドを撃つと、ツール結果が文脈を上書きして
        # 「ツール結果に GPU 型番は含まれていません」と誤答する (実測 2026-07-25。
        # 同じ会話の 1 ターン前では Radeon 890M を正しく想起できていた)。
        # create では run_command が一級のツールで、「依存を入れて」→ 学習済み
        # `pip install ...` の引き当てが本機能の主目的なのでゲートしない。
        #
        # 専用ツール (層 0.6 / 0.6b) やルール表 (層 1) の除外リストはもう要らない。
        # それらはこの層より前に評価されるため、claim されたクエリはここまで
        # 降りてこない (以前は逆順で、除外語彙を手で足し続けていた)。
        recall_allowed = is_create_mode(mode) or (
            _query_has_tool_signal(query)
            and not _has_history_recall_keywords(query)
        )
        if recall_allowed:
            cmd_recall_result = await self._judge_with_executable_command_recall(
                query, tools_registry, mode=mode,
            )
        else:
            cmd_recall_result = None
        if cmd_recall_result is not None:
            cmd_recall_result = self._finalize(
                cmd_recall_result, tools_registry, mode, query=query,
            )
            self._log_tool_decision(
                cmd_recall_result, "executable_command_recall_matched",
            )
            return cmd_recall_result


        # 5.9. ベースモデルの文法制約ツール分類 (docs/c_14 §1.3)。
        # 「ツールが要るのに撃たれない」穴を埋める最後の層。決定論層が
        # すべて外れてから実行する (決定論のシグナルの方がモデル判断より
        # 信頼できる)。内部の決定論プリゲート (``_gate_allows``) が、ツール
        # シグナルの無い雑談で推論を 1 往復も増やさないようにする。
        classified = (
            await self._judge_with_tool_classifier(
                query, tools_registry, mode, conversation, session_id,
            )
            if (self.enabled and allow_classifier) else None
        )
        if classified is not None:
            self._log_tool_decision(classified, "tool_classifier")
            return classified

        # 5.95. 差分クエリの式合成 (最後の接地手段)。
        # 分類器 (層5.9) が no_tool と判断した後だけ走る。
        synthesized = await self._judge_with_expression_synthesis(
            query, tools_registry, mode, conversation,
        )
        if synthesized is not None:
            self._log_tool_decision(synthesized, "expression_synthesis")
            return synthesized

        # 6. 全フォールバック失敗時の no_tool 結末を記録
        # (削除依頼の記録は judge() 冒頭で済ませてある)
        self._mark_blocked_if_unexecutable_command(
            query, tools_registry, mode,
        )
        no_tool_result = ToolJudgement(tool_needed=False, source="rule")
        self._log_tool_decision(
            no_tool_result, "no_match_in_any_layer",
        )
        return no_tool_result

    async def _judge_with_expression_synthesis(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
        conversation: list[dict] | None,
    ) -> "ToolJudgement | None":
        """被演算子が会話にしかない差分クエリで、式だけを合成させて calculate を撃つ。

        **判断させず合成だけ命じる**のが要点。層5.9 の分類器は「ツールが要るか」
        を判定するため、割り算のような難しい計算では ``calculate`` を選ぶが、
        引き算は自分で暗算できると判断して ``no_tool`` を返す。そして実際には
        間違える。

        実測 (2026-08-26 ライブ A/B、Qwen3.8-27B、在庫 12→9→14→12 の直後に
        「最初の在庫からいくつ減りましたか？」を尋ねる。正解 0):

        - 応答全体の正答率 **5/8** (誤答は ``6台`` / ``2台`` / ``12台``)
        - 分類器に ``calculate`` / ``no_tool`` だけ与えても **4/4 で no_tool**
        - 選択肢を与えず「式を合成せよ」と命じると **4/4 で ``12-12`` → 0**

        合成した式は ``_ungrounded_numbers`` で検証する。会話に無い数値が 1 つ
        でも混ざれば捨てる (捏造された式の結果は「正しく計算された嘘」になり、
        素の暗算より有害。``_suppress_ungrounded_calculate`` の docstring 参照)。

        発火条件を「クエリに数値が無い差分/照応形」に閉じるのは、往復 1 回
        (実測 15〜20 秒) を無関係なターンへ足さないため。被演算子がクエリ側に
        あるケースは層5.9 が既に ``calculate`` を選べる。
        """
        if not tools_registry.has("calculate"):
            return None
        # 被演算子がクエリにあるなら層5.9 で足りる。ここは「会話にしかない」形専用。
        if NUMBER_LITERAL_RE.search(query):
            return None
        if not ANAPHORIC_OPERAND_RE.search(query):
            return None
        if not looks_like_numeric_question(
            query, _recent_dialogue_text(conversation),
        ):
            return None
        client = self._llm_client
        if client is None or not hasattr(client, "generate_constrained"):
            return None

        # ⚠ 判定用の 4 ターン窓を渡してはいけない。基準値 (「最初の〜」) が
        # 窓の外にあると、窓内の最古の値で式を作って必ず間違える。
        messages = _recent_dialogue_messages(
            conversation, _SYNTHESIS_CONTEXT_TURNS,
        )
        messages.append({"role": "user", "content": query})
        messages.insert(0, {
            "role": "system",
            "content": select_locale_variant(
                EXPRESSION_SYSTEM, EXPRESSION_SYSTEM_EN,
            ),
        })
        try:
            content = await client.generate_constrained(
                messages,
                response_format=EXPRESSION_SCHEMA,
                max_tokens=CLASSIFY_MAX_TOKENS,
                # 分類器と同じ専有スロット (CLAUDE.md §6 #1)。
                id_slot=getattr(
                    client, "classifier_slot",
                    getattr(client, "background_slot", -1),
                ),
                timeout=self._tool_classifier_timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("expression synthesis failed: %s", exc)
            return None

        expression = parse_expression_response(content)
        if not expression:
            return None
        unexplained = _ungrounded_numbers(
            expression, query, _dialogue_text(conversation),
        )
        if unexplained:
            logger.info(
                "Expression synthesis rejected: %s uses numbers absent from "
                "the conversation (%s)", expression, ", ".join(unexplained),
            )
            return None
        result = ToolJudgement(
            tool_needed=True,
            tool_name="calculate",
            tool_args={"expression": expression},
            source="classifier",
        )
        return self._finalize(result, tools_registry, mode, query=query)

    _NATIVE_JUDGE_SYSTEM = (
        "あなたはツール選択器です。ユーザーの発言に答えるのに必要なツールを"
        "1つだけ選び、JSON で返します。回答文の作成は後続の工程が担当します。\n"
        "tool には下記のいずれかの名前を入れ、arg にはそのツールの主引数"
        "(式・クエリ・コマンド・パス等) を入れます。ツールが不要なら"
        'tool="none"、arg="" を返します。\n\n利用可能なツール:\n'
    )
    _NATIVE_JUDGE_SYSTEM_EN = (
        "You are a tool selector. Pick at most one tool needed to answer the "
        "user's message and return it as JSON. Writing the reply itself is "
        "handled by a later stage.\n"
        "Put the tool name in `tool` and its primary argument (expression, "
        "query, command, path, ...) in `arg`. If no tool is needed, return "
        'tool="none" with arg="".\n\nAvailable tools:\n'
    )

    async def _gate_allows(
        self, query: str, conversation: list[dict] | None,
    ) -> bool:
        """分類器を撃つかどうかの門。

        埋め込み kNN が準備できていればそちらを使い、``None`` (判定不能) や
        未準備なら従来の正規表現ゲートへ縮退する。**誤って閉じない**方向に
        倒すのは、取りこぼしのコスト (誤答) が無駄撃ちのコスト (分類器 1 回)
        より高いため。

        正規表現側は数値計算の判定だけ直近の会話も見る (被演算子の片方が前
        ターンにしか無い言い回しがあるため。``looks_like_numeric_question``)。
        """
        # **平叙の自己申告は問答無用で止める。**
        #
        # 層 5.9 はこの判定系で唯一の推論往復で、実測 34〜39 秒かかる
        # (2026-08-25 ライブ監査、Qwen3.8-27B)。同じ監査で kNN ゲートが
        # 「ツールが要る」と判定した 8 種のうち **3 種が単なる事実の申告**
        # だった:
        #
        #   「ビルドで SIGSEGV_A17X というエラーコードが出ることがあります。」(votes 3/5)
        #   「毎週火曜日の15時に定例会議があります。」                        (votes 3/5)
        #   「私の誕生日は3月14日です。」                                     (votes 3/5)
        #
        # どれもツールでは答えようがない。1 件あたり 34 秒を捨てていた。
        # 判定は層 0.6b と同じ ``is_plain_statement`` — 問い・依頼のマーカーが
        # 無く、かつ平叙の文末で終わる場合だけ True になるので、体言止めの
        # 問い合わせ (「PC のスペック」) は止めない。実測 13 クエリで
        # 誤発火 6 件を全て捕捉し、正当な 7 件は 1 件も止めなかった。
        if is_plain_statement(query):
            logger.debug(
                "Tool classifier gate: plain statement, not a request: %s",
                query[:50],
            )
            return False
        gate = self._tool_gate
        if gate is not None:
            verdict = await gate.needs_tool(query)
            if verdict is not None:
                return verdict
        return _query_has_tool_signal(query, _recent_dialogue_text(conversation))

    async def warmup_tool_gate(self) -> bool:
        """ツール要否ゲートの exemplar を埋め込む (起動後の背景タスク用)。"""
        if self._tool_gate is None:
            return False
        return await self._tool_gate.warmup()

    async def _judge_with_tool_classifier(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
        conversation: list[dict] | None = None,
        session_id: str = "",
    ) -> "ToolJudgement | None":
        """ベースモデルに文法制約 JSON でツールを選ばせる (docs/c_14 §1.3)。

        決定論層と層4 の補助タスク判定がいずれも結論を出さなかったときに走る
        最終層 (層5.9)。「ツールが要るのに撃たれない」穴を埋める。

        選ばせ方は OAI ``tools`` ではなく ``response_format`` (json_schema) の
        enum 分類。実測 (2026-08-12, Qwen3.5-27B / gemma-4-12b) では ``tools`` は
        200 で受理されてもモデルが tool_call を出さずに本文を書き始め、
        ``max_tokens`` を使い切って 15.6〜60.2 秒を捨てる (``tool_choice:
        "required"`` でも 6 件中 3 件で無視された)。json_schema は llama-server
        側の GBNF 制約なので必ず従い、出力トークン数の上限が読める。

        コスト対策として、呼ぶ前に決定論プリゲート ``_query_has_tool_signal``
        を通す。ツールシグナルの無い雑談で推論を 1 往復増やさないため。
        なお本ゲートは再現率が低く (実測 20 ケースでツールが要る 14 件中 6 件を
        遮断)、単位換算のような「ツールシグナルの無い算術」を落とす。これは
        埋め込み kNN の一次振り分けで置き換える予定 (別作業)。

        判定結果は ``_finalize(aux_guards=True)`` を通す。分類器の引数も
        モデルが自由生成したものであり、LLM 判定と同じグラウンディングの
        ガードが要るため。ただし ``hidden_tools_offered=True`` を渡して隠し
        ツール抑止だけは外す — ``build_classifier_schema`` が hidden も enum に
        載せている以上、選ばれた名前は hallucination ではない。

        Returns:
            ツールが選ばれた場合 ``ToolJudgement``。それ以外は ``None``。
        """
        if not (self._tool_classifier_enabled and self._tool_classifier_supported):
            return None
        client = self._llm_client
        if client is None or not hasattr(client, "generate_constrained"):
            return None
        # プリゲート: ツールが要らないターンでは 1 往復も増やさない。
        if not await self._gate_allows(query, conversation):
            return None

        schema = build_classifier_schema(tools_registry, mode)
        if schema is None:
            return None

        # 対話窓は **照応のあるクエリだけ** に載せる。
        #
        # プロンプトは [system=ツールメニュー (毎回同一)][対話窓][クエリ] の順で、
        # 中間の対話窓が毎ターン入れ替わるため接頭辞キャッシュが崩壊する。
        # 通常 attention なら ``--cache-reuse`` (KV シフト) が救うが、hybrid
        # recurrent モデルでは llama-server が cache_reuse ごと無効化するので
        # 救えない (launch_llama.py が起動時に warning を出している事実)。
        # 実測は ``query_needs_dialogue`` の docstring を参照。
        if query_needs_dialogue(query):
            messages = _recent_dialogue_messages(conversation)
        else:
            messages = []
        messages.append({"role": "user", "content": query})
        # 役割を宣言しないと、モデルは判定ではなく **回答本文** を書き始める
        # (詳細は _NATIVE_JUDGE_SYSTEM のコメント)。文法制約は形式を保証するが
        # 「何を選ぶか」の質は役割宣言に依存する。
        messages.insert(0, {
            "role": "system",
            "content": select_locale_variant(
                self._NATIVE_JUDGE_SYSTEM, self._NATIVE_JUDGE_SYSTEM_EN,
            ) + build_tool_menu(tools_registry, mode),
        })

        try:
            content = await client.generate_constrained(
                messages,
                response_format=schema,
                max_tokens=self._tool_classifier_max_tokens,
                # 補助判定は専有スロット固定 (CLAUDE.md §6 #1)。省略すると
                # llama-server が LCP 類似度 / LRU でスロットを選ぶため、
                # (a) 自分の前ターンの KV を再利用できずに毎回フルプリフィル
                # (2026-08-23 実測: 450〜590 トークン / 11〜16 秒 を 13 回)、
                # (b) 選ばれたのがチャットスロットならチャット側の接頭辞
                # キャッシュを破壊する、の 2 つを踏む。``generate_constrained``
                # の他の呼出元 (AuxClient / PromptEvolver) は全て固定済みで、
                # ここだけが漏れていた。
                #
                # **さらに背景スロットとも分ける** (``classifier_slot``)。
                # 背景スロットは sleep-time / aux と共有で、ターンの合間に走る
                # 補助タスクが分類器のプレフィクス (ツールメニュー 385 トークン、
                # 毎回同一) を毎回追い出していた。実測 (2026-08-25、Qwen3.8-27B):
                # 追い出される側は 422 tok / cache 0 で 37.7 秒、追い出されない
                # 側は **別のクエリでも** cache 390/418 で 11.3 秒。
                # 1 回あたり約 26 秒 (3.3 倍) 縮む。``llama.slots < 3`` の構成では
                # 従来どおり背景スロットへ倒れる。
                id_slot=getattr(
                    client, "classifier_slot",
                    getattr(client, "background_slot", -1),
                ),
                timeout=self._tool_classifier_timeout_sec,
            )
        except httpx.HTTPStatusError as exc:
            # ``response_format`` 非対応の build。リトライしても回復しないので
            # 以後この経路を使わない (毎ターン 4xx を踏まない)。
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                self._tool_classifier_supported = False
                logger.warning(
                    "Grammar tool classifier disabled for this process: "
                    "llama-server rejected the response_format payload (HTTP %s)",
                    exc.response.status_code,
                )
            else:
                logger.info("grammar tool classifier failed: %s", exc)
            return None
        except Exception as exc:
            logger.info("grammar tool classifier failed: %s", exc)
            return None

        parsed = parse_classifier_response(content, tools_registry, mode)
        if parsed is None:
            return None
        tool_name, tool_args = parsed

        result = ToolJudgement(
            tool_needed=True,
            tool_name=tool_name,
            tool_args=tool_args,
            source="classifier",
        )
        # 自由生成の search_history はセッションスコープの後処理が要る
        # (自己参照は現在セッションへ限定 / 順序リコールは横断へ拡張)。
        self._maybe_scope_session_search(result, query, session_id)
        self._maybe_expand_ordered_history_search(result, query)
        # 分類器は自由生成の arg をそのまま query に入れるため、文のまま
        # 渡ってくることがある (層 5.5 と違い縮約が掛からない)。
        self._maybe_reduce_history_query(result, query)

        result = self._finalize(
            result,
            tools_registry, mode, query=query,
            conversation=conversation, aux_guards=True,
            hidden_tools_offered=True,
        )
        if not result.tool_needed:
            return None

        # readonly 検証は ``_finalize`` ではなく **各層の出口** で適用する設計
        # (層1/5/5.5 と同じ)。``_finalize`` は mode 外の兄弟ツールへ載せ替える
        # ため (run_command → run_command_readonly)、載せ替え **後** の名前と
        # コマンドで判定しないと、破壊的コマンドが chat の読み取り専用ツールへ
        # そのまま乗ってしまう。
        command = (result.tool_args or {}).get("command")
        if isinstance(command, str) and command and self._reject_readonly(
            result.tool_name, command,
        ):
            logger.info(
                "Native tool call rejected: %s is not read-only (%s)",
                result.tool_name, command[:80],
            )
            # 「撃てなかった」ことを呼出側へ伝える。黙って no_tool に落とすと
            # base が完了報告や測定値を捏造する。
            # ただし *何を* 撃ち損ねたかで注記が変わる:
            #   - 状態を変える試み → ``_UNPERFORMED_ACTION_GUIDANCE``
            #   - 検査 (読み取り) の試み → ``_UNMEASURED_FACT_GUIDANCE``
            # 一律に action 扱いすると、``test -f`` のような読み取りにまで
            # 「状態を変える操作を実行していない」が付き、base が「確認する
            # ツールが利用できない」と誤った説明をする (2026-08-15 ターン12)。
            # 本経路は判定層がコマンドツールを選択済み = 意図が確定しているため、
            # ``_user_requested_measurement`` を待たず measurement を記録してよい
            # (``_reject_readonly`` の docstring 参照)。
            if _command_is_readonly_inspection(command):
                self._measurement_blocked = True
            else:
                self._action_blocked = True
            return None

        logger.info(
            "Tool classifier selected: %s(%s) (query=%s)",
            result.tool_name, result.tool_args, query[:50],
        )
        return result

    def _finalize(
        self,
        result: "ToolJudgement",
        tools_registry: ToolsRegistry,
        mode: str,
        *,
        query: str = "",
        conversation: list[dict] | None = None,
        aux_guards: bool = False,
        hidden_tools_offered: bool = False,
    ) -> "ToolJudgement":
        """``judge()`` の全 exit が通る唯一の後処理 funnel.

        各判定層 (rule / cartridge / learned / aux / 各種リコール・
        フォールバック) は、確定した ``ToolJudgement`` を必ず本メソッドへ通して
        から返す。**新しい抑止を足すときの編集箇所を 1 つに保つ**ことが目的。

        以前は同じガード列が層ごとに手で書き写されており、層によって適用される
        ガードの部分集合が食い違っていた (rule / learned は 4 つ、cartridge は
        2 つ、リコール系は ``_validate_tool_availability`` のみ)。抑止を足した
        当時のインシデント経路にだけガードが付き、同じ穴を持つ他の層は素通しの
        まま残る、という抜けが実際に起きていた。

        ガードは全て純粋な ``ToolJudgement -> ToolJudgement`` で、対象ツール名が
        一致しなければ何もしない。したがって「そのツールを返さない層」に適用
        しても no-op であり、順序付きリストとして一括適用して安全。

        Args:
            aux_guards: aux (層4) 専用ガードも適用するか。
                ``_suppress_hidden_tool_from_aux`` は「プロンプトのツール一覧に
                出ない名前を aux が返すのは hallucination」という前提の防衛で、
                コード側がツール名を注入する経路 (chat の
                ``run_command_readonly`` 等) に掛けると正当な判定を潰すため、
                aux 経路でのみ有効化する。grounding 系 2 つも aux の
                free-form args 向けなので同様に限定する。
            hidden_tools_offered: モデルに提示したツール一覧に hidden ツールを
                **含めた**か。上記 hallucination 前提が成り立つのは「提示して
                いない名前が返ってきた」ときだけで、文法制約ツール分類は
                ``build_classifier_schema`` が hidden も enum に載せる (hidden は
                「プロンプト一覧に出さない」印であって「使わせない」印ではない)。
                提示した上で選ばれた名前を hallucination として潰すと、chat で
                唯一の実行系ツール ``run_command_readonly`` が分類器経路から
                恒久的に到達不能になる (実インシデント 2026-08-08 ライブ監査
                ターン13: ファイル追記が 1 度も実行されないまま「追記しました。
                行数は5行です」と捏造。実ファイルは 1 行のまま無変更だった)。
        """
        ctx = GuardContext(
            tools_registry=tools_registry,
            mode=mode,
            query=query,
            conversation=conversation,
            aux_guards=aux_guards,
            hidden_tools_offered=hidden_tools_offered,
        )
        result = apply_guards(result, ctx)
        self._absorb_blocked_flags(ctx)
        return result

    def _absorb_blocked_flags(self, ctx: GuardContext) -> None:
        """ガードが立てた「実行できなかった」印をインスタンスへ引き継ぐ。

        ガードは純粋関数なので ``ctx`` にしか書かない。``judge()`` は冒頭で両
        フラグを落とすため、ここでは立った側だけを取り込めばよい。
        """
        self._measurement_blocked = (
            self._measurement_blocked or ctx.measurement_blocked
        )
        self._action_blocked = self._action_blocked or ctx.action_blocked

    # --- ガードへの薄い委譲 -------------------------------------------------
    # 実体は :mod:`backend.free.agent.tool_judge_guards` の純粋関数
    # (適用順は :data:`~backend.free.agent.tool_judge_guards.GUARD_PIPELINE`)。
    # ここに残すのは「ガードを 1 件だけ掛ける呼出面」と、``ctx`` に立った印を
    # インスタンスへ引き継ぐ責務の 2 つだけ。

    def _suppress_proximal_recall_cross_session(
        self, result: ToolJudgement, query: str,
    ) -> ToolJudgement:
        """ガード ``proximal_recall_excluded_session`` を単体で掛ける。"""
        return guards._suppress_proximal_recall_cross_session(
            result, GuardContext(query=query),
        )

    def _validate_tool_availability(
        self, result: ToolJudgement, tools_registry: ToolsRegistry, mode: str,
    ) -> ToolJudgement:
        """ガード ``tool_availability`` を単体で掛ける。"""
        ctx = GuardContext(tools_registry=tools_registry, mode=mode)
        result = guards._validate_tool_availability(result, ctx)
        self._absorb_blocked_flags(ctx)
        return result

    def _scope_list_directory_depth(
        self, result: ToolJudgement, query: str,
    ) -> ToolJudgement:
        """ガード ``immediate_children_depth`` を単体で掛ける。"""
        return guards._scope_list_directory_depth(result, GuardContext(query=query))

    def _scope_read_file_line_range(
        self, result: ToolJudgement, query: str,
    ) -> ToolJudgement:
        """ガード ``read_file_line_range`` を単体で掛ける。"""
        return guards._scope_read_file_line_range(result, GuardContext(query=query))

    def _restore_truncated_text_operand(
        self, result: ToolJudgement, conversation: list[dict] | None,
    ) -> ToolJudgement:
        """ガード ``truncated_text_operand`` を単体で掛ける。"""
        return guards._restore_truncated_text_operand(
            result, GuardContext(conversation=conversation),
        )

    def _suppress_unfetchable_fetch_url(self, result: ToolJudgement) -> ToolJudgement:
        """ガード ``unfetchable_fetch_url`` を単体で掛ける。"""
        return guards._suppress_unfetchable_fetch_url(result, GuardContext())

    def _suppress_commandless_run_command(
        self, result: ToolJudgement,
    ) -> ToolJudgement:
        """ガード ``commandless_run_command`` を単体で掛ける。"""
        return guards._suppress_commandless_run_command(result, GuardContext())

    def _suppress_hidden_tool_from_aux(
        self, result: ToolJudgement, tools_registry: ToolsRegistry,
    ) -> ToolJudgement:
        """ガード ``hidden_tool_from_aux`` を単体で掛ける。"""
        return guards._suppress_hidden_tool_from_aux(
            result, GuardContext(tools_registry=tools_registry),
        )

    def _suppress_expressionless_calculate(
        self, result: ToolJudgement,
    ) -> ToolJudgement:
        """ガード ``expressionless_calculate`` を単体で掛ける。"""
        return guards._suppress_expressionless_calculate(result, GuardContext())

    def _suppress_ungrounded_calculate(
        self,
        result: ToolJudgement,
        query: str,
        conversation: list[dict] | None,
    ) -> ToolJudgement:
        """ガード ``ungrounded_calculate`` を単体で掛ける。"""
        return guards._suppress_ungrounded_calculate(
            result, GuardContext(query=query, conversation=conversation),
        )

    def _suppress_ungrounded_read_path(
        self,
        result: ToolJudgement,
        query: str,
        conversation: list[dict] | None,
    ) -> ToolJudgement:
        """ガード ``ungrounded_read_path`` を単体で掛ける。"""
        return guards._suppress_ungrounded_read_path(
            result, GuardContext(query=query, conversation=conversation),
        )

    async def _judge_with_url_recall(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
    ) -> "ToolJudgement | None":
        """URL リコール単独で fetch_url 判定を返す (mode / enabled 非依存).

        条件:
          - ``fetch_url`` ツールが登録済み
          - ``mem_view`` / ``embedder`` が wired されている
          - ``_try_recall_url`` が閾値判定で URL を返す

        Returns:
            URL 引き当て成立時は ``ToolJudgement(fetch_url, {"url": ...})``、
            それ以外は ``None`` (通常の判定フローに falling-through する)。
        """
        if not tools_registry.has("fetch_url"):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        # クエリが実在するローカルファイルを明示参照している場合、URL recall の
        # 無条件短絡はスキップし後段の判定層へ落とす。操作対象が具体的な
        # ローカルファイルであるタスク (実インシデント: "Read <path>.xlsx /
        # Apply monthly borders...") が、過去の無関係な URL 記憶と埋め込み
        # 類似度だけで fetch_url へハイジャックされるのを決定論的に防ぐ。
        # 書込み先ディレクトリ指定 (url_write 正規フロー) は is_file()=False の
        # ため影響せず、後段の rule/learned 層が fetch_url を選べば
        # ``_maybe_recall_url`` の URL 補完も引き続き機能する。
        referenced = _extract_file_path(query)
        if referenced:
            try:
                if Path(referenced).is_file():
                    logger.info(
                        "URL recall: skipped (query references existing "
                        "local file %s)", referenced,
                    )
                    return None
            except (OSError, ValueError):
                pass
        # パスを明示しないローカルファイル参照 (「そのファイルの中身を読み込んで
        # 見せて」等) は _extract_file_path で拾えないため、上の実在チェックを
        # すり抜けて埋め込み類似度だけで fetch_url へ短絡していた (実インシデント
        # 2026-07-27 ライブ検証: 直前ターンで保存した note2.txt の読み出し依頼が
        # 過去セッションの example.com への fetch_url になった)。web 参照シグナル
        # が無いローカルファイル依頼は URL recall の対象外にする。
        if _query_targets_local_file_only(query):
            logger.info(
                "URL recall: skipped (local file reference without web signal): %s",
                query[:50],
            )
            return None
        recalled = await self._try_recall_url(query, mode=mode)
        if not recalled:
            return None
        logger.info(
            "URL recall: matched url=%s for query=%s", recalled, query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name="fetch_url",
            tool_args={"url": recalled},
            source="rule",
        )

    async def recall_url_judgement(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
    ) -> "ToolJudgement | None":
        """外部から URL recall のみを問い合わせる公開 API.

        Reactive レイヤの escalation 判定など、judge() フル実行前に
        「URL recall だけ」をチェックしたい呼び元向け。判定本体は
        ``_judge_with_url_recall`` を共有し、戻り値も同じ。
        """
        return await self._judge_with_url_recall(query, tools_registry, mode=mode)

    async def _maybe_recall_url(
        self, result: "ToolJudgement", query: str, mode: str = "create",
    ) -> None:
        """rule / learned 層が ``fetch_url`` を返したが URL が空の場合、
        過去質問で正しく fetch できた URL (``mem.world.url.*``) を引き当てる。

        引き当てが成立すると ``result.tool_args["url"]`` に補完して
        in-place で更新する。失敗 / 引き当てなしの場合は何もしない。
        ``mem_view`` / ``embedder`` が None (degraded) でも安全に no-op。
        """
        if result.tool_name != "fetch_url":
            return
        if result.tool_args and result.tool_args.get("url"):
            return
        recalled = await self._try_recall_url(query, mode=mode)
        if not recalled:
            return
        if result.tool_args is None:
            result.tool_args = {}
        result.tool_args["url"] = recalled
        logger.debug("URL recall: matched url=%s for query=%s", recalled, query[:50])

    def _maybe_scope_session_search(
        self, result: "ToolJudgement", query: str, session_id: str,
    ) -> None:
        """``search_history`` の現在セッションの扱いを code 側で強制する。

        クエリが「この会話で」等のセッション自己参照パターンに一致する場合は
        ``tool_args["session_id"]`` を強制注入して検索対象を現在セッションのみに
        限定する。session_id を渡さずに search_history を無条件許可すると、
        2026-07-17/18 の実インシデント (「この会話で一番面白かった？」が無関係な
        過去セッションの内容を誤って混同した) が再発するため。

        自己参照でない場合は逆に ``tool_args["exclude_session_id"]`` を注入して
        現在セッションを結果から外す。現在セッションの発言は既に会話コンテキスト
        へ全文が載っており再注入しても情報は増えないのに、セッション要約
        (= 会話冒頭の発言) が「独立した根拠」の顔で入り、後から訂正された内容を
        訂正前の値へ巻き戻す (2026-07-26 ライブ検証: 火曜→水曜と訂正した歯科の
        予約が、2 ターン後に検索結果のセッション要約経由で火曜へ戻った)。

        in-place で更新する。``session_id`` が空 (未提供) の場合は何もしない
        (呼出元が未対応でも安全に no-op)。
        """
        if result.tool_name != "search_history":
            return
        if not session_id:
            return
        if is_en_locale():
            patterns = _SELF_SESSION_REFERENCE_PATTERNS_EN
            self_reference = (
                not _SESSION_TOPIC_BREAK_LEAD_RE_EN.search(query)
                and any(p.search(query) for p in patterns)
            )
        else:
            patterns = _SELF_SESSION_REFERENCE_PATTERNS
            self_reference = any(p.search(query) for p in patterns)
        if result.tool_args is None:
            result.tool_args = {}
        if self_reference:
            result.tool_args["session_id"] = session_id
            logger.debug(
                "search_history scoped to current session "
                "(self-session reference): %s", query[:50],
            )
            return
        result.tool_args["exclude_session_id"] = session_id
        logger.debug(
            "search_history excludes current session (already in context): %s",
            query[:50],
        )

    def _maybe_reduce_history_query(
        self, result: "ToolJudgement", query: str,
    ) -> None:
        """``search_history`` の検索語が **文のまま** なら内容キーワードへ縮約する.

        層 5.5 の強制フォールバックは ``_reduce_ordered_history_query`` を通すが、
        分類器 (層 5.9) は文法制約 JSON の ``arg`` をそのまま
        ``tool_args["query"]`` に入れるため縮約が掛からない。ツール説明は
        「Keywords to search for, NOT the user's question verbatim」と書いて
        いるが、**説明は守られないことがある**。

        実インシデント (2026-08-25 ライブ監査): 「過去の会話から「猫」について
        話した内容を検索してください。」で検索語が生の文全体になり 0 件。
        ``HistoryManager`` の字句照合は長い疑問文を短い会話ターンに当てられない。

        縮約するのは **文らしいときだけ**。分類器が良いキーワード
        (``登壇予定日``) を出しているときに書き換えると悪化する。文らしさは
        「生クエリと同一」か「文末記号・依頼形を含む」で見る (どちらも
        キーワードには現れない)。in-place で更新する。
        """
        if result.tool_name != "search_history":
            return
        args = result.tool_args or {}
        raw = args.get("query")
        if not isinstance(raw, str) or not raw.strip():
            return
        current = raw.strip()
        if not _looks_like_sentence(current, query):
            return
        reduced = _reduce_ordered_history_query(query)
        if not reduced or reduced.strip() == current:
            return
        args["query"] = reduced
        result.tool_args = args
        logger.debug(
            "search_history query reduced (classifier passed a sentence): "
            "%r -> %r", current[:60], reduced[:60],
        )

    def _maybe_expand_ordered_history_search(
        self, result: "ToolJudgement", query: str,
    ) -> None:
        """時系列順序指定クエリの ``search_history`` で小さい limit を既定値へ引き上げる.

        実インシデント (2026-07-21 ライブ検証 ターン18): 「この会話で一番最初に
        計算させた問題は?」で aux が ``query='計算', limit=1`` を合成 →
        ``HistoryManager.search_sessions`` は字句スコア降順で limit 件に切る
        ため、時系列先頭ではなく直近の計算を返し誤答した。limit が十分なら
        turn# 付きの全マッチターンが digest に渡り、元クエリ (digest の user
        prompt に含まれる) と合わせて時系列選択が機能する (同検証 ターン42
        「すべて挙げて」が limit 既定 10 で 6 件完全列挙に成功した実績)。

        判定は aux 合成後の ``args["query"]`` ではなく**ユーザー生クエリ**
        に対して行う (合成 query では「一番最初」等の順序語が消えている)。
        引き上げのみで引き下げはしない (「直近20件」等の明示的な大 limit を
        壊さない)。挿入点は aux 層のみ — limit を合成し得るのは free-form
        args の aux 層だけで、rule/learned 層の ``_infer_tool`` に
        search_history 分岐は無く、cartridge 層は空 args、層5.5 の強制発火は
        limit を設定しない (ハンドラ既定 10 が効く)。in-place で更新する。

        aux (LFM2) は json_schema grammar を強制せず型崩れ JSON を返し得る
        (limit が "1" (文字列) や 2.0 (float) 等) ため、数値相当は int へ
        正規化してから判定する。引き上げ時は int で書き戻すため、後続の
        search_history ハンドラ (limit で slice する) への型汚染も防ぐ。
        """
        if result.tool_name != "search_history":
            return
        if not _ORDERED_HISTORY_QUERY_RE.search(query):
            return
        args = result.tool_args or {}
        limit = _coerce_positive_int(args.get("limit"))
        if limit is not None and 0 < limit < _HISTORY_SEARCH_DEFAULT_LIMIT:
            args["limit"] = _HISTORY_SEARCH_DEFAULT_LIMIT
            result.tool_args = args
            logger.debug(
                "search_history limit expanded %d -> %d for ordered query: %s",
                limit, _HISTORY_SEARCH_DEFAULT_LIMIT, query[:50],
            )

    async def _try_recall_url(self, query: str, mode: str = "create") -> str | None:
        """過去質問の URL fact から類似質問の URL を返す。

        条件:
          - ``mem_view`` / ``embedder`` が両方提供されている
          - ``tools.url_recall_enabled`` が True
          - top-K 候補のうち ``world_fact`` で subject prefix が
            ``mem.world.url.`` のもの
          - 類似度 >= ``url_recall_min_score``
          - 過去採点平均 (``_extra.score_avg``) >= ``url_recall_min_record_score``
          - profile_id が一致する (異プロファイルの URL を引かない)

        Returns:
            条件を満たす最良の URL。なければ ``None``。
        """
        tools_cfg = (self._config or {}).get("tools") or {}
        if not bool(tools_cfg.get("url_recall_enabled", True)):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        if not query:
            return None
        # embed_query は LRU キャッシュ付きの単一クエリ経路。同一ターンでは
        # 検索パイプライン (run_search_pipeline) が同じ (query, mode) で既に
        # 埋め込み済みなので、ここはキャッシュヒットになり埋め込みサーバへの
        # 往復が消える。embed() を直接呼ぶとキャッシュを迂回して毎ターン
        # 二重に埋め込んでいた (2026-07-27 実測: 1 ターンあたり +0.35s)。
        try:
            embedding = await self._embedder.embed_query(query, mode=mode)
        except Exception as exc:
            logger.warning("URL recall: embed failed: %s", exc)
            return None
        if embedding is None or len(embedding) == 0:
            return None
        import numpy as _np
        q_vec = _np.asarray(embedding, dtype=_np.float32)

        top_k = int(tools_cfg.get("url_recall_topk", 5))
        min_sim = float(tools_cfg.get("url_recall_min_score", 0.7))
        min_avg = float(tools_cfg.get("url_recall_min_record_score", 0.6))
        ttl_days = int(tools_cfg.get("url_recall_ttl_days", 30))
        ttl_seconds = float(ttl_days) * 86400.0 if ttl_days > 0 else 0.0
        try:
            candidates = self._mem_view.search_by_embedding(q_vec, top_k=top_k)
        except Exception as exc:
            logger.warning("URL recall: search_by_embedding failed: %s", exc)
            return None

        # Pro 拡張: team プロファイルの URL fact も引き当て候補に含める。
        # Free build では factory が登録されていないため allowed = {self._profile_id}
        # のみで Phase 1 と等価。
        allowed_profiles: set[str] = {self._profile_id}
        try:
            from backend.edition import get_pro_handler
            factory = get_pro_handler("url_recall_resolver_factory")
            if callable(factory):
                resolver = factory(self._config)
                if resolver is not None:
                    for pid in resolver.allowed_profile_ids():
                        if pid:
                            allowed_profiles.add(pid)
        except Exception as exc:
            logger.warning("URL recall: pro resolver init failed: %s", exc)

        import time as _time
        now = _time.time()

        # recall miss の理由を後から追えるよう、最良 (= 最高 sim) の
        # ``mem.world.url.*`` 候補を記録しておく (candidates は score 降順想定)。
        best_subject: str | None = None
        best_sim: float | None = None
        best_reason = "no_url_candidate"

        for fact, sim in candidates:
            if fact.type != "world_fact":
                continue
            if not fact.subject.startswith("mem.world.url."):
                continue
            if best_subject is None:
                best_subject, best_sim, best_reason = fact.subject, sim, "sim_below_min"
            if sim < min_sim:
                # candidates は score 降順想定。閾値未満は以降全て無効。
                break
            if fact.profile_id and fact.profile_id not in allowed_profiles:
                continue
            extra = fact._extra or {}
            url = extra.get("url")
            score_avg = float(extra.get("score_avg") or 0.0)

            # 鮮度ペナルティ: TTL 超過なら score_avg を半減して閾値判定。
            # 完全 skip ではなく penalize に留めるのは、「古いが他に候補なし」
            # の状況で min_record を下げて運用すれば引けるようにするため。
            effective_score = score_avg
            if ttl_seconds > 0.0:
                last_fetched = float(extra.get("last_fetched_at") or 0.0)
                if last_fetched > 0.0:
                    age_sec = now - last_fetched
                    if age_sec > ttl_seconds:
                        effective_score = score_avg * 0.5
                        logger.warning(
                            "URL recall: TTL exceeded for %s "
                            "(age=%.0fd, ttl=%dd), penalize score %.3f -> %.3f",
                            fact.subject, age_sec / 86400.0, ttl_days,
                            score_avg, effective_score,
                        )

            if url and effective_score >= min_avg:
                # 非 ASCII を含む URL は壊れている可能性が高い (旧 URL 抽出 regex が
                # 末尾の日本語を取り込んだ残骸など)。fetch で 404 になるため引き当てない。
                if not str(url).isascii():
                    logger.warning(
                        "URL recall: skipping non-ASCII (likely malformed) url=%s",
                        url,
                    )
                    continue
                logger.info(
                    "URL recall: match sim=%.3f (min_sim=%.2f) score_avg=%.3f "
                    "(min_record=%.2f) url=%s",
                    sim, min_sim, effective_score, min_avg, url,
                )
                return str(url)
            # sim は満たしたが score_avg/TTL で落ちたケースを記録 (最初の 1 件のみ)
            if best_subject == fact.subject and best_reason == "sim_below_min":
                best_reason = "score_avg_below_min"

        # 引き当て無し: なぜ外れたかを DEBUG で可視化する (閾値チューニングの根拠)。
        if best_subject is None:
            logger.debug(
                "URL recall: no mem.world.url candidate in top-%d for query=%r",
                top_k, query[:50],
            )
        else:
            logger.debug(
                "URL recall: no match for query=%r; best candidate subject=%s "
                "sim=%.3f (min_sim=%.2f) min_record=%.2f reason=%s",
                query[:50], best_subject, best_sim or 0.0, min_sim,
                min_avg, best_reason,
            )
        return None

    async def _judge_with_executable_command_recall(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str = "create",
    ) -> "ToolJudgement | None":
        """SemMem の過去成功コマンド引き当てで executable 判定を返す.

        ``_judge_with_url_recall`` と対称。条件:
          - 現在の mode で executable ツールが解決できる
            (create → run_command / chat → run_command_readonly)
          - ``mem_view`` / ``embedder`` が wired されている
          - ``_try_recall_executable_command`` が閾値判定でコマンドを返す

        引き当てが成立すれば aux 呼出 (5 層目 / chat early-return) より
        先に確定するため、学習済みクエリでは LLM コストがゼロになる。

        recall は subject の mode (`mem.world.executable_command.<mode>.*`) を
        フィルタしないため、chat では create 学習由来の任意コマンド (書込系
        等) が引き当たり得る。chat (readonly) のときは
        ``reject_readonly_violation`` で再検証し、違反コマンドは引き当てを
        捨てて通常フローへ落とす (実行段の readonly ラッパでも二重に弾かれる
        が、ここで捨てれば synth 等の後続層が正当なコマンドを合成し直せる)。

        Returns:
            引き当て成立時は ``ToolJudgement(<exec_tool>, {"command": ...})``、
            それ以外は ``None`` (通常の判定フローに falling-through する)。
        """
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        if not exec_tool:
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        recalled = await self._try_recall_executable_command(query, mode=mode)
        if not recalled:
            return None
        if self._reject_readonly(exec_tool, recalled):
            return None
        logger.info(
            "Executable command recall: matched command for query=%s",
            query[:50],
        )
        return ToolJudgement(
            tool_needed=True,
            tool_name=exec_tool,
            tool_args={"command": recalled},
            # "rule" と区別する。recall 由来の実行を curator が再学習すると
            # 「誤発火 → 成功記録 → fact 延命 → また誤発火」で自己強化するため、
            # sleep 側 (executable_command_curator) がこの source を見て除外する。
            source="recall",
        )

    async def _try_recall_executable_command(
        self, query: str, mode: str = "create",
    ) -> str | None:
        """過去成功した run_command を SemMem から類似クエリで引き当てる.

        ``_try_recall_url`` と対称。条件:
          - ``mem_view`` / ``embedder`` が両方提供されている
          - ``tools.executable_command_recall_enabled`` が True
          - top-K 候補のうち ``world_fact`` で subject prefix が
            ``mem.world.executable_command.`` のもの
          - 類似度 >= ``executable_command_recall_min_score``
          - 過去成功率 (``_extra.success_avg``) >=
            ``executable_command_recall_min_record_score``
          - profile_id が一致する

        Returns:
            条件を満たす最良のコマンド文字列。なければ ``None``。
        """
        tools_cfg = (self._config or {}).get("tools") or {}
        if not bool(tools_cfg.get("executable_command_recall_enabled", True)):
            return None
        if self._mem_view is None or self._embedder is None:
            return None
        if not query:
            return None
        # ``_try_recall_url`` と同じ理由で embed_query (LRU キャッシュ経路) を使う。
        try:
            embedding = await self._embedder.embed_query(query, mode=mode)
        except Exception as exc:
            logger.warning("Executable command recall: embed failed: %s", exc)
            return None
        if embedding is None or len(embedding) == 0:
            return None
        import numpy as _np
        q_vec = _np.asarray(embedding, dtype=_np.float32)

        top_k = int(tools_cfg.get("executable_command_recall_topk", 5))
        min_sim = float(tools_cfg.get("executable_command_recall_min_score", 0.7))
        min_avg = float(
            tools_cfg.get("executable_command_recall_min_record_score", 0.6),
        )
        ttl_days = int(tools_cfg.get("executable_command_recall_ttl_days", 30))
        ttl_seconds = float(ttl_days) * 86400.0 if ttl_days > 0 else 0.0
        try:
            candidates = self._mem_view.search_by_embedding(q_vec, top_k=top_k)
        except Exception as exc:
            logger.warning(
                "Executable command recall: search_by_embedding failed: %s", exc,
            )
            return None

        # 候補プールが小さいうちは top-K も success_avg も選別として機能しないため
        # (実測 2026-07-25: executable_command fact は global に 1 件のみで、
        # 類似度ゲートだけが唯一のフィルタだった)、閾値を嵩上げして保守的に倒す。
        if len(candidates) < _RECALL_SMALL_POOL_SIZE:
            min_sim += _RECALL_SMALL_POOL_MARGIN

        import time as _time
        now = _time.time()

        best_sim = candidates[0][1] if candidates else 0.0
        self._last_recall_diag = {
            "candidates": len(candidates),
            "best_sim": round(float(best_sim), 4),
            "min_sim": round(min_sim, 4),
            "min_avg": min_avg,
        }

        for fact, sim in candidates:
            if fact.type != "world_fact":
                continue
            if not fact.subject.startswith("mem.world.executable_command."):
                continue
            if sim < min_sim:
                # candidates は score 降順想定。閾値未満は以降全て無効。
                break
            if fact.profile_id and fact.profile_id != self._profile_id:
                continue
            extra = fact._extra or {}
            command = extra.get("command")
            success_avg = float(extra.get("success_avg") or 0.0)

            # 鮮度ペナルティ: TTL 超過なら success_avg を半減して閾値判定。
            effective_score = success_avg
            if ttl_seconds > 0.0:
                last_exec = float(extra.get("last_executed_at") or 0.0)
                if last_exec > 0.0:
                    age_sec = now - last_exec
                    if age_sec > ttl_seconds:
                        effective_score = success_avg * 0.5
                        logger.warning(
                            "Executable command recall: TTL exceeded "
                            "(age=%.0fd, ttl=%dd), penalize %.3f -> %.3f",
                            age_sec / 86400.0, ttl_days,
                            success_avg, effective_score,
                        )

            if command and not recalled_command_fits_query(
                str(command), str(fact.object or ""), query,
            ):
                logger.info(
                    "Executable command recall rejected: query-specific "
                    "literals missing (sim=%.4f subject=%s origin=%s)",
                    sim, fact.subject, str(fact.object or "")[:60],
                )
                self._last_recall_diag["rejected"] = "literal_mismatch"
                continue

            if command and effective_score >= min_avg:
                # URL リコール側と同粒度の観測。embed モデルを差し替えるたびに
                # sim 分布が動くため、これが無いと閾値較正が事後検証できない。
                logger.info(
                    "Executable command recall matched: sim=%.4f min_sim=%.4f "
                    "success_avg=%.3f effective=%.3f min_record=%.3f "
                    "candidates=%d subject=%s",
                    sim, min_sim, success_avg, effective_score, min_avg,
                    len(candidates), fact.subject,
                )
                self._last_recall_diag.update({
                    "sim": round(float(sim), 4),
                    "success_avg": round(success_avg, 3),
                    "effective_score": round(effective_score, 3),
                    "subject": fact.subject,
                })
                return str(command)
        logger.debug(
            "Executable command recall miss: candidates=%d best_sim=%.4f "
            "min_sim=%.4f query=%s",
            len(candidates), float(best_sim), min_sim, query[:50],
        )
        return None

    def _log_tool_decision(
        self, result: "ToolJudgement", reason: str,
    ) -> None:
        """

        chosen は ``rule`` / ``cartridge`` / ``learned`` / ``aux`` /
        ``no_tool`` のいずれかで、4 段階フォールバックのどの層で決着したかを
        identify する。``evolve`` レベル限定で実発火、それ以外は no-op。
        """
        if self._debug_logger is None:
            return
        chosen = "no_tool" if not result.tool_needed else (result.source or "rule")
        context: dict[str, Any] = {
            "tool_needed": bool(result.tool_needed),
            "tool_name": getattr(result, "tool_name", None) or "",
        }
        command = (getattr(result, "tool_args", None) or {}).get("command")
        if command:
            context["command"] = str(command)[:120]
        # 層0.5 の採否は類似度が唯一の根拠なので、decision.jsonl だけで
        # 閾値較正を検証できるよう診断値を載せる。
        if reason.startswith("executable_command_recall") and self._last_recall_diag:
            context.update(self._last_recall_diag)
        self._debug_logger.log_decision(
            decision_point="tool_call_decision",
            chosen=chosen,
            candidates=["rule", "cartridge", "learned", "llm", "recall", "no_tool"],
            reason=reason,
            context=context,
            scope="request",
        )

    async def _resolve_executable_command(
        self, query: str, readonly: bool = False,  # noqa: ARG002 - 面の互換
    ) -> str:
        """executable command をルール表 (regex) から解決する.

        Returns:
            実行可能と判定された場合のコマンド文字列。それ以外は ``""``。
        """
        return _infer_executable_command(query)

    def _judge_with_rules(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> ToolJudgement:
        """ルールベースでツール呼び出しを判定（フォールバック）

        パターンマッチした場合、クエリの内容からツール名と引数を推定する。
        知識質問（RAG で処理すべき）はツール不要と判定する。
        """
        # 明示的に書かれた算術式は calculate で決定論的に評価する。
        # 知識質問判定より前に置く: 「1234 × 5678 はいくつですか？」は
        # 「〜ですか」で知識質問にマッチし、ツール無しで base の暗算に落ちて
        # 誤答するため (実インシデント 2026-07-27 ライブ検証)。
        expression = _extract_arithmetic_expression(query)
        if expression and tools_registry.has("calculate"):
            logger.debug("Rule-based: arithmetic expression detected: %s", expression)
            return ToolJudgement(
                tool_needed=True,
                tool_name="calculate",
                tool_args={"expression": expression},
                source="rule",
            )

        # ディレクトリ列挙は決定論で解決する。対象が実在するときだけ発火し、
        # 解決できなければシグナルだけ立てて後段へ委ねる (当てずっぽうの引数で
        # 撃たない)。知識質問判定より前に置くのは算術式と同じ理由で、
        # 「〜には何がありますか」が知識質問にマッチして base の想像に落ちるため。
        if asks_directory_listing(query) and tools_registry.has("list_directory"):
            directory = resolve_listing_directory(query, get_project_root())
            if directory is not None:
                logger.debug(
                    "Rule-based: directory listing detected: %s", directory,
                )
                return ToolJudgement(
                    tool_needed=True,
                    tool_name="list_directory",
                    tool_args={"directory": directory},
                    source="rule",
                )

        # 知識質問はツール不要（RAG パイプラインで処理）
        # ただしツールパターン・ファイルパス・URL にもマッチするクエリは
        # ツール操作の可能性が高いため知識質問判定を適用しない
        has_tool_signal = _query_has_tool_signal(query)
        knowledge_patterns = select_locale_variant(_KNOWLEDGE_PATTERNS, _KNOWLEDGE_PATTERNS_EN)
        if not has_tool_signal and any(p.search(query) for p in knowledge_patterns):
            logger.debug("Rule-based: knowledge query detected, skipping tool: %s", query[:50])
            return ToolJudgement(tool_needed=False, source="rule")

        # 明示パス / URL があり、かつ具体的なツールまで決定論で解決できるなら
        # ``_TOOL_PATTERNS`` に無い言い回しでもここで確定させる。パス付きの依頼が
        # aux 層任せになっており、同じ依頼が read_file / search_history /
        # ツール未発火に割れていた (実インシデント 2026-08-04 ライブ監査:
        # 「E:/tmp/a.txt の中身を見せて、あわせて文字数も教えてください。」で
        # ツールが 1 つも走らず「存在しない」と誤答)。``_infer_tool`` は読み書きの
        # 動詞が無ければ空を返すので、パスに言及しただけの文は従来経路へ落ちる。
        if has_tool_signal:
            signal_name, signal_args = self._infer_tool(query, tools_registry, mode)
            if signal_name:
                logger.debug(
                    "Rule-based: path/URL signal resolved to %s: %s",
                    signal_name, query[:50],
                )
                return ToolJudgement(
                    tool_needed=True,
                    tool_name=signal_name,
                    tool_args=signal_args,
                    source="rule",
                )

        tool_patterns = select_locale_variant(_TOOL_PATTERNS, _TOOL_PATTERNS_EN)
        if not any(p.search(query) for p in tool_patterns):
            return ToolJudgement(tool_needed=False, source="rule")

        logger.debug("Rule-based: tool pattern matched for query: %s", query[:50])

        # ツール名と引数の推定
        tool_name, tool_args = self._infer_tool(query, tools_registry, mode)

        return ToolJudgement(
            tool_needed=True,
            tool_name=tool_name,
            tool_args=tool_args,
            source="rule",
        )

    def _judge_with_cartridge_hints(
        self,
        query: str,
        tools_registry: ToolsRegistry,
    ) -> ToolJudgement:
        """ロード済みカートリッジの tool_hints でツール呼び出しを判定

        カートリッジが定義するキーワードリストとクエリを照合し、
        マッチした場合は指定ツールへの誘導を返す。
        知識質問パターンに関係なく、カートリッジが「この分野はツール実行が有益」と
        宣言するためルールベースの知識質問判定より優先される。
        """
        if self._cartridge_manager is None:
            return ToolJudgement(tool_needed=False, source="rule")

        hints = self._cartridge_manager.get_tool_hints()
        if not hints:
            return ToolJudgement(tool_needed=False, source="rule")

        q_lower = query.lower()
        for hint in hints:
            patterns = hint.get("patterns", [])
            tool = hint.get("tool", "")
            if not tool or not patterns:
                continue
            for pattern in patterns:
                if pattern.lower() in q_lower:
                    if tools_registry.has(tool):
                        logger.debug(
                            "Cartridge hint matched: pattern=%s, tool=%s, query=%s",
                            pattern, tool, query[:50],
                        )
                        return ToolJudgement(
                            tool_needed=True,
                            tool_name=tool,
                            tool_args={},
                            source="cartridge",
                        )
                    break

        return ToolJudgement(tool_needed=False, source="rule")

    def _judge_with_learned_patterns(
        self,
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> ToolJudgement:
        """学習済み tool_routing パターンでツール呼び出しを判定

        LearnedPatternStore の tool_routing カテゴリのパターンを参照し、
        閾値以上の重みでマッチした場合にツール実行を誘導する。
        """
        if self._learned_patterns is None:
            return ToolJudgement(tool_needed=False, source="rule")

        matches = self._learned_patterns.match(query, category="tool_routing")
        if not matches:
            return ToolJudgement(tool_needed=False, source="rule")

        top_weight = matches[0][1]
        if top_weight < self._tool_routing_threshold:
            return ToolJudgement(tool_needed=False, source="rule")

        logger.debug(
            "Learned tool_routing pattern matched: top=%s (w=%.3f), query=%s",
            matches[0][0], top_weight, query[:50],
        )

        # ツール名と引数を推定（静的パターンと同じロジック）
        tool_name, tool_args = self._infer_tool(query, tools_registry, mode)
        if not tool_name:
            # _infer_tool が推定できない場合の run_command フォールバックは
            # 「引数なしの仮判定」であり、judge() 側で
            # ルール層が command を解決できた場合
            # のみ生き残る。合成不成立 (aux が not executable と判定 /
            # aux 未接続) なら _suppress_commandless_run_command が
            # no_tool に倒す — 学習パターンの字句マッチだけを根拠に実行不能な
            # run_command を返さない (2026-07-20: 学習済み「説明」が知識質問
            # にマッチし create モードで誤発火し得た件の防衛線)。
            tool_name = "run_command" if tools_registry.has("run_command") else ""
        if not tool_name:
            return ToolJudgement(tool_needed=False, source="rule")

        return ToolJudgement(
            tool_needed=True,
            tool_name=tool_name,
            tool_args=tool_args,
            source="learned",
        )

    @staticmethod
    def _infer_tool(
        query: str,
        tools_registry: ToolsRegistry,
        mode: str,
    ) -> tuple[str, dict]:
        """クエリからツール名と引数を推定する

        Returns:
            (tool_name, tool_args): 推定結果。推定できない場合は ("", {})。
        """
        q = query.lower()
        # ファイルパス抽出用のクエリ。バッククォート内コマンドの引数パスは
        # 読み書きの対象ではないため取り除く (コマンド実行分岐は生の ``query``
        # を見るのでコマンド自体は失われない)。これが無いと「コマンド
        # `dir E:\tmp\x` を実行して出力を報告して」の「出力」が下の書込み
        # パターンに一致し、コマンド引数のパスが write_file の書込み先になる。
        path_query = strip_command_literals(query)

        # URL フェッチパターン（他のパターンより優先）
        # URL を含むクエリは fetch_url で処理する（run_command + curl に落ちるのを防止）
        url_match = _URL_IN_QUERY_RE.search(query)
        if url_match and tools_registry.has("fetch_url"):
            return "fetch_url", {"url": url_match.group(1)}
        if re.search(
            r"(?:フェッチ|fetch|取得して|アクセス|ウェブ|web|サイト|site|ページ|page"
            r"|ニュース|news|ブラウズ|browse)",
            q,
        ) and tools_registry.has("fetch_url"):
            # URL がクエリに含まれていないが、フェッチ意図がある場合
            return "fetch_url", {}

        # コード検証パターン（read_file より優先）
        # 「動作する？」「正しく動く？」等はファイルを読むより構文チェックの方が確実
        if re.search(
            r"(?:動作|動[くい]|実行でき|エラー|バグ|正常|正しく動)"
            r"|(?:work|run correctly|execute|error|bug)",
            q,
        ):
            path = _extract_file_path(query)
            if path and path.endswith(".py") and tools_registry.has("verify_syntax"):
                return "verify_syntax", {"file_path": path}

        # ファイル読込みパターン
        # 「確認」「チェック」「見せて」「内容」等は実質的にファイル読み取りを必要とする。
        # カタカナ「チェック」は日本語で頻出するため明示的に含める (ASCII "check" だけ
        # ではカタカナ表記を取りこぼし、後続の write パターンへ誤って落ちる)。
        # ファイルの行数・文字数を問う質問も読み取りが要る。モデルは本文から
        # 数えても正確にならないため read_file のメタ行 (lines / chars) を
        # 使わせる (実測 2026-08-05: ツール未発火のまま「確認できません」と
        # 回答放棄した)。パス抽出済みの分岐なので誤爆はファイル参照時に限る。
        # 有無を問う語 (存在し / ありますか / exists ...) も読み取り側に含める。
        # パス抽出済みの分岐なので「ファイルの話」であることは確定しており、
        # これが無いと「E:\tmp\a.txt はまだありますか」がツール未発火のまま
        # base の記憶で断定される (2026-08-16 監査の「存在しますか」形とは
        # 語尾違いで挙動が割れていた)。
        if re.search(
            r"(?:読[みむ]込|読んで|開いて|見せて|見て|確認|チェック|確かめ"
            r"|正し[いく]|合って|内容|中身|何文字|文字数|何行|行数"
            r"|存在し|ありますか|あるか|残ってい|消えてい"
            r"|read|show|check|verify|correct|content|view|exists?"
            r"|how many (?:characters|chars|lines))",
            q,
        ):
            path = _extract_file_path(path_query)
            if path:
                # ディレクトリ指定 (配下のファイルを点検する文脈) は read_file だと
                # "Not a file" になるため list_directory に振り分ける。
                if Path(path).is_dir() and tools_registry.has("list_directory"):
                    return "list_directory", {"directory": path}
                if tools_registry.has("read_file"):
                    args: dict = {"file_path": path}
                    head_lines = _extract_head_line_count(query)
                    if head_lines is not None:
                        args["start_line"] = 1
                        args["end_line"] = head_lines
                    elif asks_file_existence_only(query):
                        # 有無だけを問う質問に全文を渡すとモデルが全文を復唱する。
                        # メタ行 (lines / chars) だけで答えられるので 1 行に絞る。
                        args["start_line"] = 1
                        args["end_line"] = 1
                    return "read_file", args

        # ファイル書き込み/出力パターン
        # ディレクトリを書込み先に取ると write_file が配下に output_<UTC>.txt を
        # 捏造する (記述的な「出力」誤マッチで read 指示がここへ落ちるケースを含む)。
        # ディレクトリは書込み対象から除外する。
        if re.search(r"(?:書[きく]込|書いて|出力|保存|生成|作成|write|save|output)", q):
            path = _extract_file_path(path_query)
            if path and not Path(path).is_dir() and tools_registry.has("write_file"):
                return "write_file", {"file_path": path}

        # コマンド実行パターン
        # "run" は ASCII 境界必須 ("running" 等の語幹への部分一致誤爆対策。
        # "What OS am I running?" が意図せずここで確定していた実績あり)。
        # "exec" は execute/executing/executed の活用形も拾う必要があるため、
        # 境界を活用語尾まで含めた明示形にする (単純な境界だと "execute" を
        # 取りこぼす)。
        if re.search(
            r"(?:実行|(?<![A-Za-z])run(?![A-Za-z])"
            r"|(?<![A-Za-z])exec(?:ute[sd]?|uting)?(?![A-Za-z]))",
            q,
        ) and tools_registry.has("run_command"):
            # ファイルパスがあれば python で実行
            path = _extract_file_path(query)
            if path and path.endswith(".py"):
                return "run_command", {"command": f'python "{path}"'}
            # バッククォート内のコマンド
            cmd_match = re.search(r'`([^`]+)`', query)
            if cmd_match:
                return "run_command", {"command": cmd_match.group(1)}
            return "run_command", {}

        # コード検索パターン (_TOOL_PATTERNS と同一の共起ガード _CODE_SEARCH_PATTERNS
        # を再利用。汎用「検索」単独は "Binary Search Tree" のような英語クラス名
        # の部分一致にも誤爆するため、コード/ファイル文脈語との共起を要求する
        # — 2026-07-22 監査で判明。裸の検索語だけで抽出パターンが確定し、
        # 無関係ファイルへの search_code 誤発火を招いていた)
        # 所在を問う言い回し (「<識別子> はどこで使われていますか」)。
        # _CODE_SEARCH_PATTERNS は「コード/ファイル語 × 検索動詞」の共起を
        # 要求するため、この言い方は**どちらの語も含まず**ルール層を素通りする。
        #
        # より一般的な _CODE_SEARCH_PATTERNS より **先に** 判定する: 所在質問は
        # 疑問詞が骨組みなので、汎用の _extract_search_pattern だと疑問詞自体を
        # 検索語に採ってしまう ("where is search_code used?" → pattern="where")。
        if (_is_code_usage_location_query(query)
                and tools_registry.is_available("search_code", mode)):
            pattern = _code_usage_location_pattern(query)
            if pattern:
                return "search_code", {"pattern": pattern}

        if (any(p.search(query) for p in _CODE_SEARCH_PATTERNS)
                and tools_registry.has("search_code")):
            # クエリからキーワードを抽出して pattern 引数に設定
            pattern = _extract_search_pattern(query)
            if pattern:
                return "search_code", {"pattern": pattern}
            return "search_code", {}

        # Python 実行可能クエリ（システム情報・数値処理・データ処理・変換）
        # これらのクエリは Python コード生成 → run_command で正確に回答できる。
        # ツール名は mode から解決する (chat は run_command_readonly)。
        exec_tool = _executable_tool_for_mode(tools_registry, mode)
        exec_query_re = select_locale_variant(_INFER_TOOL_EXEC_QUERY_RE, _INFER_TOOL_EXEC_QUERY_RE_EN)
        if (
            exec_query_re.search(q)
            and exec_tool
            and not asks_about_prior_conversation_entity(query)
        ):
            # システム情報クエリは具体的なコマンドを生成
            command = _infer_executable_command(query)
            if command:
                return exec_tool, {"command": command}
            return exec_tool, {}

        # 計算パターン (実行可能クエリ分岐より後ろに置く。「フィボナッチ数列の
        # 10番目を計算して」のように両方にマッチするクエリは、式抽出を持たない
        # calculate {} で潰さず run_command 合成経路 (aux synth) に乗せる。
        # 2026-07-21 ライブ検証 ターン35 のインシデント対策)
        if re.search(r"(?:計算|calculate)", q) and tools_registry.has("calculate"):
            return "calculate", {}

        return "", {}

    def _parse_response(self, content: str) -> ToolJudgement:
        """補助タスクの応答をパースして ToolJudgement に変換

        ``response_format=json_schema`` 制約サンプリングが効いている場合は
        ``{"tool": "...", "args": {...}}`` 形式の有効な JSON が必ず返る。
        フラグ無効化時 / 古い llama-server build / max_tokens 切断時の
        フォールバックとして共通実装 ``extract_json_object`` を経由する。
        """
        # 共通 JSON 抽出経路
        data = extract_json_object(content)
        if isinstance(data, dict):
            return _json_to_judgement(data)

        # JSON が抽出できないケースはツール不要と判定 (安全側)
        logger.warning(
            "Could not parse LLM response for tool judgement: %s",
            content[:100],
        )
        return ToolJudgement(tool_needed=False, source="llm")


def _json_to_judgement(data: dict) -> ToolJudgement:
    """JSON dict を ToolJudgement に変換

    補助タスク応答は ``response_format`` 無効 / 古い llama-server / max_tokens 切断
    時に ``json_repair`` で機械修復されるため、``tool`` / ``args`` が非想定型
    (list / str 等) になりうる。``ToolJudgement.tool_args`` は dict 契約なので、
    下流 (``deliberative._execute_tool`` の ``dict(tool_args)`` 等) が落ちないよう
    ここで強制正規化する。
    """
    tool = data.get("tool", "")
    if not isinstance(tool, str):
        tool = ""
    if not tool or tool == "no_tool":
        return ToolJudgement(tool_needed=False, source="llm")
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return ToolJudgement(
        tool_needed=True,
        tool_name=tool,
        tool_args=args,
        source="llm",
    )
