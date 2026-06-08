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

# 外部 fetch 意図パターン (確実シグナルのみ)
# URL を含む / 明示的 fetch 動詞は 100% fetch 意図とみなし RAG をスキップ。
# リアルタイムキーワード (ニュース / 株価 / 天気 等) はアシスト判定に委譲し、
# 固定キーワードリストの陳腐化を避ける (Phase 2 で learned_pattern 化予定)。
FETCH_INTENT_PATTERNS = re.compile(
    r"(https?://"
    r"|フェッチ|fetch"
    r"|アクセスして|アクセスし"
    r"|取得して|取得しなおして|取り直して|再取得"
    r"|ブラウズ|browse|ダウンロード|download)",
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

# 自明な質問パターン (RAG 不要の即 skip 確定)
# 時刻 / 日付 / 曜日 / 自己同一性 / 簡単な雑談を捕捉する。`FORCE_PATTERNS`
# (教えて / 調べて等) より優先順位が低いため、「教えて、今は何時？」のような
# 知識要求を伴うクエリは retrieve に倒れる。
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
    r"|元気？)",
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
# 一貫させた語彙 (docs/f_03_agent_engine.md §1.2 参照)。
CODE_DOC_GEN_INTENT_PATTERNS = re.compile(
    r"(?:作成|実装|生成|書いて|書く|出力|"
    r"仕様書|設計書|要件定義|計画書|手順書|README|"
    r"create|implement|generate|write)",
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

        # 1. 短すぎるクエリはスキップ
        if len(query_stripped) < 3:
            logger.debug("Necessity: skip (query too short: %d chars)", len(query_stripped))
            return "skip"

        # 2. 外部 fetch 意図 (確定): URL を含む or 明示的 fetch 動詞
        if FETCH_INTENT_PATTERNS.search(query_stripped):
            logger.debug(
                "Necessity: fetch (fetch intent pattern matched: %r)",
                query_stripped[:50],
            )
            return "fetch"

        # 3. 自明な質問 (時刻 / 日付 / 自己同一性 / 雑談): 即 skip 確定
        if TRIVIAL_QUESTION_PATTERNS.search(query_stripped):
            logger.debug(
                "Necessity: skip (trivial question pattern matched: %r)",
                query_stripped[:50],
            )
            return "skip"

        # 4. 質問マーカー (SKIP より優先)
        # 「発売日はいつ」が SKIP の "はい" substring にヒットして誤って
        # skip されないよう、QUESTION を SKIP より先に評価する。
        if QUESTION_PATTERNS.search(query_stripped):
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

        # 4.5. ファイル参照 + コード/ドキュメント生成意図 → skip
        # 例: "C:\path\spec.txt を参照してテトリスを Python で作成"
        # ユーザが提示したファイルを文脈とする新規生成タスクで、
        # local KB (SemMem / 履歴) には引き当てるべき情報がない。
        # この時点で RAG 全工程 (assist 判定 + embedding + LTM + rerank) を
        # 早期 skip して 10 秒以上のレイテンシを排除する。
        if FILE_PATH_PATTERN.search(query_stripped) and \
                CODE_DOC_GEN_INTENT_PATTERNS.search(query_stripped):
            logger.debug(
                "Necessity: skip (file path + code/doc-gen intent: %r)",
                query_stripped[:60],
            )
            return "skip"

        # 5. スキップパターン (挨拶/相槌) — 短文のみ
        if SKIP_PATTERNS.search(query_stripped) and len(query_stripped) < 20:
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

        # assist_client が無ければ degraded mode → 安全側 retrieve
        if assist_client is None:
            logger.debug("Necessity assist: skipped (assist_client is None)")
            return "retrieve"

        # tracker による発火上限チェック (本機能の quality キーは "uncertain")
        if tracker is not None:
            decision = tracker.check(
                session_id=session_id,
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

        timeout_s = float(cfg.get("timeout_s", 5.0))
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
            result = await asyncio.wait_for(
                assist_client.generate_json(
                    prompt,
                    max_tokens=32,
                    temperature=0.0,
                    purpose="retrieval_necessity_judge",
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("Necessity assist: timeout after %.1fs", timeout_s)
            if debug_logger is not None:
                debug_logger.log_decision(
                    decision_point="self_rag_necessity_path",
                    chosen="retrieve",
                    candidates=["retrieve", "fetch", "skip"],
                    reason="assist_timeout",
                    scope="request",
                )
            return "retrieve"
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
            tracker.record(session_id)
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
