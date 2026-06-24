"""フィードバック収集: 暗黙的シグナルの検出と経験バッファへの記録

学習済みパターンストアと連携し、訂正・言い直し検出時に
会話から新しいパターンキーワードを自動学習する。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from backend.free.learning.level0_instant import (
    RESPONSE_FULL_CAP,
    RESPONSE_SUMMARY_CAP,
    ExperienceBuffer,
    ExperienceEntry,
    FeedbackSignals,
    truncate_at_boundary,
)
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.learned_patterns import LearnedPatternStore

logger = get_logger("agent.feedback")

# ユーザー訂正パターン（ハードコード: 高確度）
CORRECTION_PATTERNS = [
    re.compile(r"違[うわえおっく]|違い(?:ます|ません|まし)", re.IGNORECASE),
    re.compile(r"間違[いっ]", re.IGNORECASE),
    re.compile(r"そうじゃ", re.IGNORECASE),
    re.compile(r"そうではな", re.IGNORECASE),
    re.compile(r"正しくは", re.IGNORECASE),
    re.compile(r"訂正", re.IGNORECASE),
    re.compile(r"not correct", re.IGNORECASE),
    re.compile(r"that'?s wrong", re.IGNORECASE),
    re.compile(r"^\s*actually\b", re.IGNORECASE),
]

# 言い換えパターン（同じ質問の言い直し検出用）
REPHRASE_THRESHOLD = 0.5  # 類似度閾値


class FeedbackCollector:
    """暗黙的フィードバックシグナルを収集し経験バッファに記録

    学習済みパターンストアが設定されている場合、訂正・言い直し検出時に
    前回のクエリからキーワードを抽出してパターンとして学習する。
    """

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        debug_logger: DebugLogger | None = None,
        learned_patterns: LearnedPatternStore | None = None,
        disabled: bool = False,
        base_model_name: str = "",
        embedding_model_name: str = "",
    ) -> None:
        self.buffer = experience_buffer
        self._debug_logger = debug_logger
        self._learned_patterns = learned_patterns
        self._prev_query: str | None = None
        # 直前ターンの entry と capability 使用状況 (false_negative の事後検出用)。
        # 「前ターンが capability 未使用 → 当ターンで明示訂正 → 当ターンで capability
        # 使用」の遷移を検出したら前 entry へ遡及マークし、前クエリから学習する。
        self._prev_entry: ExperienceEntry | None = None
        self._prev_routed_tool: bool = False
        self._prev_used_long_form: bool = False
        # 現在ロード中のモデル名 (GGUF ファイル名)。record() の base_model /
        # embedding_model が明示指定されないとき既定値として埋める。
        self._base_model_name = base_model_name
        self._embedding_model_name = embedding_model_name
        # 現会話セッションで record した entry の参照。会話終了時に
        # mark_conversation_ended() がまとめて conversation_ended=True にする。
        self._session_entries: list[ExperienceEntry] = []
        # 自己学習無効化フラグ (--no-learning 経由)。True の場合 record() は
        # シグナル検出も ExperienceBuffer 書込も行わずダミーの ExperienceEntry を返す
        self._disabled = disabled
        if disabled:
            logger.info(
                "FeedbackCollector initialized in disabled mode "
                "(Level 0 experience record is no-op)",
            )

    def record(
        self,
        query: str,
        response: str,
        mode: str = "chat",
        rag_used: bool = False,
        rag_source: str | None = None,
        rag_top1_score: float | None = None,
        agent_loops: int = 0,
        cartridge_ids: list[str] | None = None,
        base_model: str = "",
        embedding_model: str = "",
        long_form_used: bool = False,
        long_form_content_type: str | None = None,
        long_form_strategy: str | None = None,
        long_form_units_total: int = 0,
        long_form_units_completed: int = 0,
        long_form_validation_errors: int = 0,
        long_form_budget_used_pct: float | None = None,
        tool_routing_success: bool = False,
        tool_routing_false_positive: bool = False,
        tool_routing_false_negative: bool = False,
        long_form_success: bool = False,
        long_form_false_positive: bool = False,
        long_form_false_negative: bool = False,
        step_credits: list[dict] | None = None,
    ) -> ExperienceEntry:
        """シグナル収集 → ExperienceBuffer に記録"""
        if self._disabled:
            # 学習無効化中: シグナル検出 / パターン学習 / バッファ書込を全てスキップ。
            # 呼出側 (chat_recorder) は戻り値を直接参照しないが、署名互換のため
            # 最小限のダミーエントリを返す
            return ExperienceEntry(
                timestamp=utc_now(),
                mode=mode,
                query=query,
                response_summary=response[:RESPONSE_SUMMARY_CAP],
                response_full=truncate_at_boundary(response, RESPONSE_FULL_CAP),
                base_model=base_model or self._base_model_name,
                embedding_model=embedding_model or self._embedding_model_name,
                cartridge_ids=cartridge_ids or [],
                signals=FeedbackSignals(),
            )
        correction_text, detected_by = self._detect_correction(query)

        signals = FeedbackSignals(
            rephrased_query=self._detect_rephrase(query),
            rag_used=rag_used,
            rag_source=rag_source,
            rag_top1_score=rag_top1_score,
            agent_loops=agent_loops,
            user_correction=correction_text,
            correction_detected_by=detected_by,
            long_form_used=long_form_used,
            long_form_content_type=long_form_content_type,
            long_form_strategy=long_form_strategy,
            long_form_units_total=long_form_units_total,
            long_form_units_completed=long_form_units_completed,
            long_form_validation_errors=long_form_validation_errors,
            long_form_budget_used_pct=long_form_budget_used_pct,
            tool_routing_success=tool_routing_success,
            tool_routing_false_positive=tool_routing_false_positive,
            tool_routing_false_negative=tool_routing_false_negative,
            long_form_success=long_form_success,
            long_form_false_positive=long_form_false_positive,
            long_form_false_negative=long_form_false_negative,
            step_credits=step_credits or [],
        )

        entry = ExperienceEntry(
            timestamp=utc_now(),
            mode=mode,
            query=query,
            response_summary=response[:RESPONSE_SUMMARY_CAP],
            response_full=truncate_at_boundary(response, RESPONSE_FULL_CAP),
            base_model=base_model or self._base_model_name,
            embedding_model=embedding_model or self._embedding_model_name,
            cartridge_ids=cartridge_ids or [],
            signals=signals,
        )

        # false_negative 事後検出 (訂正ゲート + capability 弁別):
        # 「前ターンが capability 未使用 → 当ターンで明示訂正 → 当ターンで capability
        # 使用」= 前ターンは capability を要すべきだった、という低ノイズの強い証拠。
        # 前 entry へ遡及マーク (Level 1 バッチ消費側が前クエリから学習: 正しい帰属) し、
        # 即時にも前クエリから学習する。前ターンが既に capability を使っていたケースは
        # 同一ターン false_positive (chat_recorder で検出) の領分なので扱わない。
        # 訂正学習 (_learn_from_correction) より先に処理する: パターンは keyword 単位
        # キーイング (add_pattern) のため、より具体的な tool_routing/long_form の主張を
        # 一般的な correction カテゴリより優先させ shadow を防ぐ。
        current_routed_tool = tool_routing_success or tool_routing_false_positive
        if correction_text is not None and self._prev_entry is not None:
            if current_routed_tool and not self._prev_routed_tool:
                self._prev_entry.signals.tool_routing_false_negative = True
                self._learn_tool_routing_from_false_negative(self._prev_entry.query)
            if long_form_used and not self._prev_used_long_form:
                self._prev_entry.signals.long_form_false_negative = True
                self._learn_long_form_from_signal(self._prev_entry.query)

        # 訂正・言い直し検出時にパターンを学習
        if correction_text is not None:
            self._learn_from_correction(query)
        if signals.rephrased_query:
            self._learn_from_rephrase(query)

        # ツールルーティング false_negative 時: 明示注入 (テスト等) では当該クエリから学習。
        if tool_routing_false_negative:
            self._learn_tool_routing_from_false_negative(query)

        # 長文ルーティング 成功 / false_negative 時: クエリからキーワードを
        # ``category="long_form"`` として学習する。
        # success 時はクエリ全体が長文分類のヒントになるため正例として学習。
        if long_form_success or long_form_false_negative:
            self._learn_long_form_from_signal(query)

        self.buffer.record(entry)
        self._session_entries.append(entry)
        self._prev_query = query
        self._prev_entry = entry
        self._prev_routed_tool = current_routed_tool
        self._prev_used_long_form = long_form_used

        logger.info(
            "Recorded experience: mode=%s, rephrase=%s, correction=%s (by=%s)",
            mode, signals.rephrased_query,
            signals.user_correction is not None, detected_by,
        )

        # DebugLogger に Level 0 学習サイクルを記録
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(cycle_num=0, data={
                "level": 0,
                "mode": mode,
                "buffer_size": self.buffer.count,
                "rephrase": signals.rephrased_query,
                "correction": signals.user_correction is not None,
                "correction_detected_by": detected_by,
                "rag_used": signals.rag_used,
            })

        return entry

    def mark_conversation_ended(self) -> None:
        """現会話セッションで record した全エントリに conversation_ended を設定

        record() は会話途中の各応答ごとに新規 entry を作るため、会話終了時に
        当該セッションの全 entry へまとめて反映する (approach a)。マーク後は
        セッション参照と直前クエリをリセットし、次会話を新セッション扱いにする。
        buffer ローテーションで切り捨てられた entry も参照経由で安全 (生存 entry のみ
        buffer に効き、切捨て済みは GC 対象)。
        """
        if self._disabled:
            return
        for entry in self._session_entries:
            entry.signals.conversation_ended = True
        self._session_entries.clear()
        self._prev_query = None
        self._prev_entry = None
        self._prev_routed_tool = False
        self._prev_used_long_form = False

    def _detect_rephrase(self, query: str) -> bool:
        """直前の質問との類似度で言い換えを検出（簡易版: 文字重複率）"""
        if self._prev_query is None:
            return False

        # 簡易的な文字重複率
        prev_chars = set(self._prev_query)
        curr_chars = set(query)
        if not prev_chars or not curr_chars:
            return False

        overlap = len(prev_chars & curr_chars)
        union = len(prev_chars | curr_chars)
        similarity = overlap / union if union > 0 else 0

        return similarity > REPHRASE_THRESHOLD

    def _detect_correction(self, query: str) -> tuple[str | None, str | None]:
        """ユーザーの訂正パターンを検出（ハードコード + 学習済み）

        Returns:
            (correction_text, detected_by): 検出テキストと検出元
            detected_by: "hardcoded" | "learned" | None
        """
        # 1. ハードコードパターン（高確度、優先）
        for pattern in CORRECTION_PATTERNS:
            if pattern.search(query):
                return query, "hardcoded"

        # 2. 学習済みパターン (correction カテゴリのみ参照。category 未指定だと
        # long_form / tool_routing / rephrase で学習した語まで横断ヒットし、
        # 一般的な意図語が訂正として誤検知される)
        if self._learned_patterns is not None:
            matches = self._learned_patterns.match(query, category="correction")
            if matches:
                logger.debug(
                    "Learned pattern matched: %s",
                    [(kw, f"{w:.2f}") for kw, w in matches[:3]],
                )
                return query, "learned"

        return None, None

    def _learn_from_correction(self, current_query: str) -> None:
        """訂正検出時: 前回クエリからキーワードを抽出してパターンとして学習

        ユーザーが訂正した場合、前回の指示に含まれる意図キーワード
        （例: 「追記」「削除」「修正」等）を学習する。
        """
        if self._learned_patterns is None or self._prev_query is None:
            return

        keywords = self._learned_patterns.extract_intent_keywords(self._prev_query)
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="correction")

        # 現在のクエリからも意図語を抽出（訂正メッセージ自体にも学習対象あり）
        current_keywords = self._learned_patterns.extract_intent_keywords(current_query)
        for kw in current_keywords:
            self._learned_patterns.add_pattern(kw, category="correction")

        if keywords or current_keywords:
            logger.info(
                "Learned patterns from correction: prev=%s, current=%s",
                json.dumps(keywords[:5], ensure_ascii=False),
                json.dumps(current_keywords[:5], ensure_ascii=False),
            )

    def _learn_tool_routing_from_false_negative(self, query: str) -> None:
        """ツールルーティング false_negative 時: クエリからキーワードを tool_routing として学習

        ツール実行されなかったがユーザーが手動で要求した場合、
        クエリに含まれる意図キーワードを tool_routing カテゴリとして学習する。
        """
        if self._learned_patterns is None:
            return

        keywords = self._learned_patterns.extract_intent_keywords(query)
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="tool_routing")

        if keywords:
            logger.info(
                "Learned tool_routing patterns from false_negative: %s",
                json.dumps(keywords[:5], ensure_ascii=False),
            )

    def _learn_long_form_from_signal(self, query: str) -> None:
        """長文ルーティング success / false_negative 時: クエリからキーワードを学習

        長文分類が成功した、またはユーザが手動で長文を再要求した場合、
        クエリに含まれる意図キーワードを ``category="long_form"`` として学習する。
        ルータの ``_detect_long_form_learned()`` がこの語彙を参照する。
        """
        if self._learned_patterns is None:
            return

        # パス片 / URL 片 / 汎用ファイル操作語は long_form の文書種別シグナルでは
        # ないため学習から除外する (出力先指定の自己学習による誤ルーティング防止)。
        keywords = [
            kw for kw in self._learned_patterns.extract_intent_keywords(query)
            if self._learned_patterns.is_long_form_learnable(kw)
        ]
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="long_form")

        if keywords:
            logger.info(
                "Learned long_form patterns from signal: %s",
                json.dumps(keywords[:5], ensure_ascii=False),
            )

    def _learn_from_rephrase(self, current_query: str) -> None:
        """言い直し検出時: クエリからキーワードを抽出してパターンとして学習

        ユーザーが同じ質問を言い直した場合、元の意図キーワードを
        rephrase カテゴリとして学習する。
        """
        if self._learned_patterns is None or self._prev_query is None:
            return

        keywords = self._learned_patterns.extract_intent_keywords(self._prev_query)
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="rephrase")

        if keywords:
            logger.info(
                "Learned patterns from rephrase: %s",
                json.dumps(keywords[:5], ensure_ascii=False),
            )
