"""フィードバック収集: 暗黙的シグナルの検出と経験バッファへの記録

学習済みパターンストアと連携し、ツールルーティング false_negative 時に
クエリから動作指示語を tool_routing パターンとして自動学習する
(長文ルーティングは success / false_negative 時に long_form パターンを学習)。
訂正・言い直しの検出は決定論 (ハードコード正規表現 / 文字重複率) のみで、
学習パターンは使わない。
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
# learned correction 機構 (旧・層2) は 2026-07-21 に廃止した。学習される語が
# 「訂正の言い回し」ではなく「訂正が起きたときの話題語」だったため偽陽性率
# ~85% (経験 65 件の実測) に達し、Level 1 fitness / critique / few-shot /
# Level 2 cvector 対比ペアの学習信号を汚染していた。以後、訂正語彙の拡充は
# 本リストへのハードコード追加で行う (見逃しは prev_failed / same_target の
# 別層が拾うため、追加は確度の高い表現に限る。「実は」「厳密には」「〜では
# なく」単独は話題導入・比較の一般語法と識別できず見送った実績あり)。
CORRECTION_PATTERNS = [
    re.compile(r"違[うわえおっく]|違い(?:ます|ません|まし)", re.IGNORECASE),
    # 「間違え」(下一段) は旧 [いっ] が取りこぼしていた (「間違えていますよ」)
    re.compile(r"間違[いっえ]", re.IGNORECASE),
    re.compile(r"そうじゃ", re.IGNORECASE),
    re.compile(r"そうではな", re.IGNORECASE),
    re.compile(r"正しくは", re.IGNORECASE),
    re.compile(r"訂正", re.IGNORECASE),
    # 出力値の取り違え指摘 (「値が逆になっていませんか？」— learned 層廃止時の
    # 実データ真陽性から回収。仮定表現「逆になっていたら」は誤検知しないよう
    # 疑問形終端まで要求する)
    re.compile(r"逆になって(?:い)?(?:ません|ない)か", re.IGNORECASE),
    # 成果物未達の報告 + やり直し要求 (2026-07-15 の訂正 2 ターンが
    # どちらも検出漏れした語彙)
    re.compile(r"作られて(?:い)?(?:ない|ません)|できて(?:い)?(?:ない|ません)", re.IGNORECASE),
    re.compile(r"(?:し|やり|作り)直して", re.IGNORECASE),
    re.compile(r"not correct", re.IGNORECASE),
    re.compile(r"that'?s wrong", re.IGNORECASE),
    re.compile(r"^\s*actually\b", re.IGNORECASE),
]

# coding モードの実行結果報告 (2026-07-18: coding 経験に訂正シグナルがほぼ発生
# せず Level 1 fitness が無差別化する一因だった語彙)。「動かない」「エラー」
# 等は一般語彙で他モードの質問・新規依頼 (例:「ホバーしても動かないボタンに
# して」「テストが通らないという話について教えて」) と誤検知しやすいため、
# CORRECTION_PATTERNS には含めず (a) coding モード限定 (b) 直前ターンが存在する
# 場合のみ (訂正対象が無い最初のターンでは新規の質問/依頼である可能性が高い)、
# の 2 条件でゲートする (_detect_correction 参照)。
CODING_FAILURE_REPORT_PATTERNS = [
    re.compile(r"動(?:か|き)(?:ない|ません)", re.IGNORECASE),
    re.compile(r"エラー(?:が出|にな|です)", re.IGNORECASE),
    re.compile(r"テストが(?:通ら|落ち)", re.IGNORECASE),
]
# 上記語彙を含んでいても文末が新規依頼の完結形 (「〜にして」「〜作って」
# 「〜教えて」等) なら報告ではなく仕様/質問の可能性が高いため除外する
# (「ホバーしても動かないボタンにして」「テストが通らないという話について
# 教えて」の誤検知回避)。「〜(やり/し/作り)直して」で終わる依頼は
# CORRECTION_PATTERNS 側で既に検出されるため対象外にする必要はない。
_CODING_FAILURE_REPORT_EXCLUDE_RE = re.compile(
    r"(?:ください|にして|教えて|作って|実装して)[。.！!？?]*\s*$",
)

# 弱い訂正パターン: 単独では新規依頼との区別がつかないため、直前ターンの
# 失敗または同一成果物 (同じ出力先パス) の再指定を伴う場合のみ訂正とみなす。
WEAK_CORRECTION_PATTERNS = [
    re.compile(r"(?:では|じゃ)な[くい]"),
    re.compile(r"また.{0,15}(?:になって|なって)"),
]

# 応答の失敗マーカー (meta_cognitive の最終応答フォーマット "- [failed] ...")
_FAILED_MARKER_RE = re.compile(r"(?:^|\n)\s*-\s*\[failed\]", re.IGNORECASE)
_DONE_MARKER_RE = re.compile(r"(?:^|\n)\s*-\s*\[done\]", re.IGNORECASE)

# クエリ中の明示的な出力先パス (同一成果物の再指定検出用)
_QUERY_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'「」()（）]+")

# 言い換えパターン（同じ質問の言い直し検出用）
REPHRASE_THRESHOLD = 0.5  # 類似度閾値


class FeedbackCollector:
    """暗黙的フィードバックシグナルを収集し経験バッファに記録

    学習済みパターンストアが設定されている場合、ツールルーティング / 長文
    ルーティングのシグナルからパターンを学習する (訂正・言い直しからの学習は
    2026-07-21 に廃止 — ``_detect_correction`` の docstring 参照)。
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
        # 直前ターンの成否 ([failed] マーカー等から導出)。失敗直後のターンは
        # 無条件で訂正候補とみなす (2026-07-15: 訂正 2 ターンが検出漏れ)。
        self._prev_turn_failed: bool = False
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
        turn_outcome = self._derive_turn_outcome(
            response, step_credits,
            tool_routing_false_positive=tool_routing_false_positive,
            long_form_false_positive=long_form_false_positive,
        )
        if turn_outcome == "failed":
            # 失敗ターンの成功シグナルは矛盾なので failed 側に倒す
            # (偽成功が learned_patterns の正例学習 / Level 1 fitness に
            # 伝播した 2026-07-15 の再発防止)。
            tool_routing_success = False
            long_form_success = False

        correction_text, detected_by = self._detect_correction(query, mode=mode)
        # 訂正と言い直しは排他: 訂正が検出されたターンを rephrase として
        # 二重学習しない
        rephrased = (
            correction_text is None and self._detect_rephrase(query)
        )

        signals = FeedbackSignals(
            turn_outcome=turn_outcome,
            rephrased_query=rephrased,
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
        current_routed_tool = tool_routing_success or tool_routing_false_positive
        if correction_text is not None and self._prev_entry is not None:
            if current_routed_tool and not self._prev_routed_tool:
                self._prev_entry.signals.tool_routing_false_negative = True
                self._learn_tool_routing_from_false_negative(self._prev_entry.query)
            if long_form_used and not self._prev_used_long_form:
                self._prev_entry.signals.long_form_false_negative = True
                self._learn_long_form_from_signal(self._prev_entry.query)

        # 訂正・言い直し検出からのパターン学習は行わない (2026-07-21 廃止)。
        # correction: 話題語学習による偽陽性増殖 (_detect_correction の
        # docstring 参照)。rephrase: 書き込み専用の dead カテゴリで、match()
        # の参照箇所が存在しなかった (検出自体は文字重複率ベースで学習不要)。

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
        self._prev_turn_failed = turn_outcome == "failed"

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
        self._prev_turn_failed = False

    @staticmethod
    def _derive_turn_outcome(
        response: str,
        step_credits: list[dict] | None,
        *,
        tool_routing_false_positive: bool = False,
        long_form_false_positive: bool = False,
    ) -> str:
        """ターン成否 ("success" | "partial" | "failed") を決定論導出する。

        SSE 完走 = 成功ではなく、応答本文の [failed] マーカー・step_credits
        全 0・ルーティング false_positive を失敗シグナルとして扱う。
        """
        text = response or ""
        if _FAILED_MARKER_RE.search(text):
            return "partial" if _DONE_MARKER_RE.search(text) else "failed"
        if step_credits and all(
            not (c.get("credit") or 0) for c in step_credits
        ):
            return "failed"
        if tool_routing_false_positive or long_form_false_positive:
            return "failed"
        return "success"

    def _same_target_path(self, query: str) -> bool:
        """直前クエリと同じ明示出力先パスを再指定しているかを判定する。"""
        if self._prev_query is None:
            return False
        prev = _QUERY_PATH_RE.search(self._prev_query)
        curr = _QUERY_PATH_RE.search(query)
        if prev is None or curr is None:
            return False
        return prev.group(0).lower() == curr.group(0).lower()

    def _detect_rephrase(self, query: str) -> bool:
        """直前の質問との類似度で言い換えを検出（簡易版: 文字重複率）

        双方に明示的な出力先パスがあり、それが異なる場合は「類似した別の
        新規依頼」(テンプレ連続依頼等) なので rephrase としない
        (2026-07-15: 31 連続の類似依頼で偽陽性 2 件)。
        """
        if self._prev_query is None:
            return False

        prev_path = _QUERY_PATH_RE.search(self._prev_query)
        curr_path = _QUERY_PATH_RE.search(query)
        if (
            prev_path is not None
            and curr_path is not None
            and prev_path.group(0).lower() != curr_path.group(0).lower()
        ):
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

    def _detect_correction(
        self, query: str, *, mode: str = "chat",
    ) -> tuple[str | None, str | None]:
        """ユーザーの訂正パターンを検出（多段: ハードコード / 直前失敗 /
        同一成果物 + 弱パターン）

        旧・層2 (学習済み correction パターン照合) は 2026-07-21 に廃止した。
        学習語が訂正表現ではなく話題語 (「会話」「質問」「カーディナリティ」等)
        だったため、「正解です。では次の質問です」のような肯定評価+話題転換
        ターンまで訂正と誤検出し (偽陽性率 ~85%、経験 65 件の実測)、その語を
        再学習する自己強化ループで汚染が増殖していた。詳細は
        ``CORRECTION_PATTERNS`` の定義コメント参照。

        Returns:
            (correction_text, detected_by): 検出テキストと検出元
            detected_by: "hardcoded" | "prev_failed" | "same_target" | None
        """
        # 1. ハードコードパターン（高確度、優先）
        for pattern in CORRECTION_PATTERNS:
            if pattern.search(query):
                return query, "hardcoded"

        # 1b. coding モードの実行結果報告。訂正対象 (直前ターン) が無い最初の
        # ターンでは新規の質問/依頼である可能性が高く、文末が新規依頼の完結形
        # なら報告ではなく仕様/質問の可能性が高いため、いずれも除外する。
        if (
            mode == "coding"
            and self._prev_query is not None
            and not _CODING_FAILURE_REPORT_EXCLUDE_RE.search(query.strip())
        ):
            for pattern in CODING_FAILURE_REPORT_PATTERNS:
                if pattern.search(query):
                    return query, "hardcoded"

        # 2. 直前ターンが失敗 ([failed] 応答等) → 次ターンは訂正候補
        if self._prev_turn_failed:
            return query, "prev_failed"

        # 3. 同一出力先パスの再指定 + 弱い訂正語 (「〜ではなく」等)
        if self._same_target_path(query) and any(
            p.search(query) for p in WEAK_CORRECTION_PATTERNS
        ):
            return query, "same_target"

        return None, None

    def _learn_tool_routing_from_false_negative(self, query: str) -> None:
        """ツールルーティング false_negative 時: クエリからキーワードを tool_routing として学習

        ツール実行されなかったがユーザーが手動で要求した場合、
        クエリに含まれる意図キーワードを tool_routing カテゴリとして学習する。

        学習可否の判定は ``LearnedPatternStore.extract_tool_routing_keywords``
        に集約する (Level 1 バッチ側 ``_evolve_tool_routing_patterns`` と共通):

        - クエリ自体にツールシグナルが無ければ学習しない (2026-07-18:
          「読書いいですね。最近何か面白い本を読みましたか？」から感想語
          「面白」が誤学習され、雑談中に run_command 判定を誘発した実
          インシデントの再発防止。遡及ヒューリスティックはノイズが多い)。
        - 動作指示語のみ学習し、話題名詞・言語タスク語 (「説明」等) は
          除外する (2026-07-20: 学習済み「説明」w=0.630 が知識質問への
          run_command 誘導を誘発し得た件の再発防止)。
        """
        if self._learned_patterns is None:
            return
        keywords = self._learned_patterns.extract_tool_routing_keywords(query)
        if not keywords:
            logger.debug(
                "Skipping tool_routing learning: no learnable keyword "
                "in query=%s", query[:50],
            )
            return
        for kw in keywords:
            self._learned_patterns.add_pattern(kw, category="tool_routing")

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
