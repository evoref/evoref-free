"""LongFormOrchestrator — 長文生成エントリポイント

設計書 f_09_long_form_generation.md §2, §9, §10 準拠。
Router が長文判定した場合に Meta-Cognitive 層から委任される。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from backend.free.generation.code_skeleton import CodeSkeleton, update_skeleton
from backend.free.generation.content_detector import detect_content_type
from backend.free.generation.models import (
    CodeUnit,
    ContentType,
    LongFormMode,
    SectionPlan,
    extract_target_chars,
)
from backend.free.generation.rolling_context import RollingContext
from backend.free.generation.strategy_cogwriter import CogWriterStrategy
from backend.free.generation.strategy_common import resolve_generation_order
from backend.free.generation.strategy_recurrent import RecurrentStrategy
from backend.free.generation.token_budget import TokenBudget, truncate_tail
from backend.free.generation.validators import remove_code_fences, validate_python
from backend.policy_helpers import get_policy_value
from backend.utils import estimate_tokens

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = logging.getLogger("backend.free.generation.orchestrator")


class LongFormOrchestrator:
    """長文生成のオーケストレーター

    戦略の自動選択（assist_client 有無）、計画生成、ユニット逐次生成、
    レビュー、コード検証を統合管理する。
    """

    def __init__(
        self,
        main_client,
        assist_client=None,
        memory_wm=None,
        memory_stm=None,
        retriever=None,
        embedder=None,
        config: dict | None = None,
        debug_logger: DebugLogger | None = None,
        generation_params: dict | None = None,
        policy: PolicyInterpreter | None = None,
    ):
        self.main_client = main_client
        self.assist_client = assist_client
        self.memory_wm = memory_wm
        self.memory_stm = memory_stm
        self.retriever = retriever
        self.embedder = embedder
        self.config = config or {}
        self._debug_logger = debug_logger
        self._generation_params = generation_params or {}
        self._policy = policy
        # 直近 generate() の content_type ("code" / "text")。コンシューマが生成途中で
        # コード/テキストを判定するために参照する。generate() 開始時に確定する。
        self.last_content_type: str | None = None

        # Recurrent も計画 / 要約再帰をアシストモデルで実行する
        # ため ``assist_client`` を渡す。``None`` の場合は Recurrent 内部で
        # fallback_plan 単一ユニット計画 + 新セクション末尾保持にフォール
        # バックする (ベースモデル経由の JSON 抽出は行わない)。
        self.strategy: CogWriterStrategy | RecurrentStrategy = (
            CogWriterStrategy(
                main_client, assist_client, self.config, debug_logger,
                generation_params=self._generation_params,
            )
            if assist_client
            else RecurrentStrategy(
                main_client, assist_client, self.config, debug_logger,
                generation_params=self._generation_params,
            )
        )

    @property
    def strategy_name(self) -> str:
        """現在の戦略名を返す"""
        return "cogwriter" if isinstance(self.strategy, CogWriterStrategy) else "recurrent"

    def _effective_context_size(self) -> int:
        """スロットあたりの有効コンテキストサイズを返す

        llama-server は context_size を slots で等分するため、
        1 リクエストが使える実効値は context_size // slots になる。
        """
        llama_cfg = self.config.get("llama", {})
        total_ctx = llama_cfg.get("context_size", 4096)
        slots = max(llama_cfg.get("slots", 1), 1)
        return total_ctx // slots

    async def _build_plan_for_generation(
        self,
        instruction: str,
        context: dict,
        content_type: ContentType,
        mode: str,
    ) -> tuple[Any, TokenBudget]:
        """TokenBudget 算出 → 計画生成 → テキストユニット自動分割 → 依存順ソート。

        計画オブジェクトと算出済み TokenBudget のタプルを返す。
        """
        context_size = self._effective_context_size()
        budget = TokenBudget.from_context_size(
            context_size,
            content_type=content_type,
            strategy=self.strategy_name,
        )
        budget.adjust_for_small_context()

        plan = await self.strategy.create_plan(
            instruction, context, content_type, budget,
        )

        # ローカルLLMは1回の生成で安定して出力できるトークン数に限界があるため、
        # 大きすぎるテキストユニットは分割しローリングコンテキストで繋ぐ。
        if content_type == ContentType.TEXT:
            unit_target = self._lf_policy(
                "unit_target_tokens", mode,
                self.config.get("long_form", {}).get("unit_target_tokens", 800),
            )
            plan = _split_oversized_text_units(plan, unit_target)

        # 依存順ソート (コードのみ; create_plan で既にソート済みだが念のため)
        if content_type == ContentType.CODE:
            code_units = [u for u in plan.units if isinstance(u, CodeUnit)]
            if code_units:
                plan.units = resolve_generation_order(code_units)

        return plan, budget

    def _init_rolling_context(
        self,
        plan: Any,
        budget: TokenBudget,
        content_type: ContentType,
        existing_content: str,
    ) -> RollingContext:
        """RollingContext を構築し、追記モード時は既存テキスト末尾を short_term に設定する。"""
        rolling = RollingContext(plan=plan, budget=budget)
        if content_type == ContentType.CODE:
            rolling.skeleton = CodeSkeleton([], [], [], [], [])
        if existing_content and content_type == ContentType.TEXT:
            rolling.short_term = truncate_tail(
                existing_content, rolling.budget.short_term,
            )
            rolling.has_existing_context = True
        return rolling

    @staticmethod
    def _emit_plan_step(on_step, plan: Any, content_type: ContentType) -> None:
        """計画生成完了の SSE ステップを emit する。"""
        if on_step is None:
            return
        _call_step(on_step, {
            "type": "long_form_plan",
            "detail": f"{len(plan.units)} units planned ({content_type.value})",
            "status": "done",
        })

    async def _apply_review_revisions(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> AsyncIterator[str]:
        """CogWriter 戦略時のレビュー & 修正リライト。トークンを yield する。"""
        review_enabled = self.config.get("long_form", {}).get("review_enabled", True)
        if not (isinstance(self.strategy, CogWriterStrategy) and review_enabled):
            return
        revisions = await self.strategy.review(rolling, content_type)
        if on_step:
            _call_step(on_step, {
                "type": "long_form_review",
                "detail": f"{len(revisions)} issues",
                "status": "done",
            })
        max_revisions = self.config.get("long_form", {}).get("max_revisions", 3)
        for rev in revisions[:max_revisions]:
            async for token in self.strategy.revise_unit(rev, rolling, content_type):
                yield token

    def _validate_generated_code(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> tuple[int, int]:
        """コード検証 + SSE 通知 + デバッグログ。`(error_count, warning_count)` を返す。"""
        if content_type != ContentType.CODE or not rolling.generated_units:
            return 0, 0
        assembled = "\n\n".join(
            remove_code_fences(u) for u in rolling.generated_units
        )
        errors = validate_python(assembled)
        validation_errors = sum(1 for e in errors if e.severity == "error")
        warning_count = sum(1 for e in errors if e.severity == "warning")
        if on_step:
            has_errors = validation_errors > 0
            if errors:
                detail = "; ".join(str(e) for e in errors[:5])
            else:
                detail = "OK"
            if not has_errors and warning_count:
                detail = f"OK ({warning_count} warnings)"
            _call_step(on_step, {
                "type": "long_form_validate",
                "detail": detail,
                "status": "failed" if has_errors else "done",
            })
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "validate",
                "errors": [str(e) for e in errors],
                "error_count": validation_errors,
                "warning_count": warning_count,
            })
        return validation_errors, warning_count

    def _record_final_metrics(
        self,
        content_type: ContentType,
        plan: Any,
        units_completed: int,
        validation_errors: int,
        total_tokens: int,
        elapsed: float,
    ) -> None:
        """完了ログ + `last_metrics` 保存 (FeedbackCollector 用)"""
        context_size = self._effective_context_size()
        logger.info(
            "Long-form generation complete: strategy=%s, content_type=%s, "
            "units=%d/%d, tokens=%d, elapsed=%.2fs",
            self.strategy_name, content_type.value,
            units_completed, len(plan.units), total_tokens, elapsed,
        )
        self.last_metrics = {
            "content_type": content_type.value,
            "strategy": self.strategy_name,
            "units_total": len(plan.units),
            "units_completed": units_completed,
            "validation_errors": validation_errors,
            "budget_used_pct": round(total_tokens / max(context_size, 1) * 100, 1),
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed, 3),
        }

    async def generate(
        self,
        instruction: str,
        session_id: str,
        mode: str = "coding",
        on_step: Callable[[dict], Any] | None = None,
        existing_content: str = "",
        long_form_mode: LongFormMode = LongFormMode.CONTINUE,
    ) -> AsyncIterator[str]:
        """長文生成のエントリポイント。トークンを yield する。

        Args:
            instruction: ユーザーの指示
            session_id: セッションID
            mode: "coding" | "chat"
            on_step: SSEステップ通知コールバック
            existing_content: 追記モード時の既存ファイル内容
            long_form_mode: 出力モード。
                :attr:`LongFormMode.CONTINUE` (既定) は従来挙動。
                :attr:`LongFormMode.EXPAND` / :attr:`LongFormMode.SPLIT` は
                planning プロンプトと unit イベントを切り替える (P2/P3 で実装)。

        Yields:
            生成トークン文字列
        """
        t_start = time.monotonic()

        # 1. コンテンツ種別判定
        content_type = detect_content_type(instruction, mode)
        # コンシューマ (chat_streaming) が生成途中でも code/text を判定できるよう、
        # 確定した content_type をインスタンス属性に保持する (リクエストごとの一時
        # インスタンスなので並行リクエストでも競合しない)。
        self.last_content_type = content_type.value

        # 2. メモリ・RAG からコンテキスト収集
        context = await self._gather_context(instruction)
        if existing_content:
            context["existing_content"] = existing_content
        # 出力モード (EXPAND / SPLIT 等) は strategy.create_plan() に
        # プロンプト分岐用に渡す。CONTINUE は従来挙動と互換。
        context["long_form_mode"] = long_form_mode

        # 3-5. 計画 + 予算算出 + 依存順ソート
        plan, budget = await self._build_plan_for_generation(
            instruction, context, content_type, mode,
        )
        self._emit_plan_step(on_step, plan, content_type)

        # 6. ローリングコンテキスト初期化
        rolling = self._init_rolling_context(
            plan, budget, content_type, existing_content,
        )

        total_tokens = 0
        units_completed = 0
        # 長文生成 1 リクエストの総ウォールクロック上限 (低速ローカル GPU での
        # 無限ハング防止)。0 で無効。超過時はユニット境界で打ち切り部分結果を返す。
        total_timeout = float(
            self.config.get("long_form", {}).get("total_timeout_sec", 1800.0) or 0.0
        )
        timed_out = False

        for i, unit in enumerate(plan.units):
            rolling.current_unit_idx = i

            # 総時間上限チェック (ユニット境界で判定し、生成済みで打ち切る)
            if total_timeout and (time.monotonic() - t_start) > total_timeout:
                timed_out = True
                logger.warning(
                    "long_form: total budget %.0fs exceeded after %d/%d units; "
                    "stopping with partial result",
                    total_timeout, units_completed, len(plan.units),
                )
                if on_step:
                    _call_step(on_step, {
                        "type": "long_form_unit_done",
                        "detail": (
                            f"総時間 {total_timeout:.0f}s 超過のため "
                            f"{units_completed}/{len(plan.units)} ユニットで打ち切り"
                        ),
                        "status": "done",
                    })
                break

            label = unit.name if isinstance(unit, CodeUnit) else unit.heading
            if on_step:
                kind_info = f" ({unit.kind})" if isinstance(unit, CodeUnit) else ""
                _call_step(on_step, {
                    "type": "long_form_unit_start",
                    "detail": f"[{i + 1}/{len(plan.units)}] {label}{kind_info}",
                    "status": "running",
                })

            # ユニット間に段落区切りを挿入（テキスト生成時、2ユニット目以降）
            # SPLIT モードでは unit ごとに個別ファイルに書くため区切りは不要
            if (
                content_type == ContentType.TEXT
                and i > 0
                and long_form_mode != LongFormMode.SPLIT
            ):
                yield "\n\n"

            text = ""
            async for token in self.strategy.generate_unit(unit, rolling, content_type):
                text += token
                yield token

            rolling.generated_units.append(text)
            await self._update_rolling_context(rolling, text, content_type)
            unit_tokens = estimate_tokens(text)
            total_tokens += unit_tokens
            units_completed += 1

            if on_step:
                _call_step(on_step, {
                    "type": "long_form_unit_done",
                    "detail": f"[{i + 1}/{len(plan.units)}] {label}: {unit_tokens} tokens",
                    "status": "done",
                })

            # SPLIT モード: unit 完了時に per-unit ファイル書込みイベントを発行。
            # chat_streaming 側がこれを受けて 1 ファイルを即座に書き出す。
            if (
                long_form_mode == LongFormMode.SPLIT
                and content_type == ContentType.TEXT
                and isinstance(unit, SectionPlan)
                and on_step
            ):
                _call_step(on_step, {
                    "type": "long_form_unit_file",
                    "idx": i,
                    "total": len(plan.units),
                    "heading": unit.heading,
                    "file_name": unit.file_name,
                    "content": text,
                })

            if self._debug_logger:
                self._debug_logger.log_long_form_event({
                    "phase": "unit_done",
                    "unit_name": label,
                    "unit_idx": i,
                    "tokens_generated": unit_tokens,
                    "elapsed_sec": round(time.monotonic() - t_start, 3),
                })

        # 7-8. ポスト生成 (目標文字数エンフォース) + レビュー/リライト。
        # 総時間超過で打ち切った場合は省略し、生成済みユニットをそのまま確定する。
        if not timed_out:
            # 7. ポスト生成: 目標文字数エンフォースメント
            async for token in self._extend_to_target(
                rolling, instruction, content_type, on_step,
            ):
                total_tokens += 1
                yield token

            # 8. レビュー & リライト (CogWriter のみ)
            async for token in self._apply_review_revisions(
                rolling, content_type, on_step,
            ):
                yield token

        # 9. コード検証
        validation_errors, _warning_count = self._validate_generated_code(
            rolling, content_type, on_step,
        )

        # 10. メトリクス確定
        elapsed = time.monotonic() - t_start
        self._record_final_metrics(
            content_type, plan, units_completed,
            validation_errors, total_tokens, elapsed,
        )

    async def _gather_context(self, instruction: str) -> dict:
        """メモリ3層 + RAG からコンテキストを収集"""
        context: dict[str, str] = {"rag": "", "memory": ""}

        # メモリコンテキスト（WorkingMemory のターン履歴）
        if self.memory_wm is not None:
            try:
                turns = self.memory_wm.get_messages()
                # 直近のターンをメモリコンテキストとして使用
                memory_parts = []
                for turn in turns[-6:]:  # 直近6ターン
                    role = turn.get("role", "")
                    content = turn.get("content", "")[:200]
                    if content:
                        memory_parts.append(f"[{role}] {content}")
                context["memory"] = "\n".join(memory_parts)
            except Exception as e:
                logger.warning("Failed to gather memory context: %s", e)

        # RAG コンテキスト（retriever が利用可能な場合）
        if self.retriever is not None and self.embedder is not None:
            try:
                results = await self.retriever.search(instruction, top_k=3)
                if results:
                    rag_parts = []
                    for chunk_text, score, source in results[:3]:
                        rag_parts.append(f"[{source}] (score={score:.2f})\n{chunk_text[:300]}")
                    context["rag"] = "\n---\n".join(rag_parts)
            except Exception as e:
                logger.warning("Failed to gather RAG context: %s", e)

        return context

    async def _update_rolling_context(
        self, rolling: RollingContext, text: str, content_type: ContentType,
    ) -> None:
        """ローリングコンテキストを更新"""
        # 共通: 直前ユニットの末尾を保持
        rolling.short_term = truncate_tail(text, rolling.budget.short_term)

        if content_type == ContentType.CODE:
            # コード: スケルトン更新（ルールベース、LLM不要）
            if rolling.skeleton is not None:
                rolling.skeleton = update_skeleton(rolling.skeleton, text)
                if self._debug_logger:
                    self._debug_logger.log_long_form_event({
                        "phase": "skeleton_update",
                        "imports_count": len(rolling.skeleton.imports),
                        "signatures_count": len(rolling.skeleton.function_signatures),
                        "types_count": len(rolling.skeleton.type_definitions),
                    })
        else:
            # テキスト: 要約更新
            if isinstance(self.strategy, RecurrentStrategy):
                # Recurrent: LLMで要約更新
                t0 = time.monotonic()
                rolling.long_term_summary = await self.strategy.update_summary(
                    rolling.long_term_summary, text, rolling.budget,
                )
                if self._debug_logger:
                    self._debug_logger.log_long_form_event({
                        "phase": "summary_update",
                        "summary_length": len(rolling.long_term_summary),
                        "elapsed_sec": round(time.monotonic() - t0, 3),
                    })
            # CogWriter: アシストが全文参照できるため要約不要

    @staticmethod
    def _get_extend_tail(rolling: RollingContext) -> str:
        """追加生成の文脈として使う直前テキスト末尾を取得する。

        `short_term` がセットされていればそれを、なければ最後の生成ユニットの
        末尾 500 文字を返す。両方なければ空文字列。
        """
        tail = rolling.short_term
        if not tail and rolling.generated_units:
            tail = rolling.generated_units[-1][-500:]
        return tail

    @staticmethod
    def _build_extend_messages(remaining: int, tail: str) -> list[dict]:
        """追加生成用のシンプルな messages を構築する。"""
        return [
            {"role": "system", "content": (
                "あなたは物語の続きを書く作家です。"
                "以下の文章の続きを自然に書いてください。"
                "見出し行やメタ的な記述は不要です。本文のみを出力してください。"
            )},
            {"role": "user", "content": (
                f"以下の文章の続きを{remaining}文字程度書いてください。\n\n"
                f"{tail}\n\n続き:"
            )},
        ]

    def _build_extend_gen_kwargs(self, remaining: int) -> dict:
        """追加生成用の generate kwargs を構築する。"""
        unit_max = self._lf_policy("unit_max_tokens", "chat", 2000)
        max_tokens = min(unit_max, max(int(remaining * 0.6 * 1.5), 512))
        gen_kwargs: dict = {
            "stream": True,
            "temperature": self._generation_params.get("temperature", 0.7),
            "max_tokens": max_tokens,
            "id_slot": self.main_client.chat_slot,
        }
        for k in ("top_p", "top_k", "presence_penalty"):
            if k in self._generation_params:
                gen_kwargs[k] = self._generation_params[k]
        return gen_kwargs

    async def _stream_extend_round(
        self,
        remaining: int,
        tail: str,
    ) -> AsyncIterator[str]:
        """1 ラウンド分の追加生成トークンを yield する。

        例外発生時は呼び出し側で `Exception` を捕捉する想定。
        """
        messages = self._build_extend_messages(remaining, tail)
        gen_kwargs = self._build_extend_gen_kwargs(remaining)
        stream = await self.main_client.generate(messages, **gen_kwargs)
        async for token in stream:
            yield token

    async def _extend_to_target(
        self,
        rolling: RollingContext,
        instruction: str,
        content_type: ContentType,
        on_step: Callable[[dict], Any] | None,
    ) -> AsyncIterator[str]:
        """目標文字数に到達するまで追加生成を繰り返す

        計画ベースの生成完了後、合計文字数が目標の70%未満の場合に
        単純な「続き」プロンプトで直接LLMを呼び出して追加生成する。

        戦略の generate_unit は複雑なプロンプトを構築するため
        ローカルLLMが短い応答しか返さない傾向がある。
        この方法では最小限のプロンプトで確実に生成量を確保する。
        """
        target_chars = extract_target_chars(instruction, default=0)
        if target_chars <= 0 or content_type != ContentType.TEXT:
            return

        total_chars = sum(len(t) for t in rolling.generated_units)
        extend_ratio = self._lf_policy("extend_threshold_ratio", "chat", 0.7)
        threshold = int(target_chars * extend_ratio)
        max_rounds = self._lf_policy(
            "max_extend_rounds", "chat",
            self.config.get("long_form", {}).get("max_extend_rounds", 10),
        )

        extend_count = 0
        while total_chars < threshold and extend_count < max_rounds:
            extend_count += 1
            remaining = target_chars - total_chars
            tail = self._get_extend_tail(rolling)

            if on_step:
                _call_step(on_step, {
                    "type": "long_form_unit_start",
                    "detail": f"[追加生成 {extend_count}] 残り約{remaining}文字",
                    "status": "running",
                })

            # 追加生成の前に段落区切りを挿入
            yield "\n\n"

            ext_text = ""
            try:
                async for token in self._stream_extend_round(remaining, tail):
                    ext_text += token
                    yield token
            except Exception as e:  # noqa: BLE001 — LLM 呼出失敗は警告して停止
                logger.warning("Extend generation failed: %s", e)
                break

            if not ext_text.strip():
                logger.info(
                    "Extend round %d: empty output, stopping", extend_count,
                )
                break

            rolling.generated_units.append(ext_text)
            rolling.short_term = ext_text[-500:]
            total_chars += len(ext_text)

            if on_step:
                _call_step(on_step, {
                    "type": "long_form_unit_done",
                    "detail": (
                        f"[追加生成 {extend_count}] +{len(ext_text)}文字 "
                        f"(合計 {total_chars}/{target_chars}文字)"
                    ),
                    "status": "done",
                })

            logger.info(
                "Extend round %d: +%d chars (total %d/%d)",
                extend_count, len(ext_text), total_chars, target_chars,
            )


    def _lf_policy(
        self, key: str, mode: str, default: int | float,
    ) -> int | float:
        """long_form ポリシーからパラメータ取得（フォールバック付き）"""
        return get_policy_value(self._policy, "long_form", key, default, mode=mode)


def _call_step(on_step: Callable[[dict], Any], data: dict) -> None:
    """ステップコールバックを呼び出す（sync のみ、chat.py のキュー方式に合わせる）"""
    on_step(data)


def _split_oversized_text_units(
    plan: "GenerationPlan",
    unit_target_tokens: int,
) -> "GenerationPlan":
    """大きすぎるテキストユニットを複数ユニットに分割する

    ローカルLLMは1回の生成で安定して出力できるトークン数に限界がある。
    estimated_tokens が unit_target_tokens を超えるユニットを分割し、
    各ユニットがLLMの安定出力範囲内に収まるようにする。

    分割後の各サブユニットはローリングコンテキスト（short_term）で
    自然に繋がるため、テキストの一貫性は維持される。
    """

    new_units: list[CodeUnit | SectionPlan] = []
    for unit in plan.units:
        if not isinstance(unit, SectionPlan):
            new_units.append(unit)
            continue

        if unit.estimated_tokens <= unit_target_tokens:
            new_units.append(unit)
            continue

        # 分割数を算出
        n_splits = math.ceil(unit.estimated_tokens / unit_target_tokens)
        tokens_per_split = unit.estimated_tokens // n_splits

        for i in range(n_splits):
            if i == 0:
                # 最初のサブユニット: 元の key_points を引き継ぐ
                sub = SectionPlan(
                    heading=unit.heading,
                    key_points=unit.key_points,
                    estimated_tokens=tokens_per_split,
                )
            else:
                # 後続サブユニット: 前の内容の続きとして生成
                sub = SectionPlan(
                    heading=f"{unit.heading}（続き{i + 1}）",
                    key_points=["前の文章から自然に続く内容を書いてください。"],
                    estimated_tokens=tokens_per_split,
                )
            new_units.append(sub)

        logger.info(
            "Split oversized unit '%s': %d tokens -> %d sub-units x %d tokens",
            unit.heading, unit.estimated_tokens, n_splits, tokens_per_split,
        )

    plan.units = new_units
    return plan
