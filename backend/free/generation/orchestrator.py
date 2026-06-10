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

from backend.config import resolve_context_size
from backend.free.generation.code_repair import CodeRepairer, infer_language
from backend.free.generation.import_wirer import wire_imports
from backend.free.generation.code_skeleton import CodeSkeleton, update_skeleton
from backend.free.generation.content_detector import detect_content_type
from backend.free.generation.smoke_validator import (
    check_integrity,
    normalize_relative_imports,
    run_import_smoke,
)
from backend.free.generation.spec_renderer import render_spec_markdown
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
from backend.free.generation.validators import (
    ValidationError,
    collapse_runaway_repetition,
    remove_code_fences,
    validate_python,
)
from backend.policy_helpers import get_policy_value
from backend.utils import estimate_tokens

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = logging.getLogger("backend.free.generation.orchestrator")

# 長文生成の plan コンテキストへ注入する RAG 予算。``_gather_context`` は
# ``_build_plan_for_generation`` の TokenBudget 算出前に走るため、ここでは
# 固定値で plan プロンプトの肥大を防ぐ。
RAG_MAX_CHUNKS = 3
RAG_CHUNK_CHAR_CAP = 300
RAG_TOTAL_TOKEN_BUDGET = 256

# CodeUnit に file_path が無い (degraded fallback 等) 場合の既定ファイル名。
_DEFAULT_CODE_FILE = "output.py"


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
        # 直近 generate() のコード出力 (検証・修正済み assembled)。chat_streaming の
        # finalize が coding の editor/file 出力に使う (生ストリーム二重追記の解消)。
        self.last_code_output: str | None = None
        # 直近コードの推定言語 (検証を Python のみ AST 検証に限定するため)。
        self._code_language: str = "python"
        # 直近コードのファイル別 (file_path → 検証・修正済みコード)。複数ファイル
        # 計画時に Agentic 経路が複数 EditorArtifact として配信するために使う。
        self.last_code_files: dict[str, str] = {}
        # 直近生成の post-repair 検証で残った error 行 (severity=error のみ)。
        # 配信側 (make_code_artifact_generator) が「壊れたコードを成功として
        # 渡さない」ためユーザーへ提示する。
        self.last_validation_errors: list[str] = []

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
        total_ctx = resolve_context_size(self.config, "base")
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

    async def _repair_generated_code(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> None:
        """生成コードを検証ゲート付きでリペアし ``last_code_output`` に保持する。

        review 後の ``generated_units`` を assemble → 検証 → assist 修正 → 再検証。
        coding の editor/file 出力はこの結果を配信する (生ストリームの revise
        二重追記の解消)。リペア無効 / degraded 時は素の assembled をそのまま保持。
        """
        if content_type != ContentType.CODE or not rolling.generated_units:
            return
        repairer = CodeRepairer(
            self.assist_client, self.config, debug_logger=self._debug_logger,
        )
        # plan.units と generated_units は位置対応 (CODE は _extend_to_target で
        # append せず、revise は in-place 更新)。timeout 部分生成でも安全なよう
        # zip で生成済み分だけ走査する。
        coded = [
            (u, remove_code_fences(text))
            for u, text in zip(rolling.plan.units, rolling.generated_units)
            if isinstance(u, CodeUnit)
        ]
        if not coded:
            # CodeUnit が取れない (degraded plan 等) → 全体を 1 ファイル扱い。
            assembled = "\n\n".join(
                remove_code_fences(u) for u in rolling.generated_units
            )
            self._code_language = "python"
            repaired = await repairer.repair(assembled, language="python")
            self.last_code_output = repaired
            self.last_code_files = {_DEFAULT_CODE_FILE: repaired}
            if on_step and repaired != assembled:
                _call_step(on_step, {
                    "type": "long_form_repair",
                    "detail": "コードを検証・修正しました (1 file)",
                    "status": "done",
                })
            return
        # file_path 別にグルーピングし、ファイル単位で検証・修正する。
        groups: dict[str, list[str]] = {}
        for unit, text in coded:
            groups.setdefault(unit.file_path or _DEFAULT_CODE_FILE, []).append(text)
        files: dict[str, str] = {}
        changed = False
        # 複数ファイル分割時は他ファイル定義シンボル参照の undefined 誤検知を避け、
        # ファイル単位の修正を構文エラーのみに限定する (未定義名は全体検証側で扱う)。
        multi_file = len(groups) > 1
        for path, texts in groups.items():
            group_code = "\n\n".join(texts)
            repaired_group = await repairer.repair(
                group_code, language=infer_language([path]), syntax_only=multi_file,
            )
            if repaired_group != group_code:
                changed = True
            files[path] = repaired_group
        if multi_file:
            # flat 配信では解決不能な相対 import (from .x import) を先に除去し、
            # その後 wire_imports が兄弟定義から正しい絶対 import を再配線する。
            # (今回の LINE アプリ失敗の主因 `from .models import` の決定的修正)
            files = normalize_relative_imports(files)
            # 複数ファイルは相互 import が欠落するため AST ベースで補完する
            # (cross-file 配線 + stdlib 伝播)。実行時ロジックは対象外。
            files = wire_imports(files)
        self.last_code_files = files
        self.last_code_output = "\n\n".join(files.values())
        self._code_language = infer_language(list(groups.keys()))
        if on_step and changed:
            _call_step(on_step, {
                "type": "long_form_repair",
                "detail": f"コードを検証・修正しました ({len(files)} file)",
                "status": "done",
            })

    def _validate_generated_code(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> tuple[int, int]:
        """コード検証 + SSE 通知 + デバッグログ。`(error_count, warning_count)` を返す。

        リペア後の出力を対象に検証する。``validate_python`` は Python 専用のため、
        他言語では AST 検証をスキップする (Python として誤検出しないため)。

        複数ファイル生成時は ``last_code_files`` を**ファイル単位**で検証して集約する。
        全ファイルを結合して検証すると、あるファイルの未 import 参照が別ファイルの
        定義で「定義済み」と誤判定され cross-file の未定義が同一名前空間でマスクされる
        (実測: 結合検証は 0 件、ファイル単位だと未定義を検出)。import 配線
        (``wire_imports``) 後を対象とするため、配線で解決済みなら誤検知しない。
        """
        if content_type != ContentType.CODE or not rolling.generated_units:
            return 0, 0
        if getattr(self, "_code_language", "python") != "python":
            if on_step:
                _call_step(on_step, {
                    "type": "long_form_validate",
                    "detail": f"skipped (language={self._code_language})",
                    "status": "done",
                })
            return 0, 0
        files = self.last_code_files
        if files and len(files) > 1:
            errors = [
                ValidationError(
                    e.error_type, f"{path}: {e.message}", severity=e.severity,
                )
                for path in sorted(files)
                for e in validate_python(files[path])
            ]
        else:
            code = self.last_code_output
            if code is None:
                code = "\n\n".join(
                    remove_code_fences(u) for u in rolling.generated_units
                )
            errors = validate_python(code)
        validation_errors = sum(1 for e in errors if e.severity == "error")
        warning_count = sum(1 for e in errors if e.severity == "warning")
        self.last_validation_errors = [
            str(e) for e in errors if e.severity == "error"
        ]
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

    def _run_integrity_and_smoke(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> None:
        """整合ゲート (静的) + import スモークテスト (config 任意) を実行する。

        検出した**実エラー** (相対 import 残存 / エントリポイント欠落 / cross-file
        の ModuleNotFoundError / dataclass 引数違い等) を ``last_validation_errors``
        に追加し、配信側が「壊れたコードを成功として渡さない」よう可視化する。
        スモークテスト (サブプロセス) は ``long_form.code_smoke_test_enabled``
        (コード既定 False / 配布 config 既定 True) で gate する。
        """
        if content_type != ContentType.CODE or not self.last_code_files:
            return
        if getattr(self, "_code_language", "python") != "python":
            return

        files = self.last_code_files
        spec = rolling.plan.code_spec

        issues = check_integrity(files, spec)

        lf = self.config.get("long_form", {})
        warnings: list[str] = []
        if lf.get("code_smoke_test_enabled", False):
            timeout = float(lf.get("code_smoke_timeout_sec", 10.0) or 10.0)
            smoke = run_import_smoke(files, timeout_sec=timeout)
            issues.extend(smoke.errors)
            warnings = smoke.warnings

        if issues:
            self.last_validation_errors.extend(issues)
        if on_step:
            if issues:
                detail = "; ".join(issues[:5])
                status = "failed"
            else:
                detail = "OK" + (
                    f" ({len(warnings)} warnings)" if warnings else ""
                )
                status = "done"
            _call_step(on_step, {
                "type": "long_form_integrity",
                "detail": detail,
                "status": status,
            })
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "integrity",
                "errors": issues,
                "warnings": warnings,
            })

    def _attach_spec_artifact(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> None:
        """設計仕様を ``SPEC.md`` として ``last_code_files`` に添付する (CODE のみ)。

        ``rolling.plan.code_spec`` が無い (合成失敗 / 無効化) / コードが空の場合は
        何もしない。flowchart (Phase 2) があれば mermaid を埋め込む。
        """
        if content_type != ContentType.CODE:
            return
        spec = rolling.plan.code_spec
        if spec is None or not self.last_code_files:
            return
        flowchart = getattr(rolling.plan, "code_flowchart", "") or ""
        markdown = render_spec_markdown(spec, flowchart_mermaid=flowchart)
        if not markdown.strip():
            return
        self.last_code_files["SPEC.md"] = markdown
        if on_step:
            _call_step(on_step, {
                "type": "long_form_spec",
                "detail": "SPEC.md (設計仕様) を生成しました",
                "status": "done",
            })

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
        prefetched_rag: list[tuple[str, float, str]] | None = None,
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
        # prefetched_rag があれば retriever を呼ばず取得済みチャンクを再利用する。
        context = await self._gather_context(instruction, prefetched_rag=prefetched_rag)
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

            # 退化反復 (トークン生成ループ) を確定ユニットから切除する。ライブ
            # ストリームには既に流れているが、assemble / 後続ユニットのコンテキスト
            # 汚染を防ぐため保存前にクリーンにする。
            raw_len = len(text)
            text = collapse_runaway_repetition(text)
            if len(text) < raw_len and self._debug_logger:
                self._debug_logger.log_long_form_event({
                    "phase": "repetition_collapsed",
                    "unit_name": label,
                    "unit_idx": i,
                    "chars_before": raw_len,
                    "chars_after": len(text),
                })
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

        # 8.5 検証ゲート付きコードリペア (CODE のみ)。review 後の generated_units を
        # assemble → 検証 → assist 修正 → 再検証し、last_code_output に保持する。
        # 総時間超過時も最終出力品質のため実行する (max_repair_rounds で有界)。
        await self._repair_generated_code(rolling, content_type, on_step)

        # 9. コード検証 (リペア後を対象、Python のみ AST 検証)
        validation_errors, _warning_count = self._validate_generated_code(
            rolling, content_type, on_step,
        )

        # 9.5 整合ゲート (相対 import 残存 / エントリポイント) + import スモーク
        # テスト (config 任意)。AST 検証が見逃す実行時 import エラー等を捕捉する。
        self._run_integrity_and_smoke(rolling, content_type, on_step)

        # 9.6 設計仕様を SPEC.md 成果物として添付 (CODE のみ)。
        self._attach_spec_artifact(rolling, content_type, on_step)

        # 10. メトリクス確定
        elapsed = time.monotonic() - t_start
        self._record_final_metrics(
            content_type, plan, units_completed,
            validation_errors, total_tokens, elapsed,
        )

    async def _gather_context(
        self,
        instruction: str,
        prefetched_rag: list[tuple[str, float, str]] | None = None,
    ) -> dict:
        """メモリ3層 + RAG からコンテキストを収集

        ``prefetched_rag`` (search pipeline 取得済み ``(chunk_id, score, text)``)
        があれば retriever を呼ばずそれを再利用する。``None``/空なら従来の
        retriever 経路 (注入時のみ) に倒し、いずれも無ければ RAG なし。
        """
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

        # RAG コンテキスト。
        # 優先: search pipeline 取得済みの scored_chunks (prefetched_rag)。
        # long_form 経路は retriever を注入しないため二重取得を避け、上流で
        # 品質判定 / content gate を通った結果をそのまま再利用する。
        # 従来の retriever 経路は retriever 注入時 (ユニットテスト等) のみ温存。
        if prefetched_rag:
            context["rag"] = self._format_prefetched_rag(prefetched_rag)
        elif self.retriever is not None and self.embedder is not None:
            try:
                results = await self.retriever.search(
                    instruction, top_k=RAG_MAX_CHUNKS,
                )
                if results:
                    rag_parts = []
                    for chunk_text, score, source in results[:RAG_MAX_CHUNKS]:
                        rag_parts.append(
                            f"[{source}] (score={score:.2f})\n"
                            f"{chunk_text[:RAG_CHUNK_CHAR_CAP]}"
                        )
                    context["rag"] = "\n---\n".join(rag_parts)
            except Exception as e:
                logger.warning("Failed to gather RAG context: %s", e)

        return context

    @staticmethod
    def _format_prefetched_rag(
        scored_chunks: list[tuple[str, float, str]],
    ) -> str:
        """search pipeline 取得済み scored_chunks を plan コンテキスト文字列へ整形。

        ``scored_chunks`` は ``(chunk_id, score, text)`` (search pipeline 形式)。
        入力順 (salience 順) を維持し上位 ``RAG_MAX_CHUNKS`` 件を
        各 ``RAG_CHUNK_CHAR_CAP`` 字に丸め、全体 ``RAG_TOTAL_TOKEN_BUDGET``
        トークンに収まる範囲だけ採用する (plan プロンプト肥大の防止)。
        """
        parts: list[str] = []
        used_tokens = 0
        for chunk_id, score, text in scored_chunks[:RAG_MAX_CHUNKS]:
            snippet = (text or "")[:RAG_CHUNK_CHAR_CAP]
            if not snippet:
                continue
            part = f"[{chunk_id}] (score={score:.2f})\n{snippet}"
            part_tokens = estimate_tokens(part)
            # 1 件目は予算超過でも必ず含める (RAG を空にしない)。
            if parts and used_tokens + part_tokens > RAG_TOTAL_TOKEN_BUDGET:
                break
            parts.append(part)
            used_tokens += part_tokens
        return "\n---\n".join(parts)

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
