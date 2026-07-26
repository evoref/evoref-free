"""LongFormOrchestrator — 長文生成エントリポイント

設計書 f_09_long_form_generation.md §2, §9, §10 準拠。
Router が長文判定した場合に Meta-Cognitive 層から委任される。
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from backend.config import resolve_context_size_for_mode
from backend.free.generation.code_repair import CodeRepairer, infer_language
from backend.free.generation.import_wirer import wire_imports
from backend.free.generation.code_skeleton import CodeSkeleton, update_skeleton
from backend.free.generation.content_detector import detect_content_type
from backend.free.generation.document_gate import (
    evaluate_document,
    is_document_format,
)
from backend.free.generation.smoke_validator import (
    check_integrity,
    dedup_top_level_defs,
    normalize_relative_imports,
    run_import_smoke,
)
from backend.free.generation.spec_renderer import (
    render_spec_for_prompt,
    render_spec_markdown,
)
from backend.free.generation.models import (
    CodeUnit,
    ContentType,
    LongFormMode,
    SectionPlan,
    extract_target_chars,
)
from backend.free.generation.rolling_context import RollingContext
from backend.free.generation.strategy_cogwriter import CogWriterStrategy, ReviewIssue
from backend.free.generation.strategy_common import resolve_generation_order
from backend.free.generation.strategy_recurrent import RecurrentStrategy
from backend.free.generation.token_budget import TokenBudget, truncate_tail
from backend.free.generation.validators import (
    ValidationError,
    collapse_runaway_repetition,
    remove_code_fences,
    validate_python,
)
from backend.i18n_helper import get_locale
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


def _lexical_overlap(query: str, text: str) -> float:
    """クエリとチャンクの語句重なり率 (0.0-1.0)。

    埋め込み器が無い経路 (チャットの long_form) でユニット別に RAG チャンクを
    選抜するための決定論スコア。日本語も拾えるよう文字 bi-gram を併用する。
    """
    def _grams(s: str) -> set[str]:
        s = s.lower()
        words = {w for w in re.findall(r"[a-z0-9_]{2,}", s)}
        bigrams = {s[i:i + 2] for i in range(len(s) - 1) if not s[i].isspace()}
        return words | bigrams

    q = _grams(query)
    if not q:
        return 0.0
    return len(q & _grams(text)) / len(q)

# CodeUnit に file_path が無い (degraded fallback 等) 場合の既定ファイル名。
_DEFAULT_CODE_FILE = "output.py"

# 散文出力の品質ゲート (_validate_generated_text)。
# 目標文字数比の許容レンジ。plan.target_length が 0 (未指定) のときは判定しない。
_TARGET_LENGTH_MIN_RATIO = 0.5
_TARGET_LENGTH_MAX_RATIO = 1.5
# 文重複率の判定は短文で不安定なので、一定数の文が無いと評価しない。
_DUP_CHECK_MIN_SENTENCES = 12
# ユニーク文の比率がこれ未満なら degeneration とみなし error。
_MIN_UNIQUE_SENTENCE_RATIO = 0.6
# error には満たないが健全とも言えない領域は warning に落とす。
_WARN_UNIQUE_SENTENCE_RATIO = 0.8
# 重複判定で無視する短文 (定型の相槌・見出し断片を拾わないため)。
_DUP_MIN_SENTENCE_CHARS = 15

# 文分割 (日本語の句点・感嘆・疑問 + 改行 + 英文のピリオド)。
_SENTENCE_SPLIT_RE = re.compile(r"[。．!！?？\n]+")


def _sentence_dup_stats(text: str) -> tuple[int, int]:
    """``(総文数, ユニーク文数)`` を返す。空白を正規化して比較する。

    ``_DUP_MIN_SENTENCE_CHARS`` 未満の短文は除外する (箇条書きの記号や
    「はい」のような相槌が重複としてカウントされるのを避けるため)。
    """
    seen: set[str] = set()
    total = 0
    for raw in _SENTENCE_SPLIT_RE.split(text):
        s = " ".join(raw.split())
        if len(s) < _DUP_MIN_SENTENCE_CHARS:
            continue
        total += 1
        seen.add(s)
    return total, len(seen)


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
        self.retriever = retriever
        self.embedder = embedder
        self.config = config or {}
        self._debug_logger = debug_logger
        self._generation_params = generation_params or {}
        self._policy = policy
        # 直近 generate() のモード。_effective_context_size が coding_model の
        # 実窓を反映するために参照する。generate() 開始時に確定する。
        self._mode = "coding"
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
        # 直近生成の検証で残った error 行。CODE は post-repair の AST 検証
        # (severity=error のみ)、TEXT は品質ゲート (_validate_generated_text)。
        # CODE では配信側 (make_code_artifact_generator) が「壊れたコードを成功
        # として渡さない」ためユーザーへ提示する (TEXT は artifact を持たないので
        # 当該経路には乗らず、chat_streaming の警告ステップ側で surface される)。
        self.last_validation_errors: list[str] = []
        # 直近 generate() の TEXT 確定本文 (document_quality_enabled 時のみ、改稿済み
        # generated_units から組む)。chat_streaming の finalize が file/editor 出力に
        # 使い、生ストリームの revise 二重追記を解消する (CODE の last_code_output と
        # 対称)。非 document_quality / 非 TEXT では None のまま (従来の生ストリーム使用)。
        self.last_text_output: str | None = None
        # 直近 generate() の出力先拡張子 (document_quality ゲートの形式判定に使う)。
        self._target_format: str = ""
        # 直近 generate() が主題不明で確認質問を返した (ユニット生成を行わなかった)
        # かどうか。chat_streaming の finalize はこの場合 write_file を呼ばない
        # (確認質問をファイルへ書き込んでしまうことを防ぐ)。
        self.last_needed_clarification: bool = False
        # 直近 review が検出した issue 数と、max_revisions 予算内で改稿できなかった
        # 残数。TEXT の品質判定 (_validate_generated_text) が参照する。
        self._last_review_issue_count: int = 0
        self._last_unaddressed_issue_count: int = 0

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
        total_ctx = resolve_context_size_for_mode(self.config, self._mode)
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
            max_units = self.config.get("long_form", {}).get("max_units", 20)
            plan = _split_oversized_text_units(
                plan, unit_target, max_units=max_units,
            )

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
            rolling.short_term = self._fit_short_term(
                truncate_tail(existing_content, rolling.budget.short_term),
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

    @staticmethod
    def _default_clarification_question() -> str:
        """計画側が確認質問を空文字で返した場合のフォールバック文言。"""
        if get_locale() == "en":
            return "What would you like this document to be about?"
        return "どのような内容・主題の文書をご希望ですか？"

    async def _apply_review_revisions(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> AsyncIterator[str]:
        """CogWriter 戦略時のレビュー & 修正リライト。トークンを yield する。

        ``review_enabled`` (既定 True) の LLM レビューに加え、
        ``document_quality_enabled`` (既定 OFF) のとき TEXT のドキュメント出力先には
        決定論的な構造ゲート (空 / 表の列ずれ / 見出し階層 / 形式不適合) を重ねる。
        ゲート赤は export 破綻に直結するため LLM レビューより優先して有界改稿する。
        """
        self._last_review_issue_count = 0
        self._last_unaddressed_issue_count = 0
        if not isinstance(self.strategy, CogWriterStrategy):
            return
        lf = self.config.get("long_form", {})
        review_enabled = lf.get("review_enabled", True)
        max_revisions = lf.get("max_revisions", 3)

        revisions: list[ReviewIssue] = []
        if review_enabled:
            revisions = list(await self.strategy.review(rolling, content_type))
            if on_step:
                _call_step(on_step, {
                    "type": "long_form_review",
                    "detail": f"{len(revisions)} issues",
                    "status": "done",
                })

        # 決定論ドキュメント品質ゲート (TEXT のドキュメント出力先のみ、既定 OFF)。
        # 構造の欠落のみ指摘し取得データを創作させない。ゲート赤を LLM レビューより
        # 前に置き、max_revisions の予算内で優先的に改稿させる。
        if (
            content_type == ContentType.TEXT
            and lf.get("document_quality_enabled", False)
            and is_document_format(self._target_format)
        ):
            plan_headings = [
                u.heading for u in rolling.plan.units if isinstance(u, SectionPlan)
            ]
            gate_issues = evaluate_document(
                rolling.generated_units, plan_headings, self._target_format,
            )
            if on_step:
                _call_step(on_step, {
                    "type": "long_form_document_gate",
                    "detail": f"{len(gate_issues)} structural issues",
                    "status": "done",
                })
            revisions = gate_issues + revisions

        # 検出数と「予算内で改稿しきれなかった残数」を記録する。残issueは出力に
        # 残ったままなので、TEXT の成否判定 (_validate_generated_text) が error として
        # 数える。これが無いと review が issue を出しても success 判定に反映されない。
        self._last_review_issue_count = len(revisions)
        self._last_unaddressed_issue_count = max(0, len(revisions) - max_revisions)

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
        # 共有設計仕様 (契約) をリペアプロンプトへ同梱する。無いと修正が「検出
        # エラーを消すこと」だけを目標にでき、契約 (モジュール名 / データモデルの
        # フィールド名・型 / 公開シグネチャ) から逸脱した修正 (機能削除含む) を
        # エラー数減として採用しうる。
        repair_spec = render_spec_for_prompt(getattr(rolling.plan, "code_spec", None))
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
            # 生成パスの二重連結による同名 def/class 重複を決定論的に除去する。
            assembled = dedup_top_level_defs("\n\n".join(
                remove_code_fences(u) for u in rolling.generated_units
            ))
            self._code_language = "python"
            repaired = await repairer.repair(
                assembled, language="python", spec=repair_spec,
            )
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
        # 複数ファイル分割時は cross-file 参照の undefined 誤検知を避けるため undefined を
        # 修正対象から外すが、同一ファイル内で完結するエラー (構文 / dataclass 引数不整合)
        # は修正する。undefined の解決は後段の wire_imports に委ねる。
        multi_file = len(groups) > 1
        for path, texts in groups.items():
            # 同一ファイルへ複数ユニットを連結する際、同名 def/class の重複
            # (生成パス二重連結 = 後勝ちで前定義が死ぬ) を repair 前に決定論的除去。
            group_code = dedup_top_level_defs("\n\n".join(texts))
            repaired_group = await repairer.repair(
                group_code, language=infer_language([path]),
                intra_file_only=multi_file, spec=repair_spec,
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

    def _validate_generated_text(
        self,
        rolling: RollingContext,
        content_type: ContentType,
        on_step,
    ) -> tuple[int, int]:
        """散文出力の品質検証。``(error_count, warning_count)`` を返す。

        ``_validate_generated_code`` は CODE 専用のため、TEXT では従来
        ``validation_errors`` が常に 0 になり、``long_form_success`` が実質恒真だった
        (壊れた出力が Level 0 の正例として学習される)。決定論で判定できる 3 点のみ見る:

        1. review が検出し ``max_revisions`` 予算内で改稿しきれなかった残 issue
        2. 目標文字数比が ``[0.5, 1.5]`` の外 (``target_length`` 未指定時はスキップ)
        3. 文単位の重複率 (ユニーク文 / 総文 が ``_MIN_UNIQUE_SENTENCE_RATIO`` 未満)

        3 は degeneration ループの検出用。実測 (2026-07-25) では同一文が 34 回反復した
        7,192 字のメールが 43/156 = 0.28 で、正常な生成は 0.9 以上だった。
        """
        if content_type != ContentType.TEXT or not rolling.generated_units:
            return 0, 0

        text = "\n\n".join(rolling.generated_units)
        errors: list[str] = []
        warnings: list[str] = []

        unaddressed = self._last_unaddressed_issue_count
        if unaddressed > 0:
            errors.append(
                f"review issues left unaddressed: {unaddressed}"
                f" (detected={self._last_review_issue_count})",
            )

        target = getattr(rolling.plan, "target_length", 0) or 0
        if target > 0:
            ratio = len(text) / target
            if ratio < _TARGET_LENGTH_MIN_RATIO or ratio > _TARGET_LENGTH_MAX_RATIO:
                errors.append(
                    f"length {len(text)} chars is {ratio:.2f}x target {target}",
                )

        total, unique = _sentence_dup_stats(text)
        if total >= _DUP_CHECK_MIN_SENTENCES:
            uniq_ratio = unique / total
            if uniq_ratio < _MIN_UNIQUE_SENTENCE_RATIO:
                errors.append(
                    f"repetitive output: {unique}/{total} unique sentences"
                    f" ({uniq_ratio:.2f})",
                )
            elif uniq_ratio < _WARN_UNIQUE_SENTENCE_RATIO:
                warnings.append(f"unique sentence ratio {uniq_ratio:.2f}")

        self.last_validation_errors = list(errors)
        if on_step:
            _call_step(on_step, {
                "type": "long_form_validate",
                "detail": "; ".join(errors) if errors else "OK",
                "status": "failed" if errors else "done",
            })
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "validate",
                "errors": errors,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "unique_sentence_ratio": round(unique / total, 3) if total else None,
                "review_issues": self._last_review_issue_count,
            })
        return len(errors), len(warnings)

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
        if lf.get("code_smoke_test_enabled", True):
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
        session_id: str,  # noqa: ARG002
        mode: str = "coding",
        on_step: Callable[[dict], Any] | None = None,
        existing_content: str = "",
        long_form_mode: LongFormMode = LongFormMode.CONTINUE,
        prefetched_rag: list[tuple[str, float, str]] | None = None,
        file_context_block: str | None = None,
        content_type_override: "ContentType | None" = None,
        target_format: str | None = None,
    ) -> AsyncIterator[str]:
        """長文生成のエントリポイント。トークンを yield する。

        Args:
            instruction: ユーザーの指示
            session_id: セッションID
            mode: "coding" | "chat"
            on_step: SSEステップ通知コールバック
            existing_content: 追記モード時の既存ファイル内容
            file_context_block: ユーザー添付ファイルを整形したブロック。
                指定時は plan コンテキストの参考情報チャネルへ合流させる。
            long_form_mode: 出力モード。
                :attr:`LongFormMode.CONTINUE` (既定) は従来挙動。
                :attr:`LongFormMode.EXPAND` / :attr:`LongFormMode.SPLIT` は
                planning プロンプトと unit イベントを切り替える (P2/P3 で実装)。
            content_type_override: 指定時は ``detect_content_type`` をスキップして
                強制する。用途例: ドキュメント品質ゲートのテスト
                (test_document_gate.py) で TEXT 判定を強制する場合。staged
                コーディング (backend/free/loop/staged/) は eb2fca3 以降
                direct_codegen 経由の単発呼出しに移行しており、本 override は
                使用しない。
            target_format: 出力先拡張子 (``.docx`` / ``.pptx`` / ``.xlsx`` 等)。
                ``document_quality_enabled`` の決定論ドキュメント品質ゲートが対象
                形式を判定するために使う。未指定/非ドキュメント形式ではゲート非適用。

        Yields:
            生成トークン文字列
        """
        t_start = time.monotonic()
        # リクエストモードを保持 (_effective_context_size が実窓解決に参照)
        self._mode = mode
        # 出力先形式を保持 (document_quality ゲートの形式判定に参照)
        self._target_format = (target_format or "").lower()
        # 前回 generate() の確定本文が残ると finalize が古い本文を配信し得るため、
        # リクエスト冒頭で None に戻す (現状は per-request インスタンスだが防御的に)。
        self.last_text_output = None

        # 1. コンテンツ種別判定 (override 指定時は検出をスキップ)
        content_type = content_type_override or detect_content_type(instruction, mode)
        # コンシューマ (chat_streaming) が生成途中でも code/text を判定できるよう、
        # 確定した content_type をインスタンス属性に保持する (リクエストごとの一時
        # インスタンスなので並行リクエストでも競合しない)。
        self.last_content_type = content_type.value

        # 2. メモリ・RAG からコンテキスト収集
        # prefetched_rag があれば retriever を呼ばず取得済みチャンクを再利用する。
        context = await self._gather_context(
            instruction,
            prefetched_rag=prefetched_rag,
            file_context_block=file_context_block,
            mode=mode,
        )
        if existing_content:
            context["existing_content"] = existing_content
        # 出力モード (EXPAND / SPLIT 等) は strategy.create_plan() に
        # プロンプト分岐用に渡す。CONTINUE は従来挙動と互換。
        context["long_form_mode"] = long_form_mode

        # 3-5. 計画 + 予算算出 + 依存順ソート
        plan, budget = await self._build_plan_for_generation(
            instruction, context, content_type, mode,
        )

        # 指示単独では主題が決定できず、計画側が確認を求めた場合はユニット生成
        # を一切行わずそのまま確認質問を返す。【関連メモリ】の話題を主題として
        # 誤って流用したまま生成・書込みしてしまう事故を防ぐ
        # (2026-07-22 ライブ検証で判明: 「文書が欲しい」等の主題無し依頼が、
        # 直前の無関係な会話の話題で長文ドキュメントを生成・書込みしていた)。
        if plan.needs_clarification:
            self.last_needed_clarification = True
            question = plan.clarification_question or self._default_clarification_question()
            logger.info(
                "long_form: instruction has no discernible topic, asking for "
                "clarification instead of generating (instruction=%.80s)",
                instruction,
            )
            yield question
            return

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

            # ユニット別 RAG (long_form.rag_per_unit)。plan 用の一括ブロックでは
            # なく、このユニットに効く参考情報だけを絞って渡す。
            rolling.unit_rag = await self._build_unit_rag(label, prefetched_rag, mode)

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

            # 8.1 ドキュメント品質モード時は改稿済みユニットから確定本文を組み直す。
            # 生ストリーム (full_response) は revise トークンを末尾に二重追記するため
            # file/editor 出力には使えない。CODE の last_code_output と対称の TEXT 版。
            if (
                content_type == ContentType.TEXT
                and self.config.get("long_form", {}).get(
                    "document_quality_enabled", False,
                )
                and is_document_format(self._target_format)
                and rolling.generated_units
            ):
                self.last_text_output = "\n\n".join(rolling.generated_units)

        # 8.5 検証ゲート付きコードリペア (CODE のみ)。review 後の generated_units を
        # assemble → 検証 → assist 修正 → 再検証し、last_code_output に保持する。
        # 総時間超過時も最終出力品質のため実行する (max_repair_rounds で有界)。
        await self._repair_generated_code(rolling, content_type, on_step)

        # 9. 出力検証。CODE は AST 検証 (リペア後 / Python のみ)、TEXT は決定論の
        # 品質ゲート (残 review issue / 目標文字数比 / 文重複率)。TEXT 側が無いと
        # validation_errors が常に 0 になり、壊れた散文が success として学習される。
        if content_type == ContentType.TEXT:
            validation_errors, _warning_count = self._validate_generated_text(
                rolling, content_type, on_step,
            )
        else:
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
        file_context_block: str | None = None,
        mode: str = "coding",
    ) -> dict:
        """メモリ3層 + RAG からコンテキストを収集

        ``prefetched_rag`` (search pipeline 取得済み ``(chunk_id, score, text)``)
        があれば retriever を呼ばずそれを再利用する。``None``/空なら従来の
        retriever 経路 (注入時のみ) に倒し、いずれも無ければ RAG なし。

        ``file_context_block`` (ユーザー添付ファイル) があれば参考情報チャネル
        (``context["rag"]``) の先頭へ合流させる。両 strategy が ``rag`` を参照
        コンテキストとして消費するため、strategy 側の変更を要さない。
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
                    instruction, top_k=RAG_MAX_CHUNKS, mode=mode,
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

        # 添付ファイルを参考情報チャネルへ合流 (RAG チャンクより前に置く)。
        if file_context_block:
            existing_rag = context["rag"]
            context["rag"] = (
                f"{file_context_block}\n---\n{existing_rag}"
                if existing_rag else file_context_block
            )

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

    def _fit_short_term(self, text: str) -> str:
        """ローリング短期コンテキストを ``long_form.rolling_short_term_chars``
        文字以内に収める (トークン予算とは独立した文字数上限)。
        """
        limit = int(
            self.config.get("long_form", {}).get("rolling_short_term_chars", 1000),
        )
        if limit <= 0 or len(text) <= limit:
            return text
        return text[-limit:]

    async def _build_unit_rag(
        self,
        unit_label: str,
        prefetched_rag: list[tuple[str, float, str]] | None,
        mode: str,
    ) -> str:
        """ユニット単位の参考情報ブロックを組み立てる。

        ``long_form.rag_per_unit`` が有効なときだけ働き、無効なら空文字を返す
        (ユニットプロンプトの参考情報は「(なし)」のまま)。

        - retriever + embedder が注入されていれば、ユニット名で実検索する
        - 未注入 (チャット経路の既定) なら、plan 用に取得済みの
          ``prefetched_rag`` からユニット名との語句重なりで再選抜する

        いずれも ``long_form.rag_top_k_per_unit`` 件までに絞る。
        """
        lf = self.config.get("long_form", {})
        if not lf.get("rag_per_unit", True):
            return ""
        top_k = int(lf.get("rag_top_k_per_unit", 3))
        if top_k <= 0 or not unit_label:
            return ""

        if self.retriever is not None and self.embedder is not None:
            try:
                hits = await self.retriever.search(unit_label, top_k=top_k, mode=mode)
            except Exception as e:
                logger.warning("per-unit RAG search failed: %s", e)
                return ""
            return "\n---\n".join(
                f"[{source}] (score={score:.2f})\n{text[:RAG_CHUNK_CHAR_CAP]}"
                for text, score, source in hits[:top_k]
            )

        if not prefetched_rag:
            return ""
        ranked = sorted(
            prefetched_rag,
            key=lambda c: (_lexical_overlap(unit_label, c[2]), c[1]),
            reverse=True,
        )
        return self._format_prefetched_rag(ranked[:top_k])

    async def _update_rolling_context(
        self, rolling: RollingContext, text: str, content_type: ContentType,
    ) -> None:
        """ローリングコンテキストを更新"""
        # 共通: 直前ユニットの末尾を保持
        rolling.short_term = self._fit_short_term(
            truncate_tail(text, rolling.budget.short_term),
        )

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
        """追加生成用のシンプルな messages を構築する (指示文は locale 追従)。

        続き生成なので出力言語は元テキストへ追従させる (locale で上書きしない)。
        """
        if get_locale() == "en":
            system = (
                "You are a writer continuing a piece of text. "
                "Continue the text below naturally, in the same language and "
                "style as the original. Output only the body text - no "
                "headings, no meta commentary."
            )
            user = (
                f"Continue the following text with about {remaining} "
                f"characters.\n\n{tail}\n\nContinuation:"
            )
        else:
            system = (
                "あなたは文章の続きを書く作家です。"
                "以下の文章の続きを、元の文章と同じ言語・文体で自然に書いてください。"
                "見出し行やメタ的な記述は不要です。本文のみを出力してください。"
            )
            user = (
                f"以下の文章の続きを{remaining}文字程度書いてください。\n\n"
                f"{tail}\n\n続き:"
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
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
            except Exception as e:
                logger.warning("Extend generation failed: %s", e)
                break

            if not ext_text.strip():
                logger.info(
                    "Extend round %d: empty output, stopping", extend_count,
                )
                break

            rolling.generated_units.append(ext_text)
            rolling.short_term = self._fit_short_term(ext_text)
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


# 1 ユニットあたりの分割数上限。estimated_tokens は LLM 由来 (parse 時に
# strategy_common._ESTIMATED_TOKENS_MAX で切り詰め済みだが、念のため分割
# ループ自体にも上限を設ける)。これを超える指定は現実的な単一セクション
# 規模を逸脱しており、同期ループでイベントループを長時間ブロックする
# (実運用で発生したハングの直接原因)。
_MAX_SPLITS_PER_UNIT = 50


#: 続きユニットに必ず添える継続指示。内容仕様ゼロの定型文単独だと直前段落の
#: 逐語再掲が起きるため、再掲禁止を明示する (2026-07-15: 7 ファイルで冒頭
#: 段落の末尾再掲が発生)。
_CONTINUATION_NOTE = "前の文章の続きだけを書く。既に書いた文や冒頭の宣言文を再掲しない。"


def _chunk_evenly(items: list[str], n: int) -> list[list[str]]:
    """リストを n 個の連続チャンクにほぼ等分する (先頭側が大きい)。"""
    if n <= 1:
        return [list(items)]
    base, extra = divmod(len(items), n)
    chunks: list[list[str]] = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(list(items[idx:idx + size]))
        idx += size
    return chunks


def _split_oversized_text_units(
    plan: "GenerationPlan",
    unit_target_tokens: int,
    max_units: int | None = None,
) -> "GenerationPlan":
    """大きすぎるテキストユニットを複数ユニットに分割する

    ローカルLLMは1回の生成で安定して出力できるトークン数に限界がある。
    estimated_tokens が分割閾値を超えるユニットを分割し、
    各ユニットがLLMの安定出力範囲内に収まるようにする。

    - 分割閾値にはヒステリシス (target の 1.5 倍かつ最低 600) を設ける。
      parse_plan は estimated_tokens を最低 200 にクランプするため、
      L1 進化で unit_target_tokens が下限に張り付くと全ユニットが機械的に
      倍分割される (2026-07-15: 13→26 units / 464 秒)。閾値の床がこれを防ぐ。
    - ``max_units`` 指定時は分割後の総ユニット数が上限を超えないよう
      実効ターゲットを底上げする (plan 時の truncate は分割前にしか
      効かないため)。
    - 親ユニットの key_points をサブユニットへ連続分配し、続きユニットが
      内容仕様ゼロにならないようにする (逐語再掲の抑止)。
    """
    split_threshold = max(int(unit_target_tokens * 1.5), 600)

    if max_units and max_units > 0:
        text_total = sum(
            u.estimated_tokens for u in plan.units if isinstance(u, SectionPlan)
        )
        floor_target = math.ceil(text_total / max_units)
        if floor_target > unit_target_tokens:
            logger.info(
                "Raising effective unit target %d -> %d to keep total units "
                "within max_units=%d",
                unit_target_tokens, floor_target, max_units,
            )
            unit_target_tokens = floor_target
            split_threshold = max(int(unit_target_tokens * 1.5), 600)

    new_units: list[CodeUnit | SectionPlan] = []
    for unit in plan.units:
        if not isinstance(unit, SectionPlan):
            new_units.append(unit)
            continue

        if unit.estimated_tokens <= split_threshold:
            new_units.append(unit)
            continue

        # 分割数を算出 (異常値によるイベントループ長時間ブロックを防ぐため上限を設ける)
        n_splits = min(
            math.ceil(unit.estimated_tokens / unit_target_tokens),
            _MAX_SPLITS_PER_UNIT,
        )
        tokens_per_split = unit.estimated_tokens // n_splits
        point_chunks = _chunk_evenly(unit.key_points, n_splits)

        for i in range(n_splits):
            chunk = point_chunks[i] if i < len(point_chunks) else []
            if i == 0:
                # 最初のサブユニット: 分配された key_points (空なら全量) を引き継ぐ
                sub = SectionPlan(
                    heading=unit.heading,
                    key_points=chunk or unit.key_points,
                    estimated_tokens=tokens_per_split,
                )
            else:
                # 後続サブユニット: 担当分の key_points のみ。
                # heading に「（続き N）」を埋めると、その文字列が
                # strategy_common のセクション一覧経由で全ユニットのプロンプトに
                # 注入され、本文へそのまま出力される (2026-07-25 実測)。
                # 継続指示も key_points (=「本文に含めるべき要点」) ではなく
                # system プロンプト側で与える。
                sub = SectionPlan(
                    heading=unit.heading,
                    key_points=list(chunk),
                    estimated_tokens=tokens_per_split,
                    sub_index=i + 1,
                )
            new_units.append(sub)

        logger.info(
            "Split oversized unit '%s': %d tokens -> %d sub-units x %d tokens",
            unit.heading, unit.estimated_tokens, n_splits, tokens_per_split,
        )

    plan.units = new_units
    return plan
