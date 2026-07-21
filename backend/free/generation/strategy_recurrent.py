"""Recurrent 戦略（ローリングコンテキスト型 long-form 生成）

設計書 f_09_long_form_generation.md §6 準拠。
RecurrentGPT 方式のローリングコンテキストにより、メインモデルは生成 (`generate_unit`)
に専念し、計画 (`create_plan`) と要約再帰 (`update_summary`) はアシストモデルが
担当する
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from backend.exceptions import UnitGenerationError
from backend.free.generation.models import (
    CodeUnit,
    ContentType,
    GenerationPlan,
    LongFormMode,
    SectionPlan,
    extract_target_chars,
)
from backend.free.generation.rolling_context import RollingContext
from backend.free.generation.strategy_common import (
    build_code_unit_messages,
    build_text_unit_messages,
    excerpt_continuation_content,
    excerpt_for_expand,
    fallback_plan,
    finalize_plan_units,
    generate_plan_json,
    parse_plan,
    resolve_max_units,
)
from backend.free.generation.token_budget import TokenBudget
from backend.free.llm.utils import extract_content
from backend.i18n_helper import prose_language_name

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.llm.local_client import LocalClient

logger = logging.getLogger("backend.free.generation.strategy_recurrent")


# ── プロンプトテンプレート ──

_PLAN_PROMPT = """\
以下の指示に基づき、実装計画をJSON形式で生成してください。

【ユーザー指示】{instruction}
【参考情報】{context}

{format_instruction}

JSON のみ出力してください。"""

_CODE_FORMAT = """\
出力形式:
{{
  "content_type": "code",
  "title": "...",
  "target_length": 3000,
  "global_context": "Python 3.12+, type hints必須",
  "constraints": [],
  "units": [
    {{
      "kind": "function",
      "name": "...",
      "file_path": "...",
      "spec": "...",
      "depends_on": [],
      "estimated_tokens": 500
    }}
  ]
}}"""

_TEXT_FORMAT = """\
出力形式:
{{
  "content_type": "text",
  "title": "...",
  "target_length": 5000,
  "global_context": "文書の種類・文体・語り口（例: 技術文書、丁寧体）",
  "constraints": [],
  "units": [
    {{
      "heading": "...",
      "key_points": ["..."],
      "estimated_tokens": 500
    }}
  ]
}}

global_context には文書の種類と文体を具体的に記述してください。
key_points には、生成すべき具体的な内容を書いてください。
「〜について書く」のようなメタ的な記述ではなく、実際に含める内容を指定してください。"""

_CONTINUATION_PLAN_PROMPT = """\
既存テキストに基づき、ユーザー指示に沿った計画をJSON形式で生成してください。

【ユーザー指示】{instruction}
【既存テキスト】
{existing_content}
【参考情報】{context}

重要な制約:
- まず既存テキストの種類（小説、報告書、技術文書、議事録、マニュアル等）を判別してください。
- 既存テキストの文体（語り口、文末表現、語彙のレベル、構造的特徴）を分析し、\
global_context に具体的に記述してください。
- key_points には実際に生成すべき具体的な内容を書いてください。\
メタ的な記述ではなく、生成する内容そのものを指定してください。

{format_instruction}

JSON のみ出力してください。"""

# EXPAND/SPLIT モード (Recurrent 戦略): CogWriter と同じ意図で、既存テキストを
# 機能ごとに分解した詳細仕様書のアウトラインを生成させる。
_EXPAND_PLAN_PROMPT = """\
既存テキストに含まれる機能/トピックを抽出し、機能ごとに分解した詳細仕様書の
計画を JSON 形式で生成してください。

【ユーザー指示】{instruction}
【既存テキスト】
{existing_content}
【参考情報】{context}

重要な制約:
- 既存テキストを **要約せず、機能ごとに詳細化** してください。
- 5〜12 個の機能/コンポーネントを抽出し、それぞれを 1 unit として列挙してください。
- 各 unit の heading は機能名、key_points にはその機能の **詳細仕様**
  (入出力、データ構造、アルゴリズム、エラー条件、テスト観点) を具体的に箇条書き。
- estimated_tokens は各 unit で {per_unit_tokens} 程度を目安。
- target_length は全 unit 合計で {target_length} を指定してください。

{format_instruction}

JSON のみ出力してください。"""

_TEXT_UNIT_USER = """\
# 文書構成
タイトル: {title}
セクション一覧: {section_headings}

# これまでの要約
{long_term_summary}

# 直前セクション末尾
{short_term}

# 現在のセクション
見出し: {heading}
含めるべき要点: {key_points}

# 参考資料
{rag_context}

本文のみを出力してください（見出し行・メタ解説・執筆意図の説明は不要）:"""

_SUMMARY_UPDATE_PROMPT = """\
以下の既存要約に新セクションの内容を統合し、{budget}トークン以内の要約に更新してください。

【既存要約】{current_summary}
【新セクション】{new_section}

更新要約:"""


# ── Recurrent 戦略本体 ──

class RecurrentStrategy:
    """Recurrent 戦略（ローリングコンテキスト型）

    メインモデルはユニット逐次生成に専念し、計画立案 / 要約再帰 / その他の
    判定処理は ``assist_client`` (アシストモデル) に委ねる

    RecurrentGPT 方式のローリングコンテキストにより、固定サイズのコンテキスト
    ウィンドウで任意長の出力を生成する。CogWriter とは異なり ``review`` /
    ``revise_unit`` フェーズを持たないため、計画後の生成は単一パス。
    """

    def __init__(
        self,
        main_client: LocalClient,
        assist_client: AssistModelClient | None,
        config: dict,
        debug_logger=None,
        generation_params: dict | None = None,
    ):
        self.main_client = main_client
        # assist_client は CLAUDE.md §1 に従い計画 / 要約再帰の
        # 判定系処理を担う。``None`` (assist health_check 失敗の degraded
        # mode) の場合は ``fallback_plan`` 単一ユニット計画 + 新セクション
        # 末尾保持にフォールバックする (ベースモデル経由の JSON 抽出は
        # 行わない)。
        self.assist_client = assist_client
        self.config = config
        self._debug_logger = debug_logger
        self._lf_config = config.get("long_form", {})
        self._generation_params = generation_params or {}

    @property
    def client(self) -> LocalClient:
        """メインクライアント（Orchestrator の update_summary で使用）"""
        return self.main_client

    async def create_plan(
        self,
        instruction: str,
        context: dict,
        content_type: ContentType,
        budget: TokenBudget,  # noqa: ARG002
    ) -> GenerationPlan:
        """アシストモデルで計画を JSON 生成

        ``assist_client=None`` の degraded mode では即座に
        ``fallback_plan`` (単一ユニット) を返す。
        """
        t0 = time.monotonic()
        ctx_text = context.get("rag", "") or context.get("memory", "") or ""
        existing_content = context.get("existing_content", "")

        format_instruction = (
            _CODE_FORMAT if content_type == ContentType.CODE else _TEXT_FORMAT
        )
        # テキスト計画の記述言語 (locale 追従)。テンプレートは CODE/継続と共有の
        # ため呼出側で注入する。継続モードは既存テキストの言語追従が正なので
        # 付加しない (新規/EXPAND のみ)。
        lang_line = ""
        if content_type == ContentType.TEXT:
            lang_line = (
                f"\nユーザー指示に言語の明示指定が無い限り、"
                f"title・global_context・heading・key_points は"
                f"{prose_language_name()}で書いてください。"
            )

        long_form_mode: LongFormMode = context.get(
            "long_form_mode", LongFormMode.CONTINUE,
        )

        if existing_content and content_type == ContentType.TEXT and long_form_mode in (
            LongFormMode.EXPAND, LongFormMode.SPLIT,
        ):
            # EXPAND/SPLIT モード: 詳細仕様書として再構成
            excerpt = excerpt_for_expand(existing_content)
            user_target = extract_target_chars(instruction, default=0)
            target_length = max(user_target, int(len(existing_content) * 1.5), 3000)
            per_unit_tokens = max(600, target_length // 12)
            prompt = _EXPAND_PLAN_PROMPT.format(
                instruction=instruction,
                existing_content=excerpt,
                context=ctx_text[:500],
                target_length=target_length,
                per_unit_tokens=per_unit_tokens,
                format_instruction=format_instruction + lang_line,
            )
        elif existing_content and content_type == ContentType.TEXT:
            # 追記モード: 既存テキストを含む専用プロンプト
            excerpt = excerpt_continuation_content(existing_content)
            prompt = _CONTINUATION_PLAN_PROMPT.format(
                instruction=instruction,
                existing_content=excerpt,
                context=ctx_text[:500],
                format_instruction=format_instruction,
            )
        else:
            prompt = _PLAN_PROMPT.format(
                instruction=instruction,
                context=ctx_text[:500],  # コンテキスト予算を節約
                format_instruction=format_instruction + lang_line,
            )

        max_units = resolve_max_units(self._lf_config, long_form_mode)

        # content_type に応じた schema 選択・assist None / 例外フォールバックは共通化
        plan_telemetry: dict = {}
        data = await generate_plan_json(
            self.assist_client, prompt, content_type, telemetry=plan_telemetry,
        )
        if plan_telemetry.get("truncated"):
            logger.warning(
                "Plan JSON truncated (content_type=%s): planned units may be "
                "missing; output can be incomplete",
                content_type.value,
            )

        if not data or "units" not in data:
            logger.warning(
                "Plan JSON parse failed, falling back to single unit"
            )
            plan = fallback_plan(instruction, content_type)
        else:
            plan = parse_plan(data, content_type, instruction)

        # ユニット数上限 + コード用依存順ソート (共通後処理)
        finalize_plan_units(plan, max_units, content_type)

        elapsed = time.monotonic() - t0
        logger.info(
            "Plan created: content_type=%s, units=%d, elapsed=%.2fs",
            content_type.value, len(plan.units), elapsed,
        )
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "plan",
                "strategy": "recurrent",
                "content_type": content_type.value,
                "units_count": len(plan.units),
                "elapsed_sec": round(elapsed, 3),
            })

        return plan

    async def generate_unit(
        self,
        unit: CodeUnit | SectionPlan,
        rolling: RollingContext,
        content_type: ContentType,
    ) -> AsyncIterator[str]:
        """メインモデルでユニットを逐次生成（ストリーミング）"""
        config_max = self._lf_config.get("unit_max_tokens", 2000)
        unit_max_tokens = max(config_max, int(unit.estimated_tokens * 1.5))

        if content_type == ContentType.CODE:
            assert isinstance(unit, CodeUnit)
            messages = build_code_unit_messages(unit, rolling)
        else:
            assert isinstance(unit, SectionPlan)
            messages = self._build_text_unit_messages(unit, rolling)

        try:
            gen_kwargs: dict = {
                "stream": True,
                "temperature": self._generation_params.get("temperature", 0.7),
                "max_tokens": unit_max_tokens,
                "id_slot": self.main_client.chat_slot,
            }
            for k in ("top_p", "top_k", "presence_penalty"):
                if k in self._generation_params:
                    gen_kwargs[k] = self._generation_params[k]
            stream = await self.main_client.generate(messages, **gen_kwargs)
            async for token in stream:
                yield token
        except Exception as e:
            logger.error("Unit generation failed: %s", e)
            raise UnitGenerationError(
                f"Failed to generate unit: {e}"
            ) from e

    async def update_summary(
        self,
        current_summary: str,
        new_section: str,
        budget: TokenBudget,
    ) -> str:
        """アシストモデルで既存要約 + 新セクションを再要約

        ``assist_client=None`` の degraded mode では新セクション末尾を
        フォールバックとして返す (ベースモデル経由の JSON 抽出を避ける)。
        """
        if self.assist_client is None:
            logger.warning(
                "Recurrent update_summary: assist_client is None, "
                "returning new section tail as fallback",
            )
            return new_section[-300:]

        prompt = _SUMMARY_UPDATE_PROMPT.format(
            budget=budget.skeleton_or_summary,
            current_summary=current_summary or "(なし)",
            new_section=new_section[-500:],
        )
        try:
            result = await self.assist_client.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=budget.skeleton_or_summary,
                temperature=0.3,
                purpose="summarize",
            )
            return extract_content(result).strip()
        except Exception as e:
            logger.warning("Summary update failed: %s", e)
            # フォールバック: 新セクションの末尾を返す
            return new_section[-300:]

    # ── プロンプト構築 ──

    def _build_text_unit_messages(
        self,
        unit: SectionPlan,
        rolling: RollingContext,
    ) -> list[dict]:
        """テキスト生成用のメッセージを構築 (共通スケルトンに委譲)。

        Recurrent は長期要約スロットを持つため
        ``include_long_term_summary=True``。
        """
        return build_text_unit_messages(
            unit, rolling, _TEXT_UNIT_USER, include_long_term_summary=True,
        )
