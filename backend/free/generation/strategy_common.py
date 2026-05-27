"""Long-form 戦略の共通ユーティリティ

`strategy_recurrent.py` と `strategy_cogwriter.py` で AST 完全一致していた
3 関数とプロンプト定数を集約する

責務:
- 計画 JSON → :class:`GenerationPlan` 変換 (:func:`parse_plan`)
- JSON 解析失敗時の単一ユニットフォールバック (:func:`fallback_plan`)
- コードユニット生成用メッセージ構築 (:func:`build_code_unit_messages`)
- 共通プロンプトテンプレート定数

`_build_text_unit_messages` は両戦略間で差分 (recurrent 側のみ要約スロット
``long_term_summary`` を含む) があるため、本モジュールでは扱わない。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.free.generation.models import (
    CodeUnit,
    ContentType,
    GenerationPlan,
    SectionPlan,
    chars_to_tokens,
    extract_target_chars,
)

logger = logging.getLogger("backend.free.generation.strategy_common")

if TYPE_CHECKING:
    from backend.free.generation.rolling_context import RollingContext


# ── 共通プロンプトテンプレート ──

CODE_UNIT_SYSTEM = """\
あなたはPythonプログラマーです。以下の計画に従い、指定ユニットのコードを生成してください。
{global_context}"""

CODE_UNIT_USER = """\
# 実装計画
ファイル: {file_path}
全ユニット: {unit_names}

# 生成済みコード構造
{skeleton}

# 直前の生成コード
{short_term}

# 現在のユニット
種別: {kind}
名前: {name}
仕様: {spec}
依存: {depends_on}

# 参考コード
{rag_context}

コードのみ出力してください:"""

TEXT_UNIT_SYSTEM = """\
以下の計画に従い、指定セクションの本文を生成してください。
- 見出し行（# や ## など）は出力しないでください。本文のみを出力してください。
- 文章は自然な段落で区切り、1文ごとに改行を入れないでください。
- 本文の内容そのものだけを出力してください。\
執筆意図・方針・プロセスの説明などメタ的な記述は一切含めないでください。
- このセクションの目標文字数は約{unit_target_chars}文字です。必ずこの文字数に近い量を生成してください。\
短すぎる出力は不可です。
{global_context}"""

TEXT_UNIT_CONTINUATION_SYSTEM = """\
既存テキストを踏まえ、ユーザー指示に沿った内容を生成してください。以下のルールを厳守してください。
- 既存テキストの文体（語り口、文末表現、語彙、構造）を正確に維持してください。
- 生成する内容そのものだけを出力してください。\
メタ的な記述（「確認したところ」「〜する予定」「〜について記述する」等）は絶対に含めないでください。
- 見出し行（# や ## など）は出力しないでください。
- 文章は自然な段落で区切り、1文ごとに改行を入れないでください。
- 「直前テキスト末尾」から自然に繋がるように書いてください。
- このセクションの目標文字数は約{unit_target_chars}文字です。必ずこの文字数に近い量を生成してください。\
短すぎる出力は不可です。
{global_context}"""


# ── 計画パース ──

def _to_int(value: object, default: int) -> int:
    """JSON 由来の数値フィールドを安全に int 化する。

    ``json_extract`` の戦略 4 (``json_repair`` フォールバック) は型強制を
    行わないため、LLM が ``"500"`` のように文字列で返した数値がそのまま
    dict に残ることがある。下流の算術 (sum / 比較 / 乗算) が ``int + str``
    で落ちないよう parse 時点で int に揃える。
    """
    if isinstance(value, bool):
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_plan(
    data: dict,
    content_type: ContentType,
    instruction: str = "",
) -> GenerationPlan:
    """JSON 辞書から :class:`GenerationPlan` を構築する。

    ユーザー指示の文字数指定 (``extract_target_chars``) が LLM が返した
    ``target_length`` より優先される。``target_length`` に対して
    ``estimated_tokens`` の合計が不足している場合は比例スケーリングで補正する。
    """
    units_raw = data.get("units", [])
    units: list[CodeUnit | SectionPlan] = []
    for u in units_raw:
        # Defense in depth: 上流 (json_extract) が壊れた応答から非 dict 要素を
        # units として渡してきた場合に AttributeError で全体クラッシュさせず、
        # 警告して当該要素をスキップする。
        if not isinstance(u, dict):
            logger.warning(
                "parse_plan: skipping non-dict unit element (type=%s, repr=%.80s)",
                type(u).__name__, repr(u),
            )
            continue
        if content_type == ContentType.CODE:
            units.append(CodeUnit(
                kind=u.get("kind", "function"),
                name=u.get("name", "unknown"),
                file_path=u.get("file_path", ""),
                spec=u.get("spec", ""),
                depends_on=u.get("depends_on", []),
                estimated_tokens=_to_int(u.get("estimated_tokens"), 500),
            ))
        else:
            units.append(SectionPlan(
                heading=u.get("heading", ""),
                key_points=u.get("key_points", []),
                estimated_tokens=_to_int(u.get("estimated_tokens"), 500),
                file_name=u.get("file_name") or None,  # SPLIT モード時のみ非 None
            ))

    # ユーザー指示の文字数指定を優先（LLM の計画値より信頼できる）
    user_target = extract_target_chars(instruction, default=0)
    plan_target = _to_int(data.get("target_length"), 0)
    target_length = user_target if user_target > 0 else plan_target

    # target_length に基づき estimated_tokens を補正
    if target_length > 0 and units:
        target_tokens = chars_to_tokens(target_length)
        total_estimated = sum(u.estimated_tokens for u in units) or 1
        if total_estimated < target_tokens:
            scale = target_tokens / total_estimated
            for u in units:
                u.estimated_tokens = max(int(u.estimated_tokens * scale), 200)

    return GenerationPlan(
        content_type=content_type,
        title=data.get("title", ""),
        target_length=target_length,
        global_context=data.get("global_context", ""),
        constraints=data.get("constraints", []),
        units=units,
    )


def fallback_plan(
    instruction: str,
    content_type: ContentType,
) -> GenerationPlan:
    """JSON 解析失敗時の単一ユニット フォールバック計画を返す。"""
    target_length = extract_target_chars(instruction, default=1000)
    fallback_tokens = max(chars_to_tokens(target_length), 1000)

    if content_type == ContentType.CODE:
        unit: CodeUnit | SectionPlan = CodeUnit(
            kind="function",
            name="main",
            file_path="output.py",
            spec=instruction,
            depends_on=[],
            estimated_tokens=fallback_tokens,
        )
    else:
        unit = SectionPlan(
            heading="本文",
            key_points=[instruction],
            estimated_tokens=fallback_tokens,
        )
    return GenerationPlan(
        content_type=content_type,
        title="",
        target_length=target_length,
        global_context="",
        constraints=[],
        units=[unit],
    )


# ── プロンプト構築 ──

def build_code_unit_messages(
    unit: CodeUnit,
    rolling: RollingContext,
) -> list[dict]:
    """コードユニット生成用のメッセージ列を構築する。

    両戦略 (Recurrent / CogWriter) で完全に共通の純粋関数。`self` には依存しない。
    """
    budget = rolling.budget
    plan = rolling.plan

    unit_names = ", ".join(
        u.name for u in plan.units if isinstance(u, CodeUnit)
    )

    skeleton_text = ""
    if rolling.skeleton:
        skeleton_text = budget.fit_content(
            "skeleton_or_summary",
            rolling.skeleton.to_prompt(budget.skeleton_or_summary),
        )

    short_term = budget.fit_content("short_term", rolling.short_term)

    system = budget.fit_content(
        "system_prompt",
        CODE_UNIT_SYSTEM.format(global_context=plan.global_context),
    )
    user = CODE_UNIT_USER.format(
        file_path=unit.file_path,
        unit_names=unit_names,
        skeleton=skeleton_text or "(なし)",
        short_term=short_term or "(なし)",
        kind=unit.kind,
        name=unit.name,
        spec=budget.fit_content("unit_spec", unit.spec),
        depends_on=", ".join(unit.depends_on) or "(なし)",
        rag_context="(なし)",  # RAG は Orchestrator 側で注入
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
