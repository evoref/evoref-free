"""文脈予算の唯一の配り手 (c_02 §6.3 / f_03 §7.1.2)。

**窓を決める主体を 1 つにする。** 以前は WorkingMemory (``working_max_tokens``)、
``build_messages`` (``_DYN_BLOCK_RESERVE`` 固定 1600 + system 実測)、
meta_cognitive のループ予算がそれぞれ独立に窓を計算し、設定値が構造的に
達成不能でも警告するだけだった (2026-09-03 監査: 毎ターン
``history is trimmed twice``)。ここで一度だけ配る。

配分 (モードの context_size から):

    generation_reserve  = llama.max_tokens (無ければ DEFAULT_GENERATION_RESERVE)
    system_max          = context_size × prompt.system_max_share
                          (レンダラが守る上限。実測が渡されればそちら)
    dyn_reserve         = 動的ブロックの上限の合計
                          (fewshot_dynamic + semmem + rag)。**実際の注入量ではない**
    fact_slate          = prompt.fact_slate_max_tokens
    working_max         = context_size − generation_reserve − system − dyn_reserve − fact_slate

純粋関数に近い (config dict を受けて計算するだけ)。
"""

from __future__ import annotations

from dataclasses import dataclass

#: ``llama.max_tokens`` が無いときの生成予約。``inference.DEFAULT_GENERATION_RESERVE``
#: と同値。循環 import を避けるためここに複製し、テストで一致を固定する。
DEFAULT_GENERATION_RESERVE = 512

#: WM の窓の下限。極端に小さい context_size で 0 になると会話が成立しない。
WORKING_MIN_TOKENS = 512

#: ``prompt`` セクション未ロード時の既定 (schemas/prompt.py と同値)。
_PROMPT_DEFAULTS: dict[str, int | float] = {
    "system_max_share": 0.30,
    "fewshot_core_max_tokens": 300,
    "fewshot_dynamic_max_tokens": 600,
    "semmem_max_tokens": 800,
    "rag_max_tokens": 200,
    "fact_slate_max_tokens": 200,
}


@dataclass(frozen=True, slots=True)
class PromptBudgets:
    """1 モードぶんの予算配分。"""

    context_size: int
    generation_reserve: int
    system_max_tokens: int
    system_tokens: int
    dyn_reserve: int
    fact_slate_tokens: int
    working_max_tokens: int


def _prompt_cfg(config: dict) -> dict:
    raw = (config.get("prompt") or {}) if isinstance(config, dict) else {}
    merged = dict(_PROMPT_DEFAULTS)
    for key in merged:
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = value
    return merged


def dynamic_reserve(config: dict) -> int:
    """動的ブロックの固定予約 = 各ブロック上限の合計。"""
    p = _prompt_cfg(config)
    return int(p["fewshot_dynamic_max_tokens"]) + int(p["semmem_max_tokens"]) + int(p["rag_max_tokens"])


def system_max_tokens(config: dict, context_size: int) -> int:
    """モードの context_size に対する静的 system の上限。"""
    return int(context_size * float(_prompt_cfg(config)["system_max_share"]))


def resolve_budgets(
    config: dict,
    context_size: int,
    *,
    system_tokens: int | None = None,
    generation_reserve: int | None = None,
) -> PromptBudgets:
    """予算を配る。

    Args:
        config: ロード済み config dict。
        context_size: そのモードの context_size (``resolve_context_size_for_mode``)。
        system_tokens: レンダ済み system の実測トークン。``None`` なら上限
            (``system_max_tokens``) を占有として見積もる — WM の窓を決める側は
            レンダ前に呼ぶので、「レンダラが守る上限」で計算するのが安全側。
        generation_reserve: ``llama.max_tokens``。``None`` なら config から解決。
    """
    if generation_reserve is None:
        raw = ((config.get("llama") or {}) if isinstance(config, dict) else {}).get("max_tokens")
        generation_reserve = int(raw) if isinstance(raw, int) and raw > 0 else DEFAULT_GENERATION_RESERVE
    sys_max = system_max_tokens(config, context_size)
    sys_used = sys_max if system_tokens is None else int(system_tokens)
    dyn = dynamic_reserve(config)
    slate = int(_prompt_cfg(config)["fact_slate_max_tokens"])
    working = context_size - generation_reserve - sys_used - dyn - slate
    return PromptBudgets(
        context_size=int(context_size),
        generation_reserve=int(generation_reserve),
        system_max_tokens=sys_max,
        system_tokens=sys_used,
        dyn_reserve=dyn,
        fact_slate_tokens=slate,
        working_max_tokens=max(WORKING_MIN_TOKENS, working),
    )
