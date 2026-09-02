"""欠陥率ベースの共有 fitness (Level 1 evolver 群の共通土台)

PolicyParamEvolver / GenerationParamEvolver / PromptEvolver が各自に持っていた
「観測された欠陥の重み付き率」を 1 箇所へ集約する。**加点シグナルを使わない**
のが要点 — ``conversation_ended`` は :meth:`ExperienceBuffer._mark_loaded_conversations_ended`
が読み込み時に全エントリへ立てるため構造的に恒真へ寄り (実測 205/206)、
``turn_outcome`` も実測 206/206 が ``"success"`` だった。これらを主項に置くと
fitness が上限へ張り付いて選択圧が消える (docs/f_04 §8 禁則 7)。

派生シグナル (``turn_outcome == "failed"`` / long_form 検証失敗) は
:data:`DERIVED_DEFECTS` の合成キーとして扱い、呼出側が重みを足せる。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

#: 欠陥シグナルと重み (raw ``signals`` キーをそのまま truthiness 判定する)。
DEFECT_WEIGHTS: dict[str, float] = {
    "user_correction": 1.0,
    "assistant_self_retraction": 1.0,
    "rephrased_query": 0.6,
    "tool_routing_false_negative": 0.5,
    "tool_routing_false_positive": 0.5,
}

#: raw キーでは表せない派生欠陥。``weights`` にこのキーを含めた呼出側だけが使う。
DERIVED_DEFECTS: dict[str, Callable[[Mapping], bool]] = {
    # 出力が壊れていたターンの SSOT (docs/f_04 §2.5 / §8 禁則 9)。
    "turn_outcome_failed": lambda s: s.get("turn_outcome") == "failed",
    # 長文生成の成果物が検証で落ちた (create モードの主要な失敗シグナル)。
    "long_form_failed": lambda s: bool(
        s.get("long_form_used") and s.get("long_form_success") is False,
    ),
}


def signal_is_defect(signals: Mapping, key: str) -> bool:
    """``signals`` 上で欠陥キー ``key`` が立っているか (派生キー対応)。"""
    derived = DERIVED_DEFECTS.get(key)
    if derived is not None:
        return derived(signals)
    return bool(signals.get(key))


def defect_rate_fitness(
    experiences: list[dict],
    *,
    weights: Mapping[str, float] | None = None,
    window: int | None = None,
) -> float | None:
    """観測された欠陥の重み付き率から fitness を返す (1.0 = 欠陥なし)。

    Args:
        experiences: 経験 dict のリスト (``signals`` を持つ)。
        weights: 欠陥キー → 重み。省略時は :data:`DEFECT_WEIGHTS`。
        window: 指定時は末尾 ``window`` 件のみを評価する (経験は時系列順)。

    Returns:
        ``[0.0, 1.0]`` の fitness。評価対象が空なら ``None`` (呼出側が中立値や
        skip へ倒す — 「欠陥なし」の 1.0 と区別する)。
    """
    if window is not None and window > 0:
        experiences = experiences[-window:]
    if not experiences:
        return None
    table = DEFECT_WEIGHTS if weights is None else weights
    defects = 0.0
    for e in experiences:
        signals = e.get("signals") or {}
        for key, weight in table.items():
            if signal_is_defect(signals, key):
                defects += weight
    return max(0.0, min(1.0, 1.0 - defects / len(experiences)))


def has_defect_signal(
    experiences: list[dict], *, weights: Mapping[str, float] | None = None,
) -> bool:
    """欠陥シグナルが 1 件でも立っているか (= 評価に使える分散があるか)。"""
    table = DEFECT_WEIGHTS if weights is None else weights
    return any(
        signal_is_defect(e.get("signals") or {}, key)
        for e in experiences
        for key in table
    )
