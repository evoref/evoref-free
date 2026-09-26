"""欠陥率ベースの共有 fitness (Level 1 evolver 群の共通土台)

PolicyParamEvolver / GenerationParamEvolver / PromptEvolver が各自に持っていた
「観測された欠陥の重み付き率」を 1 箇所へ集約する。**加点シグナルを使わない**
のが要点 — ``conversation_ended`` は :meth:`ExperienceBuffer._mark_loaded_conversations_ended`
が読み込み時に全エントリへ立てるため構造的に恒真へ寄る。これを主項に置くと
fitness が上限へ張り付いて選択圧が消える (docs/f_04 §8 禁則 7)。

ラベルの出どころは **検証器** に寄せる (2026-09-21)。旧 :data:`DEFECT_WEIGHTS`
は字句ブール 5 種 (``user_correction`` / ``assistant_self_retraction`` /
``rephrased_query`` / ``tool_routing_false_positive`` /
``tool_routing_false_negative``) だったが、実データ 161 件で **5 種すべて 0 件**
だった。``has_defect_signal`` が恒に False を返すため
:class:`~backend.free.learning.policy_evolver.PolicyParamEvolver` と
:class:`~backend.free.learning.generation_param_evolver.GenerationParamEvolver`
は ``skipped_no_signal`` へ落ち続け、**一度も進化していなかった**。

同じ 161 件で ``turn_outcome == "failed"`` は 13 件立っており、その出どころは
:meth:`FeedbackCollector._derive_turn_outcome_with_reason` の決定論検証器
(算術矛盾 / 日本語の崩れ / ツール結果の無視 / 指示違反 …) である。字句の網目を
細かくする方向ではなく、**既に立っている検証器のラベルを読む**方向で直す
(不変則 #12 / #14 と同じ構え)。

失敗理由は :data:`OUTCOME_REASON_CHANNELS` で 4 チャネルへ畳む。1 件の失敗が
必ず 1 チャネルだけに属する分割なので、チャネル別に重みを変えても二重計上は
起きない。チャネルを分ける理由は **evolver ごとに責任範囲が違う** ため —
sampling パラメータ (temperature / top_p / top_k) は出力の崩れに効くが指示違反
には効かない。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

#: ``turn_outcome_reason`` の接頭辞 → 欠陥チャネル。理由文字列を組むのは
#: :meth:`FeedbackCollector._derive_turn_outcome_with_reason` と
#: :meth:`FeedbackCollector._apply_self_retraction` の 2 箇所だけで、ここは
#: その語彙に追従する。**前方一致**で引く (理由は ``"<接頭辞>: <詳細>"``)。
OUTCOME_REASON_CHANNELS: tuple[tuple[str, str], ...] = (
    # (1) 出力そのものが壊れた。sampling パラメータの寄与が大きい。
    ("broken JA spacing", "outcome_broken_output"),
    ("Chinese token leaked", "outcome_broken_output"),
    ("answer cut off", "outcome_broken_output"),
    # (2) 本文が自分自身 / 手元の事実と食い違う。プロンプトと推論の寄与が大きい。
    ("arithmetic contradiction", "outcome_contradiction"),
    ("conclusion contradiction", "outcome_contradiction"),
    ("response retracts", "outcome_contradiction"),
    ("retracted by assistant", "outcome_contradiction"),
    ("measured value contradiction", "outcome_contradiction"),
    ("tool result ignored", "outcome_contradiction"),
    ("date result ignored", "outcome_contradiction"),
    ("claimed completion while blocked", "outcome_contradiction"),
    ("fabricated count", "outcome_contradiction"),
    # (3) 明示された指示を守っていない。プロンプトの寄与が大きい。
    ("length constraint", "outcome_instruction"),
    ("output form", "outcome_instruction"),
    ("user echo", "outcome_instruction"),
    # (4) 実行そのものが失敗した。ルーティングとポリシーの寄与が大きい。
    ("all tasks failed", "outcome_execution"),
    ("no step credit", "outcome_execution"),
    ("routing false positive", "outcome_execution"),
)

#: 理由が :data:`OUTCOME_REASON_CHANNELS` のどれにも当たらない ``failed`` の
#: 行き先。理由が空 (旧レコード / 外部から直接 ``failed`` を書いた経路) も
#: ここへ落ちる。**失敗を取りこぼさない**ための受け皿で、ここが増えたら
#: チャネル表の更新漏れを疑う。
OUTCOME_CHANNEL_OTHER = "outcome_other"

#: 失敗チャネルの全体。``DEFECT_WEIGHTS`` 系の表を組む側が参照する。
OUTCOME_CHANNELS: tuple[str, ...] = (
    "outcome_broken_output",
    "outcome_contradiction",
    "outcome_instruction",
    "outcome_execution",
    OUTCOME_CHANNEL_OTHER,
)


def outcome_channel(signals: Mapping) -> str | None:
    """失敗ターンの欠陥チャネルを返す (失敗でなければ ``None``)。

    ``turn_outcome == "failed"`` のときだけチャネルを返す。``partial``
    (「一部のタスクが失敗」) は失敗として数えない — 完了した側の手本価値が
    残るため、成否の SSOT 側でも別扱いになっている。
    """
    if signals.get("turn_outcome") != "failed":
        return None
    reason = str(signals.get("turn_outcome_reason") or "")
    for prefix, channel in OUTCOME_REASON_CHANNELS:
        if reason.startswith(prefix):
            return channel
    return OUTCOME_CHANNEL_OTHER


def _channel_predicate(channel: str) -> Callable[[Mapping], bool]:
    """``channel`` に属する失敗かを判定する述語を返す。"""
    return lambda s: outcome_channel(s) == channel


#: raw キーでは表せない派生欠陥。``weights`` にこのキーを含めた呼出側だけが使う。
DERIVED_DEFECTS: dict[str, Callable[[Mapping], bool]] = {
    # 失敗チャネル (排他。1 件の失敗はちょうど 1 つに属する)。
    **{channel: _channel_predicate(channel) for channel in OUTCOME_CHANNELS},
    # 長文生成の成果物が検証で落ちた (create モードの主要な失敗シグナル)。
    # ``turn_outcome`` とは独立に立つ — 実測で long_form を使った 19 件のうち
    # 15 件が検証落ちしており、chat の失敗率 (8%) より一桁濃い。
    "long_form_failed": lambda s: bool(
        s.get("long_form_used") and s.get("long_form_success") is False,
    ),
}

#: 欠陥シグナルと重み。policy 進化 (``PolicyParamEvolver``) の既定表で、
#: 他の evolver はこれを土台に自分の責任範囲へ寄せた表を持つ。
#:
#: ``user_negative`` (ユーザーの 👎) を **policy 側にも入れる** (2026-09-21、
#: docs/f_04 §3.2.3 の「policy 進化の圧には使わない」を撤回)。本人が失敗と
#: 言った唯一の信号を外すと、閾値を緩めた分だけ検証器が拾えない失敗が
#: 進化の圧から抜ける。
#:
#: ``user_correction`` は **検証済み**の訂正だけが立つ
#: (``learning.correction_verifier`` が昇格させる) ので字句ブールではない。
#: 一方 ``assistant_self_retraction`` と ``tool_routing_false_positive`` は
#: ``turn_outcome`` 側の理由チャネルと同じ事象を二重計上するため外した。
#: ``rephrased_query`` (bigram cosine の閾値判定) は不変則 #12 / #14 が
#: 「語形では直さない」と決めた層なので外した。
DEFECT_WEIGHTS: dict[str, float] = {
    "user_correction": 1.0,
    "user_negative": 1.0,
    "outcome_contradiction": 1.0,
    "outcome_instruction": 0.8,
    "outcome_broken_output": 0.8,
    "outcome_execution": 0.8,
    OUTCOME_CHANNEL_OTHER: 0.6,
    "long_form_failed": 0.5,
    # ツールを撃つべきだったのに撃たなかった。検証器のどのチャネルとも重なら
    # ないので残す (立つのは明示訂正があったターンの 1 つ前だけ)。
    "tool_routing_false_negative": 0.5,
}

#: 生成パラメータ (temperature / top_p / top_k) 進化の重み。
#:
#: sampling が直接効くのは **出力の崩れ**なので、そこを重くする。逆に
#: ルーティングの失敗や指示違反は sampling をいじっても直らないため軽くする
#: — 責任のない失敗で温度を動かすと、直らないまま端まで振れる。
GENERATION_DEFECT_WEIGHTS: dict[str, float] = {
    **DEFECT_WEIGHTS,
    "outcome_broken_output": 1.0,
    "outcome_execution": 0.3,
    "tool_routing_false_negative": 0.2,
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
