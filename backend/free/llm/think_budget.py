"""purpose 別の思考予算 (要求本文の ``reasoning_budget_tokens``) の表 (docs/c_15 §2.9.3 / c_14 §3.3)。

常に思考するモデル (``reasoning.mode: always``、OFF スイッチ無し。例 LFM2.5-24B-A2B) では、
max_tokens の小さい補助呼び出しが思考で上限を使い切り本文が空になる (c_15 §1.2.1)。
呼出側は purpose からここで予算 B を引き、``think_budget`` 引数で ``LocalClient`` /
facade (``LLMClient.generate_constrained``) へ渡す。予算を実際に送るか (条件 S) は
``LocalClient`` が 1 か所で決める — toggle / none のモデルには送らない。

表に無い purpose と値が ``None`` の purpose は要求では送らず、起動フラグ
``--reasoning-budget`` (プロファイルの ``budget_default``) の既定だけが掛かる。
値はすべて仮 (実測で確定する)。
"""

from __future__ import annotations

from typing import Any

from backend.utils import estimate_tokens

#: 思考の観測値として ``result_meta`` / aux 行に載せる欄 (docs/c_15 §2.9.8)。
THINK_META_KEYS: tuple[str, ...] = (
    "completion_tokens", "reasoning_chars", "reasoning_tokens_est", "think_budget", "max_tokens_sent",
)

#: AuxClient の purpose ではない直呼びの鍵 (``PURPOSE_TIMEOUT_DEFAULTS`` に無い)。
#: ``tool_classifier`` / ``expression_synthesis`` は tool_call_judge の ``usage_purpose``、
#: ``prompt_edit`` は Level 1 の差分編集変異 (``PromptEvolver._mutate_by_edit``、facade 経由)。
DIRECT_CALL_KEYS: frozenset[str] = frozenset({
    "tool_classifier",
    "expression_synthesis",
    "prompt_edit",
})

PURPOSE_THINK_BUDGET: dict[str, int | None] = {
    # ── チャット応答パスの判定 (判定 JSON の前の思考は待ち時間と空応答の原因) ──
    "tool_classifier": 0,
    "date_intent": 0,
    "table_aggregate": 0,
    "meta_cognitive_plan": 0,
    "retrieval_chunk_gate": 0,
    "answer_conditions": 0,
    "template_fill": 0,
    # ── 式合成 (0 だと誤式が出て正答 75%。24 問の掃引で 128 / 256 / 512 は 96〜98% で同等、
    #    256 は余裕を見た値。c_15 §2.9.3) ──
    "expression_synthesis": 256,
    # ── 背景の抽出・記憶への書き込み。personal_fact_split / correction_verify は
    #    ラベル付き評価 (scripts/bench/bg_purpose_eval) で 1024 に確定。残りは測ってから ──
    "personal_fact_split": 1024,
    "slot_edit": None,
    "assertion_naming": 256,
    "url_relevance_score": None,
    "fewshot_quality_score": None,
    "pseudo_query": None,
    "note_evolution": None,
    "summarize": None,
    "conflict_resolution": None,
    "correction_verify": 1024,
    "prompt_candidate_judge": None,
    # ── Level 1 の変異 (測定後) ──
    "prompt_edit": None,
}


def resolve_think_budget(purpose: str | None) -> int | None:
    """purpose の思考予算 B (トークン数)。表に無い / 値が ``None`` なら ``None`` (送らない)。"""
    if not purpose:
        return None
    return PURPOSE_THINK_BUDGET.get(purpose)


def sent_max_tokens(client: Any, answer: int | None, budget: int | None) -> int | None:
    """``client`` へ ``answer`` と予算 ``budget`` で頼んだときに実際に送る max_tokens。

    締切・コンテキスト予算を送る値で計算するための共通入口 (c_15 §2.9.4)。実体は
    ``LocalClient.effective_max_tokens`` (facade は委譲)。メソッドはクラスから引く —
    差し替え実装 (テストの ``AsyncMock`` 等) のインスタンス属性を呼ぶと、戻り値が
    数でない / 待たれないコルーチンになるため、持たない実装は答えの上限のままにする。
    """
    if budget is None:
        return answer
    fn = getattr(type(client), "effective_max_tokens", None)
    if not callable(fn):
        return answer
    sent = fn(client, answer, budget)
    if isinstance(sent, int) and not isinstance(sent, bool):
        return sent
    return answer


def think_usage_of(data: Any, content: str | None) -> dict:
    """応答 (OAI 互換 dict) から思考の量の観測値を取り出す (docs/c_15 §2.9.8)。

    ``usage.completion_tokens`` (無ければ ``timings.predicted_n``) は思考の分を含み、
    思考だけの数は返らない (c_15 §1.2.1 の 5) ので、本文のトークン見積りを引いて推定する。
    取れない値は ``None`` (``reasoning_chars`` は思考が無ければ 0)。
    """
    if not isinstance(data, dict):
        data = {}
    usage = data.get("usage")
    timings = data.get("timings")
    completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not _is_count(completion):
        predicted = timings.get("predicted_n") if isinstance(timings, dict) else None
        completion = predicted if _is_count(predicted) else None
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message")
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    reasoning_est = None
    if completion is not None:
        reasoning_est = max(0, completion - estimate_tokens(content if isinstance(content, str) else ""))
    return {
        "completion_tokens": completion,
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "reasoning_tokens_est": reasoning_est,
    }


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "DIRECT_CALL_KEYS",
    "PURPOSE_THINK_BUDGET",
    "THINK_META_KEYS",
    "resolve_think_budget",
    "sent_max_tokens",
    "think_usage_of",
]
