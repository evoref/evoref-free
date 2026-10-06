"""取得した表の集計のパラメータを文法制約 JSON で取り、コードで集計する (docs/f_03 §4.2.2)。

計算は ``core/table_aggregate`` が全行に対して決定論で行う。ここはモデルから
**パラメータだけ** を取る 1 往復 (purpose ``table_aggregate``、``CHAT_PATH_PURPOSES``)
で、日付演算の ``date_intent`` と同じ立て付け — 分類器スロットで
``generate_constrained`` を直接呼び、共有接頭辞で KV を保つ (不変則 #1)。

表が無い・往復が失敗した・``kind: none``・パラメータが表と合わない、のどれでも
``None`` を返し、呼出側は従来どおりモデルの答えに任せる (集計を諦めるだけで、
推測の値は渡さない)。
"""

from __future__ import annotations

import json
import time
from typing import Any

from backend.exceptions import LLMTimeoutError
from backend.free.core.table_aggregate import (
    AGGREGATIONS,
    DATE_PARTS,
    DERIVES,
    AggregateError,
    AggregateResult,
    AggregateSpec,
    Table,
    describe_spec,
    find_table,
    run_aggregate,
)
from backend.free.core.intent_vocab import has_aggregation_cue
from backend.free.llm.aux_client import PURPOSE_TIMEOUT_DEFAULTS
from backend.free.llm.json_extract import extract_json_object
from backend.free.llm.json_schemas import resolve_response_format_for_purpose
from backend.free.llm.slot_prefix import apply_shared_prefix
from backend.free.llm.tps_calibration import chat_path_timeout, derived_seconds, measured_tps_of
from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("agent.table_aggregate")

PURPOSE = "table_aggregate"
#: 応答の上限トークン (パラメータ 8 つの JSON)。実測 (2026-10-05〜06) は 75〜89 トークン、
#: ``derive`` 付きで 100 トークン超。
_MAX_TOKENS = 200
#: パラメータを取る材料としてモデルへ見せるデータ行の数 (計算には全行を使う)。
_SAMPLE_ROWS = 3

_SYSTEM = (
    "You extract the parameters of ONE aggregation over a table. Do not compute "
    "anything; code computes the result over all rows. Return JSON only.\n"
    "- kind: \"aggregate\" when the request asks for numbers computed over the rows "
    "(total, sum, average, minimum, maximum, count, per group, or a change, "
    "difference or share between such per-group values); otherwise \"none\".\n"
    "- value: an arithmetic expression using only column names exactly as in the "
    "header, numbers, + - * / and parentheses (e.g. \"units * unit_price\"). "
    "Use \"\" only with agg \"count\".\n"
    "- agg: sum | mean | min | max | count.\n"
    "- group_by: header column names to group by ([] for one overall value).\n"
    "- group_date_part: \"month\" or \"year\" to group a date column by month or "
    "year, otherwise \"none\".\n"
    "- where_column / where_equals: one equality filter on a column, \"\" when none.\n"
    "- derive: a value computed from the per-group results when the request asks "
    "for it: \"pct_change\" (percent change from the previous group in key order, "
    "e.g. month-over-month), \"diff\" (difference from the previous group in key "
    "order), \"share_of_total\" (each group's percent of the total); otherwise "
    "\"none\". Code computes it; group_by gives the groups it is computed over.\n"
    "- [previous request], when given, is the user's earlier request in this "
    "conversation. Use it only to fill in what the current request leaves out (the "
    "value to aggregate, a filter); the current request decides the aggregation."
)


def _user_prompt(table: Table, query: str, task: str, previous: str = "") -> str:
    lines = [
        f"[columns] {', '.join(table.header)}",
        "[first rows]",
        *(", ".join(row) for row in table.rows[:_SAMPLE_ROWS]),
    ]
    if previous.strip() and previous.strip() != query.strip():
        lines.append(f"[previous request] {previous.strip()}")
    lines.append(f"[request] {query.strip()}")
    if task.strip() and task.strip() != query.strip():
        lines.append(f"[current step] {task.strip()}")
    return "\n".join(lines)


def parse_aggregate_spec(payload: Any) -> AggregateSpec | None:
    """往復の応答を集計のパラメータにする (純粋関数)。集計でない・形が違えば ``None``。"""
    if not isinstance(payload, dict) or payload.get("kind") != "aggregate":
        return None
    agg = payload.get("agg")
    date_part = payload.get("group_date_part", "none")
    derive = payload.get("derive", "none")
    group_by = payload.get("group_by", [])
    if agg not in AGGREGATIONS or date_part not in DATE_PARTS or derive not in DERIVES:
        return None
    if not isinstance(group_by, list) or not all(isinstance(g, str) for g in group_by):
        return None
    fields = {k: payload.get(k, "") for k in ("value", "where_column", "where_equals")}
    if not all(isinstance(v, str) for v in fields.values()):
        return None
    return AggregateSpec(
        value=fields["value"],
        agg=agg,
        group_by=tuple(g for g in group_by if g.strip()),
        group_date_part=date_part,
        where_column=fields["where_column"],
        where_equals=fields["where_equals"],
        derive=derive,
    )


def round_trip_timeout(llm_client: Any, messages: list[dict], *, carried: bool = False) -> float:
    """往復 1 回の上限秒 (docs/f_03 §4.2.2 の 2 / 6)。

    読んだターンは purpose の既定 (40 秒) を、生成速度の実測 (c_16 §7.2.3) が遅いマシンで
    だけ「推定プロンプト ÷ prefill + ``max_tokens`` ÷ decode」(余裕込み) まで延ばす (天井は
    1.5 倍)。``AuxClient`` がチャット応答パスの purpose に掛ける規則と同じ (この往復は
    ``AuxClient`` を通らない)。

    ``carried`` (持ち越した表への続きのターン) は同じ見積り — ``max_tokens`` 全部を実測の
    速度で出し切る秒に余裕と下駄 (``derived_seconds``) — を上限にする (読んだターンの上限は
    超えない)。完走する往復を途中で切らず、見積りを超えた遅れだけを打ち切る。2026-10-06:
    固定 10 秒は実測の中央値で、「前月比は？」の往復を切り増減率を暗算で誤らせた。
    未測定なら読んだターンと同じ。
    """
    base = PURPOSE_TIMEOUT_DEFAULTS.get(PURPOSE, 40.0)
    tps = measured_tps_of(llm_client)
    prompt_tokens = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
    budget = chat_path_timeout(base, tps, prompt_tokens=prompt_tokens, max_tokens=_MAX_TOKENS)
    if not carried:
        return budget
    need = derived_seconds(tps, prompt_tokens=prompt_tokens, max_tokens=_MAX_TOKENS)
    return budget if need is None else min(budget, need)


def _log_round_trip(
    llm_client: Any, messages: list[dict], content: str, elapsed: float, *,
    timeout: float, slot: int, outcome: str,
) -> None:
    """往復を requests JSONL の aux 行に残す (``AuxClient`` 経由の補助タスクと同じ形)。"""
    debug_logger = getattr(llm_client, "debug_logger", None)
    if debug_logger is None:
        return
    try:
        debug_logger.log_aux_request(
            messages_count=len(messages), response_preview=content, elapsed_sec=elapsed,
            purpose=PURPOSE, resolved_timeout=timeout, response_format_used=True,
            finish_reason=outcome, response_length=len(content), slot=slot,
        )
    except Exception:  # noqa: BLE001 - 計測の失敗で集計を失わない
        logger.debug("Table aggregate round trip was not logged", exc_info=True)


async def aggregate_retrieved_table(
    llm_client: Any, query: str, outputs: list[str], *, task: str = "",
    require_cue: bool = True, previous: str = "", carried: bool = False,
) -> AggregateResult | None:
    """取得結果の表に対し、依頼が求める集計をコードで行う (行えなければ ``None``)。

    ``outputs`` は ``read_file`` の結果の並び (別のファイルの表が 2 つ以上なら集計しない)。
    表が無いか、依頼とステップのどちらにも数値の集計・計算の手掛かり
    (``intent_vocab.has_aggregation_cue``、既存の語彙の和) が無ければ往復を撃たない
    (表を読む要約・説明のターンにレイテンシを足さない)。``require_cue=False`` は
    持ち越した資料への続きの問い用で、手掛かりの語を見ずに撃つ (docs/f_03 §4.2.2 の 6)。
    ``previous`` は直前のユーザー発話 (続きの問いが省いた量・絞り込みを補う文脈)、
    ``carried`` は持ち越した表への続きのターン (上限は :func:`round_trip_timeout`)。
    """
    if require_cue and not has_aggregation_cue(f"{query}\n{task}"):
        return None
    table = find_table(outputs)
    if table is None:
        return None
    if llm_client is None or not hasattr(llm_client, "generate_constrained"):
        return None
    response_format = resolve_response_format_for_purpose(PURPOSE)
    if response_format is None:
        return None
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_prompt(table, query, task, previous)},
    ]
    prefix = getattr(llm_client, "classifier_slot_prefix", None)
    if isinstance(prefix, str) and prefix:
        # 分類器スロットの接頭辞キャッシュを追い出さない (aux_client._with_slot_prefix)
        messages = apply_shared_prefix(prefix, messages)
    slot = getattr(llm_client, "classifier_slot", None)
    if not isinstance(slot, int) or isinstance(slot, bool):
        slot = getattr(llm_client, "background_slot", -1)
    timeout = round_trip_timeout(llm_client, messages, carried=carried)
    started = time.monotonic()
    try:
        # usage_purpose は渡さない — aux 行 (_log_round_trip) が同じターンの集計へ足す
        content = await llm_client.generate_constrained(
            messages,
            response_format=response_format,
            max_tokens=_MAX_TOKENS,
            id_slot=slot,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - 集計を諦めて従来の答えへ倒す
        elapsed = time.monotonic() - started
        outcome = "timeout" if isinstance(exc, LLMTimeoutError | TimeoutError) else "error"
        _log_round_trip(
            llm_client, messages, "", elapsed, timeout=timeout, slot=slot, outcome=outcome,
        )
        logger.info(
            "Table aggregate extraction failed after %.1fs (limit %.1fs, carried=%s): %s",
            elapsed, timeout, carried, exc,
        )
        return None
    elapsed = time.monotonic() - started
    text = content if isinstance(content, str) else ""
    _log_round_trip(llm_client, messages, text, elapsed, timeout=timeout, slot=slot, outcome="")
    logger.info(
        "Table aggregate round trip: %.1fs (limit %.1fs, carried=%s, response %d chars)",
        elapsed, timeout, carried, len(text),
    )
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        # スキーマを強制しない build では非 JSON が返り得る (分類器と同じ救済)
        payload = extract_json_object(content)
    spec = parse_aggregate_spec(payload)
    if spec is None:
        logger.info("Table aggregate: the request is not an aggregation over the table")
        return None
    try:
        result = run_aggregate(table, spec)
    except AggregateError as exc:
        logger.info(
            "Table aggregate not computed (%s): %s", exc, describe_spec(spec),
        )
        return None
    logger.info(
        "Table aggregate computed by code: %s over %d/%d rows -> %d groups",
        describe_spec(spec), result.rows_used, result.rows_total, len(result.groups),
    )
    return result
