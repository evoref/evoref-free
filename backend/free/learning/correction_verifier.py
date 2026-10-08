"""訂正候補の検証 — 字句で拾った候補を「本物の訂正」へ昇格させる。

``FeedbackCollector`` の訂正検出は正規表現だけで、除外規則は 2026-08-10 以降
**事故ごとに 1 分岐ずつ** 増えてきた。2026-09-08 のライブ監査では 100 ターンで
立った ``user_correction`` が 2 件、そのどちらもが偽陽性で、しかもその 2 件が
Level 1 採用ゲートの唯一の評価ケースになった (15 分のプロンプト進化が無意味な
2 件のために走った)。字句の網目を細かくし続ける限り、次の語形で必ずまた漏れる。

そこで役割を分ける:

- **記録時** = 候補 (``signals.correction_candidate``)。recall 重視で、精度の
  責任を負わない。チャット応答パスの軽い用途 (数値の保留 / few-shot 除外) は
  ここを見る。前ターンへの遡及 false_negative は検証の後
  (:func:`reconcile_false_negatives`) で付け外しする。
- **消費前** = 検証 (本モジュール)。Level 1 / Level 2 が読み始める前に、
  (直前アシスタント応答, 訂正発話) の組を補助タスク (``correction_verify``、
  ``background_slot``) へ渡して帰属を判定し、``assistant`` のものだけを
  ``signals.user_correction`` へ昇格させる。

LLM の判定は **単独では信用しない**。プロンプトの組み立てと、判定結果への
コード側の門 (逐語 span / 同値 / 既述) は ``backend.free.core.correction_verdict``
(:func:`~backend.free.core.correction_verdict.build_correction_verify_prompt` /
:func:`~backend.free.core.correction_verdict.check_verdict`) に集約している。
EvorefMem 側の ``memory.sleep.correction_curator`` も同じ門を使うので、片方だけ
偽陽性を弾いて反対側は書いてしまうという食い違いが起きない — 2026-09-08 夜の
監査では、アシスタントが「100 m」と正しく答えたのに、ユーザーの「訂正」も
同じ「100 m」を主張する発話 (``wrong_claim="100"`` / ``correct_value="100 m"``)
が ``assistant`` 判定で昇格していた。``wrong_claim`` と ``correct_value`` が
指す値が同じなら、それは訂正ではなく確認・言い直しであり、
:func:`~backend.free.core.correction_verdict.claims_equivalent` の門で弾く。

副産物の ``correct_value`` は eval_core / 訂正ペアの期待語に使う — 字句の
否定境界 (``ではなく`` / ``じゃなく``) 2 語に頼っていた頃は、誤り側が期待語
として eval_core に載っていた (2026-09-07 監査 F-01)。

冪等。``correction_verified_at`` が入ったエントリは二度と問い合わせない。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.free.core.correction_verdict import (
    answer_disputes_value,
    build_correction_verify_prompt,
    check_verdict,
    claims_equivalent,
    get_shared_verdict,
    response_already_states,
    reversed_restatement,
)
from backend.free.llm.json_schemas import CorrectionVerdict
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.free.learning.level0_instant import ExperienceBuffer, ExperienceEntry

logger = get_logger("learning.correction_verifier")

#: 1 回の呼出で問い合わせる候補の上限。1 件 30〜40 秒なので、溜まっていても
#: 1 サイクルを占有しないよう新しい方から打ち切る。
DEFAULT_MAX_ITEMS = 12

#: 答えが取れなかった回 (出力の切断・パース不能で空の結果) を一過性として
#: 再試行する回数。これに達したら ``no_verdict`` で閉じる — 崩れた出力しか
#: 返らない候補に毎サイクル補助タスクを払い続けない。
MAX_UNANSWERED_ATTEMPTS = 3

#: 昇格させる帰属。
_PROMOTED_TARGET = "assistant"


def _is_pending(signals) -> bool:
    """未検証の候補か。

    2026-09-08 以前に記録されたバッファは字句検出の結果を ``user_correction``
    に直接持つ (候補フィールドが無い)。検証済み印が無いそれらは **候補へ
    格下げしてから** 検証する — 移行器を別に置かず、ここで吸収する。
    """
    if getattr(signals, "correction_verified_at", None):
        return False
    if getattr(signals, "correction_candidate", None) is not None:
        return True
    return getattr(signals, "user_correction", None) is not None


def _demote_legacy(signals) -> None:
    """旧形式 (``user_correction`` 直書き) を候補へ格下げする。"""
    if (
        getattr(signals, "correction_candidate", None) is None
        and getattr(signals, "user_correction", None) is not None
    ):
        signals.correction_candidate = signals.user_correction
        signals.user_correction = None


def has_pending_candidates(experience_buf) -> bool:
    """未検証の訂正候補が 1 件でもあるか。

    呼出側が「候補が無いのに補助クライアントを組み立てる」のを避けるための
    安価な事前チェック。
    """
    return any(
        _is_pending(e.signals)
        for e in (getattr(experience_buf, "entries", None) or [])
    )


def _resolve_previous_entry(
    entries: list["ExperienceEntry"], index: int,
) -> "ExperienceEntry | None":
    """``entries[index]`` の訂正候補が指す **宛先のエントリ** を返す。

    優先順は ``corrected_entry_id`` (記録時に本文の重なりで確定した宛先) →
    同一セッションの直前ターン (訂正候補のターンは飛ばす)。取れなければ ``None``。
    """
    entry = entries[index]
    target_id = getattr(entry.signals, "corrected_entry_id", None)
    if target_id:
        for cand in entries:
            if cand.id == target_id:
                return cand
    session_id = entry.session_id
    for cand in reversed(entries[:index]):
        if cand.session_id != session_id:
            continue
        if getattr(cand.signals, "correction_candidate", None) is not None:
            continue
        return cand
    return None


def _resolve_previous_context(
    entries: list["ExperienceEntry"], index: int,
) -> tuple[str, str]:
    """``entries[index]`` の訂正候補が指す **直前アシスタント応答とその質問** を返す。

    宛先は :func:`_resolve_previous_entry`。取れなければ空文字の組。``query`` は
    ``target=self`` (ユーザー自身の過去の発言を訂正) の ``wrong_claim`` 検証に
    使う — アシスタント応答ではなく、その応答を引き出したユーザー発話の側に
    誤りがあったケースなので、span の照合元が異なる。
    """
    target = _resolve_previous_entry(entries, index)
    if target is None:
        return ("", "")
    return (target.response_full or target.response_summary or "", target.query or "")


def _routed_tool(signals) -> bool:
    return bool(
        getattr(signals, "tool_routing_success", False)
        or getattr(signals, "tool_routing_false_positive", False)
    )


#: (宛先の見逃しフラグ, 学習カテゴリ, 訂正ターンが capability を使ったか,
#: 宛先が capability を使ったか)。
_FALSE_NEGATIVE_KINDS: tuple[tuple[str, str, Callable, Callable], ...] = (
    (
        "tool_routing_false_negative", "tool_routing",
        _routed_tool, _routed_tool,
    ),
    (
        "long_form_false_negative", "long_form",
        lambda s: bool(getattr(s, "long_form_used", False)),
        lambda s: bool(getattr(s, "long_form_used", False)),
    ),
)


def reconcile_false_negatives(
    entries: list["ExperienceEntry"],
) -> list[tuple[str, str, int]]:
    """検証済みの訂正から、宛先ターンの見逃し (false_negative) の印を付け外しする。

    「宛先ターンは capability (ツール / 長文) を使わず、訂正ターンが使った」は
    宛先ターンが capability を要した証拠だが、**訂正が本物のときだけ** 成り立つ
    (不変則 #12)。そこで記録時ではなく検証の後にここで印を決める:

    - 訂正の判定が ``assistant`` (昇格中) で宛先のフラグが ``False`` → ``True`` にして +1。
    - 訂正が検証済みで判定が ``assistant`` 以外 (却下・再チェックでの格下げ) で、
      宛先のフラグが ``True`` → ``False`` にして -1。ただし同じ宛先・同じ種別を
      昇格中の別の訂正が支えているなら外さない。
    - capability の遷移が無い訂正 (= 証拠にならない) のフラグには触らない。

    冪等: 付け終えた印は再度数えない。

    Returns:
        ``[(宛先の query, "tool_routing" | "long_form", +1 | -1), ...]``。
        +1 は学習語の追加、-1 は取り消し (減衰) を呼出側に求める。
    """
    supported: set[tuple[str, str]] = set()
    revoke: list[tuple["ExperienceEntry", str, str]] = []
    out: list[tuple[str, str, int]] = []
    for index, entry in enumerate(entries):
        signals = entry.signals
        if not getattr(signals, "correction_verified_at", None):
            continue
        target = _resolve_previous_entry(entries, index)
        if target is None:
            continue
        promoted = (
            getattr(signals, "correction_verdict", None) == _PROMOTED_TARGET
            and getattr(signals, "user_correction", None) is not None
        )
        for flag, category, used_now, used_before in _FALSE_NEGATIVE_KINDS:
            if not used_now(signals) or used_before(target.signals):
                continue
            if promoted:
                supported.add((target.id, flag))
                if not getattr(target.signals, flag, False):
                    setattr(target.signals, flag, True)
                    out.append((target.query or "", category, +1))
            else:
                revoke.append((target, flag, category))
    for target, flag, category in revoke:
        if (target.id, flag) in supported:
            continue
        if getattr(target.signals, flag, False):
            setattr(target.signals, flag, False)
            out.append((target.query or "", category, -1))
    for query, category, delta in out:
        logger.info(
            "Correction verifier: %s false_negative on the corrected turn "
            "(category=%s, query=%r)",
            "marked" if delta > 0 else "revoked", category, query[:40],
        )
    return out


def _apply_verdict(
    entry: "ExperienceEntry",
    verdict: str,
    *,
    wrong_claim: str = "",
    correct_value: str = "",
) -> None:
    """検証結果をエントリへ刻む (昇格判定込み)。"""
    signals = entry.signals
    signals.correction_verified_at = utc_now()
    signals.correction_verdict = verdict
    signals.correction_wrong_claim = wrong_claim or None
    signals.correction_correct_value = correct_value or None
    if verdict == _PROMOTED_TARGET:
        signals.user_correction = signals.correction_candidate


def _log_split(entry_id: str, memory_label: str, promoted: bool) -> None:
    """記憶側と同じ生の出力から、帰属 (assistant か否か) が割れたら記録する。

    門は各側の入力 (記憶側はノートの複数ターン、学習側は経験の直前応答) で
    当てるので、同じ出力でも結論が割れうる (不変則 #14(c) の記録)。
    """
    if (memory_label == _PROMOTED_TARGET) == promoted:
        return
    logger.info(
        "Correction verdict split between memory and learning sides "
        "(entry=%s, memory=%s, learning_promoted=%s)",
        entry_id, memory_label, promoted,
    )


def recheck_promoted(entries: list["ExperienceEntry"]) -> int:
    """既に昇格済みのエントリへ **コード側の門だけ** を掛け直す (LLM を呼ばない)。

    門は後から増える (2026-09-09 に同値 / 既述、2026-09-28 に向きを追加)。過去に昇格した
    エントリは冪等マーカーで二度と問い合わせないため、門を足しても既存の
    偽陽性 (100 → 100 m) は few-shot / 採用ゲート / eval_core に残り続ける。
    記録済みの span と直前応答だけで決定論に判定できる門なので、毎回掛け直して
    落ちたものを候補へ戻し、理由を ``correction_verdict`` に記録する。

    Returns:
        格下げしたエントリ数。
    """
    demoted = 0
    for index, entry in enumerate(entries):
        signals = entry.signals
        if getattr(signals, "user_correction", None) is None:
            continue
        if getattr(signals, "correction_verdict", None) != _PROMOTED_TARGET:
            continue
        wrong = getattr(signals, "correction_wrong_claim", None) or ""
        correct = getattr(signals, "correction_correct_value", None) or ""
        prev_response, _prev_query = _resolve_previous_context(entries, index)
        reason = None
        detail = ""
        candidate = getattr(signals, "correction_candidate", None) or ""
        if reversed_restatement(wrong, correct, candidate):
            # 永続の語彙は増やさない (check_verdict と同じく invalid_span)。
            reason, detail = "invalid_span", " (reversed)"
        elif claims_equivalent(wrong, correct):
            reason = "same_value"
        elif response_already_states(correct, prev_response):
            reason = "already_stated"
        if reason is None:
            continue
        signals.user_correction = None
        signals.correction_verdict = reason
        demoted += 1
        logger.info(
            "Correction verifier: demoted a promoted entry on re-check: %s%s "
            "(entry=%s, wrong_claim=%r, correct_value=%r)",
            reason, detail, entry.id, wrong[:40], correct[:40],
        )
    return demoted


async def verify_pending_corrections(
    experience_buf: "ExperienceBuffer",
    aux_client,
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    learning_disabled: bool = False,
    unanswered: dict[str, int] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> dict:
    """未検証の訂正候補を検証し、``assistant`` のものだけを昇格させる。

    Args:
        experience_buf: 経験バッファ (``entries`` を持つもの)。
        aux_client: ``AuxClient``。``None`` なら縮退して何もしない。
        max_items: 1 回で問い合わせる上限 (新しい方から)。
        learning_disabled: ``--no-learning`` 中は no-op。
        unanswered: エントリ id → 答えが取れなかった回数。呼出を跨いで持つ
            (``MAX_UNANSWERED_ATTEMPTS`` に達したら ``no_verdict`` で閉じる)。
        should_pause: 真ならユーザーが活動中。検証を出さずに打ち切る。

    Returns:
        ``{"checked": int, "promoted": int, "rejected": int, "pending": int,
        "skipped": str | None, "demoted": int, "unanswered": int,
        "false_negatives": list[tuple[str, str, int]]}`` —
        ``demoted`` は :func:`recheck_promoted` が候補へ戻した昇格済みエントリ数、
        ``unanswered`` は答えが取れず次回へ回した数、``false_negatives`` は
        :func:`reconcile_false_negatives` の結果 (呼出側が学習語へ適用する)。
    """
    out: dict = {
        "checked": 0, "promoted": 0, "rejected": 0, "pending": 0, "skipped": None,
        "unanswered": 0, "false_negatives": [],
    }
    if unanswered is None:
        unanswered = {}
    if learning_disabled:
        out["skipped"] = "learning_disabled"
        return out

    entries: list["ExperienceEntry"] = list(
        getattr(experience_buf, "entries", None) or []
    )
    demoted = recheck_promoted(entries)
    out["demoted"] = demoted
    pending = [i for i, e in enumerate(entries) if _is_pending(e.signals)]
    out["pending"] = len(pending)
    if not pending:
        out["false_negatives"] = reconcile_false_negatives(entries)
        if demoted or out["false_negatives"]:
            flush = getattr(experience_buf, "flush", None)
            if callable(flush):
                flush()
        return out
    # 旧形式は先に候補へ格下げする。aux 未接続で戻る場合も、消費側が
    # 未検証の ``user_correction`` を読まない状態にしてから戻す。
    legacy = 0
    for i in pending:
        if getattr(entries[i].signals, "correction_candidate", None) is None:
            _demote_legacy(entries[i].signals)
            legacy += 1
    if legacy:
        logger.info(
            "Correction verifier: demoted %d legacy user_correction entries to "
            "candidates", legacy,
        )

    if aux_client is None:
        # 起動失敗 / 文法制約非対応。候補は候補のまま残し、次サイクルで再試行する
        # (黙って昇格させない = 学習側は訂正 0 件で回る)。
        out["skipped"] = "no_aux_client"
        logger.warning(
            "Correction verifier degraded: no aux client; %d candidates left "
            "unverified", len(pending),
        )
        out["false_negatives"] = reconcile_false_negatives(entries)
        if legacy or demoted or out["false_negatives"]:
            flush = getattr(experience_buf, "flush", None)
            if callable(flush):
                flush()
        return out

    for index in pending[-max_items:]:
        if should_pause is not None and should_pause():
            out["skipped"] = "chat_active"
            logger.info(
                "Correction verifier: user is active; remaining candidates wait "
                "for a quiet window",
            )
            break
        entry = entries[index]
        candidate = entry.signals.correction_candidate or ""
        prev_response, prev_query = _resolve_previous_context(entries, index)
        if not prev_response.strip():
            _apply_verdict(entry, "no_context")
            out["checked"] += 1
            out["rejected"] += 1
            continue
        # 記憶側 (Step 8.0) が同じ候補を既に問うていれば、その生の出力を使う。
        # 門は下でこちらの入力に当て直す (決定論の SSOT は correction_verdict)。
        shared = get_shared_verdict(entry.session_id, candidate)
        shared_label = shared[1] if shared is not None else None
        try:
            if shared is not None:
                parsed = shared[0]
            else:
                parsed = await aux_client.generate_json(
                    build_correction_verify_prompt(
                        prev_response, candidate, prev_user=prev_query,
                    ),
                    purpose="correction_verify",
                    max_tokens=384,
                    temperature=0.1,
                    response_schema=CorrectionVerdict,
                )
        except Exception as exc:  # noqa: BLE001 - 検証失敗で学習を止めない
            logger.warning(
                "Correction verification failed (entry=%s): %r", entry.id, exc,
            )
            if getattr(exc, "contended", False):
                # 横取りされたなら次の候補もまた横取りされる。残りは次回へ回す
                # (記憶側 Step 8.0 と同じ、2026-09-26 監査 #12)。
                break
            continue
        if not isinstance(parsed, dict) or not parsed:
            # 出力の切断・パース不能で答えが取れなかった。判定ではないので
            # 上限までは刻まずに次回へ回す (2026-09-26 監査: create の生成と
            # 併走した崩れた出力が no_verdict として刻まれ、二度と検証されなかった)。
            attempts = unanswered.get(entry.id, 0) + 1
            if attempts < MAX_UNANSWERED_ATTEMPTS:
                unanswered[entry.id] = attempts
                out["unanswered"] += 1
                logger.info(
                    "Correction verifier: no usable verdict (entry=%s, attempt %d/%d); "
                    "will retry", entry.id, attempts, MAX_UNANSWERED_ATTEMPTS,
                )
                continue
            unanswered.pop(entry.id, None)
            out["checked"] += 1
            _apply_verdict(entry, "no_verdict")
            out["rejected"] += 1
            continue
        unanswered.pop(entry.id, None)
        out["checked"] += 1

        check = check_verdict(
            parsed, candidate=candidate, prev_response=prev_response,
            prev_user=prev_query,
        )
        # 訂正への回答 (このエントリ自身の応答) が値を退けていれば昇格させない
        # (:func:`answer_disputes_value`、2026-09-11 (j) J-04: 「302 は恒久的な
        # 移転」という誤った訂正がアシスタントに退けられたのに assistant と
        # 判定され、訂正ペアの素材になりかけた)。
        disputed = check.ok and answer_disputes_value(
            entry.response_full or entry.response_summary or "", check.correct_value,
        )
        if shared_label is not None:
            _log_split(
                entry.id, shared_label,
                check.ok and not disputed and check.target == _PROMOTED_TARGET,
            )
        if disputed:
            logger.info(
                "Correction verdict disputed by the reply (entry=%s, value=%r)",
                entry.id, check.correct_value[:40],
            )
            _apply_verdict(
                entry, "disputed",
                wrong_claim=check.wrong_claim, correct_value=check.correct_value,
            )
            out["rejected"] += 1
            continue
        if check.ok and check.target == _PROMOTED_TARGET:
            _apply_verdict(
                entry, _PROMOTED_TARGET,
                wrong_claim=check.wrong_claim, correct_value=check.correct_value,
            )
            out["promoted"] += 1
            continue

        if check.reason is not None:
            logger.info(
                "Correction verdict rejected: %s (entry=%s, wrong_claim=%r, "
                "correct_value=%r)",
                check.reason, entry.id, check.wrong_claim[:40],
                check.correct_value[:40],
            )
        verdict = check.reason if check.reason is not None else check.target
        _apply_verdict(
            entry, verdict,
            wrong_claim=check.wrong_claim, correct_value=check.correct_value,
        )
        out["rejected"] += 1

    out["false_negatives"] = reconcile_false_negatives(entries)
    flush = getattr(experience_buf, "flush", None)
    if callable(flush):
        flush()
    logger.info(
        "Correction verification: checked=%d promoted=%d rejected=%d "
        "pending=%d", out["checked"], out["promoted"], out["rejected"],
        out["pending"],
    )
    return out


__all__ = [
    "DEFAULT_MAX_ITEMS",
    "MAX_UNANSWERED_ATTEMPTS",
    "has_pending_candidates",
    "recheck_promoted",
    "reconcile_false_negatives",
    "verify_pending_corrections",
]
