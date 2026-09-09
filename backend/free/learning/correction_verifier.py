"""訂正候補の検証 — 字句で拾った候補を「本物の訂正」へ昇格させる。

``FeedbackCollector`` の訂正検出は正規表現だけで、除外規則は 2026-08-10 以降
**事故ごとに 1 分岐ずつ** 増えてきた。2026-09-08 のライブ監査では 100 ターンで
立った ``user_correction`` が 2 件、そのどちらもが偽陽性で、しかもその 2 件が
Level 1 採用ゲートの唯一の評価ケースになった (15 分のプロンプト進化が無意味な
2 件のために走った)。字句の網目を細かくし続ける限り、次の語形で必ずまた漏れる。

そこで役割を分ける:

- **記録時** = 候補 (``signals.correction_candidate``)。recall 重視で、精度の
  責任を負わない。チャット応答パスの軽い用途 (遡及 false_negative マーク /
  数値の保留 / few-shot 除外) はここを見る。
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

from typing import TYPE_CHECKING

from backend.free.core.correction_verdict import (
    build_correction_verify_prompt,
    check_verdict,
    claims_equivalent,
    response_already_states,
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


def _resolve_previous_context(
    entries: list["ExperienceEntry"], index: int,
) -> tuple[str, str]:
    """``entries[index]`` の訂正候補が指す **直前アシスタント応答とその質問** を返す。

    優先順は ``corrected_entry_id`` (記録時に本文の重なりで確定した宛先) →
    同一セッションの直前ターン。どちらも取れなければ空文字の組。``query`` は
    ``target=self`` (ユーザー自身の過去の発言を訂正) の ``wrong_claim`` 検証に
    使う — アシスタント応答ではなく、その応答を引き出したユーザー発話の側に
    誤りがあったケースなので、span の照合元が異なる。
    """
    entry = entries[index]
    target_id = getattr(entry.signals, "corrected_entry_id", None)
    if target_id:
        for cand in entries:
            if cand.id == target_id:
                return (cand.response_full or cand.response_summary or "", cand.query or "")
    session_id = entry.session_id
    for cand in reversed(entries[:index]):
        if cand.session_id != session_id:
            continue
        if getattr(cand.signals, "correction_candidate", None) is not None:
            continue
        return (cand.response_full or cand.response_summary or "", cand.query or "")
    return ("", "")


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


def recheck_promoted(entries: list["ExperienceEntry"]) -> int:
    """既に昇格済みのエントリへ **コード側の門だけ** を掛け直す (LLM を呼ばない)。

    門は後から増える (2026-09-09 に同値 / 既述を追加)。過去に昇格した
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
        if claims_equivalent(wrong, correct):
            reason = "same_value"
        elif response_already_states(correct, prev_response):
            reason = "already_stated"
        if reason is None:
            continue
        signals.user_correction = None
        signals.correction_verdict = reason
        demoted += 1
        logger.info(
            "Correction verifier: demoted a promoted entry on re-check: %s "
            "(entry=%s, wrong_claim=%r, correct_value=%r)",
            reason, entry.id, wrong[:40], correct[:40],
        )
    return demoted


async def verify_pending_corrections(
    experience_buf: "ExperienceBuffer",
    aux_client,
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    learning_disabled: bool = False,
) -> dict:
    """未検証の訂正候補を検証し、``assistant`` のものだけを昇格させる。

    Args:
        experience_buf: 経験バッファ (``entries`` を持つもの)。
        aux_client: ``AuxClient``。``None`` なら縮退して何もしない。
        max_items: 1 回で問い合わせる上限 (新しい方から)。
        learning_disabled: ``--no-learning`` 中は no-op。

    Returns:
        ``{"checked": int, "promoted": int, "rejected": int, "pending": int,
        "skipped": str | None, "demoted": int}`` — ``demoted`` は
        :func:`recheck_promoted` が候補へ戻した昇格済みエントリ数。
    """
    out: dict = {
        "checked": 0, "promoted": 0, "rejected": 0, "pending": 0, "skipped": None,
    }
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
        if demoted:
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
        if legacy:
            flush = getattr(experience_buf, "flush", None)
            if callable(flush):
                flush()
        return out

    for index in pending[-max_items:]:
        entry = entries[index]
        candidate = entry.signals.correction_candidate or ""
        prev_response, prev_query = _resolve_previous_context(entries, index)
        if not prev_response.strip():
            _apply_verdict(entry, "no_context")
            out["checked"] += 1
            out["rejected"] += 1
            continue
        try:
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
            continue
        out["checked"] += 1
        if not isinstance(parsed, dict) or not parsed:
            _apply_verdict(entry, "no_verdict")
            out["rejected"] += 1
            continue

        check = check_verdict(
            parsed, candidate=candidate, prev_response=prev_response,
            prev_user=prev_query,
        )
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
    "has_pending_candidates",
    "recheck_promoted",
    "verify_pending_corrections",
]
