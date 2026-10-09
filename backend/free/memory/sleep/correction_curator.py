"""Step 8.0: 訂正候補の **検証** キュレーター

``MemoryNote.is_correction`` は字句 (「違います」「ではなく」「正しくは」…)
で立てた **候補** に過ぎない。ところが記憶側はこの 1 ビットに、検証なしで
強い力を与えていた:

- ``SemanticFact.from_correction`` — 競合解決の即時解決、単値でないスロットの
  supersede、注入の「訂正後の記録」ラベル
- ``resolve_value_anchored_attributes`` — 既存スロットの現在値でスロットを決める
- ``resolve_inherited_attributes`` — 直前の言明のスロットを丸ごと継ぐ

「訂正の形をしている」と「実際に過去の発言の誤りを指している」は別のことで、
両者を同一視した結果 2026-09-08 夜のライブ監査で 2 件の実害が出た:

- **G-01**: 物理の計算に対する訂正「違います。最初の答えを計算し直して
  ください。時速 90 km は秒速 25 m なので、4 秒では 100 m のはずです。」が
  ``mem.personal.family`` として書かれ、本物の家族ファクト 4 件を supersede した。
- **G-04**: 9 時間前のノート「…営業から「モニタと印刷で色が違う」という
  クレームが…」が引用中の語で訂正候補になり、より新しい occupation を
  supersede した。

そこで **消費する前に検証する**。学習側 (``learning.correction_verifier``) が
2026-09-08 に採った形と同じで、判定の SSOT は
:mod:`backend.free.core.correction_verdict` (プロンプト + コード側の門) を
共有する — 片方だけ厳しくすると「学習側は偽陽性を弾いたのに記憶側は書いた」
という食い違いが残る。

規約:

- 走るのは sleep-time だけ (CLAUDE.md §6 不変則 #2 / c_16 §2.1)。Step 8
  (抽出) の **直前** に置き、同じサイクルで判定と消費が揃うようにする。
- 冪等マーカーは ``correction_verified_at``。LLM が答えたときだけ立てる —
  例外 (aux の timeout 等) では立てず、次サイクルで再試行する。
- ``aux_client`` が無い (degraded) 場合は no-op。ノートは未検証のまま残り、
  消費側は「訂正でない」ではなく **通常の再言明** として扱う。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.free.core.correction_verdict import (
    answer_disputes_value,
    build_correction_verify_prompt,
    check_verdict,
    put_shared_verdict,
)
from backend.free.llm.json_schemas import CorrectionVerdict
# 候補の判定は注入の注記と共有する 1 実装 (不変則 #14(a))。本モジュールからも
# 引けるよう名前を残す。
from backend.free.memory.corrections import is_correction_candidate
from backend.free.memory.sleep._curator_common import public_notes
from backend.free.memory.sleep.curation_backoff import (
    clear_failure,
    in_cooldown,
    record_transient_failure,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.episodic.note import MemoryNote

logger = get_logger("memory.sleep.correction_curator")

#: 1 サイクルで検証に出す上限。補助タスク 1 回 60 秒
#: (``PURPOSE_TIMEOUT_DEFAULTS["correction_verify"]``) なのでアイドル窓を
#: 食い潰さないよう明示的に絞る。超過分は次サイクルへ回る
#: (``correction_verified_at`` を立てないため)。
_MAX_PER_CYCLE = 6

#: ``correction_verify`` の出力トークン上限と生成温度。
_VERIFY_MAX_TOKENS = 384
_VERIFY_TEMPERATURE = 0.1

#: ``target=self`` の照合元として遡るユーザー発話の件数。
_SELF_CONTEXT_TURNS = 3

#: 直前のアシスタント応答が同一セッションに無かったときの verdict。
#: 「検証した結果、訂正の相手が居なかった」= 訂正として消費しない。
NO_CONTEXT = "no_context"


#: ``MemoryNote.curation_failures`` のキー (curation_backoff)。
VERIFY_FAILURE_KEY = "correction_verify"


def verification_pending(note: object, now: float) -> bool:
    """訂正候補で、まだ検証されておらず **次のサイクルで検証が見込める** か。

    Step 8 (ChatExtractor) はこれが真のノートを据え置く。据え置かないと、
    Step 8.0 の aux がチャットに横取りされたサイクルで訂正候補を通常の
    再言明として消費し ``extracted_fact_ids`` を立ててしまい、次サイクルで
    検証が通っても **二度と訂正の力で再抽出されない** (2026-09-10 (h) H-12)。
    cooldown 中 (実失敗が閾値に達した) は見込めないので偽 — 永久に据え置かない。
    """
    if getattr(note, "correction_verified_at", None) is not None:
        return False
    if not is_correction_candidate(note):
        return False
    return not in_cooldown(note, VERIFY_FAILURE_KEY, now)


def _session_of(note: "MemoryNote") -> str:
    return str(getattr(note, "session_id", "") or "")


def previous_turn_context(
    ordered: list["MemoryNote"], note: "MemoryNote",
) -> tuple[str, str]:
    """同一セッションの ``(直前のアシスタント応答, それ以前のユーザー発話)``。

    ``ordered`` は ``created_at`` 昇順のノート列 (純粋関数)。訂正より前を
    後ろから辿り、最初に見つかったアシスタント発話を「誤ったかもしれない
    応答」にする。``target=self`` の照合元は **直前 1 発話では足りない** —
    自己訂正は数ターン前の自己申告を直すのが普通で (2026-09-09 検証 V01:
    「盛岡市ではなく花巻市でした」の盛岡市は 2 ターン前の自己紹介にあり、
    直前のユーザー発話は猫の話だった → 検証器は ``none`` を返した)、
    ``_SELF_CONTEXT_TURNS`` 件までのユーザー発話を古い順に連結して渡す。
    見つからなければ空文字列。
    """
    session = _session_of(note)
    created_at = float(getattr(note, "created_at", 0.0) or 0.0)
    prev_response = ""
    prev_users: list[str] = []
    for candidate in reversed(ordered):
        if candidate is note or _session_of(candidate) != session:
            continue
        if float(getattr(candidate, "created_at", 0.0) or 0.0) >= created_at:
            continue
        source = str(getattr(candidate, "source", "user") or "user")
        content = str(getattr(candidate, "content", "") or "")
        if not content.strip():
            continue
        if not prev_response:
            if source == "assistant":
                prev_response = content
            continue
        if source == "user":
            prev_users.append(content)
            if len(prev_users) >= _SELF_CONTEXT_TURNS:
                break
    return prev_response, "\n".join(reversed(prev_users))


def reply_to(ordered: list["MemoryNote"], note: "MemoryNote") -> str:
    """同一セッションで ``note`` の **直後** のアシスタント応答 (純粋関数)。

    訂正への回答。無ければ空文字列。
    """
    session = _session_of(note)
    created_at = float(getattr(note, "created_at", 0.0) or 0.0)
    for candidate in ordered:
        if candidate is note or _session_of(candidate) != session:
            continue
        if float(getattr(candidate, "created_at", 0.0) or 0.0) <= created_at:
            continue
        if str(getattr(candidate, "source", "user") or "user") != "assistant":
            continue
        content = str(getattr(candidate, "content", "") or "")
        if content.strip():
            return content
    return ""


#: 訂正の形だが、アシスタントがその場で値を退けた (受け入れなかった)。
DISPUTED = "disputed"


def mark(
    note: "MemoryNote",
    verdict: str,
    *,
    wrong_claim: str = "",
    correct_value: str = "",
    now: float,
) -> None:
    """検証結果をノートへ刻む (冪等マーカーを含む)。"""
    note.correction_verified_at = now
    note.correction_verdict = verdict
    note.correction_wrong_claim = wrong_claim
    note.correction_correct_value = correct_value


def verdict_label(check) -> str:
    """``VerdictCheck`` を note へ載せる 1 語にする (純粋関数)。

    通ったものは帰属 (``assistant`` / ``self``)。却下は理由を載せるが、
    「訂正ではない帰属」(``premise_change`` / ``none`` / ``third_party``) は
    理由 (``not_correction``) より **帰属そのもの** の方が読んで分かるので
    そちらを採る。
    """
    if check.ok:
        return check.target
    if check.reason == "not_correction" and check.target:
        return check.target
    return check.reason or "none"


def pending_notes(notes: list["MemoryNote"]) -> list["MemoryNote"]:
    """検証待ちの訂正候補を発話時刻順で返す (純粋関数)。"""
    return sorted(
        (
            # 抽出済みは Step 8 が already_extracted で飛ばすので検証しても効果が
            # 無く、1 サイクルの枠 (``_MAX_PER_CYCLE``) を占めるだけ。据え置き側
            # (``verification_pending``) は already_extracted が先に判定される。
            n for n in notes
            if is_correction_candidate(n) and not getattr(n, "extracted_fact_ids", None)
        ),
        key=lambda n: float(getattr(n, "created_at", 0.0) or 0.0),
    )


async def curate_corrections(
    notes: list["MemoryNote"],
    *,
    aux_client,
    now_provider: Callable[[], float] | None = None,
    max_per_cycle: int = _MAX_PER_CYCLE,
    should_pause: Callable[[], bool] | None = None,
    stats: dict[str, int] | None = None,
    continue_on_preempt: bool = False,
) -> int:
    """訂正候補のノートを検証し、帰属と逐語 span をノートへ刻む。

    Args:
        notes: 直近の MemoryNote 群 (通常 ``EpisodicWorkspace.notes.values()``)。
        aux_client: 検証に使う補助タスククライアント。``None`` なら no-op。
        now_provider: 時刻供給。テスト用。
        max_per_cycle: 1 サイクルの上限。
        should_pause: 真ならチャットが最近あった (静穏窓の外)。検証を出さずに
            打ち切る — 出してもチャットに横取りされるだけ (2026-09-26 監査 #12)。
        stats: 渡すと ``preempted`` (チャットに譲って検証しなかった候補数 —
            ``should_pause`` と横取り) と ``failed`` (一過性の実失敗の数) を書く。
            「検証 0 件」が見送りか欠陥かを死活監視で分けるため (2026-10-02 監査)。
        continue_on_preempt: 真なら横取りされた候補だけを数えて次の候補へ進む
            (強制実行の Full。件数は上限で絞られており、先頭の 1 件で全体を止めない)。

    Returns:
        マーカーを立てたノート数 (却下も含む)。
    """
    if stats is not None:
        stats.update(preempted=0, failed=0)
    ordered = sorted(
        public_notes(list(notes)),
        key=lambda n: float(getattr(n, "created_at", 0.0) or 0.0),
    )
    now_fn = now_provider or time.time
    # cooldown 中は据え置きも外れている (``verification_pending``)。毎サイクル
    # aux へ出して 60 秒のタイムアウトを払い続けない。
    cooldown_now = now_fn()
    candidates = [
        n for n in pending_notes(ordered)
        if not in_cooldown(n, VERIFY_FAILURE_KEY, cooldown_now)
    ]
    if not candidates:
        return 0
    if aux_client is None:
        logger.debug(
            "correction_curator: aux_client is None, %d candidate(s) left "
            "unverified", len(candidates),
        )
        return 0
    if len(candidates) > max_per_cycle:
        logger.info(
            "correction_curator: %d candidate(s), verifying the oldest %d this "
            "cycle (the rest carry over)", len(candidates), max_per_cycle,
        )
        candidates = candidates[:max_per_cycle]

    marked = 0
    for index, note in enumerate(candidates):
        if should_pause is not None and should_pause():
            logger.info(
                "correction_curator: chat is recent; verification waits for a "
                "quiet window (%d candidate(s) carried over)", len(candidates) - index,
            )
            if stats is not None:
                stats["preempted"] += len(candidates) - index
            break
        content = str(note.content or "")
        prev_response, prev_user = previous_turn_context(ordered, note)
        if not prev_response.strip():
            # 相手の応答が無ければ「誤りの指摘」は成立しない。
            mark(note, NO_CONTEXT, now=now_fn())
            marked += 1
            continue
        try:
            parsed = await aux_client.generate_json(
                build_correction_verify_prompt(
                    prev_response, content, prev_user=prev_user,
                ),
                purpose="correction_verify",
                max_tokens=_VERIFY_MAX_TOKENS,
                temperature=_VERIFY_TEMPERATURE,
                response_schema=CorrectionVerdict,
            )
        except Exception as exc:  # noqa: BLE001 - 一過性失敗でマーカーを立てない
            logger.warning(
                "correction_curator: verification failed (note=%s): %r",
                getattr(note, "id", "?"), exc,
            )
            # Step 8 が「検証待ち」として据え置く根拠 (:func:`verification_pending`)。
            # チャット併走の横取り (contended) は数えない — 静かなサイクルで
            # そのまま再試行する。実失敗が閾値に達したら cooldown に入り、
            # Step 8 は通常の再言明として消費する (永久に据え置かない)。
            contended = bool(getattr(exc, "contended", False))
            record_transient_failure(
                note, VERIFY_FAILURE_KEY, now_fn(), counts=not contended,
            )
            if contended and continue_on_preempt:
                if stats is not None:
                    stats["preempted"] += 1
                continue
            if contended:
                # 横取りされたなら次の候補もまた横取りされる。残りは次の
                # サイクルへ回す (2026-09-26 監査 #12: 11 回とも全滅)。
                if stats is not None:
                    stats["preempted"] += len(candidates) - index
                break
            if stats is not None:
                stats["failed"] += 1
            continue
        if not isinstance(parsed, dict) or not parsed:
            # 出力の切断・パース不能で答えが取れなかった。判定ではないので刻まず、
            # 例外と同じ一過性失敗として数える (2026-09-26 監査: 崩れた出力の空の
            # 結果が invalid_target として刻まれ、二度と検証されなかった)。
            logger.warning(
                "correction_curator: no usable verdict (note=%s); will retry",
                getattr(note, "id", "?"),
            )
            record_transient_failure(note, VERIFY_FAILURE_KEY, now_fn())
            if stats is not None:
                stats["failed"] += 1
            continue
        clear_failure(note, VERIFY_FAILURE_KEY)
        check = check_verdict(
            parsed,
            candidate=content,
            prev_response=prev_response,
            prev_user=prev_user,
        )
        verdict = verdict_label(check)
        # 訂正への回答が値を退けていれば、検証済み訂正として消費しない
        # (:func:`answer_disputes_value` の docstring、J-04)。
        if check.ok and answer_disputes_value(
            reply_to(ordered, note), check.correct_value,
        ):
            logger.info(
                "correction_curator: note=%s value %r was disputed by the reply; "
                "not consumed as a correction", getattr(note, "id", "?"),
                check.correct_value[:40],
            )
            verdict = DISPUTED
        mark(
            note, verdict,
            wrong_claim=check.wrong_claim,
            correct_value=check.correct_value,
            now=now_fn(),
        )
        # 学習側 (``learning.correction_verifier``) は同じ候補をこの生の出力で
        # 判定する (補助タスクを 2 度払わない)。記憶側は複数ターンの文脈で
        # 問うているので、共有の書き手はこちらだけにする。
        put_shared_verdict(_session_of(note), content, parsed, verdict=verdict)
        marked += 1
        logger.debug(
            "correction_curator: note=%s verdict=%s wrong=%r correct=%r",
            getattr(note, "id", "?"), verdict,
            check.wrong_claim[:40], check.correct_value[:40],
        )
    if marked:
        logger.info("correction_curator: verified %d correction candidate(s)", marked)
    return marked


__all__ = [
    "DISPUTED",
    "NO_CONTEXT",
    "reply_to",
    "curate_corrections",
    "mark",
    "pending_notes",
    "previous_turn_context",
    "verdict_label",
]
