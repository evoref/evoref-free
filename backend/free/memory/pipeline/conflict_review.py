"""SemMem pending 競合のユーザー解決ヘルパ (EvorefMem 内部)

``semantic_conflict_resolver.py`` (sleep-time Step 6B) が ``review_status=
"pending"`` に振り分けた競合を、ユーザーの意思決定で解決するための共通
ロジックを提供する。チャット確認フロー (``conflict_chat_judge`` /
``chat_service.maybe_resolve_pending_conflicts``) から利用する
(専用 API 層は PR #106 のメモリインスペクタ削除で撤去済)。

提供する操作:

- :func:`collect_pending_groups` — pending ファクトの (subject, predicate)
  グルーピング (読取のみ)
- :func:`apply_resolution` — keep_old / keep_new / merge の supersede 反映 +
  ``conflicts.jsonl`` の pending 行掃除 + ``conflicts_resolved.jsonl`` への
  audit 追記までを一体で実行する

書込例外について: チャット応答パスからの SemMem 書込は sleep-time に
閉じる不変則の例外 2 例目として、:func:`apply_resolution` のみ許可される
(CLAUDE.md §6.2 / docs/f_02_memory_system.md §5.2 参照)。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.free.memory.pipeline.semantic_conflict_resolver import (
    CONFLICTS_PENDING_FILENAME,
    CONFLICTS_RESOLVED_FILENAME,
)
from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.types import SemanticFact, make_fact
from backend.log_config import get_logger

logger = get_logger("memory.semantic.conflict_review")


_RESOLVED_ACTION_TO_STATUS: dict[str, str] = {
    "keep_old": "resolved_keep_old",
    "keep_new": "resolved_keep_new",
    "merge": "resolved_merged",
}


class AlreadyResolvedError(ValueError):
    """対象ファクトが既に supersede 済み (= 解決済み) の場合に送出する。

    チャット確認フロー (``chat_service.maybe_resolve_pending_conflicts``) では
    汎用 except に吸収され pending 維持で no-op になる (専用 API 層と 409 への
    対応付けは PR #106 のメモリインスペクタ削除で撤去済)。
    """


@dataclass(frozen=True)
class PendingConflictGroup:
    """pending 競合の 1 グループ (同 subject / predicate)。

    ``type`` は表示用に member facts から導出した文字列 (混在時は
    ``"a/b"`` 形式)。``facts`` は ``created_at`` 昇順。keep_new の winner は
    ``facts[-1]``、keep_old の winner は ``facts[0]``。
    """

    scope: str
    subject: str
    predicate: str
    type: str
    facts: tuple[SemanticFact, ...]

    @property
    def newest(self) -> SemanticFact:
        return self.facts[-1]

    @property
    def oldest(self) -> SemanticFact:
        return self.facts[0]


@dataclass(frozen=True)
class ResolutionResult:
    """:func:`apply_resolution` の適用結果。"""

    scope: str
    action: str
    winner_id: str
    superseded_ids: tuple[str, ...]
    new_fact_id: str | None


def collect_pending_groups(
    store: SemanticFactStore, scope: str,
) -> list[PendingConflictGroup]:
    """``review_status="pending"`` の active ファクトを (subject, predicate)
    でグルーピングして返す。

    グルーピングキーは producer ``SemanticConflictResolver._detect_groups``
    (``(subject, predicate)``) と一致させること。type を加えると、producer が
    同 (subject, predicate) で type 混在の競合を pending に振り分けた場合に
    consumer 側で type ごとに単独バケット化して両方除外され、その競合が
    チャットに一度も注入されず永久に pending のまま残る (再検出も skip される)。

    単独 pending (グループとして競合になっていないもの) は除外する。
    並びは ``(scope, subject, predicate)`` の決定的順序 — チャット注入の
    ``[C1]`` 採番と assist 判定プロンプトで同一の番号を共有するため、
    呼び出しタイミングに依らず安定であること。
    """
    pending = [
        f for f in store.all_facts(include_superseded=False)
        if f.review_status == "pending"
    ]
    buckets: dict[tuple[str, str], list[SemanticFact]] = {}
    for f in pending:
        buckets.setdefault((f.subject, f.predicate), []).append(f)

    groups: list[PendingConflictGroup] = []
    for (subject, predicate), facts in buckets.items():
        if len(facts) < 2:
            continue
        facts.sort(key=lambda f: f.created_at)
        # type 混在グループの表示用に member facts から導出する。
        type_ = "/".join(sorted({f.type for f in facts}))
        groups.append(PendingConflictGroup(
            scope=scope,
            subject=subject,
            predicate=predicate,
            type=type_,
            facts=tuple(facts),
        ))
    groups.sort(key=lambda g: (g.scope, g.subject, g.predicate, g.type))
    return groups


def append_resolution_log(
    store: SemanticFactStore,
    *,
    scope: str,
    winner: SemanticFact,
    loser_ids: list[str],
    action: str,
    new_fact_id: str | None,
    decision: str = "user",
    trace_id: str | None = None,
) -> None:
    """解消を ``conflicts_resolved.jsonl`` に追記する (audit trail)。

    ``decision`` は解決経路の識別子 — ``"user"`` (API 経由) /
    ``"user_chat"`` (チャット確認フロー経由) / ``"ttl_auto"`` (sleep-time の
    pending TTL 自動解消)。``reason`` は user 経路では ``"user_<action>"``、
    それ以外の経路では ``"<decision>_<action>"`` (例 ``"ttl_auto_keep_new"``)。
    """
    path = store.root_dir / CONFLICTS_RESOLVED_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    reason = (
        f"user_{action}"
        if decision in ("user", "user_chat")
        else f"{decision}_{action}"
    )
    entry: dict = {
        "ts": time.time(),
        "scope": scope,
        "subject": winner.subject,
        "predicate": winner.predicate,
        "type": winner.type,
        "winner_id": new_fact_id or winner.id,
        "loser_ids": loser_ids if not new_fact_id else [winner.id, *loser_ids],
        "decision": decision,
        "reason": reason,
    }
    if trace_id:
        entry["trace_id"] = trace_id
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def remove_pending_lines(
    store: SemanticFactStore, *, fact_ids: set[str],
) -> None:
    """``conflicts.jsonl`` から ``fact_ids`` を含むエントリを削除する。

    ファイル全体を書き換える (規模的に十分実用範囲)。
    """
    path = store.root_dir / CONFLICTS_PENDING_FILENAME
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("read conflicts.jsonl failed: %s", exc)
        return
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            entry = json.loads(s)
        except json.JSONDecodeError:
            kept.append(s)
            continue
        ids = {entry.get("winner_id"), *(entry.get("loser_ids") or [])}
        if ids & fact_ids:
            continue
        kept.append(s)
    path.write_text(
        ("\n".join(kept) + "\n") if kept else "",
        encoding="utf-8",
    )


def apply_resolution(
    store: SemanticFactStore,
    *,
    scope: str,
    action: str,
    winner_id: str,
    loser_ids: list[str],
    merged_object: str | None = None,
    decision_source: str = "user",
    trace_id: str | None = None,
) -> ResolutionResult:
    """pending 競合に keep_old / keep_new / merge を適用する。

    supersede 反映 → ``review_status`` 更新 → ``conflicts.jsonl`` の
    pending 行掃除 → ``conflicts_resolved.jsonl`` への audit 追記までを
    一体で実行する。

    Raises:
        ValueError: action 不正 / winner_id が loser_ids に含まれる /
            merge で merged_object 欠落。
        KeyError: winner / loser ファクトが存在しない。
        AlreadyResolvedError: 対象ファクトが既に supersede 済み。
    """
    if action not in _RESOLVED_ACTION_TO_STATUS:
        raise ValueError(f"unknown action: {action}")

    winner = store.get_fact(winner_id)
    if winner is None:
        raise KeyError(f"winner fact not found: {winner_id}")
    if winner.superseded_by is not None:
        raise AlreadyResolvedError(
            f"winner fact already superseded: {winner_id}",
        )
    losers: list[SemanticFact] = []
    for lid in loser_ids:
        if lid == winner_id:
            raise ValueError("winner_id must not appear in loser_ids")
        loser = store.get_fact(lid)
        if loser is None:
            raise KeyError(f"loser fact not found: {lid}")
        if loser.superseded_by is not None:
            raise AlreadyResolvedError(
                f"loser fact already superseded: {lid}",
            )
        losers.append(loser)

    superseded_ids: list[str] = []
    new_fact_id: str | None = None
    review_status = _RESOLVED_ACTION_TO_STATUS[action]

    if action == "merge":
        if not merged_object:
            raise ValueError("merged_object is required for action=merge")
        # winner / losers の supersedes チェーンを集約しつつ新ファクトを作成
        merged_supersedes: list[str] = []
        for f in (winner, *losers):
            for sid in f.supersedes:
                if sid not in merged_supersedes:
                    merged_supersedes.append(sid)
        new_fact = make_fact(
            subject=winner.subject,
            predicate=winner.predicate,
            object_=merged_object,
            type=winner.type,  # type: ignore[arg-type]
            scope=winner.scope,
            mode_origin=winner.mode_origin,
            confidence=max(winner.confidence, *(l.confidence for l in losers)),
        )
        added = store.add_fact(new_fact)
        new_fact_id = added.id
        # winner も含めて全件 supersede
        for f in (winner, *losers):
            try:
                store.supersede(f.id, new_fact_id)
                superseded_ids.append(f.id)
            except (KeyError, ValueError) as exc:
                logger.warning("merge supersede failed for %s: %s", f.id, exc)
        store.update_fact(
            new_fact_id,
            requires_user_review=False,
            review_status=review_status,
            supersedes=list({*merged_supersedes, *(f.id for f in (winner, *losers))}),
        )
    else:
        # keep_old / keep_new: losers のみ supersede
        for loser in losers:
            try:
                store.supersede(loser.id, winner.id)
                superseded_ids.append(loser.id)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "supersede failed for %s -> %s: %s",
                    loser.id, winner.id, exc,
                )
        store.update_fact(
            winner.id,
            requires_user_review=False,
            review_status=review_status,
        )

    # pending エントリも掃除し、audit trail を残す
    remove_pending_lines(store, fact_ids={winner_id, *loser_ids})
    append_resolution_log(
        store,
        scope=scope,
        winner=winner,
        loser_ids=[l.id for l in losers],
        action=action,
        new_fact_id=new_fact_id,
        decision=decision_source,
        trace_id=trace_id,
    )

    return ResolutionResult(
        scope=scope,
        action=action,
        winner_id=winner_id,
        superseded_ids=tuple(superseded_ids),
        new_fact_id=new_fact_id,
    )


# ──────────────────────────────────────────────────────────────────────────
# チャット注入レンダラ
# ──────────────────────────────────────────────────────────────────────────


def _short_date(ts: float) -> str:
    """created_at (epoch) を MM-DD 表記へ。不正値は空文字列。"""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_group_line(
    index: int, group: PendingConflictGroup, *, max_object_chars: int,
) -> str:
    """1 グループを ``[C1] (type) subject predicate: 旧「…」 / 新「…」`` 形式に。"""
    parts = [
        f"旧「{_truncate(f.object, max_object_chars)}」({_short_date(f.created_at)})"
        if i == 0
        else f"新「{_truncate(f.object, max_object_chars)}」({_short_date(f.created_at)})"
        for i, f in enumerate(group.facts)
    ]
    return (
        f"[C{index}] ({group.type}) {group.subject} {group.predicate}: "
        + " / ".join(parts)
    )


def render_pending_conflicts_block(
    groups: list[PendingConflictGroup],
    *,
    instruct: bool = True,
    max_groups: int = 0,
    max_object_chars: int = 80,
) -> str | None:
    """pending 競合グループをプロンプト注入用テキストへ整形する。

    採番 (``[C1]``〜) は :func:`collect_pending_groups` の決定的順序に
    従う。assist 判定プロンプト (:func:`judge_user_reply`) と同一の
    レンダリングを共有し、ターン間で番号がずれないようにする。

    Args:
        instruct: True で AI への確認指示行を含める。assist 未接続
            (回答を判定できない) のときは False で情報提示のみにする。
        max_groups: 注入するグループ数の上限 (0 = 無制限)。超過分は
            件数のみ要約する。
    """
    if not groups:
        return None
    shown = groups if max_groups <= 0 else groups[:max_groups]
    lines = ["[記憶の競合 — 未解決]"]
    if instruct:
        lines.append(
            "以下の記憶は内容が矛盾しています。会話の流れを壊さない範囲で、"
            "どちらが正しいか自然にユーザーへ確認してください (1 ターンに 1 件まで)。"
        )
    else:
        lines.append("以下の記憶は内容が矛盾しています (確認待ち)。")
    for i, g in enumerate(shown, start=1):
        lines.append(_render_group_line(i, g, max_object_chars=max_object_chars))
    if len(groups) > len(shown):
        lines.append(f"(他 {len(groups) - len(shown)} 件)")
    return "\n".join(lines)


def render_resolved_notice(
    result: ResolutionResult, group: PendingConflictGroup,
) -> str:
    """直前のユーザー回答で解決した競合の確認通知行を返す。

    同ターンの注入ブロック末尾に追加し、AI に「記憶を更新した」旨を
    一言返させる。解決後は pending ブロックの ``[C1..]`` が再収集で振り直される
    ため、``[C{index}]`` 参照は同一ターン内ですら別の競合を指し得る。それを
    避けるため、解決した競合は番号ではなく内容 (subject / predicate) で示す。
    """
    return (
        f"(直前のユーザー回答により記憶の競合「{group.subject} {group.predicate}」を "
        f"{result.action} で解決し、記憶を更新済み。"
        "応答の冒頭で一言だけ反映した旨を伝えること。)"
    )


# ──────────────────────────────────────────────────────────────────────────
# assist 回答判定
# ──────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """あなたは記憶管理アシスタント。直前の会話で、AI はユーザーに以下の記憶の競合について確認した可能性がある。

競合一覧:
{conflicts}

直前の AI 発話 (抜粋):
{assistant_message}

ユーザーの最新発話:
{user_message}

ユーザーの発話が競合のどれかへの回答であるかを判定せよ。
- 回答である場合: is_answer=true、group_index に対象番号 ([C1]=1)、action に keep_old (古い方が正しい) / keep_new (新しい方が正しい) / merge (両方正しい・統合) のいずれかを設定。merge の場合は merged_object に統合後の値を書く。
- 雑談・無関係・どちらとも取れない場合: is_answer=false, action="none"。迷ったら必ず is_answer=false にすること。
JSON のみで応答せよ。"""


async def judge_user_reply(
    assist_client,
    *,
    groups: list[PendingConflictGroup],
    user_message: str,
    last_assistant_message: str,
    max_object_chars: int = 80,
) -> dict | None:
    """ユーザー発話が pending 競合への回答かを assist で判定する。

    戻り値は検証済みの判定 dict (``group_index`` は 1 始まりで
    ``groups`` の範囲内、``action`` は keep_old/keep_new/merge のいずれか)。
    回答でない / 応答が壊れている / 例外時はすべて ``None`` (= no-op)。
    アシストが json_schema grammar を強制しないモデル (LFM2 系) でも
    安全なよう、消費側で多段バリデーションする。
    """
    if assist_client is None or not groups:
        return None
    conflicts_text = "\n".join(
        _render_group_line(i, g, max_object_chars=max_object_chars)
        for i, g in enumerate(groups, start=1)
    )
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        conflicts=conflicts_text,
        assistant_message=_truncate(last_assistant_message or "(なし)", 300),
        user_message=_truncate(user_message, 500),
    )
    try:
        result = await assist_client.generate_json(
            prompt,
            purpose="conflict_chat_judge",
            max_tokens=192,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — 判定失敗は no-op で吸収
        logger.warning("conflict_chat_judge failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    if not result.get("is_answer"):
        return None
    action = result.get("action")
    if action not in ("keep_old", "keep_new", "merge"):
        return None
    try:
        group_index = int(result.get("group_index", 0))
    except (TypeError, ValueError):
        return None
    if not (1 <= group_index <= len(groups)):
        return None
    merged_object = str(result.get("merged_object") or "")
    if action == "merge" and not merged_object.strip():
        return None
    return {
        "group_index": group_index,
        "action": action,
        "merged_object": merged_object,
    }
