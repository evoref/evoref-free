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


from backend.free.core.text_quality import carries_no_assertion
from backend.free.memory.pipeline.semantic_conflict_resolver import (
    CONFLICTS_PENDING_FILENAME,
    CONFLICTS_RESOLVED_FILENAME,
    DEFAULT_ATTRIBUTE_SIMILARITY_THRESHOLD,
    distinct_conflict_objects,
    split_by_attribute_similarity,
)
from backend.free.memory.semantic.store import SemanticFactStore
from backend.utils import estimate_tokens
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
    similarity_threshold: float = DEFAULT_ATTRIBUTE_SIMILARITY_THRESHOLD,
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
    ``[C1]`` 採番をチャット注入ブロックと共有するため、
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
        # producer が類似度で塊に割っているので、consumer 側でも同じ割り方を
        # する。しないと別々に pending になった無関係な塊が (subject, predicate)
        # だけで再合流し、producer が避けた偽の競合が表示側で復活する。
        for cluster in split_by_attribute_similarity(
            facts, similarity_threshold,
        ):
            if len(cluster) < 2:
                continue
            # 原文と [要約] だけの塊は矛盾ではなく重複。
            if len(distinct_conflict_objects(cluster)) < 2:
                continue
            # type 混在グループの表示用に member facts から導出する。
            type_ = "/".join(sorted({f.type for f in cluster}))
            groups.append(PendingConflictGroup(
                scope=scope,
                subject=subject,
                predicate=predicate,
                type=type_,
                facts=tuple(cluster),
            ))
    groups.sort(
        key=lambda g: (
            g.scope, g.subject, g.predicate, g.type, g.facts[0].created_at,
        ),
    )
    return groups


def collect_review_groups(
    store: SemanticFactStore, scope: str,
    similarity_threshold: float = DEFAULT_ATTRIBUTE_SIMILARITY_THRESHOLD,
) -> list[PendingConflictGroup]:
    """**表示用**の競合グループ。スロットの現在値まで含めて返す。

    :func:`collect_pending_groups` との違いは 2 つで、どちらも「ユーザーに何を
    見せるか」の問題。**解決 (supersede) の対象は変えない** — TTL 自動解決は
    引き続き pending だけを見る (非 pending を巻き込むと、ユーザーがレビューに
    同意していないファクトを消してしまう)。

    1. **スロットの非 pending ファクトも表示に含める。**
       競合検出より後に作られたファクトには pending が付かないため、
       pending だけを並べるとそのスロットの **最新値が欠落**する。旧/新 の
       ラベルは残ったメンバの中で付くので、古い値が「新」として提示される。

       実インシデント (2026-08-09): ``mem.personal.name`` の表示が
       ``旧「好きな季節は秋」(08-05) / 新「趣味は自転車と写真」(08-08 06:38)``
       となり、実際の最新値「趣味は登山と写真」(08-08 12:58、
       ``review_status=none``) が含まれていなかった。競合は 08-08 07:48 に
       検出され ``pinned_present`` で pending のまま滞留していたため、5 時間後に
       現れた正しい値は永久に合流できない。**この「新」ラベルはプロンプト中で
       最も強い現在値の信号**で、記憶注入 / few-shot / RAG をすべて正しくしても
       これだけで古い値が採用され続けた。

    2. **問いだけのファクトを除く。**
       「私の猫の名前と誕生日を覚えていますか。」が競合の当事者として並び、
       最新であるがゆえに winner 扱いされていた。問いは主張ではないので
       競合し得ない。注入層は同じ判定をファクトへ既に掛けている。

    3. **同一 object は 1 件に畳む。**
       ストアは append-only で同じ文が複数 id で残る。畳まないと
       ``旧「私の趣味をもう一度確認させてください。」/ 新「私の趣味をもう一度
       確認させてください。」`` のように **同じ文が旧と新の両方**に並び、
       ブロック自体が自己矛盾する (実測 2026-08-09)。
    """
    by_slot: dict[tuple[str, str], list[SemanticFact]] = {}
    for f in store.all_facts(include_superseded=False):
        if carries_no_assertion(f.object or ""):
            continue
        by_slot.setdefault((f.subject, f.predicate), []).append(f)

    groups: list[PendingConflictGroup] = []
    for base in collect_pending_groups(store, scope, similarity_threshold):
        newest_by_object: dict[str, SemanticFact] = {}
        for f in by_slot.get((base.subject, base.predicate), []):
            cur = newest_by_object.get(f.object)
            if cur is None or f.created_at > cur.created_at:
                newest_by_object[f.object] = f
        members = list(newest_by_object.values())
        if len(members) < 2:
            continue
        # スロット全件を引き戻すので、pending の塊と同じ属性を語っているものだけに
        # 絞り直す。これをしないと producer / collect_pending_groups で分けた
        # 無関係なファクトが表示段階で戻ってくる。
        members.sort(key=lambda f: f.created_at)
        pending_ids = {f.id for f in base.facts}
        members = next(
            (
                cluster
                for cluster in split_by_attribute_similarity(
                    members, similarity_threshold,
                )
                if any(f.id in pending_ids for f in cluster)
            ),
            [],
        )
        if len(members) < 2:
            continue
        members.sort(key=lambda f: f.created_at)
        groups.append(PendingConflictGroup(
            scope=base.scope,
            subject=base.subject,
            predicate=base.predicate,
            type="/".join(sorted({f.type for f in members})),
            facts=tuple(members),
        ))
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
    """1 グループを ``[C1] (type) subject predicate: 旧「…」 / 新「…」`` 形式に。

    ``新`` は **最新の 1 件だけ**。3 件以上のグループで先頭以外を全部 ``新`` に
    すると「新」が複数現れ、どれが現在値か分からなくなる (実インシデント
    2026-08-09: 3 件のグループで古い値と質問の両方に「新」が付いていた)。
    """
    facts = group.facts
    last = len(facts) - 1
    parts = [
        f"新「{_truncate(f.object, max_object_chars)}」({_short_date(f.created_at)})"
        if i == last
        else f"旧「{_truncate(f.object, max_object_chars)}」({_short_date(f.created_at)})"
        for i, f in enumerate(facts)
    ]
    return (
        f"[C{index}] ({group.type}) {group.subject} {group.predicate}: "
        + " / ".join(parts)
    )


def _render_block_lines(
    groups: list[PendingConflictGroup],
    *,
    instruct: bool,
    max_object_chars: int,
) -> list[str]:
    """ブロック本文 (ヘッダ + 指示 + グループ行) を組み立てる。"""
    lines = ["[記憶の競合 — 未解決]"]
    if instruct:
        lines.append(
            "以下の記憶は内容が矛盾しています。会話の流れを壊さない範囲で、"
            "どちらが正しいか自然にユーザーへ確認してください (1 ターンに 1 件まで)。"
        )
    else:
        lines.append("以下の記憶は内容が矛盾しています (確認待ち)。")
    for i, g in enumerate(groups, start=1):
        lines.append(_render_group_line(i, g, max_object_chars=max_object_chars))
    return lines


def _fit_groups_to_tokens(
    groups: list[PendingConflictGroup],
    max_tokens: int,
    *,
    instruct: bool,
    max_object_chars: int,
) -> list[PendingConflictGroup]:
    """トークン上限に収まる先頭 N グループを返す (最低 1 件は残す)。"""
    if not groups:
        return groups
    fitted: list[PendingConflictGroup] = []
    for g in groups:
        candidate = [*fitted, g]
        text = "\n".join(
            _render_block_lines(
                candidate, instruct=instruct, max_object_chars=max_object_chars,
            ),
        )
        if fitted and estimate_tokens(text) > max_tokens:
            break
        fitted.append(g)
    return fitted


def render_pending_conflicts_block(
    groups: list[PendingConflictGroup],
    *,
    instruct: bool = True,
    max_groups: int = 0,
    max_object_chars: int = 80,
    max_tokens: int = 0,
) -> str | None:
    """pending 競合グループをプロンプト注入用テキストへ整形する。

    採番 (``[C1]``〜) は :func:`collect_pending_groups` の決定的順序に
    従う。番号の採番規則と同一の
    レンダリングを共有し、ターン間で番号がずれないようにする。

    Args:
        instruct: True で AI への確認指示行を含める。aux 未接続
            (回答を判定できない) のときは False で情報提示のみにする。
        max_groups: 注入するグループ数の上限 (0 = 無制限)。超過分は
            件数のみ要約する。
        max_tokens: ブロック全体のトークン上限 (0 = 無制限)。グループ数上限
            だけでは member 数の多いグループを抑えられないため、トークンでも
            打ち切る。設計書 §203 の「drop されない」保証を守るため、
            **最低 1 グループは必ず残す** (上限を超えても先頭 1 件は出す)。
    """
    if not groups:
        return None
    shown = groups if max_groups <= 0 else groups[:max_groups]
    if max_tokens > 0:
        shown = _fit_groups_to_tokens(
            shown, max_tokens, instruct=instruct,
            max_object_chars=max_object_chars,
        )
    lines = _render_block_lines(
        shown, instruct=instruct, max_object_chars=max_object_chars,
    )
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
