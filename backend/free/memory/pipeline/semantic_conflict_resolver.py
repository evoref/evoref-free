"""

EvorefMem 統合仕様 における sleep-time **Step 6** の SemMem 対応分
``backend/free/memory/pipeline/conflict_resolver.py`` は ShortTermMemory (FadeMem)
向けで、本モジュールは ``SemanticFactStore`` 上で同 ``(subject, predicate)``
を持つ複数ファクトの競合を解消する。

設計仕様:

1. ``project_tag_always_manual: true`` を基本とする。
   ``project`` / ``policy`` タグの競合は ``review_status="pending"`` に
   振り分け、チャット確認フロー (``conflict_review`` /
   ``conflict_chat_judge``) で人手解決する (専用 UI は PR #106 で撤去済)
2. 例外として、``auto_for_evolved_policies: true`` かつ winner が
   ``auto_evolved=True`` の ``policy`` ファクトの場合のみ自動マージする
   (PolicyEvolver 由来の進化結果)。
3. ``default_mode: auto`` ではそれ以外のタグは原則 newest-wins で
   supersede するが、微妙ケース (同 source / ``confirm_window_hours``
   以内) は pending に振り分ける。
4. ``default_mode: manual`` では全件 pending。
5. pinned ファクトを含む競合は常に pending (誤って自動消失するのを防ぐ)。
6. ``pending_auto_resolve_days`` (既定 3 日) 超過の滞留 pending は
   ``resolve()`` 冒頭の TTL pre-pass (``_resolve_expired_pending``) が
   keep_new で自動解消する (``conflicts_resolved.jsonl`` に
   ``decision="ttl_auto"``)。**pinned / project / policy も対象**
   (チャット回答での解決経路が 2026-08-14 に撤去され、除外すると解決手段が
   ゼロになるため)。``default_mode=manual`` でも有効、0 で無効。

ファイル出力 (1 ストアあたり)::

    <store.root_dir>/
    ├── conflicts.jsonl            # 追記式: pending 状態の競合エントリ
    └── conflicts_resolved.jsonl   # 追記式: 自動解決された競合エントリ

各行スキーマ::

    {
      "ts": float,
      "scope": str,
      "subject": str,
      "predicate": str,
      "type": str,
      "winner_id": str,
      "loser_ids": [str, ...],
      "decision": "auto" | "pending",
      "reason": str
    }
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.semantic.conflict")

#: 同 ``(subject, predicate)`` のファクトを「同じ属性についての言い直し」と
#: みなす最小コサイン類似度。
#:
#: subject は属性単位へ分割される設計だが (``resolve_fact_attribute``)、
#: トリガ辞書に無い属性は ``mem.personal.user`` へフォールバックする。すると
#: **飲み物の好みと食べ物の好みが同じスロットに入り**、``(subject, predicate)``
#: だけで束ねる検出器がそれを競合と誤判定する。
#:
#: 実インシデント (2026-08-16 ライブ監査): 毎ターン注入されていた競合ブロック
#:   [C1] (personal_fact) mem.personal.user states:
#:   旧「私はコーヒーより紅茶派です。」/ 旧「…担々麺にハマってて、週2で…」/
#:   旧「コーヒー派？紅茶派？私はコーヒーを1日3杯は飲んじゃう。」…
#: は、飲み物 2 件 + 食べ物 1 件 (と各々の [要約]) を 1 つの矛盾として提示していた。
#:
#: 実測 (同ストアの実ファクト、LFM2.5-Embedding-350M):
#:   真の競合 (同じ属性の言い直し) … 0.796 / 0.798 / 0.956 / 0.963
#:   偽の競合 (別の属性)          … 0.316 / 0.329 / 0.335 / 0.371 / 0.383 / 0.418
#: 分離は完全で、0.45〜0.75 のどこを取っても正しく分かれる。競合の取りこぼしは
#: 古い値が残り続ける害があるため (2026-08-16 監査の主要因の一つ)、中央より
#: 低めの 0.55 を採る。
#:
#: ``memory.conflict_similarity_threshold`` (0.85) とは別物。あちらは STM ノートの
#: 「同一ノートか」の判定で、ここは「同じ属性についての言明か」の判定。
DEFAULT_ATTRIBUTE_SIMILARITY_THRESHOLD = 0.55


#: ``compress_turn(style="summary")`` の圧縮マークと末尾の元文字数。
_SUMMARY_MARK = "[要約] "
_SUMMARY_TAIL_RE = re.compile(r"…（\d+文字）\s*$")


def normalize_object_for_conflict(text: str) -> str:
    """競合の「値が違うか」を見るための object 正規化 (純粋関数)。

    同じ発話から原文と ``[要約]`` 版の 2 ファクトが生まれることがある。文字列
    としては違うので「値が 2 つある = 競合」と判定されるが、中身は同じ発話で
    矛盾していない。

    実インシデント (2026-08-16 ライブ監査): 「最近ハマってる食べ物ってある？
    私は担々麺に…」とその ``[要約]`` 版が「内容が矛盾しています」として毎ターン
    提示されていた。
    """
    body = (text or "").strip()
    if body.startswith(_SUMMARY_MARK):
        body = body[len(_SUMMARY_MARK):]
    body = _SUMMARY_TAIL_RE.sub("", body)
    return "".join(body.split())


def _truncation_marked(text: str) -> bool:
    """``text`` が切り詰め済みであることを **印で** 判定する (純粋関数)。

    切り詰めの経路は 2 つで、どちらも印を残す:

    - 要約 (``compress_turn(style="summary")``) — ``[要約] `` 接頭辞と
      ``…（N文字）`` 末尾
    - object の長さ制限 (``BaseExtractor.truncate``) — 末尾の ``…``
    """
    t = (text or "").strip()
    return (
        t.startswith(_SUMMARY_MARK)
        or bool(_SUMMARY_TAIL_RE.search(t))
        or t.endswith("…")
    )


def distinct_conflict_objects(facts: Iterable[SemanticFact]) -> set[str]:
    """競合判定に使う「異なる値」の集合。

    原文と ``[要約]`` は同一視する (:func:`normalize_object_for_conflict`)。
    片方が他方の接頭辞になる場合も同一視するが、**切り詰めの印がある側が
    あるときだけ** (:func:`_truncation_marked`)。

    印を要求するのは、判定が ``fact.text`` (= ``statement or object``) に
    掛かるため。``statement`` が正規化済みの短い命題で埋まると、印の無い
    「名前は小川」と「名前は小川浩之」が接頭辞一致で同じ値と見なされ、
    **本物の競合が検出されなくなる** (= 古い値が supersede されない)。
    長さの閾値では要約側の実長 (テストでは 12 文字) と命題の実長が重なって
    区別できないので、**印そのもの** を判定に使う (2026-08-26)。
    """
    keys: list[tuple[str, bool]] = []
    for f in facts:
        raw = f.text or ""
        key = normalize_object_for_conflict(raw)
        if not key:
            continue
        marked = _truncation_marked(raw)
        if any(
            _is_same_value(key, marked, k, k_marked) for k, k_marked in keys
        ):
            continue
        keys.append((key, marked))
    return {k for k, _ in keys}


def _is_same_value(a: str, a_marked: bool, b: str, b_marked: bool) -> bool:
    """``a`` と ``b`` が同じ値を指すか (切り詰め違いを含む、純粋関数)。"""
    if a == b:
        return True
    if not (a_marked or b_marked):
        return False
    return a.startswith(b) or b.startswith(a)


def split_by_attribute_similarity(
    facts: list[SemanticFact], threshold: float,
) -> list[list[SemanticFact]]:
    """同スロットのファクト群を「同じ属性について語っている塊」へ分ける。

    類似度が ``threshold`` 以上のペアを辺として連結成分を取る。埋め込みを持たない
    ファクトは判定できないので **単独では切り離さず** 全体と繋いだままにする
    (競合の取りこぼしより偽の競合の方がまし、という判断はしない — 取りこぼすと
    古い値が恒久的に残る)。

    Returns:
        塊のリスト。入力順 (created_at 昇順) は各塊の中で保たれる。
    """
    if len(facts) < 2 or threshold <= 0:
        return [facts]
    vectors: list[Any] = []
    no_embedding: list[int] = []
    for i, f in enumerate(facts):
        emb = getattr(f, "embedding", None)
        if emb is None:
            no_embedding.append(i)
            vectors.append(None)
            continue
        vec = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        vectors.append(vec / norm if norm else None)
        if norm == 0:
            no_embedding.append(i)
    # union-find
    parent = list(range(len(facts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            vi, vj = vectors[i], vectors[j]
            if vi is None or vj is None:
                # 判定できない組は繋いだままにする (取りこぼし防止)。
                union(i, j)
                continue
            if float(vi @ vj) >= threshold:
                union(i, j)

    groups: dict[int, list[SemanticFact]] = {}
    for i, f in enumerate(facts):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


CONFLICTS_PENDING_FILENAME = "conflicts.jsonl"
CONFLICTS_RESOLVED_FILENAME = "conflicts_resolved.jsonl"

# 自動解決を許可しない (基本) タグ集合
_MANUAL_BASE_TAGS: frozenset[str] = frozenset({"project", "policy"})

Decision = Literal["auto", "pending"]


@dataclass(frozen=True)
class ConflictDecision:
    """1 グループの解決判断結果"""

    decision: Decision
    reason: str
    winner: SemanticFact
    losers: tuple[SemanticFact, ...]


class SemanticConflictResolver:
    """1 ``SemanticFactStore`` 内のファクト競合を検出し解消するリゾルバ。

    インスタンスは 1 sleep-time サイクル内で使い捨てる前提とし、内部状態は
    持たない。``resolve()`` を呼び出すと検出 → 判定 → 適用 → ファイル記録
    までを 1 パスで行う。
    """

    def __init__(
        self,
        store: SemanticFactStore,
        config: dict | None = None,
        *,
        now_provider=None,
    ) -> None:
        self.store = store
        cfg = ((config or {}).get("memory", {}) or {}).get("conflict", {}) or {}
        self.default_mode: str = cfg.get("default_mode", "auto")
        self.confirm_window_sec: float = (
            float(cfg.get("confirm_window_hours", 1.0)) * 3600.0
        )
        self.project_tag_always_manual: bool = bool(
            cfg.get("project_tag_always_manual", True),
        )
        self.auto_for_evolved_policies: bool = bool(
            cfg.get("auto_for_evolved_policies", True),
        )
        # pending 競合の TTL 自動解消 (秒)。0 で無効。
        self.pending_ttl_sec: float = (
            float(cfg.get("pending_auto_resolve_days", 3.0)) * 86400.0
        )
        self.attribute_similarity_threshold: float = float(
            cfg.get(
                "attribute_similarity_threshold",
                DEFAULT_ATTRIBUTE_SIMILARITY_THRESHOLD,
            ),
        )
        self._now_provider = now_provider or time.time

    # ── public API ────────────────────────────────────────────────────

    def resolve(self) -> dict[str, int]:
        """ストア内の競合を検出 → 判定 → 適用する。

        先頭で TTL pre-pass (``_resolve_expired_pending``) を実行し、滞留した
        pending を keep_new で自動解消してから新規競合を検出する。pre-pass で
        解消した winner は singleton 化するため、同サイクルの ``_detect_groups``
        で別 active ファクトとの新競合があれば正しく拾える。

        Returns:
            ``{"detected", "auto_resolved", "pending", "groups",
            "ttl_auto_resolved"}`` のサマリ。
        """
        result = {
            "detected": 0,
            "auto_resolved": 0,
            "pending": 0,
            "groups": 0,
            "ttl_auto_resolved": 0,
        }
        self._resolve_expired_pending(result)
        groups = self._detect_groups()
        for facts in groups:
            result["groups"] += 1
            result["detected"] += len(facts)
            decision = self._decide(facts)
            self._apply(decision, result)
        if result["groups"]:
            logger.info(
                "SemMem conflict resolution: groups=%d auto=%d pending=%d "
                "(scope=%s)",
                result["groups"],
                result["auto_resolved"],
                result["pending"],
                self._infer_scope(),
            )
        return result

    # ── TTL pre-pass ──────────────────────────────────────────────────

    def _resolve_expired_pending(self, result: dict[str, int]) -> None:
        """TTL 超過の pending 競合グループを keep_new で自動解消する (pre-pass)。

        グループ内最新ファクトの ``created_at`` から ``pending_ttl_sec`` 経過した
        グループが対象。``default_mode=manual`` でも有効で、無効化は
        ``pending_auto_resolve_days=0`` を指定する。``result`` の
        ``ttl_auto_resolved`` を解消グループ数だけ加算する。

        **pinned / project / policy も対象に含める。** 以前はこの 3 種を
        「チャット回答でのみ解決」として除外していたが、**その チャット回答の
        判定経路は 2026-08-14 に撤去された** (docs/f_02 §5.3)。撤去後も除外だけが
        残っていたため、この 3 種は解決経路がゼロになり永久に pending へ滞留する。
        滞留した pending は Tier 予算の外で毎ターン最大 400 トークン注入され続け、
        しかも関連度ゲートが掛からない。

        pin を対象にしてよい理由: TTL は ``keep_new`` で、**同じスロットの古い世代を
        supersede するだけ**。最新の値は active のまま残り、supersede は削除では
        ないので内容は失われない。pin は「優先度」の指定であって「不変」の宣言では
        ない (``MemoryInjector`` 側の pin の扱いと同じ立場) うえ、pin は
        「覚えておいてください」等の語で **自動的に** 付くため、除外したままだと
        普通の会話で恒久 pending が量産される。

        自動解決そのものを望まない構成は ``pending_auto_resolve_days: 0`` で
        止められる (従来どおり)。
        """
        if self.pending_ttl_sec <= 0:
            return
        # conflict_review はモジュールトップで本モジュールの定数を import して
        # いるため、循環回避でここで遅延 import する (同一 pillar EvorefMem 内)。
        from backend.free.memory.pipeline.conflict_review import (
            AlreadyResolvedError,
            apply_resolution,
            collect_pending_groups,
        )
        scope = self._infer_scope()
        now = float(self._now_provider())
        for group in collect_pending_groups(self.store, scope):
            if now - group.newest.created_at < self.pending_ttl_sec:
                continue
            loser_ids = [f.id for f in group.facts if f.id != group.newest.id]
            try:
                apply_resolution(
                    self.store,
                    scope=scope,
                    action="keep_new",
                    winner_id=group.newest.id,
                    loser_ids=loser_ids,
                    decision_source="ttl_auto",
                )
            except AlreadyResolvedError:
                # 競合がチャット回答で既に解決済み (失敗ではなく稀な競合)。
                logger.debug(
                    "TTL pending group %s.%s already resolved (skip)",
                    group.subject, group.predicate,
                )
                continue
            except Exception as exc:
                # sleep-time は best-effort。apply_resolution の I/O 失敗
                # (OSError 等) や想定外で 1 グループが落ちても、残りの解消と
                # sleep サイクル全体を止めない。
                logger.warning(
                    "TTL auto-resolve failed for %s %s: %r",
                    group.subject, group.predicate, exc,
                )
                continue
            result["ttl_auto_resolved"] += 1
        if result["ttl_auto_resolved"]:
            logger.info(
                "SemMem pending TTL auto-resolve: %d group(s) (scope=%s)",
                result["ttl_auto_resolved"], scope,
            )

    # ── 検出 ─────────────────────────────────────────────────────────

    def _detect_groups(self) -> list[list[SemanticFact]]:
        """同 ``(subject, predicate)`` で異なる ``object`` を持つ active ファクト群を抽出する。

        ``review_status == "pending"`` のファクトは前サイクルで既に pending
        と判定済みなので再処理しない (二重通知防止)。

        スロットは属性単位に分かれている **はず** だが、トリガ辞書に無い属性は
        ``mem.personal.user`` へフォールバックするため、無関係な事実が同居する。
        そのまま束ねると偽の競合になるので、類似度で塊に割ってから競合とみなす
        (:func:`split_by_attribute_similarity`)。
        """
        active = self.store.all_facts(include_superseded=False)
        buckets: dict[tuple[str, str], list[SemanticFact]] = {}
        for f in active:
            if f.review_status == "pending":
                continue
            buckets.setdefault((f.subject, f.predicate), []).append(f)
        groups: list[list[SemanticFact]] = []
        for facts in buckets.values():
            if len(facts) < 2:
                continue
            facts.sort(key=lambda x: x.created_at)
            for cluster in split_by_attribute_similarity(
                facts, self.attribute_similarity_threshold,
            ):
                if len(cluster) < 2:
                    continue
                if len(distinct_conflict_objects(cluster)) < 2:
                    continue
                groups.append(cluster)
        return groups

    # ── 判定 ─────────────────────────────────────────────────────────

    def _decide(self, facts: list[SemanticFact]) -> ConflictDecision:
        """1 グループの自動/手動判定を返す。``facts`` は created_at 昇順。"""
        winner = facts[-1]
        losers = tuple(facts[:-1])
        types_in_group = {f.type for f in facts}

        # ユーザーが明示的に訂正したターン由来の値は、確認を挟まず即採用する。
        #
        # ``_is_borderline`` は「同 session_id」または「confirm_window_hours 以内」を
        # 微妙ケースとして pending にするが、**会話中の訂正はその両方を必ず満たす**。
        # つまり「違います、コーヒーです」のような **いちばん確度の高い訂正が
        # いちばん自動解決されない** 経路に入っていた。しかも解決経路は TTL
        # (既定 3 日) しか残っていない。
        #
        # pinned より優先する: pin は「覚えておいてください」等の語で自動的に
        # 付くので、ユーザーが後から明示的に否定した値を pin が守る理由は無い。
        # 一方 ``default_mode: manual`` と project / policy タグは尊重する
        # (前者は運用者の明示指定、後者はチャットの訂正が及ぶ対象ではない)。
        if (
            winner.from_correction
            and self.default_mode != "manual"
            and not (types_in_group & _MANUAL_BASE_TAGS)
        ):
            return ConflictDecision(
                "auto", "user_correction", winner, losers,
            )

        if any(f.pinned for f in facts):
            return ConflictDecision(
                "pending", "pinned_present", winner, losers,
            )

        if self.default_mode == "manual":
            return ConflictDecision(
                "pending", "default_manual", winner, losers,
            )

        manual_tag_hit = types_in_group & _MANUAL_BASE_TAGS
        if manual_tag_hit and self.project_tag_always_manual:
            if (
                self.auto_for_evolved_policies
                and winner.type == "policy"
                and winner.auto_evolved
                and all(f.type == "policy" for f in facts)
            ):
                return ConflictDecision(
                    "auto", "auto_evolved_policy", winner, losers,
                )
            return ConflictDecision(
                "pending",
                "project_tag_manual" if "project" in manual_tag_hit
                else "policy_tag_manual",
                winner, losers,
            )

        if self._is_borderline(winner, losers):
            return ConflictDecision(
                "pending", "confirm_window", winner, losers,
            )

        return ConflictDecision("auto", "newest_wins", winner, losers)

    def _is_borderline(
        self,
        winner: SemanticFact,
        losers: tuple[SemanticFact, ...],
    ) -> bool:
        """微妙ケース (同 source または ``confirm_window_hours`` 以内) 判定。

        - 同 source: provenance の ``session_id`` または ``note_id`` が重複
        - 時間: winner と任意 loser の created_at 差が窓以下
        """
        win_sessions, win_notes = _provenance_keys(winner)
        for loser in losers:
            if abs(winner.created_at - loser.created_at) <= self.confirm_window_sec:
                return True
            l_sessions, l_notes = _provenance_keys(loser)
            if win_sessions & l_sessions or win_notes & l_notes:
                return True
        return False

    # ── 適用 ─────────────────────────────────────────────────────────

    def _apply(self, decision: ConflictDecision, result: dict[str, int]) -> None:
        if decision.decision == "auto":
            self._apply_auto(decision)
            result["auto_resolved"] += len(decision.losers)
        else:
            self._apply_pending(decision)
            result["pending"] += 1 + len(decision.losers)

    def _apply_auto(self, decision: ConflictDecision) -> None:
        winner = decision.winner
        for loser in decision.losers:
            try:
                self.store.supersede(loser.id, winner.id)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "supersede failed for %s -> %s: %s",
                    loser.id, winner.id, exc,
                )
                continue
        # 解決 mark を winner にも残す
        try:
            self.store.update_fact(
                winner.id,
                review_status="resolved_keep_new",
            )
        except KeyError:
            pass
        self._write_jsonl(CONFLICTS_RESOLVED_FILENAME, decision)

    def _apply_pending(self, decision: ConflictDecision) -> None:
        for fact in (decision.winner, *decision.losers):
            try:
                self.store.update_fact(
                    fact.id,
                    requires_user_review=True,
                    review_status="pending",
                )
            except KeyError:
                continue
        self._write_jsonl(CONFLICTS_PENDING_FILENAME, decision)

    # ── 永続化 ────────────────────────────────────────────────────────

    def _write_jsonl(self, filename: str, decision: ConflictDecision) -> None:
        path = self.store.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": float(self._now_provider()),
            "scope": self._infer_scope(),
            "subject": decision.winner.subject,
            "predicate": decision.winner.predicate,
            "type": decision.winner.type,
            "winner_id": decision.winner.id,
            "loser_ids": [f.id for f in decision.losers],
            "decision": decision.decision,
            "reason": decision.reason,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _infer_scope(self) -> str:
        """ストアの ``root_dir`` 構造から scope 文字列を推定する。

        ``<semantic_root>/global`` or ``<semantic_root>/projects/<id>``
        を仮定する。それ以外はディレクトリ名をそのまま返す。
        """
        root = self.store.root_dir
        if root.name == "global":
            return "global"
        if root.parent.name == "projects":
            return f"project:{root.name}"
        return root.name


def _provenance_keys(
    fact: SemanticFact,
) -> tuple[set[str], set[str]]:
    """``(session_ids, note_ids)`` のタプルを返す。``None`` は除外する。"""
    sessions: set[str] = set()
    notes: set[str] = set()
    for p in fact.provenances:
        if p.session_id:
            sessions.add(p.session_id)
        if p.note_id:
            notes.add(p.note_id)
    return sessions, notes


def resolve_semmem_conflicts(
    stores: Iterable[SemanticFactStore],
    config: dict | None = None,
) -> dict[str, int]:
    """複数ストアに対して :class:`SemanticConflictResolver` を順次適用する。

    sleep-time Step 6 のヘルパ。``stores`` は ``[global_store, project_store]``
    を想定するが任意個に対応する。集計サマリを返す。
    """
    total = {
        "detected": 0,
        "auto_resolved": 0,
        "pending": 0,
        "groups": 0,
        "ttl_auto_resolved": 0,
    }
    for store in stores:
        if store is None:
            continue
        sub = SemanticConflictResolver(store, config).resolve()
        for k, v in sub.items():
            total[k] = total.get(k, 0) + v
    return total
