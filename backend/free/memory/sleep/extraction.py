"""Step 8: SemanticFact Extractor orchestration

``sleep_update.SleepTimeWorker._step8_extract_facts`` として
実装された Extractor 起動ロジックを独立 module に切り出したもの。

処理は 3 段階で構成される:

1. :class:`~backend.free.memory.extractors.chat.ChatExtractor`
   → ``global`` スコープに ``personal_fact`` / ``world_fact`` / ``preference`` /
   ``emotion`` / ``opinion`` を追記
2. :class:`~backend.free.memory.extractors.create.CreateExtractor`
   → ``project:<id>`` スコープに ``project`` / ``decision`` / ``commitment`` /
   ``create_task`` / ``create`` を追記
3. :class:`~backend.free.memory.extractors.mdp_trace.MDPTraceExtractor`
   → ``project:<id>`` スコープに ``failure_pattern`` / ``decision`` を追記
   (config で disable 可)

本 module は EvorefMem pillar 内部扱いのため SemanticFactStore を直接参照する。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.free.memory.notes.note_builder import is_single_valued_subject
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.extractors import (
        ExtractionResult,
        MDPTraceExtractor,
    )
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.stores.short_term import MemoryNote
    from backend.free.memory.notes.subject_canonicalizer import SubjectCanonicalizer

logger = get_logger("memory.sleep.extraction")


def persist_facts(
    store: "SemanticFactStore",
    result: "ExtractionResult",
    label: str,
) -> int:
    """``ExtractionResult`` のファクトを ``SemanticFactStore`` に書き込む。

    重複 ID 衝突 (極めてまれ) や書き込み失敗は warning ログにとどめ、
    sleep-time 全体は止めない。

    Args:
        store: 書き込み先ストア。
        result: ``BaseExtractor.extract`` の戻り値。
        label: ログ用ラベル (``"chat"`` / ``"create"`` / ``"mdp_trace"`` 等)。

    Returns:
        実際に書き込まれたファクト数。
    """
    written = 0
    persisted: list = []
    for fact in result.facts:
        try:
            store.add_fact(fact)
            written += 1
            persisted.append(fact)
        except Exception as exc:
            logger.warning(
                "Step 8 [%s]: failed to add fact %s: %s",
                label, fact.id, exc,
            )
    _supersede_corrected_slots(store, persisted, label)
    if written:
        logger.debug("Step 8 [%s]: persisted %d facts", label, written)
    return written


def _supersede_corrected_slots(
    store: "SemanticFactStore", persisted: list, label: str,
) -> int:
    """訂正ファクトを書いたら、同じスロットの旧世代を supersede する。

    ``SemanticFactStore.supersede`` は既に存在するが、**抽出経路からは
    一度も呼ばれていなかった**。呼んでいたのはセッション要約の昇格
    (``sleep.promotion``)、競合解決 (``conflict_review`` /
    ``SemanticConflictResolver``)、learn / loop の書き戻しだけで、
    チャット由来の属性ファクトは新旧が live のまま積み上がる。

    実データ (2026-08-27 ライブ監査、``semantic/global/facts.jsonl`` 12 件):
    全件が ``supersedes: []`` / ``superseded_by: None``。``from_correction``
    は 4 件立っているのに旧世代が 1 つも無効化されていない。結果、

    - ``mem.personal.occupation`` に「データベース管理者」と
      「ネットワークエンジニア」が同時に live
    - ``mem.personal.name`` に 3 世代 (テスト太郎 / 御堂 陽介 / 田中) が live

    となり、新規セッションでの想起が **同じ問いに毎回ちがう値** を返した
    (名前は 1 回目「御堂 陽介」/ 2 回目「田中」)。注入側の
    ``_collapse_to_current_values`` は 1 スロット 1 値へ畳むが、
    **畳む前に検索へ乗るのは全世代**で、埋め込み検索の上位に旧値が来れば
    その時点で負ける。ストア側で世代を閉じるのが本筋。

    畳む条件は ``from_correction`` が立っているか、**スロットが単値と宣言されて
    いる** こと。訂正でない再言明まで無条件に supersede すると、``pet`` のように
    1 人が複数値を持ちうるスロットで正当な値を落とすため、一括では畳まない。

    単値スロット (``fact_attributes.yaml`` の ``single_valued: true``) を条件に
    加えたのは、**訂正ではない更新** が旧値を live のまま残していたから。
    実データ (2026-08-29 ライブ監査、``semantic/global/facts.jsonl``):

    - 「先月、横浜から札幌に引っ越しました」→ ``from_correction`` が立たず、
      ``mem.personal.location`` に 横浜 / 札幌 / 名古屋 が **3 つとも live**
    - その結果、次セッションの想起が旧値を返した
      (T28: 「39歳, **横浜市**, **ソフトウェアエンジニア**」/ T29: 更新前の出張日程)
    - 自己検査も「古い情報は含まれていません」と旧値を最新だと保証した

    ``single_valued`` を宣言していないスロットの挙動は一切変わらない
    (既定 ``False``)。

    Returns:
        supersede した旧ファクト数。
    """
    superseded = 0
    #: 同一バッチで同じスロットへ複数の値が書かれたとき、**勝者を 1 つに決めて
    #: から** 畳む。素朴に「新規ファクトごとに他の live を supersede する」と、
    #: 同じ subject の 2 件が **互いを supersede** して live が 0 件になる。
    #:
    #: 実データ (2026-08-29 クリーンストア検証): 「今は千葉に住んでいます」と
    #: 「先週、千葉から神戸に引っ越しました」が同じ Full で抽出され、
    #: ``mem.personal.location`` の **2 件とも SUPERSEDED** になった
    #: (``occupation`` も同様)。想起は「確認できていません」に落ちる。
    #: 勝者は ``persisted`` の **最後** に来たもの (抽出順 = 発話順)。
    winners: dict[tuple[str, str], object] = {}
    for fact in persisted:
        if not is_single_valued_subject(getattr(fact, "subject", "") or ""):
            continue
        winners[(fact.subject, fact.predicate)] = fact
    for fact in persisted:
        if not (
            getattr(fact, "from_correction", False)
            or is_single_valued_subject(getattr(fact, "subject", "") or "")
        ):
            continue
        # 単値スロットは勝者だけが畳む側に回る (敗者は何も supersede しない)。
        winner = winners.get((fact.subject, fact.predicate))
        if winner is not None and winner is not fact:
            continue
        try:
            siblings = store.search_by_subject(
                fact.subject, include_superseded=False,
            )
        except Exception as exc:
            logger.warning(
                "Step 8 [%s]: failed to list slot %s: %s",
                label, fact.subject, exc,
            )
            continue
        for old in siblings:
            if old.id == fact.id or old.predicate != fact.predicate:
                continue
            if old.superseded_by:
                continue
            try:
                store.supersede(old.id, fact.id)
                superseded += 1
            except (KeyError, ValueError) as exc:
                # 昇格側と同じ扱い — 書き込み自体は成立しているので警告に留め、
                # 残った旧世代は競合解決 / TTL に委ねる。
                logger.warning(
                    "Step 8 [%s]: failed to supersede %s -> %s: %s",
                    label, old.id, fact.id, exc,
                )
    if superseded:
        logger.info(
            "Step 8 [%s]: superseded %d stale slot value(s) by corrections",
            label, superseded,
        )
    return superseded


#: ``mem.<kind>.<attr>`` の ``kind`` → ``FactType``。値アンカー用の逆引き。
_FACT_TYPE_BY_KIND: dict[str, str] = {
    "personal": "personal_fact",
    "world": "world_fact",
    "preference": "preference",
    "emotion": "emotion",
    "opinion": "opinion",
}


def collect_live_attribute_values(
    store: "SemanticFactStore",
) -> dict[tuple[str, str], tuple[str, ...]]:
    """live ファクトから ``{(fact_type, 属性スロット): (現在値, ...)}`` を組む。

    属性語を落とした訂正 (「さっき名古屋と言いましたが、正しくは横浜です。」)
    の宛先を決めるための材料。詳細は
    :func:`~backend.free.memory.extractors.chat.
    resolve_value_anchored_attributes`。

    ``user`` スロットは除く — 属性が解決できなかったファクトの受け皿なので、
    そこを名指しても宛先を絞れない。
    """
    values: dict[tuple[str, str], list[str]] = {}
    try:
        facts = store.all_facts(include_superseded=False)
    except Exception as exc:
        logger.warning("Step 8: failed to read live facts: %s", exc)
        return {}
    for fact in facts:
        parts = (fact.subject or "").split(".")
        if len(parts) != 3 or parts[0] != "mem":
            continue
        fact_type = _FACT_TYPE_BY_KIND.get(parts[1])
        attr = parts[2]
        if not fact_type or attr == "user":
            continue
        text = (fact.text or "").strip()
        if text:
            values.setdefault((fact_type, attr), []).append(text)
    return {key: tuple(vals) for key, vals in values.items()}


def _drop_facts_with_existing_subject(
    store: "SemanticFactStore",
    result: "ExtractionResult",
) -> int:
    """既に同一 subject の active fact が存在する ``decision`` 候補を除外する。

    ``MDPTraceExtractor`` は ``decision`` を ``mem.decision.<episode_id>``
    (エピソード毎に一意) で生成する。プロセス再起動で抽出器の in-memory
    ``_processed_episode_ids`` が失われると同一エピソードが再抽出されるが、
    既存 subject を弾くことで新しい ``fact_id`` での重複追記を防ぐ (store が
    dedup の永続状態を兼ねる)。``chat`` / ``create`` 抽出器は同一 subject の再
    アサートで内容を更新する設計のため、この dedup は MDP 経路にのみ適用する。

    ``failure_pattern`` (loop 所有) は呼出側で事前に分離され
    :func:`_persist_failure_patterns_via_view` が ``LoopFactView`` 経由で
    signature 単位の in-place occurrences 加算として書くため、本 dedup には
    渡らない (別エピソードでの同一 signature 再発を弾くと再発頻度が失われる)。
    よって本関数の対象は ``decision`` のみ。

    Returns:
        除外した件数。
    """
    if not result.facts:
        return 0
    kept = []
    dropped = 0
    for fact in result.facts:
        # decision のみ subject 一意性 dedup。failure_pattern は Step 13 に委ねる。
        if fact.type == "decision" and store.search_by_subject(
            fact.subject, include_superseded=False,
        ):
            dropped += 1
            continue
        kept.append(fact)
    if dropped:
        result.facts = kept
        logger.debug(
            "Step 8 [mdp_trace]: skipped %d duplicate-subject facts", dropped,
        )
    return dropped


def _persist_failure_patterns_via_view(
    project_store: "SemanticFactStore",
    failure_facts: list,
    project_id: str,
) -> int:
    """MDP 由来の ``failure_pattern`` を ``LoopFactView`` 経由で書き込む。

    ``failure_pattern`` は loop 所有 FactType のため、mem pillar が
    ``store.add_fact`` で直書きすると ownership enforcement を素通りする。
    :meth:`LoopFactView.write_failure_pattern` 経由にすることで owner 検証を
    通し、同一 signature を **in-place で occurrences 加算** する
    (failure_consolidator が LoopFactView を使うのと同じ前例)。MDP 抽出器が
    組み立てた JSON object (``error_type`` / ``normalized_file_path`` /
    ``last_actions`` / ``outcomes_history``) を分解して低レベル API に渡す。

    Returns:
        書き込んだ failure_pattern 数。
    """
    if not failure_facts:
        return 0
    from backend.free.memory.views.loop import LoopFactView

    view = LoopFactView(stores=[project_store], writeback_store=project_store)
    written = 0
    for f in failure_facts:
        signature = f.failure_signature or ""
        if not signature:
            continue
        try:
            payload = json.loads(f.object)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        outcomes = payload.get("outcomes_history") or []
        try:
            view.write_failure_pattern(
                project_id=project_id,
                signature=signature,
                error_type=str(payload.get("error_type", "")),
                normalized_file_path=str(payload.get("normalized_file_path", "")),
                last_actions=list(payload.get("last_actions") or []),
                outcome_label=str(outcomes[0]) if outcomes else None,
                trace_id=f.trace_id,
            )
            written += 1
        except Exception as exc:
            logger.warning(
                "Step 8 [mdp_trace]: failure_pattern write_via_view failed "
                "(sig=%s): %s", signature, exc,
            )
    if written:
        logger.debug(
            "Step 8 [mdp_trace]: persisted %d failure_pattern via LoopFactView",
            written,
        )
    return written


def extract_semantic_facts(
    notes: list["MemoryNote"],
    *,
    config: dict | None,
    store_provider: Callable[[str], "SemanticFactStore | None"] | None,
    current_project_id: str | None,
    agent_trace_dir: Path | None,
    subject_canonicalizer: "SubjectCanonicalizer | None",
    mdp_trace_extractor: "MDPTraceExtractor | None" = None,
    mdp_trace_extractor_factory: Callable[[], "MDPTraceExtractor"] | None = None,
) -> tuple[int, "MDPTraceExtractor | None"]:
    """Step 8: ChatExtractor / CreateExtractor / MDPTraceExtractor を順次実行する。

    Guards:

    - ``memory.facts.enable_extraction = False`` → no-op (``0``)
    - ``store_provider`` が ``None`` → no-op (``0``)
    - ``global`` store 取得失敗 → Chat skip
    - ``current_project_id`` 未設定 / project store 取得失敗 → Create / MDP skip

    Args:
        notes: 対象ノート群 (通常は ``ShortTermMemory.notes.values()`` のリスト)。
        config: ``memory.facts`` 配下の設定を含む設定 dict。
        store_provider: ``scope`` → ``SemanticFactStore`` を返すコールバック。
        current_project_id: 現在のプロジェクト ID。
        agent_trace_dir: ``agent_trace*.jsonl`` のディレクトリ (MDP 抽出用)。
        subject_canonicalizer: subject の正規化器
        mdp_trace_extractor: 既存の MDPTraceExtractor インスタンス
            (プロセス内で episode の二重抽出を防ぐためワーカー側で保持するもの)。
        mdp_trace_extractor_factory: ``mdp_trace_extractor`` が ``None``
            の場合に新規生成するファクトリ。未指定時は
            :class:`MDPTraceExtractor` を直接 import して生成する。

    Returns:
        ``(total_extracted, mdp_trace_extractor)`` のペア。
        第二要素は caller にキャッシュして再利用させるためのもの (初回実行で
        生成したインスタンスを返す; 2 回目以降は同じインスタンスが戻る)。
    """
    cfg_facts = (config or {}).get("memory", {}).get("facts", {}) or {}
    if not cfg_facts.get("enable_extraction", True):
        logger.debug("Step 8: extraction disabled by config")
        return 0, mdp_trace_extractor
    if store_provider is None:
        logger.debug("Step 8: no semantic store provider, skipping")
        return 0, mdp_trace_extractor

    from backend.free.memory.extractors import (
        ChatExtractor,
        CreateExtractor,
        ExtractionContext,
        MDPTraceExtractor,
    )

    max_per_session_cfg = cfg_facts.get("extraction_max_per_session", {}) or {}
    ctx = ExtractionContext(
        project_id=current_project_id,
        agent_trace_dir=agent_trace_dir,
        max_per_session={
            "chat": int(max_per_session_cfg.get("chat", 10)),
            "create": int(max_per_session_cfg.get("create", 5)),
        },
        max_pinned_per_session=int(
            cfg_facts.get("extraction_max_pinned_per_session", -1),
        ),
        canonicalizer=subject_canonicalizer,
    )

    total_extracted = 0
    try:
        global_store = store_provider("global")
    except Exception as exc:
        logger.warning("Step 8: failed to obtain global store: %s", exc)
        global_store = None

    # ── 1. ChatExtractor → global ──
    if global_store is not None:
        # 属性語を落とした訂正の宛先を決めるため、既存スロットの現在値を渡す
        # (chat.resolve_value_anchored_attributes の説明を参照)。
        ctx.live_attribute_values = collect_live_attribute_values(global_store)
        chat_result = ChatExtractor().extract(notes, ctx)
        total_extracted += persist_facts(global_store, chat_result, "chat")

    # ── 2. CreateExtractor → project ──
    project_store = None
    if current_project_id:
        try:
            project_store = store_provider(f"project:{current_project_id}")
        except Exception as exc:
            logger.warning("Step 8: failed to obtain project store: %s", exc)

    if project_store is not None:
        create_result = CreateExtractor().extract(notes, ctx)
        total_extracted += persist_facts(project_store, create_result, "create")

    # ── 3. MDPTraceExtractor → project ──
    if (
        project_store is not None
        and bool(cfg_facts.get("extract_from_mdp_trace", True))
    ):
        if mdp_trace_extractor is None:
            if mdp_trace_extractor_factory is not None:
                mdp_trace_extractor = mdp_trace_extractor_factory()
            else:
                mdp_trace_extractor = MDPTraceExtractor()
        mdp_result = mdp_trace_extractor.extract(notes, ctx)
        # failure_pattern は loop 所有なので LoopFactView 経由で書く (ownership
        # 準拠 + signature 単位の in-place occurrences 加算)。decision は mem の
        # store 直書き経路 (subject 一意 dedup) のまま。
        failure_facts = [f for f in mdp_result.facts if f.type == "failure_pattern"]
        mdp_result.facts = [
            f for f in mdp_result.facts if f.type != "failure_pattern"
        ]
        _drop_facts_with_existing_subject(project_store, mdp_result)
        total_extracted += persist_facts(
            project_store, mdp_result, "mdp_trace",
        )
        total_extracted += _persist_failure_patterns_via_view(
            project_store, failure_facts, current_project_id,
        )

    if total_extracted:
        logger.info("Step 8: extracted %d facts", total_extracted)
    else:
        logger.debug("Step 8: no facts extracted")
    return total_extracted, mdp_trace_extractor


__all__ = [
    "collect_live_attribute_values",
    "extract_semantic_facts",
    "persist_facts",
]
