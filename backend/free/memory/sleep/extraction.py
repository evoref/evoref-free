"""Step 8: SemanticFact Extractor orchestration

``sleep_update.SleepTimeWorker._step8_extract_facts`` として
実装された Extractor 起動ロジックを独立 module に切り出したもの。

処理は 3 段階で構成される:

1. :class:`~backend.free.memory.extractors.chat.ChatExtractor`
   → ``global`` スコープに ``personal_fact`` / ``world_fact`` / ``preference`` /
   ``emotion`` / ``opinion`` を追記
2. :class:`~backend.free.memory.extractors.coding.CodingExtractor`
   → ``project:<id>`` スコープに ``project`` / ``decision`` / ``commitment`` /
   ``coding_task`` / ``coding`` を追記
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
        label: ログ用ラベル (``"chat"`` / ``"coding"`` / ``"mdp_trace"`` 等)。

    Returns:
        実際に書き込まれたファクト数。
    """
    written = 0
    for fact in result.facts:
        try:
            store.add_fact(fact)
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Step 8 [%s]: failed to add fact %s: %s",
                label, fact.id, exc,
            )
    if written:
        logger.debug("Step 8 [%s]: persisted %d facts", label, written)
    return written


def _drop_facts_with_existing_subject(
    store: "SemanticFactStore",
    result: "ExtractionResult",
) -> int:
    """既に同一 subject の active fact が存在する ``decision`` 候補を除外する。

    ``MDPTraceExtractor`` は ``decision`` を ``mem.decision.<episode_id>``
    (エピソード毎に一意) で生成する。プロセス再起動で抽出器の in-memory
    ``_processed_episode_ids`` が失われると同一エピソードが再抽出されるが、
    既存 subject を弾くことで新しい ``fact_id`` での重複追記を防ぐ (store が
    dedup の永続状態を兼ねる)。``chat`` / ``coding`` 抽出器は同一 subject の再
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
        except Exception as exc:  # noqa: BLE001 — 1 件失敗で sleep を止めない
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
    """Step 8: ChatExtractor / CodingExtractor / MDPTraceExtractor を順次実行する。

    Guards:

    - ``memory.facts.enable_extraction = False`` → no-op (``0``)
    - ``store_provider`` が ``None`` → no-op (``0``)
    - ``global`` store 取得失敗 → Chat skip
    - ``current_project_id`` 未設定 / project store 取得失敗 → Coding / MDP skip

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
        CodingExtractor,
        ExtractionContext,
        MDPTraceExtractor,
    )

    max_per_session_cfg = cfg_facts.get("extraction_max_per_session", {}) or {}
    ctx = ExtractionContext(
        project_id=current_project_id,
        agent_trace_dir=agent_trace_dir,
        max_per_session={
            "chat": int(max_per_session_cfg.get("chat", 10)),
            "coding": int(max_per_session_cfg.get("coding", 5)),
        },
        max_pinned_per_session=int(
            cfg_facts.get("extraction_max_pinned_per_session", -1),
        ),
        canonicalizer=subject_canonicalizer,
    )

    total_extracted = 0
    try:
        global_store = store_provider("global")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Step 8: failed to obtain global store: %s", exc)
        global_store = None

    # ── 1. ChatExtractor → global ──
    if global_store is not None:
        chat_result = ChatExtractor().extract(notes, ctx)
        total_extracted += persist_facts(global_store, chat_result, "chat")

    # ── 2. CodingExtractor → project ──
    project_store = None
    if current_project_id:
        try:
            project_store = store_provider(f"project:{current_project_id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Step 8: failed to obtain project store: %s", exc)

    if project_store is not None:
        coding_result = CodingExtractor().extract(notes, ctx)
        total_extracted += persist_facts(project_store, coding_result, "coding")

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


__all__ = ["extract_semantic_facts", "persist_facts"]
