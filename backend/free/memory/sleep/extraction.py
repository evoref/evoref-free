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
        total_extracted += persist_facts(
            project_store, mdp_result, "mdp_trace",
        )

    if total_extracted:
        logger.info("Step 8: extracted %d facts", total_extracted)
    else:
        logger.debug("Step 8: no facts extracted")
    return total_extracted, mdp_trace_extractor


__all__ = ["extract_semantic_facts", "persist_facts"]
