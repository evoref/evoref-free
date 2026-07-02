"""Step 7.5: MDP trace → episodic LTM ingestion

``sleep_update.SleepTimeWorker._step7_5_ingest_mdp_traces`` /
``_ensure_mdp_ingester`` として実装された MDP トレース取り込みロジックを
独立 module に切り出したもの。

処理概要:

- 現 STM の private ノートに紐づく ``trace_id`` を収集し、該当エピソードの
  LTM 昇格をスキップする
- 取得した ``EpisodeRecord`` 1 件につき ``MemoryNote`` を 1 つ生成し、
  埋め込み計算後に :meth:`LongTermMemory.absorb_from_short_term` で投入する。
- embedder が未設定の場合や ingest 中の例外は warning ログにとどめ、
  sleep-time 全体は止めない。

本 module は EvorefMem pillar 内部扱い。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.stores.long_term import LongTermMemory
    from backend.free.memory.notes.mdp_ingester import MDPIngester
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.rag.embedding_backend import EmbeddingBackend

logger = get_logger("memory.sleep.mdp_ingest")


def ensure_mdp_ingester(
    *,
    cached: "MDPIngester | None",
    agent_trace_dir: Path | None,
) -> "MDPIngester | None":
    """``MDPIngester`` の lazy 初期化。

    ``cached`` が既に生成済ならそのまま返す。そうでなければ
    ``agent_trace_dir`` が存在する場合にのみ新規生成する。
    ``path_resolver`` から ``memory_dir`` を解決し、state ファイルを
    ``local/memory/mdp_ingest_state.json`` に配置する。resolver が使えない
    場合は ``agent_trace_dir`` 配下にフォールバック。

    Args:
        cached: 既存の :class:`MDPIngester` インスタンス (初回は ``None``)。
        agent_trace_dir: ``agent_trace*.jsonl`` を収集するディレクトリ。

    Returns:
        生成または既存の :class:`MDPIngester`。ディレクトリ未設定 / 存在しない
        場合は ``None`` (graceful no-op)。
    """
    if cached is not None:
        return cached
    if agent_trace_dir is None:
        return None
    log_dir = Path(agent_trace_dir)
    if not log_dir.exists():
        return None
    try:
        from backend.config import get_path_resolver
        resolver = get_path_resolver()
        mem_dir = resolver.resolve_local("memory_dir")
        state_path = Path(mem_dir) / "mdp_ingest_state.json"
    except Exception as exc:
        logger.warning("MDP ingester: failed to resolve memory_dir: %s", exc)
        state_path = log_dir / "mdp_ingest_state.json"
    from backend.free.memory.notes.mdp_ingester import MDPIngester
    return MDPIngester(log_dir, state_path)


async def ingest_mdp_traces(
    short_term: "ShortTermMemory",
    long_term: "LongTermMemory",
    embedder: "EmbeddingBackend | None",
    *,
    config: dict | None,
    agent_trace_dir: Path | None,
    current_project_id: str | None,
    cached_ingester: "MDPIngester | None",
) -> tuple[int, "MDPIngester | None"]:
    """Step 7.5 本体 — ``agent_trace*.jsonl`` を episodic LTM に取り込む。

    Guards:

    - ``memory.facts.ingest_mdp_trace_to_ltm = False`` → no-op (``0``)
    - ``ensure_mdp_ingester`` が ``None`` を返した → no-op (``0``)
    - ``embedder is None`` → no-op (``0``)

    Args:
        short_term: 現ノート収集元の :class:`ShortTermMemory`。
        long_term: ingest 先の :class:`LongTermMemory`。
        embedder: ノート埋め込み生成用の
            :class:`~backend.free.rag.embedding_backend.EmbeddingBackend`。
        config: ``memory.facts.ingest_mdp_trace_to_ltm`` を含む設定 dict。
        agent_trace_dir: ``agent_trace*.jsonl`` を収集するディレクトリ。
        current_project_id: ingest 時に note へ紐付けるプロジェクト ID。
        cached_ingester: 前回以降に生成済の :class:`MDPIngester` があれば渡す
            (プロセス内で episode の二重抽出を防ぐため)。

    Returns:
        ``(ingested_count, ingester)`` のペア。第二要素は caller 側で
        キャッシュして次回以降に再利用する。
    """
    cfg_facts = (config or {}).get("memory", {}).get("facts", {}) or {}
    if not bool(cfg_facts.get("ingest_mdp_trace_to_ltm", True)):
        logger.debug("Step 7.5: MDP ingest disabled by config")
        return 0, cached_ingester
    ingester = ensure_mdp_ingester(
        cached=cached_ingester, agent_trace_dir=agent_trace_dir,
    )
    if ingester is None:
        return 0, cached_ingester
    if embedder is None:
        logger.debug("Step 7.5: no embedder, skipping MDP ingest")
        return 0, ingester

    private_trace_ids = {
        n.trace_id for n in short_term.notes.values()
        if getattr(n, "private", False) and getattr(n, "trace_id", None)
    }
    episodes = ingester.collect_episodes(private_trace_ids=private_trace_ids)
    if not episodes:
        return 0, ingester

    notes = [
        ingester.to_memory_note(ep, project_id=current_project_id)
        for ep in episodes
    ]
    try:
        embeddings = await embedder.embed(
            [n.content for n in notes], is_query=False,
        )
    except Exception as exc:
        logger.warning("Step 7.5: MDP ingest embedding failed: %s", exc)
        return 0, ingester

    ingested = 0
    for note, emb in zip(notes, embeddings):
        try:
            note.embedding = (
                emb.astype(np.float32)
                if hasattr(emb, "astype")
                else np.asarray(emb, dtype=np.float32)
            )
            chunk_id = long_term.absorb_from_short_term(note)
            if chunk_id is not None:
                ingested += 1
        except Exception as exc:
            logger.warning(
                "Step 7.5: failed to absorb mdp note %s: %s", note.id, exc,
            )
    if ingested:
        logger.info("Step 7.5: ingested %d MDP episodes into LTM", ingested)
    return ingested, ingester


__all__ = ["ensure_mdp_ingester", "ingest_mdp_traces"]
