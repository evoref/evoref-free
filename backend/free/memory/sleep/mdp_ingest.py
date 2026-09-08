"""Step 7.5: MDP trace → エピソード記憶への取り込み

``agent_trace*.jsonl`` の 1 エピソードを 1 ノートにして
:class:`~backend.free.memory.episodic.store.EpisodicStore` の ``long`` tier へ
``put`` する。エピソードは会話ターンではなく **完了した実行の記録** なので、
``short`` (会話由来ノートの working set) には入れない — 抽出器 / キュレーター
の入力にはならず、検索でだけ効く。

- private セッションのターンに紐づく ``trace_id`` のエピソードは昇格しない
  (``private_trace_ids``。以前は STM の private ノートから集めていたが、
  private ターンは会話履歴にもストアにも残らなくなったので、呼出側が
  ``WorkingMemory`` から渡す)
- 埋め込みは snapshot 生成時にまとめて作られるので、ここでは呼ばない
- ingest 中の例外は warning にとどめ、sleep-time 全体は止めない
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.episodic.store import EpisodicStore
    from backend.free.memory.notes.mdp_ingester import MDPIngester

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


def ingest_mdp_traces(
    episodic: "EpisodicStore",
    *,
    config: dict | None,
    agent_trace_dir: Path | None,
    current_project_id: str | None,
    cached_ingester: "MDPIngester | None",
    private_trace_ids: Iterable[str] = (),
) -> tuple[int, "MDPIngester | None"]:
    """Step 7.5 本体 — ``agent_trace*.jsonl`` をエピソード記憶へ取り込む。

    Guards:

    - ``memory.facts.ingest_mdp_trace_to_ltm = False`` → no-op (``0``)
    - ``ensure_mdp_ingester`` が ``None`` を返した → no-op (``0``)

    Returns:
        ``(ingested_count, ingester)``。第二要素は caller 側でキャッシュして
        次回以降に再利用する (プロセス内で同じ episode を二度取らないため)。
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

    episodes = ingester.collect_episodes(private_trace_ids=set(private_trace_ids))
    if not episodes:
        return 0, ingester

    ingested = 0
    for episode in episodes:
        try:
            note = ingester.to_memory_note(episode, project_id=current_project_id)
            episodic.put_note(note, tier="long")
            ingested += 1
        except Exception as exc:  # noqa: BLE001 — 1 件の失敗で sleep-time を止めない
            logger.warning("Step 7.5: failed to ingest an mdp episode: %s", exc)
    if ingested:
        logger.info("Step 7.5: ingested %d MDP episode(s) into episodic memory", ingested)
    return ingested, ingester


__all__ = ["ensure_mdp_ingester", "ingest_mdp_traces"]
