"""EvorefMem 初期化 + 起動時 SemMem bootstrap

含まれる関数:

- :func:`_init_memory` : ``WorkingMemoryRegistry`` / ``EpisodicStore`` /
  ``SemanticStore`` の初期化。G0 のスキーマ版マーカー・移行器・破壊的な
  自動初期化は持たない (G1 の版は形式ごとの封筒と世代印、c_05 §0.4)。
- :func:`apply_semmem_policy_overrides` : SemMem active policy ファクト
  → ``PolicyInterpreter`` 反映。

純粋な move であり、関数本体・引数・default 値は変更していない。
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

from backend.app_state import AppState
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.episodic.store import EpisodicStore
    from backend.free.memory.semantic.store import SemanticStore
    from backend.free.memory.stores.working import WorkingMemoryRegistry

logger = get_logger("factory.memory_init")


def apply_semmem_policy_overrides(
    state: AppState,
    cfg: dict[str, Any],  # noqa: ARG001
    current_project_id: str | None,
) -> None:
    """SemMem 上の active policy ファクトを PolicyInterpreter に反映する

    メモリシステムと project_id 解決後に呼び出す。``policy_source`` が
    ``yaml`` の場合は何もしない (既存テスト保護)。``hybrid`` / ``semmem``
    の場合は ``[global, project:<id>]`` の順でストアを渡し、プロジェクト
    スコープが後勝ちで上書きする。
    """
    pi = state.policy_interpreter
    if pi is None:
        return
    if pi.policy_source == "yaml":
        return
    try:
        stores = [state.get_semantic_store("global")]
        if current_project_id:
            stores.append(state.get_semantic_store(f"project:{current_project_id}"))
        pi.set_semmem_stores(stores)
        logger.info(
            "PolicyInterpreter SemMem overrides applied (stores=%d, project_id=%s)",
            len(stores), current_project_id,
        )
    except Exception as exc:
        logger.warning(
            "Failed to apply SemMem policy overrides: %s "
            "(falling back to YAML-only values)", exc,
        )


def _init_memory(
    state: AppState, cfg: dict[str, Any], resolver: Any,
) -> tuple["WorkingMemoryRegistry", "EpisodicStore"]:
    """6. メモリシステム初期化

    WM はセッション別 (:class:`WorkingMemoryRegistry`) で、**プロセス内の窓**
    に閉じる (c_16 §4.1)。ノートの永続化は :class:`EpisodicStore`、構造化事実は
    :class:`SemanticStore` の各 1 本で、書き手は sleep-time だけ。
    """
    from backend.free.memory.episodic.store import EpisodicStore
    from backend.free.memory.semantic.store import SemanticStore
    from backend.free.memory.stores.working import WorkingMemoryRegistry
    from backend.free.rag.evidence.config import merge_rag_evidence_config

    memory_dir = resolver.resolve_local("memory_dir")
    memory_dir.mkdir(parents=True, exist_ok=True)

    # EvorefMem トリガ辞書 (pin / fact / classify) の user override 配置先。
    # 同梱 default は ``backend/free/memory/_defaults/triggers/`` 配下。
    triggers_dir = resolver.resolve_local("triggers_dir")
    # note_builder のモジュールレベル default に設定し、以降に構築される
    # ChatNoteBuilder / CreateNoteBuilder (NoteBuilder singleton 経由 + sleep
    # extractors が fresh 構築するインスタンス) が同じ user override を拾うようにする。
    from backend.free.memory.notes.note_builder import set_default_triggers_dir
    set_default_triggers_dir(triggers_dir)
    wm = WorkingMemoryRegistry(cfg)
    # 窓の寿命に合わせて応答パス側のセッション別台帳 (蓄積バッファ等) を畳む。
    # 会話は毎ターン履歴ファイルへ保存済みなので、押し出されたセッションが
    # 戻ってきても ``_ensure_session_restored`` が索引から引き直せる。
    from backend.free.api.chat.chat_recorder import clear_session_data
    wm.on_drop = clear_session_data

    # エピソード記憶 (c_16 §4.1)。実体は ``<memory_dir>/episodic``。
    # 埋め込みバックエンドは EvorefGen 側の構築後に注入される
    # (:func:`attach_episodic_embedder`) — snapshot 生成時にしか使わないので、
    # 起動順の制約にしない。
    evidence_cfg = (cfg.get("memory") or {}).get("evidence") or {}
    retention = evidence_cfg.get("retention")
    # 3 ストア (episodic / semantic / corpus) は **同じ面** を読む。
    # ここで ``ranking`` を落とすと ``memory.evidence.ranking.store_prior`` /
    # ``allow_assistant_origin_injection`` / 半減期が記憶側の順位式に一切
    # 届かない (2026-09-08 監査。corpus だけが merge を通っていた)。
    evidence_face = merge_rag_evidence_config(cfg)
    episodic = EpisodicStore(
        memory_dir,
        rag_config=evidence_face,
        retention=retention if isinstance(retention, dict) else None,
        debug_logger=getattr(state, "debug_logger", None),
    )
    try:
        episodic.load()
    except Exception as e:  # noqa: BLE001 — 記憶が読めなくても起動は続ける
        logger.warning("Episodic store load skipped: %s", e)

    # 構造化事実 (c_16 §4.2)。実体は ``<memory_dir>/semantic``。episodic と
    # 同じく埋め込みバックエンドは後から注入する (:func:`attach_semantic_embedder`)。
    semantic = SemanticStore(
        memory_dir,
        rag_config=evidence_face,
        retention=retention if isinstance(retention, dict) else None,
        know_half_life=evidence_cfg.get("know_half_life_days"),
        debug_logger=getattr(state, "debug_logger", None),
    )
    try:
        semantic.load()
    except Exception as e:  # noqa: BLE001 — 記憶が読めなくても起動は続ける
        logger.warning("Semantic store load skipped: %s", e)
    _freeze_resident_once()

    state.working_memory_registry = wm
    state.episodic_memory = episodic
    state.semantic_memory = semantic
    state._semantic_stores.clear()
    logger.info(
        "Memory system initialized (%d episodic record(s), %d fact(s))",
        len(episodic), len(semantic),
    )
    return wm, episodic


_frozen = False


def _freeze_resident_once() -> None:
    """一括ロードした常駐 (Evidence 等) を GC の走査から外す (G1 設計 §17.4)。

    以後の世代 GC が数万件の長寿命オブジェクトを毎回たどらない。先に 1 回集めて
    ゴミの循環を凍らせない。プロセスで 1 回だけ (起動し直すテストで凍結が積もらない)。
    """
    global _frozen
    if _frozen:
        return
    gc.collect()
    gc.freeze()
    _frozen = True


def attach_episodic_embedder(episodic: "EpisodicStore", embedder: Any) -> None:
    """埋め込みバックエンドを後から差し込む (EvorefGen 構築後に呼ぶ)。

    ``EpisodicStore`` は snapshot 生成時にしか埋め込みを呼ばないので、起動順を
    「Gen より先に Mem」に保ったまま後付けできる。``None`` のままだと索引は
    語彙側だけになる (縮退動作)。
    """
    if episodic is None or embedder is None:
        return
    episodic.evidence.embedding_backend = embedder


def attach_semantic_embedder(semantic: Any, embedder: Any) -> None:
    """埋め込みバックエンドを SemMem 側へ後から差し込む (EvorefGen 構築後)。

    ``SemanticStore`` も snapshot 生成時にしか埋め込みを呼ばないので、起動順を
    「Gen より先に Mem」に保ったまま後付けできる。``None`` のままだと索引は
    語彙側だけになる (縮退動作)。
    """
    if semantic is None or embedder is None:
        return
    semantic.evidence.embedding_backend = embedder
