"""設定変更時のコンポーネント再生成ヘルパー

config_api でセクション保存後、対象コンポーネントを再生成して
AppState に差し替える。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import get_config
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("api.component_reload")

# backend/free/api/config/component_reload.py から見て parents[4] がリポジトリルート。
# parents[3] (= backend/) を渡すと埋め込みキャッシュ等のデータ根が backend/userdata/ に解決され、
# 正規の起動経路 (_pillar_wirer / component_rebind) とずれた迷子データを生む。
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def follow_embedder_rebind(state: "AppState") -> None:
    """embedder の差し替えに事例ゲート (kNN) を追随させる。

    ``state.embedder`` を差し替える経路 (config API の再生成 / model migrate の
    rebind) は、``ToolCallJudge`` と ``ToolGateKNN`` が構築時の参照と旧モデル空間の
    exemplar ベクトルを握ったままにしていた (次元が同じなら縮退にも掛からず投票
    だけが狂う)。参照を差し替えて exemplar を捨て、再 warmup は背景で行う。

    属性スロットの補完ゲート (:mod:`backend.free.memory.notes.attribute_gate`)
    も同じ事情を持つ。較正で得た類似度の床は埋め込み空間に依存するので、
    ベクトルと一緒に捨てて引き直す。
    """
    import asyncio

    judge = getattr(state, "tool_call_judge", None)
    if judge is not None:
        judge.rebind_embedder(state.embedder)
        if state.embedder is not None:
            asyncio.create_task(judge.warmup_tool_gate(), name="tool_gate_rewarmup")

    # few-shot 手本の埋め込みも旧モデルの空間のもの。捨てて張り直す。
    scheduler = getattr(state, "learning_scheduler", None)
    fewshot_pool = getattr(scheduler, "_fewshot_pool", None) if scheduler else None
    if fewshot_pool is not None:
        fewshot_pool.drop_embeddings()
        if state.embedder is not None:
            asyncio.create_task(
                fewshot_pool.backfill_embeddings(state.embedder),
                name="fewshot_embedding_rewarmup",
            )

    for attr, task_name in (
        ("attribute_slot_gate", "attribute_gate_rewarmup"),
        ("retrieval_skip_gate", "retrieval_skip_gate_rewarmup"),
        ("layer_shadow", "layer_shadow_rewarmup"),
        ("write_intent_gate", "write_intent_gate_rewarmup"),
        ("statement_gate", "statement_gate_rewarmup"),
        ("edit_mode_gate", "edit_mode_gate_rewarmup"),
    ):
        gate = getattr(state, attr, None)
        if gate is None:
            continue
        gate.reset(state.embedder)
        if state.embedder is not None:
            asyncio.create_task(gate.warmup(), name=task_name)


def follow_base_model_rebind(state: "AppState", new_model_filename: str) -> dict:
    """base モデルの差し替えに Learn pillar のパーティション束ねを追随させる。

    llama-server を新 base で再起動して ``state.local_client`` を差し替えた直後
    (``/api/model/reload``) に呼ぶ。experience / base prompts / fewshot / policy
    等を新モデルの (model_key×mode) パーティションへ再バインドし、旧パーティションへは
    先に退避する (``backend.factory._learning_rebind.rebind_base_learning``)。
    失敗しても呼出側 (モデル reload 自体) は成功扱いにし、WARNING で残す。
    """
    from backend.factory._learning_rebind import rebind_base_learning

    try:
        result = rebind_base_learning(state, new_model_filename=new_model_filename)
    except Exception as e:
        logger.warning("Learning partition rebind failed after base swap: %s", e)
        return {"rebound": False, "reason": f"error: {e}"}
    if result.get("rebound"):
        logger.info(
            "Learning partition rebound after base swap: %s -> %s",
            result.get("old_stem"), result.get("new_stem"),
        )
    else:
        logger.info(
            "Learning partition rebind skipped after base swap: %s",
            result.get("reason"),
        )
    return result


async def reload_embedder(state: AppState) -> None:
    """embedding セクション変更時に embedder を再生成して差し替える。

    新しい embedder を作れてから差し替え、旧 embedder はその後で閉じる。作れなければ
    旧 embedder を残したまま例外を送出する (呼出側が ``restart_required`` を立てる)。
    """
    from backend.free.rag.embedding_factory import create_embedding_backend

    cfg = get_config()
    embedder = create_embedding_backend(
        cfg, _PROJECT_ROOT, debug_logger=state.debug_logger,
    )
    old = state.embedder
    state.embedder = embedder
    try:
        follow_embedder_rebind(state)
    except Exception:
        # 追随が途中で止まると事例ゲートが新旧どちらの空間を握るか定まらない。旧へ戻して
        # 追随し直し、新しい方は閉じる (旧は閉じない)
        state.embedder = old
        try:
            follow_embedder_rebind(state)
        except Exception as e:
            logger.warning("Re-binding the previous embedder failed: %s", e)
        if embedder is not old and hasattr(embedder, "aclose"):
            try:
                await embedder.aclose()
            except Exception as e:
                logger.warning("Failed to close the new embedder: %s", e)
        raise
    logger.info(
        "Embedder reloaded: backend=%s, model=%s",
        embedder.backend_type(), embedder.model_name(),
    )

    if old is not None and old is not embedder and hasattr(old, "aclose"):
        try:
            await old.aclose()
        except Exception as e:
            logger.warning("Failed to close old embedder: %s", e)

    # 次元整合性を再評価
    try:
        from backend.free.rag.dimension_check import check_embedding_dim_consistency
        check_embedding_dim_consistency(state)
    except Exception as e:
        logger.warning("Post-reload dimension check failed: %s", e)


async def reload_prompt_manager(state: AppState) -> None:
    """instance セクション変更時にインスタンス名を稼働中の SystemPromptManager へ写す。

    マネージャは作り直さない — LearningScheduler / LearnPillar が同じオブジェクトを握り、
    few-shot の選択器とパーティションの解決 (``wire_generating_partitions``) も
    配線時にこのオブジェクトへ注入されている。作り直すと参照が割れ、それらが外れる。
    """
    mgr = state.prompt_manager
    if mgr is None:
        raise RuntimeError("prompt manager is not initialized")
    instance_name = get_config().get("instance", {}).get("name", "evoref")
    for manager in (mgr, *list(getattr(mgr, "_partition_managers", {}).values())):
        manager.instance_name = instance_name
        # セッションに凍結した静的 system は旧い名前を含むので捨てる (名前の変更を即時に効かせる)
        manager._drop_frozen()
    logger.info("PromptManager instance name updated: instance_name=%s", instance_name)


async def reload_i18n(state: AppState) -> None:  # noqa: ARG001
    """i18n セクション変更時に locale / fallback をランタイム再適用する

    UI メッセージと LLM 生成物 prose の出力言語 (``get_locale()`` 依存) を
    再起動なしで追従させる。``prompt_locale`` はプロンプト再学習を伴う専用
    エンドポイント (``POST /api/prompts/locale``) の管轄なのでここでは触れない。
    """
    from backend.i18n_helper import init_i18n

    i18n_cfg = get_config().get("i18n", {})
    locale = i18n_cfg.get("locale", "ja")
    fallback = i18n_cfg.get("fallback", "ja")
    init_i18n(locale=locale, fallback=fallback)
    logger.info("i18n reloaded: locale=%s, fallback=%s", locale, fallback)
