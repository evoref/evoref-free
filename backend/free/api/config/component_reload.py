"""設定変更時のコンポーネント再生成ヘルパー

config_api でセクション保存後、対象コンポーネントを再生成して
AppState に差し替える。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import get_config, get_path_resolver
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("api.component_reload")

# backend/free/api/config/component_reload.py から見て parents[4] がリポジトリルート。
# parents[3] (= backend/) を渡すと cache_dir 等の相対パスが backend/local/ に解決され、
# 正規の起動経路 (_pillar_wirer / component_rebind) とずれた迷子データを生む。
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def follow_embedder_rebind(state: "AppState") -> None:
    """embedder の差し替えにツール判定器 (kNN ゲート) を追随させる。

    ``state.embedder`` を差し替える経路 (config API の再生成 / model migrate の
    rebind) は、``ToolCallJudge`` と ``ToolGateKNN`` が構築時の参照と旧モデル空間の
    exemplar ベクトルを握ったままにしていた (次元が同じなら縮退にも掛からず投票
    だけが狂う)。参照を差し替えて exemplar を捨て、再 warmup は背景で行う。
    """
    judge = getattr(state, "tool_call_judge", None)
    if judge is None:
        return
    judge.rebind_embedder(state.embedder)
    if state.embedder is None:
        return
    import asyncio

    asyncio.create_task(judge.warmup_tool_gate(), name="tool_gate_rewarmup")


def follow_base_model_rebind(state: "AppState", new_model_filename: str) -> dict:
    """base モデルの差し替えに Learn pillar のパーティション束ねを追随させる。

    llama-server を新 base で再起動して ``state.local_client`` を差し替えた直後
    (``/api/model/reload``) に呼ぶ。experience / base prompts / fewshot / policy
    等を新モデルの (model×mode) パーティションへ再バインドし、旧パーティションへは
    先に退避する (``backend.factory._learning_rebind.rebind_base_learning``)。
    失敗しても呼出側 (モデル reload 自体) は成功扱いにし、WARNING で残す。
    """
    from backend.factory._learning_rebind import rebind_base_learning

    try:
        result = rebind_base_learning(
            state, get_config(), new_model_filename=new_model_filename,
        )
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
    """embedding セクション変更時に embedder を再生成して差し替える"""
    from backend.free.rag.embedding_factory import create_embedding_backend

    cfg = get_config()

    # 旧 embedder を閉じる
    old = state.embedder
    if old is not None and hasattr(old, "aclose"):
        try:
            await old.aclose()
        except Exception as e:
            logger.warning("Failed to close old embedder: %s", e)

    try:
        embedder = create_embedding_backend(
            cfg, _PROJECT_ROOT, debug_logger=state.debug_logger,
        )
        state.embedder = embedder
        logger.info(
            "Embedder reloaded: backend=%s, model=%s",
            embedder.backend_type(), embedder.model_name(),
        )
    except Exception as e:
        logger.error("Failed to reload embedder: %s", e)
        state.embedder = None
    follow_embedder_rebind(state)

    # 次元整合性を再評価
    try:
        from backend.free.rag.dimension_check import check_embedding_dim_consistency
        check_embedding_dim_consistency(state)
    except Exception as e:
        logger.warning("Post-reload dimension check failed: %s", e)


async def reload_prompt_manager(state: AppState) -> None:
    """instance セクション変更時に SystemPromptManager を再生成して差し替える"""
    from backend.free.agent.prompt_manager import SystemPromptManager

    cfg = get_config()
    resolver = get_path_resolver()
    # base システムプロンプトは (model×mode) パーティション配下 (resolve_learning)。
    prompt_dir = resolver.resolve_learning("prompts_dir")
    instance_name = cfg.get("instance", {}).get("name", "evoref")
    state.prompt_manager = SystemPromptManager(prompt_dir, instance_name=instance_name)
    logger.info("PromptManager reloaded: instance_name=%s", instance_name)


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
