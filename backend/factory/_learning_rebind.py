"""ランタイム base モデル切替時の Learn pillar 再バインド

起動時は ``_pillar_wirer._activate_learning_partition`` が PathResolver の active
stem と ``AppState.active_base_model_slug`` を確定し、その値で experience /
base prompts / FewShotPool / PolicyParamEvolver 等が (model×mode) パーティション
(docs/f_04 §1.2) に束ねられる。本モジュールは base モデルの **ランタイム** 切替
(``/api/model/migrate`` → llama-server 再起動 → ``/api/model/reload``) の後で
同じ束ね直しを再起動なしに行う。

- :func:`bind_active_base_model` — resolver の active stem と state のスラグを
  確定する共有ヘルパ (起動時 / rebind 共用)。
- :func:`rebind_base_learning` — 旧パーティションへ退避 → active stem 切替 →
  各コンポーネントを新パーティションへ向け直して再ロード。
- :func:`install_rebind_hook` — ``LearningScheduler`` が ModelState との食い違いを
  検知したときの自己修復フックを注入する。

共有 (非パーティション) の ``LearnedPatternStore`` と、embedding モデル軸で
分離されている embed_instruction には触れない。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState
    from backend.config import PathResolver

logger = get_logger("factory.learning_rebind")


def bind_active_base_model(
    resolver: "PathResolver", state: "AppState", base_filename: str,
) -> tuple[str, str] | None:
    """resolver の active モデル stem と ``state.active_base_model_slug`` を確定する。

    Returns:
        ``(stem, slug)``。``base_filename`` が空、または slug 化に失敗した場合は
        partition を無効化 (flat) して ``None``。
    """
    from backend.free.memory.notes.subject_ns import model_slug

    name = Path(base_filename or "").name
    if not name:
        logger.warning("Learning partition: no base model identity; staying flat")
        resolver.set_active_model_stem(None)
        state.active_base_model_slug = ""
        return None
    stem = Path(name).stem
    try:
        slug = model_slug(name)
    except Exception as exc:
        logger.warning("Learning partition: model_slug failed (%s); staying flat", exc)
        resolver.set_active_model_stem(None)
        state.active_base_model_slug = ""
        return None
    resolver.set_active_model_stem(stem)
    state.active_base_model_slug = slug
    return stem, slug


def rebind_base_learning(
    state: "AppState", cfg: dict[str, Any], *, new_model_filename: str,
) -> dict[str, Any]:
    """base モデル切替後に Learn pillar を新 (model×mode) パーティションへ束ね直す。

    手順:
      1. 旧パーティションへ退避 (learning_state / policy_evolver / exploration /
         yaml モードの fewshot_pool、経験バッファ)。
      2. ``PathResolver`` の active stem と ``state.active_base_model_slug`` を新
         モデルへ切替 (:func:`bind_active_base_model`)。
      3. PolicyInterpreter を再スコープ (SemMem override 再適用)、
         SystemPromptManager / AuxPromptManager の prompt_dir を差替えて再ロード、
         ExperienceBuffer を新パーティションへ rebind、FeedbackCollector の
         base_model 名を更新、LearningScheduler 経由で learning_state /
         PolicyParamEvolver / FewShotPool (再 bootstrap) / FeedbackPipe /
         GenerationParamEvolver を再バインド、PolicyAdjuster を再スコープ。

    Args:
        state: AppState (Learn pillar 構築済)。
        cfg: 現在の config dict (``learning.partition_by_base_model`` の参照用)。
        new_model_filename: 新 base モデルの GGUF ファイル名 (パス可、name を使う)。

    Returns:
        ``{"rebound": bool, "reason": str | None, "old_stem", "new_stem",
        "base_model_id", "prompt_dir", "experience_file", "fewshot_total"}``。
        partition 無効 / 同一モデル / Level 1 実行中は ``rebound=False`` で理由を返す。
    """
    from backend.config import get_path_resolver

    resolver = get_path_resolver()
    new_name = Path(new_model_filename or "").name
    partition_enabled = resolver.partition_enabled and bool(
        (cfg.get("learning") or {}).get("partition_by_base_model", True),
    )
    if not partition_enabled:
        return {"rebound": False, "reason": "partition_disabled"}
    if not new_name:
        return {"rebound": False, "reason": "no_model_identity"}

    old_stem = resolver.active_model_stem
    new_stem = Path(new_name).stem
    if old_stem == new_stem:
        return {"rebound": False, "reason": "unchanged", "old_stem": old_stem,
                "new_stem": new_stem}

    scheduler = getattr(state, "learning_scheduler", None)
    if scheduler is not None and scheduler.running:
        return {"rebound": False, "reason": "learning_running",
                "old_stem": old_stem, "new_stem": new_stem}

    # 1. 旧パーティションへ退避 (active stem を動かす前にパスを確定する)
    old_exp_file = resolver.resolve_learning("experience_file")
    if scheduler is not None:
        scheduler.save_partition_state()

    # 2. active stem / slug を切替
    bound = bind_active_base_model(resolver, state, new_name)
    if bound is None:
        return {"rebound": False, "reason": "bind_failed",
                "old_stem": old_stem, "new_stem": new_stem}
    stem, slug = bound

    # 3. 各コンポーネントを新パーティションへ
    policy = getattr(state, "policy_interpreter", None)
    if policy is not None:
        policy.set_base_model_id(slug)

    prompt_dir = resolver.resolve_learning("prompts_dir")
    prompt_mgr = getattr(state, "prompt_manager", None)
    if prompt_mgr is not None:
        prompt_mgr.rebind_prompt_dir(prompt_dir)

    aux_mgr = getattr(state, "aux_prompt_manager", None)
    if aux_mgr is not None:
        # AuxPromptManager は base 軸パーティション (resolve_aux_prompt_dir)。
        # 構築時と同じ _load_all を新ディレクトリで走らせる。
        aux_mgr.prompt_dir = resolver.resolve_aux_prompt_dir()
        aux_mgr.contents.clear()
        aux_mgr.metas.clear()
        aux_mgr._load_all()

    exp_file = resolver.resolve_learning("experience_file")
    fc = getattr(state, "feedback_collector", None)
    if fc is not None:
        fc.buffer.rebind(exp_file, previous=old_exp_file)
        fc.rebind_base_model(new_name)

    fewshot_total = None
    if scheduler is not None:
        scheduler.rebind_learning_partition(
            model_stem=stem,
            base_model_id=slug,
            generation_deltas_file=resolver.resolve_learning("generation_deltas_file"),
        )
        pool = getattr(scheduler, "_fewshot_pool", None)
        if pool is not None:
            fewshot_total = pool.total_count

    learn = getattr(state, "learn", None)
    adjuster = getattr(learn, "policy_adjuster", None) if learn is not None else None
    if adjuster is not None:
        adjuster.set_base_model_id(slug)

    logger.info(
        "Learn pillar rebound to base model partition: %s -> %s (slug=%s)",
        old_stem, stem, slug,
    )
    return {
        "rebound": True,
        "reason": None,
        "old_stem": old_stem,
        "new_stem": stem,
        "base_model_id": slug,
        "prompt_dir": str(prompt_dir),
        "experience_file": str(exp_file),
        "fewshot_total": fewshot_total,
    }


def install_rebind_hook(state: "AppState") -> None:
    """``LearningScheduler`` にランタイム切替検知時の自己修復フックを注入する。

    ``_base_model_changed`` が ModelState との食い違いを見つけたとき、Level 1 を
    止める代わりに :func:`rebind_base_learning` をその場で呼ぶ。config は呼出
    時点の ``get_config()`` を使う (migrate 後の in-memory 同期値)。
    """
    scheduler = getattr(state, "learning_scheduler", None)
    if scheduler is None:
        return
    from backend.config import get_config

    scheduler.set_partition_rebind_hook(
        lambda filename: rebind_base_learning(
            state, get_config(), new_model_filename=filename,
        ),
    )
