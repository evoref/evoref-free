"""ランタイム base モデル切替時の Learn pillar 再バインド

起動時は ``_pillar_wirer._activate_learning_partition`` が PathResolver の active
model_key と ``AppState.active_base_model_slug`` (= model_key) を確定し、その値で
experience / base prompts / FewShotPool / PolicyParamEvolver 等が (model_key×mode)
パーティション (docs/f_04 §1.2.0) に束ねられる。本モジュールは base モデルの **ランタイム** 切替
(``/api/model/migrate`` → llama-server 再起動 → ``/api/model/reload``) の後で
同じ束ね直しを再起動なしに行う。

- :func:`bind_active_base_model` — resolver の active model_key と state のスラグを
  確定する共有ヘルパ (起動時 / rebind 共用)。
- :func:`rebind_base_learning` — 旧パーティションへ退避 → active model_key 切替 →
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


def model_path_for(resolver: "PathResolver", filename: str) -> Path:
    """GGUF のファイル名からモデルのパスを決める (``model_state`` はファイル名だけを持つ)。

    ``model_paths.base_model`` と同じ名前ならそのパス、違えば同じディレクトリの
    同名ファイル (モデル移行は base_model のディレクトリの中で行う)。
    """
    base = resolver.resolve_model("base_model")
    name = Path(filename).name
    return base if base.name == name else base.parent / name


def bind_active_base_model(
    resolver: "PathResolver", state: "AppState", base_model: str | Path,
) -> tuple[str, str] | None:
    """resolver の active model_key と ``state.active_base_model_slug`` を確定する。

    ``learn.*`` subject のモデル次元は model_key (``mk_<hex16>``、c_05 §0.5.7)。

    Returns:
        ``(stem, model_key)``。``base_model`` が空なら ``None`` (束縛を外し、
        resolve_learning が ``model_paths.base_model`` から導出し直す)。
    """
    raw = Path(str(base_model or ""))
    if not raw.name:
        logger.warning("Learning partition: no base model identity")
        resolver.bind_active_model(None)
        state.active_base_model_slug = ""
        return None
    # ファイル名だけ (model_state 由来) なら base_model のディレクトリで探す
    key = resolver.bind_active_model(
        raw if raw.parent != Path() else model_path_for(resolver, raw.name),
    )
    assert key is not None
    state.active_base_model_slug = key
    return raw.stem, key


def rebind_base_learning(
    state: "AppState", *, new_model_filename: str,
) -> dict[str, Any]:
    """base モデル切替後に Learn pillar を新 (model_key×mode) パーティションへ束ね直す。

    手順:
      1. 旧パーティションへ退避 (learning_state / policy_evolver / exploration /
         yaml モードの fewshot_pool、経験バッファ)。
      2. ``PathResolver`` の active model_key と ``state.active_base_model_slug`` を新
         モデルへ切替 (:func:`bind_active_base_model`)。
      3. PolicyInterpreter を再スコープ (進化対象ポリシーの置き場 + SemMem override)、
         SystemPromptManager / AuxPromptManager の prompt_dir を差替えて再ロード、
         ExperienceBuffer を新パーティションへ rebind、FeedbackCollector の
         base_model 名を更新、LearningScheduler 経由で learning_state /
         PolicyParamEvolver / FewShotPool (再 bootstrap) / FeedbackPipe /
         GenerationParamEvolver を再バインド、PolicyAdjuster を再スコープ。

    Args:
        state: AppState (Learn pillar 構築済)。
        new_model_filename: 新 base モデルの GGUF ファイル名 (パス可、name を使う)。

    Returns:
        ``{"rebound": bool, "reason": str | None, "old_stem", "new_stem",
        "base_model_id", "prompt_dir", "experience_file", "fewshot_total"}``。
        同一モデル (model_key が同じ) / Level 1 実行中は ``rebound=False`` で理由を返す。
    """
    from backend.config import get_path_resolver

    resolver = get_path_resolver()
    new_name = Path(new_model_filename or "").name
    if not new_name:
        return {"rebound": False, "reason": "no_model_identity"}

    old_stem = resolver.active_model_stem
    new_stem = Path(new_name).stem
    new_path = model_path_for(resolver, new_name)
    if resolver.model_key_for(new_path) == resolver.active_model_key:
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

    # 2. active model_key / slug を切替
    bound = bind_active_base_model(resolver, state, new_path)
    if bound is None:
        return {"rebound": False, "reason": "bind_failed",
                "old_stem": old_stem, "new_stem": new_stem}
    stem, slug = bound

    # 3. 各コンポーネントを新パーティションへ
    policy = getattr(state, "policy_interpreter", None)
    if policy is not None:
        policy.rebind_partition(resolver.resolve_learning("evolved_policies_dir"), slug)

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

    _rebind_lora_partition(state, scheduler)

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


def _rebind_lora_partition(state: "AppState", scheduler: Any) -> None:
    """base LoRA のパスをキャッシュしている保持者を新パーティションへ向け直す。

    起動時は ``_pillar_wirer._wire_sleep_scheduler_models`` が Pro のパスを
    SleepTimeScheduler と Level 2 の version_manager に焼き込むため、
    切替後に呼び直さないと旧モデルのアダプタを指したままになる。Pro が保持する
    :class:`LoRAVersionManager` (CartridgeChangeHandler / ProLearnComponents 経由)
    は Free から import できないので、Pro が登録したハンドラへ委譲する。
    """
    from backend.edition import get_pro_handler

    pro_path = get_pro_handler("pro_learning_path")
    if pro_path is None:
        return
    lora_path = pro_path("lora_adapter")
    versions_dir = pro_path("lora_versions_dir")

    sleep_scheduler = getattr(state, "sleep_scheduler", None)
    if sleep_scheduler is not None:
        sleep_scheduler.set_lora_path(lora_path)

    vm_cls = get_pro_handler("lora_version_manager")
    if vm_cls is not None and scheduler is not None:
        try:
            scheduler.set_version_manager(vm_cls(versions_dir, lora_path))
        except Exception as exc:
            logger.warning("Base LoRA version_manager rebind skipped: %s", exc)

    rebind_pro_vm = get_pro_handler("rebind_lora_version_manager")
    if rebind_pro_vm is not None:
        try:
            rebind_pro_vm(state, versions_dir, lora_path)
        except Exception as exc:
            logger.warning("Pro LoRA version_manager rebind skipped: %s", exc)


def install_rebind_hook(state: "AppState") -> None:
    """``LearningScheduler`` にランタイム切替検知時の自己修復フックを注入する。

    ``_base_model_changed`` が ModelState との食い違いを見つけたとき、Level 1 を
    止める代わりに :func:`rebind_base_learning` をその場で呼ぶ。
    """
    scheduler = getattr(state, "learning_scheduler", None)
    if scheduler is None:
        return
    scheduler.set_partition_rebind_hook(
        lambda filename: rebind_base_learning(state, new_model_filename=filename),
    )
