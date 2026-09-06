"""Pillar 配線 / DI

含まれるシンボル:

- 計測ヘルパー : :func:`_timed` / :func:`_timed_task`
- pillar 内の個別 ``_init_*`` ヘルパー (LLM / 埋め込み / リランカー /
  カートリッジ / 学習サイクル / Pro Learn 注入 / loop driver / テーマ /
  model state / エディションダウングレード)
- pillar build エントリポイント : :func:`_build_base_context` /
  :func:`_build_gen_pillar` / :func:`_build_mem_pillar` /
  :func:`_build_gen_pillar_retrieval` / :func:`_build_learn_pillar` /
  :func:`_build_loop_pillar` / :func:`_finalize_base`
- DI トップレベル : :class:`_LifespanContext` / :class:`_BaseContext` /
  :func:`wire_pillars`

純粋な move であり、関数本体・引数・default 値は変更していない。
``_build_base_context`` のみ ``_init_config`` / ``_init_logging`` /
``_init_i18n`` / ``_init_local_dirs`` を ``backend.app_factory`` から
lazy import する (循環 import 回避)。PR3 で ``_bootstrap.py`` に分離後、
top-level import に置換される予定。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from backend.app_state import AppState
from backend.log_config import get_logger
from backend.trace_context import generate_trace_id, set_trace_id

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.agent.aux_prompt_manager import AuxPromptManager
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.agent.prompt_manager import SystemPromptManager
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.learning.level0_instant import ExperienceBuffer
    from backend.free.learning.scheduler import LearningScheduler
    from backend.free.llm.aux_client import AuxClient
    from backend.free.llm.local_client import LocalClient
    from backend.free.memory.scheduler import SleepTimeScheduler
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.bm25_retriever import BM25Retriever
    from backend.free.rag.retriever import HybridRetriever
    from backend.free.rag.vector_store import VectorStore
    from backend.pillars import GenPillar, LearnPillar, LoopPillar, MemPillar

logger = get_logger("factory.pillar_wirer")

# シャットダウンフックの型エイリアス（Pro ライフサイクルの shutdown 部分）
type ProShutdownHook = Callable[[AppState], Awaitable[None]] | None
# Develop シャットダウンフック (Pro と同形)
type DevelopShutdownHook = Callable[[AppState], Awaitable[None]] | None


@contextmanager
def _timed(timings: dict[str, float], name: str):
    """コンポーネント初期化の所要時間を計測（ms）"""
    start = time.monotonic()
    try:
        yield
    finally:
        timings[name] = (time.monotonic() - start) * 1000


async def _timed_task(timings: dict[str, float], name: str, coro):
    """並列起動タスクの所要時間を計測（ms）"""
    start = time.monotonic()
    try:
        return await coro
    finally:
        timings[name] = (time.monotonic() - start) * 1000


# ──────────────────────────────────────────────────────────────────────────
# Pillar build helpers — 元 app_factory の `_init_*` 群。挙動は変えない。
# ──────────────────────────────────────────────────────────────────────────


def _base_model_path_or_none(cfg: dict[str, Any]) -> "Path | None":
    """base モデルの実ファイルパス。解決できなければ ``None``。

    起動プローブ予算の算出専用。ここで失敗しても起動は続けたいので、
    PathResolver の例外は握って ``None`` へ倒す (既定予算が使われる)。
    """
    try:
        from backend.config import PathResolver

        path = PathResolver(cfg).resolve_model("base_model")
    except Exception:
        return None
    return path if path.exists() else None


async def _init_llama_server(
    state: AppState, cfg: dict[str, Any], debug_logger: "DebugLogger",
) -> "LocalClient | None":
    """5. llama-server 接続"""
    from backend.free.llm._base_client import (
        resolve_startup_probe_timeout,
        wait_for_server_ready,
    )
    from backend.free.llm.local_client import LocalClient
    from backend.free.llm.model_metadata import fetch_model_metadata

    llama_cfg = cfg.get("llama", {})
    llama_host = llama_cfg.get("host", "127.0.0.1")
    llama_port = llama_cfg.get("port", 8080)
    llama_url = f"http://{llama_host}:{llama_port}"

    try:
        # 起動レース対策: llama-server プロセスが listen するまで /health を
        # ポーリング。失敗しても続行 → 既存の degraded mode に倒れる。
        # 予算はモデルサイズに追随させる (固定 30 秒では 16GB クラスのロードに
        # 負け、AuxClient が恒久 unwired になる — 2026-09-03 ライブ監査)。
        await wait_for_server_ready(
            f"{llama_url}/health", label="llama-server (base)",
            timeout=resolve_startup_probe_timeout(
                _base_model_path_or_none(cfg), label="llama-server (base)",
            ),
        )
        metadata = await fetch_model_metadata(llama_url, debug_logger=debug_logger)
        from backend.config import (
            resolve_client_reasoning,
            resolve_context_size,
            resolve_enable_thinking,
        )
        base_enable_thinking = resolve_enable_thinking(
            cfg, "base",
            explicit=llama_cfg.get("enable_thinking"),
            chat_template=getattr(metadata, "chat_template", None),
        )
        think_budget, on_runaway = resolve_client_reasoning(cfg, "base")
        # config の slots は宣言値。llama-server が実際に確保したスロット数
        # (``/props`` の total_slots) より多いと、``id_slot=2`` 等の要求が
        # 存在しないスロットを指して 400 になる。少ない方へ丸めて警告する。
        cfg_slots = int(llama_cfg.get("slots", 1) or 1)
        total_slots = int(getattr(metadata, "total_slots", 0) or 0)
        slots = cfg_slots
        if total_slots > 0 and total_slots != cfg_slots:
            slots = min(cfg_slots, total_slots)
            logger.warning(
                "llama.slots=%d does not match llama-server total_slots=%d; "
                "using %d (restart llama-server after changing config.yaml)",
                cfg_slots, total_slots, slots,
            )
        client = LocalClient(
            llama_url,
            metadata,
            cache_prompt=llama_cfg.get("cache_prompt", True),
            slots=slots,
            enable_thinking=base_enable_thinking,
            stream_first_token_timeout=llama_cfg.get(
                "stream_first_token_timeout_sec", 60.0,
            ),
            debug_logger=debug_logger,
            client_think_budget=think_budget,
            on_runaway=on_runaway,
            # 送信前コンテキスト超過ガード用 (slots>1 でも launch_llama が
            # --kv-unified を自動付与するため per-slot でも full n_ctx)。
            # ``llama.context_size`` の既定は None (プロファイル委譲) なので
            # ``.get(..., 4096)`` では None が入ってガードが無効化されていた。
            # 起動フラグ ``-c`` と同じ優先順位で解決する。
            context_size=resolve_context_size(cfg, "base"),
        )
        if await client.health_check():
            state.local_client = client
            logger.info(
                "llama-server connected: %s (cache_prompt=%s, slots=%d, "
                "enable_thinking=%s, n_ctx=%s)",
                llama_url, client._cache_prompt, client._slots,
                client._enable_thinking, client._context_size,
            )
            if client.background_slot < 0:
                logger.warning(
                    "llama.slots=%d: no dedicated background slot — aux / "
                    "sleep-time requests will share KV with chat (set slots >= 2)",
                    client._slots,
                )
            _start_capability_probe(
                client, cfg, metadata, llama_url, llama_cfg, debug_logger,
            )
            return client
        logger.warning("llama-server health check failed: %s", llama_url)
    except Exception as e:
        logger.warning("llama-server not available: %s", e)
    return None


def _start_capability_probe(
    client: "LocalClient",
    cfg: dict[str, Any],
    metadata: Any,
    llama_url: str,
    llama_cfg: dict,
    debug_logger: "DebugLogger",
) -> None:
    """ベースモデルの実挙動プローブをバックグラウンドで起動する (docs/c_15)。

    ``runtime.capability_probe`` (既定 True) が False なら no-op。プローブ完了時に
    ``client.capabilities`` を確定し、観測 reasoning mode で ``enable_thinking`` を
    再解決して in-place 更新する。起動はブロックせず、完了までは prior (宣言) で動作。
    失敗時は prior を維持 (degraded 安全)。
    """
    if not (cfg.get("runtime", {}) or {}).get("capability_probe", True):
        return

    import asyncio

    from backend.config import resolve_enable_thinking, resolve_reasoning_mode
    from backend.free.llm.capability import (
        PROBE_REASONING_MAX_TOKENS,
        make_llama_chat_fn,
        probe_model_capabilities,
        probe_timeout_sec,
    )

    chat_template = getattr(metadata, "chat_template", None)
    params_b = float(getattr(metadata, "params_b", 0.0) or 0.0)

    async def _run() -> None:
        try:
            declared = resolve_reasoning_mode(cfg, "base", chat_template=chat_template)
            # モデルサイズに追随したタイムアウト (固定 120 秒は 27B+ の iGPU で
            # 足りず、観測が恒久的に None になっていた)
            probe_timeout = probe_timeout_sec(params_b, PROBE_REASONING_MAX_TOKENS)
            snapshot = await probe_model_capabilities(
                model_id=getattr(metadata, "model_id", ""),
                template_family=getattr(metadata, "template_family", "unknown"),
                declared_reasoning_mode=declared,
                chat_fn=make_llama_chat_fn(
                    llama_url,
                    debug_logger=debug_logger,
                    timeout=probe_timeout,
                    # プローブは背景処理。チャットの初回応答と KV を分離する
                    id_slot=client.background_slot,
                ),
                probe_json=True,  # 補助タスクの JSON purpose も base が担う
            )
            client.capabilities = snapshot
            # 観測を使う (docs/c_15): </think> を閉じないモデルは暴走 reasoning
            # の watchdog を必ず武装する。プロファイルが budget=0 (無効) でも、
            # 実機が閉じないなら上限無しのまま流す理由が無い。
            if snapshot.closes_think_tags is False and client._client_think_budget <= 0:
                client._client_think_budget = _RUNAWAY_THINK_BUDGET_DEFAULT
                logger.warning(
                    "Capability probe: model emits <think> without closing it; "
                    "arming client think watchdog (budget=%d chunks)",
                    _RUNAWAY_THINK_BUDGET_DEFAULT,
                )
            new_enable = resolve_enable_thinking(
                cfg, "base",
                explicit=llama_cfg.get("enable_thinking"),
                chat_template=chat_template,
                observed_reasoning_mode=snapshot.effective_reasoning_mode,
            )
            if new_enable != client._enable_thinking:
                logger.info(
                    "Capability probe adjusted base enable_thinking: %s -> %s "
                    "(observed reasoning mode=%s)",
                    client._enable_thinking, new_enable,
                    snapshot.effective_reasoning_mode,
                )
                client._enable_thinking = new_enable
            await _calibrate_token_estimate(client, llama_url)
        except Exception as e:
            logger.warning("Capability probe task failed (prior retained): %s", e)

    client._capability_probe_task = asyncio.create_task(_run())


#: プローブが「</think> を閉じない」と観測したときに武装する watchdog の chunk
#: 上限 (``ProfileReasoningConfig.client_think_budget`` の既定値と同じ)。
_RUNAWAY_THINK_BUDGET_DEFAULT = 512


#: CJK 項の較正に使うサンプル。**日本語のみ**にすること — ASCII を混ぜると比が
#: 薄まるうえ、コードは逆に過小評価側なので係数の意味が壊れる
#: (``backend.utils._CJK_TOKEN_SCALE`` の解説を参照)。
_TOKEN_CALIBRATION_SAMPLE = (
    "あなたの名前はアリスです。ユーザーに名前を聞かれた場合や自己紹介を求められた"
    "場合は、この名前で答えてください。質問に直接答えることを最優先し、前置きや"
    "質問の復唱、断り書きから始めず、最初の文で答えの核心を述べてください。"
    "挨拶や雑談、単純な事実の質問には短く答え、手順や比較、複数の論点を含む質問は"
    "見出しや箇条書きで構造化します。聞かれていない周辺知識は付け足しません。"
    "曖昧な質問には自分の解釈を一文で示してから答えるか、確認の質問を一つだけ"
    "返してください。過去の記録と今回の会話が食い違う場合は今回の会話を優先します。"
)


async def _calibrate_token_estimate(client: Any, llama_url: str) -> None:  # noqa: ARG001
    """``estimate_tokens`` の CJK 項を実トークナイザで較正する。

    素の推定 (CJK 1 文字 ≒ 1 トークン) は Qwen3 系の日本語で 1.5 倍前後の過大評価に
    なり、予算が推定トークン建てである以上そのぶんコンテキストを使い残す。
    ``/tokenize`` を 1 回だけ叩いて比を取り、以後の予算計算へ反映する。
    失敗しても係数 1.0 のまま = 従来動作なので、起動は止めない。
    """
    from backend.utils import estimate_tokens, set_cjk_token_scale

    sample = _TOKEN_CALIBRATION_SAMPLE
    actual = await client.count_tokens(sample)
    if not actual:
        logger.warning(
            "CJK token scale calibration skipped: /tokenize returned no tokens",
        )
        return
    raw = estimate_tokens(sample)
    scale = set_cjk_token_scale(raw, actual)
    logger.info(
        "CJK token scale calibrated: raw=%d actual=%d (over-estimate %.2fx) "
        "-> scale=%.3f",
        raw, actual, (raw / actual) if actual else 0.0, scale,
    )


async def _init_pro_gen_pillar(
    state: AppState, cfg: dict[str, Any], resolver: Any,
) -> ProShutdownHook:
    """5b. Pro Gen pillar setup に委譲

    Pro エディション起動時は :func:`backend.pro.setup_pro_gen` が登録されており、
    :class:`ProGenPillar` を返す。``state.pro.gen`` に保持して後続の
    :func:`setup_pro_learn` が参照できるようにする
    サポートを撤去したため、本フックは Pro 拡張 (WidgetProxyManager /
    ターミナル等) の初期化のみを担う。
    """
    from backend.edition import get_pro_pillar_setup, get_pro_shutdown
    from backend.pillars import ProState

    pro_gen_setup = get_pro_pillar_setup("gen")
    pro_shutdown = get_pro_shutdown()

    if pro_gen_setup:
        try:
            pro_gen_pillar = await pro_gen_setup(cfg, resolver, state)
            if state.pro is None:
                state.pro = ProState(gen=pro_gen_pillar)
            else:
                state.pro.gen = pro_gen_pillar
            logger.info("Pro Gen pillar setup completed")
        except Exception as e:
            logger.warning("Pro Gen pillar setup failed: %s", e)
    else:
        logger.info("Pro Gen pillar not registered (Free edition)")
    return pro_shutdown


async def _init_develop_gen_pillar(
    state: AppState, cfg: dict[str, Any], resolver: Any,
) -> DevelopShutdownHook:
    """Develop Gen pillar setup に委譲 (Pro と並列)。

    Develop エディション起動時に :func:`backend.develop.setup_develop` で
    ``register_develop_pillar_setup("gen", ...)`` が呼ばれていれば、その setup を
    実行し :class:`DevelopGenPillar` を ``state.develop.gen`` に保持する。
    現状はスケルトン段階のため、未登録時は何もしない (no-op)。
    """
    from backend.edition import get_develop_pillar_setup, get_develop_shutdown
    from backend.pillars import DevelopState

    develop_gen_setup = get_develop_pillar_setup("gen")
    develop_shutdown = get_develop_shutdown()

    if develop_gen_setup:
        try:
            develop_gen_pillar = await develop_gen_setup(cfg, resolver, state)
            if state.develop is None:
                state.develop = DevelopState(gen=develop_gen_pillar)
            else:
                state.develop.gen = develop_gen_pillar
            logger.info("Develop Gen pillar setup completed")
        except Exception as e:
            logger.warning("Develop Gen pillar setup failed: %s", e)
    else:
        logger.debug("Develop Gen pillar not registered (skeleton stage)")
    return develop_shutdown


def _init_llm_client(
    state: AppState, client: "LocalClient | None",
) -> None:
    """5c. LLMClient 統合クライアント作成"""
    from backend.free.llm.llm_client import LLMClient
    if client:
        state.llm_client = LLMClient(local=client)
        logger.info("LLMClient initialized (local llama-server)")


def _track_background_task(
    state: AppState, coro: "Awaitable[Any]", *, name: str,
) -> "asyncio.Task[Any]":
    """fire-and-forget の背景タスクを ``state.background_tasks`` に保持して起動する。

    イベントループはタスクを弱参照しか持たないため、戻り値を捨てると実行途中で
    GC され得る。完了時に集合から自ら抜ける。イベントループ外では
    ``asyncio.create_task`` と同じく ``RuntimeError``。
    """
    task = asyncio.create_task(coro, name=name)
    state.background_tasks.add(task)
    task.add_done_callback(state.background_tasks.discard)
    return task


def _init_aux_client(
    state: AppState,
    cfg: dict[str, Any],
    debug_logger: "DebugLogger",
    local_client: Any,
) -> "AuxClient | None":
    """5d. 補助タスク用クライアント初期化

    補助タスク (sleep-time / 学習 / 長文プラン / create の計画・仕様合成) は
    ベースモデルの専有スロットで回す。専用サーバもヘルスチェックも要らず、
    ベースが接続できていれば必ず使える。ベース未接続時のみ ``None``。
    """
    if local_client is None or not hasattr(local_client, "generate_constrained"):
        logger.warning(
            "Aux client not wired: base llama-server is unavailable — "
            "sleep-time LLM stages, learning cycles, long-form planning and "
            "create-mode synthesis will use their fallback paths",
        )
        return None

    from backend.free.llm.aux_client import AuxClient

    aux_client = AuxClient(local_client, config=cfg, debug_logger=debug_logger)
    state.aux_client = aux_client
    logger.info(
        "Aux client wired on the base model (background slot=%d)",
        local_client.background_slot,
    )
    return aux_client


def _start_quality_probes(
    state: AppState,
    cfg: dict[str, Any],
    resolver: Any,
    debug_logger: "DebugLogger",
    *,
    local_client: Any,
    embedder: Any,
) -> None:
    """モデル切替を検知した役割だけ出力品質プローブを走らせる。

    ``capability_probe`` が観測するのは形式 (reasoning 分離 / json 強制) だけで
    **出力の質は見ない**。base を Qwen3.5-9B → gemma-4-12b へ切り替えた際、
    日本語の語間空白が 1% → 76〜83% に悪化したのに起動も capability probe も
    正常で、27 ターン会話するまで気付けなかった (2026-08-02)。その穴を塞ぐ。

    毎起動は走らせない。``local/model_quality.json`` に「役割ごとに最後に検査した
    モデル名」を記録し、**変わったときだけ**カナリアを投げる (iGPU では 3 役割で
    分単位かかるため)。起動はブロックせず、失敗しても prior のまま継続する。
    """
    if not (cfg.get("runtime", {}) or {}).get("capability_probe", True):
        return

    from backend.free.llm.quality_probe import (
        QualityProbeStore,
        log_probe_result,
        probe_embed_quality,
        probe_text_quality,
    )

    model_paths = cfg.get("model_paths", {}) or {}
    targets: list[tuple[str, str, Any]] = [
        ("base", model_paths.get("base_model") or "", local_client),
        ("embedding", model_paths.get("embed_model") or "", embedder),
    ]
    raw_paths = {role: raw for role, raw, _ in targets}

    try:
        store = QualityProbeStore(resolver.resolve_local("model_quality_file"))
    except Exception as exc:
        logger.warning("Quality probe store unavailable: %s", exc)
        return
    state.model_quality = store

    pending = [
        (role, Path(raw).name, client)
        for role, raw, client in targets
        if client is not None
        and store.needs_probe(role, Path(raw).name if raw else "")
    ]
    if not pending:
        return
    logger.info(
        "Quality probe scheduled for changed models: %s",
        ", ".join(f"{role}={name}" for role, name, _ in pending),
    )

    async def _run() -> None:
        for role, model_name, client in pending:
            try:
                baseline = _resolve_role_quality_baseline(raw_paths[role])
                if role == "embedding":
                    result = await probe_embed_quality(
                        model=model_name,
                        embed_fn=_make_probe_embed_fn(client),
                        baseline=baseline,
                        expected_dim=(cfg.get("embedding", {}) or {}).get("dim"),
                    )
                else:
                    from backend.free.llm.capability import (
                        make_llama_chat_fn,
                        probe_timeout_sec,
                    )
                    from backend.free.llm.quality_probe import _PROBE_MAX_TOKENS
                    url = getattr(client, "url", "")
                    if not url:
                        continue
                    params_b = float(
                        getattr(getattr(client, "metadata", None), "params_b", 0.0) or 0.0,
                    )
                    # モデルサイズに追随 (固定 90 秒だと 27B+ の iGPU で 200 トークン
                    # の decode が収まらない)
                    quality_timeout = probe_timeout_sec(params_b, _PROBE_MAX_TOKENS)
                    result = await probe_text_quality(
                        role=role,
                        model=model_name,
                        chat_fn=make_llama_chat_fn(
                            url, debug_logger=debug_logger, timeout=quality_timeout,
                            # 品質プローブも背景処理。チャットと KV を分離する
                            id_slot=client.background_slot,
                        ),
                        baseline=baseline,
                        # 本番のチャットパスと同じ thinking 設定で測る。既定のまま
                        # 投げると reasoning モデルでは思考が生成上限を食い潰し、
                        # content が空のまま「観測不能」になる (実機で確認)。
                        enable_thinking=_probe_enable_thinking(client),
                    )
                log_probe_result(result)
                store.record(result)
            except Exception as exc:
                logger.warning(
                    "Quality probe failed for role=%s (skipped): %s", role, exc,
                )

    state._quality_probe_task = asyncio.create_task(_run())


def _probe_enable_thinking(client: Any) -> bool | None:
    """クライアントが本番で使っている ``enable_thinking`` を取り出す。

    品質プローブは**ユーザーが実際に受け取る本文**を測るので、本番と同じ設定で
    投げる必要がある。設定が違えば別のものを測ってしまう (既定のまま投げたところ、
    reasoning モデルで思考が生成上限を食い潰し content が空になった)。

    ``LocalClient._enable_thinking`` は起動時に ``resolve_enable_thinking`` で
    解決済みの値なので、ここでは読むだけで再導出しない (二重解決は乖離のもと)。
    ``None`` は「送らない」を意味する。
    """
    value = getattr(client, "_enable_thinking", None)
    if value is not None:
        return bool(value)
    return None


def _resolve_role_quality_baseline(model_path: str) -> Any:
    """モデルプロファイルから ``quality_baseline`` を解決する。

    プロファイルに ``quality_baseline:`` が無い / 読めない場合は既定閾値 —
    プロファイルはあくまで既定の**調整**で、無くても検査自体は成立させる。
    モデル別層 (by-model) が効くので、同 arch でもサイズ・量子化ごとに
    実測済みの弱点を宣言できる。
    """
    from backend.free.llm.quality_probe import resolve_quality_baseline

    if not model_path:
        return resolve_quality_baseline(None)
    try:
        from backend.config import get_project_root
        from scripts.launch_llama import load_model_profile_for

        root = get_project_root()
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = root / resolved
        profile = load_model_profile_for(resolved, root)
    except Exception as exc:
        logger.debug(
            "quality baseline profile load failed (%s): %s", model_path, exc,
        )
        return resolve_quality_baseline(None)
    return resolve_quality_baseline(profile)


def _make_probe_embed_fn(embedder: Any) -> Any:
    """``EmbeddingBackend`` を品質プローブ用の ``EmbedFn`` へ薄く包む。

    プローブ文はクエリでもドキュメントでもない中立のカナリアなので
    ``is_query=False`` (prefix なし) で埋め込む — instruction prefix が乗ると
    測っているのがモデルではなく prefix になってしまう。
    """
    async def _embed(texts: Any) -> Any:
        return await embedder.embed(list(texts), is_query=False)

    return _embed


def _init_cartridge_manager(state: AppState, cfg: dict[str, Any], resolver: Any) -> None:
    """6b. カートリッジマネージャ初期化"""
    from backend.free.rag.cartridge_manager import CartridgeManager

    try:
        cartridges_dir = resolver.resolve_local("cartridges_dir")
        rag_cfg = cfg.get("rag", {})
        cart_mgr = CartridgeManager(
            cartridges_dir, rag_config=rag_cfg, debug_logger=state.debug_logger,
        )
        state.cartridge_manager = cart_mgr
        logger.info(
            "CartridgeManager initialized: dir=%s, installed=%d",
            cartridges_dir, len(cart_mgr.list_cartridges()),
        )
    except Exception as e:
        logger.warning("CartridgeManager init skipped: %s", e)


def _load_experience_buffer(resolver: Any) -> tuple["ExperienceBuffer", Path]:
    """7a. 経験バッファ読込み（破損時は空のバッファで継続）"""
    from backend.free.learning.level0_instant import ExperienceBuffer

    exp_buf = ExperienceBuffer()
    # base 経験は (model×mode) partition 配下 (resolve_learning)。partition 無効時は
    # resolve_local 同等 (flat)。
    exp_file = resolver.resolve_learning("experience_file")
    # 保存先は存在有無に依らず先に束ねる (record 時の自動保存が初回起動から効く)。
    exp_buf.bind(exp_file)
    if exp_file.exists():
        try:
            exp_buf.load(exp_file)
            logger.info("Experience buffer loaded: %d entries", len(exp_buf.entries))
        except Exception as e:
            logger.warning("Experience buffer load failed: %s", e)
    return exp_buf, exp_file


def _load_learned_patterns(
    cfg: dict[str, Any], resolver: Any, policy_interpreter: "PolicyInterpreter",
) -> tuple["LearnedPatternStore", Path]:
    """学習済みパターンストア初期化"""
    from backend.free.agent.learned_patterns import LearnedPatternStore

    learned_patterns_store = LearnedPatternStore(cfg, policy=policy_interpreter)
    patterns_file = resolver.resolve_learning("learned_patterns_file")
    if patterns_file.exists():
        try:
            learned_patterns_store.load(patterns_file)
            logger.info("Learned patterns loaded: %d patterns", learned_patterns_store.count)
        except Exception as e:
            logger.warning("Learned patterns load failed: %s", e)
    return learned_patterns_store, patterns_file


def _init_learning_core(
    state: AppState,
    cfg: dict[str, Any],
    resolver: Any,
    debug_logger: "DebugLogger",
    policy_interpreter: "PolicyInterpreter",
) -> tuple[
    "ExperienceBuffer",
    Path,
    "LearnedPatternStore",
    Path,
    "SystemPromptManager",
    "AuxPromptManager",
    "SleepTimeScheduler",
]:
    """7. 学習サイクル初期化（経験バッファ・パターン・プロンプト・SleepTimeScheduler）"""
    from backend.free.agent.feedback import FeedbackCollector
    from backend.free.agent.prompt_manager import SystemPromptManager
    from backend.free.memory.scheduler import SleepTimeScheduler

    # 7a. 経験バッファ + 学習済みパターン + フィードバック収集
    exp_buf, exp_file = _load_experience_buffer(resolver)
    learned_patterns_store, patterns_file = _load_learned_patterns(
        cfg, resolver, policy_interpreter,
    )
    state.learned_patterns_store = learned_patterns_store

    # CartridgeManager に学習済みパターンストアを注入
    if state.cartridge_manager is not None:
        state.cartridge_manager.set_learned_patterns(learned_patterns_store)

    # 経験に現在ロード中のモデル名 (GGUF ファイル名) を刻む。Level 2 base=C の
    # build_contrastive_pairs は current_model でこの base_model を一致フィルタする
    # ため、空のままだと母集団が seed のみに縮退する (cvector が学習信号を失う)。
    _model_paths = cfg.get("model_paths", {})
    _base_model_name = Path(_model_paths.get("base_model") or "").name
    _embed_model_name = Path(_model_paths.get("embed_model") or "").name
    fc = FeedbackCollector(
        exp_buf, debug_logger=debug_logger,
        learned_patterns=learned_patterns_store,
        disabled=state.learning_disabled,
        base_model_name=_base_model_name,
        embedding_model_name=_embed_model_name,
    )
    state.feedback_collector = fc

    # 7b. プロンプトマネージャ（§6.6.4: モード別システムプロンプト）
    # base システムプロンプト (chat.md / create.md + meta + history + learning_state)
    # は (model×mode) partition 配下 (resolve_learning)。補助タスクのプロンプトは
    # モードに依存しない共有軸 (resolve_aux_prompt_dir) に置き、両者が同一
    # prompts_dir を共有しないよう分離する。
    base_prompt_dir = resolver.resolve_learning("prompts_dir")
    aux_prompt_dir = resolver.resolve_aux_prompt_dir()
    instance_name = cfg.get("instance", {}).get("name", "evoref")
    prompt_mgr = SystemPromptManager(base_prompt_dir, instance_name=instance_name)
    state.prompt_manager = prompt_mgr

    # 7b-2. 補助タスクプロンプトマネージャ（§7.1.2: タスク別プロンプト）
    from backend.free.agent.aux_prompt_manager import AuxPromptManager

    aux_prompt_mgr = AuxPromptManager(aux_prompt_dir)
    state.aux_prompt_manager = aux_prompt_mgr
    logger.info(
        "AuxPromptManager initialized: %d tasks loaded",
        len(aux_prompt_mgr.contents),
    )

    # 7c. SleepTimeScheduler
    sleep_scheduler = state.sleep_scheduler
    if sleep_scheduler is None:
        # bg_task wrapper の outcome.jsonl 記録のため debug_logger を注入
        sleep_scheduler = SleepTimeScheduler(cfg, debug_logger=debug_logger)
        state.sleep_scheduler = sleep_scheduler

    return (
        exp_buf, exp_file,
        learned_patterns_store, patterns_file,
        prompt_mgr, aux_prompt_mgr,
        sleep_scheduler,
    )


async def _init_embedding(
    state: AppState,
    cfg: dict[str, Any],
    project_root: Path,
    debug_logger: "DebugLogger",
) -> "EmbeddingBackend | None":
    """7d. EmbeddingBackend ファクトリ生成"""
    embedder = None
    try:
        from backend.free.rag.embedding_factory import create_embedding_backend

        embedder = create_embedding_backend(cfg, project_root, debug_logger=debug_logger)
        state.embedder = embedder
        logger.info(
            "EmbeddingBackend initialized: backend=%s, model=%s",
            embedder.backend_type(), embedder.model_name(),
        )
    except Exception as e:
        logger.warning("EmbeddingBackend init skipped: %s", e)
    return embedder


async def _check_embedding_dim(state: AppState, cfg: dict[str, Any]) -> None:
    """7d-1b. 埋め込み次元整合性チェック

    ``embedding.auto_reindex_on_mismatch=True`` のときは mismatch 検出後に
    ``run_reindex`` を呼んで自動再構築する (運用者の手動 ``evoref reindex``
    実行を不要にする)。Mismatch がない or auto reindex 無効のときは従来通り
    ``state.embedding_dim_mismatch`` フラグだけを立てる。
    """
    try:
        from backend.free.rag.dimension_check import check_embedding_dim_consistency
        mismatch = check_embedding_dim_consistency(state)
    except Exception as e:
        logger.warning("Embedding dimension check failed: %s", e)
        return

    # SemMem fact 埋め込みの stale 検知 (RAG とは別ストア)。embed swap 後の
    # 取り残しを reembed-facts へ誘導する。recall miss のみで誤結果は出ないため
    # ブロックはせず WARNING のみ。
    try:
        from backend.free.memory.semantic.stale_guard import (
            warn_if_semmem_reembed_required,
        )
        warn_if_semmem_reembed_required()
    except Exception as e:
        logger.debug("SemMem stale guard check skipped: %s", e)

    if not mismatch:
        return

    auto_reindex = bool(
        (cfg.get("embedding") or {}).get("auto_reindex_on_mismatch", False)
    )
    if not auto_reindex:
        logger.warning(
            "Embedding dimension mismatch detected. Run 'evoref reindex' to "
            "rebuild, or set embedding.auto_reindex_on_mismatch=true to "
            "auto-rebuild on next startup.",
        )
        return

    logger.warning(
        "Embedding dimension mismatch detected. "
        "auto_reindex_on_mismatch=true → running reindex now...",
    )
    try:
        from backend.free.rag.reindex import run_reindex
        result = await run_reindex(state)
        logger.info(
            "Auto reindex complete: rag=%d chunks, %d cartridges, %.2fs",
            result.rag_chunks, len(result.cartridges_rebuilt),
            result.elapsed_sec,
        )
    except Exception as e:
        logger.error(
            "Auto reindex failed: %s. RAG remains in degraded state. "
            "Run 'evoref reindex' manually.", e,
        )


def _build_bm25_index(
    vs: "VectorStore | None",
    cfg: dict[str, Any] | None = None,
) -> "BM25Retriever | None":
    """メインベクトルストアの BM25 索引を構築する。

    **1 インスタンスを全経路で共有する。** チャット応答経路の LTM
    ハイブリッド検索・sleep-time のプレフィックス再構築・``HybridRetriever``
    (ベンチ / 長文生成) が同じ索引を見ないと、片方だけ新しいチャンクを
    知っている状態が生まれる。

    BM25 パラメータ (k1/b/delta/trigram/ASCII split/stopword) を config から反映。
    ストアが空でもインスタンスは返す (後で sleep-time が張り直す)。
    """
    if vs is None:
        return None
    try:
        from backend.free.rag.bm25_retriever import (
            BM25Retriever,
            DEFAULT_STOPWORD_BIGRAMS,
            build_index_from_vector_store,
        )

        rag_cfg = (cfg or {}).get("rag", {}) if cfg else {}
        stop_cfg = rag_cfg.get("bm25_stopword_bigrams", None)
        if stop_cfg is None:
            stopwords: list[str] | None = list(DEFAULT_STOPWORD_BIGRAMS)
        else:
            # 明示空リストはストップワード無効化、非空なら指定リストを使用
            stopwords = list(stop_cfg)

        bm25 = BM25Retriever(
            k1=float(rag_cfg.get("bm25_k1", 1.5)),
            b=float(rag_cfg.get("bm25_b", 0.75)),
            delta=float(rag_cfg.get("bm25_delta", 1.0)),
            use_trigrams=bool(rag_cfg.get("bm25_use_trigrams", False)),
            split_ascii=bool(rag_cfg.get("bm25_split_ascii", True)),
            stopwords=stopwords,
        )
        n = build_index_from_vector_store(bm25, vs)
        logger.info("BM25 index initialized: %d chunks", n)
        return bm25
    except Exception as e:
        logger.warning("BM25 index init skipped: %s", e)
        return None


def _build_hybrid_retriever(
    vs: "VectorStore | None",
    embedder: "EmbeddingBackend | None",
    debug_logger: "DebugLogger",
    policy_interpreter: "PolicyInterpreter",
    cfg: dict[str, Any] | None = None,
    bm25: "BM25Retriever | None" = None,
) -> "HybridRetriever | None":
    """7d-2. HybridRetriever 構築（ベンチマーク・長文生成の unit 検索で使用）

    融合パラメータ (fusion_method/rrf_k/bm25_weight/vector_weight) を config から
    反映する。BM25 索引は :func:`_build_bm25_index` が作った共有インスタンスを
    受け取る (無ければ本関数内で作る)。
    """
    if not (vs and embedder):
        return None
    try:
        from backend.free.rag.retriever import HybridRetriever

        rag_cfg = (cfg or {}).get("rag", {}) if cfg else {}
        if bm25 is None:
            bm25 = _build_bm25_index(vs, cfg)
        if bm25 is None:
            return None

        hybrid_retriever = HybridRetriever(
            vector_store=vs,
            bm25_retriever=bm25,
            embedder=embedder,
            fusion_method=str(rag_cfg.get("fusion_method", "rrf")),
            rrf_k=int(rag_cfg.get("rrf_k", 60)),
            bm25_weight=float(rag_cfg.get("bm25_weight", 0.3)),
            vector_weight=float(rag_cfg.get("vector_weight", 0.7)),
            debug_logger=debug_logger,
            policy=policy_interpreter,
            config=cfg,
        )
        logger.info("HybridRetriever initialized for benchmark")
        return hybrid_retriever
    except Exception as e:
        logger.warning("HybridRetriever init skipped: %s", e)
        return None


def _init_lazy_contextual(
    state: AppState,
    cfg: dict[str, Any],
    vector_store: "VectorStore | None",
    embedder: "EmbeddingBackend | None",
) -> None:
    """LazyContextualPrefixService を初期化する

    ``rag.contextual_prefix.enabled=true`` かつ ``mode=lazy`` のときのみ
    構築する。それ以外はスキップし ``state.lazy_contextual`` は ``None``
    のままで、search_pipeline 側の hook は自動的に no-op 化される。
    ``aux_client`` / ``embedder`` / ``vector_store`` のいずれかが
    欠落していても no-op。
    """
    rag_cfg = cfg.get("rag", {}) or {}
    cp_cfg = rag_cfg.get("contextual_prefix", {}) or {}
    enabled = bool(cp_cfg.get("enabled", True))
    mode = str(cp_cfg.get("mode", "eager"))
    if not (enabled and mode == "lazy"):
        logger.info(
            "LazyContextualPrefixService not initialized "
            "(enabled=%s, mode=%s)", enabled, mode,
        )
        return
    if not (state.aux_client and embedder and vector_store):
        logger.info(
            "LazyContextualPrefixService init skipped: "
            "aux_client=%s, embedder=%s, vector_store=%s",
            state.aux_client is not None,
            embedder is not None,
            vector_store is not None,
        )
        return
    try:
        from backend.free.rag.contextual_prefix import ContextualPrefixGenerator
        from backend.free.rag.lazy_contextual import LazyContextualPrefixService

        generator = ContextualPrefixGenerator(state.aux_client, cfg)
        service = LazyContextualPrefixService(
            generator=generator,
            embedder=embedder,
            vector_store=vector_store,
            config=cfg,
            debug_logger=state.debug_logger,
        )
        state.lazy_contextual = service
        logger.info(
            "LazyContextualPrefixService initialized (mode=%s, threshold=%d, min_tokens=%d)",
            service.mode, service.lazy_hit_threshold, service.min_chunk_tokens,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("LazyContextualPrefixService init skipped: %s", e)


def _init_memory_threshold_calibration(
    resolver, cfg: dict[str, Any], short_term, embedder,
) -> None:
    """記憶検索の品質 3 閾値を、埋め込みモデルに合わせた較正値へ差し替える。

    ``rag.self_rag.threshold_mode`` が ``manual`` なら何もしない (config の静的値)。
    ``auto`` のときは指紋一致のキャッシュがあれば同期で適用し、無ければ代理クエリ
    の再埋め込みが要るためバックグラウンドタスクで較正する。較正が終わるまでは
    config の静的値へ縮退する (起動を遅らせない)。
    """
    from backend.free.rag.memory_threshold_calibration import (
        embedder_fingerprint,
        load_calibration,
        set_active_calibration,
    )

    rag_cfg = cfg.get("rag", {}) or {}
    mode = str((rag_cfg.get("self_rag") or {}).get("threshold_mode", "auto"))
    if mode != "auto":
        set_active_calibration(None)
        logger.info("Memory threshold calibration disabled (threshold_mode=manual)")
        return
    if embedder is None or short_term is None:
        logger.info(
            "Memory threshold calibration skipped: embedder=%s short_term=%s",
            embedder is not None, short_term is not None,
        )
        return

    try:
        fingerprint = embedder_fingerprint(embedder.model_name(), embedder.dim())
        memory_dir = resolver.resolve_local("memory_dir")
    except Exception as e:  # pragma: no cover - 起動は止めない
        logger.warning("Memory threshold calibration setup failed: %s", e)
        return

    cached = load_calibration(memory_dir, fingerprint)
    if cached:
        set_active_calibration(cached)
        logger.info(
            "Memory threshold calibration loaded from cache (%s): %s",
            fingerprint, cached,
        )
        return

    # 起動時に条件を満たさなくても諦めない。較正は **起動時 1 回きり** だった
    # ため、ノート 0 件で起動するとそのプロセスの生涯にわたって config の静的
    # 閾値が使われ続けていた (実測 2026-08-27: その後 STM が 105 件まで増えても
    # 再較正されず、Quality: low が 12/12。観測 top_score の最大 0.428 に対し
    # relevance_threshold 0.65 は到達不能)。sleep-time Full の末尾から
    # ``retry_pending_calibration()`` が拾えるよう、バインド済みの再試行口を
    # 登録しておく。
    from backend.free.rag.memory_threshold_calibration import (
        set_pending_recalibration,
    )

    set_pending_recalibration(
        lambda: _run_memory_threshold_calibration(
            memory_dir, fingerprint, short_term, embedder,
        ),
    )

    try:
        asyncio.create_task(
            _run_memory_threshold_calibration(
                memory_dir, fingerprint, short_term, embedder,
            ),
            name="memory_threshold_calibration",
        )
    except RuntimeError:  # イベントループ外 (同期テスト等) では単に見送る
        logger.info("Memory threshold calibration deferred: no running event loop")


async def _run_memory_threshold_calibration(
    memory_dir, fingerprint: str, short_term, embedder,
) -> None:
    """代理クエリを再埋め込みして閾値を較正し、キャッシュへ保存する。

    代理クエリは **保存済みのユーザー発話**。質問文と平叙文では埋め込みが
    非対称なので、ノート↔ノートで測ると系統的に高く出て閾値が過大になる。
    """
    import numpy as np

    from backend.free.rag.memory_threshold_calibration import (
        MIN_NOTES,
        MIN_QUERIES,
        compute_calibration,
        save_calibration,
        set_active_calibration,
    )

    try:
        notes = [n for n in short_term.notes.values() if n.embedding is not None]
        if len(notes) < MIN_NOTES:
            logger.info(
                "Memory threshold calibration skipped: only %d notes with "
                "embeddings (need %d); falling back to config thresholds",
                len(notes), MIN_NOTES,
            )
            return
        row_of = {note.id: i for i, note in enumerate(notes)}
        proxies = [n for n in notes if getattr(n, "source", "user") == "user"]
        if len(proxies) < MIN_QUERIES:
            logger.info(
                "Memory threshold calibration skipped: only %d user-source notes "
                "usable as proxy queries (need %d)", len(proxies), MIN_QUERIES,
            )
            return

        query_vecs = await embedder.embed(
            [n.content for n in proxies], is_query=True, mode="chat",
        )
        note_vecs = np.stack([n.embedding for n in notes])
        self_index = [row_of[n.id] for n in proxies]

        result = compute_calibration(
            note_vecs, np.asarray(query_vecs), self_index=self_index,
        )
        if not result.get("ok"):
            logger.info(
                "Memory threshold calibration not applied: %s", result.get("reason"),
            )
            return
        set_active_calibration(result["thresholds"])
        save_calibration(memory_dir, fingerprint, result)
        logger.info(
            "Memory threshold calibration applied (%s): %s",
            fingerprint, result["thresholds"],
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pragma: no cover - 較正失敗は縮退で吸収する
        logger.warning("Memory threshold calibration failed: %s", e)


def _init_judge_tracker(state: AppState) -> None:
    """Self-RAG quality_judge のセッション / クエリ単位カウンタを初期化する

    config の ``rag.self_rag.content_gate`` 設定読取は消費側で都度行うため、
    ここではトラッカーのみ生成する。enabled=False の場合でも tracker 自体は
    生成して問題ない (判定時に disabled skip)。
    """
    from backend.free.rag.judge_usage_tracker import JudgeUsageTracker
    state.judge_tracker = JudgeUsageTracker()
    # conflict_chat_judge のセッション内発火上限用に別インスタンスを生成
    # (RAG 判定とカウントを混在させない)。
    state.conflict_judge_tracker = JudgeUsageTracker()

    logger.info(
        "JudgeUsageTracker initialized (self_rag + conflict_chat_judge)",
    )


def _init_sleep_time_worker(
    state: AppState,
    cfg: dict[str, Any],
    sleep_scheduler: "SleepTimeScheduler",
    stm: "ShortTermMemory",
    ltm: Any,
    embedder: "EmbeddingBackend | None",
    vs: "VectorStore | None",
    exp_buf: "ExperienceBuffer",
    debug_logger: "DebugLogger",
    learned_patterns_store: "LearnedPatternStore",
    policy_interpreter: "PolicyInterpreter",
    aux_prompt_mgr: "AuxPromptManager",
    bm25_retriever: "BM25Retriever | None" = None,
) -> None:
    """7e. SleepTimeWorker（EmbeddingBackend + FadeMemScorer が必要）"""
    try:
        from backend.free.memory.pipeline.lightmem_scorer import FadeMemScorer
        from backend.free.memory.sleep_update import SleepTimeWorker

        if embedder is None:
            logger.warning("SleepTimeWorker init skipped: no embedder available")
            return

        scorer = FadeMemScorer(cfg, policy=policy_interpreter)

        # ── Step 8 Extractor 用配線 ──
        # ``state.get_semantic_store`` をプロバイダとして渡し、抽出器が
        # global / project スコープのストアを lazy に取得できるようにする。
        semantic_provider = state.get_semantic_store
        # 現在のプロジェクト ID は project_resolver で導出
        # 失敗しても sleep-time 全体を止めない (graceful degrade)。
        current_project_id: str | None = None
        try:
            from backend.free.memory.project_resolver import resolve_project_id
            from pathlib import Path as _Path
            current_project_id = resolve_project_id(_Path.cwd()).project_id
        except Exception as exc:
            logger.warning("Step 8 wiring: project_id resolution failed: %s", exc)

        # subject 正規化器も渡す (extractor 内で subject の表記ゆれを吸収)。
        subject_canonicalizer = None
        try:
            from backend.free.memory.notes.subject_canonicalizer import (
                build_default_canonicalizer,
            )
            from backend.config import get_path_resolver
            mem_cfg = (cfg or {}).get("memory", {}) or {}
            sd_cfg = mem_cfg.get("subject_dictionary", {}) or {}
            if bool(sd_cfg.get("enabled", True)):
                resolver = get_path_resolver()
                dict_file_rel = sd_cfg.get(
                    "file", "local/memory/semantic/subject_dictionary.json",
                )
                dict_path = resolver.root / dict_file_rel
                facts_cfg = mem_cfg.get("facts", {}) or {}
                bypass = facts_cfg.get(
                    "extraction_skip_subject_canonicalize_regex",
                    r"^(loop|learn|mem)\.",
                )
                subject_canonicalizer = build_default_canonicalizer(
                    dictionary_path=dict_path, bypass_regex=bypass,
                )
        except Exception as exc:
            logger.warning("Step 8 wiring: subject canonicalizer init failed: %s", exc)

        # agent_trace*.jsonl は AgentTraceStore が常設ディレクトリへ日付付き
        # ファイル名 (``agent_trace_YYYY-MM-DD.jsonl``) で出力する (develop
        # フラグ非依存)。ディレクトリそのものを Step 8 / MDPIngester に渡す。
        agent_trace_dir = _agent_trace_dir()

        # Step 10 でアーカイブ後にキャッシュ済 SemanticFactStore を破棄
        def _semantic_invalidator(scope: str) -> None:
            state._semantic_stores.pop(scope, None)

        worker = SleepTimeWorker(
            stm, ltm, embedder, scorer, cfg,
            experience_buf=exp_buf,
            debug_logger=debug_logger,
            learned_patterns=learned_patterns_store,
            vector_store=vs,
            cartridge_manager=state.cartridge_manager,
            policy=policy_interpreter,
            aux_prompt_manager=aux_prompt_mgr,
            semantic_store_provider=semantic_provider,
            current_project_id=current_project_id,
            agent_trace_dir=agent_trace_dir,
            subject_canonicalizer=subject_canonicalizer,
            semantic_store_invalidator=_semantic_invalidator,
        )
        # 新しく昇格したチャンクを語彙検索でも引けるよう、共有 BM25
        # インスタンスを worker に渡す (sleep-time が索引を張り直す)。
        if bm25_retriever is not None:
            worker.set_bm25_retriever(bm25_retriever)
        sleep_scheduler.set_worker(worker)
        logger.info(
            "SleepTimeWorker initialized (project_id=%s, agent_trace_dir=%s)",
            current_project_id, agent_trace_dir,
        )
    except Exception as e:
        logger.warning("SleepTimeWorker init skipped: %s", e)


def _init_learning_scheduler(
    state: AppState,
    cfg: dict[str, Any],
    exp_buf: "ExperienceBuffer",
    prompt_mgr: "SystemPromptManager",
    debug_logger: "DebugLogger",
    policy_interpreter: "PolicyInterpreter",
    learned_patterns_store: "LearnedPatternStore",
    resolver: Any,
) -> "LearningScheduler":
    """7f. LearningScheduler (Level 1/2) + Evolver / Critique / FewShot 接続"""
    from backend.free.learning.scheduler import LearningScheduler

    learning_scheduler = LearningScheduler(
        config=cfg,
        experience_buf=exp_buf,
        prompt_manager=prompt_mgr,
        debug_logger=debug_logger,
        policy=policy_interpreter,
        disabled=state.learning_disabled,
        resolver=resolver,
    )
    learning_scheduler.set_learned_patterns(learned_patterns_store)

    # 7f-1b. EmbedInstructionEvolver (Level 1 phase3, f_04 §4.2)
    # embedder が instruction-aware の場合のみ embed 検索指示プロンプトの進化が有効。
    if state.embedder is not None:
        learning_scheduler.set_embedder(state.embedder)
        # 候補 instruction の実測評価器を注入 (Learn→Gen は EmbedEvalProtocol の
        # duck-typing 注入で境界を保つ)。embedder + vector_store が揃い
        # instruction-aware のときのみ有効。未注入時は記録済み rag_top1_score 平均へ
        # degrade する。
        if (
            state.vector_store is not None
            and hasattr(state.embedder, "supports_instructions")
            and state.embedder.supports_instructions()
        ):
            from backend.free.rag.embed_instruction_eval import EmbedInstructionEval
            _emb_cfg = cfg.get("embedding", {}) or {}
            learning_scheduler.set_embed_eval(EmbedInstructionEval(
                embedder=state.embedder,
                vector_store=state.vector_store,
                query_template=_emb_cfg.get(
                    "query_template", "Instruct: {task}\nQuery: {query}",
                ),
            ))

    # 7f-1b'. base prompt 採用ゲートの実測評価器 (Level 1 phase1、f_04 §4.5)
    # 欠陥率 fitness は候補に無反応なので、採用は「現行 vs 最良候補を失敗ケースで
    # 再生成・採点した差」で決める。ベース未接続時は未注入 = 採用しない。
    # Learn→Gen は PromptEvalProtocol の duck-typing 注入で境界を保つ。
    if state.llm_client is not None:
        from backend.free.llm.prompt_candidate_eval import PromptCandidateEval
        learning_scheduler.set_prompt_eval(PromptCandidateEval(
            state.llm_client,
            config=cfg,
            debug_logger=debug_logger,
            should_abort=learning_scheduler.should_yield,
        ))

    # 7f-1c. GenerationParamEvolver (Level 1 phase6)
    # 進化したデルタは config.get_generation_params() が読む
    # ``local_paths.generation_deltas_file`` (base モデルパーティション配下) に
    # 永続化される (reader と同一の resolve_learning 経由)。
    from backend.free.learning.generation_param_evolver import GenerationParamEvolver

    gen_delta_file = resolver.resolve_learning("generation_deltas_file")
    learning_scheduler.set_generation_param_evolver(
        GenerationParamEvolver(delta_file=gen_delta_file),
    )

    # 7f-2. PolicyParamEvolver + ExplorationController
    # learning.policy.evolve_writeback=semmem の場合は
    # SemMem ストアを注入し、進化結果を policy ファクトとして書き戻す。
    # 旧 ``harness:`` セクションから ``learning.policy.*`` へ移行済
    from backend.free.learning.exploration_controller import ExplorationController
    from backend.free.learning.policy_evolver import PolicyParamEvolver

    learning_policy_cfg = ((cfg or {}).get("learning") or {}).get("policy") or {}
    evolve_writeback = learning_policy_cfg.get("evolve_writeback", "yaml")
    learn_view: Any | None = None
    semmem_writeback_scope: str = "global"
    if evolve_writeback == "semmem":
        try:
            from backend.free.memory.views.learn import LearnFactView

            global_store = state.get_semantic_store("global")
            stores: list[Any] = [global_store]
            writeback_store: Any = global_store
            semmem_writeback_scope = "global"
            try:
                from pathlib import Path as _Path

                from backend.free.memory.project_resolver import resolve_project_id
                project_id = resolve_project_id(_Path.cwd()).project_id
            except Exception as exc:
                logger.warning(
                    "policy_evolver semmem wiring: project_id resolution failed: %s",
                    exc,
                )
                project_id = None
            if project_id:
                project_store = state.get_semantic_store(f"project:{project_id}")
                stores.append(project_store)
                writeback_store = project_store
                semmem_writeback_scope = f"project:{project_id}"
            learn_view = LearnFactView(
                stores=stores, writeback_store=writeback_store,
            )
        except Exception as exc:
            logger.warning(
                "policy_evolver semmem wiring failed: %s "
                "(falling back to evolve_writeback=yaml)", exc,
            )
            learn_view = None
            semmem_writeback_scope = "global"
            evolve_writeback = "yaml"

    exploration_controller = ExplorationController(debug_logger=debug_logger)
    exploration_controller.load(prompt_mgr.prompt_dir / "exploration_state.json")
    policy_evolver = PolicyParamEvolver(
        policy=policy_interpreter,
        exploration=exploration_controller,
        debug_logger=debug_logger,
        learn_view=learn_view,
        semmem_writeback_scope=semmem_writeback_scope,
        evolve_writeback=evolve_writeback,
        # #8: proven delta 昇格の目標 confidence (PolicyInterpreter の適用閾値と一致)。
        activation_min_confidence=float(
            learning_policy_cfg.get("activation_min_confidence", 0.7),
        ),
        # base 学習パーティションの active モデル: policy 書き戻し subject へ
        # ``learn.policy.<model>.*`` としてモデル次元を埋め込む (partition 無効時は空)。
        base_model_id=state.active_base_model_slug,
    )
    # 旧 JSON ステートが残っていれば読み込みのみ行う (semmem モードでも
    # fitness 履歴の継続性を保つため)。書き出しは scheduler 側で
    # is_semmem_writeback_active() を見て分岐する。
    policy_evolver.load(prompt_mgr.prompt_dir / "policy_evolver_state.json")
    learning_scheduler.set_policy_param_evolver(policy_evolver)
    logger.info(
        "PolicyParamEvolver initialized (evolve_writeback=%s, scope=%s, view=%s)",
        evolve_writeback, semmem_writeback_scope, learn_view is not None,
    )

    # 7f-3. CritiqueSynthesizer
    # 批評はアイドル窓 (Level 1) の処理なのでベースモデルで回す。
    # purpose="critique_synthesis" は PURPOSE_SCHEMAS 登録済みなので
    # AuxClient が文法制約経路へ流す。
    from backend.free.learning.critique_synthesizer import CritiqueSynthesizer

    critique_client = learning_scheduler.resolve_idle_task_client(state.local_client)
    critique_synthesizer = CritiqueSynthesizer(
        llm_client=critique_client,
        debug_logger=debug_logger,
    )
    learning_scheduler.set_critique_synthesizer(critique_synthesizer)
    logger.info(
        "CritiqueSynthesizer initialized (client=%s)",
        type(critique_client).__name__ if critique_client else "none",
    )

    # 7f-4. FewShotPool
    # learning.policy.evolve_writeback=semmem
    # の場合は SemMem ストアを注入し、新規 example を ``learn.fewshot.<mode>.<id>`` の
    # fewshot ファクトとして書き戻す。bootstrap 時に SemMem 上の active
    # ファクトから in-memory プールを再構築する。
    # 旧 ``harness:`` セクションから ``learning.policy.*`` へ移行済
    from backend.free.learning.fewshot_pool import FewShotPool

    learning_cfg = cfg.get("learning", {})
    fewshot_pool = FewShotPool(
        pool_size=learning_cfg.get("fewshot_pool_size", 50),
        min_fitness=learning_cfg.get("fewshot_min_fitness", 0.7),
        min_quality_score=learning_cfg.get("fewshot_min_quality_score", 0.5),
        max_examples=learning_cfg.get("fewshot_max_examples", 3),
        diversity_threshold=learning_cfg.get("fewshot_diversity_threshold", 0.8),
        debug_logger=debug_logger,
        learn_view=learn_view,
        semmem_writeback_scope=semmem_writeback_scope,
        evolve_writeback=evolve_writeback,
        # active モデル: fewshot 書き戻し / bootstrap を ``learn.fewshot.<model>.*``
        # にスコープする (partition 無効時は空でレガシー全件)。
        base_model_id=state.active_base_model_slug,
    )
    if fewshot_pool.is_semmem_writeback_active():
        # SemMem を source of truth として in-memory プールを再構築。
        # 旧 JSON ステートが残っていても上書きされる。
        fewshot_pool.bootstrap_from_semmem()
    else:
        fewshot_pool.load(prompt_mgr.prompt_dir / "fewshot_pool.json")
    learning_scheduler.set_fewshot_pool(fewshot_pool)
    # 推論時 query 依存 few-shot 選択器を prompt_manager に注入 (Loop→Learn は
    # FewShotSelector Protocol + duck-typing で境界を保つ)。読み込み系のため
    # learning_disabled でも注入する (fewshot_pool 構築自体スキップされない)。
    prompt_mgr.set_fewshot_selector(
        fewshot_pool, k=learning_cfg.get("fewshot_max_examples", 3),
    )
    logger.info(
        "FewShotPool initialized (evolve_writeback=%s, scope=%s, total=%d)",
        fewshot_pool.evolve_writeback,
        semmem_writeback_scope,
        fewshot_pool.total_count,
    )
    # 手本の埋め込みを背景で張る。埋め込みが載るまでは従来の文字 bi-gram で
    # 選ぶので、起動を待たせる必要はない (tool_gate_warmup と同じ扱い)。
    # 永続化しない設計なので、毎起動ここで張り直す (pool_size 件だけ)。
    if state.embedder is not None:
        try:
            _track_background_task(
                state,
                fewshot_pool.backfill_embeddings(state.embedder),
                name="fewshot_embedding_warmup",
            )
        except RuntimeError:  # イベントループ外 (同期テスト等) では見送る
            logger.info("Fewshot embedding warmup deferred: no running event loop")

    # 7f-5. FeedbackPipe — 品質ゲート結果 → 学習サイクル 還流
    from backend.free.learning.feedback_pipe import FeedbackPipe

    feedback_cfg = (learning_cfg or {}).get("feedback_pipe", {}) or {}
    feedback_pipe = FeedbackPipe(
        feedback_cfg,
        learn_view=learn_view,
        writeback_scope=semmem_writeback_scope,
        fewshot_pool=fewshot_pool,
        critique_synthesizer=critique_synthesizer,
        debug_logger=debug_logger,
    )
    # critique policy の subject を PolicyInterpreter が読む ``learn.policy.<model>.`` 前置に揃える
    feedback_pipe.set_base_model_id(getattr(state, "active_base_model_slug", "") or "")
    learning_scheduler.set_feedback_pipe(feedback_pipe)
    logger.info(
        "FeedbackPipe initialized (enabled=%s, weight=%.2f)",
        feedback_pipe.enabled,
        feedback_pipe.weight_semmem_success,
    )

    state.learning_scheduler = learning_scheduler
    return learning_scheduler


def _wire_sleep_scheduler_models(
    state: AppState,
    sleep_scheduler: "SleepTimeScheduler",
    learning_scheduler: "LearningScheduler",
    resolver: Any,
) -> None:
    """7g (前半). SleepTimeScheduler に LLM クライアント / モデルパスを設定"""
    # sleep-time / Level 1 の LLM ステージはベースモデルで回す。
    # is_user_active() の協調 yield 判定は chat_in_flight()/in_flight_chat_count を
    # 持つ LLMClient ファサード (state.llm_client) 前提。生の LocalClient
    # (state.local_client) にはこの属性が無くgetattrデフォルト(0)に落ちるため、
    # フォアグラウンド生成中でも Level1/Level2 の yield ゲートが機能しなかった
    # (2026-07-24 実機検証で判明、staged create 実行中も Level2 real-eval が
    # 抑止されず本番 llama-server とリソース競合していた)。
    if state.llm_client:
        sleep_scheduler.set_llm_client(state.llm_client)

    sleep_scheduler.set_learning_scheduler(learning_scheduler)

    # LoRA パスは不在でも常に設定する: Level 2 初回 full-train (bootstrap) が
    # この位置に初期アダプタを書き込むため、存在判定は trainer 側で行う。
    # resolve_learning 経由に統一 (Issue #251 修正): 従来 resolve_local (flat)
    # で構築していたため、partition_by_base_model=true 環境でもここだけモデル
    # パーティションから孤立していた。mode は明示せず resolver.active_mode
    # (起動時既定 "chat") に委ねる — レガシー "model" スキームでは mode を無視して
    # 従来と同じ flat 相当のパスを返すため、この変更で挙動が変わるのは
    # partition_by_base_model=true のユーザーのみ (元々 partition される "べき"
    # だった箇所の是正)。
    lora_path = resolver.resolve_learning("lora_adapter")
    sleep_scheduler.set_lora_path(lora_path)

    base_model_path = resolver.resolve_model("base_model")
    if base_model_path.exists():
        sleep_scheduler.set_base_model_path(base_model_path)

    # Level 2 (base) の version_manager / eval_core_manager を running scheduler に
    # 後注入する (Pro 限定、未注入だと _check_and_run_base が None で停止する)。
    from backend.edition import get_pro_handler

    vm_cls = get_pro_handler("lora_version_manager")
    if vm_cls is not None:
        try:
            versions_dir = resolver.resolve_learning("lora_versions_dir")
            learning_scheduler.set_version_manager(vm_cls(versions_dir, lora_path))
        except Exception as e:
            logger.debug("Base LoRA version_manager injection skipped: %s", e)

    ecm_cls = get_pro_handler("eval_core_manager")
    if ecm_cls is not None:
        try:
            eval_core_path = resolver.resolve_learning("eval_core_file")
            learning_scheduler.set_eval_core_manager(ecm_cls(eval_core_path))
        except Exception as e:
            logger.debug("eval_core_manager injection skipped: %s", e)


def _inject_level2_runner(
    learning_scheduler: "LearningScheduler", debug_logger: "DebugLogger",
) -> None:
    """7h. Level2Runner 注入（Pro）"""
    try:
        from backend.pro.learning.level2_trainer import Level2Runner
        runner = Level2Runner(learning_scheduler, debug_logger=debug_logger)
        learning_scheduler.set_level2_runner(runner)
        logger.info("Level 2 runner injected into LearningScheduler")
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Level 2 runner injection failed: %s", e)


async def _inject_learn_components(
    state: AppState,
    cfg: dict[str, Any],
    resolver: Any,
    project_root: Path,
    learn_pillar: "LearnPillar",
) -> None:
    """7i. Pro Learn pillar setup を呼び出し、Pro 学習コンポーネントを注入する。

    (pro_pillar_setup("learn")) に全面委譲し、本関数は pillar 構築の薄い
    ラッパに縮退した。結果は ``state.pro.learn`` へ載せるため戻り値は持たない。
    """
    from backend.edition import get_pro_pillar_setup
    from backend.pillars import ProGenPillar, ProState

    pro_learn_setup = get_pro_pillar_setup("learn")
    if pro_learn_setup is None:
        return None

    pro_gen = state.pro.gen if state.pro is not None else None
    if pro_gen is None:
        pro_gen = ProGenPillar()

    try:
        pro_learn_pillar = await pro_learn_setup(
            cfg, resolver, state, learn_pillar, pro_gen,
            project_root=project_root,
        )
    except Exception as e:
        logger.warning("Pro Learn pillar setup failed: %s", e)
        return None

    if state.pro is None:
        state.pro = ProState(learn=pro_learn_pillar)
    else:
        state.pro.learn = pro_learn_pillar
    return None


def _start_level1_loop(
    sleep_scheduler: "SleepTimeScheduler", learning_scheduler: "LearningScheduler",
) -> None:
    """7l. Level 1 独立常駐ループの起動"""
    # 起動時に SUSPENDED な session があれば次の tick で resume される
    if learning_scheduler.has_active_session():
        logger.info(
            "Level 1 SUSPENDED session detected at startup; "
            "will resume on next loop tick",
        )
    sleep_scheduler.start_level1_loop()
    logger.info("Level 1 independent loop started")
    # Level 2 も再起動耐性のある独立常駐ループで起動する (overdue/idle 発火)。
    sleep_scheduler.start_level2_loop()
    logger.info("Level 2 independent loop started")
    logger.info("Learning cycle initialized")


def _init_tools(
    state: AppState,
    cfg: dict[str, Any],
    client: "LocalClient | None",
    learned_patterns_store: "LearnedPatternStore",
    embedder: "EmbeddingBackend | None" = None,
) -> None:
    """7j. ToolsRegistry + ToolCallJudge + ReactiveAgent 初期化（3層エージェントディスパッチ用）"""
    from backend.free.agent.tools_registry import ToolsRegistry
    from backend.free.agent.tools.builtin import register_builtin_tools
    from backend.free.agent.tool_call_judge import ToolCallJudge
    from backend.free.agent.reactive import ReactiveAgent
    from backend.free.history.history_manager import get_history_manager

    tools_reg = ToolsRegistry()
    # aux_client は Gen pillar 構築時 (_init_aux_client) に state へ載っている。
    # base 未接続なら None で、LLM 委譲ツールは LocalClient 直呼びへ縮退する。
    register_builtin_tools(
        tools_reg, cfg, client,
        history_manager=get_history_manager(),
        aux_client=state.aux_client,
    )

    # Pro 拡張ツール: ``register_pro_tools`` ハンドラが
    # 登録されていれば呼び出してレジストリを拡張する。Free 版では未登録のため
    # no-op。Pro 側 (backend/pro/__init__.py::setup_pro) から登録される。
    from backend.edition import get_pro_handler
    pro_register_tools = get_pro_handler("register_pro_tools")
    if callable(pro_register_tools):
        try:
            pro_register_tools(tools_reg, state, cfg)
        except Exception as e:
            logger.warning("Pro tools registration failed: %s", e)

    state.tools_registry = tools_reg
    logger.info("ToolsRegistry initialized: %d tools", tools_reg.count)

    # URL リコール用 MemFactView (global scope) — チャット応答時に
    # ``mem.world.url.*`` を読み取って fetch_url 補完するために使う。
    # global store が取得できない (early init / disabled) 場合は None。
    mem_view = None
    try:
        global_store = state.get_semantic_store("global")
        if global_store is not None:
            from backend.free.memory.views.mem import MemFactView
            mem_view = MemFactView(global_store)
    except Exception as exc:
        logger.warning("URL recall mem_view init skipped: %s", exc)

    tool_judge = ToolCallJudge(
        config=cfg,
        cartridge_manager=state.cartridge_manager,
        learned_patterns=learned_patterns_store,
        debug_logger=state.debug_logger,
        mem_view=mem_view,
        embedder=embedder,
        # ベースモデルの文法制約ツール分類用 (docs/c_14 §1.3)。
        # ``generate_constrained`` を持つ LocalClient を直接渡す
        # (LLMClient ファサードは response_format を透過しない)。
        llm_client=state.local_client,
    )
    state.tool_call_judge = tool_judge
    logger.info("ToolCallJudge initialized: enabled=%s", tool_judge.enabled)

    # ツール要否ゲートの exemplar 埋め込み (実測 ~5.6 秒 / 137 件)。起動を
    # 待たせないので背景タスクにし、完了までは従来の正規表現ゲートへ縮退する。
    try:
        _track_background_task(
            state, tool_judge.warmup_tool_gate(), name="tool_gate_warmup",
        )
    except RuntimeError:  # イベントループ外 (同期テスト等) では見送る
        logger.info("Tool gate warmup deferred: no running event loop")

    # Reactive 層を常駐化 (リクエスト毎生成だと LRU キャッシュが温まらない)。
    # 既定 cache (100 件 / TTL 300s) のまま。LLM 非依存なので構築コストはほぼゼロ。
    state.reactive_agent = ReactiveAgent()
    logger.info("ReactiveAgent initialized (resident)")


def _agent_trace_dir() -> Path | None:
    """MDP トレース常設ストアのディレクトリ (``local_paths.agent_trace_dir``)。"""
    try:
        from backend.config import get_path_resolver
        return get_path_resolver().resolve_local("agent_trace_dir")
    except Exception as exc:
        logger.warning("agent_trace_dir unresolved, MDP trace store disabled: %s", exc)
        return None


def _init_agent_tracer(state: AppState, debug_logger: "DebugLogger") -> None:
    """7k. AgentTracer 初期化（MDP トレース: 常設ストア + develop 時の JSONL）"""
    from backend.free.agent.agent_tracer import AgentTracer
    from backend.free.agent.agent_trace_store import AgentTraceStore

    trace_dir = _agent_trace_dir()
    trace_store = AgentTraceStore(trace_dir) if trace_dir is not None else None
    agent_tracer = AgentTracer(debug_logger=debug_logger, trace_store=trace_store)
    state.agent_tracer = agent_tracer
    logger.info("AgentTracer initialized (trace_dir=%s)", trace_dir)


def _init_loop_driver(
    state: AppState,
    cfg: dict[str, Any],
    project_root: Path,
    aux_client: "AuxClient | None",
) -> None:
    """7m. LoopDriver + TaskExecutor 初期化

    - ``loop.enabled=False`` の場合は driver を作らずに終了
    - ``loop.executor=noop`` なら ``NoOpExecutor``、``ralph`` なら
      ``RalphExecutor`` を構築して DI する
    - ``RalphExecutor`` には ``DefaultHarness`` + ``ActionRunner`` +
      PolicyInterpreter.get_all を差し込む
    - 起動 bootstrap は既存の ``bootstrap_loop_context_at_startup`` に任せる
    """
    loop_cfg = (cfg or {}).get("loop", {}) or {}
    if not loop_cfg.get("enabled", True):
        logger.info("LoopDriver: disabled by config (loop.enabled=false)")
        return
    from backend.free.harness.base import DefaultHarness
    from backend.free.loop.action_runner import (
        ActionRunnerError,
        build_action_runner_config,
        ActionRunner,
    )
    from backend.free.loop.driver import LoopDriver
    from backend.free.loop.executor import NoOpExecutor
    from backend.free.loop.quality_gate import build_default_gates
    from backend.free.loop.ralph_executor import RalphExecutor
    from backend.free.memory.views.harness import HarnessFactView
    from backend.free.memory.views.loop import LoopFactView
    from backend.schemas import LoopQualityGatesConfig

    executor_kind = str(loop_cfg.get("executor", "ralph")).lower()

    # ActionRunner / QualityGate は Ralph 構築時のみ必要
    executor: object | None = None
    if executor_kind == "noop":
        executor = NoOpExecutor()
    else:
        try:
            ar_cfg = build_action_runner_config(loop_cfg.get("sandbox") or {})
            action_runner = ActionRunner(config=ar_cfg, repo_root=project_root)
        except ActionRunnerError as exc:
            logger.warning(
                "LoopDriver: ActionRunner config invalid: %s — loop disabled",
                exc,
            )
            return

        gates_raw = loop_cfg.get("quality_gates") or {}
        try:
            gates_cfg = LoopQualityGatesConfig(**gates_raw)
        except Exception as exc:
            logger.warning(
                "LoopDriver: quality_gates config invalid: %s (using defaults)",
                exc,
            )
            gates_cfg = LoopQualityGatesConfig()
        gates = build_default_gates(
            gates_cfg, repo_root=project_root,
        )

        def _policy_provider(mode: str) -> dict[str, object] | None:
            pi = state.policy_interpreter
            if pi is None:
                return None
            try:
                return pi.get_all(mode)
            except Exception:
                return None

        # Harness は HarnessFactView (read-only) で SemMem を参照する
        stores: list = [state.get_semantic_store("global")]
        current_project = state.current_project_id
        if current_project:
            stores.append(state.get_semantic_store(f"project:{current_project}"))
        harness_view = HarnessFactView(stores=stores)

        harness = DefaultHarness(
            harness_view=harness_view,
            mode="create",
            policy_provider=_policy_provider,
        )
        executor = RalphExecutor(
            harness=harness,
            action_runner=action_runner,
            aux_client=aux_client,
            quality_gates=gates,
            max_actions_per_task=int(loop_cfg.get("max_actions_per_task", 10)),
            policy_provider=_policy_provider,
        )

    def _view_provider(project_id: str) -> LoopFactView:
        """project_id ごとに ``LoopFactView`` を生成する

        ``stores=[global, project]`` + ``writeback_store=project`` の標準構成。
        """
        global_store = state.get_semantic_store("global")
        project_store = state.get_semantic_store(f"project:{project_id}")
        return LoopFactView(
            stores=[global_store, project_store],
            writeback_store=project_store,
        )

    # artifact_hook を結線 — ラルフループの成果物を SemMem の project
    # スコープへ即時書き込み (冪等、重複スキップ)
    from backend.free.loop.artifact_writer import make_loop_artifact_hook
    artifact_hook = make_loop_artifact_hook(_view_provider)

    from backend.free.loop.events import LoopEventBus
    event_bus = LoopEventBus(
        max_queue_size=int(loop_cfg.get("event_bus_max_queue", 128)),
    )

    def _build_sleep_time_hook(app_state: AppState):
        """`loop.sleep_time_every_n` 反復ごとに sleep-time Light を回すフック。

        ``sleep_scheduler`` が未構築 (degraded / ``--no-learning``) の場合は
        ``None`` を返し、LoopDriver 側でフック無効として扱われる。
        """
        scheduler = getattr(app_state, "sleep_scheduler", None)
        if scheduler is None:
            return None

        async def _hook() -> None:
            await scheduler.run_light_now()

        return _hook

    driver = LoopDriver(
        view_provider=_view_provider,
        executor=executor,  # type: ignore[arg-type]
        max_iterations=int(loop_cfg.get("max_iterations", 50)),
        max_wall_time_sec=float(loop_cfg.get("max_wall_time_sec", 1800.0)),
        max_consecutive_failures=int(
            loop_cfg.get("max_consecutive_failures", 3),
        ),
        tick_interval_sec=float(loop_cfg.get("tick_interval_sec", 0.0)),
        sleep_time_every_n=int(loop_cfg.get("sleep_time_every_n", 5)),
        sleep_time_hook=_build_sleep_time_hook(state),
        on_gate_fail=str(loop_cfg.get("on_gate_fail", "retry")),
        retry_limit_per_task=int(loop_cfg.get("retry_limit_per_task", 2)),
        artifact_hook=artifact_hook,
        event_bus=event_bus,
        # に記録 (decision_point=``loop_continue_or_abort`` / ``quality_gate_action``)
        debug_logger=state.debug_logger,
    )
    state.loop_driver = driver  # type: ignore[attr-defined]
    logger.info(
        "LoopDriver initialized: executor=%s max_iter=%d",
        executor_kind, driver._max_iterations,  # type: ignore[attr-defined]
    )


def _init_theme_manager(state: AppState, cfg: dict[str, Any], resolver: Any) -> None:
    """8. テーママネージャ初期化"""
    from backend.free.themes.theme_service import ThemeManager

    themes_dir = resolver.resolve_local("themes_dir")
    theme_mgr = ThemeManager(themes_dir, cfg)
    state.theme_manager = theme_mgr
    logger.info("ThemeManager initialized: themes_dir=%s", themes_dir)


def _served_model_filename(state: Any) -> str:
    """llama-server が **実際にロードしているモデル**のファイル名を返す。

    ``/props`` 由来の ``ModelMetadata.model_id`` (model_alias → model_path →
    default_generation_settings.model の順で解決済み) を basename 化する。
    取得できない (degraded / 未接続) 場合は空文字。
    """
    client = getattr(state, "local_client", None) if state is not None else None
    model_id = getattr(getattr(client, "metadata", None), "model_id", "") or ""
    return Path(model_id).name if model_id else ""


def _validate_served_model(
    cfg: dict[str, Any], state: Any, expected_filename: str,
) -> None:
    """9a. 実際に serve 中のモデルが config / model_state と一致するか検証する。

    モデル移行 (`/migrate-model`) は ``model_state.json`` と学習パーティションを
    切り替えるが、**稼働中の llama-server は差し替えない** (再起動が要る)。
    2026-08-12 の調査では、この乖離に気付かないまま **62.8 時間**別モデルが
    serve され、学習パーティションが別モデルの出力で汚染されていた。config と
    ``model_state.json`` が一致していても発生するため、3 者目として ``/props``
    を照合する。

    起動をブロックはしない (再起動すれば解消する運用上の乖離であり、degraded で
    ``model_id`` が取れないケースもあるため)。WARNING + ``AppState`` へ記録する。
    """
    served = _served_model_filename(state)
    if not served or not expected_filename or served == expected_filename:
        return

    recommendation = (
        "Restart llama-server (scripts/evoref-ctl.bat stop && start) so the "
        "served model matches the configured one. Until then, chat responses "
        "and any learning written to the active partition come from "
        f"'{served}', not '{expected_filename}'."
    )
    logger.warning(
        "Served model mismatch: llama-server /props=%s, expected=%s. %s",
        served, expected_filename, recommendation,
    )
    if state is not None:
        state.served_model_mismatch = {
            "served_filename": served,
            "expected_filename": expected_filename,
            "recommendation": recommendation,
        }


def _validate_model_state(
    cfg: dict[str, Any], resolver: Any, state: Any = None,
) -> None:
    """9. model_state.json 検証

    config.yaml の base_model と model_state.json の current_filename を比較し、
    不一致があれば ERROR ログに記録する。さらに `AppState.model_state_mismatch`
    に詳細を保持し、API (`GET /api/model/state`) 経由で UI / CLI が参照できる
    ようにする。`model_migration.strict_startup_check` が true のときは
    RuntimeError を送出して起動をブロックする。

    加えて **実際に serve 中のモデル** (`/props`) を第 3 の照合先として見る
    (`_validate_served_model`)。config と model_state が一致していても、
    llama-server が古いモデルを掴んだままのことがある。
    """
    try:
        from backend.free.core.model_migration import ModelState

        model_state_path = resolver.resolve_local("model_state_file")
        ms = ModelState(model_state_path)
        if not ms.current_filename:
            ms.initialize_from_config(cfg)

        config_base_model = cfg.get("model_paths", {}).get("base_model") or ""
        config_filename = Path(config_base_model).name if config_base_model else ""

        # config が意図するモデルを基準に、実際の serve 状態を照合する
        # (config 未設定なら model_state を基準にする)。
        _validate_served_model(cfg, state, config_filename or ms.current_filename)

        mismatch = (
            ms.current_filename
            and config_filename
            and ms.current_filename != config_filename
        )
        if not mismatch:
            return

        recommendation = (
            f"Run '/migrate-model --new-model {config_base_model}' (or POST "
            f"/api/model/migrate) to safely align model_state.json with "
            f"config.yaml. Alternatively, revert config.yaml model_paths."
            f"base_model to '{ms.current_filename}'. Skipping migration "
            f"risks invalidating LoRA adapters and Level 1/2 evolution "
            f"parameters learned for the previous base model."
        )
        logger.error(
            "Model mismatch detected: model_state.json=%s, config.yaml=%s. %s",
            ms.current_filename, config_filename, recommendation,
        )

        info = {
            "current_filename": ms.current_filename,
            "config_filename": config_filename,
            "config_base_model": config_base_model,
            "recommendation": recommendation,
        }
        if state is not None:
            state.model_state_mismatch = info

        migration_cfg = cfg.get("model_migration", {})
        auto_migrate = bool(migration_cfg.get("auto_migrate_on_startup", False))
        strict = bool(migration_cfg.get("strict_startup_check", False))

        if auto_migrate and state is not None:
            try:
                _auto_migrate_base_model(
                    cfg, resolver, state, ms,
                    new_model_path=config_base_model,
                    old_filename=ms.current_filename,
                )
                state.model_state_mismatch = None
                return
            except Exception as exc:
                logger.error(
                    "Auto-migrate on startup failed: %s", exc, exc_info=True,
                )
                if not strict:
                    return
                raise RuntimeError(
                    "Startup blocked: auto-migrate failed and "
                    "strict_startup_check is enabled. "
                    f"model_state.json={ms.current_filename} vs "
                    f"config.yaml={config_filename}. Error: {exc}",
                ) from exc

        if strict:
            raise RuntimeError(
                "Startup blocked by model_migration.strict_startup_check: "
                f"model_state.json={ms.current_filename} vs "
                f"config.yaml={config_filename}. {recommendation}",
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.debug("model_state.json validation skipped: %s", e)


def _validate_component_model_state(
    cfg: dict[str, Any], resolver: Any, state: Any = None,
) -> None:
    """9b. component (embed) の model_state 検証

    base は `_validate_model_state` が担当。本関数は component の
    config↔model_state 不一致を ERROR ログ + `AppState.component_state_mismatches`
    に surface する (現状 component は無検証で desync しても黙るため)。
    component は migrate ボタン経由なら同期されるが、config 直書きや手動編集で
    desync しうる。`model_migration.strict_startup_check` が true なら起動をブロック。
    """
    try:
        from backend.free.core.model_migration import (
            ModelState,
            detect_mismatches,
        )

        ms = ModelState(resolver.resolve_local("model_state_file"))
        if not ms.current_filename:
            ms.initialize_from_config(cfg)

        mismatches = detect_mismatches(ms, cfg)
        component_mm = {k: v for k, v in mismatches.items() if k != "base_model"}
        if not component_mm:
            return

        for cfg_key, mm in component_mm.items():
            logger.error(
                "Component model mismatch: %s model_state.json=%s, config.yaml=%s. "
                "Run the component migrate (POST /api/model/<component>/migrate) or "
                "revert config.yaml model_paths.%s to '%s'.",
                cfg_key, mm["model_state"], mm["config"], cfg_key, mm["model_state"],
            )
        if state is not None:
            state.component_state_mismatches = component_mm

        if bool(cfg.get("model_migration", {}).get("strict_startup_check", False)):
            raise RuntimeError(
                "Startup blocked by model_migration.strict_startup_check: "
                f"component model mismatch {sorted(component_mm)}",
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.debug("component model_state validation skipped: %s", e)


def _auto_migrate_base_model(
    cfg: dict[str, Any],
    resolver: Any,
    state: Any,
    model_state: Any,
    *,
    new_model_path: str,
    old_filename: str,
) -> None:
    """起動時 auto-migrate

    `_validate_model_state` から呼ばれる。既に初期化済みの AppState から
    prompt_manager / experience_buf / learning_scheduler / stm を取り出して
    `ModelMigrator.migrate()` を実行する。
    """
    from backend.edition import get_pro_handler
    from backend.free.core.model_migration import ModelMigrator

    project_root = resolver.root

    experience_buf = None
    fc = getattr(state, "feedback_collector", None)
    if fc is not None:
        experience_buf = getattr(fc, "buffer", None)

    prompt_manager = getattr(state, "prompt_manager", None)
    learning_scheduler = getattr(state, "learning_scheduler", None)

    eval_core_mgr = None
    EvalCoreManager = get_pro_handler("eval_core_manager")
    if EvalCoreManager is not None:
        try:
            eval_core_path = resolver.resolve_learning("eval_core_file")
            eval_core_mgr = EvalCoreManager(eval_core_path)
        except Exception as exc:
            logger.debug("eval_core_manager unavailable during auto-migrate: %s", exc)

    stm = None
    get_memory_system = getattr(state, "get_memory_system", None)
    if callable(get_memory_system):
        mem = get_memory_system()
        if mem:
            _, stm, _ = mem

    logger.warning(
        "Auto-migrate on startup: %s -> %s. "
        "Restart llama-server to unload stale LoRA/model.",
        old_filename, Path(new_model_path).name,
    )
    migrator = ModelMigrator(
        config=cfg,
        project_root=project_root,
        model_state=model_state,
        experience_buf=experience_buf,
        prompt_manager=prompt_manager,
        eval_core_manager=eval_core_mgr,
        learning_scheduler=learning_scheduler,
        short_term_memory=stm,
    )
    result = migrator.migrate(
        new_model_path=new_model_path,
        try_lora=False,
        regenerate_context=False,
        dry_run=False,
    )
    logger.info(
        "Auto-migrate on startup completed: %s -> %s (lora=%s)",
        result.old_model, result.new_model, result.lora_action,
    )


def _check_edition_downgrade(resolver: Any) -> None:
    """10. エディションダウングレード検出"""
    from backend.edition import check_downgrade
    local_dir = resolver.resolve_local("memory_dir").parent  # local/
    check_downgrade(local_dir)


# ──────────────────────────────────────────────────────────────────────────
# Pillar wiring 本体
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _LifespanContext:
    """`lifespan` の startup → shutdown 間で持ち回される shutdown 必要オブジェクト。

    各フィールドは shutdown フェーズで `_shutdown_*` ヘルパーへ渡される。
    """

    sleep_scheduler: Any
    learning_scheduler: Any
    wm: Any  # WorkingMemoryRegistry (全セッションを shutdown で drain する)
    stm: Any
    resolver: Any
    exp_buf: Any
    exp_file: Any
    learned_patterns_store: Any
    patterns_file: Any
    pro_shutdown: Any
    instance_name: str
    # Develop エディション起動時のみ非 None
    # (`backend.develop.setup_develop` が `register_develop_shutdown` を
    # 呼んだ場合)。Pro と並列に shutdown フェーズで呼び出される。
    develop_shutdown: Any = None
    # develop=evolve 時のみ非 None。LogIngestor + PolicyAdjuster
    # + bridge bg task の 3 つを shutdown フェーズで cleanup する。
    log_ingestor: Any = None
    policy_adjuster: Any = None
    log_ingestor_bridge_task: Any = None


@dataclass
class _BaseContext:
    """``wire_pillars`` の base フェーズ成果物 (pillar 間の受け渡し用)。"""

    cfg: dict[str, Any]
    debug_logger: "DebugLogger"
    resolver: Any
    policy_interpreter: "PolicyInterpreter"


def _log_gpu_cpu_placement(cfg: dict[str, Any], project_root: Path) -> None:
    """llama-server 4 モデルの GPU/CPU 配置と推定 VRAM 合計を INFO に出力する

    `scripts/launch_llama.py` の推定ロジックを流用し、ベース / 補助タスク /
    埋め込み / リランカーの ``-ngl`` と GGUF ファイルサイズから使用 VRAM を
    見積もる。``runtime.total_vram_budget_mb`` が設定されている場合は予算と
    合算値を比較し、超過時に WARNING ログを出力する (起動はブロックしない。
    ブロック挙動は launch_llama.py --all 側に限定)。
    """
    try:
        import importlib.util

        launch_py = project_root / "scripts" / "launch_llama.py"
        if not launch_py.exists():
            logger.debug("GPU/CPU placement log skipped: %s not found", launch_py)
            return
        spec = importlib.util.spec_from_file_location("_launch_llama_probe", launch_py)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # バックエンド起動時は subprocess オーバーヘッドを避けるため Tier 2
        # (GGUF サイズベース) のみ使う。Tier 1 (llama-fit-params)
        # は ``scripts/launch_llama.py --all`` 側でのみ実行される。
        estimates = mod.estimate_vram_usage_mb(
            cfg, project_root, prefer_fit_params=False,
        )
        summary_lines = mod.format_placement_summary(estimates)
        total_vram_mb = sum(e.get("vram_mb", 0) for e in estimates.values())
        budget_mb = (cfg.get("runtime") or {}).get("total_vram_budget_mb")

        logger.info("GPU/CPU placement (llama-server components):")
        for line in summary_lines:
            logger.info(line.rstrip())
        if budget_mb is None:
            logger.info(
                "  total estimated VRAM: %d MB (runtime.total_vram_budget_mb not set)",
                total_vram_mb,
            )
        else:
            if total_vram_mb > int(budget_mb):
                logger.warning(
                    "  total estimated VRAM %d MB exceeds runtime.total_vram_budget_mb "
                    "(%d MB). Run `python scripts/launch_llama.py --all` to re-check, "
                    "or set embedding.gpu_layers to 0 (CPU fallback).",
                    total_vram_mb, int(budget_mb),
                )
            else:
                logger.info(
                    "  total estimated VRAM: %d MB / budget %d MB (OK)",
                    total_vram_mb, int(budget_mb),
                )
    except Exception as exc:
        logger.debug("GPU/CPU placement log failed: %s", exc)


async def _build_base_context(
    state: AppState, project_root: Path, timings: dict[str, float],
) -> _BaseContext:
    """横断基盤の初期化 (config / logging / i18n / local_dirs / llama_manager)。

    pillar 構築の前に必要な前提 (cfg / debug_logger / resolver / policy) を
    揃える。本フェーズは全 pillar が依存する「pillar 外」の共有基盤であり、
    CLAUDE.md §8 でいう横断基盤 (config / debug_logger / trace_context / i18n /
    edition / app_factory) に相当する。
    """
    from backend.factory._bootstrap import (
        _init_config,
        _init_i18n,
        _init_local_dirs,
        _init_logging,
    )

    with _timed(timings, "config"):
        cfg = _init_config(project_root)
    with _timed(timings, "logging"):
        debug_logger = _init_logging(state, cfg, project_root)
    with _timed(timings, "i18n"):
        _init_i18n(cfg)
    with _timed(timings, "local_dirs"):
        resolver, policy_interpreter = _init_local_dirs(state, project_root, cfg)

    # GPU/CPU 配置サマリ。設定駆動の情報なので logging 完了後すぐに出す
    with _timed(timings, "gpu_cpu_placement"):
        _log_gpu_cpu_placement(cfg, project_root)

    # LlamaProcessManager を early に作る (opt-in 動作)
    with _timed(timings, "llama_manager"):
        from backend.free.core.llama_process_manager import LlamaProcessManager
        pm_cfg = (cfg.get("process_manager") or {})
        state.llama_manager = LlamaProcessManager(
            project_root,
            health_timeout=int(pm_cfg.get("health_timeout", 120)),
            stop_timeout=int(pm_cfg.get("stop_timeout", 10)),
        )

    return _BaseContext(
        cfg=cfg,
        debug_logger=debug_logger,
        resolver=resolver,
        policy_interpreter=policy_interpreter,
    )


async def _build_gen_pillar(
    state: AppState,
    base: _BaseContext,
    project_root: Path,
    timings: dict[str, float],
) -> tuple["GenPillar", ProShutdownHook, DevelopShutdownHook]:
    """EvorefGen pillar の core 部分 (LLM / 埋め込み) を構築する。

    並列 I/O (llama_server / pro_gen_pillar / develop_gen_pillar /
    embedding) で TaskGroup 内でまとめて初期化し、LLMClient ファサードを
    組み立てる。Pro エディション起動時は ``setup_pro_gen`` が ``state.pro.gen`` を、
    Develop エディション起動時は ``setup_develop_gen`` が ``state.develop.gen`` を
    設定する (Develop はスケルトン段階で通常 no-op)。

    Returns:
        (gen_pillar, pro_shutdown_hook, develop_shutdown_hook) — retrieval
        (HybridRetriever 等) は :func:`_build_gen_pillar_retrieval` で後追いで
        追加する。
    """
    cfg = base.cfg
    debug_logger = base.debug_logger
    resolver = base.resolver

    async with asyncio.TaskGroup() as tg:
        t_llama = tg.create_task(
            _timed_task(
                timings, "llama_server",
                _init_llama_server(state, cfg, debug_logger),
            ),
        )
        t_pro_gen = tg.create_task(
            _timed_task(
                timings, "pro_gen_pillar", _init_pro_gen_pillar(state, cfg, resolver),
            ),
        )
        t_develop_gen = tg.create_task(
            _timed_task(
                timings, "develop_gen_pillar",
                _init_develop_gen_pillar(state, cfg, resolver),
            ),
        )
        t_embed = tg.create_task(
            _timed_task(
                timings, "embedding",
                _init_embedding(state, cfg, project_root, debug_logger),
            ),
        )
    client = t_llama.result()
    pro_shutdown = t_pro_gen.result()
    develop_shutdown = t_develop_gen.result()
    embedder = t_embed.result()

    with _timed(timings, "llm_client"):
        _init_llm_client(state, client)

    with _timed(timings, "aux_client"):
        aux_client = _init_aux_client(state, cfg, debug_logger, client)

    _start_quality_probes(
        state, cfg, resolver, debug_logger,
        local_client=client, embedder=embedder,
    )

    from backend.pillars import GenPillar
    gen = GenPillar(
        local_client=client,
        llm_client=state.llm_client,
        aux_client=aux_client,
        embedder=embedder,
    )
    return gen, pro_shutdown, develop_shutdown


async def _build_mem_pillar(
    state: AppState,
    base: _BaseContext,
    gen: "GenPillar",  # noqa: ARG001
    timings: dict[str, float],
) -> "MemPillar":
    """EvorefMem pillar を構築する (memory + cartridge + sleep-time + bootstrap)。

    依存方向: Mem は最下流だが、SleepTimeWorker は Gen の embedder / aux_client
    に依存するため ``gen`` を受け取る。SemMem への書込は MemFactView 経由、
    他 pillar (Loop/Learn) からは Fact View 経由でのみアクセスされる。
    """
    from backend.factory._memory_init import (
        _init_memory,
        apply_semmem_policy_overrides,
        bootstrap_loop_context_at_startup,
    )

    cfg = base.cfg
    debug_logger = base.debug_logger
    resolver = base.resolver

    with _timed(timings, "memory"):
        wm, stm, ltm, vs = _init_memory(state, cfg, resolver)
    await _timed_task(
        timings, "embedding_dim_check",
        _check_embedding_dim(state, cfg),
    )
    with _timed(timings, "cartridge_manager"):
        _init_cartridge_manager(state, cfg, resolver)

    # PolicyInterpreter ↔ SemMem 連携
    with _timed(timings, "policy_semmem_overrides"):
        try:
            from pathlib import Path as _Path

            from backend.free.memory.project_resolver import resolve_project_id
            project_id_for_policy = resolve_project_id(_Path.cwd()).project_id
        except Exception as exc:
            logger.warning(
                "policy_semmem_overrides: project_id resolution failed: %s", exc,
            )
            project_id_for_policy = None
        state.current_project_id = project_id_for_policy
        apply_semmem_policy_overrides(state, cfg, project_id_for_policy)

    # Loop startup bootstrap — Mem から Tier 1 素材を再構築する
    with _timed(timings, "loop_startup_bootstrap"):
        bootstrap_loop_context_at_startup(
            state, cfg, project_id_for_policy, debug_logger=debug_logger,
        )

    # Sleep-time 中核 (SleepTimeScheduler は Mem 所有、SleepTimeWorker の注入に必要)
    with _timed(timings, "sleep_scheduler"):
        from backend.free.memory.scheduler import SleepTimeScheduler
        # bg_task wrapper の outcome.jsonl 記録のため debug_logger を注入
        sleep_scheduler = SleepTimeScheduler(cfg, debug_logger=debug_logger)
        state.sleep_scheduler = sleep_scheduler
        # Full 実行の直前に **全セッション** の WM → STM スナップショットを
        # 走らせる (f_02 §1.2 経路 (c) / §4.3)。押し出しが起きていない進行中
        # セッションでは、これが Step 8 抽出への唯一の供給経路になる。吸収処理
        # (エコー落とし) は api 層にあるため、mem pillar からは注入で受ける。
        from backend.free.api.chat.chat_recorder import snapshot_all_wm_to_stm

        sleep_scheduler.set_pre_full_flush(
            lambda: snapshot_all_wm_to_stm(wm, stm),
        )

    from backend.pillars import MemPillar
    return MemPillar(
        working_memory=None,
        working_memory_registry=wm,
        short_term_memory=stm,
        long_term_memory=ltm,
        vector_store=vs,
        cartridge_manager=state.cartridge_manager,
        sleep_scheduler=sleep_scheduler,
        current_project_id=project_id_for_policy,
    )


def _build_gen_pillar_retrieval(
    state: AppState,
    base: _BaseContext,
    gen: "GenPillar",
    mem: "MemPillar",
    timings: dict[str, float],
) -> None:
    """Gen pillar の retrieval 層 (HybridRetriever)。

    Mem pillar の ``vector_store`` / ``cartridge_manager`` に依存するため、
    Mem 構築後に呼び出す。Gen pillar に ``hybrid_retriever`` を事後付与する。
    """
    cfg = base.cfg
    debug_logger = base.debug_logger
    policy_interpreter = base.policy_interpreter

    rag_cfg = (cfg or {}).get("rag", {}) or {}
    with _timed(timings, "bm25_index"):
        bm25 = _build_bm25_index(mem.vector_store, cfg)
        gen.bm25_retriever = bm25
        # チャット応答経路の LTM をハイブリッド (ベクトル + BM25 RRF) にする。
        # ``rag.hybrid_search`` は長らくチャット経路では no-op だったが、
        # ここで実際に効くようになった (config.yaml.example の NOTE も更新済)。
        if (
            bm25 is not None
            and mem.long_term_memory is not None
            and bool(rag_cfg.get("hybrid_search", True))
        ):
            mem.long_term_memory.set_bm25_retriever(
                bm25, rrf_k=int(rag_cfg.get("rrf_k", 60)),
            )
            logger.info("LTM hybrid retrieval enabled (vector + BM25 RRF)")
    with _timed(timings, "hybrid_retriever"):
        hybrid_retriever = _build_hybrid_retriever(
            mem.vector_store, gen.embedder,
            debug_logger, policy_interpreter, cfg, bm25=bm25,
        )
        gen.hybrid_retriever = hybrid_retriever
    with _timed(timings, "lazy_contextual"):
        _init_lazy_contextual(state, cfg, mem.vector_store, gen.embedder)
    with _timed(timings, "memory_threshold_calibration"):
        _init_memory_threshold_calibration(
            base.resolver, cfg, mem.short_term_memory, gen.embedder,
        )
    with _timed(timings, "judge_tracker"):
        _init_judge_tracker(state)


def _wire_fewshot_pool_to_sleep_worker(
    state: AppState, learning_scheduler: Any, mem: "MemPillar",
) -> None:
    """FewShotPool を SleepTimeWorker へ後付けする (構築順の都合)。

    どちらかが未構築 (--no-learning / degraded) なら黙って no-op。
    """
    pool = getattr(learning_scheduler, "_fewshot_pool", None)
    scheduler = getattr(mem, "sleep_scheduler", None)
    worker = getattr(scheduler, "_worker", None) if scheduler else None
    if pool is None or worker is None:
        return
    setter = getattr(worker, "set_fewshot_pool", None)
    if setter is None:
        return
    setter(pool)
    logger.info("FewShotPool wired to SleepTimeWorker for embedding backfill")


async def _build_learn_pillar(
    state: AppState,
    base: _BaseContext,
    project_root: Path,
    gen: "GenPillar",
    mem: "MemPillar",
    timings: dict[str, float],
) -> tuple["LearnPillar", Path, Path]:
    """EvorefLearn pillar を構築する (experience / scheduler / evolver / Pro 拡張)。

    依存: Mem (SleepTimeScheduler / SemMem view) + Gen (aux_client)。
    Pro エディションでは ``inject_learn_components`` 経由で Level 2 ランナーと
    ``ProLearnComponents`` が注入され、``state.pro.learn`` に集約される。

    Returns:
        (learn_pillar, experience_file_path, learned_patterns_file_path)
    """
    cfg = base.cfg
    debug_logger = base.debug_logger
    resolver = base.resolver
    policy_interpreter = base.policy_interpreter

    # Learn 中核 (experience / patterns / prompt / aux_prompt / sleep_scheduler)
    # SleepTimeScheduler は Mem 所有のため、_init_learning_core が使う
    # state.sleep_scheduler は既に Mem pillar 構築時に設定済み。
    with _timed(timings, "learning_core"):
        (
            exp_buf, exp_file,
            learned_patterns_store, patterns_file,
            prompt_mgr, aux_prompt_mgr,
            _sleep_scheduler_ignored,  # Mem が既に set 済み、重複生成を許容
        ) = _init_learning_core(state, cfg, resolver, debug_logger, policy_interpreter)

    # SleepTimeWorker を Mem の sleep_scheduler に注入 (Mem + Learn が連携するため
    # Learn フェーズで wire する — SleepTimeWorker 構築には Learn の exp_buf /
    # learned_patterns_store / aux_prompt_mgr が必要)
    # --no-learning 時は SleepTimeWorker / Level1 loop / Level2 runner /
    # Pro 学習コンポーネント注入をスキップする (LearningScheduler 本体は構築維持)
    learning_disabled = state.learning_disabled
    if not learning_disabled:
        with _timed(timings, "sleep_time_worker"):
            _init_sleep_time_worker(
                state, cfg, mem.sleep_scheduler, mem.short_term_memory,
                mem.long_term_memory, gen.embedder, mem.vector_store,
                exp_buf, debug_logger, learned_patterns_store,
                policy_interpreter, aux_prompt_mgr,
                bm25_retriever=gen.bm25_retriever,
            )
    else:
        logger.info("SleepTimeWorker setup skipped (learning disabled)")

    with _timed(timings, "learning_scheduler"):
        learning_scheduler = _init_learning_scheduler(
            state, cfg, exp_buf, prompt_mgr, debug_logger,
            policy_interpreter, learned_patterns_store, resolver,
        )

    # FewShotPool は LearningScheduler 構築の中で作られるため、SleepTimeWorker
    # (先に構築済み) へはここで後から渡す。新規に採用された手本の埋め込みを
    # sleep-time で張るのに使う — 埋め込みが載るまでその手本は密ベクトル選択の
    # 候補にならない (STM ノートが embed 工程を通るまで注入対象にならないのと
    # 同じ契約)。
    _wire_fewshot_pool_to_sleep_worker(state, learning_scheduler, mem)

    with _timed(timings, "component_wiring"):
        _wire_sleep_scheduler_models(
            state, mem.sleep_scheduler, learning_scheduler, resolver,
        )
        if not learning_disabled:
            _inject_level2_runner(learning_scheduler, debug_logger)
        else:
            logger.info("Level 2 runner injection skipped (learning disabled)")

    from backend.pillars import LearnPillar
    learn = LearnPillar(
        scheduler=learning_scheduler,
        experience_buffer=exp_buf,
        learned_patterns_store=learned_patterns_store,
        prompt_manager=prompt_mgr,
        aux_prompt_manager=aux_prompt_mgr,
        feedback_collector=state.feedback_collector,
    )

    # Pro Learn pillar setup (ProLearnComponents を LearningScheduler に注入する)
    if not learning_disabled:
        with _timed(timings, "pro_learn_setup"):
            await _inject_learn_components(
                state, cfg, resolver, project_root, learn,
            )
    else:
        logger.info("Pro learn components injection skipped (learning disabled)")

    # Level 1 独立常駐ループ起動
    if not learning_disabled:
        with _timed(timings, "level1_loop_start"):
            _start_level1_loop(mem.sleep_scheduler, learning_scheduler)
    else:
        logger.info("Level 1 independent loop start skipped (learning disabled)")

    return learn, exp_file, patterns_file


def _build_loop_pillar(
    state: AppState,
    base: _BaseContext,
    project_root: Path,
    gen: "GenPillar",
    mem: "MemPillar",  # noqa: ARG001
    learn: "LearnPillar",
    timings: dict[str, float],
) -> "LoopPillar":
    """EvorefLoop pillar を構築する (ツール / エージェント / LoopDriver)。

    ``loop.enabled=false`` の場合は ``LoopDriver=None`` / ``enabled=False`` で
    返す。tools / agent_tracer は loop driver の有無に関係なく常時初期化する
    (チャット経由のエージェントレイヤーでも使うため)。
    """
    cfg = base.cfg
    debug_logger = base.debug_logger

    with _timed(timings, "tools"):
        _init_tools(
            state, cfg, gen.local_client,
            learn.learned_patterns_store,
            embedder=gen.embedder,
        )
    with _timed(timings, "agent_tracer"):
        _init_agent_tracer(state, debug_logger)

    loop_cfg = (cfg or {}).get("loop", {}) or {}
    enabled = bool(loop_cfg.get("enabled", True))

    with _timed(timings, "loop_driver"):
        if enabled:
            _init_loop_driver(state, cfg, project_root, gen.aux_client)

    from backend.pillars import LoopPillar
    driver = getattr(state, "loop_driver", None) if enabled else None
    return LoopPillar(driver=driver, enabled=enabled)


# ──────────────────────────────────────────────────────────────────────────
# LogIngestor + PolicyAdjuster pipeline (develop=evolve 限定)
# ──────────────────────────────────────────────────────────────────────────


async def _bridge_log_ingestor_to_policy_adjuster(
    ingestor: Any, adjuster: Any,
) -> None:
    """LogIngestor.stream_pairs() → PolicyAdjuster.consume() の bg task body。

    ``CancelledError`` 受信時は ``flush_all()`` を呼んでから抜ける (lifespan
    shutdown 時の最終 flush 保証)。集約結果の SemMem 書込は consume 内で
    閾値判定により自動発火するため、本ブリッジ自身は最小限のループに留める。
    """
    try:
        async for pair in ingestor.stream_pairs():
            try:
                await adjuster.consume(pair)
            except Exception as exc:
                # 個別 pair の処理失敗で bridge 全体を停止させない
                logger.warning("PolicyAdjuster.consume failed: %s", exc)
                continue
    except asyncio.CancelledError:
        raise
    finally:
        # 最終 flush: 蓄積中で閾値到達済の bucket を確実に書き出す
        try:
            await adjuster.flush_all()
        except Exception as exc:
            logger.warning("PolicyAdjuster.flush_all on cancel failed: %s", exc)


async def _init_evolve_pipeline(
    state: AppState,
    project_root: Path,
    learn: "LearnPillar",
    loop_pillar: "LoopPillar",
) -> None:
    """``develop=evolve`` 時に LogIngestor + PolicyAdjuster + bridge を起動する。

    起動条件:

    * ``state.develop_level == "evolve"`` のみアクティブ化。それ以外
      (``"off"`` / ``"debug"`` / ``"investigate"``) では JSONL が出力されない
      ため動かしても無意味。
    * ``loop_pillar.enabled=False`` でもアクティブ化する: 自律ループは無効
      でもチャット経路の decision/outcome は記録されうるため、消費側だけは
      動かす意義がある。

    結線:

    1. LogIngestor: ``debug_log_dir`` = ``local/logs/debug/``、``state_path``
       = ``local/state/log_ingestor.json``
    2. LearnFactView: ``global`` + ``project:<id>`` (解決可能なら) を読み、
       writeback は ``project:<id>`` (なければ ``global``) — PolicyParamEvolver
       の semmem 経路と同じ scope 戦略
    3. PolicyAdjuster: 上記 view を注入
    4. bridge bg task: ``_bridge_log_ingestor_to_policy_adjuster`` を
       ``asyncio.create_task`` で起動

    結果は :attr:`LoopPillar.log_ingestor` / :attr:`LearnPillar.policy_adjuster`
    と ``state.log_ingestor_bridge_task`` (lifespan ctx 経由で shutdown) に格納。
    """
    if state.develop_level != "evolve":
        return

    from backend.free.learning.policy_adjuster import PolicyAdjuster
    from backend.free.loop.log_ingestor import LogIngestor
    from backend.free.memory.views.learn import LearnFactView

    from backend.config import PathResolver, get_config

    debug_log_dir = project_root / "local" / "logs" / "debug"
    try:
        _cfg_for_paths = get_config() or {}
    except Exception:
        _cfg_for_paths = {}
    state_path = PathResolver(_cfg_for_paths, project_root).resolve_local(
        "log_ingestor_file",
    )

    # LearnFactView: PolicyParamEvolver と同じ scope 戦略 (global + project)
    global_store = state.get_semantic_store("global")
    stores: list[Any] = [global_store]
    writeback_store: Any = global_store
    try:
        from backend.free.memory.project_resolver import resolve_project_id
        project_id = resolve_project_id(Path.cwd()).project_id
    except Exception as exc:
        logger.warning(
            "evolve pipeline: project_id resolution failed (using global only): %s",
            exc,
        )
        project_id = None
    if project_id:
        project_store = state.get_semantic_store(f"project:{project_id}")
        stores.append(project_store)
        writeback_store = project_store
    learn_view = LearnFactView(stores=stores, writeback_store=writeback_store)

    ingestor = LogIngestor(
        debug_log_dir=debug_log_dir,
        state_path=state_path,
    )
    adjuster = PolicyAdjuster(
        learn_view=learn_view,
        scope=f"project:{project_id}" if project_id else "global",
    )
    adjuster.set_base_model_id(getattr(state, "active_base_model_slug", "") or "")

    await ingestor.start()
    bridge_task = asyncio.create_task(
        _bridge_log_ingestor_to_policy_adjuster(ingestor, adjuster),
        name="log_ingestor_bridge",
    )

    # pillar に格納
    loop_pillar.log_ingestor = ingestor
    learn.policy_adjuster = adjuster
    state.log_ingestor_bridge_task = bridge_task

    logger.info(
        "evolve pipeline started: debug_log_dir=%s state_path=%s scope=%s",
        debug_log_dir, state_path, adjuster.scope,
    )


def _migrate_mode_rename(base: "_BaseContext") -> None:
    """旧モード名 ``"coding"`` のローカルデータを ``"create"`` へ一度きり移行する。

    マーカー (``local/.mode_renamed_create_v1``) 済みなら即 no-op。移行はユーザー
    データの整形であって起動要件ではないため、失敗しても WARNING で継続する。
    """
    from backend.free.core.mode_rename_migrator import ModeRenameMigrator

    try:
        ModeRenameMigrator(base.resolver).migrate_if_needed()
    except Exception as exc:
        logger.warning("Mode rename migration skipped: %s", exc)


def _activate_learning_partition(base: "_BaseContext", state: AppState) -> None:
    """base 学習パーティションを有効化する (active stem 確定 + flat→partition 移行)。

    resolver の active モデル stem を ``ModelState.current_filename`` から確定し、
    SemMem ``learn.*`` 用スラグを ``state.active_base_model_slug`` に保持、構築済
    PolicyInterpreter を当該モデルへ再スコープし、一度きりの flat→partition 非破壊
    移行を実行する。``partition_by_base_model=false`` 時は no-op (レガシー flat)。

    **必ず Learn pillar 構築 (_build_learn_pillar) の前に呼ぶこと** —
    experience / SystemPromptManager / FewShotPool / PolicyParamEvolver が
    resolve_learning / active slug をこの時点の値で確定するため。
    """
    cfg = base.cfg
    resolver = base.resolver
    if not resolver.partition_enabled:
        logger.info("Learning partition disabled (legacy flat layout)")
        return

    # embed_instruction は embedding モデル単位のパーティション (base モデルとは
    # 独立軸)。以降の base モデル identity 依存の早期 return に影響されないよう
    # ここで確定する。
    embed_filename = Path(cfg.get("model_paths", {}).get("embed_model") or "").name
    resolver.set_active_embedding_model_stem(
        Path(embed_filename).stem if embed_filename else None,
    )

    from backend.free.core.learning_partition_migrator import LearningPartitionMigrator
    from backend.free.core.model_migration import ModelState
    from backend.free.memory.notes.subject_ns import model_slug

    # active モデル = config の base_model (= 実際に起動する llama-server のモデル)。
    # 学習コンポーネントはこのモデルのパーティションを指す。
    active_filename = Path(cfg.get("model_paths", {}).get("base_model") or "").name
    if not active_filename:
        logger.warning("Learning partition: no base model identity; staying flat")
        return

    # stem / slug の確定は _learning_rebind と共有 (ランタイム切替でも同じ規則で束ねる)
    from backend.factory._learning_rebind import bind_active_base_model

    bound = bind_active_base_model(resolver, state, active_filename)
    if bound is None:
        return
    stem, _ = bound

    # PolicyInterpreter (base 構築済) を active モデルへ再スコープ。
    if base.policy_interpreter is not None:
        base.policy_interpreter.set_base_model_id(state.active_base_model_slug)

    # 一度きり flat→partition 非破壊移行。producer = flat データの生成元
    # (model_state.current_filename、前回起動モデル)。active(config) と異なる場合、
    # flat データは producer のパーティションへ移り active(新) は空 = ゼロから学習。
    producer_filename = active_filename
    try:
        ms = ModelState(resolver.resolve_local("model_state_file"))
        if ms.current_filename:
            producer_filename = ms.current_filename
    except Exception as exc:
        logger.debug("partition activation: ModelState read failed: %s", exc)

    migrator = LearningPartitionMigrator(
        resolver, cfg, develop_level=state.develop_level,
    )
    migrator.migrate_if_needed(producer_filename)
    # level2_adapter_partition=="model_mode" 初回有効化時、既存の (model のみ)
    # パーティション済みアダプタを "chat" バケットへ移す (独立マーカー、
    # partition_by_base_model の移行タイミングとは非同期に発火しうる)。
    migrator.migrate_adapter_partition_mode_if_needed(stem)
    logger.info(
        "Learning partition active: stem=%s slug=%s (migration producer=%s)",
        stem, state.active_base_model_slug, Path(producer_filename).stem,
    )


def _finalize_base(
    state: AppState, base: _BaseContext, timings: dict[str, float],
) -> None:
    """横断基盤の仕上げ (テーマ / model_state 検証 / エディションダウングレード)。"""
    cfg = base.cfg
    resolver = base.resolver

    with _timed(timings, "theme_manager"):
        _init_theme_manager(state, cfg, resolver)
    with _timed(timings, "model_state"):
        _validate_model_state(cfg, resolver, state)
        _validate_component_model_state(cfg, resolver, state)
    with _timed(timings, "edition_check"):
        _check_edition_downgrade(resolver)


async def wire_pillars(
    state: AppState, project_root: Path,
) -> tuple[_LifespanContext, dict[str, float]]:
    """4 pillar を依存順に構築し、shutdown 用 context + pillar 別 timings を返す。

    依存方向:
        ``EvorefGen → EvorefMem ← EvorefLoop ← EvorefLearn``

    実際の構築順は Gen core → Mem → Gen retrieval → Learn → Loop で、
    retrieval (HybridRetriever) のみ Mem (vector_store) への依存のため
    Mem 構築後に Gen pillar へ事後付与する。

    各フェーズの所要時間は ``timings`` dict に記録し、pillar 単位のサマリ
    (``pillar_gen`` / ``pillar_mem`` / ``pillar_loop`` / ``pillar_learn``) と
    個別コンポーネント時間の両方を含む。
    """
    # startup フェーズ全体の trace_id を設定。これにより起動時リトライの
    # debug JSONL エントリに trace_id が付与され、セッションと区別可能になる。
    set_trace_id(generate_trace_id())
    timings: dict[str, float] = {}

    # Base (横断基盤)
    with _timed(timings, "pillar_base"):
        base = await _build_base_context(state, project_root, timings)

    # 旧モード名 "coding" → "create" の一度きり移行。パーティション有効化より前に
    # 行い、以降のパス解決 (prompts/<mode>.md や <stem>/<mode>/) が現行名で当たる
    # ようにする。partition_by_base_model の有無や base モデル identity には
    # 依存しないため _activate_learning_partition の内側には置かない。
    with _timed(timings, "mode_rename_migration"):
        _migrate_mode_rename(base)

    # base 学習パーティション有効化 (active stem 確定 + flat→partition 一度きり移行)。
    # Learn pillar 構築より前に行い、experience / base prompts / fewshot / policy が
    # 当該 (model×mode) パーティションを指すようにする。
    with _timed(timings, "learning_partition"):
        _activate_learning_partition(base, state)

    # Gen pillar (core): LLM / 埋め込み / リランカー + Pro Gen 拡張 + Develop Gen 拡張
    with _timed(timings, "pillar_gen"):
        gen, pro_shutdown, develop_shutdown = await _build_gen_pillar(
            state, base, project_root, timings,
        )
        state.gen = gen

    # Mem pillar: WM/STM/LTM + SemMem + Sleep-time
    # auto_reindex_on_mismatch=true 時に embedding_dim_check 内で
    # async run_reindex を呼ぶため、_build_mem_pillar 自体が async。
    mem = await _timed_task(
        timings, "pillar_mem",
        _build_mem_pillar(state, base, gen, timings),
    )
    state.mem = mem

    # Gen retrieval (Mem 依存): HybridRetriever
    with _timed(timings, "pillar_gen_retrieval"):
        _build_gen_pillar_retrieval(state, base, gen, mem, timings)

    # Learn pillar: experience / LearningScheduler / Evolver + Pro Learn 拡張
    with _timed(timings, "pillar_learn"):
        learn, exp_file, patterns_file = await _build_learn_pillar(
            state, base, project_root, gen, mem, timings,
        )
        state.learn = learn
        # ランタイムの base 切替を検知したら Level 1 を止めずに再バインドする自己修復フック
        from backend.factory._learning_rebind import install_rebind_hook

        install_rebind_hook(state)

    # Loop pillar: tools / agent tracer / LoopDriver
    with _timed(timings, "pillar_loop"):
        loop_pillar = _build_loop_pillar(
            state, base, project_root, gen, mem, learn, timings,
        )
        state.loop = loop_pillar

    # develop=evolve 時のみ LogIngestor + PolicyAdjuster pipeline
    # を起動。Loop / Learn pillar 両方が構築済の状態で結線する (cross-pillar
    # 結線は app_factory 層で行うのが規約)。
    with _timed(timings, "evolve_pipeline"):
        try:
            await _init_evolve_pipeline(state, project_root, learn, loop_pillar)
        except Exception as exc:
            # evolve pipeline は補助機能のため、起動失敗してもアプリ全体を
            # 落とさない (WARN ログのみ)。loop / learn 自体には影響しない。
            logger.warning("evolve pipeline initialization failed: %s", exc)

    # 横断基盤仕上げ
    with _timed(timings, "pillar_base_finalize"):
        _finalize_base(state, base, timings)

    instance_name = base.cfg.get("instance", {}).get("name", "evoref")
    ctx = _LifespanContext(
        sleep_scheduler=mem.sleep_scheduler,
        learning_scheduler=learn.scheduler,
        wm=mem.working_memory_registry,
        stm=mem.short_term_memory,
        resolver=base.resolver,
        exp_buf=learn.experience_buffer,
        exp_file=exp_file,
        learned_patterns_store=learn.learned_patterns_store,
        patterns_file=patterns_file,
        pro_shutdown=pro_shutdown,
        instance_name=instance_name,
        develop_shutdown=develop_shutdown,
        log_ingestor=loop_pillar.log_ingestor,
        policy_adjuster=learn.policy_adjuster,
        log_ingestor_bridge_task=getattr(state, "log_ingestor_bridge_task", None),
    )
    return ctx, timings
