"""統合検索パイプライン: 3層メモリ + Self-RAG（asyncio.gather 並列検索）"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context
from backend.free.rag.chunk_content_gate import ChunkContentGate, GateConfig
from backend.free.rag.self_rag_judge import (
    QualityThresholds,
    RetrievalNecessityJudge,
    RetrievalQualityJudge,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.core.stage_timer import StageTimer
    from backend.free.memory.views.mem import MemFactView
    from backend.free.rag.assist_judge_tracker import AssistJudgeUsageTracker
    from backend.free.rag.lazy_contextual import LazyContextualPrefixService

logger = get_logger("memory.search_pipeline")

# STM / LTM / カートリッジ用スレッドプール（設計書 4.10.1）
_search_executor = ThreadPoolExecutor(max_workers=3)


@dataclass
class SearchResult:
    """統合検索の結果"""
    sources: list[tuple[str, float, str]] = field(default_factory=list)
    quality: str = "low"
    from_memory: bool = False
    skipped: bool = False


def _resolve_fetch_multiplier(cfg: dict) -> int:
    """候補拡張倍率を解決する。

    ``rag.fetch_multiplier`` (既定 1 = 拡張なし) を用いて LTM / カートリッジの
    取得件数を ``top_k * N`` へ広げ、「広く取って絞る」第1段として候補プールを
    確保する (STM は ``stm_top_k`` 固定で拡張対象外)。値は [1, 5] にクランプする。
    """
    rag_cfg = cfg.get("rag") or {}
    multiplier = rag_cfg.get("fetch_multiplier", 1)
    try:
        multiplier = int(multiplier)
    except (TypeError, ValueError):
        multiplier = 1
    return max(1, min(5, multiplier))


def _resolve_search_params(
    policy: PolicyInterpreter | None,
    rag_cfg: dict,
    mode: str,
) -> tuple[int, int, float]:
    """ポリシー優先で `(top_k, stm_top_k, noise_sigma)` を解決する。

    ポリシー未設定 / キー欠落時は config + ハードコードのデフォルトへフォールバック。
    """
    top_k = rag_cfg.get("top_k", 5)
    stm_top_k = 3
    noise_sigma = 0.05
    if policy is None:
        return top_k, stm_top_k, noise_sigma
    try:
        top_k = policy.get("search", "top_k", mode)
    except KeyError:
        pass
    try:
        stm_top_k = policy.get("search", "stm_top_k", mode)
    except KeyError:
        pass
    try:
        noise_sigma = policy.get("search", "noise_sigma", mode)
    except KeyError:
        pass
    return top_k, stm_top_k, noise_sigma


async def _search_stm_layer(
    short_term, query_vec: np.ndarray, stm_top_k: int,
) -> list[tuple[str, float, str]]:
    """Layer 2 短期記憶検索 (< 1ms)。失敗時は空リスト。"""
    loop = asyncio.get_running_loop()
    try:
        hits = await run_in_executor_with_context(
            loop, _search_executor, short_term.retrieve_top_k, query_vec, stm_top_k,
        )
        results = [(note.id, score, note.content) for note, score in hits]
        logger.debug(
            "Step 2 STM: %d hits, scores=[%s]",
            len(results),
            ", ".join(f"{s:.3f}" for _, s, _ in results),
        )
        return results
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("STM search failed: %s", e)
        return []


async def _search_ltm_layer(
    long_term, query_vec: np.ndarray, top_k: int,
) -> list[tuple[str, float, str]]:
    """Layer 3 長期記憶検索 (< 5ms)。LTM 未設定 / 失敗時は空リスト。"""
    if long_term is None:
        return []
    loop = asyncio.get_running_loop()
    try:
        results = await run_in_executor_with_context(
            loop, _search_executor, long_term.search, query_vec, top_k,
        )
        logger.debug("Step 3 LTM: %d results", len(results))
        return results
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("LTM search failed: %s", e)
        return []


async def _search_cartridge_layer(
    cartridge_mgr, query_vec: np.ndarray, top_k: int,
    timeout_ms: int = 0,
) -> list[tuple[str, float, str]]:
    """カートリッジ検索。マネージャ未設定 / 失敗時は空リスト。

    ``timeout_ms`` が 1 以上の場合、検索全体にタイムアウトを適用する
。タイムアウト時は空リストを返し、チャット応答を
    止めないようにする。
    """
    if cartridge_mgr is None or not hasattr(cartridge_mgr, "search"):
        return []
    loop = asyncio.get_running_loop()
    try:
        coro = run_in_executor_with_context(
            loop, _search_executor, cartridge_mgr.search, query_vec, top_k,
        )
        if timeout_ms > 0:
            results = await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
        else:
            results = await coro
        logger.debug("Step 3b cartridge: %d results", len(results))
        return results
    except asyncio.TimeoutError:
        logger.warning(
            "Cartridge search timed out after %d ms (L3); "
            "returning empty results to keep chat responsive",
            timeout_ms,
        )
        return []
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Cartridge search failed: %s", e)
        return []


def _resolve_assist_judge_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.assist_judge`` セクションを安全に取り出す

    セクション欠落時はデフォルト値相当の dict を返し、呼び出し側で
    キーの存在チェックを不要にする。旧 ``rag.assist_judge_enabled``
    フラット構造は廃止済み (後方互換なし)。
    """
    self_rag_cfg = rag_cfg.get("self_rag") or {}
    aj_cfg = self_rag_cfg.get("assist_judge") or {}
    return {
        "enabled": bool(aj_cfg.get("enabled", True)),
        "max_per_session": int(aj_cfg.get("max_per_session", 5)),
        "max_per_query": int(aj_cfg.get("max_per_query", 1)),
        "only_when_quality": list(aj_cfg.get("only_when_quality", ["medium"])),
    }


def _resolve_assist_necessity_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.assist_necessity`` セクションを安全に取り出す

    検索必要性のハイブリッド判定 (ルール → uncertain 時のみアシスト) を
    制御する。``only_when_quality`` は ``AssistJudgeUsageTracker`` 互換の
    キーで、本機能の quality ラベルは ``"uncertain"`` のみを使う。
    """
    self_rag_cfg = rag_cfg.get("self_rag") or {}
    an_cfg = self_rag_cfg.get("assist_necessity") or {}
    return {
        "enabled": bool(an_cfg.get("enabled", True)),
        "max_per_session": int(an_cfg.get("max_per_session", 10)),
        "max_per_query": int(an_cfg.get("max_per_query", 1)),
        "only_when_quality": list(an_cfg.get("only_when_quality", ["uncertain"])),
        "timeout_s": float(an_cfg.get("timeout_s", 5.0)),
        "context_turns": int(an_cfg.get("context_turns", 2)),
    }


def _resolve_necessity_recall_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.necessity_recall`` セクションを安全に取り出す

    embedding 決定論的リコール (``rag_judge_recall.try_recall_necessity``) の
    閾値設定。url/executable_command recall と同型。
    """
    cfg = ((rag_cfg.get("self_rag") or {}).get("necessity_recall")) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "topk": int(cfg.get("topk", 5)),
        "min_score": float(cfg.get("min_score", 0.75)),
        "min_record_score": float(cfg.get("min_record_score", 0.65)),
        "ttl_days": int(cfg.get("ttl_days", 14)),
    }


def _resolve_quality_recall_cfg(rag_cfg: dict) -> dict:
    """``rag.self_rag.quality_recall`` セクションを安全に取り出す (necessity_recall と対称)。"""
    cfg = ((rag_cfg.get("self_rag") or {}).get("quality_recall")) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "topk": int(cfg.get("topk", 5)),
        "min_score": float(cfg.get("min_score", 0.75)),
        "min_record_score": float(cfg.get("min_record_score", 0.65)),
        "ttl_days": int(cfg.get("ttl_days", 14)),
    }


async def _maybe_assist_judge_quality(
    quality_judge: RetrievalQualityJudge,
    quality: str,
    merged: list[tuple[str, float, str]],
    query: str,
    assist_client,
    rag_cfg: dict,
    *,
    session_id: str = "default",
    tracker: "AssistJudgeUsageTracker | None" = None,
    debug_logger: "DebugLogger | None" = None,
    assist_experience_recorder: "Callable[[str, str, str, float], None] | None" = None,
    mem_view: "MemFactView | None" = None,
    query_vec: np.ndarray | None = None,
) -> str:
    """marginal 境界 + アシスト有効時のみ LLM 品質再判定を行う

    発火条件は ``rag.self_rag.assist_judge`` の ``enabled`` /
    ``only_when_quality`` / ``max_per_session`` / ``max_per_query`` で
    制御する。上限超過・無効・quality 非該当の場合は ``debug_logger``
    へ ``op="assist_judge"`` + ``assist_judge_skipped_reason`` を記録し、
    ルールベース判定をそのまま返す。

    assist 呼出の前に、まず embedding 決定論的リコール
    (``rag_judge_recall.try_recall_quality``) を試す。ヒットすれば
    tracker 予算を消費せず、assist_client が None (degraded) でも
    LLM 呼出をスキップして即決定する。対象は assist と同じ
    ``only_when_quality`` (既定 ``medium``) のみ。
    """
    aj_cfg = _resolve_assist_judge_cfg(rag_cfg)

    if quality in aj_cfg["only_when_quality"]:
        quality_recall_cfg = _resolve_quality_recall_cfg(rag_cfg)
        if quality_recall_cfg["enabled"] and mem_view is not None and query_vec is not None:
            from backend.free.memory.pipeline.rag_judge_recall import try_recall_quality
            recalled = await try_recall_quality(query_vec, mem_view, quality_recall_cfg)
            if recalled is not None:
                label, recall_sim = recalled
                logger.debug(
                    "Step 5b quality recall: %s (sim=%.3f)", label, recall_sim,
                )
                if debug_logger is not None:
                    debug_logger.log_decision(
                        decision_point="self_rag_judge_path",
                        chosen="embedding_recall",
                        candidates=["rule_based", "assist_judge", "embedding_recall"],
                        reason="embedding_recall_matched",
                        context={
                            "rule_based_quality": quality,
                            "recalled_quality": label,
                            "similarity": recall_sim,
                        },
                        scope="request",
                    )
                return label

    # 前提条件: assist_client が無ければ判定不能 (skip ログは出さない)
    if assist_client is None:
        return quality

    # tracker が無いとセッション累計を追えないので、記録対象から外す
    # (tracker=None のフローはテスト経路でのみ発生する想定)
    if tracker is None:
        if not aj_cfg["enabled"]:
            return quality
        if quality not in aj_cfg["only_when_quality"]:
            return quality
        new_quality = await quality_judge.judge_with_assist(
            query, merged, assist_client, quality,
            record_assist=assist_experience_recorder,
        )
        logger.debug("Step 5b assist judge (no tracker): %s", new_quality)
        return new_quality

    decision = tracker.check(
        session_id=session_id,
        namespace="quality",
        quality=quality,
        query_count=0,
        config=aj_cfg,
    )
    if not decision.allowed:
        logger.debug(
            "Step 5b assist judge skipped: reason=%s, session=%d, query=%d",
            decision.reason, decision.session_count, decision.query_count,
        )
        if debug_logger is not None:
            debug_logger.log_assist_judge(
                query_preview=query,
                rule_based_quality=quality,
                used=False,
                final_quality=quality,
                session_count=decision.session_count,
                query_count=decision.query_count,
                skipped_reason=decision.reason,
            )
        return quality

    import time as _time
    started = _time.perf_counter()
    new_quality = await quality_judge.judge_with_assist(
        query, merged, assist_client, quality,
        record_assist=assist_experience_recorder,
    )
    elapsed = _time.perf_counter() - started
    session_count_after = tracker.record(session_id, namespace="quality")
    logger.debug(
        "Step 5b assist judge: rule=%s -> final=%s (session_count=%d)",
        quality, new_quality, session_count_after,
    )
    if debug_logger is not None:
        # session_count は "発火後" の累計を記録することで、skip ログ側の
        # "発火前" の数値と組み合わせて発火履歴を時系列で追える。
        debug_logger.log_assist_judge(
            query_preview=query,
            rule_based_quality=quality,
            used=True,
            final_quality=new_quality,
            session_count=session_count_after,
            query_count=decision.query_count + 1,
            elapsed_sec=elapsed,
        )
    return new_quality


async def _try_quality_expansion(
    quality: str,
    merged: list[tuple[str, float, str]],
    quality_judge: RetrievalQualityJudge,
    query: str,
    query_vec: np.ndarray,
    working_mem,
    long_term,
    top_k: int,
    noise_sigma: float,
) -> tuple[list[tuple[str, float, str]], str]:
    """品質不足 (low) 時にクエリ拡張で再検索を試みる。`(merged, quality)` を返す。"""
    if quality != "low" or not merged:
        return merged, quality
    logger.debug("Quality low — attempting query expansion")
    expanded_results = await _expand_and_research(
        query, query_vec, working_mem, long_term, top_k,
        noise_sigma=noise_sigma,
    )
    if not expanded_results:
        return merged, quality
    merged = _merge_results(merged, expanded_results, [])
    quality = quality_judge.judge(merged)
    logger.debug("After expansion: %d results, quality=%s", len(merged), quality)
    return merged, quality


def _log_memory_search_state(
    debug_logger,
    context_count: int,
    stm_results: list,
    ltm_results: list,
    semmem_stats: dict | None = None,
) -> None:
    """DebugLogger 設定時のみメモリ検索状態を記録する。

    ``semmem_stats`` が与えられた場合は memory.jsonl の ``semmem`` フィールド
    として埋め込む
    """
    if debug_logger is None:
        return
    debug_logger.log_memory_state(
        session_id="unified_search",
        memory_dump={
            "working_turns": context_count,
            "stm_notes": len(stm_results),
            "ltm_vectors": len(ltm_results),
        },
        semmem_stats=semmem_stats,
    )


async def unified_search(
    query: str,
    query_vec: np.ndarray,
    working_mem,
    short_term,
    long_term,
    cartridge_mgr=None,
    config: dict | None = None,
    assist_client=None,
    debug_logger=None,
    mode: str = "chat",
    policy: PolicyInterpreter | None = None,
    timer: "StageTimer | None" = None,
    semmem_stats: dict | None = None,
    lazy_contextual: "LazyContextualPrefixService | None" = None,
    *,
    session_id: str = "default",
    assist_judge_tracker: "AssistJudgeUsageTracker | None" = None,
    necessity_prompt: str | None = None,
    quality_prompt: str | None = None,
    assist_experience_recorder: "Callable[[str, str, str, float], None] | None" = None,
    mem_view: "MemFactView | None" = None,
) -> SearchResult:
    """統合検索パイプライン: Self-RAG + 3層メモリ

    デフォルトはルールベース + ベクトル演算で完結（ベースモデル呼び出しゼロ）。
    ``rag.self_rag.assist_judge.enabled=true`` (既定) 時、
    ``only_when_quality`` に該当した検索品質判定 (既定 ``medium``) で
    アシストモデル LLM を併用し marginal な結果を救済する
    ``max_per_session`` / ``max_per_query`` の発火上限とセッション単位の
    カウンタを追加し、ユーザ体感レイテンシへの影響を抑制する。
    STM / LTM / カートリッジ検索を asyncio.gather で並列実行する。
    ``rag.fetch_multiplier`` が 2 以上の場合、LTM / カートリッジの取得件数を
    `top_k * N` に拡張する (STM は `stm_top_k` 固定)。

    Args:
        session_id: assist_judge の発火カウンタキー
            ``run_search_pipeline`` が ``WorkingMemory.session_id`` か
            フロントエンド指定 session_id を渡す。
        assist_judge_tracker: セッション単位のカウンタ。``None`` の場合、
            セッション上限は評価されずルールベース + quality gate のみで
            判定する (テスト経路互換)。
    """
    cfg = config or {}
    rag_cfg = cfg.get("rag", {})
    top_k, stm_top_k, noise_sigma = _resolve_search_params(policy, rag_cfg, mode)
    multiplier = _resolve_fetch_multiplier(cfg)
    fetch_k = top_k * multiplier
    logger.debug(
        "unified_search: query=%r, top_k=%d, fetch_k=%d (mult=%d)",
        query[:80], top_k, fetch_k, multiplier,
    )

    # Step 1: Self-RAG 検索必要性判定 (rule + uncertain 時のみアシスト)
    # necessity_prompt は AssistPromptManager (task=rag_necessity) 由来の編集可能
    # 指示部。composition 層から plain str で注入され、None なら judge 側既定に倒す。
    necessity_judge = RetrievalNecessityJudge(necessity_instructions=necessity_prompt)
    full_context = working_mem.get_context()
    context_count = len(full_context)
    # 末尾ターンは現在のユーザクエリ自身 (chat service が search 前に
    # WorkingMemory.add_turn 済) なので、アシストプロンプトの "最新のクエリ"
    # と重複しないよう除外する。末尾が user role でない (テスト経路等) なら
    # 全件をそのまま渡す。
    if full_context and full_context[-1].get("role") == "user":
        recent_context = full_context[:-1]
    else:
        recent_context = full_context
    necessity_cfg = _resolve_assist_necessity_cfg(rag_cfg)
    necessity_recall_cfg = _resolve_necessity_recall_cfg(rag_cfg)

    rule_necessity = necessity_judge.judge_rule_only(query, context_count)
    necessity: str | None = None
    if rule_necessity != "uncertain":
        necessity = rule_necessity
    elif necessity_recall_cfg["enabled"] and mem_view is not None:
        from backend.free.memory.pipeline.rag_judge_recall import try_recall_necessity
        recalled = await try_recall_necessity(query_vec, mem_view, necessity_recall_cfg)
        if recalled is not None:
            necessity, recall_sim = recalled
            logger.info(
                "Necessity recall: %s (sim=%.3f, query=%r)",
                necessity, recall_sim, query[:50],
            )
            if debug_logger is not None:
                debug_logger.log_decision(
                    decision_point="self_rag_necessity_path",
                    chosen=necessity,
                    candidates=["retrieve", "fetch", "skip"],
                    reason="embedding_recall_matched",
                    context={"similarity": recall_sim},
                    scope="request",
                )
    if necessity is None:
        if necessity_cfg["enabled"] and assist_client is not None:
            necessity = await necessity_judge.judge_with_assist(
                query,
                context_count,
                assist_client,
                recent_context=recent_context,
                session_id=session_id,
                tracker=assist_judge_tracker,
                debug_logger=debug_logger,
                config=necessity_cfg,
                record_assist=assist_experience_recorder,
            )
        else:
            necessity = necessity_judge.judge(query, context_count=context_count)
    logger.debug("Step 1 necessity: %s (context_count=%d)", necessity, context_count)
    # `fetch` は外部 fetch_url 委譲シグナル — RAG パイプラインは `skip` と同等に
    # 即終了し、ToolCallJudge / fetch_url ツールに委ねる。
    if necessity in ("skip", "fetch"):
        logger.info(
            "Search skipped (necessity=%s) for query: %s", necessity, query[:50],
        )
        return SearchResult(skipped=True, from_memory=True)

    # Step 2-3: STM / LTM / カートリッジを asyncio.gather で並列実行（設計書 4.10.1）
    # fetch_multiplier >= 2 のときは fetch_k 件を取得し、後段で top_k に絞る
    cart_timeout_ms = int(rag_cfg.get("cartridge_search_timeout_ms", 3000))
    stm_results, ltm_results, cart_results = await asyncio.gather(
        _search_stm_layer(short_term, query_vec, stm_top_k),
        _search_ltm_layer(long_term, query_vec, fetch_k),
        _search_cartridge_layer(
            cartridge_mgr, query_vec, fetch_k, timeout_ms=cart_timeout_ms,
        ),
    )

    # Lazy Contextual Retrieval — LTM hit について on-demand で
    # プレフィックス生成を非同期タスクとして起動する。fire-and-forget なので
    # 本 retrieval のレイテンシには影響せず、次回以降の同一 chunk ヒット時に
    # contextual_text が使えるようになる。
    if lazy_contextual is not None and lazy_contextual.is_active and ltm_results:
        ltm_chunk_ids = [cid for cid, _, _ in ltm_results]
        asyncio.create_task(
            lazy_contextual.on_retrieval_hits(ltm_chunk_ids),
            name=f"lazy_contextual_hits[{len(ltm_chunk_ids)}]",
        )

    # Step 4: 結果マージ。merged_raw は生スコア (品質判定 / gate / クエリ拡張用)、
    # merged は最終順位付け用。score_normalization でクロスレイヤ正規化を適用し、
    # STM/LTM/cartridge の異種スコアスケールの歪みを吸収する。
    norm = rag_cfg.get("score_normalization", "none")
    merged_raw = _merge_results(stm_results, ltm_results, cart_results)
    if norm == "none":
        merged = merged_raw
    else:
        merged = _merge_results(
            stm_results, ltm_results, cart_results, normalization=norm,
        )
    logger.debug(
        "Step 4 merge: %d unique results after dedup (normalization=%s)",
        len(merged_raw), norm,
    )

    # Step 4.5: 取得直後の内容精査ゲート — 低価値 chunk を pruning し、後続の
    # 品質判定 / クエリ拡張の候補数を縮小する。
    # coding mode を主対象 (chat mode は近似重複除去のみ)。marginal band の
    # prose のみ assist で 1 回関連性判定する (assist 無/cap 超過/error は純ルール)。
    # gate は生スコア (relevance_floor は cosine 前提) で判定するため merged_raw に
    # 適用し、残った chunk_id 集合を正規化側 merged にも射影する。
    gate_cfg = GateConfig.from_rag_cfg(rag_cfg)
    if gate_cfg.enabled and merged_raw:
        merged_raw = await ChunkContentGate(
            gate_cfg, debug_logger=debug_logger,
        ).filter(
            query, merged_raw, mode,
            assist_client=assist_client,
            tracker=assist_judge_tracker,
            session_id=session_id,
        )
        if norm == "none":
            merged = merged_raw
        else:
            kept = {cid for cid, _, _ in merged_raw}
            merged = [t for t in merged if t[0] in kept]
        logger.debug("Step 4.5 content gate: %d results after prune", len(merged_raw))

    # Step 5: Self-RAG 品質判定 + (オプション) アシスト強化（< 0.1ms）
    # 判定は常に生スコア (merged_raw) に対して行う。品質3閾値は cosine 分布前提の
    # ため、正規化スコアを渡すと閾値の意味が崩れる。
    thresholds = QualityThresholds.from_config(rag_cfg)
    # decision.jsonl に記録 (decision_point=``self_rag_judge_path``)
    quality_judge = RetrievalQualityJudge(
        thresholds, debug_logger=debug_logger,
        quality_instructions=quality_prompt,
    )
    quality = quality_judge.judge(merged_raw)
    logger.debug("Step 5 quality: %s", quality)
    if timer is not None:
        timer.start("assist_judge_ms")
    try:
        quality = await _maybe_assist_judge_quality(
            quality_judge, quality, merged_raw, query, assist_client, rag_cfg,
            session_id=session_id,
            tracker=assist_judge_tracker,
            debug_logger=debug_logger,
            assist_experience_recorder=assist_experience_recorder,
            mem_view=mem_view,
            query_vec=query_vec,
        )
    finally:
        if timer is not None:
            timer.stop("assist_judge_ms")

    # Step 6: 品質不足時のクエリ拡張フォールバック (生スコアで再検索)。
    # 拡張が発火した場合 (quality=="low" の稀ケース) は結果集合が変わるため、
    # 正規化側 merged も生順 (拡張結果込み) にフォールバックして確実に届ける。
    expanded, quality = await _try_quality_expansion(
        quality, merged_raw, quality_judge, query, query_vec,
        working_mem, long_term, top_k, noise_sigma,
    )
    if expanded is not merged_raw:
        merged_raw = expanded
        merged = expanded

    # Step 7: 最終順位付け (merged はスコア降順) から top_k 件を採用
    final_sources = merged[:top_k]

    # Step 7.5: カートリッジ公平性保証
    # ロード済みカートリッジが上位選別で完全に欠落するのを防ぐ
    final_sources = _ensure_cartridge_fairness(
        final_sources, cart_results, top_k,
    )

    logger.info(
        "Search completed: %d results, quality=%s, from_memory=%s",
        len(final_sources), quality, bool(stm_results),
    )
    _log_memory_search_state(
        debug_logger, context_count, stm_results, ltm_results,
        semmem_stats=semmem_stats,
    )

    return SearchResult(
        sources=final_sources,
        quality=quality,
        from_memory=bool(stm_results),
    )


def _cartridge_id_of(chunk_id: str) -> str | None:
    """カートリッジ由来のチャンク ID から cart_id 部分を抽出する。

    `cartridge_manager.search` は ``"<cart_id>:<original_chunk_id>"`` 形式で
    chunk_id を返す。STM/LTM 由来の chunk_id は通常 ":" を含まないため、
    プレフィックス分離で十分判別可能。
    """
    if ":" not in chunk_id:
        return None
    return chunk_id.split(":", 1)[0]


def _ensure_cartridge_fairness(
    final: list[tuple[str, float, str]],
    cart_results: list[tuple[str, float, str]],
    top_k: int,
) -> list[tuple[str, float, str]]:
    """ロード済みカートリッジの公平な代表を保証する

    `cart_results` (cartridge_manager.search の生出力) に含まれる各
    カートリッジのうち、最終結果 `final` から完全に欠落しているものがあれば、
    その入力チャンクの先頭 (= ラウンドロビン優先順位の最高) を最終結果に
    強制的に含める。

    グローバルスコアソートが特定カートリッジを完全に除外する
    挙動を補正することが目的。複数カートリッジが存在しないケースでは
    `final` をそのまま返す (no-op)。

    既に `final` の長さが `top_k` 未満なら末尾追加し、満杯の場合は最低
    スコアの非カートリッジ要素または別カートリッジで重複代表されている
    要素を差し替える。

    注入エントリのスコアは `final` 内の最低スコアへ揃える。`final` は
    score_normalization 適用済み [0,1] であり、
    `cart_results` の生スコア (cosine*priority、無上限) を持ち込むと下流の
    SalienceRanker min-max が歪むため。強制注入 = 最下位相当の意味付け。
    """
    if not cart_results or not final:
        return final

    floor_score = min(s for _, s, _ in final)

    input_carts: list[str] = []
    seen_input: set[str] = set()
    cart_top_chunk: dict[str, tuple[str, float, str]] = {}
    for entry in cart_results:
        cid = _cartridge_id_of(entry[0])
        if cid is None:
            continue
        if cid not in seen_input:
            seen_input.add(cid)
            input_carts.append(cid)
            cart_top_chunk[cid] = entry

    if len(input_carts) < 2:
        # カートリッジが 1 つ以下なら公平性問題は起きない
        return final

    output_carts = {
        c
        for cid, _, _ in final
        for c in (_cartridge_id_of(cid),)
        if c is not None
    }

    missing = [c for c in input_carts if c not in output_carts]
    if not missing:
        return final

    # 既存の chunk_id 集合 (重複混入防止)
    existing_ids = {cid for cid, _, _ in final}
    result = list(final)

    for cart_id in missing:
        raw = cart_top_chunk[cart_id]
        if raw[0] in existing_ids:
            continue
        chunk = (raw[0], floor_score, raw[2])
        if len(result) < top_k:
            result.append(chunk)
            existing_ids.add(chunk[0])
            output_carts.add(cart_id)
            continue
        # 満杯: 差し替え対象を選ぶ
        # 優先度1: 非カートリッジ (STM/LTM) のうち最低スコア
        # 優先度2: 既に他で代表されているカートリッジの最低スコア
        replace_idx = _find_replaceable_index(result, output_carts)
        if replace_idx is None:
            continue
        result[replace_idx] = chunk
        existing_ids.add(chunk[0])
        output_carts.add(cart_id)

    logger.debug(
        "cartridge fairness: input_carts=%s, missing=%s, applied=%d",
        input_carts, missing, len(missing),
    )
    return result


def _find_replaceable_index(
    final: list[tuple[str, float, str]],
    output_carts: set[str],  # noqa: ARG001
) -> int | None:
    """差し替え可能な要素のインデックスを返す

    優先度:
        1. 非カートリッジ要素 (STM/LTM 由来) のうち最終位置 (≒最低スコア)
        2. 複数カートリッジが代表されている場合の重複代表側の最終位置

    どれも該当しない場合 (1 カートリッジしかなく満杯) は None。
    """
    # 優先度 1: 非カートリッジ要素を末尾から探す
    for i in range(len(final) - 1, -1, -1):
        if _cartridge_id_of(final[i][0]) is None:
            return i

    # 優先度 2: 同じカートリッジが複数回代表されているケースで末尾を差し替え
    cart_counts: dict[str, int] = {}
    for cid, _, _ in final:
        c = _cartridge_id_of(cid)
        if c is not None:
            cart_counts[c] = cart_counts.get(c, 0) + 1

    for i in range(len(final) - 1, -1, -1):
        c = _cartridge_id_of(final[i][0])
        if c is not None and cart_counts.get(c, 0) > 1:
            cart_counts[c] -= 1
            return i

    return None


def _normalize_layer(
    layer: list[tuple[str, float, str]],
    method: str,
) -> list[tuple[str, float, str]]:
    """1 層 (STM/LTM/cartridge) 内でスコアを正規化する。

    ``minmax``: 層内 min-max で [0, 1] へ写像。レンジ 0 (1 件 / 全同値) は
        0 に潰さず一律 1.0 とする (単独層を層間で不利にしないため)。
    ``rank``: 1 位=1.0 … 最下位=1/n の線形ランク減衰。n=1 は 1.0。
    ``none`` / 未知: 入力をそのまま返す (no-op)。空層は空リスト。

    combined スコアをスカラとして正規化するため、STM の blended
    (cosine*0.6+lightmem*0.4) や cartridge の cosine*priority の内訳は壊さない
    (層内相対順位は不変、層間スケールのみ揃う)。cartridge priority は無上限だが
    層内 min-max が層内相対順位を保ったまま [0,1] に収めるため歪みを吸収する。
    """
    n = len(layer)
    if n == 0:
        return []
    if method == "minmax":
        scores = [s for _, s, _ in layer]
        lo = min(scores)
        hi = max(scores)
        rng = hi - lo
        if rng <= 1e-12:
            return [(cid, 1.0, text) for cid, _, text in layer]
        return [(cid, (s - lo) / rng, text) for cid, s, text in layer]
    if method == "rank":
        order = sorted(range(n), key=lambda i: -layer[i][1])
        norm = [0.0] * n
        for rank, idx in enumerate(order):
            norm[idx] = (n - rank) / n
        return [(layer[i][0], norm[i], layer[i][2]) for i in range(n)]
    return layer


def _merge_results(
    *result_lists: list[tuple[str, float, str]],
    normalization: str = "none",
) -> list[tuple[str, float, str]]:
    """結果をマージしてスコア降順でソート（重複排除）。

    ``normalization`` が ``minmax`` / ``rank`` のとき、各 result_list を 1 層と
    みなして層内でスコアを正規化してからマージし、STM/LTM/cartridge の異種
    スコアスケールを吸収する。``none`` (既定) は従来どおり生スコアでマージする。
    重複は最初に出現した層の要素を採用する (先勝ち、スコアに非依存=現状一致)。
    """
    seen: set[str] = set()
    merged: list[tuple[str, float, str]] = []

    for results in result_lists:
        for chunk_id, score, text in _normalize_layer(results, normalization):
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append((chunk_id, score, text))

    merged.sort(key=lambda x: -x[1])
    return merged


async def _expand_and_research(
    query: str,  # noqa: ARG001
    query_vec: np.ndarray,
    working_mem,
    long_term,
    top_k: int,
    noise_sigma: float = 0.05,
) -> list[tuple[str, float, str]]:
    """クエリベクトル摂動による簡易再検索（LLM なし）。

    直近の会話コンテキストがある場合に限り、クエリベクトルを微小ノイズで
    摂動させて近傍を再取得する。会話コンテキストの内容自体は (キーワード抽出
    等で) 検索条件に反映しない簡易拡張。
    """
    # 直近 3 ターンに非空の発話があるときだけ拡張する。
    has_context = any(
        (turn.get("content", "") or "").strip()
        for turn in working_mem.get_context()[-3:]
    )
    if not has_context or long_term is None:
        return []

    # クエリベクトルを少し摂動させて再検索（簡易的な拡張）
    noise = np.random.randn(*query_vec.shape).astype(np.float32) * noise_sigma
    expanded_vec = query_vec + noise
    norm = np.linalg.norm(expanded_vec)
    if norm > 0:
        expanded_vec = expanded_vec / norm

    loop = asyncio.get_running_loop()
    try:
        return await run_in_executor_with_context(
            loop, _search_executor, long_term.search, expanded_vec, top_k,
        )
    except asyncio.CancelledError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.warning("Expanded search failed: %s", e)
        return []
