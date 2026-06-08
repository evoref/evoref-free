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
from backend.free.rag.reranker_skip import (
    evaluate_reranker_skip,
    is_skip_config_active,
)
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
    from backend.free.rag.assist_judge_tracker import AssistJudgeUsageTracker
    from backend.free.rag.lazy_contextual import LazyContextualPrefixService
    from backend.free.rag.reranker_backend import RerankerBackend

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


def _resolve_candidates_multiplier(
    cfg: dict, reranker: "RerankerBackend | None",
) -> int:
    """reranker 有効時の候補拡張倍率を解決する。

    無効時は 1（拡張なし）を返す。
    """
    if reranker is None or not getattr(reranker, "is_active", False):
        return 1
    reranker_cfg = cfg.get("reranker") or {}
    multiplier = reranker_cfg.get("candidates_multiplier", 3)
    try:
        multiplier = int(multiplier)
    except (TypeError, ValueError):
        multiplier = 3
    return max(1, multiplier)


def _resolve_rerank_candidate_cap(cfg: dict, top_k: int, multiplier: int) -> int:
    """リランカーへ投入する候補数の上限を解決する

    計測の結果 merged が STM(3) + LTM(fetch_k) + cart(fetch_k) のユニオン
    として 25 件前後まで膨らみ、リランカーが 1 件あたり ~55ms かかる構造が
    判明した。本上限はリランカーへ渡す前に hybrid score 上位のみへ縮小し、
    forward pass を線形に短縮する。検索品質判定 (Step 5) は merged 全件で
    動作するため、この上限は順位付けの計算量にのみ影響する。

    未指定時のデフォルトは ``top_k * candidates_multiplier`` (例: 5*3=15)。
    ``top_k`` 未満には絶対に下げない（リランカーが top_k を返せなくなるため）。
    """
    reranker_cfg = cfg.get("reranker") or {}
    cap = reranker_cfg.get("rerank_candidate_cap")
    if cap is None:
        return max(top_k, top_k * multiplier)
    try:
        return max(top_k, int(cap))
    except (TypeError, ValueError):
        return max(top_k, top_k * multiplier)


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


async def _maybe_assist_judge_quality(
    quality_judge: RetrievalQualityJudge,
    quality: str,
    merged: list[tuple[str, float, str]],
    query: str,
    assist_client,
    rag_cfg: dict,
    reranker: "RerankerBackend | None" = None,
    *,
    session_id: str = "default",
    tracker: "AssistJudgeUsageTracker | None" = None,
    debug_logger: "DebugLogger | None" = None,
    assist_experience_recorder: "Callable[[str, str, str, float], None] | None" = None,
) -> str:
    """marginal 境界 + アシスト有効時のみ LLM 品質再判定を行う

    発火条件は ``rag.self_rag.assist_judge`` の ``enabled`` /
    ``only_when_quality`` / ``max_per_session`` / ``max_per_query`` で
    制御する。上限超過・無効・quality 非該当の場合は ``debug_logger``
    へ ``op="assist_judge"`` + ``assist_judge_skipped_reason`` を記録し、
    ルールベース判定をそのまま返す。

    リランカーが有効な場合は LLM 品質再判定をスキップする
    リランカーがクロスエンコーダで merged を強力に再順位付けするため、
    assist LLM による medium→low 再分類 (→ クエリ拡張トリガー) の効果は
    限定的。1 回の assist LLM 呼び出し (generate_json, 中央値 ~500-800ms)
    が `search_ms` 残差の長い裾を支配していたため削減対象とする。
    """
    aj_cfg = _resolve_assist_judge_cfg(rag_cfg)

    # 前提条件: assist_client が無ければ判定不能 (skip ログは出さない)
    if assist_client is None:
        return quality

    # リランカー有効時は skip (skip ログは出さない — 既存挙動維持)
    if reranker is not None and getattr(reranker, "is_active", False):
        logger.debug(
            "Step 5b assist judge skipped: reranker active"
        )
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
    session_count_after = tracker.record(session_id)
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
    reranker: "RerankerBackend | None" = None,
    timer: "StageTimer | None" = None,
    semmem_stats: dict | None = None,
    lazy_contextual: "LazyContextualPrefixService | None" = None,
    *,
    session_id: str = "default",
    assist_judge_tracker: "AssistJudgeUsageTracker | None" = None,
    necessity_prompt: str | None = None,
    quality_prompt: str | None = None,
    assist_experience_recorder: "Callable[[str, str, str, float], None] | None" = None,
) -> SearchResult:
    """統合検索パイプライン: Self-RAG + 3層メモリ + (任意) リランカー

    デフォルトはルールベース + ベクトル演算で完結（ベースモデル呼び出しゼロ）。
    ``rag.self_rag.assist_judge.enabled=true`` (既定) 時、
    ``only_when_quality`` に該当した検索品質判定 (既定 ``medium``) で
    アシストモデル LLM を併用し marginal な結果を救済する
    ``max_per_session`` / ``max_per_query`` の発火上限とセッション単位の
    カウンタを追加し、ユーザ体感レイテンシへの影響を抑制する。
    STM / LTM / カートリッジ検索を asyncio.gather で並列実行する。
    `reranker` 指定時かつ `is_active=True` の場合、各層の取得件数を
    `top_k * candidates_multiplier` に拡張し、マージ後にリランカーで
    `top_k` 件まで再順位付けする

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
    multiplier = _resolve_candidates_multiplier(cfg, reranker)
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
    # reranker 有効時は fetch_k 件を取得し、後段でリランカーが top_k に絞る
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

    # Step 4: 結果マージ + スコア正規化（< 0.5ms）
    merged = _merge_results(stm_results, ltm_results, cart_results)
    logger.debug("Step 4 merge: %d unique results after dedup", len(merged))

    # Step 4.5: 取得直後の内容精査ゲート — 低価値 chunk を pruning し、後続の
    # 品質判定 / クエリ拡張 / reranker forward pass の候補数を縮小する。
    # coding mode を主対象 (chat mode は近似重複除去のみ)。marginal band の
    # prose のみ assist で 1 回関連性判定する (assist 無/cap 超過/error は純ルール)。
    gate_cfg = GateConfig.from_rag_cfg(rag_cfg)
    if gate_cfg.enabled and merged:
        merged = await ChunkContentGate(
            gate_cfg, debug_logger=debug_logger,
        ).filter(
            query, merged, mode,
            assist_client=assist_client,
            tracker=assist_judge_tracker,
            session_id=session_id,
        )
        logger.debug("Step 4.5 content gate: %d results after prune", len(merged))

    # Step 5: Self-RAG 品質判定 + (オプション) アシスト強化（< 0.1ms）
    thresholds = QualityThresholds.from_config(rag_cfg)
    # decision.jsonl に記録 (decision_point=``self_rag_judge_path``)
    quality_judge = RetrievalQualityJudge(
        thresholds, debug_logger=debug_logger,
        quality_instructions=quality_prompt,
    )
    quality = quality_judge.judge(merged)
    logger.debug("Step 5 quality: %s", quality)
    if timer is not None:
        timer.start("assist_judge_ms")
    try:
        quality = await _maybe_assist_judge_quality(
            quality_judge, quality, merged, query, assist_client, rag_cfg,
            reranker=reranker,
            session_id=session_id,
            tracker=assist_judge_tracker,
            debug_logger=debug_logger,
            assist_experience_recorder=assist_experience_recorder,
        )
    finally:
        if timer is not None:
            timer.stop("assist_judge_ms")

    # Step 6: 品質不足時のクエリ拡張フォールバック
    merged, quality = await _try_quality_expansion(
        quality, merged, quality_judge, query, query_vec,
        working_mem, long_term, top_k, noise_sigma,
    )

    # Step 7: リランカー適用
    # 上位 N 件にキャップしてリランカー forward pass を短縮
    # reranker.skip の quality-aware 条件で高 confidence 時は短絡
    # ``mode`` を reranker へ伝搬し instruction-aware に切替
    candidate_cap = _resolve_rerank_candidate_cap(cfg, top_k, multiplier)
    final_sources = await _maybe_rerank(
        reranker, query, merged, top_k,
        timer=timer, candidate_cap=candidate_cap,
        cfg=cfg, debug_logger=debug_logger, mode=mode,
    )

    # Step 7.5: カートリッジ公平性保証
    # ロード済みカートリッジが reranker / 上位選別で完全に欠落するのを防ぐ
    final_sources = _ensure_cartridge_fairness(
        final_sources, cart_results, top_k,
    )

    logger.info(
        "Search completed: %d results, quality=%s, from_memory=%s, reranked=%s",
        len(final_sources), quality, bool(stm_results),
        bool(reranker and getattr(reranker, "is_active", False) and merged),
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


async def _maybe_rerank(
    reranker: "RerankerBackend | None",
    query: str,
    merged: list[tuple[str, float, str]],
    top_k: int,
    timer: "StageTimer | None" = None,
    candidate_cap: int | None = None,
    cfg: dict | None = None,
    debug_logger: "DebugLogger | None" = None,
    mode: str = "chat",
) -> list[tuple[str, float, str]]:
    """リランカー有効時に merged を再順位付けし top_k 件返す。

    無効/空入力/失敗時は単純に `merged[:top_k]` を返す。
    `timer` 指定時は `rerank_ms` ステージとして計測を記録する。
    `candidate_cap` 指定時は hybrid score 上位 N 件のみリランカーへ渡し、
    forward pass のコストを抑える。``None`` または
    ``len(merged)`` 以上の値は無視され、従来通り全件投入する。

    ``cfg`` に ``reranker.skip`` のしきい値が設定されている場合
    融合後 top スコア / gap / 候補数を評価し、高 confidence 時は cross-encoder
    forward pass を skip する。skip 時は ``debug_logger.log_rerank_skipped``
    を介して ``rag.jsonl`` に ``op="rerank_skip"`` エントリを書き出す。
    """
    if not merged:
        return []
    if reranker is None or not getattr(reranker, "is_active", False):
        return merged[:top_k]

    if candidate_cap is not None and 0 < candidate_cap < len(merged):
        candidates = merged[:candidate_cap]
    else:
        candidates = merged

    # quality-aware skip 条件を評価
    if is_skip_config_active(cfg):
        decision = evaluate_reranker_skip(candidates, cfg)
        if decision.should_skip:
            logger.debug(
                "Reranker skipped by quality gate: reason=%s, top_score=%.4f, "
                "gap=%.4f, candidates=%d",
                decision.reason, decision.top_score,
                decision.gap, decision.candidates_count,
            )
            if debug_logger is not None:
                debug_logger.log_rerank_skipped(
                    query_preview=query,
                    candidates_count=decision.candidates_count,
                    reason=decision.reason,
                    top_score=decision.top_score,
                    gap=decision.gap,
                    source="unified_search",
                )
            return candidates[:top_k]

    docs = [text for _, _, text in candidates]
    try:
        if timer is not None:
            timer.start("rerank_ms")
        # mode 別 instruction を reranker へ伝搬
        ranked = await reranker.rerank(query, docs, top_k, mode=mode)
    except Exception as e:  # noqa: BLE001 — 失敗時は元順序にフォールバック
        logger.warning("Reranker failed in unified_search: %s", e)
        return merged[:top_k]
    finally:
        if timer is not None:
            timer.stop("rerank_ms")

    if not ranked:
        return merged[:top_k]
    # rerank スコアで置換しつつ chunk_id / text を保持
    return [
        (candidates[idx][0], float(score), candidates[idx][2])
        for idx, score in ranked
        if 0 <= idx < len(candidates)
    ][:top_k]


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

    リランカー / グローバルスコアソートが特定カートリッジを完全に除外する
    挙動を補正することが目的。複数カートリッジが存在しないケースでは
    `final` をそのまま返す (no-op)。

    既に `final` の長さが `top_k` 未満なら末尾追加し、満杯の場合は最低
    スコアの非カートリッジ要素または別カートリッジで重複代表されている
    要素を差し替える。
    """
    if not cart_results or not final:
        return final

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
        chunk = cart_top_chunk[cart_id]
        if chunk[0] in existing_ids:
            continue
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
    output_carts: set[str],
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


def _merge_results(
    *result_lists: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    """結果をマージしてスコア降順でソート（重複排除）"""
    seen: set[str] = set()
    merged: list[tuple[str, float, str]] = []

    for results in result_lists:
        for chunk_id, score, text in results:
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append((chunk_id, score, text))

    merged.sort(key=lambda x: -x[1])
    return merged


async def _expand_and_research(
    query: str,
    query_vec: np.ndarray,
    working_mem,
    long_term,
    top_k: int,
    noise_sigma: float = 0.05,
) -> list[tuple[str, float, str]]:
    """ルールベースのクエリ拡張で再検索（LLM なし）"""
    # 会話コンテキストからキーワードを抽出して拡張
    context_keywords: list[str] = []
    for turn in working_mem.get_context()[-3:]:
        content = turn.get("content", "")
        words = content.split()[:5]
        context_keywords.extend(words)

    if not context_keywords or long_term is None:
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
