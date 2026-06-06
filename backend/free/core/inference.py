"""推論パイプライン: messages リスト組み立て"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.free.api.chat.chat_constants import (
    DEFAULT_CONTEXT_SIZE, DEFAULT_GENERATION_RESERVE,
    DEFAULT_MAX_TOKENS, DEFAULT_WORKING_MAX_TOKENS,
)
from backend.config import resolve_context_size
from backend.log_config import get_logger
from backend.utils import compress_turn, estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from backend.free.core.salience_ranker import SalienceRanker

logger = get_logger("core.inference")


_RAG_HEADER = "以下の参考情報を踏まえて回答してください:\n\n"
_FILES_HEADER = "以下のファイルがコンテキストとして提供されています:\n\n"


def _build_file_section(fc: dict) -> str:
    """単一の file_context エントリを 1 セクション文字列へ整形する。"""
    filename = fc.get("filename", "unknown")
    chunks = fc.get("chunks", [])
    if chunks:
        return f"[ファイル: {filename}]\n" + "\n\n".join(chunks)
    return f"[ファイル: {filename}]"


def _inject_file_contexts(
    file_contexts: list[dict] | None,
    remaining: int,
    total_fc: int,
) -> tuple[str | None, int, int]:
    """ファイルコンテキストブロックを構築する。

    予算内に収まる範囲でセクションを連結し、`(block, new_remaining, injected_count)`
    を返す。注入対象がなければ `(None, remaining, 0)`。
    """
    if not file_contexts or remaining <= 0:
        return None, remaining, 0

    file_sections: list[str] = []
    for fc in file_contexts:
        section = _build_file_section(fc)
        cost = _estimate_tokens(section)
        if cost > remaining:
            logger.debug(
                "File context dropped: %s (%d tokens, remaining=%d)",
                fc.get("filename", "unknown"), cost, remaining,
            )
            break
        file_sections.append(section)
        remaining -= cost

    if not file_sections:
        return None, remaining, 0

    files_text = "\n\n---\n\n".join(file_sections)
    files_block = f"{_FILES_HEADER}{files_text}"
    # ヘッダー・セパレータのオーバーヘッドを補正
    overhead = (
        _estimate_tokens(files_block)
        - sum(_estimate_tokens(s) for s in file_sections)
    )
    remaining -= overhead
    injected = len(file_sections)
    logger.debug(
        "File contexts injected: %d/%d files, %d tokens",
        injected, total_fc, _estimate_tokens(files_block),
    )
    return files_block, remaining, injected


def _format_rag_block(entries: list[str]) -> str:
    """RAG 参照ブロックの最終的な system 文字列を生成する。"""
    return f"{_RAG_HEADER}" + "\n\n".join(entries)


def _inject_rag_salience(
    rag_scored_chunks: list[tuple[str, float, str]],
    salience_ranker: SalienceRanker,
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """サリエンスランカーで RAG チャンクを選別し block を返す。"""
    header_overhead = _estimate_tokens(_RAG_HEADER)
    chunk_budget = remaining - header_overhead
    ranked_texts = salience_ranker.rank(rag_scored_chunks, chunk_budget)
    if not ranked_texts:
        return None, remaining, 0

    selected_entries = [
        f"[参考情報 {i + 1}]\n{text}" for i, text in enumerate(ranked_texts)
    ]
    rag_block = _format_rag_block(selected_entries)
    rag_block_tokens = _estimate_tokens(rag_block)
    remaining -= rag_block_tokens
    injected = len(selected_entries)
    logger.debug(
        "RAG chunks (salience): %d/%d, %d tokens",
        injected, total_rag, rag_block_tokens,
    )
    return rag_block, remaining, injected


def _inject_rag_fallback(
    rag_chunks: list[str],
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """スコア降順 + 予算逐次選別で RAG block を構築する (フォールバック経路)。"""
    selected_entries: list[str] = []
    for i, chunk in enumerate(rag_chunks):
        entry = f"[参考情報 {i + 1}]\n{chunk}"
        cost = _estimate_tokens(entry)
        if cost > remaining:
            logger.debug(
                "RAG chunk %d/%d dropped (%d tokens, remaining=%d)",
                i + 1, total_rag, cost, remaining,
            )
            break
        selected_entries.append(entry)
        remaining -= cost

    if not selected_entries:
        return None, remaining, 0

    rag_block = _format_rag_block(selected_entries)
    overhead = (
        _estimate_tokens(rag_block)
        - sum(_estimate_tokens(e) for e in selected_entries)
    )
    remaining -= overhead
    injected = len(selected_entries)
    logger.debug(
        "RAG chunks injected: %d/%d, %d tokens",
        injected, total_rag, _estimate_tokens(rag_block),
    )
    return rag_block, remaining, injected


def _select_rag_block(
    rag_chunks: list[str] | None,
    rag_scored_chunks: list[tuple[str, float, str]] | None,
    salience_ranker: SalienceRanker | None,
    remaining: int,
    total_rag: int,
) -> tuple[str | None, int, int]:
    """サリエンス経路 / フォールバック経路を選んで RAG block を返す。"""
    if remaining <= 0:
        return None, remaining, 0
    if salience_ranker and rag_scored_chunks:
        return _inject_rag_salience(
            rag_scored_chunks, salience_ranker, remaining, total_rag,
        )
    if rag_chunks:
        return _inject_rag_fallback(rag_chunks, remaining, total_rag)
    return None, remaining, 0


def _select_semmem_block(
    semmem_block: str | None,
    remaining: int,
) -> tuple[str | None, int]:
    """MemoryInjector 由来のメモリブロックを予算内なら採用する。

    ブロックは MemoryInjector 側で既に tier 予算内に整形済みのため、
    ここでは全体 context に収まるかの二次チェックのみ行い、収まらなければ
    破棄する (部分切り出しはしない)。
    """
    if not semmem_block or remaining <= 0:
        return None, remaining
    labeled = f"[関連する記憶]\n{semmem_block}"
    cost = _estimate_tokens(labeled)
    if cost > remaining:
        logger.debug(
            "semmem block dropped (%d tokens > remaining %d)", cost, remaining,
        )
        return None, remaining
    logger.debug("semmem block injected: %d tokens", cost)
    return labeled, remaining - cost


def build_messages(
    system_prompt: str,
    history: list[dict],
    rag_chunks: list[str] | None = None,
    file_contexts: list[dict] | None = None,
    working_max_tokens: int = DEFAULT_WORKING_MAX_TOKENS,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_tokens: int | None = None,
    rag_scored_chunks: list[tuple[str, float, str]] | None = None,
    salience_ranker: SalienceRanker | None = None,
    semmem_block: str | None = None,
) -> list[dict]:
    """
    messages リストを組み立て、トークン予算内に収める。

    テンプレート変換は LocalClient に委譲するため、
    ここではロール・内容の組み立てのみを担う。

    予算 = context_size - generation_reserve から
    system → file_contexts → rag_chunks → history の優先順で配分。

    サリエンスランカーが指定されている場合、RAG チャンクを5因子スコアで
    再評価し、トークン予算内で情報量を最大化するチャンク集合を選別する。

    多くの LLM テンプレート（Qwen3.5 等）は system メッセージが
    先頭に 1 つだけであることを要求するため、system_prompt / file_contexts /
    rag_chunks をすべて単一の system メッセージに結合して返す。

    Args:
        system_prompt: インスタンス名プレフィックス付きのシステムプロンプト。
        file_contexts: ファイルコンテキストのリスト。
            各要素は {"filename": str, "chunks": list[str]} の辞書。
        context_size: コンテキストウィンドウサイズ（トークン数）。
        max_tokens: 生成予約トークン数。None の場合は 512 をデフォルトとする。
        rag_scored_chunks: スコア付き検索結果 [(chunk_id, score, text), ...]。
            salience_ranker と同時に指定した場合、rag_chunks より優先される。
        salience_ranker: BudgetMem 式サリエンスランカー。
    """
    generation_reserve = max_tokens if max_tokens is not None else DEFAULT_GENERATION_RESERVE
    budget = context_size - generation_reserve
    total_rag = len(rag_chunks) if rag_chunks else 0
    total_fc = len(file_contexts) if file_contexts else 0

    sys_parts: list[str] = [system_prompt]
    sys_tokens = _estimate_tokens(system_prompt)
    remaining = budget - sys_tokens

    logger.debug(
        "build_messages: budget=%d (context_size=%d - %d), "
        "system=%d tokens, remaining=%d, "
        "rag_chunks=%d, file_contexts=%d",
        budget, context_size, generation_reserve, sys_tokens, remaining,
        total_rag, total_fc,
    )

    # 2. ファイルコンテキスト（ユーザー明示 → RAG より優先）
    fc_block, remaining, injected_fc = _inject_file_contexts(
        file_contexts, remaining, total_fc,
    )
    if fc_block:
        sys_parts.append(fc_block)

    # 2.5 セマンティックメモリ注入（MemoryInjector、RAG より優先）
    semmem_part, remaining = _select_semmem_block(semmem_block, remaining)
    if semmem_part:
        sys_parts.append(semmem_part)

    # 3. RAG チャンク（サリエンス優先 → フォールバック）
    rag_block, remaining, injected_rag = _select_rag_block(
        rag_chunks, rag_scored_chunks, salience_ranker, remaining, total_rag,
    )
    if rag_block:
        sys_parts.append(rag_block)

    # 単一 system メッセージへ結合
    messages: list[dict] = [
        {"role": "system", "content": "\n\n".join(sys_parts)}
    ]

    # 4. 会話履歴（残余予算と working_max_tokens の小さい方で管理）
    history_budget = max(0, min(remaining, working_max_tokens))
    trimmed = _trim_history(history, history_budget)
    messages.extend(trimmed)

    logger.debug(
        "build_messages complete: %d messages "
        "(history %d→%d, rag %d/%d, files %d/%d)",
        len(messages), len(history), len(trimmed),
        injected_rag, total_rag, injected_fc, total_fc,
    )

    return messages


def build_messages_for_loop(
    system: str,
    history: list[dict],
    compacted_steps: list,
    cfg: dict,
) -> list[dict]:
    """Meta-Cognitive ループ用の messages 組み立て。

    build_messages と同じ予算管理だが、RAG チャンクの代わりに
    圧縮済みステップ結果を注入する。

    Args:
        system: システムプロンプト
        history: 会話履歴
        compacted_steps: StepResult のリスト（圧縮済み）
        cfg: config.yaml の辞書
    """
    ctx = resolve_context_size(cfg, "base")
    max_tok = cfg.get("llama", {}).get("max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_GENERATION_RESERVE
    generation_reserve = max_tok if max_tok else DEFAULT_GENERATION_RESERVE
    budget = ctx - generation_reserve
    remaining = budget - _estimate_tokens(system)

    step_parts: list[str] = []
    for s in compacted_steps:
        step_text = f"[Step {s.iteration + 1}: {s.tool_name}]\n{s.output}"
        step_tokens = _estimate_tokens(step_text)
        if remaining - step_tokens < 0:
            break
        step_parts.append(step_text)
        remaining -= step_tokens

    if step_parts:
        sys_content = system + "\n\n" + "\n\n".join(step_parts)
    else:
        sys_content = system

    working_max = cfg.get("memory", {}).get("working_max_tokens", DEFAULT_WORKING_MAX_TOKENS)
    trimmed_history = _trim_history(history, min(remaining, working_max))

    logger.debug(
        "build_messages_for_loop: %d steps injected, remaining=%d",
        len(step_parts), remaining,
    )

    return [{"role": "system", "content": sys_content}, *trimmed_history]


def _trim_history(
    history: list[dict],
    max_tokens: int,
) -> list[dict]:
    """トークン予算に収まるよう、古い履歴を圧縮・削除する。

    トークン推定は estimate_tokens() を使用
    （CJK: 1文字≒1トークン、ASCII: 4文字≒1トークン）。
    """
    if not history:
        logger.debug("_trim_history: empty history, nothing to trim")
        return []

    result: list[dict] = []
    total_tokens = 0

    # 新しいターンから逆順に追加
    for turn in reversed(history):
        content = turn.get("content", "")
        estimated_tokens = _estimate_tokens(content)

        if total_tokens + estimated_tokens > max_tokens:
            # 予算超過: 残りの古いターンを圧縮
            compressed = compress_turn(turn)
            compressed_tokens = _estimate_tokens(compressed["content"])
            if total_tokens + compressed_tokens <= max_tokens:
                result.insert(0, compressed)
                total_tokens += compressed_tokens
                logger.debug(
                    "_trim_history: compressed turn (role=%s, %d->%d tokens)",
                    turn.get("role"), estimated_tokens, compressed_tokens,
                )
            else:
                logger.debug(
                    "_trim_history: dropped turn (role=%s, %d tokens) — "
                    "budget exhausted (%d/%d)",
                    turn.get("role"), estimated_tokens, total_tokens, max_tokens,
                )
            break
        else:
            result.insert(0, turn)
            total_tokens += estimated_tokens

    logger.debug(
        "_trim_history: %d/%d turns kept, %d estimated tokens (max=%d)",
        len(result), len(history), total_tokens, max_tokens,
    )
    return result

