"""推論パイプライン: messages リスト組み立て"""

from __future__ import annotations

import re
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

# 動的コンテキストブロックと生クエリの境界に挟む固定文。
# few-shot 例 / 参考情報をユーザー発言と混同させないための区切り。固定文字列
# なのでターン内の prefix divergence (KV キャッシュ) には影響しない (どうせ tail)。
_DYNAMIC_CONTEXT_DELIMITER = (
    "\n\n---\n"
    "上記はシステムが用意した参考情報・応答例です。"
    "以下のユーザーの発言にのみ回答してください。\n\n"
)


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


#: few-shot ブロックの token 上限。無上限だとセッション中に数千 token へ膨張し
#: (2026-07-15: 2739 tokens → 3 回全ドロップ = 学習効果ゼロ、通常時も履歴予算を
#: 圧迫)、all-or-nothing ドロップで無意味化する。上限内へ例単位で切り詰める。
_FEWSHOT_TOKEN_CAP = 600

#: format_fewshot_section の例区切り (### Example N)
_FEWSHOT_EXAMPLE_SPLIT_RE = re.compile(r"(?=^### Example \d+$)", re.MULTILINE)


def _truncate_fewshot_to_budget(block: str, budget: int) -> str | None:
    """few-shot ブロックを例単位で budget token 内に切り詰める。

    ヘッダ (## Few-shot Examples) + 先頭から入る分の例だけを残す。
    1 例も入らない場合は None。
    """
    parts = _FEWSHOT_EXAMPLE_SPLIT_RE.split(block)
    if len(parts) <= 1:
        return block if _estimate_tokens(block) <= budget else None
    header, examples = parts[0], parts[1:]
    kept = header
    for ex in examples:
        candidate = kept + ex
        if _estimate_tokens(candidate) > budget:
            break
        kept = candidate
    if kept.strip() == header.strip():
        return None
    return kept.rstrip() + "\n"


def _select_fewshot_block(
    fewshot_block: str | None,
    remaining: int,
) -> tuple[str | None, int]:
    """few-shot ブロックを予算内へ切り詰めて採用する。

    ``format_fewshot_section`` の出力は先頭に改行を含むため lstrip して返す
    (動的ブロックの先頭要素になるため)。予算は ``remaining`` と
    ``_FEWSHOT_TOKEN_CAP`` の小さい方。超過分は例単位で部分注入する
    (all-or-nothing ドロップだと膨張時に学習効果がゼロになる)。
    """
    if not fewshot_block or remaining <= 0:
        return None, remaining
    block = fewshot_block.lstrip("\n")
    if not block:
        return None, remaining
    budget = min(remaining, _FEWSHOT_TOKEN_CAP)
    cost = _estimate_tokens(block)
    if cost > budget:
        truncated = _truncate_fewshot_to_budget(block, budget)
        if truncated is None:
            logger.debug(
                "fewshot block dropped (%d tokens > budget %d, no example fits)",
                cost, budget,
            )
            return None, remaining
        new_cost = _estimate_tokens(truncated)
        logger.debug(
            "fewshot block truncated: %d -> %d tokens (budget %d)",
            cost, new_cost, budget,
        )
        return truncated, remaining - new_cost
    logger.debug("fewshot block injected: %d tokens", cost)
    return block, remaining - cost


def _prepend_dynamic_block(trimmed: list[dict], dyn_text: str) -> bool:
    """trimmed の最後の user メッセージ content 先頭に動的ブロックを前置する。

    最後の user メッセージを **新しい dict で置換** する (入力要素は mutate しない。
    ``_trim_history`` が返す未圧縮ターンは history の dict 参照を共有するため)。
    動的ブロックは ``dyn_text + デリミタ + 元 content`` の順で、生クエリは末尾に残る
    (deliberative のツール結果はさらにその後ろへ追記されるため両立)。

    user メッセージが見つからなければ ``False`` を返す (呼び出し側が system へ fallback)。
    """
    for i in range(len(trimmed) - 1, -1, -1):
        if trimmed[i].get("role") == "user":
            original = trimmed[i].get("content", "")
            new_content = dyn_text + _DYNAMIC_CONTEXT_DELIMITER + original
            trimmed[i] = {**trimmed[i], "content": new_content}
            return True
    return False


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
    fewshot_block: str | None = None,
    history_min_tokens: int = 0,
) -> list[dict]:
    """
    messages リストを組み立て、トークン予算内に収める。

    テンプレート変換は LocalClient に委譲するため、
    ここではロール・内容の組み立てのみを担う。

    予算 = context_size - generation_reserve から
    system → 最新 user ターン予約 → fewshot → file_contexts → semmem → rag_chunks
    → history (残余) の優先順で配分。最新 user ターン (現在の質問) は動的ブロックに
    先立って ``min(質問トークン, working_max_tokens, 残予算)`` を予約し、履歴トリムで
    消失しないことを保証する (組み立て結果は history が user を含む限り user ロール ≥ 1)。

    サリエンスランカーが指定されている場合、RAG チャンクを5因子スコアで
    再評価し、トークン予算内で情報量を最大化するチャンク集合を選別する。

    **KV キャッシュ対応レイアウト**: system メッセージは ``system_prompt`` のみ
    (query 非依存・静的) とし、query 依存の動的部 (few-shot / file / semmem / RAG)
    は最後の user メッセージの content 先頭に前置する。これにより llama-server の
    prefix KV キャッシュが ``system + 過去履歴`` の範囲で再利用され、再プリフィルは
    最後の user ターンのみに限定される。gemma 系の「system は先頭 1 個」制約は維持。

    Args:
        system_prompt: インスタンス名プレフィックス付きの静的システムプロンプト。
        file_contexts: ファイルコンテキストのリスト。
            各要素は {"filename": str, "chunks": list[str]} の辞書。
        context_size: コンテキストウィンドウサイズ（トークン数）。
        max_tokens: 生成予約トークン数。None の場合は 512 をデフォルトとする。
        rag_scored_chunks: スコア付き検索結果 [(chunk_id, score, text), ...]。
            salience_ranker と同時に指定した場合、rag_chunks より優先される。
        salience_ranker: BudgetMem 式サリエンスランカー。
        fewshot_block: ``format_fewshot_section`` 整形済みの few-shot ブロック
            (query 依存)。動的ブロックの先頭に置く。``None`` / 空なら付与しない。
        history_min_tokens: 過去履歴の最低確保トークン数 (床)。動的ブロック配分前に
            予約し、予算圧迫時でも直近の会話文脈が丸ごと締め出されるのを防ぐ。
            実履歴量・残予算・working_max_tokens でキャップされ、履歴が現在の質問
            のみの場合は 0 に縮退する。0 (既定) で無効。
    """
    generation_reserve = max_tokens if max_tokens is not None else DEFAULT_GENERATION_RESERVE
    budget = context_size - generation_reserve
    total_rag = len(rag_chunks) if rag_chunks else 0
    total_fc = len(file_contexts) if file_contexts else 0

    sys_tokens = _estimate_tokens(system_prompt)
    remaining = budget - sys_tokens

    # 最新 user ターン (現在の質問) のトークンを動的ブロック配分に先立って予約する。
    # 予約は残予算を上限とし、収まらない分は _trim_history の最新ターン圧縮保持が吸収する。
    reserved_latest = 0
    if history and history[-1].get("role") == "user":
        reserved_latest = min(
            _estimate_tokens(history[-1].get("content", "")),
            working_max_tokens,
            max(0, remaining),
        )
        remaining -= reserved_latest

    # 過去履歴の最低確保 (床)。実履歴量・残予算・working_max でキャップし、
    # 履歴が現在の質問のみ (新規セッション) の場合は 0 に縮退する —
    # 空回りの予約で動的ブロックを痩せさせない。
    hist_floor = 0
    if history_min_tokens > 0 and len(history) > 1:
        past_tokens = sum(
            _estimate_tokens(t.get("content", "")) for t in history[:-1]
        )
        hist_floor = min(
            history_min_tokens,
            past_tokens,
            max(0, remaining),
            max(0, working_max_tokens - reserved_latest),
        )
        remaining -= hist_floor

    logger.debug(
        "build_messages: budget=%d (context_size=%d - %d), "
        "system=%d tokens, reserved_latest=%d, remaining=%d, "
        "rag_chunks=%d, file_contexts=%d",
        budget, context_size, generation_reserve, sys_tokens, reserved_latest,
        remaining, total_rag, total_fc,
    )

    # 動的ブロック (query 依存) を優先順に積む。system には含めない。
    dyn_parts: list[str] = []

    # 1. few-shot 例（query 依存、動的ブロック先頭）
    fewshot_part, remaining = _select_fewshot_block(fewshot_block, remaining)
    if fewshot_part:
        dyn_parts.append(fewshot_part)

    # 2. ファイルコンテキスト（ユーザー明示 → RAG より優先）
    fc_block, remaining, injected_fc = _inject_file_contexts(
        file_contexts, remaining, total_fc,
    )
    if fc_block:
        dyn_parts.append(fc_block)

    # 2.5 セマンティックメモリ注入（MemoryInjector、RAG より優先）
    semmem_part, remaining = _select_semmem_block(semmem_block, remaining)
    if semmem_part:
        dyn_parts.append(semmem_part)

    # 3. RAG チャンク（サリエンス優先 → フォールバック）
    rag_block, remaining, injected_rag = _select_rag_block(
        rag_chunks, rag_scored_chunks, salience_ranker, remaining, total_rag,
    )
    if rag_block:
        dyn_parts.append(rag_block)

    # 静的 system メッセージ (動的部は含めない)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # 4. 会話履歴（予約分 + 床 + 残余予算。上限は working_max_tokens）
    history_budget = min(
        working_max_tokens, reserved_latest + hist_floor + max(0, remaining),
    )
    if len(history) > 1 and history_budget <= reserved_latest:
        logger.warning(
            "build_messages: context budget squeeze — past history got 0 tokens "
            "(budget=%d, system=%d, reserved_latest=%d, fewshot=%d, files=%d, "
            "semmem=%d, rag=%d)",
            budget, sys_tokens, reserved_latest,
            _estimate_tokens(fewshot_part) if fewshot_part else 0,
            _estimate_tokens(fc_block) if fc_block else 0,
            _estimate_tokens(semmem_part) if semmem_part else 0,
            _estimate_tokens(rag_block) if rag_block else 0,
        )
    trimmed = _trim_history(history, history_budget)

    # 5. 動的ブロックを最後の user メッセージ先頭へ前置 (KV キャッシュ対応)。
    #    user ターンが無い場合のみ従来どおり system へ結合して情報を落とさない。
    if dyn_parts:
        dyn_text = "\n\n".join(dyn_parts)
        if not _prepend_dynamic_block(trimmed, dyn_text):
            messages[0] = {
                "role": "system",
                "content": system_prompt + "\n\n" + dyn_text,
            }

    messages.extend(trimmed)

    # 不変則ガード: history に user ターンがあるのに組み立て結果に user が無い場合、
    # 圧縮した最新 user ターンを末尾へ再掲する (予約 + 最新ターン保持により通常経路では
    # 到達しない最終防衛線。発火は契約違反のシグナル。末尾が assistant の履歴では
    # 時系列順が崩れるが、user 不在によるテンプレート 400 の回避を優先する)。
    if not any(m.get("role") == "user" for m in messages):
        last_user = next(
            (t for t in reversed(history) if t.get("role") == "user"), None,
        )
        if last_user is not None:
            recovered = compress_turn(last_user)
            messages.append(recovered)
            logger.warning(
                "build_messages: no user turn survived assembly; re-appended "
                "compressed latest user turn (%d tokens)",
                _estimate_tokens(recovered.get("content", "")),
            )

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

    **最新ターン (通常は現在の user 質問) は予算超過でも drop しない**: 予算に
    収まる長さへ圧縮して保持する (下限は compress_turn 既定の 200 字 ≈ 203 トークン
    で、超過分は呼び出し側の generation_reserve が吸収する)。

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
            if not result:
                # 最新ターンは drop せず、予算連動で圧縮して保持する
                # (この分岐では total_tokens == 0 のため残予算 = max_tokens)。
                compressed = compress_turn(turn, max_chars=max(max_tokens, 200))
                compressed_tokens = _estimate_tokens(compressed.get("content", ""))
                result.insert(0, compressed)
                total_tokens += compressed_tokens
                logger.warning(
                    "_trim_history: latest turn exceeds budget, kept compressed "
                    "(role=%s, %d->%d tokens, max=%d)",
                    turn.get("role"), estimated_tokens, compressed_tokens,
                    max_tokens,
                )
            else:
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

