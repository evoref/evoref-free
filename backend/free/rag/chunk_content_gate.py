"""取得直後の chunk 内容精査ゲート (heuristics-first + 境界 assist)

``unified_search`` の Step 4 マージ直後に走り、低価値 chunk を pruning して
quality judge / query expansion の候補数を縮小する。
coding mode を主対象とし、安価なヒューリスティック (relevance floor /
近似重複除去 / コードシグナル) で大半を裁き、判断に迷う marginal band の
prose チャンクだけアシストモデルで 1 回だけ関連性を判定する
(``AssistJudgeUsageTracker`` で発火上限を抑制)。

degraded mode (``assist_client=None``) / cap 超過 / error / timeout では
ヒューリスティックのみで確定し、例外は決して投げない (応答パスを止めない)。

chat mode では近似重複除去のみを行い、relevance floor / コードシグナル /
assist は適用しない (チャットの recall を退行させないため)。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.free.core.session_mode import is_coding_mode
from backend.free.rag.bm25_retriever import tokenize_ja
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.rag.assist_judge_tracker import AssistJudgeUsageTracker

logger = get_logger("rag.chunk_content_gate")

# marginal band の assist 判定で各 chunk を切り詰める文字数 (self_rag_judge と同等)。
_TRUNCATE = 150
# dedup の token-set Jaccard を計算する候補数の上限。超過時は O(n^2) を避け skip。
_DEDUP_MAX = 50
# marginal band assist の wall-clock 上限 (秒)。超過時は全 keep にフォールバック。
_ASSIST_TIMEOUT_S = 5.0

# コード片を含むかの安価判定パターン (coding mode)。
_CODE_FENCE = re.compile(r"```")
_INDENT_BLOCK = re.compile(r"(?m)^[ \t]{4,}\S")
_CODE_KEYWORDS = re.compile(
    r"\b(?:def|class|return|import|function|const|let|var|public|private"
    r"|async|await|struct|enum|interface|namespace)\b"
)
_CODE_SYMBOLS = re.compile(r"=>|->|::|==|!=|[(){}\[\];]")

_GATE_INSTRUCTIONS = (
    "あなたは検索結果の関連性を判定するアシスタントです。"
    "以下の候補チャンクのうち、ユーザーのクエリに答えるうえで関連するものだけを選び、"
    "その 0 始まりインデックスを配列で返してください。"
    "関連しないチャンクは除外します。判断に迷う場合は含めてください。"
)


def _has_code_signal(text: str) -> bool:
    """安価なヒューリスティックで text がコード片を含むか判定する。

    フェンス / インデントブロックは単独で強いシグナル。それ以外は
    コードキーワードと記号/演算子の併発を要求し、散発的な記号のみの
    散文を誤検出しないようにする。
    """
    if not text:
        return False
    if _CODE_FENCE.search(text) or _INDENT_BLOCK.search(text):
        return True
    return bool(_CODE_KEYWORDS.search(text)) and bool(_CODE_SYMBOLS.search(text))


def _parse_relevant_indices(res: object, count: int) -> set[int] | None:
    """assist 応答から有効な relevant_indices 集合を取り出す。

    dict でない / list でない応答は ``None`` (= 判定不能 → 全 keep)。
    範囲外・bool は無視する。空 list は「関連なし」の正当な判定として
    空集合を返す (呼出側で marginal を drop)。
    """
    if not isinstance(res, dict):
        return None
    raw = res.get("relevant_indices")
    if not isinstance(raw, list):
        return None
    out: set[int] = set()
    for i in raw:
        if isinstance(i, bool):
            continue
        if isinstance(i, int) and 0 <= i < count:
            out.add(i)
    return out


@dataclass(frozen=True)
class GateConfig:
    """``rag.self_rag.content_gate`` の解決済み設定。"""

    enabled: bool = False
    relevance_floor: float = 0.45
    marginal_band: float = 0.10
    min_keep: int = 3
    dedup_jaccard: float = 0.85
    coding_code_signal: bool = True
    assist_enabled: bool = True
    max_per_session: int = 5
    max_per_query: int = 1

    @classmethod
    def from_rag_cfg(cls, rag_cfg: dict | None) -> GateConfig:
        """``rag.self_rag.content_gate`` を防御的に取り出す (欠落時は既定)。"""
        cfg = ((rag_cfg or {}).get("self_rag") or {}).get("content_gate") or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            relevance_floor=float(cfg.get("relevance_floor", 0.45)),
            marginal_band=float(cfg.get("marginal_band", 0.10)),
            min_keep=int(cfg.get("min_keep", 3)),
            dedup_jaccard=float(cfg.get("dedup_jaccard", 0.85)),
            coding_code_signal=bool(cfg.get("coding_code_signal", True)),
            assist_enabled=bool(cfg.get("assist_enabled", True)),
            max_per_session=int(cfg.get("max_per_session", 5)),
            max_per_query=int(cfg.get("max_per_query", 1)),
        )


@dataclass
class GateResult:
    """ゲート処理の結果 (debug ログ / テスト用の内訳付き)。"""

    kept: list[tuple[str, float, str]]
    dropped_floor: int = 0
    dropped_dedup: int = 0
    assist_used: bool = False
    assist_dropped: int = 0
    assist_skipped_reason: str = ""


class ChunkContentGate:
    """取得直後の chunk 内容精査ゲート。"""

    def __init__(self, cfg: GateConfig, *, debug_logger: DebugLogger | None = None):
        self._cfg = cfg
        self._debug_logger = debug_logger

    async def filter(
        self,
        query: str,
        merged: list[tuple[str, float, str]],
        mode: str,
        *,
        assist_client=None,
        tracker: AssistJudgeUsageTracker | None = None,
        session_id: str = "default",
    ) -> list[tuple[str, float, str]]:
        """heuristics-first prune + marginal band assist。

        ``merged`` は ``(chunk_id, score, text)`` の score 降順リスト。
        pruning 済みリスト (score 降順、``min_keep`` 未満にしない) を返す。
        ゲート無効 / 候補が ``min_keep`` 以下なら無加工で返す。例外は投げない。
        """
        cfg = self._cfg
        if not cfg.enabled or len(merged) <= cfg.min_keep:
            return merged
        try:
            result = await self._run(
                query, merged, mode, assist_client, tracker, session_id,
            )
        except Exception as e:
            logger.warning(
                "chunk content gate failed, returning merged unchanged: %s", e,
            )
            return merged
        if self._debug_logger is not None:
            self._debug_logger.log_content_gate(
                query_preview=query,
                mode=mode,
                input_count=len(merged),
                kept_count=len(result.kept),
                dropped_floor=result.dropped_floor,
                dropped_dedup=result.dropped_dedup,
                assist_used=result.assist_used,
                assist_dropped=result.assist_dropped,
                assist_skipped_reason=result.assist_skipped_reason,
            )
        return result.kept

    async def _run(
        self, query, merged, mode, assist_client, tracker, session_id,
    ) -> GateResult:
        cfg = self._cfg
        if not is_coding_mode(mode):
            # chat mode: 近似重複除去のみ (recall を退行させない)
            kept, dropped = self._dedup(merged)
            return GateResult(kept=kept, dropped_dedup=dropped)

        # coding mode: フル精査
        floor = cfg.relevance_floor
        band_hi = floor + cfg.marginal_band

        # (a) relevance floor + min_keep backfill (merged は score 降順)
        above = [c for c in merged if c[1] >= floor]
        if len(above) < cfg.min_keep:
            below = [c for c in merged if c[1] < floor]
            above = above + below[: cfg.min_keep - len(above)]
        dropped_floor = len(merged) - len(above)

        # (b) 近似重複除去
        deduped, dropped_dedup = self._dedup(above)

        # (c) 区分: clear (>=band_hi) は無条件 keep / code 持ち marginal は cheap keep /
        #     prose-only marginal は assist 判定対象
        clear_idx: set[int] = set()
        code_idx: set[int] = set()
        prose_idx: list[int] = []
        for i, (_cid, score, text) in enumerate(deduped):
            if score >= band_hi:
                clear_idx.add(i)
            elif cfg.coding_code_signal and _has_code_signal(text):
                code_idx.add(i)
            else:
                prose_idx.append(i)

        result = GateResult(
            kept=[], dropped_floor=dropped_floor, dropped_dedup=dropped_dedup,
        )

        # (d) marginal prose を境界 assist で判定
        kept_prose_idx = await self._judge_marginal(
            query, prose_idx, deduped, assist_client, tracker, session_id, result,
        )

        kept_idx = clear_idx | code_idx | kept_prose_idx
        # min_keep guard: deduped は score 降順なので上位から補填
        if len(kept_idx) < cfg.min_keep:
            for i in range(len(deduped)):
                if len(kept_idx) >= cfg.min_keep:
                    break
                kept_idx.add(i)
        result.kept = [deduped[i] for i in range(len(deduped)) if i in kept_idx]
        return result

    def _dedup(
        self, chunks: list[tuple[str, float, str]],
    ) -> tuple[list[tuple[str, float, str]], int]:
        """token-set Jaccard で近似重複を除去する (score 降順前提、高スコア優先)。"""
        cfg = self._cfg
        if cfg.dedup_jaccard >= 1.0 or len(chunks) <= 1 or len(chunks) > _DEDUP_MAX:
            return list(chunks), 0
        kept: list[tuple[str, float, str]] = []
        kept_sets: list[frozenset[str]] = []
        dropped = 0
        for cid, score, text in chunks:
            tok = frozenset(tokenize_ja(text, split_ascii=True))
            is_dup = False
            if tok:
                for ks in kept_sets:
                    inter = len(tok & ks)
                    if inter == 0:
                        continue
                    union = len(tok | ks)
                    if union and inter / union >= cfg.dedup_jaccard:
                        is_dup = True
                        break
            if is_dup:
                dropped += 1
            else:
                kept.append((cid, score, text))
                kept_sets.append(tok)
        return kept, dropped

    async def _judge_marginal(
        self, query, prose_idx, deduped, assist_client, tracker, session_id, result,
    ) -> set[int]:
        """prose-only marginal チャンクを assist で関連判定し、keep する deduped index 集合を返す。

        assist 無効 / 未接続 / cap 超過 / timeout / error / 不正応答では
        全 prose を keep する (heuristic-only, 安全側)。
        """
        cfg = self._cfg
        if not prose_idx:
            return set()
        if not cfg.assist_enabled or assist_client is None:
            result.assist_skipped_reason = (
                "assist_unavailable" if assist_client is None else "assist_disabled"
            )
            return set(prose_idx)
        if tracker is not None:
            quota = {
                "enabled": cfg.assist_enabled,
                "max_per_session": cfg.max_per_session,
                "max_per_query": cfg.max_per_query,
                "only_when_quality": ["content_gate"],
            }
            decision = tracker.check(
                session_id=session_id, namespace="content_gate",
                quality="content_gate",
                query_count=0, config=quota,
            )
            if not decision.allowed:
                result.assist_skipped_reason = decision.reason
                return set(prose_idx)

        formatted = "\n".join(
            f"[{n}] {deduped[idx][2][:_TRUNCATE]}"
            for n, idx in enumerate(prose_idx)
        )
        prompt = f"{_GATE_INSTRUCTIONS}\n\nクエリ: {query}\n\n候補:\n{formatted}"
        try:
            # timeout は generate_json 側の purpose 別 (realtime) 総予算強制に
            # 一本化する (外側 wait_for との二重ラップは反応的タイムアウト較正を
            # 機能不全にする。self_rag_judge.py と同じ不具合パターン)。
            res = await assist_client.generate_json(
                prompt, max_tokens=64, temperature=0.1,
                purpose="retrieval_chunk_gate", list_key="relevant_indices",
                timeout=_ASSIST_TIMEOUT_S,
            )
        except (TimeoutError, asyncio.TimeoutError):
            result.assist_skipped_reason = "timeout"
            return set(prose_idx)
        except Exception as e:
            logger.debug("content gate assist failed: %s", e)
            result.assist_skipped_reason = "error"
            return set(prose_idx)

        relevant_local = _parse_relevant_indices(res, len(prose_idx))
        if relevant_local is None:
            result.assist_skipped_reason = "invalid_response"
            return set(prose_idx)
        if tracker is not None:
            tracker.record(session_id, namespace="content_gate")
        result.assist_used = True
        kept = {prose_idx[n] for n in relevant_local}
        result.assist_dropped = len(prose_idx) - len(kept)
        return kept
