"""llama-server ``/v1/rerank`` 経由の再順位スコア (c_16 §7.2.1)。

cross-encoder のリランカー (例: japanese-bge-reranker-v2-m3) を ``--reranking`` で
起動した専用 llama-server (既定 :8083) に、クエリ 1 本と候補文書を渡してスコアを得る。

``rag.rerank.mode: on`` のとき、検索パイプライン (``unified_search`` の Step 6.8) が floor を
通った corpus の候補を渡して並べ替える。

**チャット経路の待ち時間に直接乗る** ので、CLAUDE.md §6 #10 のリトライ (3 回) の例外として
試行 1 回 + 締切 (``deadline_ms``) で打ち切る (docs/e_03 §4.4)。失敗・締切超過・同値の
スコア (旧リランカーを撤去させた退化) は ``None`` を返し、呼び手は cosine 順のまま続ける。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from backend.free.llm._base_client import BaseHTTPClient
from backend.log_config import get_logger

logger = get_logger("rag.rerank_llamacpp")

RERANK_PATH = "/v1/rerank"
HEALTH_PATH = "/health"

#: スコアの幅 (最大 − 最小) がこれ未満なら「全候補同値」とみなす (退化)。
FLAT_SCORE_EPSILON = 1e-3
#: 全スコアの絶対値がこれ未満なら「全部 0」とみなす (旧リランカーの 0.0 / ~1e-22)。
ZERO_SCORE_EPSILON = 1e-6
#: rag JSONL に残す上位スコアの件数。
TOP_SCORES_LOGGED = 5
#: 遮断器: 直近この回数の結果が窓。窓が埋まるまでは遮断しない。
BREAKER_WINDOW = 10
#: 窓の失敗率がこれ以上なら遮断する。
BREAKER_FAILURE_RATE = 0.7
#: 遮断して HTTP を送らない秒数。明けたら 1 回だけ試す。
BREAKER_COOLDOWN_S = 600
#: 起動時にサーバへ届かなかったときの冷却秒 (``open_breaker``)。読み込み中のサーバを待つ間隔で、
#: 明けた試しが失敗しても同じ秒数で開き直し、復帰した時点で ``BREAKER_COOLDOWN_S`` に戻す。
BREAKER_STARTUP_COOLDOWN_S = 30
#: 接続の確立だけの上限秒 (読み取りは締切のまま)。相手は localhost なので接続は数 ms で済み、Windows は
#: 閉じたポートへの接続拒否を SYN 再送で 1 秒前後遅らせる (実測 1010 ms で締切に化けた) — 早く諦めて区別する。
RERANK_CONNECT_TIMEOUT_S = 0.25
#: 1 組 (query + doc) をサーバへ送るときの近似トークン数の上限。rerank サーバの ``-ub 2048`` は 1 組の
#: 最大長以上が必須で、超えると要求全体が 500 になる (``scripts/launch_llama.py``)。クエリ 400 +
#: 文書 1400 + 特殊トークン 4 で 2048 に 1 割強の余裕を残す。
RERANK_DOCUMENT_TOKENS = 1400
#: クエリの近似トークン数の上限。理由は ``RERANK_DOCUMENT_TOKENS`` と同じ。
RERANK_QUERY_TOKENS = 400
#: 近似の重み (1/10 トークン単位の整数)。XLM-R のトークナイザは CJK ≒ 1 字 1 トークン、ASCII ≒ 3.5〜4 字
#: 1 トークン — 非 ASCII は 1.0、ASCII は多め (超過しない側) の 0.3 に見積もる。
_NON_ASCII_WEIGHT_TENTHS = 10
_ASCII_WEIGHT_TENTHS = 3


def approx_rerank_tokens(text: str) -> float:
    """リランカーのトークナイザで見た ``text`` の近似トークン数 (非 ASCII 1.0 / ASCII 0.3)。"""
    n_ascii = len(text.encode("ascii", "ignore"))
    return ((len(text) - n_ascii) * _NON_ASCII_WEIGHT_TENTHS + n_ascii * _ASCII_WEIGHT_TENTHS) / 10


def clip_to_rerank_tokens(text: str, budget: int) -> str:
    """近似トークン数 (:func:`approx_rerank_tokens`) が ``budget`` を超えないよう先頭から切る。"""
    if approx_rerank_tokens(text) <= budget:
        return text
    limit = budget * 10
    used = 0
    for i, ch in enumerate(text):
        used += _ASCII_WEIGHT_TENTHS if ch.isascii() else _NON_ASCII_WEIGHT_TENTHS
        if used > limit:
            return text[:i]
    return text


def rerank_pair_tokens(query: str, document: str) -> float:
    """1 組 (query + document) を送るときの近似トークン数 (送る前に切った後の長さで数える)。"""
    return approx_rerank_tokens(clip_to_rerank_tokens(query, RERANK_QUERY_TOKENS)) + approx_rerank_tokens(
        clip_to_rerank_tokens(document, RERANK_DOCUMENT_TOKENS),
    )


#: 1 組ごとに近似トークンへ足す固定費 (近似トークン換算)。本番の実測 (n=152) で 1 組 4.6 ms ÷ 0.204 ms/実tok ≈ 22.5 と
#: 区切りの特殊トークン 4 の計 26.5 実トークンを、文書の 実 ≈ 0.819 × 近似 で近似へ直した値 (クエリは近似に含まれる)。
RERANK_PAIR_OVERHEAD_TOKENS = 32.0


def rerank_pair_cost(query: str, document: str) -> float:
    """予算で数える 1 組のコスト (:func:`rerank_pair_tokens` + :data:`RERANK_PAIR_OVERHEAD_TOKENS`)。"""
    return rerank_pair_tokens(query, document) + RERANK_PAIR_OVERHEAD_TOKENS


def fit_rerank_pool(
    query: str, documents: Sequence[str], *, max_count: int, token_budget: float | None,
) -> int:
    """``documents`` の先頭から何件を送るか。件数の上限とトークン予算の早い方で止める。

    1 回の処理時間は送るトークン数と組の数にほぼ比例する (c_16 §7.2.1) ので、各組を
    :func:`rerank_pair_cost` で数え、短い文書なら上限まで、長い文書なら予算で止める。
    ``token_budget`` が ``None`` なら件数の上限だけ。
    """
    limit = max(0, min(int(max_count), len(documents)))
    if token_budget is None:
        return limit
    used = 0.0
    for i in range(limit):
        used += rerank_pair_cost(query, documents[i])
        if used > token_budget:
            return i
    return limit


def build_rerank_payload(query: str, documents: Sequence[str]) -> dict[str, Any]:
    """``/v1/rerank`` の要求本文 (全件のスコアを返させる)。

    クエリと各文書は近似トークン数の上限 (``RERANK_QUERY_TOKENS`` / ``RERANK_DOCUMENT_TOKENS``) で
    先頭から切る。切るのは送る文字列だけで、呼び手の本文 (注入される本文) は変えない。
    """
    return {
        "query": clip_to_rerank_tokens(query, RERANK_QUERY_TOKENS),
        "documents": [clip_to_rerank_tokens(doc, RERANK_DOCUMENT_TOKENS) for doc in documents],
        "top_n": len(documents),
    }


def parse_rerank_response(data: Any, n_docs: int) -> tuple[list[float], int | None]:
    """``/v1/rerank`` の応答を **入力順** のスコア列にする。

    llama-server は ``results`` をスコア降順に並べて返すことがあるので ``index`` で戻す。
    件数・index・スコアの型が合わなければ ``ValueError`` (呼び手は失敗として扱う)。

    Returns:
        (入力順のスコア, ``usage.prompt_tokens`` — 無ければ ``None``)
    """
    if not isinstance(data, Mapping):
        raise ValueError("rerank response is not an object")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("rerank response has no results list")
    scores: list[float | None] = [None] * n_docs
    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("rerank result is not an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n_docs:
            raise ValueError(f"rerank result index out of range: {index!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"rerank result score is not a finite number: {score!r}")
        if scores[index] is not None:
            raise ValueError(f"rerank result index duplicated: {index}")
        scores[index] = float(score)
    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise ValueError(f"rerank response lacks scores for {len(missing)} of {n_docs} documents")
    usage = data.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        prompt_tokens = None
    return [float(s) for s in scores if s is not None], prompt_tokens


def degenerate_reason(
    scores: Sequence[float], *, documents: Sequence[str] | None = None,
) -> str | None:
    """スコアが退化していれば理由 (``all_zero`` / ``flat``)、健全なら ``None``。

    ``all_zero`` は 1 件でも判定する — 健全な cross-encoder の logit が絶対値 1e-6 未満に
    なることはまず無く、旧リランカーの退化 (0.0 / ~1e-22) はそのまま 1 件でも現れる
    (sleep-time の疑似クエリの採点は 1 件ずつ送る、c_16 §7.2.1 の第 5 段階 B)。
    ``flat`` は比べる相手が要るので 2 件以上のときだけ。``documents`` (``scores`` と同じ並び) を
    渡すと、同じ本文は最初の 1 件だけで比べる — 同じ本文の同点は正当で、一意な本文が 2 件未満なら
    ``flat`` を判定しない。
    """
    if not scores:
        return None
    if all(abs(s) < ZERO_SCORE_EPSILON for s in scores):
        return "all_zero"
    if documents is not None and len(documents) == len(scores):
        first: dict[str, float] = {}
        for doc, score in zip(documents, scores):
            first.setdefault(doc, score)
        scores = list(first.values())
    if len(scores) < 2:
        return None
    if max(scores) - min(scores) < FLAT_SCORE_EPSILON:
        return "flat"
    return None


class RerankClient(BaseHTTPClient):
    """rerank 用 llama-server のクライアント (EvorefGen)。

    自己テストの結果 (配置・1 件あたり ms) と、そこから決めた候補の詰め方を属性に持つ —
    呼び手 (検索経路の Step 6.8 / ``search_history``) は :func:`fit_rerank_pool` で
    ``candidates`` (件数の上限) と ``token_budget`` (近似トークンの予算、``None`` なら件数だけ) の
    早い方まで渡す。
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8083,
        deadline_ms: int = 1000,
        candidates: int = 20,
        token_budget: float | None = None,
        placement: str = "",
        ms_per_doc: float | None = None,
        debug_logger=None,
        pseudo_query_calibration: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # httpx 自身の timeout は締切より少し長く取り、打ち切りは asyncio.wait_for に一本化する。
        # 接続だけは短く切り、届かないサーバを締切超過と区別する。
        super().__init__(
            timeout=httpx.Timeout(deadline_ms / 1000.0 + 1.0, connect=RERANK_CONNECT_TIMEOUT_S),
        )
        self.base_url = f"http://{host}:{port}"
        self.deadline_ms = int(deadline_ms)
        self.candidates = int(candidates)
        self.token_budget = token_budget
        self.placement = placement
        self.ms_per_doc = ms_per_doc
        self.debug_logger = debug_logger
        #: 疑似クエリの品質検査の較正 (``pseudo_query_gate.PseudoQueryGateCalibration``、
        #: c_17 §3.15)。起動時にこのモデルの model_key の項目を読む。``None`` なら捨てない。
        self.pseudo_query_calibration = pseudo_query_calibration
        self._clock = clock
        #: 直近の結果 (True=成功 / False=None を返した)。遮断器の窓。
        self._window: deque[bool] = deque(maxlen=BREAKER_WINDOW)
        #: 遮断を始めた時刻 (``clock`` の値)。遮断していなければ ``None``。
        self._opened_at: float | None = None
        #: 今の遮断の冷却秒 (窓で開けば ``BREAKER_COOLDOWN_S``、``open_breaker`` は指定値)。
        self._cooldown_s: float = BREAKER_COOLDOWN_S
        #: 冷却後の試し 1 回の席。持ち主の呼びを識別するトークン (空きなら ``None``)。
        self._trial_token: object | None = None

    @property
    def breaker_open(self) -> bool:
        """遮断中か (冷却が明けて試しを待つ間も遮断中として数える)。"""
        return self._opened_at is not None

    @property
    def breaker_remaining_s(self) -> float | None:
        """冷却の残り秒 (遮断していなければ ``None``、冷却が明けていれば 0)。"""
        if self._opened_at is None:
            return None
        return max(0.0, self._opened_at + self._cooldown_s - self._clock())

    @property
    def breaker_blocking(self) -> bool:
        """今 ``rerank`` (``breaker=True``) を呼べば遮断器に弾かれるか。

        冷却が残る間と、冷却明けの試しが飛行中の間だけ真。冷却が明けて試しの席が空いていれば偽
        (呼び手が送らないと試しが起きず復帰しない)。
        """
        if self._opened_at is None:
            return False
        return self._trial_token is not None or self._clock() < self._opened_at + self._cooldown_s

    def open_breaker(self, cooldown_s: float) -> None:
        """遮断した状態にする (起動時にサーバへ届かなかったとき)。明けたら試し 1 回で復帰を確かめる。"""
        self._opened_at = self._clock()
        self._cooldown_s = float(cooldown_s)
        self._window.clear()
        self._trial_token = None

    def reset_breaker(self) -> None:
        """遮断器を閉じて窓を空にする (サーバの起動 / 再起動 / 再接続の後)。

        飛行中の試しの席も捨てる — 前のサーバへの試しの結果で遮断し直さないため。
        """
        if self._opened_at is not None:
            logger.info("Rerank circuit reset after the server was (re)started")
        self._opened_at = None
        self._cooldown_s = BREAKER_COOLDOWN_S
        self._window.clear()
        self._trial_token = None

    def _breaker_admit(self) -> tuple[bool, object | None]:
        """(送ってよいか, 試しの席のトークン)。冷却明けは試し 1 回だけ通し、その呼びだけがトークンを持つ。"""
        if self._opened_at is None:
            return True, None
        if self.breaker_blocking:
            return False, None
        token = object()
        self._trial_token = token
        logger.info("Rerank circuit half-open; trying one request")
        return True, token

    def _release_trial(self, token: object | None) -> None:
        """試しの席を返す (持ち主のトークンのときだけ。試し以外の呼びは席に触れない)。"""
        if token is not None and self._trial_token is token:
            self._trial_token = None

    def _breaker_record(self, ok: bool, reason: str, token: object | None) -> None:
        if token is not None:
            if self._trial_token is not token:
                return
            self._trial_token = None
            if ok:
                self._opened_at = None
                self._cooldown_s = BREAKER_COOLDOWN_S
                self._window.clear()
                logger.info("Rerank circuit closed; reranker recovered")
            else:
                self._opened_at = self._clock()
                logger.warning(
                    "Rerank circuit re-opened (%s); skipping rerank for %d s", reason, self._cooldown_s,
                )
            return
        if self._opened_at is not None:
            return  # 遮断中に終わった遅れた呼びは窓にも冷却にも触れない
        self._window.append(ok)
        if len(self._window) < BREAKER_WINDOW:
            return
        failures = self._window.count(False)
        if failures / BREAKER_WINDOW >= BREAKER_FAILURE_RATE:
            self._opened_at = self._clock()
            self._cooldown_s = BREAKER_COOLDOWN_S
            logger.warning(
                "Rerank circuit opened (%d of last %d failed, last: %s); skipping rerank for %d s",
                failures, BREAKER_WINDOW, reason, BREAKER_COOLDOWN_S,
            )

    async def health_check(self) -> bool:
        """``/health`` が 200 か (リトライしない、docs/e_03 §4.1)。"""
        try:
            resp = await self._get_http_client().get(
                f"{self.base_url}{HEALTH_PATH}", timeout=httpx.Timeout(2.0, connect=RERANK_CONNECT_TIMEOUT_S),
            )
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def rerank(
        self, query: str, documents: Sequence[str], *, ids: Sequence[str] | None = None,
        breaker: bool = True,
    ) -> list[float] | None:
        """``documents`` の入力順のスコア。失敗・締切超過・退化は ``None`` (試行 1 回)。

        ``ids`` (``documents`` と同じ並び) を渡すと、並べ替えの前後の id 順と上位スコアを
        rag JSONL に残す (``op="rerank"``)。

        ``breaker=False`` は遮断器を通らない呼び (sleep-time の疑似クエリ採点)。遮断中でも送り、
        窓にも数えず、半開の試しの席も取らない — chat 経路の遮断判断を背景処理が動かさないため。
        """
        if not documents:
            return []
        started = time.perf_counter()
        allowed, trial = self._breaker_admit() if breaker else (True, None)
        if not allowed:
            logger.debug("Rerank skipped (circuit_open) for %d documents", len(documents))
            if self.debug_logger is not None:
                self.debug_logger.log_rerank(
                    n_in=len(documents), elapsed_ms=0.0, ok=False, reason="circuit_open",
                    prompt_tokens=None, deadline_ms=self.deadline_ms,
                    ids_before=list(ids) if ids is not None else None, ids_after=None, top_scores=None,
                )
            return None
        reason = ""
        prompt_tokens: int | None = None
        scores: list[float] | None = None
        recorded = False
        payload = build_rerank_payload(query, documents)
        try:
            try:
                resp = await asyncio.wait_for(
                    self._get_http_client().post(f"{self.base_url}{RERANK_PATH}", json=payload),
                    timeout=self.deadline_ms / 1000.0,
                )
                resp.raise_for_status()
                parsed, prompt_tokens = parse_rerank_response(resp.json(), len(documents))
                bad = degenerate_reason(parsed, documents=payload["documents"])
                if bad is None:
                    scores = parsed
                else:
                    reason = f"degenerate_{bad}"
            except TimeoutError:
                reason = "deadline"
            except httpx.HTTPStatusError as e:
                reason = f"http_{e.response.status_code}"
            except (httpx.ConnectError, httpx.ConnectTimeout):
                reason = "connect_error"
            except httpx.HTTPError as e:
                reason = f"http_error:{type(e).__name__}"
            except ValueError as e:
                reason = "bad_response"
                logger.debug("rerank response rejected: %s", e)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if breaker:
                self._breaker_record(scores is not None, reason, trial)
            recorded = True
        finally:
            if trial is not None and not recorded:
                self._release_trial(trial)  # 例外・キャンセルでも試しの席を必ず返す
        if scores is None:
            logger.info(
                "Rerank skipped (%s) after %.0f ms for %d documents; keeping cosine order",
                reason, elapsed_ms, len(documents),
            )
        if self.debug_logger is not None:
            ids_before = list(ids) if ids is not None else None
            ids_after = top_scores = None
            if ids_before is not None and scores is not None and len(ids_before) == len(scores):
                order = sorted(range(len(scores)), key=lambda i: -scores[i])
                ids_after = [ids_before[i] for i in order]
                top_scores = [round(scores[i], 4) for i in order[:TOP_SCORES_LOGGED]]
            self.debug_logger.log_rerank(
                n_in=len(documents),
                elapsed_ms=elapsed_ms,
                ok=scores is not None,
                reason=reason,
                prompt_tokens=prompt_tokens,
                deadline_ms=self.deadline_ms,
                ids_before=ids_before,
                ids_after=ids_after,
                top_scores=top_scores,
            )
        return scores


__all__ = [
    "BREAKER_COOLDOWN_S",
    "BREAKER_FAILURE_RATE",
    "BREAKER_STARTUP_COOLDOWN_S",
    "BREAKER_WINDOW",
    "FLAT_SCORE_EPSILON",
    "RERANK_CONNECT_TIMEOUT_S",
    "RERANK_DOCUMENT_TOKENS",
    "RERANK_PAIR_OVERHEAD_TOKENS",
    "RERANK_PATH",
    "RERANK_QUERY_TOKENS",
    "ZERO_SCORE_EPSILON",
    "RerankClient",
    "approx_rerank_tokens",
    "build_rerank_payload",
    "clip_to_rerank_tokens",
    "degenerate_reason",
    "fit_rerank_pool",
    "parse_rerank_response",
    "rerank_pair_cost",
    "rerank_pair_tokens",
]
