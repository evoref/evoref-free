"""LLM クライアントの HTTP プラミング共通化

`LocalClient` / `AuxClient` / `LlamaCppEmbedder`
は llama-server への HTTP 呼び出しという共通の責務を持ち、以下のロジックが
重複していた:

- `httpx.AsyncClient` の遅延初期化と `aclose()` のライフサイクル管理
- 一時的な I/O 失敗 (ConnectError / ReadTimeout / RemoteProtocolError /
  502/503/504/429) に対するリトライループ
- 指数バックオフ + jitter による thundering herd 抑制

本モジュールはこれらを `BaseHTTPClient` 基底クラスと
`async_retry_http_call` ヘルパに集約する。リトライは ``tenacity`` (Apache
2.0 / 推移依存ゼロ) で宣言的に組み立てる。

リトライ対象 / 非対象の整理:
- **リトライ対象** (transient I/O 失敗): ``httpx.ConnectError`` /
  ``httpx.ReadTimeout`` / ``httpx.WriteTimeout`` / ``httpx.PoolTimeout`` /
  ``httpx.RemoteProtocolError`` / ``httpx.HTTPStatusError`` のうち
  ``retryable_statuses`` (既定 ``{429, 500, 502, 503, 504}``) に該当するもの
- **非リトライ対象**: 上記以外 (4xx の大半 / 500 以外の 5xx 等)。
  ``httpx.ConnectTimeout`` も含むが、これは ``ConnectError`` 経由で
  ``TimeoutException`` の派生として扱われる
- **生成 POST の例外**: ``LocalClient._generate_sync`` / ``generate_constrained`` /
  ``generate_with_logprobs`` は ``GENERATION_RETRYABLE_EXCEPTIONS`` を使い、
  推論中の ``ReadTimeout`` を再 POST しない (二重投入 / スロット KV 汚染 /
  purpose timeout の ×3 膨張を防ぐ)。``timeout × MAX_ATTEMPTS`` が壁時計の
  worst case になるのは接続フェーズの失敗が連続した場合のみ
- **健全性チェック例外**: 起動時 ``health_check`` はリトライしない
  (CLAUDE.md §1 「health_check 失敗は ``None`` 注入でデグラ継続」)。
  各クライアントの ``health_check`` は本ヘルパを経由せず単発で叩く

ストリーミング (`/v1/chat/completions` の SSE) はストリーム途中で
切断された場合に再開できないため本ヘルパの対象外。`LocalClient._generate_stream`
は単発呼び出しのまま、外側で TimeoutException → LLMTimeoutError に変換する。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.log_config import get_logger
from backend.trace_context import get_trace_id

logger = get_logger("llm._base_client")


# 共通リトライ定数 (LocalClient / AuxClient / Embedder で同一値)。
# ``stop_after_attempt(MAX_ATTEMPTS)`` で全試行回数 (初回含む) を制御する。
# 旧実装の ``MAX_RETRIES=3`` (= 4 試行) から ``MAX_ATTEMPTS=3`` (= 3 試行) に変更し、
# worst case のバックオフ待機 (0.5 + 1.0 + 2.0 = 3.5s + jitter) を抑える。
# 注意: ここで制御するのは試行回数とバックオフ待機のみで、``timeout`` は
# **per-attempt** に適用される。ただし生成 POST
# (``GENERATION_RETRYABLE_EXCEPTIONS``) は ``ReadTimeout`` をリトライしないため、
# ``timeout × MAX_ATTEMPTS`` の壁時計になり得るのは **接続フェーズの失敗**
# (ConnectError / WriteTimeout / PoolTimeout / 5xx・429) が連続した場合だけで、
# 推論そのものが遅い場合は 1 回の ``timeout`` で確定する。
MAX_ATTEMPTS = 3
RETRY_WAIT_INITIAL = 0.5  # 秒、wait_exponential_jitter の初期値
RETRY_WAIT_MAX = 4.0  # 秒、wait_exponential_jitter の上限

# llama-server / OpenAI 互換サーバ向けのデフォルトリトライ対象ステータス。
# 504 (Gateway Timeout) は llama-server 自体は出さないが、リバースプロキシ
# 経由の構成で発生し得るため含める。Anthropic 等で 529 (overloaded) を
# 含めたい場合は呼び出し側で指定する。
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# リトライ対象の transient I/O 例外。``httpx.ConnectError`` は llama-server
# 起動直後の一時的失敗、``ReadTimeout`` は推論中の応答停止、
# ``RemoteProtocolError`` はネットワーク瞬断や keep-alive 切断時に発生する。
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

# 生成 POST (``/v1/chat/completions`` の非ストリーミング) 用のリトライ対象。
# ``ReadTimeout`` を **含めない**: 推論が遅くて read が切れたリクエストを再 POST
# すると、llama-server 側では前の生成がまだ走っている (二重投入) うえ、chat
# スロットへ落ちた場合は KV を汚す。aux の purpose timeout も per-attempt なので
# 3 回で ×3 に膨れていた。read が切れた生成は 1 回で確定させ、呼出側の
# ``LLMTimeoutError`` 分岐に任せる。接続フェーズの失敗 (バイトが流れる前) だけ
# を再試行する。
GENERATION_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


class HealthLogGate:
    """health check の結果を **状態が変わったときだけ** ログするためのゲート。

    UI は ``/api/status`` と ``/api/system/vram_status`` を数秒間隔でポーリングし、
    1 回のポーリングが base / aux / embed の 3 系統の health check を起こす。
    毎回 DEBUG を出すと ``--develop`` 時の backend.log が同一文言で埋まり、実信号が
    埋没する (実測 2026-07-25: 1 日の DEBUG 約 16,600 行 = 全体の大半が
    ``Health check: status=200, healthy=True`` と ``GET /api/status`` の反復。
    調査のたびに grep -v で除外する必要があった)。

    同じ ``(status, healthy)`` が続く間は抑制し、抑制した回数を次の変化時に添える。
    """

    __slots__ = ("_last", "_suppressed")

    def __init__(self) -> None:
        self._last: tuple[int, bool] | None = None
        self._suppressed = 0

    def should_log(self, status: int, healthy: bool) -> bool:
        """前回と異なる結果なら ``True``。同一なら抑制して ``False``。"""
        key = (status, healthy)
        if self._last == key:
            self._suppressed += 1
            return False
        self._last = key
        return True

    def take_suppressed(self) -> int:
        """直前の変化までに抑制した件数を返し、カウンタを 0 に戻す。"""
        n = self._suppressed
        self._suppressed = 0
        return n


class BaseHTTPClient:
    """`httpx.AsyncClient` の遅延初期化とクローズを共通化する基底クラス

    サブクラスは ``__init__`` で ``super().__init__(timeout=...)`` を呼び、
    ``self._get_http_client()`` 経由で HTTP クライアントを取得する。
    シャットダウン時は ``await self.aclose()`` を呼ぶ。
    """

    def __init__(self, *, timeout: float) -> None:
        self._http_timeout: float = timeout
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """`httpx.AsyncClient` を遅延初期化して再利用する"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
        return self._http_client

    async def aclose(self) -> None:
        """HTTP クライアントを閉じる (シャットダウン時に呼ぶ)"""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None


T = TypeVar("T")


def _make_status_retry_predicate(
    retryable_statuses: frozenset[int] | set[int],
) -> Callable[[BaseException], bool]:
    """``HTTPStatusError`` のうち retryable status のみを True とする述語を返す。

    ``tenacity.retry_if_exception`` のコールバックは bool を期待するため、
    クロージャで ``retryable_statuses`` を束ねる。
    """

    def _predicate(exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in retryable_statuses
        return False

    return _predicate


def _make_before_sleep_callback(
    request_label: str,
    retry_logger: Callable[[int, BaseException, float], None] | None,
) -> Callable[[RetryCallState], None]:
    """tenacity の ``before_sleep`` コールバックを生成する。

    待機前に呼ばれ、attempt 番号と直前の例外を WARNING ログに残す。
    ``retry_logger`` が指定されていれば、追加で構造化ログ
    (``DebugLogger.log_retry_attempt`` 等) にも転送する。
    """

    def _before_sleep(state: RetryCallState) -> None:
        # state.next_action が None の場合 (= 最終試行) は呼ばれないので無視。
        if state.outcome is None or state.next_action is None:
            return
        exc = state.outcome.exception()
        if exc is None:
            return
        wait_sec = state.next_action.sleep
        attempt = state.attempt_number  # 1-based
        # status code が分かる場合はメッセージに含める
        status_part = ""
        if isinstance(exc, httpx.HTTPStatusError):
            status_part = f" (status={exc.response.status_code})"
        elif isinstance(exc, httpx.TimeoutException):
            status_part = " (timeout)"
        logger.warning(
            "%s failed%s, retry %d/%d, waiting %.2fs: %s",
            request_label, status_part,
            attempt, MAX_ATTEMPTS - 1, wait_sec,
            exc.__class__.__name__,
        )
        if retry_logger is not None:
            try:
                retry_logger(attempt, exc, wait_sec)
            except Exception:  # pragma: no cover - 安全側
                logger.debug("retry_logger raised; ignored", exc_info=True)

    return _before_sleep


async def async_retry_http_call(
    operation: Callable[[], Awaitable[T]],
    *,
    request_label: str = "HTTP request",
    retryable_statuses: frozenset[int] | set[int] = RETRYABLE_STATUS_CODES,
    retry_logger: Callable[[int, BaseException, float], None] | None = None,
    retryable_exceptions: tuple[type[BaseException], ...] = _RETRYABLE_EXCEPTIONS,
) -> T:
    """tenacity ベースの HTTP リトライヘルパ

    ``operation`` を ``MAX_ATTEMPTS`` 回まで試行する。
    リトライ対象は ``retryable_exceptions`` (既定 ``_RETRYABLE_EXCEPTIONS``) および
    ``HTTPStatusError`` のうち ``retryable_statuses`` に含まれるもの。
    その他の例外 (4xx / 非 retryable な 5xx 等) はキャッチせず即時伝播。
    生成 POST は ``retryable_exceptions=GENERATION_RETRYABLE_EXCEPTIONS`` を渡し、
    推論中の ``ReadTimeout`` を再 POST しない (二重投入の防止)。

    全試行が失敗した場合は最後に発生した例外を再 raise する
    (``reraise=True`` 動作)。呼び出し側は必要に応じて ``TimeoutException``
    などを独自例外に変換できる。

    Args:
        operation: 試行する非同期操作 (引数なしで T を返す)
        request_label: ログ出力時のラベル (例: ``"LLM request"``)。
            ``DebugLogger.log_retry_attempt`` の ``label`` フィールドにも
            転送される。
        retryable_statuses: リトライ対象とする HTTP ステータスコード集合。
            既定 ``RETRYABLE_STATUS_CODES``。
        retry_logger: 待機前に呼ばれる構造化ログコールバック。
            ``(attempt: int, exc: BaseException, wait_sec: float) -> None``。
            ``DebugLogger.log_retry_attempt`` を bind する想定。
        retryable_exceptions: リトライ対象とする例外型のタプル。
    """
    retry_predicate = (
        retry_if_exception_type(retryable_exceptions)
        | retry_if_exception(_make_status_retry_predicate(retryable_statuses))
    )

    last_exc: BaseException | None = None
    async for attempt in AsyncRetrying(
        retry=retry_predicate,
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential_jitter(
            initial=RETRY_WAIT_INITIAL, max=RETRY_WAIT_MAX,
        ),
        before_sleep=_make_before_sleep_callback(request_label, retry_logger),
        reraise=True,
    ):
        with attempt:
            try:
                return await operation()
            except BaseException as e:
                last_exc = e
                raise

    # AsyncRetrying は reraise=True で必ず例外を上げるか値を返すため、
    # ここに到達することは構造的にない。型チェッカ向けのフォールバック。
    assert last_exc is not None  # pragma: no cover
    raise last_exc  # pragma: no cover


def make_retry_logger(
    debug_logger,
    *,
    backend: str,
    purpose: str = "",
) -> Callable[[int, BaseException, float], None] | None:
    """``DebugLogger.log_retry_attempt`` を bind したリトライログ関数を返す。

    ``debug_logger`` が ``None`` または ``log_retry_attempt`` 属性を
    持たない場合は ``None`` を返し、``async_retry_http_call`` 内の
    構造化ログ送出を抑制する。

    本関数を経由することで、各クライアント側は
    ``DebugLogger`` の有無を意識せず ``retry_logger`` 引数を組み立てられる。
    """
    if debug_logger is None:
        return None
    if not hasattr(debug_logger, "log_retry_attempt"):
        return None

    def _emit(attempt: int, exc: BaseException, wait_sec: float) -> None:
        status: int | None = None
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
        # ``debug_logger.log_retry_attempt(...)`` を直接呼び出す
        # (test_debug_logger_coverage.py の AST 検出パターンに合わせるため)。
        debug_logger.log_retry_attempt(
            backend=backend,
            purpose=purpose,
            attempt=attempt,
            wait_sec=wait_sec,
            exception=type(exc).__name__,
            status_code=status,
            trace_id=get_trace_id() or "",
        )

    return _emit


# 起動時 probe 用定数。``scripts/launch_llama.py:wait_for_health`` の
# ``timeout=30`` / 1 秒間隔と挙動を揃える。
STARTUP_PROBE_TIMEOUT_SEC = 30.0
STARTUP_PROBE_INTERVAL_SEC = 1.0

#: 一度も TCP 接続が成立しない場合の打ち切り。サーバが起動していない構成で
#: 起動を長時間ブロックしないための別予算 (モデルサイズとは無関係)。
STARTUP_PROBE_NO_LISTEN_SEC = 30.0

#: GGUF 1GB あたりに見込むロード秒数。コールドキャッシュ + 低速ディスクでも
#: 待ち切れる値を採る (上限は ``STARTUP_PROBE_MAX_SEC``)。実測は 16.3GB / 33.2s
#: (約 2s/GB) だが、これは page cache が温まっていた測定なので余裕を持たせる。
STARTUP_PROBE_SEC_PER_GB = 15.0
STARTUP_PROBE_MAX_SEC = 600.0


async def wait_for_server_ready(
    health_url: str,
    *,
    timeout: float = STARTUP_PROBE_TIMEOUT_SEC,
    interval: float = STARTUP_PROBE_INTERVAL_SEC,
    label: str = "llama-server",
) -> bool:
    """``GET <health_url>`` をポーリングして 200 OK もしくはタイムアウトまで待つ。

    backend lifespan 起動時のレース対策。``fetch_model_metadata`` や
    ``health_check`` を呼ぶ前段で使用し、llama-server プロセスがまだ listen
    していない間の ``ConnectError`` / 503 を吸収して 200 を待つ。

    本ヘルパは例外を raise せず常に bool を返す。タイムアウト時は WARNING
    を 1 行出力するのみで、後続の ``health_check`` / ``fetch_model_metadata``
    の失敗パスがそのまま既存の degraded mode (``None`` 返却) を担う。

    Args:
        health_url: ``http://host:port/health`` 形式のヘルスチェック URL
        timeout: 総タイムアウト秒数 (既定 30 秒)
        interval: ポーリング間隔秒数 (既定 1 秒)
        label: ログに付与するサーバ識別子 (例: ``"llama-server (base)"``)

    Returns:
        200 OK を受信できれば ``True``、タイムアウトすれば ``False``。
    """
    loop = asyncio.get_event_loop()
    started = loop.time()
    attempt = 0
    # 一度でも TCP が繋がったか。繋がったうえで 503 が返るのは「モデルをロード
    # 中」で、待てば必ず開く。一度も繋がらないのは「そもそも起動していない」で、
    # いくら待っても開かない。両者を同じ上限で扱うと、モデルが大きい構成では
    # 起動レースに負け、サーバ不在の構成では起動が無駄に長時間ブロックされる。
    connected = False
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            attempt += 1
            try:
                resp = await client.get(health_url)
                connected = True
                if resp.status_code == 200:
                    if attempt > 1:
                        logger.info(
                            "%s ready after %d probe(s), %.1fs: %s",
                            label, attempt, loop.time() - started, health_url,
                        )
                    return True
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
            ):
                pass
            budget = timeout if connected else min(timeout, STARTUP_PROBE_NO_LISTEN_SEC)
            if loop.time() - started >= budget:
                logger.warning(
                    "%s not ready within %.1fs (%d probes, %s): %s",
                    label, budget, attempt,
                    "still loading" if connected else "never accepted a connection",
                    health_url,
                )
                return False
            await asyncio.sleep(interval)


def resolve_startup_probe_timeout(
    model_path: "str | Path | None", *, label: str = "llama-server",
) -> float:
    """モデルファイルサイズから起動プローブの上限を導く (純粋関数に近い I/O)。

    **固定 30 秒はモデルサイズに追随しない。** 実測 (2026-09-03 ライブ監査):
    16.3GB の Qwen3.8-27B Q4_K_M のロードに 33.2 秒かかり、30 秒の予算を
    2 桁パーセント超過した。その結果 ``fetch_model_metadata`` が 503 を掴んで
    ``local_client=None`` となり、**AuxClient がプロセス寿命の間ずっと
    未配線**になった (MetaCognitive の計画生成と CritiqueSynthesizer が丸ごと縮退)。

    上限を伸ばしても **正常系のコストはゼロ** — ``/health`` が 200 を返した
    瞬間に戻るので、待つのは実際にロード中の間だけ。サーバ不在で無駄に待つ
    ケースは ``STARTUP_PROBE_NO_LISTEN_SEC`` (接続が一度も成立しない場合の
    別予算) が抑える。

    サイズが読めない場合 (パス未解決 / 未ダウンロード) は既定へ倒す。
    """
    if not model_path:
        return STARTUP_PROBE_TIMEOUT_SEC
    try:
        size_bytes = Path(model_path).stat().st_size
    except OSError:
        return STARTUP_PROBE_TIMEOUT_SEC
    size_gb = size_bytes / (1024 ** 3)
    budget = max(
        STARTUP_PROBE_TIMEOUT_SEC,
        min(size_gb * STARTUP_PROBE_SEC_PER_GB, STARTUP_PROBE_MAX_SEC),
    )
    if budget > STARTUP_PROBE_TIMEOUT_SEC:
        logger.info(
            "%s startup probe budget %.0fs (model %.1f GB)", label, budget, size_gb,
        )
    return budget
