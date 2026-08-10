"""アシスト llama-server のオンデマンド常駐管理 (docs/c_14 §1.2)

``assist_model.residency: on_demand`` (既定) のとき、アシスト llama-server は
**チャット応答パスでは起動も実行もしない**。実際にアシストが要るのは

- アイドル窓 (Full sleep-time / Level 1 / Level 2 学習)
- create モード (staged パイプライン)

の 2 つだけなので、その間だけプロセスを起動し、用が済んだら停止する。
チャット中の GPU 帯域と VRAM をベースモデルへ明け渡すのが目的。

設計上の要点:

1. **``AssistModelClient`` オブジェクトは差し替えない**。``_pillar_wirer`` が
   20 箇所以上へコンストラクタ注入しており、``AppState.set_assist_client()``
   が同期するのは 4 つだけなので、停止のたびに ``None`` へ倒すと残りは stale な
   参照を掴む。プロセスだけを起動・停止し、クライアントは同一オブジェクトの
   まま生かして ``allows()`` ゲートで fast-fail させる。
   ``BaseHTTPClient._get_http_client`` は閉じた接続プールを遅延再生成するため、
   停止時に ``aclose()`` しても次回そのまま使える (死んだ keep-alive を捨てる
   効果もある)。
2. **新しいキューは作らない**。アイドル判定 (``SleepTimeScheduler``) と
   プロセス制御 (``server_control``) は既にあるので、それらを繋ぐだけ。
3. **無限リトライしない**。起動失敗は ``failed`` + バックオフで次の窓まで待つ。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState
    from backend.free.llm.assist_client import AssistModelClient

logger = get_logger("llm.assist_residency")

ResidencyState = Literal["stopped", "starting", "ready", "stopping", "failed"]

#: 起動に失敗した後、次の ``acquire`` を受け付けるまでのバックオフ (秒)。
#: アイドル窓は 10 分間隔で回るため、失敗した窓を 1 回飛ばす程度の値にする。
FAILURE_BACKOFF_SEC = 600.0


class AssistResidencyManager:
    """アシスト llama-server の起動・停止と、呼出可否ゲートを一元管理する。

    ``always`` モードでは全メソッドが従来挙動 (常駐前提) の薄いパススルーに
    なり、``on_demand`` のときだけプロセス制御が働く。
    """

    def __init__(
        self,
        state: "AppState",
        cfg: dict[str, Any],
        *,
        on_first_ready: Callable[["AssistModelClient"], Awaitable[None]] | None = None,
    ) -> None:
        """
        Args:
            state: ``assist_client`` を保持する AppState。
            cfg: 実体 config。``assist_model.residency`` 等を読む。
            on_first_ready: 初回 ready 時に 1 度だけ呼ばれるフック。
                ``update_params_from_server`` と capability probe の起動を
                ``_pillar_wirer`` 側から注入するために使う (Gen pillar を
                factory 層に依存させないため)。
        """
        self._state = state
        self._cfg = cfg
        self._on_first_ready = on_first_ready

        assist_cfg = cfg.get("assist_model", {}) or {}
        self._configured = bool(
            assist_cfg.get("enabled", True) and assist_cfg.get("local"),
        )
        self._on_demand = (
            self._configured
            and str(assist_cfg.get("residency", "on_demand")) == "on_demand"
        )
        self._stop_after_batch = bool(assist_cfg.get("stop_after_batch", True))
        self._start_timeout = float(assist_cfg.get("idle_start_timeout_sec", 120.0))

        self._lock = asyncio.Lock()
        self._status: ResidencyState = "stopped"
        self._refcount = 0
        self._holders: dict[str, int] = {}
        self._stop_requested = False
        self._retry_after = 0.0
        self._first_ready_done = False
        self._start_task: asyncio.Task[bool] | None = None
        # モデル差し替え中など「プロセスは管理下だが今は呼べない」状態。
        # residency mode に関係なくゲートを閉じる。
        self._suspended = False

    # ── 状態参照 ──────────────────────────────────────────────

    @property
    def on_demand(self) -> bool:
        """オンデマンド制御が有効か (``always`` / 未設定なら False)。"""
        return self._on_demand

    @property
    def status(self) -> ResidencyState:
        """現在の常駐状態。``always`` では常に ``"ready"`` を返す。"""
        if self._suspended:
            return "stopping"
        if not self._on_demand:
            return "ready" if self._state.assist_client is not None else "stopped"
        return self._status

    @property
    def holders(self) -> dict[str, int]:
        """現在 residency を掴んでいる理由と件数 (観測用のコピー)。"""
        return dict(self._holders)

    def is_ready(self) -> bool:
        """アシストへ実際にリクエストを投げてよい状態か。"""
        if self._state.assist_client is None:
            return False
        if self._suspended:
            return False
        if not self._on_demand:
            return True
        return self._status == "ready"

    def suspend(self, reason: str) -> None:
        """モデル差し替え等の間、呼出を一時的に拒否する (``always`` でも効く)。

        chat⇄create のアシストモデル差し替え中は llama-server が落ちている。
        以前はここで ``state.assist_client = None`` にしていたが、コンストラクタ
        注入済みの参照が stale になるため、クライアントは生かしたままゲートだけ
        閉じる (docs/c_14 §1.2)。
        """
        self._suspended = True
        logger.info("Assist residency: suspended (reason=%s)", reason)

    def resume(self, reason: str) -> None:
        """``suspend`` を解除する。"""
        if not self._suspended:
            return
        self._suspended = False
        logger.info("Assist residency: resumed (reason=%s)", reason)

    def allows(self, purpose: str) -> bool:  # noqa: ARG002 - purpose は将来の粒度制御用
        """``AssistModelClient`` が呼出直前に引くゲート。

        ``on_demand`` では residency が ready でない限り一律で拒否する。
        purpose 別の例外は設けない — チャット応答パスの purpose だけを
        通すとプロセスを起動する必要が生じ、設計目的 (チャット中は起動しない)
        と矛盾するため。
        """
        return self.is_ready()

    # ── 取得・解放 ────────────────────────────────────────────

    async def acquire(self, reason: str) -> bool:
        """アシストを ready にして参照カウントを 1 増やす。

        既に ready なら起動せずカウントだけ増やす。``always`` モードでは
        クライアントの有無をそのまま返す (プロセス制御はしない)。

        Returns:
            ready になったか。False の場合カウントは増えない。
        """
        if not self._on_demand:
            return self._state.assist_client is not None
        if not self._configured:
            return False

        async with self._lock:
            if self._status == "ready":
                self._add_holder(reason)
                return True
            if self._status == "failed" and time.monotonic() < self._retry_after:
                logger.debug(
                    "Assist residency: acquire(%s) skipped (backoff for %.0fs more)",
                    reason, self._retry_after - time.monotonic(),
                )
                return False

            self._stop_requested = False
            self._start_task = asyncio.create_task(self._start(reason))
            try:
                started = await self._start_task
            except asyncio.CancelledError:
                # request_stop() による割込。呼出元 (アイドル窓) は中止する。
                logger.info("Assist residency: start cancelled (reason=%s)", reason)
                started = False
            finally:
                self._start_task = None

            if started:
                self._add_holder(reason)
            return started

    def note_external_start(self, reason: str) -> None:
        """呼出側が自前でアシストを起動済みのときに residency へ登録する。

        create モードの ``model_paths.assist_create_model`` 差し替えは
        ``/api/mode/switch`` の ``_restart_assist_server`` が model_override 付きで
        行うため、ここでプロセスを起こし直さず状態だけ引き継ぐ。
        """
        if not self._on_demand:
            return
        self._status = "ready"
        self._retry_after = 0.0
        self._stop_requested = False
        self._add_holder(reason)
        logger.info(
            "Assist residency: ready (external start, reason=%s, holders=%s)",
            reason, self._holders,
        )

    async def release(self, reason: str) -> None:
        """参照カウントを 1 減らし、0 になったら停止する。"""
        if not self._on_demand:
            return
        async with self._lock:
            self._remove_holder(reason)
            if self._refcount > 0:
                logger.debug(
                    "Assist residency: release(%s), still held by %s",
                    reason, self._holders,
                )
                return
            if not self._stop_after_batch:
                logger.debug(
                    "Assist residency: release(%s), keeping resident "
                    "(stop_after_batch=false)", reason,
                )
                return
            await self._stop(reason)

    async def request_stop(self, reason: str) -> None:
        """進行中の起動を打ち切り、即座に停止する (ユーザー入力の割込用)。

        ロックを取る前に起動タスクを cancel することで、``acquire`` が
        health 待ちの最中でも待たされずに停止できる。
        """
        if not self._on_demand:
            return
        self._stop_requested = True
        task = self._start_task
        if task is not None and not task.done():
            task.cancel()
        async with self._lock:
            self._holders.clear()
            self._refcount = 0
            await self._stop(reason)

    async def shutdown(self) -> None:
        """lifespan shutdown からの停止 (バックオフを無視して確実に落とす)。"""
        if not self._on_demand:
            return
        await self.request_stop("shutdown")

    # ── 内部 ──────────────────────────────────────────────────

    def _add_holder(self, reason: str) -> None:
        self._holders[reason] = self._holders.get(reason, 0) + 1
        self._refcount += 1

    def _remove_holder(self, reason: str) -> None:
        remaining = self._holders.get(reason, 0) - 1
        if remaining > 0:
            self._holders[reason] = remaining
        else:
            self._holders.pop(reason, None)
        self._refcount = max(0, self._refcount - 1)

    async def _start(self, reason: str) -> bool:
        """llama-server (assist) を起動し、既存クライアントで疎通を確認する。

        ``_lock`` 保持中に呼ばれる。クライアントは新規生成せず、
        ``state.assist_client`` の同一オブジェクトをそのまま使う。
        """
        client = self._state.assist_client
        if client is None:
            logger.warning(
                "Assist residency: no assist client wired — cannot start (reason=%s)",
                reason,
            )
            self._mark_failed()
            return False

        # server_control は API 層だが、プロセス制御の 3 経路フォールバックが
        # ここにしか無い。import 循環 (server_control → AppState) を避けるため
        # 関数内 import に留める (mode.py の _restart_assist_server と同じ形)。
        from backend.free.api.system.server_control import start_server_process

        self._status = "starting"
        t0 = time.monotonic()
        logger.info("Assist residency: starting assist server (reason=%s)", reason)

        try:
            ok, message = await asyncio.wait_for(
                start_server_process("assist", self._cfg),
                timeout=self._start_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Assist residency: start timed out after %.0fs (reason=%s)",
                self._start_timeout, reason,
            )
            await self._force_stop()
            self._mark_failed()
            return False

        if not ok:
            logger.warning(
                "Assist residency: start failed (reason=%s): %s", reason, message,
            )
            self._mark_failed()
            return False

        if self._stop_requested:
            logger.info("Assist residency: stop requested during start (reason=%s)", reason)
            await self._force_stop()
            self._status = "stopped"
            return False

        if not await client.health_check():
            logger.warning(
                "Assist residency: health check failed after start (reason=%s)", reason,
            )
            await self._force_stop()
            self._mark_failed()
            return False

        self._status = "ready"
        self._retry_after = 0.0
        logger.info(
            "Assist residency: ready in %.1fs (reason=%s, %s)",
            time.monotonic() - t0, reason, message,
        )

        if not self._first_ready_done and self._on_first_ready is not None:
            self._first_ready_done = True
            try:
                await self._on_first_ready(client)
            except Exception as e:
                # パラメータ同期 / capability probe の失敗で窓を潰さない。
                logger.warning("Assist residency: on_first_ready hook failed: %s", e)

        return True

    async def _stop(self, reason: str) -> None:
        """llama-server (assist) を停止する。``_lock`` 保持中に呼ばれる。"""
        if self._status in ("stopped", "stopping"):
            return
        self._status = "stopping"
        logger.info("Assist residency: stopping assist server (reason=%s)", reason)
        await self._force_stop()
        self._status = "stopped"

    async def _force_stop(self) -> None:
        """プロセス停止 + 接続プールの破棄。状態遷移は呼出側の責務。"""
        from backend.free.api.system.server_control import (
            stop_server_process,
            wait_port_released,
        )

        client = self._state.assist_client
        if client is not None:
            # オブジェクトは捨てない。_get_http_client が閉じたプールを
            # 遅延再生成するため、次回起動時はそのまま使える。
            try:
                await client.aclose()
            except Exception as e:
                logger.debug("Assist residency: client aclose failed: %s", e)

        try:
            await asyncio.to_thread(
                stop_server_process, "assist", self._cfg, self._state,
            )
            await asyncio.to_thread(wait_port_released, "assist", self._cfg, 10.0)
        except Exception as e:
            logger.warning("Assist residency: stop failed: %s", e)

    def _mark_failed(self) -> None:
        self._status = "failed"
        self._retry_after = time.monotonic() + FAILURE_BACKOFF_SEC
