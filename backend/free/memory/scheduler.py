"""Sleep-time update スケジューラ

Trigger A: LLM 応答生成開始直後に Light 実行（§8.1）
Trigger B: 最終応答送信から full_idle_minutes 後に Full 実行
Trigger C: Level 1 独立常駐ループ
Trigger D: 夜間（schedule_hour）に Level 2 をトリガー
ユーザー入力で実行中の処理を協調的に yield
"""

import asyncio
import time
from datetime import timedelta, timezone, tzinfo
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.log_config import get_logger
from backend.trace_context import generate_trace_id, trace_id_var
from backend.utils import utc_now_dt

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("memory.scheduler")


def _emit_bg_task_outcome(
    debug_logger: "DebugLogger | None",
    *,
    task_name: str,
    success: bool,
    duration_ms: float,
    extra: dict | None = None,
) -> None:
    """bg_task wrapper の終了 outcome を ``outcome.jsonl`` に記録する

    ``evolve`` レベル限定。``debug_logger`` が ``None`` の場合は no-op。
    ``kind`` は ``"bg_task"`` 固定で、``quality_signals.task_name`` で
    light / full / level1_loop / level2_schedule を識別する。
    """
    if debug_logger is None:
        return
    signals: dict = {"task_name": task_name}
    if extra:
        signals.update(extra)
    debug_logger.log_outcome(
        kind="bg_task",
        success=success,
        duration_ms=duration_ms,
        quality_signals=signals,
    )


class SleepTimeScheduler:
    """Sleep-time update のスケジューリングと実行管理"""

    DEFAULT_LOCAL_TZ = "Asia/Tokyo"

    def __init__(self, config: dict, debug_logger: "DebugLogger | None" = None):
        learning = config.get("learning", {})
        self._debug_logger: "DebugLogger | None" = debug_logger
        schedule_cfg = config.get("schedule", {}) or {}
        self.local_tz_name: str = schedule_cfg.get("local_tz", self.DEFAULT_LOCAL_TZ)
        self._local_tz = self._resolve_local_tz(self.local_tz_name)
        # Trigger B: Full sleep-time のアイドル閾値
        self.full_idle_minutes: float = learning.get("full_idle_minutes", 10)
        # Trigger C: Level 1 のアイドル閾値（独立常駐ループから参照）
        self.level1_idle_minutes: float = learning.get("level1_idle_minutes", 30)
        # Trigger C: Level 1 ループの再評価間隔（秒）
        self.level1_recheck_interval_sec: float = float(
            learning.get("level1_recheck_interval_sec", 60)
        )
        self.active_minutes: float = learning.get("active_minutes", 5)
        # level2_schedule_hour は優先窓のヒント (_seconds_until_hour で利用可能)。
        # 実発火は再起動耐性のある overdue/idle 判定 (Level 2 常駐ループ) で行う。
        self.level2_schedule_hour: int = learning.get("level2_schedule_hour", 3)
        self.level2_recheck_interval_sec: float = float(
            learning.get("level2_recheck_interval_sec", 300)
        )
        self.level2_overdue_hours: float = float(
            learning.get("level2_overdue_hours", 24.0)
        )

        self._last_user_input: float = 0.0
        self._last_response: float = 0.0
        self._light_task: asyncio.Task | None = None
        self._full_task: asyncio.Task | None = None
        # Level 1 / Level 2 独立常駐ループ
        self._level1_loop_task: asyncio.Task | None = None
        self._level2_loop_task: asyncio.Task | None = None
        self._worker = None  # SleepTimeWorker, set via set_worker()
        self._llm_client = None  # LocalClient for Full mode
        self._assist_llm_client = None  # AssistModelClient for Full mode (preferred)
        self._learning_scheduler = None  # LearningScheduler for Level 1/2
        self._lora_path = None  # Path to LoRA adapter for Level 2
        self._base_model_path = None  # Path to base model GGUF for LoRA target resolution
        self._assist_lora_path = None  # Path to assist LoRA adapter for Level 2
        self._assist_model_path = None  # Path to assist model GGUF for LoRA target resolution
        self._running = False

    def _resolve_local_tz(self, tz_name: str) -> tzinfo:
        """設定 tz 名を tzinfo に解決する（解決不能時は段階的に縮退）

        Windows の ``zoneinfo`` は ``tzdata`` パッケージ無しだと IANA tz DB を
        参照できず、任意の tz 名で ``ZoneInfoNotFoundError`` になる。その場合に
        startup を落とさないよう、未知の tz 名 → 既定 (Asia/Tokyo) → UTC の順で
        縮退する。最終的に UTC に倒れた場合は ``self.local_tz_name`` も更新する。
        """
        try:
            return ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            pass
        if tz_name != self.DEFAULT_LOCAL_TZ:
            try:
                tz = ZoneInfo(self.DEFAULT_LOCAL_TZ)
                logger.warning(
                    "Unknown schedule.local_tz=%r, falling back to %s",
                    tz_name, self.DEFAULT_LOCAL_TZ,
                )
                self.local_tz_name = self.DEFAULT_LOCAL_TZ
                return tz
            except ZoneInfoNotFoundError:
                pass
        logger.warning(
            "Time zone database unavailable (tz=%r); install 'tzdata'. "
            "Falling back to UTC for schedule calculations.",
            tz_name,
        )
        self.local_tz_name = "UTC"
        return timezone.utc

    def set_worker(self, worker) -> None:
        """SleepTimeWorker を設定"""
        self._worker = worker

    def set_llm_client(self, client) -> None:
        """Full 版で使用する LLM クライアントを設定（ベースモデル、フォールバック用）"""
        self._llm_client = client

    def set_assist_llm_client(self, client) -> None:
        """Full 版で使用するアシストモデルクライアントを設定（優先）"""
        self._assist_llm_client = client

    def set_learning_scheduler(self, scheduler) -> None:
        """Level 1/2 学習スケジューラを設定"""
        self._learning_scheduler = scheduler
        if scheduler is not None:
            scheduler.set_user_active_checker(self.is_user_active)

    def set_lora_path(self, path) -> None:
        """Level 2 で使用する LoRA アダプタパスを設定"""
        self._lora_path = path

    def set_base_model_path(self, path) -> None:
        """Level 2 で使用するベースモデル GGUF パスを設定（LoRA ターゲット自動判定用）"""
        self._base_model_path = path

    def set_assist_lora_path(self, path) -> None:
        """Level 2 で使用するアシストモデル LoRA アダプタパスを設定"""
        self._assist_lora_path = path

    def set_assist_model_path(self, path) -> None:
        """Level 2 で使用するアシストモデル GGUF パスを設定"""
        self._assist_model_path = path

    def on_user_input(self) -> None:
        """ユーザー入力通知: 実行中のタスクをキャンセル"""
        self._last_user_input = time.time()

        # 実行中のワーカーをキャンセル
        if self._worker is not None:
            self._worker.cancel()

        # Light タスクをキャンセル
        if self._light_task is not None and not self._light_task.done():
            self._light_task.cancel()
            self._light_task = None
            logger.info("Light task cancelled by user input")

        # Full タスクをキャンセル
        if self._full_task is not None and not self._full_task.done():
            self._full_task.cancel()
            self._full_task = None
            logger.info("Full task cancelled by user input")

        # Level 1/2 学習を協調 yield
        # 現世代を完走させて Level1Session に進捗を残し、次回 tick で resume する
        if self._learning_scheduler is not None:
            self._learning_scheduler.cancel(graceful=True)
        # Level 2 常駐ループは停止しない (各 tick で is_user_active を見て自律 skip)。

    def on_llm_start(self) -> None:
        """LLM 応答生成開始通知: Trigger A (Light) を即座に実行（§8.1）

        ベースモデルの推論と並列に、アシストモデルで sleep-time ステップ1〜5を実行する。
        """
        if self._worker is None:
            return

        # 既存の Light タスクをキャンセル
        if self._light_task is not None and not self._light_task.done():
            self._light_task.cancel()

        self._light_task = asyncio.create_task(
            self._run_light()
        )

    def on_response_sent(self) -> None:
        """応答完了通知: Trigger B (Full) タイマーをリセット（§8.1）"""
        self._last_response = time.time()

        if self._worker is None:
            return

        # 既存の Full タスクをキャンセル
        if self._full_task is not None and not self._full_task.done():
            self._full_task.cancel()

        self._full_task = asyncio.create_task(
            self._schedule_full()
        )
        # Level 2 は on_response_sent では起動しない。再起動耐性のある
        # 独立常駐ループ (start_level2_loop) が overdue/idle で発火する。

    def is_user_active(self) -> bool:
        """ユーザーがフォアグラウンド対応中かどうか

        判定順:
        1. ベース／アシスト LLM の in_flight_chat_count > 0 → アクティブ
        2. _last_user_input から active_minutes 未満 → アクティブ
        3. それ以外 → 非アクティブ

        in_flight_chat_count を主条件とすることで、「メッセージ送信中」と
        「応答を読んでいる短い無動作期間」を正しく区別できる。
        """
        for client in (self._llm_client, self._assist_llm_client):
            if client is None:
                continue
            try:
                if int(getattr(client, "in_flight_chat_count", 0)) > 0:
                    return True
            except Exception:
                # Mock やテストダブル等で属性アクセスが失敗しても致命的にはしない
                continue

        if self._last_user_input == 0:
            return False
        elapsed = time.time() - self._last_user_input
        return elapsed < self.active_minutes * 60

    async def _run_light(self) -> None:
        """Trigger A: LLM 生成開始直後に Light 版を即座に実行（遅延なし）"""
        # bg_task wrapper として trace_id を発行し、終了時に
        # outcome.jsonl へ結末記録 (evolve 限定)。
        token = trace_id_var.set(generate_trace_id())
        t_start = time.monotonic()
        success = False
        cancelled = False
        try:
            if self._worker is None:
                success = True  # no-op は成功扱い
                return

            logger.info("Trigger A: starting Light sleep-time update (on_llm_start)")
            self._running = True
            await self._worker.run_light()
            success = True
        except asyncio.CancelledError:
            logger.debug("Light task cancelled")
            cancelled = True
            raise
        except Exception as e:
            logger.error("Light sleep-time update failed: %s", e, exc_info=True)
        finally:
            self._running = False
            elapsed_ms = (time.monotonic() - t_start) * 1000
            _emit_bg_task_outcome(
                self._debug_logger,
                task_name="light",
                success=success,
                duration_ms=elapsed_ms,
                extra={"cancelled": cancelled},
            )
            trace_id_var.reset(token)

    async def _schedule_full(self) -> None:
        """Trigger B: full_idle_minutes 後に Full 版を実行する。

        以降: Level 1 のキックは行わない。Level 1 は
        独立常駐ループ `_schedule_level1_loop` で判定される（f_04 §5.3）。
        """
        # bg_task wrapper として trace_id を発行し、終了時に
        # outcome.jsonl へ結末記録 (evolve 限定)。
        token = trace_id_var.set(generate_trace_id())
        t_start = time.monotonic()
        success = False
        cancelled = False
        executed = False
        skipped_reason: str | None = None
        try:
            await asyncio.sleep(self.full_idle_minutes * 60)

            if self._worker is None:
                success = True
                skipped_reason = "no_worker"
                return

            # ユーザーがアクティブなら実行しない
            if self.is_user_active():
                logger.info("Trigger B: skipped, user is active")
                success = True
                skipped_reason = "user_active"
                return

            logger.info("Trigger B: starting Full sleep-time update")
            self._running = True
            executed = True
            # sleep-time の LLM ステージはアシストモデルでのみ実行する。
            # アシスト未接続 (degraded) 時はベースモデルへフォールバックせず
            # None を渡し、run_full が Step 5.8-10 をクリーンスキップする
            # (c_14 §6.2: degraded はメモリ抽出をスキップし次回起動で再実行)。
            # ベースモデルにメモリ抽出/要約/判定をさせない不変則
            # (CLAUDE.md §6.1) とも整合する。run_full の各 LLM 呼出は
            # purpose= 付きで LocalClient が受けられないため、フォールバックは
            # 元々 TypeError を握り潰す死に配線だった。
            llm_for_sleep = self._assist_llm_client
            await self._worker.run_full(llm_for_sleep)
            success = True

        except asyncio.CancelledError:
            logger.debug("Full schedule cancelled")
            cancelled = True
            raise
        except Exception as e:
            logger.error("Full sleep-time update failed: %s", e, exc_info=True)
        finally:
            self._running = False
            elapsed_ms = (time.monotonic() - t_start) * 1000
            extra: dict = {"cancelled": cancelled, "executed": executed}
            if skipped_reason is not None:
                extra["skipped_reason"] = skipped_reason
            _emit_bg_task_outcome(
                self._debug_logger,
                task_name="full",
                success=success,
                duration_ms=elapsed_ms,
                extra=extra,
            )
            trace_id_var.reset(token)

    # ── Level 1 独立常駐ループ (f_04 §5.3) ───────

    def start_level1_loop(self) -> None:
        """Level 1 独立常駐ループを起動する。

        アプリ起動時に一度だけ呼ぶ（lifespan startup）。既に起動済みなら no-op。
        ループ自体はシャットダウンまで生き続け、`level1_recheck_interval_sec`
        ごとに以下の判定を行う:
            (1) SUSPENDED な session があれば最優先で resume
            (2) 優先キュー要求があれば実行
            (3) 通常トリガー: level1_idle_minutes アイドル + 経験数達成
        """
        if self._level1_loop_task is not None and not self._level1_loop_task.done():
            return
        self._level1_loop_task = asyncio.create_task(self._schedule_level1_loop())
        logger.info(
            "Level 1 loop started (interval=%.0fs, idle_threshold=%.1fmin)",
            self.level1_recheck_interval_sec, self.level1_idle_minutes,
        )

    async def stop_level1_loop(self) -> None:
        """Level 1 常駐ループを停止する（lifespan shutdown 用）"""
        if self._level1_loop_task is None or self._level1_loop_task.done():
            return
        self._level1_loop_task.cancel()
        try:
            await self._level1_loop_task
        except (asyncio.CancelledError, Exception):
            pass
        self._level1_loop_task = None
        logger.info("Level 1 loop stopped")

    def is_level1_idle(self) -> bool:
        """Level 1 用のアイドル判定（level1_idle_minutes 基準）"""
        if self._last_user_input == 0:
            return False
        elapsed = time.time() - self._last_user_input
        return elapsed >= self.level1_idle_minutes * 60

    async def _schedule_level1_loop(self) -> None:
        """常駐ループ。f_04 §5.3 の判定順序に従って Level 1 を起動する。

        bg_task wrapper として最外殻に trace_id を発行し
        ループ停止 (cancel) 時に outcome.jsonl へ結末記録する。
        ループ内部の Level 1 cycle ごとの outcome は ``_run_level1`` /
        ``run_or_resume_level1`` 側で別 trace_id を発行して emit される。
        """
        token = trace_id_var.set(generate_trace_id())
        t_start = time.monotonic()
        cancelled_ok = False
        logger.info("Level 1 loop entering main wait")
        while True:
            try:
                await asyncio.sleep(self.level1_recheck_interval_sec)

                ls = self._learning_scheduler
                if ls is None:
                    continue

                llm_for_level1 = self._llm_client or self._assist_llm_client
                if llm_for_level1 is None:
                    # LLM 未接続。次 tick で再評価
                    continue

                # (1) SUSPENDED session を最優先で resume
                if ls.has_active_session():
                    if self.is_user_active():
                        continue
                    logger.info("Level 1 loop: resuming SUSPENDED session")
                    await ls.run_or_resume_level1(
                        llm_client=llm_for_level1,
                        reason="resume",
                        relax_threshold=False,
                    )
                    continue

                # (2) 優先キューを処理
                req = ls.peek_priority_request()
                if req is not None:
                    if self.is_user_active():
                        continue
                    logger.info(
                        "Level 1 loop: processing priority request reason=%s",
                        req.reason,
                    )
                    result = await ls.run_or_resume_level1(
                        llm_client=llm_for_level1,
                        reason=req.reason,
                        relax_threshold=req.relax_ratio < 1.0,
                    )
                    # yield されなかった場合のみキューから取り除く
                    if not result.get("yielded", False):
                        ls.pop_priority_request()
                    continue

                # (3) 通常トリガー: idle + 経験数
                if not self.is_level1_idle():
                    continue
                if self.is_user_active():
                    continue
                if not ls.has_enough_experiences(strict=True):
                    continue
                # 前回 Level 1 完了以降の新規経験が閾値未満なら同じデータでの
                # 空回しを避けるためスキップ
                if not ls.has_enough_new_experiences():
                    logger.debug(
                        "Level 1 loop: skipped, not enough NEW experiences "
                        "since last run"
                    )
                    continue
                logger.info("Level 1 loop: starting scheduled idle session")
                await ls.run_or_resume_level1(
                    llm_client=llm_for_level1,
                    reason="idle",
                    relax_threshold=False,
                )

            except asyncio.CancelledError:
                logger.info("Level 1 loop cancelled")
                cancelled_ok = True
                break
            except Exception as e:
                logger.error("Level 1 loop iteration failed: %s", e, exc_info=True)
                # ループ自体は継続（次 interval で再評価）

        # bg_task wrapper の終了 outcome を emit
        elapsed_ms = (time.monotonic() - t_start) * 1000
        _emit_bg_task_outcome(
            self._debug_logger,
            task_name="level1_loop",
            # 常駐ループは「停止 (cancel)」が正常終了。例外で抜けた場合は False。
            success=cancelled_ok,
            duration_ms=elapsed_ms,
            extra={"cancelled": cancelled_ok},
        )
        trace_id_var.reset(token)

    # ── Level 2 独立常駐ループ (再起動耐性) ───────

    def start_level2_loop(self) -> None:
        """Level 2 独立常駐ループを起動する (lifespan startup で一度だけ)。

        旧実装は 03:00 まで ``asyncio.sleep`` するため、それ以前の再起動で
        毎回キャンセルされ Level 2 が永遠に発火しなかった。本ループは
        ``level2_recheck_interval_sec`` ごとに overdue (前回実行から
        ``level2_overdue_hours`` 超過) + 非アクティブを判定して発火するため、
        再起動を跨いでも ``_last_level2_run`` (永続化) を基準に継続する。
        """
        if self._level2_loop_task is not None and not self._level2_loop_task.done():
            return
        self._level2_loop_task = asyncio.create_task(self._schedule_level2_loop())
        logger.info(
            "Level 2 loop started (interval=%.0fs, overdue=%.1fh)",
            self.level2_recheck_interval_sec, self.level2_overdue_hours,
        )

    async def stop_level2_loop(self) -> None:
        """Level 2 常駐ループを停止する (lifespan shutdown 用)"""
        if self._level2_loop_task is None or self._level2_loop_task.done():
            return
        self._level2_loop_task.cancel()
        try:
            await self._level2_loop_task
        except (asyncio.CancelledError, Exception):
            pass
        self._level2_loop_task = None
        logger.info("Level 2 loop stopped")

    async def _schedule_level2_loop(self) -> None:
        """常駐ループ。overdue + 非アクティブで Level 2 をトリガーする。

        bg_task wrapper として最外殻に trace_id を発行し、ループ停止 (cancel)
        時に outcome.jsonl へ結末記録する。
        """
        token = trace_id_var.set(generate_trace_id())
        t_start = time.monotonic()
        cancelled_ok = False
        overdue_sec = self.level2_overdue_hours * 3600.0
        while True:
            try:
                await asyncio.sleep(self.level2_recheck_interval_sec)

                ls = self._learning_scheduler
                if ls is None:
                    continue
                if self.is_user_active():
                    continue
                # 前回実行から overdue_hours 未満なら待機 (未実行は inf=即 overdue)
                if ls.seconds_since_level2_run() < overdue_sec:
                    continue

                triggered = False
                try:
                    triggered = bool(ls.check_level2(
                        is_user_active=self.is_user_active(),
                        lora_path=self._lora_path,
                        # 現在の base モデルファイル名でモデル隔離フィルタを有効化。
                        # FeedbackCollector が刻む base_model (= 同じ GGUF ファイル名)
                        # と一致するため、現モデルの経験のみが Level 2 に渡る。
                        current_model=(
                            self._base_model_path.name
                            if self._base_model_path else ""
                        ),
                        base_model_path=self._base_model_path,
                        assist_lora_path=self._assist_lora_path,
                        assist_model_path=self._assist_model_path,
                    ))
                except Exception as e:
                    logger.error("Level 2 check failed: %s", e, exc_info=True)
                if triggered:
                    logger.info("Level 2 LoRA tuning triggered (overdue loop)")
                    _emit_bg_task_outcome(
                        self._debug_logger,
                        task_name="level2_schedule",
                        success=True,
                        duration_ms=0.0,
                        extra={"triggered": True},
                    )

            except asyncio.CancelledError:
                logger.info("Level 2 loop cancelled")
                cancelled_ok = True
                break
            except Exception as e:
                logger.error("Level 2 loop iteration failed: %s", e, exc_info=True)
                # ループ自体は継続 (次 interval で再評価)

        elapsed_ms = (time.monotonic() - t_start) * 1000
        _emit_bg_task_outcome(
            self._debug_logger,
            task_name="level2_loop",
            success=cancelled_ok,
            duration_ms=elapsed_ms,
            extra={"cancelled": cancelled_ok},
        )
        trace_id_var.reset(token)

    @staticmethod
    def _seconds_until_hour(target_hour: int, local_tz: tzinfo) -> float:
        """指定ローカル時刻 (tz-aware) までの秒数を計算

        内部時刻は UTC で扱い、設定された ``schedule.local_tz`` で
        ローカル時刻に変換したうえで次回発火時刻を算出する。サーバの
        実行 tz に依存しない。
        """
        now_utc = utc_now_dt()
        now_local = now_utc.astimezone(local_tz)
        target_local = now_local.replace(
            hour=target_hour, minute=0, second=0, microsecond=0,
        )
        if target_local <= now_local:
            target_local += timedelta(days=1)
        target_utc = target_local.astimezone(timezone.utc)
        return (target_utc - now_utc).total_seconds()

    @property
    def running(self) -> bool:
        return self._running
