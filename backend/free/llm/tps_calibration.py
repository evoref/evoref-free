"""生成速度 (prefill / decode の tok/s) の実測 — ``cache/tps_calibration.json`` (c_16 §7.2.3)。

llama-server は応答の ``timings`` に ``prompt_per_second`` (prefill) と ``predicted_per_second``
(decode) を載せる (非ストリームの応答本体と、ストリームの最終チャンク)。``LocalClient`` がそれを
:meth:`TpsTracker.observe` へ渡し、``model_key`` ごとに EWMA で畳む。ストリーム締切・補助タスク /
エージェントの上限は、:data:`MIN_SAMPLES` に届くまでは **現行の定数のまま** で、届いた後は実測から
導いた値で **延ばす方向にだけ** 動く (速いマシンでハング検知を過敏にしない)。

保存は derived (PC 固有の測定値、読めなければ捨てて測り直す) で、**毎ターンは書かない**:
メモリ上で畳み、最後の保存から :data:`SAVE_INTERVAL_SEC` 経ったときと終了時 (:func:`flush_tps_tracker`)
にだけ書き手スレッド (``backend.io.writer_thread``) へ積む — 観測はチャット応答パスで起きるので、
イベントループの上で ``AtomicWriter`` を呼ばない (c_05 §0.5.9 と同じ理由)。

**PC の指紋** は他の cache と同じ :class:`~backend.free.rag.rerank_selftest.PcInfo` で、backend は GPU 名を
取れないので :func:`~backend.free.rag.rerank_selftest.pc_mismatch` (ホスト名・CPU・コア数・RAM GB) で比べ、
違えば保存値を捨てる (別の PC の速度で締切を組まない)。
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backend.free.rag.rerank_selftest import PcInfo, pc_mismatch
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger

logger = get_logger("llm.tps_calibration")

#: EWMA の重み (新しいサンプルの寄与)。0.2 なら直近 10 回ほどで入れ替わる。
EWMA_ALPHA = 0.2
#: これだけの (外れ値を除いた) サンプルが溜まるまで「未測定」(現行の定数のまま)。
MIN_SAMPLES = 5
#: プロセスごと・モデルごとに捨てる最初のサンプル数 (ロード直後の冷えた KV / 計算バッファの確保)。
WARMUP_SKIP = 1
#: prefill のサンプルに要る最小の評価トークン数。短いと固定費 (スロット確保・テンプレート展開) が
#: 支配して tok/s が実力より大きく低く出る (接頭辞キャッシュが効いた数トークンの要求が典型)。
MIN_PREFILL_TOKENS = 64
#: decode のサンプルに要る最小の生成トークン数 (数トークンの応答は最初のトークンの遅れが支配する)。
MIN_DECODE_TOKENS = 16
#: 受け付ける tok/s の範囲 (外は計測の事故とみなす)。
MIN_SANE_TPS = 0.05
MAX_SANE_TPS = 100_000.0
#: 保存の最小間隔 (秒)。観測のたびには書かない。
SAVE_INTERVAL_SEC = 60.0

#: 実測から締切を導くときの余裕係数 (所要秒の見積りに掛ける倍率)。
#:
#: 併走 (チャットと背景の生成が重なる) の減速は **EWMA が既に含んでいる** — 観測は重なった回も
#: 重なっていない回も同じ台帳へ畳むので、平均の tok/s は併走の分だけ遅く出る。ここで更に併走の
#: 減速 (2026-09-03 監査: 2 倍以上) を掛けると余裕を二重に取り、prefill 40 tok/s の前提に対して
#: 80 tok/s 未満の PC がすべて延びていた (チャット応答パスの補助判定ではそのまま TTFT に乗る)。
#: 1.3 は単発の揺れ (補助タスクの反応的較正 ``aux_client._CALIB_P95_HEADROOM`` と同じ余裕率) だけを
#: 見る値。延びるのは実測が現行の前提の 1.3 倍を下回る PC (ストリームなら prefill 52 / decode 1.3
#: tok/s 未満) だけで、70〜80 tok/s の PC は現行の定数のまま。
TIMEOUT_HEADROOM = 1.3
#: 実測から導く上限に足す固定の下駄 (秒)。HTTP 往復・スロット確保・テンプレート展開の揺れ。
TIMEOUT_BASE_SEC = 10.0
#: 実測で延ばすときの天井 (現行値の倍率)。測定の事故で上限が際限なく伸びないようにする
#: (補助タスクの反応的較正の天井 ``_CALIB_MAX_SCALE`` と同じ 3 倍)。
MEASURED_MAX_SCALE = 3.0
#: **ユーザーを同期で待たせる** 補助判定 (チャット応答パスの purpose・ツール分類器・計画生成・
#: capability プローブ) を実測で延ばすときの天井 (現行値の倍率)。延ばした分はそのまま応答前の
#: 待ち (TTFT) に乗るので、ストリームの総締切 (:data:`MEASURED_MAX_SCALE`) より低く抑える。
CHAT_PATH_MAX_SCALE = 1.5


# ── 形式 ─────────────────────────────────────────────────


@persisted()
@dataclass
class TpsEntry:
    """1 モデルの実測 (EWMA と外れ値を除いたサンプル数)。"""

    prefill_tps: float = 0.0
    prefill_samples: int = 0
    decode_tps: float = 0.0
    decode_samples: int = 0
    updated_at: str = ""
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class TpsCalibrationRecord:
    """``cache/tps_calibration.json`` の payload。``pc`` は測った PC (違えば読まない)。"""

    pc: PcInfo = field(default_factory=PcInfo)
    #: ``model_key`` → 実測。
    models: dict[str, TpsEntry] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


#: 形式 (c_05 §0.7.1)。PC 固有の測定値で作り直せる (derived) が、溜まるまで締切が現行の定数に
#: 戻るので ``evoref reset`` でも残す (``--include-cache`` まで)。
TPS_CALIBRATION_FORMAT = register_format(FormatSpec(
    format_id="cache.tps_calibration",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="cache/tps_calibration.json",
    retention=(
        "one record; one entry per model_key, EWMA folded in memory and written at most once per "
        "SAVE_INTERVAL_SEC and on shutdown; a record measured on another PC is ignored and replaced"
    ),
    export=False,
    keep_on_reset=True,
    records=(TpsCalibrationRecord,),
))


class TpsCalibrationFile(VersionedJsonFile):
    """``cache/tps_calibration.json`` の読み書き (derived: 読めなければ捨てて測り直す)。"""

    FORMAT = TPS_CALIBRATION_FORMAT
    _state_logger = logger

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.record: TpsCalibrationRecord | None = None

    def _to_payload(self) -> Any:
        if self.record is None:
            raise ValueError("no tps calibration record to save")
        return codec_for(TpsCalibrationRecord).encode(self.record)

    def _from_payload(self, payload: Any) -> None:
        self.record = codec_for(TpsCalibrationRecord).decode(payload)


def load_tps_calibration(path: Path | str) -> TpsCalibrationRecord | None:
    """保存済みのレコード (無い / 読めなければ ``None``)。"""
    store = TpsCalibrationFile(path)
    return store.record if store.load() else None


def save_tps_calibration(path: Path | str, record: TpsCalibrationRecord) -> bool:
    """レコードを書く (AtomicWriter)。書けたら ``True``。"""
    store = TpsCalibrationFile(path)
    store.record = record
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return store.save()


# ── 判定 (純関数) ─────────────────────────────────────────


@dataclass(frozen=True)
class TpsEstimate:
    """締切の計算に使う実測。:data:`MIN_SAMPLES` に届いていない側は ``None`` (未測定)。"""

    prefill_tps: float | None = None
    decode_tps: float | None = None
    prefill_samples: int = 0
    decode_samples: int = 0


def ewma(previous: float, sample: float, count: int, alpha: float = EWMA_ALPHA) -> float:
    """``count`` 件畳んだ ``previous`` に ``sample`` を足した EWMA (最初の 1 件はそのまま)。"""
    if count <= 0:
        return sample
    return previous + alpha * (sample - previous)


def _sane(value: Any) -> float | None:
    try:
        tps = float(value)
    except (TypeError, ValueError):
        return None
    return tps if MIN_SANE_TPS <= tps <= MAX_SANE_TPS else None


def samples_from_timings(timings: Any) -> tuple[float | None, float | None]:
    """``timings`` から (prefill tok/s, decode tok/s)。外れ値は ``None``。

    prefill は ``prompt_n`` (接頭辞キャッシュで飛ばした分を除く実評価数) が
    :data:`MIN_PREFILL_TOKENS` 以上、decode は ``predicted_n`` が :data:`MIN_DECODE_TOKENS` 以上のときだけ。
    """
    if not isinstance(timings, dict):
        return None, None
    prefill = decode = None
    try:
        if int(timings.get("prompt_n") or 0) >= MIN_PREFILL_TOKENS:
            prefill = _sane(timings.get("prompt_per_second"))
        if int(timings.get("predicted_n") or 0) >= MIN_DECODE_TOKENS:
            decode = _sane(timings.get("predicted_per_second"))
    except (TypeError, ValueError):
        return None, None
    return prefill, decode


def estimate_of(entry: TpsEntry | None, min_samples: int = MIN_SAMPLES) -> TpsEstimate | None:
    """保存形の実測を締切用に。どちらの側も届いていなければ ``None``。"""
    if entry is None:
        return None
    prefill = entry.prefill_tps if entry.prefill_samples >= min_samples and entry.prefill_tps > 0 else None
    decode = entry.decode_tps if entry.decode_samples >= min_samples and entry.decode_tps > 0 else None
    if prefill is None and decode is None:
        return None
    return TpsEstimate(prefill, decode, entry.prefill_samples, entry.decode_samples)


def extend_only(current: float, measured: float | None, ceiling: float) -> float:
    """``current`` を ``measured`` まで延ばす (縮めない)。延ばすのは ``ceiling`` まで。

    ``ceiling`` が ``current`` 未満なら ``current`` (天井で現行値を割らない)。
    """
    if measured is None:
        return current
    return max(current, min(measured, ceiling))


def derived_seconds(
    tps: TpsEstimate | None, *, prompt_tokens: int, max_tokens: int,
    headroom: float = TIMEOUT_HEADROOM, base: float = TIMEOUT_BASE_SEC,
) -> float | None:
    """「出力トークン ÷ decode tps + prompt ÷ prefill tps」に余裕と下駄を足した秒 (c_16 §7.2.3)。

    両側とも測定済みのときだけ (片側だけだと見積りが欠けて短く出る)。
    """
    if tps is None or tps.prefill_tps is None or tps.decode_tps is None:
        return None
    work = max(0, int(max_tokens)) / tps.decode_tps + max(0, int(prompt_tokens)) / tps.prefill_tps
    return base + headroom * work


def chat_path_timeout(
    current: float, tps: TpsEstimate | None, *, prompt_tokens: int, max_tokens: int,
    ceiling: float | None = None,
) -> float:
    """チャット応答パスの上限を実測で延ばす (未測定なら ``current``、天井は既定で現行の 1.5 倍)。

    ユーザーを同期で待たせる判定なので、天井は :data:`CHAT_PATH_MAX_SCALE`。
    """
    cap = ceiling if ceiling is not None else current * CHAT_PATH_MAX_SCALE
    return extend_only(
        current, derived_seconds(tps, prompt_tokens=prompt_tokens, max_tokens=max_tokens), cap,
    )


#: 実測で延ばす対象の ``agent.*`` のタイムアウト (既定のままのときだけ延ばす)。
LINKED_AGENT_TIMEOUT_KEYS: tuple[str, ...] = ("tool_classifier_timeout_sec", "llm_call_timeout")


def is_explicit_agent_timeout(agent_cfg: dict, key: str) -> bool:
    """``agent.<key>`` が schema の既定から変えられているか (変えていれば実測で延ばさない)。

    検証後の config は既定で埋まっているので「書いたか」は区別できない。既定と違う値を
    利用者の明示とみなす (c_16 §7.2.3 の ``manual``)。キーが無ければ既定扱い。
    """
    if not isinstance(agent_cfg, dict) or key not in agent_cfg:
        return False
    from backend.schemas._common import AgentConfig

    field_info = AgentConfig.model_fields.get(key)
    if field_info is None:
        return True
    try:
        return float(agent_cfg[key]) != float(field_info.default)
    except (TypeError, ValueError):
        return True


def measured_tps_of(client: Any) -> TpsEstimate | None:
    """クライアントの ``measured_tps()`` (無い / 差し替え実装なら ``None``)。"""
    fn = getattr(client, "measured_tps", None) if client is not None else None
    # 非同期の差し替え (AsyncMock 等) は呼ばない (コルーチンを作るだけで値を返さない)
    if not callable(fn) or inspect.iscoroutinefunction(fn):
        return None
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - 締切の見積りで呼出を止めない
        return None
    return value if isinstance(value, TpsEstimate) else None


# ── 実測の台帳 ────────────────────────────────────────────


class TpsTracker:
    """``model_key`` ごとの実測をメモリで畳み、間引いて保存する。スレッド安全。"""

    def __init__(
        self,
        path: Path | str | None,
        current_pc: PcInfo,
        *,
        clock: Callable[[], float] = time.monotonic,
        save_interval: float = SAVE_INTERVAL_SEC,
        submit: Callable[[Callable[[], Any]], Any] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.current_pc = current_pc
        self._clock = clock
        self._save_interval = save_interval
        #: 保存の実行先 (既定は書き手スレッド)。テストは同期の関数を渡す。
        self._submit = submit
        self._entries: dict[str, TpsEntry] = {}
        #: (model_key, "prefill" | "decode") → このプロセスで捨てた最初のサンプル数。
        self._warm: dict[tuple[str, str], int] = {}
        self._dirty = False
        self._last_save = clock()
        self._lock = threading.Lock()

    def load(self) -> bool:
        """保存値を読む。別の PC で測った値・読めない値は捨てる。読めたら ``True``。"""
        if self.path is None:
            return False
        record = load_tps_calibration(self.path)
        if record is None:
            return False
        if pc_mismatch(record.pc, self.current_pc):
            logger.info("Ignoring tps calibration measured on another PC (%s)", self.path)
            return False
        with self._lock:
            self._entries = dict(record.models)
        logger.info("Loaded tps calibration for %d model(s)", len(record.models))
        return True

    def observe(self, model_key: str, timings: Any) -> bool:
        """1 応答の ``timings`` を畳む。何か畳んだら ``True`` (間隔が来ていれば保存も積む)。"""
        if not model_key:
            return False
        prefill, decode = samples_from_timings(timings)
        if prefill is None and decode is None:
            return False
        from backend.utils import utc_now

        changed = False
        with self._lock:
            entry = self._entries.get(model_key) or TpsEntry()
            for kind, sample in (("prefill", prefill), ("decode", decode)):
                if sample is None:
                    continue
                warm_key = (model_key, kind)
                if self._warm.get(warm_key, 0) < WARMUP_SKIP:
                    self._warm[warm_key] = self._warm.get(warm_key, 0) + 1
                    continue
                if kind == "prefill":
                    entry.prefill_tps = ewma(entry.prefill_tps, sample, entry.prefill_samples)
                    entry.prefill_samples += 1
                else:
                    entry.decode_tps = ewma(entry.decode_tps, sample, entry.decode_samples)
                    entry.decode_samples += 1
                changed = True
            if changed:
                entry.updated_at = utc_now()
                self._entries[model_key] = entry
                self._dirty = True
        if changed:
            self.maybe_save()
        return changed

    def entry(self, model_key: str) -> TpsEntry | None:
        with self._lock:
            found = self._entries.get(model_key)
            return replace(found) if found is not None else None

    def estimate(self, model_key: str) -> TpsEstimate | None:
        """締切に使う実測 (未測定なら ``None``)。"""
        if not model_key:
            return None
        with self._lock:
            return estimate_of(self._entries.get(model_key))

    def maybe_save(self) -> bool:
        """前回の保存から間隔が経っていて変更があれば保存を積む。積んだら ``True``。"""
        with self._lock:
            if not self._dirty or self._clock() - self._last_save < self._save_interval:
                return False
        return self.flush()

    def flush(self) -> bool:
        """変更があれば今すぐ保存を積む (終了時)。積んだら ``True``。"""
        if self.path is None:
            return False
        with self._lock:
            if not self._dirty:
                return False
            record = TpsCalibrationRecord(
                pc=self.current_pc,
                models={k: replace(v) for k, v in self._entries.items()},
            )
            self._dirty = False
            self._last_save = self._clock()
        path = self.path

        def _write() -> None:
            if not save_tps_calibration(path, record):
                logger.warning("Failed to save tps calibration to %s", path)

        submit = self._submit or _submit_to_writer
        try:
            submit(_write)
        except Exception as e:  # noqa: BLE001 - derived の保存失敗で応答を止めない
            logger.warning("Failed to queue tps calibration save: %s", e)
        return True


def _submit_to_writer(fn: Callable[[], Any]) -> Any:
    """書き手スレッドで ``fn`` を走らせる (未起動ならその場で実行される)。"""
    from backend.io.writer_thread import default_writer

    return default_writer().call(fn, format_id=TPS_CALIBRATION_FORMAT.format_id)


# ── プロセスの既定 ────────────────────────────────────────

_tracker: TpsTracker | None = None
_tracker_lock = threading.Lock()


def configure_tps_tracker(
    path: Path | str | None, current_pc: PcInfo | None = None, **kwargs: Any,
) -> TpsTracker:
    """プロセスの既定の台帳を作って保存値を読む (起動時に 1 回)。

    未設定の間は観測も見積りもしない (単体テストで実測が別のテストの締切へ漏れない)。
    同じ置き場で呼び直したら既存の台帳を返す (メモリ上の実測を捨てない)。
    """
    global _tracker
    with _tracker_lock:
        if _tracker is not None and path is not None and _tracker.path == Path(path):
            return _tracker
        if current_pc is None:
            from backend.free.rag.rerank_selftest import collect_pc_info

            current_pc = collect_pc_info([])
        tracker = TpsTracker(path, current_pc, **kwargs)
        tracker.load()
        _tracker = tracker
        return tracker


def get_tps_tracker() -> TpsTracker | None:
    """プロセスの既定の台帳 (未設定なら ``None``)。"""
    return _tracker


def reset_tps_tracker() -> None:
    """既定の台帳を外す (テスト用)。"""
    global _tracker
    with _tracker_lock:
        _tracker = None


def flush_tps_tracker() -> bool:
    """既定の台帳の未保存分を書く (終了時)。"""
    tracker = _tracker
    return tracker.flush() if tracker is not None else False


__all__ = [
    "CHAT_PATH_MAX_SCALE",
    "EWMA_ALPHA",
    "LINKED_AGENT_TIMEOUT_KEYS",
    "MEASURED_MAX_SCALE",
    "MIN_DECODE_TOKENS",
    "MIN_PREFILL_TOKENS",
    "MIN_SAMPLES",
    "SAVE_INTERVAL_SEC",
    "TIMEOUT_BASE_SEC",
    "TIMEOUT_HEADROOM",
    "TPS_CALIBRATION_FORMAT",
    "TpsCalibrationFile",
    "TpsCalibrationRecord",
    "TpsEntry",
    "TpsEstimate",
    "TpsTracker",
    "WARMUP_SKIP",
    "chat_path_timeout",
    "configure_tps_tracker",
    "derived_seconds",
    "estimate_of",
    "ewma",
    "extend_only",
    "flush_tps_tracker",
    "get_tps_tracker",
    "is_explicit_agent_timeout",
    "load_tps_calibration",
    "measured_tps_of",
    "reset_tps_tracker",
    "samples_from_timings",
    "save_tps_calibration",
]
