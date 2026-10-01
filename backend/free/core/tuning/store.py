"""``cache/auto_tune.json`` (形式 ``cache.auto_tune`` v1、c_16 §7.2.3 / c_05 §0.7.1)。

環境調整の結果 — PC の指紋・項目ごとの値 / 反映の可否 / 再起動の要否と、環境移行の確認状態
(``decision``)。書き手は起動スクリプト / ``evoref tune`` / 管理画面の実行で、backend は読むだけ。
derived (作り直せる) なので読めなければ捨てて測り直す。``store/`` の外なので readonly の影響を受けない。

埋め込みの配置 (``cache/embed_placement.json``) とリランカーの自己テスト (``cache/rerank_selftest.json``)
は従来のファイルが正本のまま。``items`` にはその要約を載せるだけ。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.free.core.tuning.items import TuneItem
from backend.free.rag.rerank_selftest import PcInfo
from backend.io.codec import codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger

logger = get_logger("core.tuning.store")

#: 環境移行の確認状態。``auto`` は確認していない (新規・同じ PC)。
Decision = Literal["auto", "accepted", "declined", "pending", "unchanged"]
DECISIONS: tuple[str, ...] = ("auto", "accepted", "declined", "pending", "unchanged")


@persisted()
@dataclass
class AutoTuneRecord:
    """``cache/auto_tune.json`` の payload。

    ``fingerprint`` / ``pc`` / ``measured_at`` は最後に項目を測った PC と時刻 (確認だけを書いた
    レコードでは空)。``decided_*`` は ``decision`` を記録した PC — 確認はその PC にだけ効く
    (``decided_pc`` は GPU 名を取れない backend が軸で照合するため)。
    """

    fingerprint: str = ""
    pc: PcInfo = field(default_factory=PcInfo)
    measured_at: str = ""
    items: dict[str, TuneItem] = field(default_factory=dict)
    decision: Decision = "auto"
    decided_at: str = ""
    decided_fingerprint: str = ""
    decided_pc: PcInfo = field(default_factory=PcInfo)
    #: 稼働中の実行で予約した項目 (base に効く見積り)。base が止まっている次の起動 / 再起動で
    #: 再計算し、保存した項目から外す (``resolve.resolve_tuned`` / ``evoref tune``)。
    recompute: list[str] = field(default_factory=list)
    _extra: dict[str, Any] | None = None


#: 形式 (c_05 §0.7.1)。PC 固有の測定値で作り直せる (derived) が、測り直しは一時サーバを起こすので
#: ``evoref reset`` でも残す (``--include-cache`` まで)。
AUTO_TUNE_FORMAT = register_format(FormatSpec(
    format_id="cache.auto_tune",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="cache/auto_tune.json",
    retention=(
        "one record; items are rewritten per item on each tune run (startup when the PC "
        "fingerprint changes and the move is accepted, or a manual run), the decision on each "
        "migration answer; environmental failures are not written"
    ),
    export=False,
    keep_on_reset=True,
    records=(AutoTuneRecord,),
))


class AutoTuneFile(VersionedJsonFile):
    """``cache/auto_tune.json`` の読み書き (derived: 読めなければ捨てて測り直す)。"""

    FORMAT = AUTO_TUNE_FORMAT
    _state_logger = logger

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.record: AutoTuneRecord | None = None

    def _to_payload(self) -> Any:
        if self.record is None:
            raise ValueError("no auto-tune record to save")
        return codec_for(AutoTuneRecord).encode(self.record)

    def _from_payload(self, payload: Any) -> None:
        self.record = codec_for(AutoTuneRecord).decode(payload)


def load_auto_tune(path: Path | str) -> tuple[AutoTuneRecord | None, str]:
    """保存済みのレコードと、読んだときの分類 (``absent`` / ``current`` / ``corrupt`` 等)。"""
    store = AutoTuneFile(path)
    if store.load():
        return store.record, str(store.last_status or "current")
    return None, str(store.last_status or "absent")


def save_auto_tune(path: Path | str, record: AutoTuneRecord) -> bool:
    """レコードを書く (AtomicWriter)。書けたら ``True``。"""
    store = AutoTuneFile(path)
    store.record = record
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return store.save()


#: 読み直して書く操作 (:func:`update_auto_tune`) の同一プロセス内の排他。runner (スレッド) と
#: 起動スクリプトの見積り・予約・確認の答えが同じレコードを読み書きするので、読んでから書くまでの間に
#: 別の書き込みが入って消えないようにする (プロセスをまたぐ排他はしない)。
RECORD_LOCK = threading.RLock()


def update_auto_tune(path: Path | str, mutate: Callable[[AutoTuneRecord], None]) -> AutoTuneRecord | None:
    """最新のレコードを読み直して ``mutate`` を当て、書く (キー単位のマージ)。書けなければ ``None``。"""
    with RECORD_LOCK:
        record = load_auto_tune(path)[0] or AutoTuneRecord()
        mutate(record)
        return record if save_auto_tune(path, record) else None


def save_item(path: Path | str, item: TuneItem, current: PcInfo, stamp: str) -> AutoTuneRecord | None:
    """1 項目を最新のレコードへ書き足す。書けなければ ``None``。

    別の PC で測った項目が残るレコードは項目を捨ててから書く (指紋だけ今の PC に書き換えると、
    測り直していない項目まで「この PC の結果」として反映される)。予約 (``recompute``) から外す。
    """

    def put(record: AutoTuneRecord) -> None:
        if record.fingerprint and record.fingerprint != current.digest:
            record.items = {}
        record.items[item.key] = item
        record.recompute = [k for k in record.recompute if k != item.key]
        record.fingerprint, record.pc, record.measured_at = current.digest, current, stamp

    return update_auto_tune(path, put)


def schedule_recompute(path: Path | str, keys: list[str] | tuple[str, ...]) -> bool:
    """項目を再計算の予約に足す (他の内容は保つ)。書けたら ``True``。"""

    def add(record: AutoTuneRecord) -> None:
        record.recompute = [*record.recompute, *(k for k in keys if k not in record.recompute)]

    ok = update_auto_tune(path, add) is not None
    if not ok:
        logger.warning("Failed to schedule auto-tune recompute of %s in %s", ", ".join(keys), path)
    return ok


@dataclass(frozen=True)
class TunePaths:
    """環境調整が読み書きする 3 ファイル (``PathResolver.LAYOUT`` のキーから解決する)。"""

    embed_placement: Path
    rerank_selftest: Path
    auto_tune: Path


def resolve_tune_paths(cfg: dict[str, Any], project_root: Path, resolver: Any = None) -> TunePaths:
    """3 ファイルの置き場を ``PathResolver`` で決める。

    ``resolver`` を渡せばそのデータ根 (backend の起動時の resolver)、無ければ ``EVOREF_DATA_ROOT`` →
    ``<project_root>/userdata`` (起動スクリプトと同じ解決)。

    実体は :func:`_resolve_paths` (モジュールの名前で引く)。この関数を名前で import した先
    (gate / runner / resolve / API / 起動スクリプト) も、テストの隔離が ``_resolve_paths`` を
    差し替えれば同じ置き場を見る (backend/conftest.py)。
    """
    return _resolve_paths(cfg, project_root, resolver)


def _resolve_paths(cfg: dict[str, Any], project_root: Path, resolver: Any = None) -> TunePaths:
    """:func:`resolve_tune_paths` の実体。"""
    if resolver is None:
        from backend.config import PathResolver

        resolver = PathResolver(cfg, project_root)
    return TunePaths(
        embed_placement=resolver.resolve_local("embed_placement_file"),
        rerank_selftest=resolver.resolve_local("rerank_selftest_file"),
        auto_tune=resolver.resolve_local("auto_tune_file"),
    )


__all__ = [
    "AUTO_TUNE_FORMAT",
    "DECISIONS",
    "RECORD_LOCK",
    "AutoTuneFile",
    "AutoTuneRecord",
    "Decision",
    "TunePaths",
    "load_auto_tune",
    "resolve_tune_paths",
    "save_auto_tune",
    "save_item",
    "schedule_recompute",
    "update_auto_tune",
]
