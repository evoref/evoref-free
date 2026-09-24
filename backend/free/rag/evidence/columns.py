"""カラム型サイドカー ``columns.npz`` (c_16 §5.3)

snapshot の各行を numpy 配列に落として、マスク・鮮度・順位付けを **Python
ループ無し** で計算できるようにする。本文 (``records.jsonl``) は top-k だけ
lazy に読み、常駐させるのは id とカラムだけ。

| 配列 | dtype | 用途 |
|---|---|---|
| ``ids`` | ``S24`` (ASCII) | 行 → id |
| ``as_of_epoch`` / ``observed_epoch`` / ``valid_until_epoch`` | float64 (NaN=null) | 鮮度・失効 |
| ``half_life_days`` | float32 (NaN=null) | 減衰 |
| ``confidence`` | float32 | 順位 |
| ``flags`` | uint8 bitfield | private / secret / pinned / retracted / superseded / assistant / unknown |
| ``origin`` / ``kind`` / ``veracity`` | uint8 enum | マスク・origin 優先 |
| ``namespace_id`` / ``tier`` / ``package_idx`` | int16 | 部分集合の選択 |
| ``claim_hash64`` | uint64 | claim_key の先頭 64bit (畳み込み) |

``namespace_id`` / ``package_idx`` の文字列↔id 表は npz に入れず、隣の
``columns_tables.json`` に置く (人が読めること + npz に可変長文字列を混ぜない)。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.rag.evidence.types import (
    KIND_IDS,
    ORIGIN_IDS,
    TIER_IDS,
    UNKNOWN_I16,
    UNKNOWN_U8,
    VERACITY_IDS,
    Evidence,
    claim_hash64,
)
from backend.io import AtomicWriter, atomic_write_text
from backend.io.id_registry import ID_COLUMN_WIDTH, fits_id_column
from backend.log_config import get_logger
from backend.utils import parse_utc

logger = get_logger("rag.evidence.columns")

#: ``flags`` uint8 bitfield のビット名 (c_16 §5.3)。**値を変えない**。
FLAG_PRIVATE = 1 << 0
FLAG_SECRET = 1 << 1
FLAG_PINNED = 1 << 2
FLAG_RETRACTED = 1 << 3
FLAG_SUPERSEDED = 1 << 4
FLAG_ASSISTANT_ORIGIN = 1 << 5
#: 未知の列挙値を持つ行 (:attr:`Evidence.ignored`)。保持するが使わない (c_05 §0.4.5)。
#: kind / origin / veracity は列値の番兵 (255) でも分かるが、store / confidentiality /
#: note の tier・mode は列を持たないのでこのビットで表す。
FLAG_UNKNOWN_ENUM = 1 << 6

#: ファイル名。
COLUMNS_FILE = "columns.npz"
TABLES_FILE = "columns_tables.json"

#: 1 日の秒数 (age_days 換算)。
_SECONDS_PER_DAY = 86400.0

#: ``ids`` の dtype。ID 台帳の最長 (``<prefix>_<hex16>``) が収まる幅 (c_05 §0.5.5 / R10)。
_ID_DTYPE = f"S{ID_COLUMN_WIDTH}"


@dataclass(slots=True)
class EvidenceColumns:
    """snapshot 1 版分のカラム集合。"""

    ids: np.ndarray
    as_of_epoch: np.ndarray
    observed_epoch: np.ndarray
    valid_until_epoch: np.ndarray
    half_life_days: np.ndarray
    confidence: np.ndarray
    flags: np.ndarray
    origin: np.ndarray
    kind: np.ndarray
    veracity: np.ndarray
    namespace_id: np.ndarray
    tier: np.ndarray
    package_idx: np.ndarray
    claim_hash64: np.ndarray
    #: 文字列↔id 表 (``namespace`` / ``package``)。id は配列の値と対応。
    namespaces: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(len(self.ids))

    def id_index(self) -> dict[bytes, int]:
        """id (ASCII バイト列) → 行番号。引くときは :func:`encode_id` で符号化する。"""
        return {v: i for i, v in enumerate(self.ids.tolist())}


def encode_id_column(ids: Sequence[str]) -> np.ndarray:
    """id の列を固定幅の ``S24`` にする。

    numpy は幅を超える値を**黙って切り詰める**ので、超過と非 ASCII はここで例外に
    する (c_05 §0.5.5: 固定幅の列は長さ超過で例外)。
    """
    for value in ids:
        if not fits_id_column(value):
            raise ValueError(
                f"evidence id {value!r} does not fit the {_ID_DTYPE} id column "
                f"(ASCII, at most {ID_COLUMN_WIDTH} bytes)",
            )
    return np.array([value.encode("ascii") for value in ids], dtype=_ID_DTYPE)


def encode_id(record_id: str) -> bytes | None:
    """1 つの id を列のキーへ (ASCII でなければ ``None`` = 列に無い)。"""
    try:
        return record_id.encode("ascii")
    except (UnicodeEncodeError, AttributeError):
        return None


def _epoch(value: str | None) -> float:
    """ISO 8601 文字列 → epoch 秒 (float)。``None`` / 解釈不能は NaN。"""
    dt = parse_utc(value)
    return float("nan") if dt is None else dt.timestamp()


def _flags_of(record: Evidence) -> int:
    flags = 0
    if record.private:
        flags |= FLAG_PRIVATE
    if record.confidentiality == "secret":
        flags |= FLAG_SECRET
    if record.pinned:
        flags |= FLAG_PINNED
    if record.veracity == "retracted":
        flags |= FLAG_RETRACTED
    if record.superseded_by:
        flags |= FLAG_SUPERSEDED
    if record.origin == "assistant":
        flags |= FLAG_ASSISTANT_ORIGIN
    if record.ignored:
        flags |= FLAG_UNKNOWN_ENUM
    return flags


def build_columns(records: Sequence[Evidence] | Iterable[Evidence]) -> EvidenceColumns:
    """レコード列からカラム集合を組む。

    レコード → 配列への詰め替えは 1 回だけ Python で回す (構築時)。以降の
    マスク・順位計算は numpy のベクトル演算だけで行う。
    """
    rows = list(records)
    namespaces: list[str] = []
    packages: list[str] = []
    ns_index: dict[str, int] = {}
    pkg_index: dict[str, int] = {}

    ids: list[str] = []
    as_of: list[float] = []
    observed: list[float] = []
    valid_until: list[float] = []
    half_life: list[float] = []
    confidence: list[float] = []
    flags: list[int] = []
    origin: list[int] = []
    kind: list[int] = []
    veracity: list[int] = []
    namespace_id: list[int] = []
    tier: list[int] = []
    package_idx: list[int] = []
    hashes: list[int] = []

    for record in rows:
        ids.append(record.id)
        as_of.append(_epoch(record.as_of))
        observed.append(_epoch(record.observed_at))
        valid_until.append(_epoch(record.valid_until))
        half_life.append(
            float("nan") if record.half_life_days is None
            else float(record.half_life_days),
        )
        confidence.append(float(record.confidence))
        flags.append(_flags_of(record))
        origin.append(ORIGIN_IDS.get(record.origin, UNKNOWN_U8))
        kind.append(KIND_IDS.get(record.kind, UNKNOWN_U8))
        veracity.append(VERACITY_IDS.get(record.veracity, UNKNOWN_U8))
        tier.append(TIER_IDS.get(record.tier, UNKNOWN_I16))
        hashes.append(claim_hash64(record.claim_key))

        ns = record.namespace
        if ns:
            if ns not in ns_index:
                ns_index[ns] = len(namespaces)
                namespaces.append(ns)
            namespace_id.append(ns_index[ns])
        else:
            namespace_id.append(UNKNOWN_I16)

        package = record.attrs.get("package_id")
        if isinstance(package, str) and package:
            if package not in pkg_index:
                pkg_index[package] = len(packages)
                packages.append(package)
            package_idx.append(pkg_index[package])
        else:
            package_idx.append(UNKNOWN_I16)

    return EvidenceColumns(
        ids=encode_id_column(ids),
        as_of_epoch=np.array(as_of, dtype=np.float64),
        observed_epoch=np.array(observed, dtype=np.float64),
        valid_until_epoch=np.array(valid_until, dtype=np.float64),
        half_life_days=np.array(half_life, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        flags=np.array(flags, dtype=np.uint8),
        origin=np.array(origin, dtype=np.uint8),
        kind=np.array(kind, dtype=np.uint8),
        veracity=np.array(veracity, dtype=np.uint8),
        namespace_id=np.array(namespace_id, dtype=np.int16),
        tier=np.array(tier, dtype=np.int16),
        package_idx=np.array(package_idx, dtype=np.int16),
        claim_hash64=np.array(hashes, dtype=np.uint64),
        namespaces=namespaces,
        packages=packages,
    )


#: npz に収める配列名 (順序は保存/読込で共有)。
_ARRAY_FIELDS: tuple[str, ...] = (
    "ids", "as_of_epoch", "observed_epoch", "valid_until_epoch",
    "half_life_days", "confidence", "flags", "origin", "kind", "veracity",
    "namespace_id", "tier", "package_idx", "claim_hash64",
)


def save_columns(snapshot_dir: Path | str, columns: EvidenceColumns) -> Path:
    """``columns.npz`` + ``columns_tables.json`` を原子的に書き出す。"""
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / COLUMNS_FILE
    arrays = {name: getattr(columns, name) for name in _ARRAY_FIELDS}
    with AtomicWriter(npz_path, mode="wb") as f:
        np.savez(f, **arrays)
    atomic_write_text(
        directory / TABLES_FILE,
        json.dumps(
            {"namespaces": columns.namespaces, "packages": columns.packages},
            ensure_ascii=False, indent=2,
        ),
    )
    return npz_path


def load_columns(snapshot_dir: Path | str) -> EvidenceColumns:
    """``columns.npz`` を読み込む (無ければ空のカラム集合)。"""
    directory = Path(snapshot_dir)
    npz_path = directory / COLUMNS_FILE
    if not npz_path.exists():
        return build_columns([])
    with np.load(str(npz_path), allow_pickle=False) as data:
        arrays = {name: data[name] for name in _ARRAY_FIELDS}
    if arrays["ids"].dtype.kind == "U":  # G1-4 以前の版 (``<U16``) は読み替える
        arrays["ids"] = np.char.encode(arrays["ids"], "ascii")
    tables_path = directory / TABLES_FILE
    namespaces: list[str] = []
    packages: list[str] = []
    if tables_path.exists():
        try:
            raw: Any = json.loads(tables_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("failed to read %s: %s (ids only)", tables_path, e)
            raw = {}
        if isinstance(raw, dict):
            namespaces = [str(v) for v in raw.get("namespaces", [])]
            packages = [str(v) for v in raw.get("packages", [])]
    return EvidenceColumns(
        namespaces=namespaces, packages=packages, **arrays,
    )


def active_mask(
    columns: EvidenceColumns,
    now_epoch: float,
    include_private: bool = False,
) -> np.ndarray:
    """注入候補になりうる行の bool マスク (c_16 §5.3 / §7.3-1)。

    ``~retracted & ~superseded & ~secret & (valid_until 未到来)``。
    ``include_private=False`` では ``private`` も落とす (private セッション外)。
    未知の列挙値の行 (列値 255 / :data:`FLAG_UNKNOWN_ENUM`) も落とす — 除外の関門は
    ここと :func:`~backend.free.rag.evidence.store.is_active` だけ (c_05 §0.5.3)。
    """
    flags = columns.flags
    blocked = FLAG_RETRACTED | FLAG_SUPERSEDED | FLAG_SECRET | FLAG_UNKNOWN_ENUM
    if not include_private:
        blocked |= FLAG_PRIVATE
    mask = (flags & np.uint8(blocked)) == 0
    unknown = np.uint8(UNKNOWN_U8)
    mask &= (columns.kind != unknown) & (columns.origin != unknown) & (columns.veracity != unknown)
    valid_until = columns.valid_until_epoch
    not_expired = np.isnan(valid_until) | (valid_until > now_epoch)
    return mask & not_expired


def freshness(columns: EvidenceColumns, now_epoch: float) -> np.ndarray:
    """減衰係数 float32 (c_16 §7.2)。

    ``half_life_days`` が null の行は 1.0。それ以外は
    ``0.5 ** (age_days / half_life_days)``、age は ``as_of`` (無ければ
    ``observed_at``) からの経過。時刻が両方 null の行も 1.0 (減衰の基準が
    無いものを勝手に古くしない)。
    """
    reference = np.where(
        np.isnan(columns.as_of_epoch), columns.observed_epoch, columns.as_of_epoch,
    )
    age_days = np.maximum((now_epoch - reference) / _SECONDS_PER_DAY, 0.0)
    half_life = columns.half_life_days.astype(np.float64)
    decayable = ~np.isnan(half_life) & (half_life > 0.0) & ~np.isnan(age_days)
    safe_half_life = np.where(decayable, half_life, 1.0)
    safe_age = np.where(decayable, age_days, 0.0)
    decayed = np.power(0.5, safe_age / safe_half_life)
    return np.where(decayable, decayed, 1.0).astype(np.float32)


__all__ = [
    "COLUMNS_FILE",
    "FLAG_ASSISTANT_ORIGIN",
    "FLAG_PINNED",
    "FLAG_PRIVATE",
    "FLAG_RETRACTED",
    "FLAG_SECRET",
    "FLAG_SUPERSEDED",
    "FLAG_UNKNOWN_ENUM",
    "TABLES_FILE",
    "EvidenceColumns",
    "active_mask",
    "build_columns",
    "encode_id",
    "encode_id_column",
    "freshness",
    "load_columns",
    "save_columns",
]
