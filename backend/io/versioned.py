"""版付き JSON 状態ファイルの基底 (c_05 §0.4 / §0.5.1 / §0.5.8)。

G0 の ``JsonStateStore`` (learning) と ``JsonStateFile`` (rag.evidence) を統合する 1 本。

封筒::

    {"format_id": "learning.policy_evolver", "format_version": 1,
     "written_at": "...Z",
     "producer": {"component": "...", "app_version": "...", "edition": "free"},
     "payload": {...}}

- キー順は dict の組み立て順で固定し ``payload`` を最後に置く (途中で切れても版が
  読める)。書き出しは常に現行の版を刻む。封筒に世代は持たない。
- 読み取りの分類 (:data:`ReadStatus`):

  - ``absent`` — ファイルが無い。
  - ``current`` / ``migrated`` — 読めた (古い版は ``MIGRATIONS`` を vN→vN+1 で連鎖)。
  - ``newer`` — 版がアプリより新しい、または解けない圧縮形式。**読まず、保存も拒否**。
  - ``foreign`` — 別の形式 / G1 の封筒でない (G0 を含む)。G0 とはデータ互換を持たない
    (c_05 §0.3) ので、読まず・書き換えずに readonly。
  - ``unmigratable`` — 古い版だが移行器が欠けている。読めないが壊してもいけないので readonly。
  - ``corrupt`` — 版が現行以下で読めない。``<name>.corrupt-<utcstamp>`` へ改名して
    既定の状態で続ける (degraded)。**改名に失敗したら readonly** — 黙って既定値で
    上書きしない (G0 はここで全消失していた)。

  readonly と退避は SoT (sot / system) だけ。derived / volatile は読めなければ捨てて
  既定から作り直す (c_05 §0.1 / §0.4.5)。
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from backend.io import format_health, jsoncodec
from backend.io.atomic import atomic_write_bytes
from backend.io.format_registry import FormatSpec
from backend.log_config import get_logger

logger = get_logger("io.versioned")

ReadStatus = Literal[
    "absent", "current", "migrated", "newer", "foreign", "unmigratable", "corrupt",
]

#: 読めない / 書けない (書き戻すと壊す) 分類。
READONLY_STATUSES: frozenset[str] = frozenset({"newer", "foreign", "unmigratable"})

#: 自分が解けない圧縮形式の先頭バイト (c_05 §0.5.13: corrupt ではなく newer)。
_FOREIGN_COMPRESSION_MAGIC: tuple[bytes, ...] = (
    b"\x28\xb5\x2f\xfd",  # zstd
    b"\xfd7zXZ\x00",  # xz
    b"BZh",  # bz2
)
_GZIP_MAGIC = b"\x1f\x8b"

Migration = Callable[[Any], Any]
#: ペイロードの型 (JSON のオブジェクトか配列)。
JsonPayload = dict[str, Any] | list[Any]


@dataclass(frozen=True, slots=True)
class ReadResult:
    """:func:`read_versioned` の結果。``payload`` は ``current`` / ``migrated`` のときだけ有効。"""

    status: ReadStatus
    payload: Any = None
    version: int | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("current", "migrated")


def producer_edition() -> Literal["free", "pro"]:
    """書き手のエディション (Develop は pro、c_05 §0.1)。"""
    try:
        from backend.edition import Edition, current_edition

        return "pro" if current_edition() >= Edition.PRO else "free"
    except Exception:
        return "free"


def _app_version() -> str:
    try:
        from backend.version import get_runtime_version

        return str(get_runtime_version())
    except Exception:
        return ""


def build_envelope(
    *, format_id: str, format_version: int, payload: Any, component: str,
) -> dict[str, Any]:
    """封筒を組み立てる (``payload`` が最後)。"""
    from backend.utils import utc_now

    return {
        "format_id": format_id,
        "format_version": format_version,
        "written_at": utc_now(),
        "producer": {
            "component": component,
            "app_version": _app_version(),
            "edition": producer_edition(),
        },
        "payload": payload,
    }


def write_versioned(
    path: Path | str,
    *,
    format_id: str,
    format_version: int,
    payload: Any,
    component: str,
    fsync: bool = True,
    indent: int | None = None,
) -> None:
    """``payload`` を封筒で包んで原子的に書く。失敗は送出する。"""
    envelope = build_envelope(
        format_id=format_id, format_version=format_version,
        payload=payload, component=component,
    )
    atomic_write_bytes(Path(path), jsoncodec.dumps_bytes(envelope, indent=indent), fsync=fsync)


def read_versioned(
    path: Path | str,
    *,
    format_id: str,
    format_version: int,
    migrations: Mapping[int, Migration] | None = None,
) -> ReadResult:
    """封筒を読んで分類する。ファイルには触らない (改名は :func:`quarantine`)。

    current 以外の分類は ``data_health.formats`` へ載せる (:mod:`backend.io.format_health`)。
    """
    result = _classify(Path(path), format_id, format_version, migrations)
    detail = result.detail or (f"format_version {result.version}" if result.version is not None else "")
    format_health.observe_read(format_id, path, result.status, detail)
    return result


def _classify(
    path: Path,
    format_id: str,
    format_version: int,
    migrations: Mapping[int, Migration] | None,
) -> ReadResult:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ReadResult("absent")
    except OSError as e:
        # 読めない理由が権限・ロックなら中身は無事かもしれない — 改名も上書きもしない。
        return ReadResult("foreign", detail=f"unreadable: {e}")
    return read_versioned_bytes(
        data, format_id=format_id, format_version=format_version, migrations=migrations,
    )


def read_versioned_bytes(
    data: bytes,
    *,
    format_id: str,
    format_version: int,
    migrations: Mapping[int, Migration] | None = None,
) -> ReadResult:
    """封筒のバイト列 (zip の中のファイル等) を :func:`read_versioned` と同じ規則で分類する。

    ファイルではないので ``absent`` にはならず、``data_health.formats`` にも載せない。
    """
    if data.startswith(_FOREIGN_COMPRESSION_MAGIC):
        return ReadResult("newer", detail="unsupported compression")
    if data.startswith(_GZIP_MAGIC):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError) as e:
            return ReadResult("corrupt", detail=f"gzip: {e}")
    try:
        raw = jsoncodec.loads(data)
    except ValueError as e:
        return ReadResult("corrupt", detail=f"json: {e}")
    if not (isinstance(raw, dict) and "format_id" in raw and "format_version" in raw):
        return ReadResult("foreign", detail="not a G1 envelope")
    if raw["format_id"] != format_id:
        return ReadResult("foreign", detail=f"format_id {raw['format_id']!r}")
    version = raw["format_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return ReadResult("corrupt", detail=f"format_version {version!r}")
    if version > format_version:
        return ReadResult("newer", version=version)
    if "payload" not in raw:
        return ReadResult("corrupt", version=version, detail="payload missing")
    payload = raw["payload"]
    if version == format_version:
        return ReadResult("current", payload, version)
    steps = migrations or {}
    for v in range(version, format_version):
        step = steps.get(v)
        if step is None:
            return ReadResult("unmigratable", version=version, detail=f"no migration from v{v}")
        try:
            payload = step(payload)
        except Exception as e:
            return ReadResult("corrupt", version=version, detail=f"migration from v{v}: {e}")
    return ReadResult("migrated", payload, version)


def read_payload(path: Path | str) -> Any:
    """G1 の封筒から ``payload`` だけを返す (テスト / 点検スクリプト用、分類はしない)。"""
    raw = jsoncodec.loads(Path(path).read_bytes())
    if not (isinstance(raw, dict) and "format_id" in raw and "payload" in raw):
        raise ValueError(f"{path} is not a G1 envelope")
    return raw["payload"]


def write_payload(path: Path | str, payload: Any, spec: FormatSpec, *, component: str = "") -> None:
    """``spec`` の現行版で ``payload`` を書く (テスト / 点検スクリプト用)。"""
    write_versioned(
        path, format_id=spec.format_id, format_version=spec.version,
        payload=payload, component=component, fsync=False,
    )


def quarantine(path: Path | str, kind: str = "corrupt") -> Path | None:
    """``path`` を ``<name>.<kind>-<utcstamp>`` へ改名する。失敗なら ``None``。

    既存の退避と名前が衝突したら ``-1`` / ``-2`` … を足す (上書きしない)。
    """
    from backend.io.readonly import DataReadonlyError, guard_write
    from backend.utils import utc_compact_stamp

    path = Path(path)
    try:
        guard_write(path)
    except DataReadonlyError:
        return None  # readonly 中は退避もしない (呼出側は readonly で続ける)
    base = f"{path.name}.{kind}-{utc_compact_stamp()}"
    target = path.with_name(base)
    n = 0
    while target.exists():
        n += 1
        target = path.with_name(f"{base}-{n}")
    try:
        path.rename(target)
    except OSError as e:
        logger.error("Failed to quarantine %s as %s: %s", path, target.name, e)
        return None
    return target


class VersionedJsonFile:
    """版付き JSON 状態ファイルの基底。

    サブクラスは形式台帳の宣言 ``FORMAT`` (:class:`FormatSpec`) を持ち、
    :meth:`_to_payload` / :meth:`_from_payload` を実装する。版を上げたら ``FORMAT``
    の版を上げ、``MIGRATIONS[旧版]`` に旧版 → 次の版の純関数を足す (刈り込まない、
    c_05 §0.4.1)。fsync は分類で決まる (sot / system だけ、c_05 §0.5.8)。

    ``path`` はコンストラクタで固定するか、:meth:`save` / :meth:`load` の引数で渡す。
    読み込みの失敗はログに出して ``False`` を返す。保存の失敗は既定でログに出して
    ``False`` を返し (呼出側のホットパスを巻き添えにしない)、``RAISE_ON_SAVE_ERROR``
    なら送出する (失うと起動できない manifest 等)。
    """

    FORMAT: ClassVar[FormatSpec | None] = None
    MIGRATIONS: ClassVar[Mapping[int, Migration]] = {}
    RAISE_ON_SAVE_ERROR: ClassVar[bool] = False
    #: ログの出し先 (サブクラスが自分のロガーに差し替える)。
    _state_logger: ClassVar[logging.Logger | None] = None

    #: 既定のパス (``__init__`` を呼ばない子クラスでは ``None`` のまま)。
    path: Path | None = None
    #: 読み取りで書き戻すと壊す状態になった (以後の save を拒否する)。
    readonly: bool = False
    #: 直近の読み取りの分類。
    last_status: ReadStatus | None = None

    def __init__(self, path: Path | str | None = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None

    # ── public API ──

    def save(self, path: Path | str | None = None) -> bool:
        """状態を封筒で包んで原子的に書く。書けた場合だけ ``True``。"""
        target = self._target(path)
        spec = self._spec()
        if self.readonly:
            self._logger().warning(
                "Skipping save of %s to %s: the on-disk file must not be overwritten (%s)",
                type(self).__name__, target, self.last_status,
            )
            return False
        try:
            write_versioned(
                target,
                format_id=spec.format_id,
                format_version=spec.version,
                payload=self._to_payload(),
                component=self._component(),
                fsync=spec.klass in ("sot", "system"),
                indent=2 if spec.human_edited else None,
            )
        except (OSError, TypeError, ValueError) as e:
            if self.RAISE_ON_SAVE_ERROR:
                raise
            self._logger().warning("Failed to save %s to %s: %s", type(self).__name__, target, e)
            return False
        self._on_save_success(target)
        return True

    def load(self, path: Path | str | None = None) -> bool:
        """状態を復元する。読めた場合だけ ``True`` (それ以外は状態を変えない)。"""
        target = self._target(path)
        spec = self._spec()
        result = read_versioned(
            target,
            format_id=spec.format_id,
            format_version=spec.version,
            migrations=self.MIGRATIONS,
        )
        self.last_status = result.status
        if result.status == "absent":
            self._on_load_missing(target)
            return False
        if result.ok:
            try:
                self._from_payload(result.payload)
            except Exception as e:
                result = ReadResult("corrupt", version=result.version, detail=f"payload: {e}")
                self.last_status = "corrupt"
            else:
                self._on_load_success(target)
                return True
        detail = result.detail or f"format_version {result.version} > {spec.version}"
        if spec.klass in ("derived", "volatile"):
            # 作り直せる / 互換対象外: 読めなければ捨てて既定から (c_05 §0.1 / §0.4.5)。
            self._logger().warning(
                "Discarding %s at %s (%s: %s); it will be rebuilt",
                type(self).__name__, target, result.status, detail,
            )
            return False
        if result.status in READONLY_STATUSES:
            self.readonly = True
            self._logger().error(
                "Refusing to load %s from %s (%s: %s). The file is left untouched and "
                "will not be overwritten.",
                type(self).__name__, target, result.status, detail,
            )
            return False
        self._quarantine(target, detail)
        return False

    # ── 内部 ──

    def _quarantine(self, target: Path, detail: str) -> None:
        moved = quarantine(target)
        if moved is None:
            self.readonly = True
            self._logger().error(
                "%s at %s is unreadable (%s) and could not be moved aside; "
                "running read-only so it is not overwritten",
                type(self).__name__, target, detail,
            )
            return
        self._logger().warning(
            "%s at %s is unreadable (%s); moved to %s and starting from defaults",
            type(self).__name__, target, detail, moved.name,
        )

    def _target(self, path: Path | str | None) -> Path:
        if path is not None:
            return Path(path)
        if self.path is None:
            raise ValueError(f"{type(self).__name__}: no path given")
        return self.path

    def _spec(self) -> FormatSpec:
        if self.FORMAT is None:
            raise TypeError(f"{type(self).__name__} must declare FORMAT")
        return self.FORMAT

    def _component(self) -> str:
        return type(self).__name__

    def _logger(self) -> logging.Logger:
        return self._state_logger or logger

    # ── サブクラスが実装する ──

    def _to_payload(self) -> Any:
        raise NotImplementedError

    def _from_payload(self, payload: Any) -> None:
        raise NotImplementedError

    # ── 任意フック ──

    def _on_save_success(self, path: Path) -> None:
        """保存成功時。既定は無音。"""

    def _on_load_success(self, path: Path) -> None:
        """読み込み成功時。既定は無音。"""

    def _on_load_missing(self, path: Path) -> None:
        """ファイルが無かったとき。既定は無音。"""


class VersionedPayloadFile(VersionedJsonFile):
    """ペイロードを素のまま持つ版付きファイル (関数で読み書きしていたストア向け)。

    ``f = VersionedPayloadFile(SPEC, path); f.load(); f.payload = ...; f.save()``。
    分類・退避・readonly の規則は :class:`VersionedJsonFile` と同じ。

    ``decode`` / ``encode`` を渡すと ``payload`` は型付きの値になる (読むときに
    ``decode(ペイロード)``、書くときに ``encode(payload)``)。``decode`` の例外は
    ``corrupt`` (SoT は退避) として扱う。
    """

    def __init__(
        self,
        spec: FormatSpec,
        path: Path | str | None = None,
        *,
        migrations: Mapping[int, Migration] | None = None,
        component: str = "",
        state_logger: logging.Logger | None = None,
        decode: Callable[[Any], Any] | None = None,
        encode: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(path)
        self._format = spec
        self._migrations = dict(migrations or {})
        self._component_name = component
        self._instance_logger = state_logger
        self._decode = decode
        self._encode = encode
        #: 読めたペイロード (読めなければ ``None``)。save はこれを書く。
        self.payload: Any = None

    def _spec(self) -> FormatSpec:
        return self._format

    def _component(self) -> str:
        return self._component_name or self._format.format_id

    def _logger(self) -> logging.Logger:
        return self._instance_logger or logger

    def load(self, path: Path | str | None = None) -> bool:
        self.MIGRATIONS = self._migrations  # type: ignore[misc]
        return super().load(path)

    def _to_payload(self) -> Any:
        return self.payload if self._encode is None else self._encode(self.payload)

    def _from_payload(self, payload: Any) -> None:
        self.payload = payload if self._decode is None else self._decode(payload)


__all__ = [
    "READONLY_STATUSES",
    "JsonPayload",
    "ReadResult",
    "ReadStatus",
    "VersionedJsonFile",
    "VersionedPayloadFile",
    "build_envelope",
    "producer_edition",
    "quarantine",
    "read_payload",
    "read_versioned",
    "read_versioned_bytes",
    "write_payload",
    "write_versioned",
]
