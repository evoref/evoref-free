"""モデル識別子 ``compat_key`` / ``model_key`` / ``model_label`` (c_05 §0.5.7)。

- ``compat_key`` = GGUF の ``general.architecture`` + テンソル名・形状の digest。
  **LoRA / cvector が載るかの判定だけ**に使う (量子化・fine-tune を区別しない)。
- ``model_key`` = ``compat_key`` + データ長 + 重みの標本 digest。標本はテンソル
  データ領域の先頭を基準に blk 0 / 中央 / 最終の attn・ffn テンソルと ``output``
  から 64KiB ずつ取る。学習パーティション・``learn.*`` subject・LoRA の適用先・
  埋め込みディレクトリに使う。``tokenizer.chat_template`` を直してヘッダ長が
  変わっても key は変わらない (オフセットはデータ領域の先頭からの相対)。
- ``model_label`` = ファイル名の stem。表示用で、digest を UI に出さない。

計算結果は ``model_registry`` (derived) に ``(path, size, mtime_ns)`` をキーに
保存する。プロセス内にも同じキーのメモを持つ (2 回目以降は stat 1 回)。
テストは :func:`set_model_key_computer` で計算を差し替える。

置き場は pillar 外 (横断基盤、全 pillar から参照可)。
"""

from __future__ import annotations

import hashlib
import mmap
import os
import re
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import derived_id, is_valid_id
from backend.io.versioned import VersionedJsonFile
from backend.log_config import get_logger

logger = get_logger("model_key")

#: 1 テンソルから取る標本の長さ。
SAMPLE_BYTES = 64 * 1024
#: ``general.alignment`` が無いときの GGUF の既定。
_DEFAULT_ALIGNMENT = 32
_GGUF_MAGIC = b"GGUF"

_BLOCK_RE = re.compile(r"blk\.(\d+)\.")

MODEL_REGISTRY_FORMAT = register_format(FormatSpec(
    format_id="model_registry",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="store/model_registry.json",
    retention="one entry per model file path (replaced when size or mtime changes)",
))


class ModelKeyError(ValueError):
    """GGUF が読めず ``model_key`` を計算できない。"""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """1 モデルファイルの識別子。

    Attributes:
        model_key: ``mk_<hex16>``。
        compat_key: ``<architecture>:<hex16>``。
        architecture: ``general.architecture`` (無ければ空文字)。
        model_label: 表示名 (ファイル名の stem)。
        data_start: テンソルデータ領域の先頭 (ファイル先頭からのバイト数)。
        data_length: テンソルデータ領域の長さ。
    """

    model_key: str
    compat_key: str
    architecture: str
    model_label: str
    data_start: int
    data_length: int


def model_label(model_path: Path | str) -> str:
    """表示用のモデル名 (ファイル名の stem)。"""
    name = Path(str(model_path)).name
    return Path(name).stem if name else ""


def provisional_model_key(filename: str) -> str:
    """GGUF が読めないモデルの仮の key (ファイル名由来、実 key とは衝突しない)。

    モデルファイルが無い (未ダウンロード・テスト) 構成でもパスを決められるように
    するための縮退。実ファイルが置かれると key は実 key へ変わる。
    """
    digest = hashlib.sha256(b"unresolved\0" + Path(filename or "").name.encode("utf-8"))
    return derived_id("mk_", digest.hexdigest())


# ── GGUF の読み取り (bytes + unpack_from の 1 パス) ──

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_FIXED_SIZES = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


def _read_string(buf: Any, off: int) -> tuple[str, int]:
    (n,) = _U64.unpack_from(buf, off)
    off += 8
    return bytes(buf[off:off + n]).decode("utf-8", errors="replace"), off + n


def _skip_value(buf: Any, off: int, vtype: int) -> int:
    size = _FIXED_SIZES.get(vtype)
    if size is not None:
        return off + size
    if vtype == _TYPE_STRING:
        return off + 8 + _U64.unpack_from(buf, off)[0]
    if vtype == _TYPE_ARRAY:
        (etype,) = _U32.unpack_from(buf, off)
        (count,) = _U64.unpack_from(buf, off + 4)
        off += 12
        size = _FIXED_SIZES.get(etype)
        if size is not None:
            return off + size * count
        if etype == _TYPE_STRING:
            unpack = _U64.unpack_from
            for _ in range(count):
                off += 8 + unpack(buf, off)[0]
            return off
        for _ in range(count):
            off = _skip_value(buf, off, etype)
        return off
    raise ModelKeyError(f"unsupported GGUF value type: {vtype}")


def _pick_sample_tensors(names: list[str]) -> list[str]:
    """標本を取るテンソル (blk 0 / 中央 / 最終の attn・ffn と ``output``)。"""
    blocks: dict[int, list[str]] = {}
    for name in names:
        match = _BLOCK_RE.match(name)
        if match:
            blocks.setdefault(int(match.group(1)), []).append(name)
    picked: list[str] = []
    if blocks:
        order = sorted(blocks)
        chosen = sorted({order[0], order[len(order) // 2], order[-1]})
        for block in chosen:
            members = blocks[block]
            for part in ("attn", "ffn"):
                hit = next((n for n in members if part in n[len(f"blk.{block}."):]), None)
                if hit is not None and hit not in picked:
                    picked.append(hit)
    if "output.weight" in names:
        picked.append("output.weight")
    return picked


def compute_model_identity(model_path: Path | str) -> ModelIdentity:
    """GGUF を読んで識別子を計算する (キャッシュなし)。

    Raises:
        ModelKeyError: 読めない / GGUF でない / 壊れている。
    """
    path = Path(model_path)
    try:
        with path.open("rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < 24:
                raise ModelKeyError(f"{path.name}: too small for GGUF")
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                return _identity_from_buffer(buf, size, path)
    except ModelKeyError:
        raise
    except (OSError, ValueError, struct.error, MemoryError) as e:
        raise ModelKeyError(f"{path.name}: {e}") from e


def _identity_from_buffer(buf: Any, size: int, path: Path) -> ModelIdentity:
    if bytes(buf[0:4]) != _GGUF_MAGIC:
        raise ModelKeyError(f"{path.name}: not a GGUF file")
    (version,) = _U32.unpack_from(buf, 4)
    if version < 2:
        raise ModelKeyError(f"{path.name}: GGUF v{version} is not supported")
    n_tensors, n_kv = struct.unpack_from("<QQ", buf, 8)
    off = 24
    architecture = ""
    alignment = _DEFAULT_ALIGNMENT
    for _ in range(n_kv):
        key, off = _read_string(buf, off)
        (vtype,) = _U32.unpack_from(buf, off)
        off += 4
        if key == "general.architecture" and vtype == _TYPE_STRING:
            architecture, off = _read_string(buf, off)
        elif key == "general.alignment" and vtype == 4:
            (alignment,) = _U32.unpack_from(buf, off)
            off += 4
        else:
            off = _skip_value(buf, off, vtype)
    tensors: dict[str, tuple[tuple[int, ...], int]] = {}
    for _ in range(n_tensors):
        name, off = _read_string(buf, off)
        (n_dims,) = _U32.unpack_from(buf, off)
        dims = struct.unpack_from(f"<{n_dims}Q", buf, off + 4)
        off += 4 + 8 * n_dims
        (tensor_offset,) = _U64.unpack_from(buf, off + 4)
        off += 12
        tensors[name] = (tuple(dims), tensor_offset)
    if alignment <= 0:
        raise ModelKeyError(f"{path.name}: bad alignment {alignment}")
    data_start = off + (-off % alignment)
    if data_start > size:
        raise ModelKeyError(f"{path.name}: tensor info runs past the end of the file")
    data_length = size - data_start

    shape = hashlib.sha256(architecture.encode("utf-8") + b"\n")
    for name in sorted(tensors):
        dims = tensors[name][0]
        shape.update(f"{name}:{','.join(map(str, dims))}\n".encode())
    compat_key = f"{architecture or 'unknown'}:{shape.hexdigest()[:16]}"

    weights = hashlib.sha256(f"{compat_key}\n{data_length}\n".encode())
    for name in _pick_sample_tensors(list(tensors)):
        start = data_start + tensors[name][1]
        end = min(start + SAMPLE_BYTES, size)
        weights.update(f"{name}@{tensors[name][1]}\n".encode())
        weights.update(buf[start:end])
    return ModelIdentity(
        model_key=derived_id("mk_", weights.hexdigest()),
        compat_key=compat_key,
        architecture=architecture,
        model_label=model_label(path),
        data_start=data_start,
        data_length=data_length,
    )


# ── model_registry (derived) ──


class ModelRegistry(VersionedJsonFile):
    """``store/model_registry.json`` — パス → ``(size, mtime_ns)`` と識別子。

    ペイロード: ``{"entries": {"<絶対パス>": {"size", "mtime_ns", "model_key",
    "compat_key", "architecture", "data_start", "data_length"}}}``。パスごとに
    1 件で、size か mtime が変わったら置き換える。
    """

    FORMAT = MODEL_REGISTRY_FORMAT
    _state_logger = logger

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.entries: dict[str, dict[str, Any]] = {}
        self.load()

    def _to_payload(self) -> dict[str, Any]:
        return {"entries": self.entries}

    def _from_payload(self, payload: Any) -> None:
        entries = payload.get("entries") if isinstance(payload, dict) else None
        self.entries = dict(entries) if isinstance(entries, dict) else {}

    def lookup(self, path: str, size: int, mtime_ns: int) -> ModelIdentity | None:
        raw = self.entries.get(path)
        if not isinstance(raw, dict):
            return None
        if raw.get("size") != size or raw.get("mtime_ns") != mtime_ns:
            return None
        key = raw.get("model_key")
        if not is_valid_id(key, "mk_"):
            return None
        return ModelIdentity(
            model_key=str(key),
            compat_key=str(raw.get("compat_key") or ""),
            architecture=str(raw.get("architecture") or ""),
            model_label=model_label(path),
            data_start=int(raw.get("data_start") or 0),
            data_length=int(raw.get("data_length") or 0),
        )

    def remember(self, path: str, size: int, mtime_ns: int, identity: ModelIdentity) -> None:
        self.entries[path] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "model_key": identity.model_key,
            "compat_key": identity.compat_key,
            "architecture": identity.architecture,
            "data_start": identity.data_start,
            "data_length": identity.data_length,
        }


# ── 解決 (メモ → registry → 計算) ──

ModelKeyComputer = Callable[[Path], ModelIdentity]

_lock = threading.Lock()
_computer: ModelKeyComputer = compute_model_identity
_memo: dict[tuple[str, int, int], ModelIdentity] = {}
_registry: ModelRegistry | None = None
_warned_unresolved: set[str] = set()


def set_model_key_computer(computer: ModelKeyComputer | None) -> ModelKeyComputer:
    """識別子の計算を差し替える (テスト用)。``None`` で既定へ戻す。前の計算を返す。"""
    global _computer
    with _lock:
        previous = _computer
        _computer = computer or compute_model_identity
        _memo.clear()
        _warned_unresolved.clear()
    return previous


def use_model_registry(path: Path | str | None) -> None:
    """計算結果を保存する ``model_registry`` を指定する (``None`` で保存しない)。

    serve だけが指定する (単一書き手、c_05 §0.5.8)。CLI などの別プロセスは
    プロセス内のメモだけで済ませる。
    """
    global _registry
    with _lock:
        _registry = ModelRegistry(path) if path is not None else None


def clear_model_key_cache() -> None:
    """プロセス内のメモを捨てる (テスト用)。"""
    with _lock:
        _memo.clear()
        _warned_unresolved.clear()


def model_identity(model_path: Path | str) -> ModelIdentity:
    """``model_path`` の識別子 (メモ → registry → 計算)。

    Raises:
        ModelKeyError: ファイルが無い / GGUF として読めない。
    """
    path = Path(model_path)
    try:
        st = path.stat()
    except OSError as e:
        raise ModelKeyError(f"{path.name}: {e}") from e
    resolved = str(path.resolve())
    memo_key = (resolved, st.st_size, st.st_mtime_ns)
    hit = _memo.get(memo_key)
    if hit is not None:
        return hit
    with _lock:
        registry = _registry
        computer = _computer
    identity = registry.lookup(resolved, st.st_size, st.st_mtime_ns) if registry else None
    if identity is None:
        identity = computer(path)
        if registry is not None:
            with _lock:
                registry.remember(resolved, st.st_size, st.st_mtime_ns, identity)
                try:
                    registry.save()
                except Exception as e:  # noqa: BLE001 — derived の保存失敗は計算し直すだけ
                    logger.debug("model_registry not saved: %s", e)
    with _lock:
        _memo[memo_key] = identity
    return identity


def model_key_for(model_path: Path | str) -> str:
    """``model_path`` の ``model_key``。読めなければファイル名由来の仮 key。

    仮 key を返したときはパスごとに 1 回 WARNING を出す。
    """
    try:
        return model_identity(model_path).model_key
    except ModelKeyError as e:
        name = Path(str(model_path)).name
        if name not in _warned_unresolved:
            _warned_unresolved.add(name)
            logger.warning(
                "Cannot compute model_key of %s (%s); using a provisional key "
                "derived from the file name",
                name, e,
            )
        return provisional_model_key(name)


__all__ = [
    "MODEL_REGISTRY_FORMAT",
    "SAMPLE_BYTES",
    "ModelIdentity",
    "ModelKeyComputer",
    "ModelKeyError",
    "ModelRegistry",
    "clear_model_key_cache",
    "compute_model_identity",
    "model_identity",
    "model_key_for",
    "model_label",
    "provisional_model_key",
    "set_model_key_computer",
    "use_model_registry",
]
