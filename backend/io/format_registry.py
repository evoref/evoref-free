"""形式台帳 (c_05 §0.7)。

ディスク上の 1 種類のファイル / ストア = 1 形式。形式ごとに ``format_id`` と整数の
版を持ち、分類 (sot / derived / volatile / system) と書き手エディションを宣言する。

``backend/io`` は pillar を import しない — spec は各 pillar 側で宣言して
:func:`register_format` で登録する (Pro は :func:`register_pro_format`)。
停止中の CLI (doctor 等) は登録関数を明示的に呼ぶ。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, get_args

FormatClass = Literal["sot", "derived", "volatile", "system"]
Writer = Literal["free", "pro"]
Encoding = Literal["json", "jsonl", "npy", "npz", "bin", "md", "yaml", "zip", "dir"]
EnumPolicy = Literal["closed", "open"]

_FORMAT_ID_RE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
#: ``path_key`` の置換子 (1 つのパス要素の一部または全部に当たる)。
PLACEHOLDER_RE = re.compile(r"<[a-z][a-z0-9_-]*>")
#: 任意の深さに当たるパス要素 (``path_key`` に高々 1 つ)。末尾の ``/**`` はディレクトリの木。
GLOBSTAR = "**"


def _path_key_problem(path_key: str, encodings: tuple[str, ...]) -> str | None:
    """``path_key`` の文法 (:mod:`backend.io.ledger_files`) に反していれば理由を返す。"""
    if not path_key:
        return "path_key is required"
    segments = path_key.split("/")
    if path_key.startswith("/") or "\\" in path_key or "" in segments:
        return f"path_key must be a relative posix path: {path_key!r}"
    if any(s in (".", "..") for s in segments):
        return f"path_key must not contain . or ..: {path_key!r}"
    if segments.count(GLOBSTAR) > 1:
        return f"path_key may contain ** at most once: {path_key!r}"
    for segment in segments:
        if segment == GLOBSTAR:
            continue
        rest = PLACEHOLDER_RE.sub("", segment)
        if "*" in rest or "<" in rest or ">" in rest:
            return f"bad segment {segment!r} in path_key {path_key!r}"
    tree = segments[-1] == GLOBSTAR
    if tree != (encodings == ("dir",)):
        return f"a directory tree (trailing /**) is declared with encodings ('dir',) and only then: {path_key!r}"
    return None


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """1 形式の宣言。

    Attributes:
        format_id: ``<領域>.<名前>`` (小文字・数字・``_``、``.`` 区切り)。
        version: 現行の版 (1 始まり)。書き出しは常にこの版を刻む。
        klass: 分類。``sot`` は凍結・移行対象、``derived`` は作り直せる、
            ``volatile`` は互換対象外、``system`` は世代印・ロック・予約パス。
        writers: 書き手エディション (Develop は pro)。
        path_key: データ根からの相対パスの型。``<mk>`` 等の置換子は 1 つのパス要素
            (の一部) に当たり、``**`` の要素は任意の深さに当たる (末尾の ``/**`` は
            その下の木全体で、``encodings=("dir",)`` と組にする)。照合の規則は
            :mod:`backend.io.ledger_files`。
        retention: 保持方針の要約 (c_05 §0.5.10)。
        export: export の対象か。
        encodings: ディスク上の符号化。
        enums: 列挙名 → ``closed`` / ``open`` (c_05 §0.5.3)。
        keep_on_reset: ``derived`` のうち作り直しが高価で reset でも残すもの。
        records: この形式が持つレコード型 (:mod:`backend.io.codec` の永続 dataclass)。
            フィールドの名前・型・既定値が lock (c_05 §0.7.2 R1) に凍結される。
        human_edited: 利用者が手で直すファイル。字下げして書く (互換には関わらない)。
    """

    format_id: str
    version: int
    klass: FormatClass
    writers: frozenset[Writer]
    path_key: str
    retention: str = ""
    export: bool = False
    encodings: tuple[Encoding, ...] = ("json",)
    enums: Mapping[str, EnumPolicy] = field(default_factory=dict)
    keep_on_reset: bool = False
    records: tuple[type, ...] = ()
    human_edited: bool = False

    def __post_init__(self) -> None:
        if not _FORMAT_ID_RE.fullmatch(self.format_id):
            raise ValueError(f"invalid format_id: {self.format_id!r}")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError(f"{self.format_id}: version must be an int >= 1, got {self.version!r}")
        if self.klass not in get_args(FormatClass):
            raise ValueError(f"{self.format_id}: unknown class {self.klass!r}")
        writers = frozenset(self.writers)
        if not writers or not writers <= set(get_args(Writer)):
            raise ValueError(f"{self.format_id}: writers must be a non-empty subset of free/pro")
        object.__setattr__(self, "writers", writers)
        encodings = tuple(self.encodings)
        unknown = [e for e in encodings if e not in get_args(Encoding)]
        if not encodings or unknown:
            raise ValueError(f"{self.format_id}: bad encodings {encodings!r}")
        object.__setattr__(self, "encodings", encodings)
        problem = _path_key_problem(self.path_key, encodings)
        if problem:
            raise ValueError(f"{self.format_id}: {problem}")
        bad = {k: v for k, v in self.enums.items() if v not in get_args(EnumPolicy)}
        if bad:
            raise ValueError(f"{self.format_id}: enum policy must be closed/open: {bad!r}")
        object.__setattr__(self, "enums", MappingProxyType(dict(self.enums)))
        object.__setattr__(self, "records", tuple(self.records))
        if self.keep_on_reset and self.klass != "derived":
            raise ValueError(f"{self.format_id}: keep_on_reset applies to derived formats only")


class FormatRegistry:
    """``format_id`` → :class:`FormatSpec`。同じ id に別の宣言を重ねると拒否する。"""

    def __init__(self) -> None:
        self._specs: dict[str, FormatSpec] = {}
        self._lock = threading.Lock()

    def register(self, spec: FormatSpec) -> FormatSpec:
        """登録する。同一内容の再登録は無視 (停止中の CLI が登録関数を呼び直すため)。"""
        with self._lock:
            known = self._specs.get(spec.format_id)
            if known is not None and known != spec:
                raise ValueError(f"format {spec.format_id!r} is already registered with a different spec")
            self._specs[spec.format_id] = spec
            return spec

    def get(self, format_id: str) -> FormatSpec:
        """宣言を引く。未登録なら ``KeyError``。"""
        return self._specs[format_id]

    def find(self, format_id: str) -> FormatSpec | None:
        return self._specs.get(format_id)

    def all(self) -> tuple[FormatSpec, ...]:
        """登録済みの宣言 (``format_id`` 順)。"""
        return tuple(self._specs[k] for k in sorted(self._specs))

    def read_by(self, edition: Writer) -> tuple[FormatSpec, ...]:
        """実行中エディションが readonly の判定に使う形式 (c_05 §0.4.3)。

        Free は ``writers`` に pro しか無い形式を見ない。
        """
        if edition == "pro":
            return self.all()
        return tuple(s for s in self.all() if "free" in s.writers)

    def register_all(self, specs: Iterable[FormatSpec]) -> None:
        for spec in specs:
            self.register(spec)


#: プロセス既定の台帳。
FORMATS = FormatRegistry()


def register_format(spec: FormatSpec) -> FormatSpec:
    """既定の台帳へ登録する。"""
    return FORMATS.register(spec)


#: Pro だけが書くデータの置き場 (``PathResolver.LAYOUT["pro_dir"]``、c_05 §0.4.2)。
PRO_STORE_PREFIX = "store/pro/"


def register_pro_format(spec: FormatSpec) -> FormatSpec:
    """Pro の形式を既定の台帳へ登録する (書き手は pro だけ、置き場は ``store/pro/`` の下)。"""
    if spec.writers != frozenset({"pro"}) or not spec.path_key.startswith(PRO_STORE_PREFIX):
        raise ValueError(
            f"{spec.format_id}: a Pro format must be written by pro only "
            f"under {PRO_STORE_PREFIX} (got writers={sorted(spec.writers)}, path_key={spec.path_key!r})",
        )
    return register_format(spec)


def get_format(format_id: str) -> FormatSpec:
    return FORMATS.get(format_id)


__all__ = [
    "FORMATS",
    "PRO_STORE_PREFIX",
    "EnumPolicy",
    "FormatClass",
    "FormatRegistry",
    "FormatSpec",
    "Writer",
    "get_format",
    "register_format",
    "register_pro_format",
]
