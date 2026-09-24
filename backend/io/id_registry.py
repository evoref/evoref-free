"""ID 台帳 (c_05 §0.5.5)。

ランダム ID は ``<prefix>_<hex16>`` (64bit) の 1 文法。接頭辞 (``_`` を含めて 2〜5 文字)
はこの台帳にだけ登録し、重複はテストで拒否する。接頭辞の無い bare hex は禁止。

- 文法の検査は発番時と API 入口だけ。読み手は長さで弾かない。
- 固定幅の列 (``ids`` = ``S24``) に収まることと ASCII 限定を lock で固定する
  (:data:`ID_COLUMN_WIDTH`)。
- **位置カウンタを鍵にしない**。
- trace_id は対象外 (不変則 #7)。session_id は uuid4 で別文法。
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

#: 乱数部の 16 進桁数 (64bit)。
HEX_DIGITS = 16
#: 派生カラム ``ids`` の幅 (numpy ``S24``)。
ID_COLUMN_WIDTH = 24

_PREFIX_RE = re.compile(r"[a-z]{1,4}_")


@dataclass(frozen=True, slots=True)
class IdPrefix:
    """1 接頭辞の宣言。

    Attributes:
        prefix: ``_`` で終わる接頭辞 (``ev_`` 等)。
        meaning: 何の ID か。
        random: 乱数で発番できる。
        derived: 内容由来の決定論 ID を作れる (例: corpus チャンク / model_key)。
    """

    prefix: str
    meaning: str
    random: bool = True
    derived: bool = False

    @property
    def max_length(self) -> int:
        return len(self.prefix) + HEX_DIGITS


#: 接頭辞の台帳。足すときは重複と列幅を ``test_id_registry`` が検査する。
ID_PREFIXES: tuple[IdPrefix, ...] = (
    IdPrefix("t_", "conversation turn"),
    IdPrefix("task_", "loop task"),
    IdPrefix("ep_", "agent trace episode"),
    IdPrefix("run_", "create run"),
    IdPrefix("fs_", "few-shot example"),
    IdPrefix("exp_", "learning experience"),
    IdPrefix("ev_", "evidence record (random: episodic / semantic, derived: corpus chunk)", derived=True),
    IdPrefix("ks_", "knowledge source"),
    IdPrefix("ki_", "knowledge item"),
    IdPrefix("pc_", "prompt candidate"),
    IdPrefix("file_", "uploaded session file"),
    IdPrefix("pq_", "corpus pseudo query (target chunk + position + text)", random=False, derived=True),
    IdPrefix("mk_", "model key (compat + weight sample digest)", random=False, derived=True),
)

_BY_PREFIX: dict[str, IdPrefix] = {p.prefix: p for p in ID_PREFIXES}
_PATTERNS: dict[str, re.Pattern[str]] = {
    p.prefix: re.compile(re.escape(p.prefix) + rf"[0-9a-f]{{{HEX_DIGITS}}}") for p in ID_PREFIXES
}


def _known(prefix: str) -> IdPrefix:
    try:
        return _BY_PREFIX[prefix]
    except KeyError:
        raise ValueError(f"unregistered id prefix: {prefix!r}") from None


def new_id(prefix: str) -> str:
    """``<prefix><hex16>`` を発番する。内容由来だけの接頭辞は発番できない。"""
    spec = _known(prefix)
    if not spec.random:
        raise ValueError(f"{prefix!r} ids are derived from content, not generated")
    return prefix + secrets.token_hex(HEX_DIGITS // 2)


def derived_id(prefix: str, digest_hex: str) -> str:
    """内容由来の ID (``digest_hex`` の先頭 16 桁)。"""
    spec = _known(prefix)
    if not spec.derived:
        raise ValueError(f"{prefix!r} ids are random; use new_id()")
    head = digest_hex[:HEX_DIGITS].lower()
    if len(head) != HEX_DIGITS or any(c not in "0123456789abcdef" for c in head):
        raise ValueError("digest must be at least 16 hex digits")
    return prefix + head


def id_pattern(prefix: str) -> str:
    """``prefix`` の文法の正規表現 (``^…$`` で固定。API の path 検証などに渡す)。"""
    return "^" + _PATTERNS[_known(prefix).prefix].pattern + "$"


def is_valid_id(value: object, prefix: str) -> bool:
    """``value`` が ``prefix`` の文法に合うか (発番時 / API 入口の検査用)。"""
    return isinstance(value, str) and _PATTERNS[_known(prefix).prefix].fullmatch(value) is not None


def fits_id_column(value: object) -> bool:
    """``value`` が固定幅の id 列 (``S24``) に切り詰めなしで入るか (ASCII 限定)。

    文法 (接頭辞・桁数) は問わない — 読み手は長さで弾かないので、列に入るかだけを見る。
    """
    if not isinstance(value, str) or not value.isascii():
        return False
    return 0 < len(value) <= ID_COLUMN_WIDTH


def id_prefix(value: str) -> IdPrefix | None:
    """``value`` の接頭辞の宣言 (台帳外なら ``None``)。"""
    head, sep, _ = value.partition("_")
    return _BY_PREFIX.get(head + sep) if sep else None


def validate_prefixes(prefixes: tuple[IdPrefix, ...] = ID_PREFIXES) -> None:
    """台帳の不変条件 (重複なし・文法・列幅・ASCII)。違反は ``ValueError``。"""
    seen: set[str] = set()
    for p in prefixes:
        if p.prefix in seen:
            raise ValueError(f"duplicate id prefix: {p.prefix!r}")
        seen.add(p.prefix)
        if not _PREFIX_RE.fullmatch(p.prefix):
            raise ValueError(f"id prefix must be 1-4 lowercase letters + '_': {p.prefix!r}")
        if not (p.random or p.derived):
            raise ValueError(f"{p.prefix!r} can be neither generated nor derived")
        if p.max_length > ID_COLUMN_WIDTH:
            raise ValueError(f"{p.prefix!r} ids exceed the {ID_COLUMN_WIDTH}-byte id column")


validate_prefixes()


__all__ = [
    "HEX_DIGITS",
    "ID_COLUMN_WIDTH",
    "ID_PREFIXES",
    "IdPrefix",
    "derived_id",
    "fits_id_column",
    "id_pattern",
    "id_prefix",
    "is_valid_id",
    "new_id",
    "validate_prefixes",
]
