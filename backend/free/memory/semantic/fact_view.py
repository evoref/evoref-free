"""``SemanticFact`` の読み取りビューと常駐の表 (G1 設計 §17.4 / c_05 §1.4)。

常駐は ``Evidence`` 1 本。``SemanticStore`` は id → ``Evidence`` だけを持ち、消費者へ
渡す ``SemanticFact`` は :class:`FactView` — 元の ``Evidence`` を指すだけのビュー
(コピーを持たない) — として作る。

- **構築はコピーしない**。属性は読むときに ``Evidence`` から導き
  (:func:`~backend.free.memory.semantic.fact.evidence_to_fact` と同じ対応)、
  導いた値はビューの中に memo する。可変な値 (リスト / 集合 / dict) は写しを
  返すので、ビューを書き換えても常駐の ``Evidence`` は変わらない。
- **生きているビューは id ごとに 1 つ** (弱参照の表)。更新で ``Evidence`` が
  差し替わると、そのビューを新しい ``Evidence`` へ付け替える — 呼出側が握っている
  ファクトにも更新が見える (G0 の「常駐オブジェクトをその場で書き換える」と同じ面)。
  誰も握っていないビューは残らない (常駐メモリに数えない)。
- 代入と、書き手が永続形に載らない属性へ入れた値は **作業値** — 表が id ごとの
  差分として持ち (``Evidence`` から導ける値は持たない)、稼働中は見えて再起動で消える
  (G0 の常駐オブジェクトと同じ)。永続化の経路は ``SemanticStore.update_fact``
  (→ patch API) だけ。
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from dataclasses import MISSING, fields
from typing import Any

from backend.free.memory.semantic.fact import (
    KNOWN_FACT_TYPES,
    FactRecordError,
    _epoch,
    _provenances_from,
)
from backend.free.memory.types import SemanticFact
from backend.free.rag.evidence import Evidence
from backend.utils import utc_to_epoch

_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(SemanticFact))
_KNOWN: frozenset[str] = frozenset(_FIELD_NAMES)
#: 永続形を持たない属性 (値があれば常に作業値。numpy 配列なので == で比べない)。
_WORKING_ONLY: frozenset[str] = frozenset({"embedding"})


def check_fact_record(record: Evidence) -> None:
    """``record`` を ``SemanticFact`` として読めるか (読めなければ :class:`FactRecordError`)。

    :func:`~backend.free.memory.semantic.fact.evidence_to_fact` と同じ条件
    (``attrs.fact_type`` の欠損・未知値) を、作業型を作らずに調べる。
    """
    fact_type = (record.attrs or {}).get("fact_type")
    if not fact_type:
        raise FactRecordError(f"fact evidence without attrs.fact_type: {record.id}")
    if not isinstance(fact_type, str) or fact_type not in KNOWN_FACT_TYPES:
        raise FactRecordError(f"unknown fact_type {fact_type!r}: {record.id}")


# ── 属性の導出 (evidence_to_fact と同じ対応) ──────────────────────────────


def _structured(ev: Evidence, key: str) -> Any:
    structured = ev.structured
    return None if structured is None else structured.get(key)


def _object(ev: Evidence) -> str:
    return str(_structured(ev, "object") or ev.text or "")


def _created_at(ev: Evidence) -> float:
    return _epoch(ev.as_of or ev.observed_at)


def _trace_id(ev: Evidence) -> str | None:
    provenance = ev.provenance[0] if ev.provenance else {}
    return str(provenance.get("trace_id") or "") or None


def _extra(ev: Evidence) -> dict[str, Any]:
    extra = dict(ev._extra or {})
    for key, value in (ev.attrs or {}).items():
        if key != "fact_type" and key not in _KNOWN:
            extra[key] = value
    return extra


#: コア側から導く属性 (attrs に同名のキーが無いとき)。無い名前は SemanticFact の既定値。
_CORE: dict[str, Callable[[Evidence], Any]] = {
    "id": lambda ev: ev.id,
    "subject": lambda ev: str(_structured(ev, "subject") or "unknown"),
    "predicate": lambda ev: str(_structured(ev, "predicate") or ""),
    "object": _object,
    "statement": lambda ev: ev.text if ev.text != _object(ev) else None,
    "type": lambda ev: ev.attrs.get("fact_type"),
    "scope": lambda ev: ev.scope or "global",
    "lang": lambda ev: ev.lang or "",
    "provenances": lambda ev: _provenances_from(ev.provenance),
    "confidence": lambda ev: float(ev.confidence),
    "origin": lambda ev: ev.origin,
    "veracity": lambda ev: ev.veracity,
    "contradicts": lambda ev: list(ev.contradicts or []),
    "pinned": lambda ev: bool(ev.pinned),
    "superseded_by": lambda ev: ev.superseded_by,
    "created_at": _created_at,
    "accessed_at": lambda ev: _epoch(ev.last_used_at, _created_at(ev)),
    "private": lambda ev: bool(ev.private),
    "trace_id": _trace_id,
    "_extra": _extra,
}


def _defaults() -> dict[str, Callable[[], Any]]:
    """``SemanticFact`` の既定値 (コアから導かない属性の値)。"""
    out: dict[str, Callable[[], Any]] = {}
    for f in fields(SemanticFact):
        if f.default_factory is not MISSING:
            out[f.name] = f.default_factory
        elif f.default is not MISSING:
            out[f.name] = lambda value=f.default: value
    return out


_DEFAULTS = _defaults()


def _from_attrs(name: str, value: Any) -> Any:
    """attrs の値を作業型の属性へ (evidence_to_fact の attrs ループと同じ変換)。"""
    if name == "session_ids":
        return set(value or ())
    if name == "pin_locked_until":
        return utc_to_epoch(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _derive(ev: Evidence, name: str) -> Any:
    attrs = ev.attrs
    if name != "id" and name in attrs:  # attrs の同名キーが勝つ (evidence_to_fact と同じ)
        return _from_attrs(name, attrs[name])
    core = _CORE.get(name)
    if core is not None:
        return core(ev)
    return _DEFAULTS[name]()


class FactView(SemanticFact):
    """``Evidence`` から作る ``SemanticFact`` の読み取りビュー (コピーを持たない)。

    ``FactView.of(record)`` で作る (``__init__`` は通さない)。:attr:`evidence` が元の
    ``Evidence`` そのもの。
    """

    __slots__ = ("_ev", "_memo", "_table")

    @classmethod
    def of(cls, record: Evidence, table: FactTable | None = None) -> FactView:
        view = object.__new__(cls)
        view._ev = record
        view._memo = None
        view._table = table
        return view

    @property
    def evidence(self) -> Evidence:
        """このビューが指している常駐の ``Evidence``。"""
        return self._ev

    def _rebind(self, record: Evidence) -> None:
        """更新後の ``Evidence`` へ付け替える (導いた値の memo は捨てる)。"""
        self._ev = record
        self._memo = None

    def materialize(self) -> SemanticFact:
        """今の属性値 (作業値を含む) を持つ、ビューではない ``SemanticFact`` を作る。"""
        fact = object.__new__(SemanticFact)
        for name in _FIELD_NAMES:
            setattr(fact, name, getattr(self, name))
        return fact

    def __reduce__(self) -> tuple[Any, ...]:
        # 複製・pickle は常駐の表から切り離した素の SemanticFact にする
        return (_rebuild_fact, ({name: getattr(self, name) for name in _FIELD_NAMES},))


def _rebuild_fact(values: dict[str, Any]) -> SemanticFact:
    fact = object.__new__(SemanticFact)
    for name, value in values.items():
        setattr(fact, name, value)
    return fact


def _view_property(name: str) -> property:
    def fget(self: FactView) -> Any:
        memo = self._memo
        if memo is not None and name in memo:
            return memo[name]
        table = self._table
        if table is not None:
            working = table._working.get(self._ev.id)
            if working is not None and name in working:
                return working[name]
        value = _derive(self._ev, name)
        if memo is None:
            self._memo = {name: value}
        else:
            memo[name] = value
        return value

    def fset(self: FactView, value: Any) -> None:
        memo = self._memo
        if memo is not None:
            memo.pop(name, None)
        table = self._table
        if table is not None and self._ev.id in table._records:
            table._working.setdefault(self._ev.id, {})[name] = value
        elif memo is None:
            self._memo = {name: value}
        else:
            memo[name] = value

    return property(fget, fset, doc=f"``SemanticFact.{name}`` (Evidence から導く)")


for _name in _FIELD_NAMES:
    setattr(FactView, _name, _view_property(_name))
del _name


def _working_delta(record: Evidence, fact: SemanticFact) -> dict[str, Any]:
    """``fact`` の属性のうち ``record`` から導けない値 (永続化されない作業値)。"""
    delta: dict[str, Any] = {}
    for name in _FIELD_NAMES:
        value = getattr(fact, name)
        if name in _WORKING_ONLY:
            if value is not None:
                delta[name] = value
        elif value != _derive(record, name):
            delta[name] = value
    return delta


class FactTable:
    """id → 常駐の ``Evidence`` と、生きている :class:`FactView` の弱参照の表。

    ``SemanticStore`` の在メモリの写し。値は ``Evidence`` 1 本で、``get`` /
    ``values`` はビューを返す。``Evidence`` から導けない作業値 (``embedding`` と、
    永続形に載らない属性へ書き手が入れた値) だけを id ごとの差分として持つ —
    G0 の常駐オブジェクトと同じく、稼働中は見えて再起動で消える。
    """

    __slots__ = ("_records", "_views", "_working")

    def __init__(self) -> None:
        self._records: dict[str, Evidence] = {}
        self._views: weakref.WeakValueDictionary[str, FactView] = weakref.WeakValueDictionary()
        self._working: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, fact_id: object) -> bool:
        return fact_id in self._records

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def record(self, fact_id: str) -> Evidence | None:
        """常駐の ``Evidence`` (ビューを作らない)。"""
        return self._records.get(fact_id)

    def records(self) -> list[Evidence]:
        return list(self._records.values())

    def _view(self, fact_id: str, record: Evidence) -> FactView:
        view = self._views.get(fact_id)
        if view is None:
            view = self._views[fact_id] = FactView.of(record, self)
        return view

    def get(self, fact_id: str, default: Any = None) -> Any:
        record = self._records.get(fact_id)
        return default if record is None else self._view(fact_id, record)

    def __getitem__(self, fact_id: str) -> FactView:
        return self._view(fact_id, self._records[fact_id])

    def values(self) -> list[FactView]:
        return [self._view(fid, record) for fid, record in self._records.items()]

    def items(self) -> list[tuple[str, FactView]]:
        return [(fid, self._view(fid, record)) for fid, record in self._records.items()]

    def put(self, record: Evidence, working: SemanticFact | None = None) -> FactView:
        """``record`` を常駐させ、生きているビューがあれば付け替えて返す。

        ``working`` は書き手が渡した作業型 (``add_fact`` / ``update_fact``)。永続形に
        載らなかった値を作業値として残す。
        """
        fact_id = record.id
        self._records[fact_id] = record
        if working is not None:
            delta = _working_delta(record, working)
            if delta:
                self._working[fact_id] = delta
            else:
                self._working.pop(fact_id, None)
        view = self._views.get(fact_id)
        if view is None:
            view = self._views[fact_id] = FactView.of(record, self)
        else:
            view._rebind(record)
        return view

    def pop(self, fact_id: str, default: Any = None) -> Any:
        """表から外す。返すビューは外す直前の ``Evidence`` と作業値を持ったまま。"""
        record = self._records.pop(fact_id, None)
        if record is None:
            return default
        working = self._working.pop(fact_id, None)
        view = self._views.pop(fact_id, None) or FactView.of(record)
        view._table = None
        if working:
            view._memo = {**(view._memo or {}), **working}
        return view


__all__ = ["FactTable", "FactView", "check_fact_record"]
