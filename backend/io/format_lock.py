"""形式台帳の lock (c_05 §0.7.2 の縮小版: R1 / R9)。

登録済みの形式を JSON に描き出し、``tests/fixtures/formats/lock.json`` (配布物外) と
突き合わせる。**版を上げずに互換を壊す変更**をここで止める。

同じ版のまま許す変更 (c_05 §0.4.4):

- 既定値を持つフィールドの追加 (任意フィールド)
- ``open`` な列挙への値の追加
- ``derived`` / ``volatile`` の形式の変更 (作り直せる / 互換対象外)

それ以外 (フィールドの削除・型 / 既定値の変更・必須化・``closed`` な列挙への値追加・
列挙値の削除・分類 / 書き手 / 符号化 / 置き場 / 列挙方針の変更) は ``sot`` / ``system``
では版を上げる。版を下げるのは常に違反。

形式の版を上げたら、その形式の側のデータ世代 (リリース用の定数) も上げる
(設計 r6 §4.3 / e_02 §6.2)。lock はトップに世代も持ち、版が上がった形式があるのに
世代が lock と同じなら違反にする:

- 共有形式 (``writers`` に free を含む) → ``backend.free.__version__.DATA_GENERATION``
- Pro 専用形式 (``writers`` が pro だけ) → ``backend.pro.__version__.PRO_DATA_GENERATION``

lock はトップに G0 検出の署名 (``g0_signatures``、起動ゲートの ``G0_SIGNATURES``) も
持つ (設計 r6 §15.10)。署名を削るのは違反 (既存の G0 のデータを見落として黙って新規扱い
になる)、足すのは更新待ち。定数は ``backend/factory`` にあるので :mod:`backend.formats` が
渡す (``backend/io`` は起動の層を import しない)。

使い方 (宣言モジュールを揃える必要があるので入口は :mod:`backend.formats`)::

    python -m backend.formats --check
    python -m backend.formats --update   # 違反があれば書かない
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any, Literal

from backend.io.codec import EXTRA_FIELD, codec_for
from backend.io.format_registry import FORMATS, FormatRegistry, FormatSpec

LOCK_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "formats" / "lock.json"

_STRICT_CLASSES = frozenset({"sot", "system"})


# ── 描き出し ──


def _describe_type(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if tp is Any:
        return "any"
    if tp in (str, int, float, bool):
        return tp.__name__
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return {"optional": _describe_type(args[0])}
    if origin is Literal:
        return {"literal": sorted(typing.get_args(tp), key=str)}
    if origin is list:
        (item,) = typing.get_args(tp) or (Any,)
        return {"list": _describe_type(item)}
    if origin is dict:
        _, value = typing.get_args(tp) or (str, Any)
        return {"dict": _describe_type(value)}
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return {"record": describe_record(tp)}
    return repr(tp)


def _describe_default(f: dataclasses.Field[Any]) -> Any:
    if f.default is not dataclasses.MISSING:
        value = f.default
    elif f.default_factory is not dataclasses.MISSING:
        value = f.default_factory()
    else:
        return {"required": True}
    try:
        json.dumps(value)
    except TypeError:
        return {"repr": repr(value)}
    return {"value": value}


def describe_record(cls: type) -> dict[str, Any]:
    """永続 dataclass のフィールド表 (名前・型・既定値、フィールド順)。"""
    codec = codec_for(cls)
    hints = typing.get_type_hints(cls)
    return {
        "name": cls.__qualname__,
        "omit_defaults": codec.omit_defaults,
        "fields": {
            f.name: {"type": _describe_type(hints[f.name]), "default": _describe_default(f)}
            for f in dataclasses.fields(cls)
            if f.name in codec.known
        },
    }


def describe_format(spec: FormatSpec) -> dict[str, Any]:
    return {
        "version": spec.version,
        "class": spec.klass,
        "writers": sorted(spec.writers),
        "path_key": spec.path_key,
        "encodings": list(spec.encodings),
        "enums": dict(sorted(spec.enums.items())),
        "keep_on_reset": spec.keep_on_reset,
        "records": [describe_record(r) for r in spec.records],
    }


#: lock のトップに置く世代のキー → 定数名 (違反の文面に出す)。
GENERATION_CONSTANTS: dict[str, str] = {
    "generation": "DATA_GENERATION",
    "pro_generation": "PRO_DATA_GENERATION",
}


def current_generations() -> dict[str, int]:
    """リリース用のデータ世代。Pro 未同梱なら ``pro_generation`` は無い。"""
    from backend.free.__version__ import DATA_GENERATION

    out = {"generation": DATA_GENERATION}
    try:
        from backend.pro.__version__ import PRO_DATA_GENERATION  # type: ignore[import-not-found]
    except ImportError:
        return out
    out["pro_generation"] = PRO_DATA_GENERATION
    return out


def build_lock(
    registry: FormatRegistry = FORMATS,
    generations: dict[str, int] | None = None,
    g0_signatures: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """lock の描き出し。``g0_signatures`` は G0 検出の署名 (起動ゲートの定数を呼び手が渡す)。"""
    gens = current_generations() if generations is None else generations
    lock: dict[str, Any] = {**gens, "formats": {s.format_id: describe_format(s) for s in registry.all()}}
    if g0_signatures is not None:
        lock["g0_signatures"] = sorted(g0_signatures)
    return lock


def generation_key(writers: list[str]) -> str:
    """形式の版上げで上げる世代のキー (書き手が pro だけなら Pro 側)。"""
    return "pro_generation" if set(writers) == {"pro"} else "generation"


# ── 比較 ──


def _record_breaks(old: dict[str, Any], new: dict[str, Any], enums: dict[str, str], where: str) -> list[str]:
    """同じ版のまま許されないレコードの変更を列挙する。"""
    breaks: list[str] = []
    if old["omit_defaults"] != new["omit_defaults"]:
        breaks.append(f"{where}: omit_defaults changed")
    for name, spec in old["fields"].items():
        if name not in new["fields"]:
            breaks.append(f"{where}.{name}: field removed")
            continue
        after = new["fields"][name]
        if spec["default"] != after["default"]:
            breaks.append(f"{where}.{name}: default changed")
        breaks.extend(_type_breaks(spec["type"], after["type"], enums.get(name), f"{where}.{name}", enums))
    for name, spec in new["fields"].items():
        if name not in old["fields"] and spec["default"] == {"required": True}:
            breaks.append(f"{where}.{name}: required field added")
    return breaks


def _type_breaks(old: Any, new: Any, policy: str | None, where: str, enums: dict[str, str]) -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
        (key,) = old.keys()
        if key == "literal":
            removed = [v for v in old["literal"] if v not in new["literal"]]
            added = [v for v in new["literal"] if v not in old["literal"]]
            out = [f"{where}: enum values removed {removed}"] if removed else []
            if added and policy != "open":
                out.append(f"{where}: values added to a {policy or 'closed'} enum {added}")
            return out
        if key == "record":
            return _record_breaks(old["record"], new["record"], enums, where)
        return _type_breaks(old[key], new[key], policy, where, enums)
    return [] if old == new else [f"{where}: type changed"]


def _generation_changes(saved: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    stale: list[str] = []
    for key, const in GENERATION_CONSTANTS.items():
        before, after = saved.get(key), current.get(key)
        if after is None or before == after:
            continue
        if before is None:
            stale.append(f"{const}: recorded {after}")
        elif after < before:
            violations.append(f"{const}: went down {before} -> {after}")
        else:
            stale.append(f"{const}: {before} -> {after}")
    return violations, stale


def _g0_signature_changes(saved: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """G0 検出の署名 (設計 r6 §15.10)。削るのは違反 (既存の G0 のデータを見落とす)、足すのは更新待ち。"""
    if "g0_signatures" not in current:
        return [], []
    after = set(current["g0_signatures"])
    if "g0_signatures" not in saved:
        return [], [f"g0_signatures: recorded {len(after)}"]
    before = set(saved["g0_signatures"])
    violations = [
        f"G0 signature removed: {sig} (G0 data matching it would no longer be detected)"
        for sig in sorted(before - after)
    ]
    return violations, [f"G0 signature added: {sig}" for sig in sorted(after - before)]


def compare(saved: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(違反, 更新が要るだけの差分) を返す。"""
    violations, stale = _generation_changes(saved, current)
    g0_violations, g0_stale = _g0_signature_changes(saved, current)
    violations.extend(g0_violations)
    stale.extend(g0_stale)
    old_formats = saved.get("formats", {})
    new_formats = current["formats"]
    for fid, new in new_formats.items():
        old = old_formats.get(fid)
        if old is None:
            stale.append(f"{fid}: new format")
            continue
        if old == new:
            continue
        if new["version"] < old["version"]:
            violations.append(f"{fid}: version went down {old['version']} -> {new['version']}")
            continue
        if new["version"] > old["version"]:
            # 書き手が変わった版上げは共有形式側に数える (pro だけ → 共有でも逆でも Free が関わる)
            key = generation_key(old["writers"]) if old["writers"] == new["writers"] else "generation"
            before, after = saved.get(key), current.get(key)
            if before is not None and after is not None and after <= before:
                violations.append(
                    f"{fid}: version {old['version']} -> {new['version']} without bumping "
                    f"{GENERATION_CONSTANTS[key]} (still {after})"
                )
            else:
                stale.append(f"{fid}: changed")
            continue
        if old["class"] not in _STRICT_CLASSES:
            stale.append(f"{fid}: changed")
            continue
        breaks = [
            f"{fid}: {key} changed"
            for key in ("class", "writers", "path_key", "encodings", "enums", "keep_on_reset")
            if old[key] != new[key]
        ]
        old_records = {r["name"]: r for r in old["records"]}
        for record in new["records"]:
            before = old_records.pop(record["name"], None)
            if before is None:
                breaks.append(f"{fid}: record {record['name']} added")
            else:
                breaks.extend(_record_breaks(before, record, new["enums"], f"{fid}:{record['name']}"))
        breaks.extend(f"{fid}: record {name} removed" for name in old_records)
        if breaks:
            violations.extend(f"{b} (bump the format version)" for b in breaks)
        else:
            stale.append(f"{fid}: compatible change")
    stale.extend(f"{fid}: format removed" for fid in old_formats if fid not in new_formats)
    return violations, stale


def load_saved(path: Path = LOCK_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"formats": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_lock(lock: dict[str, Any], path: Path = LOCK_PATH) -> None:
    from backend.io.atomic import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
