"""図形 DSL の解析と正規化 — ``.pptx`` / ``.odp`` 共通

設計書: [docs/f_11_file_export.md](../../docs/f_11_file_export.md) §4.2。

Markdown に図形の記法が無いため、``shapes`` を言語識別子に持つフェンスへ
JSON 配列を書く独自 DSL を定めている。座標・サイズの単位は **cm 固定**
(ODF がネイティブに cm を取り、python-pptx も ``Cm()`` を持つので、同じ数値が
両形式でそのまま通る)。
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any

from backend.log_config import get_logger

logger = get_logger("export.shapes")

#: 描ける図形の種類。
KINDS = frozenset({"rect", "oval", "line"})

_DEFAULT_FILL = "#4472C4"
_DEFAULT_LINE = "#000000"
_DEFAULT_LINE_WIDTH_PT = 2.0


@dataclass(frozen=True)
class Shape:
    """1 個の図形 (単位はすべて cm、線幅のみ pt)。"""

    kind: str
    x: float = 0.0
    y: float = 0.0
    w: float = 4.0
    h: float = 2.0
    x2: float = 0.0
    y2: float = 0.0
    fill: str | None = _DEFAULT_FILL
    line: str = _DEFAULT_LINE
    width_pt: float = _DEFAULT_LINE_WIDTH_PT
    text: str = ""


def parse_shapes_source(source: str) -> list[dict[str, Any]]:
    """``shapes`` フェンスの中身 (JSON 配列) を dict のリストにする。

    壊れた JSON は **図形ゼロ** として扱い、文書全体は落とさない。
    """
    text = (source or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as e:
        logger.warning("shapes block is not valid JSON, ignoring: %r", e)
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        logger.warning("shapes block must be a JSON array, got %s", type(data).__name__)
        return []
    return [item for item in data if isinstance(item, dict)]


def looks_like_shapes_payload(text: str) -> list[dict[str, Any]] | None:
    """フェンスの言語指定が無くても図形 DSL だと判る本文か。

    ローカルモデルは ```shapes の言語指定を落として JSON 配列だけを吐くことが
    ある (2026-09-16 実測: 「青い四角形と赤い直線と円を描いて」で、正しい図形
    JSON が **フェンス無しの段落** として出た)。プロンプトの指示だけに頼ると
    そこで機能へ到達できなくなるので、決定論で拾い直す。

    誤検出を避けるため条件は厳しくする: JSON 配列で、空でなく、**全要素が
    dict で kind が既知** のときだけ図形とみなす。1 つでも外れたら ``None``。
    """
    stripped = (text or "").strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, list) or not data:
        return None
    for item in data:
        if not isinstance(item, dict):
            return None
        if str(item.get("kind", "")).strip().lower() not in KINDS:
            return None
    return data


def _number(spec: dict[str, Any], key: str, default: float) -> float:
    value = spec.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("shape %r has a non-numeric %s=%r; using %s", spec.get("kind"), key, value, default)
        return default


#: キーが無かったことを「明示の null」と区別するための番兵。
_ABSENT = object()


def _color(value: Any, default: str | None) -> str | None:
    """``#RRGGBB`` を検証する。

    キーが無ければ ``default``、**明示の ``null`` なら ``None``** (塗りなし)。
    """
    if value is _ABSENT:
        return default
    if value is None:
        return None
    if not isinstance(value, str):
        logger.warning("shape color %r is not a string; using %s", value, default)
        return default
    text = value.strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            logger.warning("shape color %r is not #RRGGBB; using %s", value, default)
            return default
        return text.upper()
    logger.warning("shape color %r is not #RRGGBB; using %s", value, default)
    return default


def normalize_shape(spec: dict[str, Any]) -> Shape | None:
    """1 個の図形定義を検証して ``Shape`` にする。不正なら ``None``。"""
    kind = str(spec.get("kind", "")).strip().lower()
    if kind not in KINDS:
        logger.warning("unknown shape kind %r (expected one of %s)", spec.get("kind"), sorted(KINDS))
        return None

    if kind == "line":
        fill = None
        line = _color(spec.get("line", _ABSENT), _DEFAULT_LINE) or _DEFAULT_LINE
        return Shape(
            kind=kind,
            x=_number(spec, "x1", 0.0),
            y=_number(spec, "y1", 0.0),
            x2=_number(spec, "x2", 4.0),
            y2=_number(spec, "y2", 0.0),
            fill=fill,
            line=line,
            width_pt=_number(spec, "width", _DEFAULT_LINE_WIDTH_PT),
        )

    fill = _color(spec.get("fill", _ABSENT), _DEFAULT_FILL)
    # 枠線色の既定は塗りと同色 (指定が無いときに黒枠が浮かないように)。
    line = _color(spec.get("line", _ABSENT), fill or _DEFAULT_LINE) or _DEFAULT_LINE
    return Shape(
        kind=kind,
        x=_number(spec, "x", 0.0),
        y=_number(spec, "y", 0.0),
        w=_number(spec, "w", 4.0),
        h=_number(spec, "h", 2.0),
        fill=fill,
        line=line,
        width_pt=_number(spec, "width", _DEFAULT_LINE_WIDTH_PT),
        text=str(spec.get("text", "") or ""),
    )


def normalize_shapes(specs: list[dict[str, Any]]) -> list[Shape]:
    """図形定義の列を検証する。不正なものは **その図形だけ** 落とす。"""
    shapes = [normalize_shape(spec) for spec in specs]
    return [s for s in shapes if s is not None]


def rgb_tuple(color: str) -> tuple[int, int, int]:
    """``#RRGGBB`` を ``(r, g, b)`` にする。"""
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
