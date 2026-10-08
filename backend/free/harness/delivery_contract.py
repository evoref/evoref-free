"""制作ステージの納品契約 (純粋関数)。

「成功」と報告する前に、成果物が依頼の名指しと形式を満たすかを構造で確かめる。
語彙では判定しない — 名指したファイル名が成果物に在るか、拡張子の形式として読めるか、
だけを見る (f_10 §7、2026-10-08 実機監査: reservation_er.md が Python 5 本になり、
library_api.yaml が日本語の散文のまま、いずれも success と報告された)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

_FENCE_RE = re.compile(r"\A\s*```[\w+-]*\n(.*?)\n```\s*\Z", re.DOTALL)


def structured_parse_error(filename: str, content: str) -> str:
    """``.yaml`` / ``.yml`` / ``.json`` の成果物が形式として読めないとき、その理由 (読めれば空)。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        return ""
    fenced = _FENCE_RE.match(content)
    body = fenced.group(1) if fenced else content
    try:
        if suffix == ".json":
            json.loads(body)
        else:
            parsed = yaml.safe_load(body)
            if not isinstance(parsed, (dict, list)):
                return f"{filename}: not a YAML mapping or list"
    except (ValueError, yaml.YAMLError) as exc:
        first_line = (str(exc).splitlines() or [""])[0]
        return f"{filename}: not valid {suffix.lstrip('.').upper()} ({first_line})"
    return ""


def missing_named_deliverables(expected: list[str], produced: list[str]) -> list[str]:
    """依頼が名指した新規ファイル (``expected``) のうち、成果物に無いもの (名前の大文字小文字は区別しない)。"""
    have = {Path(p).name.lower() for p in produced}
    return [n for n in expected if Path(n).name.lower() not in have]
