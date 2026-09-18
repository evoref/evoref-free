"""ProjectMap の id 導出 (c_16 §4.4 / §3.5)

パッケージ id はプロジェクトルートの絶対パスから、``code_node`` の
Evidence id は **同一性** (path + node_type + qualname) から決定論導出する。
どちらも中身ではなく位置カウンタでもない鍵なので、再走査しても・別 PC で
開いても同じ値になる。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: ``package.json`` の ``_extra.kind`` (c_16 §4.4)。
PROJECT_MAP_KIND = "project_map"

#: パッケージ id の接頭辞。``PACKAGE_ID_RE`` (corpus/package.py) を満たす。
_PACKAGE_ID_PREFIX = "pm-"

#: ``code_node`` の Evidence id の接頭辞 (c_16 §3.5)。
_CODE_NODE_ID_PREFIX = "ev_"


def project_map_package_id(root: Path) -> str:
    """プロジェクトルートから corpus パッケージ id を作る (c_16 §4.4)。

    ``"pm-" + sha256(正規化した絶対 posix パス)`` の先頭 12 hex。同じルートを
    指す限り、走査するたびに・別 PC で開いても同じ id になる。
    """
    normalized = Path(root).expanduser().resolve().as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{_PACKAGE_ID_PREFIX}{digest}"


def code_node_id(path: str, node_type: str, qualname: str) -> str:
    """``code_node`` の Evidence id (c_16 §3.5)。

    ``path`` + ``node_type`` + ``qualname`` という **同一性** から導出する
    (本文の内容からではない)。版を跨いで同じシンボルが同じ id を持つため、
    failure_pattern / artifact の鍵として使える。本文が変わっても id は
    変わらない — 版そのものが不変なので、内容の変化は版番号が担う。
    """
    payload = f"code_node\x00{path}\x00{node_type}\x00{qualname}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{_CODE_NODE_ID_PREFIX}{digest}"


__all__ = ["PROJECT_MAP_KIND", "code_node_id", "project_map_package_id"]
