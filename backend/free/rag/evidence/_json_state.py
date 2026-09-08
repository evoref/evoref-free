"""永続化エンベロープ + 版数ガード (c_05 §0.5.1) の最小実装。

``backend/free/learning/json_state_store.py`` の :class:`JsonStateStore` と
**同じ規約** を実装するが、あちらは EvorefLearn pillar に属するモジュールで、
EvorefGen (``backend/free/rag/``) からの top-level import は pillar 境界違反に
なる (``backend/free/tests/test_pillar_boundary.py``)。View 層のような公開
インターフェースも無いため、必要な部分 —

- ``{"schema_version", "written_at", "producer", "payload"}`` のエンベロープ
- **未対応の新しい版は読まず、書き戻しも拒否する** (旧版で書き戻すと新版の
  フィールドを落として壊す。2026-09-05 監査)
- ``AtomicWriter`` 経由の書き込み (manifest は ``fsync=True``)

だけをここへ写した。将来この規約を共有基盤 (``backend/io/``) へ引き上げる
なら、両者を差し替えること。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.io import AtomicWriter
from backend.log_config import get_logger

logger = get_logger("rag.evidence.json_state")

#: 「読むことを拒否した」を表す番兵 (``None`` は正当なペイロードなので使えない)。
REFUSED = object()


def _app_version() -> str:
    """アプリ版数 (取得できなければ空文字)。producer 記録用。"""
    try:
        from backend.version import get_runtime_version

        return str(get_runtime_version())
    except Exception:
        return ""


class JsonStateFile:
    """1 つの JSON 状態ファイル (エンベロープ + 版数ガード付き)。

    サブクラスは :meth:`_to_payload` / :meth:`_from_payload` を実装する。

    Args:
        path: 書き込み先。
        fsync: ``True`` で ``os.fsync`` する。壊れると起動不能・回復不能な
            もの (manifest / 索引) は ``True``。
    """

    #: このストアのレコード版。意味を変える変更で上げ、migrator を用意する。
    SCHEMA_VERSION: int = 1

    def __init__(self, path: Path | str, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self._fsync = fsync
        #: ディスク上のファイルが未対応の新しい版で、書き戻すと壊す状態。
        self.readonly: bool = False

    # ── public API ──

    def save(self) -> None:
        """状態をエンベロープで包んで原子的に書き出す。"""
        if self.readonly:
            logger.warning(
                "Skipping save of %s to %s: on-disk file is a newer schema "
                "version and would be downgraded",
                type(self).__name__, self.path,
            )
            return
        text = json.dumps(self._wrap(self._to_payload()), ensure_ascii=False, indent=2)
        with AtomicWriter(self.path, fsync=self._fsync) as f:
            f.write(text)

    def load(self) -> bool:
        """状態を復元する。読めた場合のみ ``True``。

        ファイル不在 / 破損 / 未対応の新しい版では状態を変えずに ``False``。
        """
        if not self.path.exists():
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to read %s from %s: %s", type(self).__name__, self.path, e,
            )
            return False
        payload = self._unwrap(raw)
        if payload is REFUSED:
            return False
        try:
            self._from_payload(payload)
        except Exception as e:
            logger.warning(
                "Failed to deserialize %s from %s: %s",
                type(self).__name__, self.path, e,
            )
            return False
        return True

    # ── エンベロープ ──

    def _wrap(self, payload: Any) -> dict[str, Any]:
        from backend.utils import utc_now

        return {
            "schema_version": self.SCHEMA_VERSION,
            "written_at": utc_now(),
            "producer": {
                "component": type(self).__name__,
                "app_version": _app_version(),
            },
            "payload": payload,
        }

    def _unwrap(self, raw: Any) -> Any:
        """エンベロープを剥がす。新しい版なら :data:`REFUSED` を返す。"""
        if not (isinstance(raw, dict) and "payload" in raw and "schema_version" in raw):
            return raw  # エンベロープ以前の素のペイロード
        version = raw.get("schema_version")
        if isinstance(version, int) and version > self.SCHEMA_VERSION:
            self.readonly = True
            self._logger().error(
                "Refusing to load %s from %s: schema_version %s is newer than "
                "supported %s. The file is left untouched.",
                type(self).__name__, self.path, version, self.SCHEMA_VERSION,
            )
            return REFUSED
        return raw["payload"]

    def _logger(self) -> logging.Logger:
        return logger

    # ── 抽象メソッド ──

    def _to_payload(self) -> Any:
        raise NotImplementedError

    def _from_payload(self, payload: Any) -> None:
        raise NotImplementedError


__all__ = ["REFUSED", "JsonStateFile"]
