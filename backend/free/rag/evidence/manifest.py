"""`manifest.json` — ストアの現在状態 (c_16 §5.1)

``fsync=True`` で書く。壊れると active 版が分からず起動不能になるため。
未対応の新しい ``schema_version`` は **読まず、書き戻しもしない**
(c_05 §0.5.1 / :mod:`backend.free.rag.evidence._json_state`)。

payload:

```jsonc
{ "store": "episodic", "record_version": 1, "active_snapshot": "v0007",
  "embedding_model_id": "bge-m3", "embedding_dim": 1024,
  "chunker_version": 1, "lexical_version": 1,
  "retention": { ... §5.4 ... },
  "events_since_snapshot": 128,
  "next_snapshot_seq": 8,
  "folded_through": {"month": "2026-09", "line": 412} }
```

``folded_through`` は c_16 §5.1 の例には無いが、「snapshot が事象ログのどこ
まで畳んだか」を持たないと読み手が同じ事象を二度適用する / 取りこぼす。
版番号 ``next_snapshot_seq`` も同じ理由で持つ (ディレクトリの ``max + 1`` は
GC で版が消えると後戻りし、既存の版名を再利用してしまう)。

payload の **既知でないキーは捨てずに ``_extra`` へ退避** し、書き戻しでその
まま復元する (c_05 §0.5.2)。物理 GC の記録 (``_extra["gc"]``) のように、
version をまたいで足される小さな注記もここに載る。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.free.rag.evidence._json_state import JsonStateFile
from backend.free.rag.evidence.events import EventPosition
from backend.free.rag.evidence.snapshot import version_name
from backend.free.rag.evidence.types import RECORD_VERSION
from backend.log_config import get_logger

logger = get_logger("rag.evidence.manifest")

MANIFEST_FILE = "manifest.json"

#: 保持方針の既定 (c_16 §5.4)。**無宣言は監査で落とす**ので、必ず manifest に
#: 書き出す。
DEFAULT_RETENTION: dict[str, Any] = {
    "short_days": 14,
    "long_max_records": 50000,
    "idx_max_records": 20000,
    "events_keep_months": 3,
    "snapshots_keep": 3,
}


#: payload のトップレベル既知キー。これ以外は ``_extra`` へ退避する。
_KNOWN_PAYLOAD_KEYS = frozenset({
    "store",
    "record_version",
    "active_snapshot",
    "embedding_model_id",
    "embedding_dim",
    "chunker_version",
    "lexical_version",
    "retention",
    "events_since_snapshot",
    "next_snapshot_seq",
    "folded_through",
})


class EvidenceManifest(JsonStateFile):
    """1 ストア分の manifest。"""

    SCHEMA_VERSION = 1

    def __init__(self, store_dir: Path | str, store_name: str = "episodic") -> None:
        super().__init__(Path(store_dir) / MANIFEST_FILE, fsync=True)
        self.store = store_name
        self.record_version: int = RECORD_VERSION
        self.active_snapshot: str = ""
        self.embedding_model_id: str = ""
        self.embedding_dim: int = 0
        self.chunker_version: int = 1
        self.lexical_version: int = 1
        self.retention: dict[str, Any] = dict(DEFAULT_RETENTION)
        self.events_since_snapshot: int = 0
        self.next_snapshot_seq: int = 1
        self.folded_through: EventPosition = EventPosition()
        #: payload の ``_extra`` (未知キーの退避先 + 小さな注記)。
        self.extra: dict[str, Any] = {}

    # ── 便宜 API ──

    def retention_value(self, key: str) -> Any:
        """保持方針を 1 件引く (未宣言なら既定)。"""
        value = self.retention.get(key)
        return DEFAULT_RETENTION.get(key) if value is None else value

    def note_extra(self, key: str, value: Any) -> None:
        """``_extra`` に注記を 1 件書く (``save`` は呼び出し側で)。"""
        self.extra[str(key)] = value

    def take_next_version(self) -> str:
        """次の版名を発番してカウンタを進める (``save`` は呼び出し側で)。

        カウンタは単調増加。GC で古い版を消しても後戻りしないので、削除済みの
        版名を別内容で再利用することがない。
        """
        name = version_name(self.next_snapshot_seq)
        self.next_snapshot_seq += 1
        return name

    # ── 永続化 ──

    def _to_payload(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "record_version": int(self.record_version),
            "active_snapshot": self.active_snapshot,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dim": int(self.embedding_dim),
            "chunker_version": int(self.chunker_version),
            "lexical_version": int(self.lexical_version),
            "retention": dict(self.retention),
            "events_since_snapshot": int(self.events_since_snapshot),
            "next_snapshot_seq": int(self.next_snapshot_seq),
            "folded_through": self.folded_through.to_payload(),
            "_extra": dict(self.extra),
        }

    def _from_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("manifest payload must be an object")
        self.store = str(payload.get("store") or self.store)
        self.record_version = int(payload.get("record_version") or RECORD_VERSION)
        self.active_snapshot = str(payload.get("active_snapshot") or "")
        self.embedding_model_id = str(payload.get("embedding_model_id") or "")
        self.embedding_dim = int(payload.get("embedding_dim") or 0)
        self.chunker_version = int(payload.get("chunker_version") or 1)
        self.lexical_version = int(payload.get("lexical_version") or 1)
        retention = payload.get("retention")
        self.retention = {
            **DEFAULT_RETENTION,
            **(retention if isinstance(retention, dict) else {}),
        }
        self.events_since_snapshot = int(payload.get("events_since_snapshot") or 0)
        self.next_snapshot_seq = int(payload.get("next_snapshot_seq") or 1)
        self.folded_through = EventPosition.from_payload(payload.get("folded_through"))
        extra = payload.get("_extra")
        self.extra = dict(extra) if isinstance(extra, dict) else {}
        # 既知でないトップレベルキーも落とさずに退避する (c_05 §0.5.2)。
        for key, value in payload.items():
            if key not in _KNOWN_PAYLOAD_KEYS and key != "_extra":
                self.extra[key] = value


__all__ = ["DEFAULT_RETENTION", "MANIFEST_FILE", "EvidenceManifest"]
