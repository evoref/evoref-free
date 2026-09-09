"""事象ログ (c_16 §5.2) — `put` / `patch` / `retract` / `touch` の追記のみ

稼働中の snapshot / 索引は書き換えず、**版を積んで指す**のが c_16 の不変則。
書き手 (sleep-time) はここへ 1 行追記するだけで、読み手は「snapshot + それ
以降の事象」を畳んで現在状態を得る。

- 月次ファイル ``events/<yyyy-mm>.jsonl``
- 1 行 = ``{"_version", "op", "id", "at", "by", "payload"}``
- 追記は :class:`backend.io.JSONLAppendStore` (同一プロセス内の単一 writer 前提。
  **別プロセスからの書込は禁止** — advisory lock を取らない)
- ``touch`` は複数 id を 1 事象にまとめる (毎ターンの ``last_used_at`` 更新で
  行数を膨らませない)

畳み込み位置 (:class:`EventPosition`) は「どのファイルの何行目まで snapshot に
畳んだか」。manifest に保存し、次回はその続きだけ読む。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from backend.io import JSONLAppendStore
from backend.log_config import get_logger
from backend.utils import utc_now, utc_now_dt

logger = get_logger("rag.evidence.events")

#: 事象行の版。行の形を変えるときに上げる。
EVENT_VERSION = 1

EventOp = Literal["put", "patch", "retract", "touch"]

#: 月次ファイル名 (``2026-09.jsonl``) の stem 形式。
_MONTH_FORMAT = "%Y-%m"

#: 月の順序比較に使う番兵 (読めない stem を最古に寄せる)。
_UNPARSED_MONTH = (-1, -1)


def month_key(month: str) -> tuple[int, int]:
    """月 stem (``2026-09``) → 比較用の ``(year, month)``。

    **文字列の辞書順で比べない** (c_05 §0.5 / CLAUDE.md §6-11)。``2026-9``
    のように 0 詰めを欠いた stem や 5 桁年が 1 つ混じるだけで順序が壊れ、
    畳み込み位置と保持方針が別の月を指す。読めない stem は最古扱いにして
    「未畳み込みの事象を消す」側へ倒さない。
    """
    parts = str(month).split("-")
    if len(parts) != 2:
        return _UNPARSED_MONTH
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return _UNPARSED_MONTH


@dataclass(frozen=True, slots=True)
class EventPosition:
    """snapshot が畳み終わった事象ログ上の位置。

    ``month`` より前のファイルは全て畳み済み、``month`` のファイルは先頭から
    ``line`` 行 (物理行数) まで畳み済みを意味する。``month`` が空文字なら
    「まだ何も畳んでいない」。
    """

    month: str = ""
    line: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {"month": self.month, "line": int(self.line)}

    @classmethod
    def from_payload(cls, payload: Any) -> "EventPosition":
        if not isinstance(payload, dict):
            return cls()
        month = payload.get("month")
        line = payload.get("line")
        return cls(
            month=str(month) if isinstance(month, str) else "",
            line=int(line) if isinstance(line, int) and line >= 0 else 0,
        )

    def _sort_key(self) -> tuple[tuple[int, int], int]:
        """位置の順序比較キー (月は ``(year, month)`` に解いてから比べる)。"""
        return (month_key(self.month), self.line)


class EvidenceEventLog:
    """月次 JSONL の追記専用事象ログ。

    Args:
        events_dir: ``<store>/events/`` ディレクトリ。
        by: 書き手 (``sleep_time`` / ``ralph_loop`` 等)。行の ``by`` に入る。
    """

    def __init__(self, events_dir: Path | str, *, by: str = "unknown") -> None:
        self.events_dir = Path(events_dir)
        self.by = by
        self._stores: dict[str, JSONLAppendStore[dict[str, Any]]] = {}

    # ── 書き込み ──

    def append(
        self,
        op: EventOp,
        record_id: str,
        payload: dict[str, Any],
        *,
        by: str | None = None,
        at: str | None = None,
    ) -> dict[str, Any]:
        """1 事象を追記して、書いた行 (dict) を返す。"""
        event = {
            "_version": EVENT_VERSION,
            "op": op,
            "id": record_id,
            "at": at or utc_now(),
            "by": by or self.by,
            "payload": payload,
        }
        self._store_for(self._month_of(event["at"])).append(event)
        return event

    def append_put(self, record: dict[str, Any], *, by: str | None = None) -> dict[str, Any]:
        """完全な Evidence レコードを ``put`` する。"""
        record_id = str(record.get("id") or "")
        if not record_id:
            raise ValueError("put event requires record['id']")
        return self.append("put", record_id, {"record": record}, by=by)

    def append_patch(
        self, record_id: str, fields: dict[str, Any], *, by: str | None = None,
    ) -> dict[str, Any]:
        """変更フィールドだけを ``patch`` する (c_16 §5.2)。"""
        if not fields:
            raise ValueError("patch event requires at least one field")
        return self.append("patch", record_id, {"fields": dict(fields)}, by=by)

    def append_retract(
        self, record_id: str, reason: str, *, by: str | None = None,
    ) -> dict[str, Any]:
        """``veracity=retracted`` + 理由。物理削除はしない (監査可能性)。"""
        return self.append("retract", record_id, {"reason": reason}, by=by)

    def append_touch(
        self, record_ids: list[str], *, by: str | None = None, at: str | None = None,
    ) -> dict[str, Any] | None:
        """複数 id の ``last_used_at`` を **1 事象** で更新する。

        id が空なら何も書かず ``None`` を返す。
        """
        ids = [rid for rid in dict.fromkeys(record_ids) if rid]
        if not ids:
            return None
        stamp = at or utc_now()
        return self.append(
            "touch", "", {"ids": ids, "last_used_at": stamp}, by=by, at=stamp,
        )

    # ── 読み込み ──

    def months(self) -> list[str]:
        """存在する月次ファイルの stem を昇順で返す。"""
        if not self.events_dir.exists():
            return []
        return sorted(
            (p.stem for p in self.events_dir.glob("*.jsonl")), key=month_key,
        )

    def current_position(self) -> EventPosition:
        """現在の末尾位置 (次回の畳み込み開始点)。

        **畳み込みの前に呼び、``iter_since(..., until=pos)`` へ渡すこと。**
        後から呼ぶと、畳んでいる最中に追記された事象を「畳み済み」と誤記録して
        取りこぼす。
        """
        months = self.months()
        if not months:
            return EventPosition()
        last = months[-1]
        return EventPosition(month=last, line=self._count_lines(self._path(last)))

    def iter_since(
        self, marker: EventPosition | None, *, until: EventPosition | None = None,
    ) -> Iterator[dict[str, Any]]:
        """``marker`` の続きから ``until`` までの事象を古い順に返す。

        壊れた行は WARNING を出して飛ばすが、**位置は物理行で数える** —
        飛ばした行のぶんカウントがずれると、次回の畳み込みが同じ事象を
        二度読む / 読み落とす。
        """
        start = marker or EventPosition()
        stop = until
        start_key = month_key(start.month) if start.month else None
        stop_key = month_key(stop.month) if stop is not None and stop.month else None
        for month in self.months():
            key = month_key(month)
            if start_key is not None and key < start_key:
                continue
            if stop_key is not None and key > stop_key:
                break
            skip = start.line if month == start.month else 0
            limit: int | None = None
            if stop is not None and month == stop.month:
                limit = stop.line
            yield from self._iter_file(self._path(month), skip=skip, limit=limit)

    def _iter_file(
        self, path: Path, *, skip: int, limit: int | None,
    ) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if limit is not None and index >= limit:
                    break
                if index < skip:
                    continue
                text = line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.warning("skipping malformed event line %s:%d: %s", path, index, e)
                    continue
                if isinstance(event, dict) and event.get("op"):
                    yield event
                else:
                    logger.warning("skipping event without op at %s:%d", path, index)

    def count_since(self, marker: EventPosition | None) -> int:
        """``marker`` 以降の事象数 (manifest の ``events_since_snapshot`` 用)。"""
        return sum(1 for _ in self.iter_since(marker))

    # ── 保持 (c_16 §5.4: events_keep_months) ──

    def prune(self, keep_months: int, *, folded_through: EventPosition | None = None) -> list[str]:
        """畳み込み済みの古い月次ファイルを削除する。

        ``folded_through`` の月以降は **消さない** (まだ snapshot に入って
        いない事象を落とすとデータが消える)。

        Returns:
            削除した月の一覧。
        """
        if keep_months < 1:
            return []
        months = self.months()
        if len(months) <= keep_months:
            return []
        boundary = month_key(months[-keep_months])
        folded_month = (folded_through or EventPosition()).month
        folded_key = month_key(folded_month) if folded_month else None
        removed: list[str] = []
        for month in months[:-keep_months]:
            key = month_key(month)
            if key >= boundary:
                continue
            if folded_key is not None and key >= folded_key:
                continue  # 未畳み込みの事象を落とさない
            try:
                self._path(month).unlink()
            except OSError as e:
                logger.warning("failed to prune event file %s: %s", month, e)
                continue
            self._stores.pop(month, None)
            removed.append(month)
        if removed:
            logger.info(
                "Pruned %d event file(s) in %s: %s",
                len(removed), self.events_dir, ", ".join(removed),
            )
        return removed

    # ── 内部 ──

    def _path(self, month: str) -> Path:
        return self.events_dir / f"{month}.jsonl"

    def _month_of(self, at: str) -> str:
        """事象時刻から月次ファイル名を決める (解釈できなければ現在月)。"""
        from backend.utils import parse_utc

        dt = parse_utc(at)
        if dt is None:
            dt = utc_now_dt()
        return dt.strftime(_MONTH_FORMAT)

    def _store_for(self, month: str) -> JSONLAppendStore[dict[str, Any]]:
        """月次ファイルの追記ストア (compaction は **しない**)。

        事象ログは履歴そのものなので、キー重複の後勝ち圧縮を掛けてはいけない。
        ``key_of`` は走査時の生存集合構築にしか使われないため、行を一意に
        識別できる値 (時刻 μs + op + id) を返す。
        """
        store = self._stores.get(month)
        if store is None:
            store = JSONLAppendStore[dict[str, Any]](
                self._path(month),
                serialize=lambda e: json.dumps(e, ensure_ascii=False),
                deserialize=json.loads,
                key_of=lambda e: f"{e.get('at', '')}|{e.get('op', '')}|{e.get('id', '')}",
            )
            self._stores[month] = store
        return store

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)


__all__ = [
    "EVENT_VERSION",
    "EventOp",
    "EventPosition",
    "EvidenceEventLog",
    "month_key",
]
