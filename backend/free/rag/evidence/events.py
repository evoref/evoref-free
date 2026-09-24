"""事象ログ (c_16 §5.2) — `create` / `put` / `patch` / `retract` / `touch` の追記のみ

稼働中の snapshot / 索引は書き換えず、**版を積んで指す**のが c_16 の不変則。
書き手 (sleep-time) はここへ 1 行追記するだけで、読み手は「snapshot + それ
以降の事象」を畳んで現在状態を得る。

- 月次ファイル ``events/<yyyy-mm>.jsonl``
- 1 行 = ``{"_v", "op", "id", "at", "by", "payload"}`` (``_v`` は行の版、c_05 §0.5.1)。
  版が新しい行は :class:`EvidenceVersionError` (読み手はストアごと readonly)
- 追記は :class:`backend.io.JSONLAppendStore` (同一プロセス内の単一 writer 前提。
  **別プロセスからの書込は禁止** — advisory lock を取らない)
- ``touch`` は複数 id を 1 事象にまとめる (毎ターンの ``last_used_at`` 更新で
  行数を膨らませない)

畳み込み位置 (:class:`EventPosition`) は「どのファイルの何行目まで snapshot に
畳んだか」。manifest に保存し、次回はその続きだけ読む。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from backend.free.rag.evidence.types import EvidenceVersionError
from backend.io import JSONLAppendStore
from backend.io.format_registry import FormatSpec, register_format
from backend.log_config import get_logger
from backend.utils import utc_now, utc_now_dt

logger = get_logger("rag.evidence.events")

#: 事象行の版。行の形を変えるときに上げる。
EVENT_VERSION = 1

#: ``create`` は新しい id の追加 (既存の id なら書き手が拒否する)、``put`` は明示の
#: 全置換 (c_16 §5.2 の G1 の事象)。
EventOp = Literal["create", "put", "patch", "retract", "touch"]

#: 事象ログの形式 (c_05 §0.7.1)。``op`` は ``closed`` — 未知の op の追加は版上げ
#: (c_05 §0.4.4)。
EVIDENCE_EVENT_FORMAT = register_format(FormatSpec(
    format_id="evidence.event",
    version=EVENT_VERSION,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/memory/<store>/events/<yyyy-mm>.jsonl",
    retention="events_keep_months after folding (c_16 §5.4)",
    export=True,
    encodings=("jsonl",),
    enums={"op": "closed"},
))

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
        by: 書き手 (``sleep_time`` 等)。行の ``by`` に入る。
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
            "_v": EVENT_VERSION,
            "op": op,
            "id": record_id,
            "at": at or utc_now(),
            "by": by or self.by,
            "payload": payload,
        }
        self._store_for(self._month_of(event["at"])).append(event)
        return event

    def append_create(self, record: dict[str, Any], *, by: str | None = None) -> dict[str, Any]:
        """新しい id の完全な Evidence レコードを ``create`` する。

        既存の id かどうかの検査は書き手 (``EvidenceStore.create``) が行う。
        """
        record_id = str(record.get("id") or "")
        if not record_id:
            raise ValueError("create event requires record['id']")
        return self.append("create", record_id, {"record": record}, by=by)

    def append_put(self, record: dict[str, Any], *, by: str | None = None) -> dict[str, Any]:
        """完全な Evidence レコードを ``put`` する。"""
        record_id = str(record.get("id") or "")
        if not record_id:
            raise ValueError("put event requires record['id']")
        return self.append("put", record_id, {"record": record}, by=by)

    def append_patch(
        self,
        record_id: str,
        fields: dict[str, Any],
        *,
        unset: list[str] | tuple[str, ...] = (),
        by: str | None = None,
    ) -> dict[str, Any]:
        """変更フィールドだけを ``patch`` する (c_16 §5.2)。

        ``unset`` は消すフィールド / キー (``"valid_until"`` / ``"attrs.<key>"`` 等)。
        値の削除を null で表さない (null は「値が無い」を書く)。
        """
        if not fields and not unset:
            raise ValueError("patch event requires at least one field or unset")
        payload: dict[str, Any] = {"fields": dict(fields)}
        if unset:
            payload["unset"] = list(dict.fromkeys(unset))
        return self.append("patch", record_id, payload, by=by)

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

        壊れた行 (``_v`` の無い行を含む) は WARNING を出して飛ばすが、**位置は物理行で
        数える** — 飛ばした行のぶんカウントがずれると、次回の畳み込みが同じ事象を
        二度読む / 読み落とす。``_v`` が新しい行は :class:`EvidenceVersionError`
        (飛ばさない。知らない形の事象を畳んで書き戻すと失う)。
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
        # 多バイト文字の途中で切れた行は置換文字で読み、壊れた行として飛ばす
        # (strict だと 1 行で月ファイル全体の読み出しが落ちる)。行の位置は物理行で
        # 数えたまま (c_05 §0.5.8)。
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for index, line in enumerate(f):
                if limit is not None and index >= limit:
                    break
                if index < skip:
                    continue
                text = line.strip()
                if not text:
                    continue
                if "\x00" in text:
                    logger.warning("skipping event line with NUL at %s:%d", path, index)
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.warning("skipping malformed event line %s:%d: %s", path, index, e)
                    continue
                if not (isinstance(event, dict) and event.get("op")):
                    logger.warning("skipping event without op at %s:%d", path, index)
                    continue
                version = event.get("_v")
                if not isinstance(version, int) or isinstance(version, bool):
                    logger.warning("skipping event without _v at %s:%d", path, index)
                    continue
                if version > EVENT_VERSION:
                    raise EvidenceVersionError(
                        f"event at {path}:{index} has _v {version}, newer than the supported {EVENT_VERSION}",
                    )
                yield event

    def count_since(self, marker: EventPosition | None) -> int:
        """``marker`` 以降の事象数 (manifest の ``events_since_snapshot`` 用)。"""
        return sum(1 for _ in self.iter_since(marker))

    # ── 耐久性 (c_16 §5.2 / c_05 §0.5.8) ──

    def fsync_range(self, start: EventPosition, end: EventPosition) -> int:
        """``start`` から ``end`` までを含む月ファイルを fsync する (snapshot の前)。

        snapshot が畳んだ事象は prune の後は版の records だけが原本になるので、
        版を作る前に事象側を先に耐久化する。

        Returns:
            fsync したファイル数。
        """
        if not end.month:
            return 0
        start_key = month_key(start.month) if start.month else None
        end_key = month_key(end.month)
        synced = 0
        for month in self.months():
            key = month_key(month)
            if (start_key is not None and key < start_key) or key > end_key:
                continue
            with self._path(month).open("r+b") as f:
                os.fsync(f.fileno())
            synced += 1
        return synced

    def pad_to(self, position: EventPosition) -> int:
        """月ファイルの物理行数が ``position.line`` に足りなければ空行で埋める。

        クラッシュで末尾の行が失われると、manifest の ``folded_through`` が実際の
        行数より先を指す。そのまま追記すると新しい事象が「畳み済み」の位置に
        入り、以後ずっと読み飛ばされる (c_16 §5.2)。欠けた行は空行として計上し、
        追記は本当の末尾の後ろへ続ける。途中で切れた最終行は改行で終端するだけで
        切り詰めない (物理行の位置を保つ)。readonly 中は呼ばない。

        月ファイルそのものが無い場合は作らない — corpus は畳んだ後に ``events/``
        を意図して消す (版が履歴そのもの、c_16 §2.1)。

        Returns:
            足した空行の数。
        """
        if not position.month or position.line <= 0:
            return 0
        path = self._path(position.month)
        if not path.exists():
            return 0
        have = self._count_lines(path)
        missing = position.line - have
        if missing <= 0:
            return 0
        with path.open("ab") as f:
            if f.tell() > 0:
                with path.open("rb") as tail:
                    tail.seek(-1, os.SEEK_END)
                    if tail.read(1) not in (b"\n", b"\r"):
                        f.write(b"\n")
            f.write(b"\n" * missing)
            f.flush()
            os.fsync(f.fileno())
        logger.warning(
            "Event file %s had %d line(s) but the manifest folded through line %d; "
            "padded %d empty line(s)", path, have, position.line, missing,
        )
        return missing

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
                row_version=EVENT_VERSION,
            )
            self._stores[month] = store
        return store

    @staticmethod
    def _count_lines(path: Path) -> int:
        """物理行数 (:meth:`_iter_file` の ``enumerate`` と同じ数え方)。

        テキストモードで 1 行ずつ回すと 50k 行の月ファイルで約 0.2 秒ループが
        止まる (版の生成のたびに呼ばれる)。バイトのまま数える: テキストモードの
        改行は LF / CRLF / 単独の CR で、改行で終わらない末尾も 1 行。
        """
        if not path.exists():
            return 0
        lines = 0
        last = b""
        carried_cr = False
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                lines += chunk.count(b"\n")
                if b"\r" in chunk:  # 普段の事象ログは LF だけ (走査を 1 回で済ませる)
                    lines += chunk.count(b"\r") - chunk.count(b"\r\n")
                if carried_cr and chunk.startswith(b"\n"):
                    lines -= 1  # CRLF がチャンクの境目で割れた
                carried_cr = chunk.endswith(b"\r")
                last = chunk[-1:]
        if last and last not in (b"\n", b"\r"):
            lines += 1
        return lines


__all__ = [
    "EVENT_VERSION",
    "EventOp",
    "EventPosition",
    "EvidenceEventLog",
    "month_key",
]
