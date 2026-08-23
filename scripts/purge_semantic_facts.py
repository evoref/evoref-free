"""SemMem から特定のファクトを取り消す (append-only の版ログへ tombstone を追記)。

``facts.jsonl`` は **append-only の版ログ** なので、行を消すのではなく
``superseded_by`` を立てた新しい版を追記して無効化する。読み出し側
(``all_facts(include_superseded=False)``) はこれで対象を返さなくなる。

用途: 抽出のバグで書かれた誤ファクトの後始末。実インシデント
(2026-08-23 ライブ監査): 訂正マーカーだけを根拠に候補化された算術訂正が
``mem.personal.birthday`` / ``mem.preference.food`` へ書かれた
(``note_builder.candidate_fact_tags`` で恒久対応済み)。既存の行は寿命が
尽きるまで注入され続けるため、明示的に取り消す。

    python scripts/purge_semantic_facts.py --contains "7006653" --dry-run
    python scripts/purge_semantic_facts.py --contains "7006653"

ログは英語固定 (リポジトリ規約)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOMBSTONE = "purged_by_operator"


def _iter_fact_files(local_root: Path) -> list[Path]:
    base = local_root / "memory" / "semantic"
    if not base.exists():
        return []
    return sorted(base.rglob("facts.jsonl"))


def _load_versions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _live_facts(rows: list[dict]) -> dict[str, dict]:
    """fact_id ごとの最新版のうち、まだ superseded されていないもの。"""
    latest: dict[str, dict] = {}
    for row in rows:
        fid = str(row.get("id") or row.get("fact_id") or "")
        if fid:
            latest[fid] = row
    return {
        fid: row for fid, row in latest.items()
        if not row.get("superseded_by")
    }


def purge(local_root: Path, needle: str, *, dry_run: bool) -> int:
    total = 0
    for path in _iter_fact_files(local_root):
        rows = _load_versions(path)
        live = _live_facts(rows)
        targets = [
            row for row in live.values()
            if needle in json.dumps(row, ensure_ascii=False)
        ]
        if not targets:
            continue
        print(f"[purge] {path}: {len(targets)} live fact(s) match {needle!r}")
        for row in targets:
            print(
                f"  - {row.get('subject')} {row.get('predicate')} "
                f"{str(row.get('object'))[:70]!r}",
            )
        total += len(targets)
        if dry_run:
            continue
        with path.open("a", encoding="utf-8") as f:
            for row in targets:
                tomb = dict(row)
                tomb["superseded_by"] = _TOMBSTONE
                f.write(json.dumps(tomb, ensure_ascii=False) + "\n")
        print(f"[purge] appended {len(targets)} tombstone version(s)")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Supersede SemMem facts whose JSON contains a substring",
    )
    parser.add_argument("--contains", required=True, help="substring to match")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-root", default=None)
    args = parser.parse_args(argv)

    root = (
        Path(args.local_root)
        if args.local_root
        else Path(__file__).resolve().parent.parent / "local"
    )
    n = purge(root, args.contains, dry_run=args.dry_run)
    print(f"[purge] {'would supersede' if args.dry_run else 'superseded'}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
