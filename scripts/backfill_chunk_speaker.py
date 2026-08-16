"""LTM チャンクの ``speaker`` (発話者) を会話履歴から遡及補完する。

``LongTermMemory.absorb_from_short_term`` は 2026-08-16 から
``metadata.json`` の ``speaker`` に発話者 (user / assistant) を残すように
なった。検索側はこれを見て「同じ質問を繰り返したターンでは、その質問への
**前回の回答** を参考情報として返さない」判定を行う
(``memory.pipeline.search_pipeline._drop_past_answers``)。

それ以前に取り込まれたチャンクには ``speaker`` が無く、判定が効かない。
本スクリプトは ``local/history/**/*.json`` の role 付きターンとチャンク本文を
突き合わせて後から埋める。埋まらないチャンク (mdp_trace / 要約など) は
そのまま残す。

使い方::

    python scripts/backfill_chunk_speaker.py            # dry-run (既定)
    python scripts/backfill_chunk_speaker.py --apply    # 実際に書き込む

``--apply`` は ``metadata.json`` を上書きする前に ``.bak`` を作る。
ベクトル本体・チャンク本文には触れないので、失敗しても検索は壊れない。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize(text: str) -> str:
    """突き合わせ用の正規化 (空白のみ除去。曖昧一致はしない)。"""
    return "".join((text or "").split())


def _index_history_turns(history_dir: Path) -> dict[str, str]:
    """``{正規化本文: role}`` を作る。role が食い違う本文は捨てる。"""
    roles: dict[str, str] = {}
    conflicts: set[str] = set()
    for path in sorted(history_dir.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for turn in data.get("turns") or []:
            key = _normalize(turn.get("content"))
            role = turn.get("role")
            if not key or not role:
                continue
            if key in roles and roles[key] != role:
                conflicts.add(key)
            roles[key] = role
    for key in conflicts:
        roles.pop(key, None)
    return roles


def _resolve_speaker(text: str, roles: dict[str, str]) -> str | None:
    """チャンク本文から発話者を引く (完全一致 → 接頭辞一致)。"""
    key = _normalize(text)
    if not key:
        return None
    exact = roles.get(key)
    if exact:
        return exact
    # 要約・切り詰めで前方一致になっているケースを拾う。
    for hist_key, role in roles.items():
        if hist_key.startswith(key) or key.startswith(hist_key):
            return role
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に metadata.json を書き換える (既定は dry-run)",
    )
    parser.add_argument(
        "--local", default=str(PROJECT_ROOT / "local"),
        help="local ディレクトリ (既定: <repo>/local)",
    )
    args = parser.parse_args(argv)

    local = Path(args.local)
    meta_path = local / "vectors" / "metadata.json"
    chunks_dir = local / "vectors" / "chunks"
    history_dir = local / "history"

    if not meta_path.exists():
        print(f"metadata.json not found: {meta_path}")
        return 1
    if not history_dir.exists():
        print(f"history directory not found: {history_dir}")
        return 1

    roles = _index_history_turns(history_dir)
    print(f"indexed {len(roles)} history turns")

    entries = json.loads(meta_path.read_text(encoding="utf-8"))
    stats: Counter[str] = Counter()
    changed = 0
    for entry in entries:
        if entry.get("_store_info"):
            continue
        if entry.get("speaker"):
            stats["already"] += 1
            continue
        chunk_path = chunks_dir / f"{entry.get('id')}.txt"
        if not chunk_path.exists():
            stats["no_chunk"] += 1
            continue
        speaker = _resolve_speaker(
            chunk_path.read_text(encoding="utf-8", errors="replace"), roles,
        )
        if speaker is None:
            stats["unresolved"] += 1
            continue
        entry["speaker"] = speaker
        stats[f"resolved:{speaker}"] += 1
        changed += 1

    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")
    print(f"would set speaker on {changed} chunk(s)")

    if not changed:
        return 0
    if not args.apply:
        print("dry-run (pass --apply to write)")
        return 0

    backup = meta_path.with_suffix(".json.bak")
    shutil.copy2(meta_path, backup)
    meta_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"wrote {meta_path} (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
