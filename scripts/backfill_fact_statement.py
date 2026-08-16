"""既存の SemanticFact に ``statement`` と属性 subject を遡及付与する。

2026-08-16 から抽出器は

- ``statement`` — 発話原文から会話の足場を落とした命題
  (``memory.extractors.chat.normalize_statement``)
- 属性単位の ``subject`` — ``mem.personal.beverage`` / ``mem.personal.food`` 等
  (``notes.note_builder.resolve_fact_attribute``)

を書くようになった。それ以前に抽出されたファクトは ``statement=None`` /
``subject=mem.<kind>.user`` のままで、``[関連する記憶]`` に原文が並び続ける。

本スクリプトは既存ファクトの ``object`` から同じ規則を適用して埋める。
``object`` (証拠) には触らない。埋め込みは ``statement`` から作り直したいので
クリアし、次の sleep-time Step 8.8 が張り直す。

使い方::

    python scripts/backfill_fact_statement.py            # dry-run (既定)
    python scripts/backfill_fact_statement.py --apply    # 実際に書き込む

対象は ``mem.personal.*`` / ``mem.preference.*`` / ``mem.emotion.*`` /
``mem.opinion.*`` の chat 由来ファクトのみ。内部索引 (session summary /
mdp_trace / executable_command) と world_fact は触らない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.free.memory.extractors.chat import (  # noqa: E402
    _KIND_BY_TAG,
    normalize_statement,
)
from backend.free.memory.notes.note_builder import (  # noqa: E402
    ChatNoteBuilder,
    resolve_fact_attribute,
)
from backend.free.memory.notes.subject_ns import make_mem_subject  # noqa: E402
from backend.free.memory.semantic.store import SemanticFactStore  # noqa: E402

#: 対象 FactType (chat extractor が作る user 主語のもの)。
_TARGET_TYPES = ("personal_fact", "preference", "emotion", "opinion")


def _iter_stores(semantic_root: Path):
    """global と各 project の SemanticFactStore を返す。"""
    if (semantic_root / "global").exists():
        yield SemanticFactStore(semantic_root / "global")
    projects = semantic_root / "projects"
    if projects.exists():
        for d in sorted(p for p in projects.iterdir() if p.is_dir()):
            yield SemanticFactStore(d)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に書き込む (既定は dry-run)",
    )
    parser.add_argument(
        "--local", default=str(PROJECT_ROOT / "local"),
        help="local ディレクトリ (既定: <repo>/local)",
    )
    args = parser.parse_args(argv)

    semantic_root = Path(args.local) / "memory" / "semantic"
    if not semantic_root.exists():
        print(f"semantic store not found: {semantic_root}")
        return 1

    builder = ChatNoteBuilder()
    total = touched = subj_changed = stmt_added = 0

    for store in _iter_stores(semantic_root):
        for fact in list(store.all_facts(include_superseded=False)):
            if fact.type not in _TARGET_TYPES:
                continue
            total += 1
            obj = fact.object or ""
            triggers = tuple(builder.fact_triggers.get(fact.type, ()))
            changes: dict = {}

            statement = normalize_statement(obj, triggers)
            if statement and statement != fact.statement:
                changes["statement"] = statement

            attr = resolve_fact_attribute(obj, fact.type, mode="chat")
            if attr:
                kind = _KIND_BY_TAG.get(fact.type)
                if kind:
                    new_subject = make_mem_subject(kind, attr)
                    if new_subject != fact.subject:
                        changes["subject"] = new_subject

            if not changes:
                continue
            touched += 1
            if "subject" in changes:
                subj_changed += 1
            if "statement" in changes:
                stmt_added += 1
            print(f"  {fact.id}")
            print(f"    subject  : {fact.subject}"
                  f"{' -> ' + changes['subject'] if 'subject' in changes else ''}")
            print(f"    object   : {obj[:62]}")
            if "statement" in changes:
                print(f"    statement: {changes['statement'][:62]}")
            if args.apply:
                # 埋め込みは statement から作り直す。None にすると
                # sleep-time Step 8.8 が次サイクルで張り直す。
                # touch=False — 保守処理はアクセスではない。
                store.update_fact(
                    fact.id, touch=False, embedding=None, **changes,
                )

    print()
    print(f"対象ファクト: {total} 件")
    print(f"更新: {touched} 件 (subject 変更 {subj_changed} / statement 付与 {stmt_added})")
    if not args.apply:
        print("dry-run (書き込むには --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
