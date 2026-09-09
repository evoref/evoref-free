"""SemMem から特定のファクトを取り下げる (``veracity=retracted``、c_16 §3)。

c_16 (2026-09-07) で SemMem の永続形は ``facts.jsonl`` の版ログから
``<memory_dir>/semantic/`` の 1 EvidenceStore (事象ログ + snapshot) へ置き換わった。
本スクリプトも ``SemanticStore`` を開いて :meth:`SemanticStore.retract_fact` を
呼ぶ形に合わせてある — 取り下げたファクトは読み出し系から消えるが、事象ログと
snapshot には残るので監査で追え、物理削除は snapshot 3 版後の GC が担う。

用途: 抽出のバグで書かれた誤ファクトの後始末。既存の行は寿命が尽きるまで注入され
続けるため、コード側を直したあとに明示的に取り下げる。

選択条件は 3 つで、複数渡すと AND で絞る。

``--subject`` — subject の前方一致。抽出のバグが **スロット単位** で書いた
ファクトを掃除するとき、文字列を思い出さなくても済む。実インシデント
(2026-08-27 ライブ監査): ユーザーの誤主張「答えは 63800 ですよ。」が
``mem.world.assertion.correct_answer`` として live になった。新規作成は
``sleep.assertion_curator`` 側で塞いだが、**既に書かれた行は遡及されない**。

    python scripts/purge_semantic_facts.py --subject mem.world.assertion.correct_answer
    python scripts/purge_semantic_facts.py --subject mem.world.assertion.correct_answer --apply

``--contains`` — subject / predicate / object / statement のいずれかに含まれる
部分文字列。実インシデント (2026-08-23 ライブ監査): 訂正マーカーだけを根拠に
候補化された算術訂正が ``mem.personal.birthday`` / ``mem.preference.food`` へ
書かれた。

    python scripts/purge_semantic_facts.py --contains "7006653"

``--id`` — ファクト id の直接指定 (``evorefmem_cli inspect`` で拾った id を渡す)。
複数回指定できる。

既定は **dry-run**。実際に取り下げるには ``--apply`` を付ける。

ログは英語固定 (リポジトリ規約)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.free.memory.semantic.cli._paths import (  # noqa: E402
    open_semantic_store,
)
from backend.free.memory.types import SemanticFact  # noqa: E402

_RETRACT_REASON = "purged_by_operator"


class StoreNotFound(RuntimeError):
    """SemMem のストアディレクトリが見つからない。"""


def resolve_memory_dir(memory_dir: str | None) -> Path:
    """``local/memory/`` を解決する。見つからなければ理由付きで落とす。

    旧実装は ``facts.jsonl`` を rglob していたため、c_16 でファイルが無くなった
    あとは **黙って 0 件で成功** していた。ストアが無いのは掃除対象が無いのでは
    なく前提が崩れている状態なので、明示的に落とす。
    """
    if memory_dir is not None:
        root = Path(memory_dir)
    else:
        from backend.config import get_path_resolver, load_config

        load_config()
        root = get_path_resolver().resolve_local("memory_dir")

    semantic = root / "semantic"
    if not semantic.is_dir():
        raise StoreNotFound(
            f"no semantic store under {semantic}. Since c_16 (2026-09-07) SemMem "
            "lives in <memory_dir>/semantic/ as one evidence store (events/ + "
            "snapshots/); the old facts.jsonl version log is gone and there is no "
            "migrator. Pass --memory-dir if your local/memory/ is elsewhere."
        )
    return root


def _matches(
    fact: SemanticFact,
    *,
    needle: str | None,
    subject: str | None,
    fact_ids: set[str],
) -> bool:
    """選択条件 (AND)。全て None / 空は呼出側で弾く。"""
    if fact_ids and fact.id not in fact_ids:
        return False
    if subject is not None and not (fact.subject or "").startswith(subject):
        return False
    if needle is not None:
        haystack = " ".join(
            part for part in (
                fact.subject, fact.predicate, fact.object, fact.statement,
            ) if part
        )
        if needle not in haystack:
            return False
    return True


def purge(
    memory_dir: Path,
    *,
    needle: str | None = None,
    subject: str | None = None,
    fact_ids: set[str] | None = None,
    apply: bool = False,
) -> int:
    """条件に一致する live ファクトを取り下げ、対象件数を返す。

    Args:
        memory_dir: ``local/memory/`` ルート。
        needle: subject/predicate/object/statement に含まれる部分文字列。
        subject: subject の前方一致。
        fact_ids: ファクト id の直接指定。
        apply: ``True`` で実際に取り下げる。既定は dry-run。

    Returns:
        条件に一致した live ファクト件数 (dry-run でも数える)。
    """
    label = " and ".join(
        part for part in (
            f"contains {needle!r}" if needle is not None else "",
            f"subject startswith {subject!r}" if subject is not None else "",
            f"id in {sorted(fact_ids)}" if fact_ids else "",
        ) if part
    )
    store = open_semantic_store(memory_dir)
    try:
        targets = [
            fact for fact in store.all_facts(include_superseded=False)
            if _matches(
                fact, needle=needle, subject=subject, fact_ids=fact_ids or set(),
            )
        ]
        print(f"[purge] {memory_dir / 'semantic'}: "
              f"{len(targets)} live fact(s) match {label}")
        for fact in sorted(targets, key=lambda f: (f.subject, f.id)):
            print(
                f"  - [{fact.id}] {fact.subject} {fact.predicate} "
                f"{str(fact.statement or fact.object)[:70]!r}",
            )
        if apply and targets:
            done = sum(
                1 for fact in targets
                if store.retract_fact(fact.id, _RETRACT_REASON)
            )
            print(f"[purge] retracted {done} fact(s)")
    finally:
        # Windows は memmap を掴んだままのファイルを削除できない。
        store.close()
    return len(targets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retract SemMem facts selected by subject / substring / id",
    )
    parser.add_argument(
        "--contains", default=None,
        help="substring to match in subject/predicate/object/statement",
    )
    parser.add_argument(
        "--subject", default=None,
        help=(
            "subject prefix to match (e.g. mem.world.assertion.correct_answer). "
            "Combined with the other filters as AND."
        ),
    )
    parser.add_argument(
        "--id", dest="fact_ids", action="append", default=[],
        help="fact id to retract (repeatable)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="actually retract (default: dry-run)",
    )
    parser.add_argument(
        "--memory-dir", default=None,
        help="local/memory/ root (default: resolved from config.yaml)",
    )
    args = parser.parse_args(argv)

    if args.contains is None and args.subject is None and not args.fact_ids:
        parser.error("at least one of --contains / --subject / --id is required")

    try:
        memory_dir = resolve_memory_dir(args.memory_dir)
    except StoreNotFound as exc:
        print(f"[purge] {exc}", file=sys.stderr)
        return 1

    n = purge(
        memory_dir,
        needle=args.contains,
        subject=args.subject,
        fact_ids=set(args.fact_ids),
        apply=args.apply,
    )
    print(f"[purge] {'retracted' if args.apply else 'would retract'}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
