"""evoref forget サブコマンド — 記憶ファクトを ID 指定で取り下げる (f_06 §2.7)。

    evoref forget --id FACT_ID [--id ...] [--with-notes] [--follow-chain] [--reason TEXT] [--apply]

取り下げは ``SemanticStore.retract_fact`` (``veracity=retracted``、c_16 §3) で、物理削除ではない。
事象ログに ``{reason, by: "cli.forget"}`` が残る。**停止中のみ**: 単一書き手ロック
(``store/.writer.lock``) を取るので、serve の稼働中は拒否する。既定は dry-run で、
``--apply`` を付けたときだけ書く。

取り下げの副作用は 2 つあり、dry-run で先に見せる。

- 取り下げたファクトに ``superseded_by`` で置き換えられていた旧値は live に戻る
  (``SemanticStore._clear_dangling_supersession``)。汚染が旧値にも及ぶときは
  ``--follow-chain`` で旧値も続けて取り下げる。
- 根拠の episodic ノートは既定では残り、注入にも残る (``duplicate_of_live_fact`` は減点で
  落とさない)。誤りが利用者の発話そのものにあるときは ``--with-notes`` で一緒に取り下げる。

多値スロット (``multi_valued``) は兄弟の値を巻き込まないよう ID 単体だけを対象にする
(CLAUDE.md 不変則 #13)。一括指定 (``--subject`` 等) は持たない。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import PathResolver
from backend.data_root import DataRootError, resolve_data_root, store_root
from backend.io.writer_lock import WriterLockHeld, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("cli.forget")

#: 取り下げ事象の書き手 (事象ログの ``by``)。
FORGET_BY = "cli.forget"


@dataclass(slots=True)
class ForgetPlan:
    """取り下げの計画 (dry-run で表示し、``--apply`` で実行する)。"""

    #: 取り下げるファクトの id (指定順、``--follow-chain`` の旧値を含む)。
    fact_ids: list[str] = field(default_factory=list)
    #: 指定されたが見つからない (または取り下げ済みの) id。
    missing: list[str] = field(default_factory=list)
    #: 取り下げで live に戻る旧値の id (``--follow-chain`` なしのとき)。
    revived: list[str] = field(default_factory=list)
    #: 同じ subject に残る live の別ファクトの id (多値スロットの兄弟)。
    siblings: list[str] = field(default_factory=list)
    #: 多値スロットのファクトの id (兄弟は巻き込まない)。
    multi_valued: list[str] = field(default_factory=list)
    #: 根拠ノートの id (存在し取り下げ済みでないもの)。
    note_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ForgetReport:
    """``--apply`` の結果。"""

    retracted_facts: int = 0
    retracted_notes: int = 0


def _note_ids_of(fact) -> list[str]:
    """ファクトの根拠ノートの id (provenance の ``note_id``、順序を保ち重複を除く)。"""
    seen: dict[str, None] = {}
    for prov in getattr(fact, "provenances", None) or ():
        note_id = getattr(prov, "note_id", None)
        if note_id:
            seen.setdefault(note_id, None)
    return list(seen)


def plan_forget(semantic, episodic, fact_ids: list[str], *, follow_chain: bool) -> ForgetPlan:
    """``fact_ids`` の取り下げ計画を作る (ストアは変えない)。"""
    from backend.free.memory.notes.note_builder import (
        is_multi_valued_subject,
        is_span_only_fold_subject,
    )

    plan = ForgetPlan()
    targets: dict[str, None] = {}
    for fact_id in fact_ids:
        if semantic.get_fact(fact_id) is None:
            plan.missing.append(fact_id)
        else:
            targets.setdefault(fact_id, None)
    # 旧値の連鎖: 取り下げる id を指していた superseded_by が外れて live に戻る
    queue = list(targets)
    revived: dict[str, None] = {}
    while queue:
        for old_id in semantic.supersedes_of(queue.pop()):
            if old_id in targets or old_id in revived:
                continue
            revived[old_id] = None
            if follow_chain:
                targets[old_id] = None
                queue.append(old_id)
    plan.fact_ids = list(targets)
    plan.revived = [] if follow_chain else list(revived)

    note_ids: dict[str, None] = {}
    subjects: dict[str, None] = {}
    for fact_id in plan.fact_ids:
        fact = semantic.get_fact(fact_id)
        subject = str(fact.subject or "")
        subjects.setdefault(subject, None)
        if is_multi_valued_subject(subject) or is_span_only_fold_subject(subject):
            plan.multi_valued.append(fact_id)
        for note_id in _note_ids_of(fact):
            record = episodic.evidence.get(note_id)
            if record is not None and record.veracity != "retracted":
                note_ids.setdefault(note_id, None)
    plan.note_ids = list(note_ids)
    for subject in subjects:
        for other in semantic.search_by_subject(subject, include_superseded=False):
            if other.id not in targets and other.id not in plan.siblings:
                plan.siblings.append(other.id)
    return plan


def forget_facts(
    semantic, episodic, plan: ForgetPlan, *, reason: str, with_notes: bool,
) -> ForgetReport:
    """計画どおりに取り下げる (書き手ロックは呼出側が持つ)。"""
    report = ForgetReport()
    text = f"forget: {reason}" if reason else "forget"
    for fact_id in plan.fact_ids:
        if semantic.retract_fact(fact_id, text, by=FORGET_BY):
            report.retracted_facts += 1
    if with_notes:
        for note_id in plan.note_ids:
            if episodic.retract_note(note_id, text, by=FORGET_BY) is not None:
                report.retracted_notes += 1
    semantic.save_manifest()
    if with_notes:
        episodic.evidence.save_manifest()
    return report


def _build_parser() -> argparse.ArgumentParser:
    from backend.i18n_helper import msg

    parser = argparse.ArgumentParser(prog="evoref forget", description=msg("cli.help_forget"))
    parser.add_argument(
        "--id", dest="fact_ids", action="append", required=True, metavar="FACT_ID",
        help="Fact id to retract (repeatable)",
    )
    parser.add_argument(
        "--with-notes", action="store_true",
        help="Also retract the episodic notes the fact was extracted from",
    )
    parser.add_argument(
        "--follow-chain", action="store_true",
        help="Also retract the older values that would become live again",
    )
    parser.add_argument("--reason", default="", help="Why (recorded in the event log)")
    parser.add_argument("--apply", action="store_true", help="Actually retract (default: dry-run)")
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    return parser


def _describe(fact) -> str:
    return f"[{fact.id}] {fact.subject} {fact.predicate} {str(fact.statement or fact.object)[:70]!r}"


def _print_plan(console, semantic, plan: ForgetPlan, *, with_notes: bool) -> None:
    from backend.i18n_helper import msg

    def show(text: str) -> None:
        # ファクトの本文に [...] が入りうるので rich のマークアップとして解釈させない
        console.print(text, markup=False, highlight=False)

    for fact_id in plan.fact_ids:
        show(msg("cli.forget_target", fact=_describe(semantic.get_fact(fact_id))))
    for fact_id in plan.multi_valued:
        show(msg("cli.forget_multi_valued", id=fact_id))
    for fact_id in plan.siblings:
        show(msg("cli.forget_sibling", fact=_describe(semantic.get_fact(fact_id))))
    for fact_id in plan.revived:
        show(msg("cli.forget_revived", fact=_describe(semantic.get_fact(fact_id))))
    for note_id in plan.note_ids:
        show(msg("cli.forget_note_retract" if with_notes else "cli.forget_note_kept", id=note_id))


def run_forget(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref forget``)。"""
    from backend.free.cli.config_loader import _find_project_root
    from backend.free.cli.renderer import create_console, render_error
    from backend.i18n_helper import init_i18n, msg

    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console(no_color=not (sys.stdout and sys.stdout.isatty()))
    try:
        data_root = resolve_data_root(args.data_root, root=_find_project_root())
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return 1
    try:
        lock = acquire_writer_lock(store_root(data_root))
    except WriterLockHeld as e:
        render_error(console, msg("cli.forget_locked", detail=str(e)))
        return 1
    try:
        return _run_locked(console, data_root, args)
    finally:
        lock.release()


def _run_locked(console, data_root: Path, args: argparse.Namespace) -> int:
    from backend.free.cli.renderer import render_error, render_info
    from backend.free.memory.episodic.store import EpisodicStore
    from backend.free.memory.semantic.store import STORE_DIRNAME, SemanticStore
    from backend.free.rag.evidence.store import EvidenceStoreReadonlyError
    from backend.i18n_helper import msg

    memory_dir = PathResolver.layout_path(data_root, "memory_dir")
    if not (memory_dir / STORE_DIRNAME).is_dir():
        render_error(console, msg("cli.forget_no_store", path=str(memory_dir / STORE_DIRNAME)))
        return 1
    semantic = SemanticStore(memory_dir)
    episodic = EpisodicStore(memory_dir)
    try:
        semantic.load()
        episodic.load()
        plan = plan_forget(
            semantic, episodic, list(dict.fromkeys(args.fact_ids)), follow_chain=args.follow_chain,
        )
        for fact_id in plan.missing:
            render_error(console, msg("cli.forget_not_found", id=fact_id))
        if not plan.fact_ids:
            return 1
        _print_plan(console, semantic, plan, with_notes=args.with_notes)
        if not args.apply:
            render_info(console, msg("cli.forget_dry_run", count=len(plan.fact_ids)))
            return 1 if plan.missing else 0
        try:
            report = forget_facts(
                semantic, episodic, plan, reason=args.reason, with_notes=args.with_notes,
            )
        except EvidenceStoreReadonlyError as e:
            render_error(console, msg("cli.forget_readonly", detail=str(e)))
            return 1
        render_info(
            console,
            msg("cli.forget_done", facts=report.retracted_facts, notes=report.retracted_notes),
        )
        return 1 if plan.missing else 0
    finally:
        semantic.close()
        episodic.close()


if __name__ == "__main__":
    sys.exit(run_forget(sys.argv[1:]))
