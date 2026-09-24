"""形式台帳へ宣言を登録するモジュールの一覧 (c_05 §0.7)。

形式の宣言 (``FormatSpec``) は各 pillar のモジュールが import 時に登録する。
アプリ本体は配線で全モジュールを import するので自然に揃うが、停止中の CLI
(lock の更新・doctor) とテストは :func:`load_all_formats` で明示的に揃える。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from typing import Any

#: 形式を宣言するモジュール。宣言を足したらここにも足す
#: (``backend/tests/test_format_ledger.py`` が片落ちを検出する)。
DECLARING_MODULES: tuple[str, ...] = (
    "backend.free.agent.agent_trace_store",
    "backend.free.agent.aux_prompt_manager",
    "backend.free.agent.learned_pattern_store",
    "backend.free.agent.prompt_ledger",
    "backend.free.agent.prompt_manager",
    "backend.free.cli.command_handlers",
    "backend.free.core.model_migration",
    "backend.free.core.policy_interpreter",
    "backend.free.generation.token_budget",
    "backend.free.history.history_manager",
    "backend.free.learning.exploration_controller",
    "backend.free.learning.fewshot_pool",
    "backend.free.learning.generation_delta_store",
    "backend.free.learning.learning_state_store",
    "backend.free.learning.level0_instant",
    "backend.free.learning.level1_session",
    "backend.free.learning.policy_evolver",
    "backend.free.llm.aux_calibration_store",
    "backend.free.llm.quality_probe",
    "backend.free.loop.log_ingestor",
    "backend.free.loop.staged.run_record",
    "backend.free.memory._defaults",
    "backend.free.memory.episodic.progress",
    "backend.free.memory.local_state_store",
    "backend.free.memory.notes.mdp_ingester",
    "backend.free.memory.notes.subject_canonicalizer",
    "backend.free.memory.pipeline.semantic_conflict_resolver",
    "backend.free.memory.semantic.sources",
    "backend.free.optimizer.embed_instruction_evolver",
    "backend.free.optimizer.prompt_evolver",
    "backend.free.rag.corpus.calibration",
    "backend.free.rag.corpus.pseudo_queries",
    "backend.free.rag.corpus.store",
    "backend.free.rag.dimension_check",
    "backend.free.rag.embedding_cache",
    "backend.free.rag.evidence.events",
    "backend.free.rag.evidence.manifest",
    "backend.free.rag.evidence.snapshot",
    "backend.free.rag.evidence.store",
    "backend.free.rag.evidence.types",
    "backend.free.rag.memory_threshold_calibration",
    "backend.free.rag.projectmap.fingerprint",
    "backend.io.generation_seal",
    "backend.io.ledger_files",
    "backend.io.writer_lock",
    "backend.liveness",
    "backend.model_key",
)

#: Pro の形式を宣言するモジュール (``register_pro_format``)。Pro が同梱されている
#: ときだけ読む (Free 単独の配布物には無い)。
PRO_DECLARING_MODULES: tuple[str, ...] = (
    "backend.pro.cartridge_creator",
    "backend.pro.eval_core_manager",
    "backend.pro.learning.level2_trainer",
    "backend.pro.version_manager",
)


def load_all_formats() -> None:
    """宣言モジュールを全て import して既定の台帳を揃える (Pro は同梱時だけ)。"""
    for name in DECLARING_MODULES:
        importlib.import_module(name)
    if importlib.util.find_spec("backend.pro") is None:
        return
    for name in PRO_DECLARING_MODULES:
        importlib.import_module(name)


def build_current_lock() -> dict[str, Any]:
    """全宣言を揃えた現行の lock (形式・世代・G0 検出の署名)。"""
    from backend.factory._data_gate import G0_SIGNATURES
    from backend.io.format_lock import build_lock

    load_all_formats()
    return build_lock(g0_signatures=G0_SIGNATURES)


def freeze_fixtures() -> list[Any]:
    """無い golden fixture だけを凍結する (c_05 §0.7.2 R4)。既存の fixture は書き換えない。

    データ根は台帳の完全性テストと同じ実際の書き手で作る (開発用。配布物には無い)。
    """
    from backend.tests.golden_fixtures import freeze, frozen_formats
    from backend.tests.test_ledger_completeness import build_data_root

    load_all_formats()
    specs = list(frozen_formats("free"))
    write_pro = None
    try:  # Free の配布物に Pro は無い (無ければ Free の形式だけ凍結する)
        from backend.pro.tests.test_ledger_completeness_pro import write_pro
    except ImportError:
        pass
    else:
        specs += frozen_formats("pro")
    return freeze(specs, lambda tmp: build_data_root(tmp, write_pro=write_pro))


def main(argv: list[str] | None = None) -> int:
    """lock (c_05 §0.7.2) の検査と更新、golden fixture の凍結。違反 1 / 更新待ち 2 / 一致 0。"""
    from backend.io.format_lock import LOCK_PATH, compare, load_saved, write_lock

    parser = argparse.ArgumentParser(description="format ledger lock and golden fixtures (c_05 §0.7.2)")
    parser.add_argument("--check", action="store_true", help="report only (default)")
    parser.add_argument("--update", action="store_true", help="rewrite the lock when there are no violations")
    parser.add_argument(
        "--freeze-fixtures", action="store_true",
        help="write the golden fixture tests/fixtures/formats/g1/<format_id>/v<version>/ of every "
             "sot / system format whose current version has none, using the real writers "
             "(existing fixtures are frozen and never overwritten; after a version bump this "
             "adds the new version's fixture next to the old one)",
    )
    args = parser.parse_args(argv)

    if args.freeze_fixtures:
        for path in freeze_fixtures():
            print(f"froze {path}")
        return 0

    current = build_current_lock()
    violations, stale = compare(load_saved(), current)
    for line in violations:
        print(f"VIOLATION {line}")
    for line in stale:
        print(f"update    {line}")
    if violations:
        return 1
    if args.update and stale:
        write_lock(current)
        print(f"wrote {LOCK_PATH}")
        return 0
    return 2 if stale else 0


__all__ = [
    "DECLARING_MODULES",
    "PRO_DECLARING_MODULES",
    "build_current_lock",
    "freeze_fixtures",
    "load_all_formats",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
