"""Step 5.9: corpus パッケージの疑似クエリ生成 (f_01 §6.4)。

応答パスで採用された corpus チャンク (``CorpusStore.pq_hits``) と、任意の
backfill 件数ぶんの未生成チャンクに対し、補助タスクで「このチャンクが答える
問い」を作り、各パッケージの :class:`PseudoQueryIndex` へ書いて版を積む。

本 module は EvorefMem pillar 内部扱いだが、実質は EvorefGen の
CartridgeManager / PseudoQueryGenerator を操作するオーケストレーション層
(旧 Step 5.8 の contextual は 2026-09-12 に撤去)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.rag.cartridge_manager import CartridgeManager

logger = get_logger("memory.sleep.pseudo_query")


def _pq_config(config: dict) -> dict:
    return ((config.get("rag") or {}).get("pseudo_query") or {})


#: 静穏窓の既定 (秒)。チャットが終わってからこの秒数は Step 5.9 を始めない。
DEFAULT_QUIET_SECONDS = 60.0


def quiet_seconds(config: dict) -> float:
    """``rag.pseudo_query.quiet_seconds`` (f_01 §6.4)。"""
    try:
        return float(_pq_config(config).get("quiet_seconds", DEFAULT_QUIET_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_QUIET_SECONDS


def _heading_context(corpus, package_id: str, evidence_id: str) -> str:
    """チャンクの見出し経路「親見出し › 節見出し」(無ければ空、f_01 §6.4)。"""
    heading_path = getattr(corpus, "heading_path", None)
    if heading_path is None:
        return ""
    try:
        path = list(heading_path(package_id, evidence_id))
    except Exception:  # noqa: BLE001 — 文脈は任意、生成を止めない
        return ""
    # 文書題 (最上位) は添えない — 問いに文書名が混ざって薄まる (f_01 §6.4)。
    if len(path) > 1:
        path = path[1:]
    return " › ".join(path)


def collect_targets(
    cartridge_manager: "CartridgeManager",
    *,
    max_per_cycle: int,
    backfill_per_cycle: int,
) -> dict[str, list[str]]:
    """生成対象を ``{package_id: [evidence_id, ...]}`` で返す (上限込み)。

    優先順は (0) 取りこぼした問いの語彙候補 (misses、既に問いを持つチャンクも
    対象)、(1) 応答パスで採用されたチャンク、(2) backfill (snapshot 行順)。
    (1)(2) は既に問いを持つチャンクを除く。バッファは drain するので、上限で
    切れた分は次サイクルで再びヒットしたときに拾う (lazy の定義どおり)。
    """
    corpus = cartridge_manager.corpus
    misses = getattr(corpus, "pq_misses", None)
    miss_ids = misses.drain() if misses is not None else []
    hit_ids = corpus.pq_hits.drain()
    by_package: dict[str, list[str]] = {}
    seen: set[str] = set()
    budget = max(0, int(max_per_cycle))

    def push(package_id: str, evidence_id: str) -> bool:
        key = f"{package_id}:{evidence_id}"
        if key in seen:
            return True
        if sum(len(v) for v in by_package.values()) >= budget:
            return False
        seen.add(key)
        by_package.setdefault(package_id, []).append(evidence_id)
        return True

    covered: dict[str, set[str]] = {}

    def is_covered(package_id: str, evidence_id: str) -> bool:
        package = corpus.get(package_id)
        if package is None or package.pseudo_queries is None:
            return True  # 索引を持てないパッケージは対象外
        if package_id not in covered:
            covered[package_id] = package.pseudo_queries.covered_target_ids()
        return evidence_id in covered[package_id]

    for chunk_id in miss_ids:
        package_id, sep, evidence_id = chunk_id.partition(":")
        if not sep or not evidence_id:
            continue
        package = corpus.get(package_id)
        if package is None or package.pseudo_queries is None:
            continue
        if not push(package_id, evidence_id):
            return by_package

    for chunk_id in hit_ids:
        package_id, sep, evidence_id = chunk_id.partition(":")
        if not sep or not evidence_id:
            continue
        if is_covered(package_id, evidence_id):
            continue
        if not push(package_id, evidence_id):
            return by_package

    remaining_backfill = max(0, int(backfill_per_cycle))
    if remaining_backfill <= 0:
        return by_package
    for package_id in corpus.loaded_ids:
        package = corpus.get(package_id)
        if package is None or package.pseudo_queries is None:
            continue
        snapshot = package.store.snapshot
        if snapshot is None:
            continue
        for row in range(len(snapshot)):
            if remaining_backfill <= 0:
                return by_package
            evidence_id = snapshot.id_at(row)
            if not evidence_id or is_covered(package_id, evidence_id):
                continue
            if not push(package_id, evidence_id):
                return by_package
            remaining_backfill -= 1
    return by_package


async def generate_pseudo_queries(
    aux_client: Any,
    *,
    config: dict,
    cartridge_manager: "CartridgeManager | None",
    is_cancelled: Callable[[], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> int:
    """Step 5.9 本体。問いを書いたチャンク数を返す。

    Args:
        aux_client: 補助タスククライアント (``None`` なら何もしない)。
        should_pause: ``True`` を返したらチャンク境界で打ち切る (チャット開始
            への協調 yield)。残りは次サイクルで拾う。
    """
    if aux_client is None or cartridge_manager is None:
        return 0
    cfg = _pq_config(config)
    if not bool(cfg.get("enabled", True)):
        return 0
    from backend.free.rag.pseudo_query import PseudoQueryGenerator, PseudoQueryPreempted

    targets = collect_targets(
        cartridge_manager,
        max_per_cycle=int(cfg.get("max_per_cycle", 50)),
        backfill_per_cycle=int(cfg.get("backfill_per_cycle", 0)),
    )
    total = sum(len(v) for v in targets.values())
    if total == 0:
        return 0
    logger.info("Step 5.9: generating pseudo queries for %d chunk(s)", total)

    generator = PseudoQueryGenerator(
        aux_client,
        questions_per_chunk=int(cfg.get("questions_per_chunk", 2)),
        max_chunk_chars=int(cfg.get("max_chunk_chars", 1200)),
    )
    corpus = cartridge_manager.corpus
    written = 0
    preempted = False
    for package_id, evidence_ids in targets.items():
        package = corpus.get(package_id)
        if package is None or package.pseudo_queries is None:
            continue
        index = package.pseudo_queries
        snapshot = package.store.snapshot
        if snapshot is None:
            continue
        added_here = 0
        for evidence_id in evidence_ids:
            if (is_cancelled and is_cancelled()) or (should_pause and should_pause()):
                logger.info("Step 5.9: paused/cancelled after %d chunk(s)", written)
                break
            row = snapshot.row_of(evidence_id)
            if row is None:
                continue
            text = snapshot.text_at(row)
            take_hints = getattr(corpus, "take_pq_hints", None)
            hints = take_hints(f"{package_id}:{evidence_id}") if take_hints else []
            try:
                questions = await generator.generate(
                    text, hint_questions=hints,
                    context=_heading_context(corpus, package_id, evidence_id),
                )
            except PseudoQueryPreempted:
                # チャットが来た。残りは次サイクルに回す (横取りの往復で
                # GPU を奪い合わない)。書いた分はこの後 commit する。
                logger.info(
                    "Step 5.9: yielded to chat after %d chunk(s); "
                    "remaining targets retry in a later cycle", written,
                )
                preempted = True
                break
            if not questions:
                continue
            hinted_from = getattr(generator, "hinted_from", None)
            try:
                index.add(
                    evidence_id, questions,
                    hinted_from=hinted_from(hints) if hinted_from else None,
                )
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    "Step 5.9: failed to add pseudo queries for %s:%s: %s",
                    package_id, evidence_id, e,
                )
                continue
            added_here += 1
            written += 1
        if added_here:
            try:
                await index.commit()
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    "Step 5.9: failed to commit pseudo-query index for %s: %s",
                    package_id, e,
                )
        if preempted or (is_cancelled and is_cancelled()) or (should_pause and should_pause()):
            break
    if written:
        logger.info("Step 5.9: wrote pseudo queries for %d chunk(s)", written)
        # 問いが増えたので corpus 側の棒を導き直す (署名一致なら no-op、f_01 §6.6)。
        recalibrate = getattr(cartridge_manager, "recalibrate_corpus", None)
        if recalibrate is not None:
            try:
                await recalibrate()
            except Exception as e:  # noqa: BLE001 — 較正の失敗で sleep-time を止めない
                logger.warning("Step 5.9: corpus recalibration failed: %s", e)
    return written


__all__ = ["collect_targets", "generate_pseudo_queries"]
