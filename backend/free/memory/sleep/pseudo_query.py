"""Step 5.9: corpus パッケージの疑似クエリ生成 (f_01 §6.4)。

応答パスで採用された corpus チャンク (``CorpusStore.pq_hits``) と、任意の
backfill 件数ぶんの未生成チャンクに対し、補助タスクで「このチャンクが答える
問い」を作り、各パッケージの :class:`PseudoQueryIndex` へ書いて版を積む。

本 module は EvorefMem pillar 内部扱いだが、実質は EvorefGen の
CartridgeManager / PseudoQueryGenerator を操作するオーケストレーション層
(旧 Step 5.8 の contextual は 2026-09-12 に撤去)。
"""

from __future__ import annotations

import asyncio
import time
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

#: 保存前の品質検査 (c_16 §7.2.1 の第 5 段階 B) の件数。Full の結果辞書に必ず載せる
#: (再順位段が無い回・失敗を握り潰した回も 0、死活監視は c_07 §7.1)。
RERANK_SCORED_KEY = "pseudo_query_rerank_scored"
RERANK_DROPPED_KEY = "pseudo_query_rerank_dropped"
RERANK_UNSCORED_KEY = "pseudo_query_rerank_unscored"
#: 採点を試みた問いの数 (``RERANK_SCORED_KEY`` の入力件数)。再順位段がある回だけ付ける。
RERANK_SCORED_INPUT_KEY = RERANK_SCORED_KEY + "_input"
#: 前のサイクルが put したまま commit しなかった問いのうち、この回に畳んだ数 (f_01 §6.4)。
#: 事象は問い 1 件 = 1 行なので、畳んだ事象の数と同じ。
#: Full の結果辞書に必ず載せる (無い回も 0)。
CARRIED_OVER_KEY = "pseudo_query_carried_over"


def empty_rerank_stats() -> dict[str, int]:
    """品質検査の件数の初期値 (全部 0、入力件数は付けない)。"""
    return {RERANK_SCORED_KEY: 0, RERANK_DROPPED_KEY: 0, RERANK_UNSCORED_KEY: 0}


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
            covered[package_id] = set(package.pseudo_queries.covered_target_ids())
            # 品質検査で問いを全部捨てたチャンクは、このプロセスの間は作り直さない
            # (backfill が同じチャンクで毎サイクル予算を使い切らないように)。
            screened = getattr(package.pseudo_queries, "screened_out_target_ids", None)
            if screened is not None:
                covered[package_id] |= set(screened())
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


async def _screen_questions(
    package_id: str,
    generated: list[tuple[str, str, list[str], int | None]],
    *,
    reranker: Any,
    stats: dict[str, int],
    is_cancelled: Callable[[], bool] | None,
    should_pause: Callable[[], bool] | None,
) -> None:
    """生成した問いを元チャンクと組にして採点し、判定点が反対した問いを空文字にする。

    c_16 §7.2.1 の第 5 段階 B / c_17 §3.15。``generated`` の問いのリストをその場で書き換える
    (空文字は ``PseudoQueryIndex.add`` が飛ばすが位置は進めるので、残った問いの id・
    ``from_hint`` は変わらない)。各呼出の前に静穏窓とキャンセルを見て、来ていたら残りは
    採点せずに残す (``unscored``)。採点の縮退・棄権も残す。
    """
    from backend.free.rag.corpus.pseudo_queries import pseudo_query_id
    from backend.free.rag.pseudo_query_gate import evaluate_pseudo_query

    calibration = getattr(reranker, "pseudo_query_calibration", None)
    pending = [
        (item, position)
        for item in generated
        for position, question in enumerate(item[2])
        if question
    ]
    for done, (item, position) in enumerate(pending):
        if (is_cancelled and is_cancelled()) or (should_pause and should_pause()):
            stats[RERANK_UNSCORED_KEY] += len(pending) - done
            logger.info(
                "Step 5.9: stopped scoring pseudo queries for chat after %d of %d; "
                "the rest are kept unscored", done, len(pending),
            )
            return
        evidence_id, text, questions, hinted_from = item
        question = questions[position]
        stats[RERANK_SCORED_INPUT_KEY] = stats.get(RERANK_SCORED_INPUT_KEY, 0) + 1
        try:
            scores = await reranker.rerank(question, [text], ids=[f"{package_id}:{evidence_id}"])
        except Exception as e:  # noqa: BLE001 — 採点の失敗で生成した問いを失わない
            logger.warning("Step 5.9: pseudo-query scoring failed: %s: %s", type(e).__name__, e)
            scores = None
        logit = float(scores[0]) if scores is not None and len(scores) == 1 else None
        if logit is not None:
            stats[RERANK_SCORED_KEY] += 1
        verdict = evaluate_pseudo_query(
            logit, calibration,
            package_id=package_id, target_id=evidence_id, position=position,
            pq_id=pseudo_query_id(evidence_id, position, question.strip()),
            from_hint=hinted_from is not None and position >= hinted_from,
        )
        if verdict.band == "skip":
            questions[position] = ""
            stats[RERANK_DROPPED_KEY] += 1


def _count_questions(generated: list[tuple[str, str, list[str], int | None]]) -> int:
    return sum(1 for item in generated for q in item[2] if q)


def _decisions_recorded() -> bool:
    from backend.free.rag.pseudo_query_gate import decisions_recorded

    return decisions_recorded()


def _add_generated(
    index: Any, package_id: str, generated: list[tuple[str, str, list[str], int | None]],
) -> int:
    """``generated`` を索引へ書いて空にする (put は同期)。書いたチャンク数を返す。

    問いを全部捨てたチャンクは書かず、このプロセスの間は作り直さない印を付ける
    (:func:`collect_targets`)。
    """
    written = 0
    items = list(generated)
    generated.clear()
    for evidence_id, _text, questions, hinted in items:
        if not any(questions):
            mark = getattr(index, "mark_screened_out", None)
            if mark is not None:
                mark(evidence_id)
            continue
        try:
            index.add(evidence_id, questions, hinted_from=hinted)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(
                "Step 5.9: failed to add pseudo queries for %s:%s: %s",
                package_id, evidence_id, e,
            )
            continue
        written += 1
    return written


async def _commit_carried_over(
    corpus: Any, stats: dict[str, int], is_cancelled: Callable[[], bool] | None,
) -> int:
    """前のサイクルが put したまま commit しなかったパッケージを畳む。畳んだ事象の数を返す。

    Full の取消は生成済みの問いを put してから上げるが commit (埋め込み) は走らせないので、
    そのパッケージが次のサイクルの対象に入らなければ問いは事象ログに残ったまま検索に出ない
    (f_01 §6.4、2026-09-30 実機: manual の 14 件)。対象の有無にかかわらずここで畳む。
    事象は問い 1 件 = 1 行 (``put_many``) なので、数は畳んだ問いの数と同じ。
    1 パッケージの失敗 (読めない索引・I/O) は WARNING にして次のパッケージへ進む (readonly は黙って飛ばす)。
    """
    folded = 0
    for package_id in list(getattr(corpus, "loaded_ids", ())):
        if is_cancelled and is_cancelled():
            break
        package = corpus.get(package_id)
        index = getattr(package, "pseudo_queries", None) if package is not None else None
        if index is None or getattr(index, "uncommitted_count", None) is None:
            continue
        try:
            pending = index.uncommitted_count()
            if pending <= 0:
                continue
            if index.store.readonly:
                # readonly は起動時に理由付きで WARNING 済み。畳めないので毎サイクル言い直さない。
                continue
            await index.commit()
        except Exception as e:  # noqa: BLE001 — 1 パッケージの失敗で Step 5.9 (と Full) を止めない
            logger.warning(
                "Step 5.9: failed to commit carried-over pseudo queries for %s: %s: %s",
                package_id, type(e).__name__, e,
            )
            continue
        logger.info(
            "Step 5.9: committed %d pseudo-query event(s) left uncommitted by an earlier cycle for %s",
            pending, package_id,
        )
        folded += pending
    if folded:
        stats[CARRIED_OVER_KEY] = stats.get(CARRIED_OVER_KEY, 0) + folded
    return folded


def _report_rerank(
    stats: dict[str, int], reranker: Any, debug_logger: Any, *, cancelled: bool = False,
) -> None:
    """品質検査の件数をログと memory JSONL (``op="pseudo_query_rerank"``) に残す (採点を試みた回だけ)。"""
    attempted = stats.get(RERANK_SCORED_INPUT_KEY, 0) + stats[RERANK_UNSCORED_KEY]
    if reranker is None or attempted <= 0:
        return
    calibrated = getattr(reranker, "pseudo_query_calibration", None) is not None
    logger.info(
        "Step 5.9: scored %d pseudo-query question(s) with the reranker, dropped %d, "
        "left %d unscored (gate %s%s)",
        stats[RERANK_SCORED_KEY], stats[RERANK_DROPPED_KEY], stats[RERANK_UNSCORED_KEY],
        "calibrated" if calibrated else "uncalibrated", ", cancelled" if cancelled else "",
    )
    if debug_logger is not None:
        payload: dict[str, Any] = {**stats, "calibrated": calibrated}
        if cancelled:
            payload["cancelled"] = True
        debug_logger.log_memory_op("pseudo_query_rerank", payload)


async def generate_pseudo_queries(
    aux_client: Any,
    *,
    config: dict,
    cartridge_manager: "CartridgeManager | None",
    is_cancelled: Callable[[], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    reranker: Any = None,
    stats: dict[str, int] | None = None,
    debug_logger: Any = None,
) -> int:
    """Step 5.9 本体。問いを書いたチャンク数を返す。

    Args:
        aux_client: 補助タスククライアント (``None`` なら何もしない)。
        should_pause: ``True`` を返したらチャンク境界で打ち切る (チャット開始
            への協調 yield)。残りは次サイクルで拾う。
        reranker: 再順位のクライアント (``GenPillar.reranker``)。あれば 1 パッケージ分の
            生成の後に問いを採点し、較正済みの判定点が反対した問いを書かない
            (c_16 §7.2.1 の第 5 段階 B)。``None`` なら従来どおり全部書く。
        stats: 品質検査の件数を書き込む辞書 (:func:`empty_rerank_stats` の形)。
        debug_logger: 品質検査の件数を memory JSONL (``op="pseudo_query_rerank"``) に残す。
    """
    stats = stats if stats is not None else empty_rerank_stats()
    if reranker is not None:
        stats.setdefault(RERANK_SCORED_INPUT_KEY, 0)
    if cartridge_manager is None:
        return 0
    cfg = _pq_config(config)
    if not bool(cfg.get("enabled", True)):
        # 無効は機能の停止 — 検索も疑似クエリを使わない (f_01 §6.5) ので、畳んでも埋め込みが無駄になる。
        return 0
    # 生成の前に畳む — この回も取消されうるので、後ろに置くと取消が続く間は畳まれない。
    # 畳むのに要るのは埋め込みだけなので、補助タスク (base) が落ちていても畳む。
    carried_over = await _commit_carried_over(cartridge_manager.corpus, stats, is_cancelled)
    if aux_client is None:
        if carried_over:
            await _recalibrate(cartridge_manager)
        return 0
    from backend.free.rag.pseudo_query import PseudoQueryGenerator, PseudoQueryPreempted

    targets = collect_targets(
        cartridge_manager,
        max_per_cycle=int(cfg.get("max_per_cycle", 20)),
        backfill_per_cycle=int(cfg.get("backfill_per_cycle", 0)),
    )
    try:
        budget_seconds = float(cfg.get("budget_seconds", 180.0))
    except (TypeError, ValueError):
        budget_seconds = 180.0
    deadline = (time.monotonic() + budget_seconds) if budget_seconds > 0 else None
    total = sum(len(v) for v in targets.values())
    if total == 0:
        if carried_over:
            await _recalibrate(cartridge_manager)
        return 0
    logger.info("Step 5.9: generating pseudo queries for %d chunk(s)", total)

    generator = PseudoQueryGenerator(
        aux_client,
        questions_per_chunk=int(cfg.get("questions_per_chunk", 1)),
        max_chunk_chars=int(cfg.get("max_chunk_chars", 1200)),
    )
    corpus = cartridge_manager.corpus
    written = 0
    preempted = False
    try:
        for package_id, evidence_ids in targets.items():
            package = corpus.get(package_id)
            if package is None or package.pseudo_queries is None:
                continue
            index = package.pseudo_queries
            snapshot = package.store.snapshot
            if snapshot is None:
                continue
            added_here = 0
            # 再順位段があれば、生成を 1 パッケージ分終えてから採点する (base の decode と
            # リランカーを同じ GPU で交互に走らせない)。要素は (evidence_id, 採点に使う本文,
            # 問い, hinted_from)。再順位段が無ければ従来どおり 1 チャンクごとにすぐ書く。
            generated: list[tuple[str, str, list[str], int | None]] = []
            # 採点に渡した問いの数と、渡す前の入力件数 (取消の回に採点しなかった残りを数える)。
            screening: tuple[int, int] | None = None
            try:
                for evidence_id in evidence_ids:
                    done = written + added_here + len(generated)
                    if (is_cancelled and is_cancelled()) or (should_pause and should_pause()):
                        logger.info("Step 5.9: paused/cancelled after %d chunk(s)", done)
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        logger.info(
                            "Step 5.9: time budget (%.0fs) spent after %d chunk(s); "
                            "remaining targets retry in a later cycle", budget_seconds, done,
                        )
                        preempted = True
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
                            "remaining targets retry in a later cycle", done,
                        )
                        preempted = True
                        break
                    if not questions:
                        continue
                    hinted_from = getattr(generator, "hinted_from", None)
                    generated.append((
                        evidence_id, (text or "").strip()[: getattr(generator, "max_chunk_chars", 1200)],
                        list(questions),
                        hinted_from(hints) if hinted_from else None,
                    ))
                    if reranker is None:
                        added_here += _add_generated(index, package_id, generated)
                if reranker is not None and generated:
                    if preempted:
                        # 横取り・予算切れで畳んだ回は採点しない (チャットに GPU を譲る)。
                        stats[RERANK_UNSCORED_KEY] += _count_questions(generated)
                    elif (
                        getattr(reranker, "pseudo_query_calibration", None) is None
                        and not _decisions_recorded()
                    ):
                        # 較正前の採点は記録 (較正の材料) だけが目的。記録されない構成では省く。
                        stats[RERANK_UNSCORED_KEY] += _count_questions(generated)
                    else:
                        screening = (_count_questions(generated), stats.get(RERANK_SCORED_INPUT_KEY, 0))
                        await _screen_questions(
                            package_id, generated, reranker=reranker, stats=stats,
                            is_cancelled=is_cancelled, should_pause=should_pause,
                        )
            except BaseException:
                # Full の取消 (CancelledError) や想定外の例外でも、生成済みの問いは失わない —
                # 採点済みの判定はそのままに、残りは採点せずに書いてから上げ直す (put は同期)。
                # commit (埋め込み) は取消の応答を遅らせるので走らせない — 次のサイクルが
                # _commit_carried_over で畳む。
                if reranker is not None and generated:
                    if screening is None:
                        stats[RERANK_UNSCORED_KEY] += _count_questions(generated)
                    else:
                        to_score, input_before = screening
                        tried = stats.get(RERANK_SCORED_INPUT_KEY, 0) - input_before
                        stats[RERANK_UNSCORED_KEY] += max(0, to_score - tried)
                _add_generated(index, package_id, generated)
                raise
            added_here += _add_generated(index, package_id, generated)
            written += added_here
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
        if written or carried_over:
            await _recalibrate(cartridge_manager)
    except BaseException as exc:
        # 本流の commit・再較正の最中の取消も含め、採点の件数をその場で残す (結果辞書は
        # Full ごと捨てられる。件数はワーカーが次の Full へ持ち越す、c_16 §7.2.1)。
        try:
            _report_rerank(
                stats, reranker, debug_logger,
                cancelled=isinstance(exc, asyncio.CancelledError),
            )
        except Exception as e:  # noqa: BLE001 — 記録の失敗で取消・例外を握り替えない
            logger.warning("Step 5.9: failed to record pseudo-query scoring: %s", e)
        raise
    _report_rerank(stats, reranker, debug_logger)
    if written:
        logger.info("Step 5.9: wrote pseudo queries for %d chunk(s)", written)
    return written


async def _recalibrate(cartridge_manager: Any) -> None:
    """問いが増えたので corpus 側の棒を導き直す (署名一致なら no-op、f_01 §6.6)。"""
    recalibrate = getattr(cartridge_manager, "recalibrate_corpus", None)
    if recalibrate is None:
        return
    try:
        await recalibrate()
    except Exception as e:  # noqa: BLE001 — 較正の失敗で sleep-time を止めない
        logger.warning("Step 5.9: corpus recalibration failed: %s", e)


__all__ = [
    "CARRIED_OVER_KEY",
    "RERANK_DROPPED_KEY",
    "RERANK_SCORED_INPUT_KEY",
    "RERANK_SCORED_KEY",
    "RERANK_UNSCORED_KEY",
    "collect_targets",
    "empty_rerank_stats",
    "generate_pseudo_queries",
]
