"""Step 8.8: 埋め込みが無い SemanticFact の遡及生成

``MemoryInjector`` の関連度ゲートは、埋め込みが無いファクトを「判定不能なので
通す」で素通りさせる (:mod:`backend.free.memory.pipeline.injector`)。ノートは
Step 1 が毎サイクル埋め込みを埋めるのでこの緩和が一時的なのに対し、**ファクト
には埋め込みを生成する経路が curator 系 (Step 8.5-8.7) にしか無かった**ため、
extractor 経由 (``personal_fact`` / ``preference``) と history 昇格経由
(``decision``) のファクトは永久に ``embedding=None`` のままだった。

実測 (2026-08-07 ライブ監査、``local/memory/semantic/global/facts.jsonl``):
21 件中 15 件 (decision 8 / personal_fact 5 / preference 2) が埋め込み無し。
結果として関連度ゲートが事実上無効化され、「12345 × 6789 はいくつですか？」の
プロンプトにまで、無関係な誕生日・猫の名前・過去の ``read_file`` の生出力が
毎ターン注入されていた。

抽出側 (:mod:`backend.free.memory.sleep.extraction`) は同期関数で embedder を
持たないため、sleep-time の非同期ステップとして遡及生成する。CLAUDE.md §6 #2
(SemMem 書込は sleep-time に閉じる) に整合する。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore

logger = get_logger("memory.sleep.fact_embedding")

#: 1 バッチあたりのファクト数。Step 1 (ノート埋め込み) と同じ粒度。
_BATCH_SIZE = 16

#: 埋め込みへ渡す前に切り詰める上限。embed サーバの context 超過で
#: バッチ全体が 400 で落ちるのを防ぐ (Step 1 と同じ理由)。
_MAX_CHARS = 2000

#: 1 サイクルで処理する上限。初回は溜まった全件が対象になるため、sleep-time が
#: 埋め込み待ちで長時間占有されないよう分割する (残りは次サイクル)。
_MAX_PER_CYCLE = 200


def _target_scopes(current_project_id: str | None) -> list[str]:
    """対象 scope のリストを返す (``global`` + current project)。"""
    scopes: list[str] = ["global"]
    if current_project_id:
        scopes.append(f"project:{current_project_id}")
    return scopes


async def backfill_fact_embeddings(
    store_provider: Callable[[str], "SemanticFactStore | None"],
    embedder,
    *,
    current_project_id: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> int:
    """``embedding`` が無いファクトへ遡及的に埋め込みを生成する。

    Args:
        store_provider: ``scope`` を受けてストア (または ``None``) を返す。
        embedder: ``embed(texts, is_query=False)`` を持つ埋め込みバックエンド。
            ``None`` (degraded) なら no-op。
        current_project_id: 現在のプロジェクト ID。未設定時は global のみ。
        is_cancelled: True を返したら途中で打ち切る。

    Returns:
        埋め込みを生成できたファクト数。
    """
    if embedder is None:
        logger.debug("Step 8.8: no embedder (degraded), skipping")
        return 0

    embedded = 0
    for scope in _target_scopes(current_project_id):
        if is_cancelled is not None and is_cancelled():
            break
        try:
            store = store_provider(scope)
        except Exception as exc:
            logger.warning("Step 8.8: failed to obtain store %s: %s", scope, exc)
            continue
        if store is None:
            continue

        targets = [
            f for f in store.all_facts(include_superseded=False)
            if f.embedding is None and (f.object or "").strip()
        ]
        if not targets:
            continue
        remaining = _MAX_PER_CYCLE - embedded
        if remaining <= 0:
            logger.info(
                "Step 8.8: per-cycle cap (%d) reached; remaining facts will be "
                "embedded on the next cycle", _MAX_PER_CYCLE,
            )
            break
        targets = targets[:remaining]

        for start in range(0, len(targets), _BATCH_SIZE):
            if is_cancelled is not None and is_cancelled():
                break
            batch = targets[start:start + _BATCH_SIZE]
            # 正規化済みの命題があればそちらを埋め込む (検索も命題で当たる)。
            texts = [(f.text or "")[:_MAX_CHARS] for f in batch]
            try:
                vecs = await embedder.embed(texts, is_query=False)
            except Exception as exc:
                # バッチ全滅を 1 件の失敗で招かないよう、次バッチへ進む。
                # 未処理分は embedding=None のまま次サイクルで再挑戦される。
                logger.warning(
                    "Step 8.8: batch embed failed (%d facts) in %s: %s",
                    len(batch), scope, exc,
                )
                continue
            if vecs is None or len(vecs) != len(batch):
                logger.warning(
                    "Step 8.8: embed returned %s vectors for %d facts in %s",
                    "no" if vecs is None else len(vecs), len(batch), scope,
                )
                continue
            for fact, vec in zip(batch, vecs):
                try:
                    # touch=False — 保守処理はアクセスではない。accessed_at を
                    # 更新すると recency スコアと GC 判定が一斉に歪む。
                    #
                    # flush_embedding=False — 1 件ごとに vectors.npy を全書き
                    # 出しすると 1 サイクル最大 200 件で 200 回の全書き出しに
                    # なる。バッチ末尾で 1 回だけ flush する。
                    store.update_fact(
                        fact.id, touch=False, flush_embedding=False,
                        embedding=np.asarray(vec, dtype=np.float32),
                    )
                    embedded += 1
                except Exception as exc:
                    logger.warning(
                        "Step 8.8: failed to persist embedding for %s: %s",
                        fact.id, exc,
                    )
            try:
                store.flush_embeddings()
            except Exception as exc:
                # 落ちても次サイクルで embedding=None の分が再挑戦される。
                logger.warning(
                    "Step 8.8: failed to flush embeddings in %s: %s", scope, exc,
                )

    if embedded:
        logger.info("Step 8.8: backfilled embeddings for %d fact(s)", embedded)
    return embedded


__all__ = ["backfill_fact_embeddings"]
