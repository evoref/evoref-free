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

**次元の違う埋め込みモデルへの切替** も本ステップが吸収する (2026-09-02 監査
M20)。以前は ``embedding is None`` だけが対象だったため、モデル切替後は
``search_by_embedding`` が ``query dim mismatch`` を投げ続け、URL / コマンド
リコールが恒久的に不発だった。ストアの次元が embedder と食い違っていたら
active model を切り替え (manifest swap 込み)、旧次元のベクトルを持つファクトも
対象に入れて、同じ per-cycle 上限の下で順に埋め直す。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.free.memory.sleep._curator_common import embed_kwargs_for_subject
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


def _embedder_dim(embedder: Any) -> int | None:
    """embedder の出力次元 (``dim()`` メソッドまたは ``dim`` 属性)。不明なら ``None``。"""
    d = getattr(embedder, "dim", None)
    if callable(d):
        try:
            d = d()
        except Exception:
            return None
    return int(d) if isinstance(d, int) and d > 0 else None


def _embedder_model_id(embedder: Any, dim: int) -> str:
    """embedder の ``model_name()`` から manifest 用 model_id を導く。

    ``model_name`` を持たない (fixture 等) 場合は次元だけで区別する。
    """
    from backend.free.memory.semantic.manifest import normalize_embedding_model_id

    name = getattr(embedder, "model_name", None)
    if callable(name):
        try:
            name = name()
        except Exception:
            name = None
    if isinstance(name, str) and name.strip():
        return normalize_embedding_model_id(name)
    return f"dim{dim}"


def _memory_dir_of(store: "SemanticFactStore") -> Path | None:
    """ストアの root から ``<memory_dir>`` (``semantic/manifest.json`` の親の親) を探す。"""
    from backend.free.memory.semantic.manifest import MANIFEST_FILENAME

    for parent in Path(store.root_dir).parents:
        if parent.name == "semantic" and (parent / MANIFEST_FILENAME).exists():
            return parent.parent
    return None


def _is_stale_dim(fact, dim: int) -> bool:
    """``fact.embedding`` が現行 embedder と別次元か。"""
    emb = fact.embedding
    return emb is not None and int(np.asarray(emb).reshape(-1).shape[0]) != dim


def _switch_store_to_current_model(
    store: "SemanticFactStore", *, model_id: str, dim: int, scope: str,
    swapped_manifests: set[Path],
) -> None:
    """ストアの active 埋め込みモデルを現行 embedder のものへ切り替える。

    manifest は memory_dir 単位で 1 回だけ swap する (global / project で共有)。
    """
    from backend.free.memory.semantic.embedding_store import swap_active_model_id

    store.switch_embedding_model(model_id)
    memory_dir = _memory_dir_of(store)
    if memory_dir is not None and memory_dir not in swapped_manifests:
        try:
            swap_active_model_id(memory_dir, model_id, new_dim=dim)
            swapped_manifests.add(memory_dir)
        except Exception as exc:
            logger.warning(
                "Step 8.8: manifest swap failed at %s: %s", memory_dir, exc,
            )
    logger.warning(
        "Step 8.8: embedding dim changed in %s; switched active model to %s "
        "(dim=%d) and will re-embed stored facts incrementally", scope, model_id, dim,
    )


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

    dim = _embedder_dim(embedder)
    swapped_manifests: set[Path] = set()
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

        # ストアの次元が embedder と食い違っていたら、書込先を現行モデルの
        # 空ストアへ切り替える (旧次元のまま upsert すると dim mismatch)。
        store_dim = getattr(store, "embedding_dim", None)
        if dim is not None and store_dim is not None and store_dim != dim:
            _switch_store_to_current_model(
                store, model_id=_embedder_model_id(embedder, dim), dim=dim,
                scope=scope, swapped_manifests=swapped_manifests,
            )
        targets = [
            f for f in store.all_facts(include_superseded=False)
            if (f.object or "").strip()
            and (f.embedding is None or (dim is not None and _is_stale_dim(f, dim)))
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

        for batch in _batches(targets):
            if is_cancelled is not None and is_cancelled():
                break
            # 正規化済みの命題があればそちらを埋め込む (検索も命題で当たる)。
            texts = [(f.text or "")[:_MAX_CHARS] for f in batch]
            # 内部索引 (executable command 等) は query 側で埋める —
            # 側の定義は ``_curator_common.INDEX_EMBED_IS_QUERY`` が SSOT。
            kwargs = embed_kwargs_for_subject(batch[0].subject) or {"is_query": False}
            try:
                vecs = await embedder.embed(texts, **kwargs)
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


def _batches(targets: list) -> list[list]:
    """``_BATCH_SIZE`` 件ずつ、かつ **埋め込み側 (kwargs) が同じもの同士** で束ねる。"""
    groups: dict[tuple, list] = {}
    for f in targets:
        kw = embed_kwargs_for_subject(f.subject)
        key = tuple(sorted(kw.items())) if kw else ()
        groups.setdefault(key, []).append(f)
    out: list[list] = []
    for members in groups.values():
        for start in range(0, len(members), _BATCH_SIZE):
            out.append(members[start:start + _BATCH_SIZE])
    return out


__all__ = ["backfill_fact_embeddings"]
