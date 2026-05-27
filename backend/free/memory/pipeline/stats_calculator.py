"""メモリ統計計算ヘルパー

API ハンドラから抽出した純粋関数群。
FastAPI / app_state / ドメインオブジェクトのインスタンス生成には依存せず、
引数として受け取った値のみから統計値を算出する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol


class _NoteLike(Protocol):
    """統計計算に必要なノートの最小インターフェース"""

    embedding: Any
    evolution_pending: bool
    lightmem_score: float


def count_pending_embeddings(notes: Iterable[_NoteLike]) -> int:
    """埋め込み未計算のノート数を数える"""
    return sum(1 for n in notes if n.embedding is None)


def count_pending_evolution(notes: Iterable[_NoteLike]) -> int:
    """進化待ちフラグが立っているノート数を数える"""
    return sum(1 for n in notes if n.evolution_pending)


def average_lightmem_score(notes: Iterable[_NoteLike]) -> float:
    """LightMem スコアの平均を算出する (ノートが空なら 0.0)"""
    scores = [n.lightmem_score for n in notes]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def index_size_mb(index_path: Path | None) -> float:
    """インデックスファイルサイズを MB 単位で返す (存在しなければ 0.0)"""
    if index_path is None:
        return 0.0
    if not index_path.exists():
        return 0.0
    return index_path.stat().st_size / (1024 * 1024)


def count_unique_sources(metadata: Iterable[dict[str, Any]]) -> int:
    """メタデータ配列から source フィールドのユニーク数を返す"""
    return len(set(m.get("source", "") for m in metadata))


def compute_stm_stats(notes: Iterable[_NoteLike]) -> dict[str, Any]:
    """STM 統計 (pending_emb / pending_evo / avg_score) をまとめて計算する

    ノートのイテレータを一度しか走査しないよう list に展開する。
    返り値は API スキーマ (ShortTermMemoryStats) のフィールドに対応するキー名。
    """
    notes_list = list(notes)
    return {
        "pending_embeddings": count_pending_embeddings(notes_list),
        "pending_evolution": count_pending_evolution(notes_list),
        "avg_lightmem_score": round(average_lightmem_score(notes_list), 3),
    }


def compute_ltm_stats(
    chunks: int,
    index_path: Path | None,
    metadata: Iterable[dict[str, Any]] | None,
) -> dict[str, Any]:
    """LTM 統計 (chunks / index_size_mb / sources) をまとめて計算する"""
    size_mb = index_size_mb(index_path)
    sources = count_unique_sources(metadata) if metadata is not None else 0
    return {
        "chunks": chunks,
        "index_size_mb": round(size_mb, 3),
        "sources": sources,
    }
