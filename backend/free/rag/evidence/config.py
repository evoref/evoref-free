"""`EvidenceStore` へ渡す設定面の組み立て (c_16 §9)

``rag.quantization`` / ``memmap_threshold`` / ``cluster_index`` と
``memory.evidence.lexical`` / ``ranking`` / ``retention`` /
``know_half_life_days`` は config.yaml 上では別セクションだが、ストアは
**1 つのオブジェクト** から読む。3 ストア (episodic / semantic / corpus) が
同じ面を見るように、畳み込みはここ 1 箇所に置く。

corpus だけがこの関数を通り、episodic / semantic は ``ranking`` を落とした
別の組み立てを使っていたため、``memory.evidence.ranking.store_prior`` が
記憶側の順位式に一切届いていなかった (2026-09-08 監査)。畳み込みを分けない。
"""

from __future__ import annotations

from typing import Any

#: ``memory.evidence`` から ``rag`` 面へ重ねるキー (c_16 §9)。
EVIDENCE_OVERLAY_KEYS: tuple[str, ...] = (
    "lexical",
    "ranking",
    "retention",
    "know_half_life_days",
)


def merge_rag_evidence_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """``rag`` セクションに c_16 §9 の ``memory.evidence.*`` を重ねた dict を作る。

    :class:`~backend.free.rag.evidence.store.EvidenceStore` は
    ``rag.quantization`` / ``memmap_threshold`` / ``cluster_index`` と、
    ``memory.evidence.lexical`` / ``ranking`` / ``retention`` /
    ``know_half_life_days`` を **同じオブジェクトから** 読む (c_16 §9 の
    「``rag.*`` は現行キーを全ストアで使う」)。設定の 2 セクションをここで
    1 枚に畳んでから渡す。
    """
    merged: dict[str, Any] = dict(cfg.get("rag") or {})
    evidence = (cfg.get("memory") or {}).get("evidence") or {}
    for key in EVIDENCE_OVERLAY_KEYS:
        value = evidence.get(key)
        if value is not None:
            merged[key] = value
    return merged


__all__ = ["EVIDENCE_OVERLAY_KEYS", "merge_rag_evidence_config"]
