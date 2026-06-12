"""候補 instruction の検索品質を実測する抽象 (EmbedEvalProtocol)

EmbedInstructionEvolver (EvorefLearn) が候補 instruction の検索品質を
offline で実測するための純抽象。Learn pillar は本 Protocol にのみ依存し、
埋め込み器・ベクトルストア (EvorefGen / EvorefMem) の具象を import しない。
実体 (``backend.free.rag.embed_instruction_eval.EmbedInstructionEval``) は
本 Protocol を明示継承せず構造的部分型 (duck typing) で満たし、wire 時に
注入する (依存方向 Gen→Learn の越境を作らないため)。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedEvalProtocol(Protocol):
    """候補 instruction で query を再埋め込み・再検索し top1 を実測する抽象。"""

    async def score_candidate(
        self, candidate: str, queries: list[str],
    ) -> float | None:
        """``candidate`` を検索 instruction として ``queries`` を再埋め込み・
        再検索し、各クエリの top1 検索スコア (cosine) の平均を返す。

        実測不能時 (ベクトルストア空 / 埋め込み器障害 / instruction 非対応) は
        ``None`` を返し、呼出側は従来のシグナルベース fitness へ degrade する。
        """
        ...
