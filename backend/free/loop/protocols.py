"""

EvorefLoop (loop / harness / agent pillar) が own する fact type
(task / progress_marker / failure_pattern / artifact) の書込 API を
Protocol として固定する。``LoopDriver`` か独立クラスが
本 Protocol を実装し、現在の module 関数 (``write_progress_marker`` /
``write_failure_note`` / ``write_artifact``) を束ねる想定。

他 pillar (特に harness / agent) は 本 Protocol 経由に
切り替える。

設計原則 (CLAUDE.md §3 / `docs/f_02_memory_system.md` §6):
- 最小 API 原則: Loop owner facts の書込 3 メソッドのみ
- Mem pillar の ``SemanticFact`` を返すが、書込先ストアは
  実装側が ``store_provider`` 等で保持する想定
- Protocol ファイルは Mem と同 pillar 内の型のみ参照する
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from backend.free.memory.types import SemanticFact


@runtime_checkable
class LoopWriteAPIProtocol(Protocol):
    """EvorefLoop owner facts の書込 API。

    task / progress_marker / failure_pattern / artifact の owner は
    EvorefLoop であるため、他 pillar はこれら fact を **直接 add/update
    してはならない**。必要な場合は本 Protocol 経由で書込 API を呼ぶ。

    最小 API:
    - ``write_progress_marker``: 自律ループのタスク進捗マーカー永続化
    - ``write_failure_pattern``: 品質ゲート失敗時の failure_pattern 永続化
    - ``write_artifact``: ラルフループの成果物 (ファイル編集) トレース永続化

    シグネチャは 実装される具象クラスの型に合わせる
    ``**kwargs`` 余地を残し、詳細パラメータ (``trace_id`` / ``iteration`` /
    ``gate_passed`` 等) は実装側で消費する。
    """

    def write_progress_marker(
        self,
        *,
        project_id: str,
        task_id: str,
        title: str = "",
        status: str = "done",
        **kwargs: Any,
    ) -> SemanticFact:
        """``progress_marker`` ファクトを idempotent に書き込む。"""
        ...

    def write_failure_pattern(
        self,
        *,
        project_id: str,
        gate_result: Any,
        mdp_steps: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> SemanticFact:
        """``failure_pattern`` ファクトを即時書き込みする。

        ``gate_result`` は EvorefLoop 内の ``QualityGate`` 出力を想定。
        Protocol としては ``Any`` で緩めに宣言し、実装側で具体型
        (``backend.free.loop.quality_gate.GateResult``) を使う。
        """
        ...

    def write_artifact(
        self,
        *,
        project_id: str,
        task_id: str,
        entry: Any,
        gate_passed: bool,
        **kwargs: Any,
    ) -> SemanticFact | None:
        """``artifact`` ファクトを冪等に書き込む (重複は ``None``)。

        ``entry`` は EvorefLoop 内の ``ArtifactEntry`` を想定。
        """
        ...


__all__ = ["LoopWriteAPIProtocol"]
