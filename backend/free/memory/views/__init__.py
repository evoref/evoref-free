"""

4 pillar アーキテクチャ における Fact View 層の公開インターフェース
本パッケージは :class:`~backend.free.memory.semantic.store.SemanticFactStore`
への pillar 別アクセスを制御する 4 つの View を提供する:

- :class:`MemFactView`     — EvorefMem 共有基盤 (全 FactType フルアクセス)
- :class:`LoopFactView`    — EvorefLoop の書込 (task / progress_marker /
  failure_pattern / artifact) + 他 pillar 読取
- :class:`LearnFactView`   — EvorefLearn の書込 (policy / fewshot) +
  他 pillar 読取
- :class:`HarnessFactView` — ハーネス層の読取専用 (policy / fewshot /
  failure_pattern のみ)

共通基盤は :mod:`backend.free.memory.views.base` に集約する。

## 導入範囲

**View 層** として、各 pillar (`loop/` / `learning/` /
`harness/`) からのアクセスを集約する。既存ストア
(``SemanticFactStore``) テストは未改修のまま通過する。
"""

from backend.free.memory.views.base import (
    FactViewBase,
    PillarReadAccessError,
    SubjectNamespaceError,
    WriteOwnershipError,
)
from backend.free.memory.views.harness import HarnessFactView
from backend.free.memory.views.learn import LearnFactView
from backend.free.memory.views.loop import LoopBootstrapResult, LoopFactView
from backend.free.memory.views.mem import MemFactView

__all__ = [
    "FactViewBase",
    "HarnessFactView",
    "LearnFactView",
    "LoopBootstrapResult",
    "LoopFactView",
    "MemFactView",
    "PillarReadAccessError",
    "SubjectNamespaceError",
    "WriteOwnershipError",
]
