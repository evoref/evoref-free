"""制作ステージ (staged / longform) の共通ポート — ``ProductionHarness`` (Phase 3a)

create の「制作」を meta のタスクループの 1 ステージとして動かすための境界
(f_03_agent_engine.md §4.4)。``StagedCodeHarness``
(:mod:`backend.free.loop.staged.harness`) と ``LongFormHarness``
(:mod:`backend.free.generation.harness`) の 2 アダプタが実装し、composition
層 (``chat.py::make_production_stage``) が選択する。

pillar 境界: 本モジュールは ``backend/free/harness/`` (EvorefLoop) に属する。
``EditorArtifact`` は ``backend.free.agent.meta_cognitive_tasks`` 由来だが、
agent も EvorefLoop 配下なので top-level import で問題ない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from backend.free.agent.meta_cognitive_tasks import EditorArtifact

__all__ = [
    "ProductionEvent",
    "ProductionHarness",
    "ProductionRequest",
    "ProductionResult",
    "RunRecorder",
]

#: ``ProductionResult.exit_kind`` の取りうる値。
ExitKind = str  # "done" | "timeout" | "cancelled" | "error" | "blocked"


@dataclass(frozen=True)
class ProductionRequest:
    """制作ステージへの依頼 (f_03 §4.4 のポート表)。"""

    instruction: str
    brief: str = ""
    #: "code" | "text"
    content_type: str = "code"
    #: "file" | "editor" | "chat"
    output_target: str = "file"
    session_id: str = ""
    run_id: str = ""
    workspace_dir: str = ""
    #: ``time.monotonic()`` 系の締切。ステージ側はこれを上限にステージ内の
    #: 予算 (f_10 §3 の 3 層) を組む。
    deadline_monotonic: float = 0.0
    #: 問い返し (``needs_input``、Phase 3b) から再開する元 run_id。未指定
    #: (``None``) は新規 run。f_03 §4.4 / f_10 §7。
    resume_of: str | None = None


@dataclass(frozen=True)
class ProductionEvent:
    """制作ステージの進捗イベント。SSE step と ``events.jsonl`` の共通形 (f_10 §7)。"""

    kind: str
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionResult:
    """制作ステージの結果。"""

    artifacts: list[EditorArtifact] = field(default_factory=list)
    #: "done" | "timeout" | "cancelled" | "error" | "blocked"
    exit_kind: str = "done"
    question: str | None = None
    metrics: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    #: ``finish_reason=length`` で切れたまま採用された生成のラベル列と、そのときの
    #: max_tokens。meta が ``sse.output_truncated`` の開示に使う (deliberative と同じ)。
    truncated_steps: tuple[str, ...] = ()
    truncated_max_tokens: int | None = None


@runtime_checkable
class RunRecorder(Protocol):
    """制作ステージの run 記録ポート (f_10 §7 / f_08 §2.3、Phase 4)。

    ``StagedCodeHarness`` は同じ部品 (``WorkspaceManager`` / ``RunRecordStore`` /
    ``RunEventLog``、:mod:`backend.free.loop.staged`) を直接使う。
    ``LongFormHarness`` (EvorefGen) は pillar 境界のためこの Protocol 越しにしか
    触らない — composition 層 (``chat.py``) が
    :class:`~backend.free.loop.staged.run_recorder.StagedRunRecorder` を注入する。
    """

    #: workspace のルートディレクトリ (``ProductionResult.notes`` へ載せる用)。
    root: Path

    def start(
        self,
        *,
        session_id: str,
        request_id: str,
        mode: str,
        query: str,
        output_target: str,
        brief_tokens: int = 0,
        resume_of: str | None = None,
    ) -> None:
        """run 開始を記録する。"""
        ...

    def append_event(self, kind: str, payload: dict) -> None:
        """1 イベントを events.jsonl へ追記する (SSE へ出す前に呼ぶこと)。"""
        ...

    def write_src(self, path: str, content: str) -> None:
        """生成物の途中経過 / 確定本文を ``src/<path>`` へ書き戻す。"""
        ...

    def finish(self, exit_kind: str, *, template: str = "", tasks_failed: int = 0) -> None:
        """run 終端を記録する (``done`` / ``cancelled`` / ``error`` 等)。

        ``template`` は構成テンプレートで seed した場合の来歴鍵 (c_05 §0.6)。
        seed していない run は既定の空文字のまま。``tasks_failed`` は流れた上で
        欠けたものの件数 (f_10 §7。1 以上なら ``incomplete`` と導出される)。
        """
        ...


@runtime_checkable
class ProductionHarness(Protocol):
    """制作ステージの共通インタフェース。

    ``StagedCodeHarness`` / ``LongFormHarness`` が構造的に実装する
    (`typing.Protocol` — 明示的な継承は不要)。
    """

    name: str

    async def run(
        self,
        req: ProductionRequest,
        *,
        on_event,
        is_cancelled,
    ) -> ProductionResult:
        """制作を実行する。

        Args:
            req: 制作依頼。
            on_event: ``async def on_event(evt: ProductionEvent) -> None``。
            is_cancelled: ``def is_cancelled() -> bool``。呼出元のキャンセル
                フラグを読む (ポーリング)。
        """
        ...
