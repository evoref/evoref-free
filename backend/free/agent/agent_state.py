"""エージェントの実行時状態（リマインダー・デバッグログ用）"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """エージェントの現在状態（リマインダー・デバッグログ用）"""

    # ループ状態
    agent_layer: str = "reactive"
    current_iteration: int = 0
    max_iterations: int = 3

    # ツール実行状態
    consecutive_tool_failures: int = 0
    last_tool_name: str = ""
    last_error: str = ""
    pending_tool: str | None = None
    pending_args: dict = field(default_factory=dict)
    #: このターンで「状態を変える操作を選んだが実行できなかった」か。
    #:
    #: ``ToolCallJudge`` はプロセス唯一の共有インスタンスで、判定中に ``await``
    #: を挟む。呼出側が後から属性を読むと、チャットが 2 本重なったときに他方の
    #: ``judge()`` がリセット済みで値が消える。``AgentState`` はターンごとに
    #: 生成されるので、判定結果をここへ写して以後の読み手に渡す
    #: (``ToolJudgement.action_blocked`` のコメント参照)。
    action_blocked: bool = False

    # 出力状態
    expected_format: str | None = None  # 'json' | 'unified_diff' | None
    last_output: str = ""

    # コンテキスト状態
    context_usage_pct: int = 0

    def on_tool_success(self, tool_name: str) -> None:
        """ツール実行成功時の状態更新"""
        self.consecutive_tool_failures = 0
        self.last_tool_name = tool_name

    def on_tool_failure(self, tool_name: str, error: str) -> None:
        """ツール実行失敗時の状態更新"""
        self.consecutive_tool_failures += 1
        self.last_tool_name = tool_name
        self.last_error = error
