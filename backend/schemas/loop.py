"""自律ループ driver 関連スキーマ"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoopQualityGatesConfig(BaseModel):
    """`loop.quality_gates` セクション

    設定値の存在のみ受け入れ、実際のゲート実行は別層で接続する。
    """

    model_config = ConfigDict(extra="forbid")

    pytest: bool = True
    typecheck: bool = True
    lint: bool = False


class LoopSandboxConfig(BaseModel):
    """`loop.sandbox` セクション

    ``ActionRunner`` の安全性境界を定義する。

    - ``allowed_write_roots`` : ``edit_file`` Action で書込可能なディレクトリの
      allowlist。``repo_root`` を基準にした相対パスを受け付ける。
      パストラバーサル (``Path.resolve()`` 後に allowlist 外) は拒否する。
    - ``allowed_commands`` : ``run_command`` Action で実行可能なコマンド名
      (引数列の先頭) の allowlist。``"*"`` のみを含む場合は全許可
      (検証用; 本番環境では明示推奨)。
    - ``command_timeout_sec`` : ``run_command`` の 1 回あたりタイムアウト秒。
    """

    model_config = ConfigDict(extra="forbid")

    allowed_write_roots: list[str] = Field(
        default_factory=lambda: ["local/loop_sandbox"],
    )
    allowed_commands: list[str] = Field(
        default_factory=lambda: ["python", "node", "npm", "git"],
    )
    command_timeout_sec: float = Field(default=120.0, gt=0.0, le=1800.0)


class LoopConfig(BaseModel):
    """自律ループ driver 設定

    `task` 型 SemanticFact を駆動源とする自律実行ループの動作を制御する。

    で ``executor`` / ``tick_interval_sec`` / ``max_actions_per_task`` /
    ``max_wall_time_sec`` / ``max_consecutive_failures`` を追加、
    で ``sandbox`` を追加した
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    driver: Literal["semmem_task"] = "semmem_task"
    executor: Literal["ralph", "noop"] = "ralph"
    max_iterations: int = Field(default=50, ge=1)
    max_actions_per_task: int = Field(default=10, ge=1)
    max_wall_time_sec: float = Field(default=1800.0, gt=0.0)
    max_consecutive_failures: int = Field(default=3, ge=1)
    tick_interval_sec: float = Field(default=0.0, ge=0.0, le=60.0)
    context_reset_threshold_tokens: int = Field(default=24000, ge=1)
    sleep_time_every_n: int = Field(default=5, ge=1)
    on_gate_fail: Literal["retry", "skip", "abort"] = "retry"
    retry_limit_per_task: int = Field(default=2, ge=0)
    # SSE 進捗ストリーム用イベントバスのキュー上限 (subscriber 単位)
    event_bus_max_queue: int = Field(default=128, ge=1, le=4096)
    quality_gates: LoopQualityGatesConfig = Field(
        default_factory=LoopQualityGatesConfig,
    )
    sandbox: LoopSandboxConfig = Field(default_factory=LoopSandboxConfig)
