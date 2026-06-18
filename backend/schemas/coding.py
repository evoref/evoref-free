"""コーディングモードのパイプライン設定スキーマ。

``coding:`` トップレベルセクション。``loop:`` (自律ループ周回の設定) とは別建てに
し、staged パイプラインのチューニングが自律ループ設定に干渉しないようにする。

- ``pipeline``: ``"staged"`` で仕様書→コード→テストの多段パイプライン、
  ``"longform"`` (既定) で従来の LongFormOrchestrator 1 リクエスト生成。
- ``staged_enabled``: pipeline 設定と独立したキルスイッチ。``false`` で staged を
  即時無効化し longform にフォールバックする。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CodingStagedConfig(BaseModel):
    """staged コーディングパイプラインの動作設定。"""

    model_config = ConfigDict(extra="forbid")

    test_stage_enabled: bool = Field(
        default=True,
        description="test 工程 (生成+pytest 実行+リペア) を有効化する。false で生成のみ",
    )
    flowchart_enabled: bool = Field(
        default=True,
        description="spec 工程で設計フローチャート (mermaid) を合成し spec.md/flowchart.md "
                    "に残して code/test 生成へ注入する。false で無効",
    )
    max_repair_rounds: int = Field(
        default=2, ge=0, le=5,
        description="test 工程の失敗リペア最大回数",
    )
    max_test_regen_rounds: int = Field(
        default=2, ge=0, le=5,
        description="生成テストが src の実 API (arity/属性) に整合しない時に "
                    "test のみ再生成する最大回数 (src は不変・準ゲート)",
    )
    entry_smoke_exec_enabled: bool = Field(
        default=True,
        description="エントリ有界実行スモーク (stdlib モック下でクラス構築+引数不要"
                    "公開メソッド呼び出し) を advisory で実施する。false で静的検査のみ",
    )
    entry_smoke_timeout_sec: float = Field(
        default=10.0, gt=0.0, le=120.0,
        description="エントリ有界実行スモークのサブプロセスタイムアウト",
    )
    spec_max_tokens: int = Field(
        default=1536, ge=256, le=8192,
        description="spec.md 生成の最大トークン",
    )
    spec_timeout_sec: float = Field(
        default=120.0, gt=0.0, le=1800.0,
        description="spec 工程の assist 生成タイムアウト (coding_spec_doc)",
    )
    test_timeout_sec: float = Field(
        default=120.0, gt=0.0, le=1800.0,
        description="生成テストの pytest サブプロセスのタイムアウト",
    )
    total_timeout_sec: float = Field(
        default=900.0, gt=0.0, le=7200.0,
        description="staged コーディング 1 リクエスト全体のウォールクロック上限",
    )
    max_iterations: int = Field(
        default=60, ge=1, le=1000,
        description="専用 LoopDriver の max_iterations (= 1 + 2*モジュール数 を見込む)",
    )
    cleanup_workspace: bool = Field(
        default=False,
        description="リクエスト完了時に temp ワークスペースを削除する。"
                    "false で継続ターン/デバッグのため保持",
    )


class CodingConfig(BaseModel):
    """``coding:`` トップレベル設定。"""

    model_config = ConfigDict(extra="forbid")

    pipeline: Literal["staged", "longform"] = Field(
        default="longform",
        description="コーディングモードの生成方式。既定は従来の longform",
    )
    staged_enabled: bool = Field(
        default=True,
        description="staged パイプラインのキルスイッチ (pipeline と独立)",
    )
    staged: CodingStagedConfig = Field(default_factory=CodingStagedConfig)
