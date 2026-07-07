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
        description="test 工程のうち advisory なユニットテスト生成+pytest 実行を"
                    "有効化する。false で生成のみ (smoke gate は smoke_gate_enabled で"
                    "別途制御)",
    )
    smoke_gate_enabled: bool = Field(
        default=True,
        description="test 工程のうち決定論的 import スモーク + spec/flowchart 注入"
                    "リペアループを有効化する。test_stage_enabled (advisory ユニット"
                    "テスト) とは独立。false でスキップ (終端の import スモークのみ残る)",
    )
    flowchart_enabled: bool = Field(
        default=True,
        description="spec 工程でフロー構造 (FlowSpec) を合成し、flowchart.md の "
                    "mermaid と spec.md の Processing flow 節を同一データから"
                    "決定論レンダリングして code/test 生成へ注入する。false で無効",
    )
    flow_part_synthesis_enabled: bool = Field(
        default=False,
        description="flow_spec_synthesis が (架橋修復込みで) 2 回とも検証不合格の"
                    "場合のエスカレーション。Component/モジュール単位で小規模"
                    "サブグラフを部分合成→決定論結合する。false なら従来通り"
                    "決定論フォールバック (fallback_flow) へ直接縮退する",
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
        default=6144, ge=256, le=8192,
        description="spec.md 生成の最大トークン。構造化 spec (## Module: / "
                    "### Component: の Signature/Attributes/Behavior/Constants "
                    "バレット) の詳細度を賄う (上限であり生成量目標ではない。"
                    "切断リトライ = 8192 での全文再生成の発生を抑える)",
    )
    spec_deepen_enabled: bool = Field(
        default=True,
        description="spec 工程でモジュール節を 1 節 1 assist 呼出で実装水準 "
                    "(メソッド毎挙動・属性・定数) まで深化させる。ガード棄却時は"
                    "原節維持の best-effort",
    )
    spec_conformance_enabled: bool = Field(
        default=True,
        description="spec 宣言契約 (Signature/メソッド/arity) と生成コードの"
                    "決定論照合を test 工程の smoke gate に合流させる。"
                    "false で観測記録のみに縮退 (誤検知時の運用弁)",
    )
    code_max_tokens: int = Field(
        default=4096, ge=512, le=16384,
        description="code 工程 (単一ファイルの直接生成) の最大トークン。"
                    "切断時のみ倍に広げて 1 回再生成する",
    )
    spec_timeout_sec: float = Field(
        default=600.0, gt=0.0, le=1800.0,
        description="spec 工程の assist 生成タイムアウト (coding_spec_doc / "
                    "coding_spec_deepen)。明示指定のため assist の反応的較正より"
                    "優先される。iGPU 実測 (7-13 t/s) で 6144 tok 級の生成を賄う "
                    "(timeout は description への全損フォールバックで救済が無い)",
    )
    test_timeout_sec: float = Field(
        default=120.0, gt=0.0, le=1800.0,
        description="生成テストの pytest サブプロセスのタイムアウト",
    )
    total_timeout_sec: float = Field(
        default=2400.0, gt=0.0, le=7200.0,
        description="staged コーディング 1 リクエスト全体のウォールクロック上限。"
                    "spec 詳細化 (深化パス + フロー詳細化) の増分を見込む",
    )
    part_generation_enabled: bool = Field(
        default=False,
        description="code 工程をコンポーネント部分生成→決定論結合で行う。spec に "
                    "'## Module:'/'### Component:' 構造がある場合のみ発動し、無ければ"
                    "単発生成へ自動フォールバックする",
    )
    part_max_tokens: int = Field(
        default=1536, ge=256, le=8192,
        description="部分 1 個の生成 max_tokens。切断時のみ倍に広げて 1 回再生成",
    )
    part_max_parts: int = Field(
        default=4, ge=2, le=8,
        description="1 ファイルの最大部分数。超過 component は spec 順の連続グループ"
                    "へ決定論的に併合する",
    )
    max_spec_revision_rounds: int = Field(
        default=1, ge=0, le=3,
        description="test 不合格時に spec 該当節を assist で点検・改訂して再生成する"
                    "サイクルのワークスペース全体での上限。0 で無効",
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
