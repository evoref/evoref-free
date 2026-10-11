"""`/api/learning/growth` のレスポンス Pydantic スキーマ (Level 2 (成長)、docs/f_04 §6.8)。

フロント側の型は ``frontend/src/lib/free/api/growth.ts`` (``test_frontend_type_parity`` が
フィールドの片落ちを検出する)。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AutoAdoptModel(BaseModel):
    """自動採用の設定と、いま効いているか (効かなければ満たさなかった最初の条件)。"""

    configured: bool = False
    effective: bool = False
    reason: str = ""


class GrowthGateModel(BaseModel):
    """採用ゲートの結果 (§4.5 の一対比較ゲート。測っていない値は null)。"""

    name: str = ""
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    noise_floor: int | None = None
    sample_wins: int | None = None
    sample_losses: int | None = None
    sample_ties: int | None = None
    guards_passed: int | None = None
    guards_failed: int | None = None
    case_digest: str = ""


class GrowthProposalSummary(BaseModel):
    """未決の提案の要約 (候補の本文は詳細でだけ返す)。"""

    proposal_id: str
    asset_kind: str
    mode: str
    parent_version: str = ""
    current_version: str = ""
    #: 提案の後に現行の版が変わった (採用すると 409 ``stale_parent``)。
    stale: bool = False
    created_at: str = ""
    expires_at: str = ""
    measure: GrowthGateModel | None = None
    candidate_deltas: dict[str, float] = Field(default_factory=dict)
    regression_case_ids: list[str] = Field(default_factory=list)
    guard_ids: list[str] = Field(default_factory=list)


class GrowthProposalDetail(GrowthProposalSummary):
    """提案の詳細 (差分表示用に候補と現行の本文を持つ)。"""

    candidate_text: str = ""
    current_text: str = ""
    current_deltas: dict[str, float] = Field(default_factory=dict)


class GrowthLedgerRowModel(BaseModel):
    """成長の台帳の 1 行 (ID だけ。本文は持たない)。"""

    row_id: str
    written_at: str = ""
    proposal_id: str = ""
    op: str
    asset_kind: str = ""
    mode: str = ""
    parent_version: str = ""
    candidate_digest: str = ""
    decided_by: str | None = None
    gate: GrowthGateModel | None = None
    reason: str = ""
    adoption_id: str | None = None
    regression_case_ids: list[str] = Field(default_factory=list)
    experience_ids: list[str] = Field(default_factory=list)
    level1_session_id: str | None = None
    model_key: str = ""


class RegressionSummaryModel(BaseModel):
    """回帰台帳の件数 (状態別・役割別・源別)。"""

    active: int = 0
    guards: int = 0
    pressure: int = 0
    flaky: int = 0
    quarantined: int = 0
    retired: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    by_mode: dict[str, dict[str, int]] = Field(default_factory=dict)
    last_run_at: str | None = None


class GrowthAdoptionModel(BaseModel):
    """採用した版の記録 (guard 監視と巻き戻しの提案)。"""

    asset_kind: str
    mode: str
    decided_by: str = ""
    proposal_id: str = ""
    adopted_version: str = ""
    adopted_at: str | None = None
    monitoring: bool = False
    #: 利用者が採用した版で回帰が起きた (自動では巻き戻さない、f_04 §6.6)。
    rollback_suggested: bool = False
    suggestion_reason: str = ""


class GrowthResponse(BaseModel):
    """``GET /api/learning/growth``。"""

    auto_adopt: AutoAdoptModel = Field(default_factory=AutoAdoptModel)
    proposals: list[GrowthProposalSummary] = Field(default_factory=list)
    recent: list[GrowthLedgerRowModel] = Field(default_factory=list)
    regression: RegressionSummaryModel = Field(default_factory=RegressionSummaryModel)
    adoptions: list[GrowthAdoptionModel] = Field(default_factory=list)


class GrowthActionResponse(BaseModel):
    """採用・却下・巻き戻しの結果。"""

    asset_kind: str
    mode: str
    proposal_id: str | None = None
    #: 操作の後の現行版。
    version: str | None = None
    row_id: str | None = None
    adoption_id: str | None = None


class GrowthRollbackRequest(BaseModel):
    asset_kind: Literal["prompt", "generation_params"]
    mode: str


class RegressionRetireResponse(BaseModel):
    case_id: str
    status: str
    changed: bool = False


class GrowthSummaryModel(BaseModel):
    """``/api/learning/status`` に載せる Level 2 (成長) の要約。"""

    auto_adopt_configured: bool = False
    auto_adopt_effective: bool = False
    auto_adopt_reason: str = ""
    pending_proposals: int = 0
    guards: int = 0
    pressure: int = 0
    rollback_suggested: int = 0
