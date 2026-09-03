"""`prompt` セクション — system プロンプトと文脈予算の配分 (f_03 §7.1 / c_02 §6.3)"""

from pydantic import BaseModel, ConfigDict, Field


class PromptConfig(BaseModel):
    """`prompt` セクション

    2026-09-03 の設計変更で導入。system プロンプトは規則台帳から決定論で
    レンダされ (f_03 §7.1.1)、文脈の予算は **1 つの関数**
    (``core.prompt_budget.resolve_budgets``) が配る。ここにある値はその関数の
    入力で、それぞれの「上限」を宣言する。実際の注入量ではなく上限の合計を
    動的ブロックの固定予約に使う (c_02 §6.3) — RAG のヒット件数が履歴の
    切り落とし位置を動かさないようにするため。
    """

    model_config = ConfigDict(extra="forbid")

    #: モードの context_size に対する静的 system の上限比。レンダラは超える
    #: とき priority の低い非 protected 規則から落とす (f_03 §7.1.2)。
    system_max_share: float = Field(default=0.30, ge=0.05, le=0.6)
    #: 静的接頭辞の末尾に置く few-shot コア集合の上限 (c_02 §6.3)。
    fewshot_core_max_tokens: int = Field(default=300, ge=0)
    #: query 依存で最後の user へ前置する few-shot の最大例数。
    fewshot_dynamic_max: int = Field(default=1, ge=0)
    #: query 依存 few-shot ブロックのトークン上限 (旧 ``_FEWSHOT_TOKEN_CAP``)。
    fewshot_dynamic_max_tokens: int = Field(default=600, ge=0)
    #: ``[関連する記憶]`` ブロックの上限 (MemoryInjector の tier 予算と揃える)。
    semmem_max_tokens: int = Field(default=800, ge=0)
    #: ``[参考情報]`` (RAG) ブロックの上限。
    rag_max_tokens: int = Field(default=200, ge=0)
    #: 押し出したターンの事実スレート (f_02 §1.2) の上限。0 で無効。
    fact_slate_max_tokens: int = Field(default=200, ge=0)
    #: 規則の削除 / verify-only 降格の根拠に要する最小観測ターン数 (f_03 §3.5.1)。
    rule_stats_min_turns: int = Field(default=200, ge=1)
    #: Level 1 の 1 ランで許す規則 delete 件数 (f_04 §4.5.2)。
    max_deletes_per_run: int = Field(default=1, ge=0)
