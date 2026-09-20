"""クリエイトモードのパイプライン設定スキーマ。

``create:`` トップレベルセクション。``loop:`` (自律ループ周回の設定) とは別建てに
し、staged パイプラインのチューニングが自律ループ設定に干渉しないようにする。

- ``pipeline``: ``"staged"`` で仕様書→コード→テストの多段パイプライン、
  ``"longform"`` (既定) で従来の LongFormOrchestrator 1 リクエスト生成。
- ``staged_enabled``: pipeline 設定と独立したキルスイッチ。``false`` で staged を
  即時無効化し longform にフォールバックする。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: verify の承認済み argv で許す置換プレースホルダ (c_16 §4.5.4)。要素単位の
#: 一致だけを許し、文字列連結 (``--out={file}``) は受けない — manifest 側の
#: ``verify.args`` 検証 (``backend/free/rag/corpus/language.py`` の
#: ``VERIFY_ARG_PLACEHOLDERS``) と同じ語彙を、config 側でも複製する
#: (schemas は corpus [EvorefGen pillar] を import しない層のため)。
_VERIFY_ARG_PLACEHOLDERS = frozenset({"{file}", "{workspace}"})


def _is_bare_command_name(name: str) -> bool:
    """パス区切り・``:``・``.``・空白を含まない素の名前か (manifest の
    ``verify.executable`` 検証と同じ規則の複製)。"""
    return bool(name) and name == name.strip() and not any(c in name for c in "/\\:. ")


def _validate_verify_command(argv: object) -> list[str]:
    """``create.staged.verify.commands`` の 1 件を検証する。

    argv[0] は manifest の ``executable`` と同じ「素の名前」規則、残りの
    要素は ``{file}``/``{workspace}`` とちょうど一致するか中括弧を含まない
    文字列のどちらか。違反は起動時に ``ValueError`` (pydantic が
    ``ValidationError`` へ包む)。
    """
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError(
            f"create.staged.verify.commands entry must be a non-empty list of "
            f"strings: {argv!r}",
        )
    executable, *args = argv
    if not _is_bare_command_name(executable):
        raise ValueError(
            "create.staged.verify.commands entry's first element (executable) "
            f"must be a bare name without path separators, ':', '.', or "
            f"whitespace: {executable!r}",
        )
    for a in args:
        if a in _VERIFY_ARG_PLACEHOLDERS:
            continue
        if "{" in a or "}" in a:
            raise ValueError(
                f"create.staged.verify.commands argument {a!r} must be exactly "
                "{file}/{workspace} or contain no braces (no string concatenation)",
            )
    return [executable, *args]

#: c_16 の ``_REMOVED_MEMORY_KEYS_REJECTED`` と同じ作法 (CLAUDE.md §7):
#: 機能ごと消えたキーは黙って捨てず、理由付きで起動時に拒否する。
#: ``create.dispatch`` は 3a-2 (2026-09-19) で撤去 — create のディスパッチは
#: meta の production_stage 経路の 1 本になった (f_03_agent_engine.md §4.4)。
_REMOVED_CREATE_KEYS_REJECTED: dict[str, str] = {
    "dispatch": (
        "create dispatch is unified onto the meta production_stage path; "
        "the legacy \"legacy\" dispatch (_dispatch_long_form / "
        "stream_staged_create) was removed"
    ),
}


class StagedVerifyConfig(BaseModel):
    """言語パックの検証コマンド (c_16 §4.5.4)。既定 OFF。

    パッケージ由来の宣言でプロセスを起動する唯一の経路 — ``enabled`` を
    明示的に ``true`` にした PC だけが実行し、``commands`` に無い argv は
    パックが宣言していても無視される。**承認単位は実行ファイル名ではなく
    argv 全体** — 実行ファイル名だけを allow-list にすると、PC の持ち主が
    ``python`` や ``node`` を許可した瞬間、パックはその処理系の ``-c`` /
    ``-e`` 経由で任意コードを実行する verify を宣言できてしまう
    (2026-09-20 レビューで指摘)。パックは承認済みのコマンド (argv) を
    選ぶだけで、引数も増やせない。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="言語パックの verify コマンドを実行する。false (既定) では"
                    "install は通るが verify の宣言を一切読まない",
    )
    commands: list[list[str]] = Field(
        default_factory=list,
        description="実行を許す argv (コマンド全体) の allow-list "
                    "(例: [[\"node\", \"--check\", \"{file}\"], [\"cargo\", \"check\"]])。"
                    "パック側の宣言は、承認済み argv のどれかと完全一致した"
                    "ときだけ実行される。PC の持ち主だけが広げられる",
    )

    @field_validator("commands")
    @classmethod
    def _validate_commands(cls, value: list[list[str]]) -> list[list[str]]:
        return [_validate_verify_command(argv) for argv in value]

    timeout_sec: float = Field(
        default=60.0, gt=0.0, le=600.0,
        description="宣言側の timeout_sec の上限。実際のタイムアウトは "
                    "min(宣言値, これ)",
    )


class CreateStagedConfig(BaseModel):
    """staged クリエイトパイプラインの動作設定。"""

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
        description="spec 工程でモジュール節を 1 節 1 呼出で実装水準 "
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
        description="spec 工程の生成タイムアウト (create_spec_doc / "
                    "create_spec_deepen)。明示指定のため反応的較正より"
                    "優先される。iGPU 実測 (7-13 t/s) で 6144 tok 級の生成を賄う "
                    "(timeout は description への全損フォールバックで救済が無い)",
    )
    test_timeout_sec: float = Field(
        default=120.0, gt=0.0, le=1800.0,
        description="生成テストの pytest サブプロセスのタイムアウト",
    )
    total_timeout_sec: float = Field(
        default=2400.0, gt=0.0, le=7200.0,
        description="staged クリエイト 1 リクエスト全体のウォールクロック上限。"
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
        description="test 不合格時に spec 該当節を LLM で点検・改訂して再生成する"
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
    verify: StagedVerifyConfig = Field(default_factory=StagedVerifyConfig)


class BriefConfig(BaseModel):
    """ProductionBrief (f_08 §2.2) の予算設定。

    create モードのターン入口で 1 回だけ決定論で組む不変ブリーフの、総予算と
    節ごとの上限。ターン中の全 LLM 呼出 (計画/spec/深化/フロー/code/test/
    修復/見直し/unit/レビュー) のプロンプト先頭に同じ bytes を置く。
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(
        default=2000, ge=0, le=8192,
        description="ブリーフ全体の総予算。超過時は code_map → attachments → "
                    "references → prior_work → memory → facts の順で節ごと削る",
    )
    facts: int = Field(
        default=200, ge=0, le=4096,
        description="Facts 節 (fact slate、chat の _append_fact_slate と同じ材料) の上限",
    )
    memory: int = Field(
        default=600, ge=0, le=4096,
        description="Memory 節 (SemMem 注入ブロック) の上限",
    )
    prior_work: int = Field(
        default=400, ge=0, le=4096,
        description="Prior work 節 (直前の長文成果物) の上限",
    )
    references: int = Field(
        default=400, ge=0, le=4096,
        description="References 節 (RAG 採用チャンク) の上限",
    )
    attachments: int = Field(
        default=400, ge=0, le=4096,
        description="Attachments 節 (添付ファイルブロック) の上限",
    )
    code_map: int = Field(
        default=400, ge=0, le=4096,
        description="Code map 節 (ProjectMap 近傍) の上限",
    )


class CreateConfig(BaseModel):
    """``create:`` トップレベル設定。"""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_removed_keys(cls, data):
        """撤去済みキー (:data:`_REMOVED_CREATE_KEYS_REJECTED`) を理由付きで拒否する。

        ``MemoryConfig.reject_removed_keys`` (backend/schemas/memory.py) と同じ作法。
        """
        if isinstance(data, dict):
            for key, reason in _REMOVED_CREATE_KEYS_REJECTED.items():
                if key in data:
                    raise ValueError(
                        f"create.{key} was removed: {reason}. "
                        "Remove the line from config.yaml "
                        "(see docs/f_03_agent_engine.md §4.4).",
                    )
        return data

    pipeline: Literal["staged", "longform"] = Field(
        default="longform",
        description="クリエイトモードの生成方式。既定は従来の longform",
    )
    staged_enabled: bool = Field(
        default=True,
        description="staged パイプラインのキルスイッチ (pipeline と独立)",
    )
    turn_timeout_sec: float = Field(
        default=3600.0, gt=0.0,
        description="create モードの 1 ターン (1 リクエスト) 全体のウォール"
                    "クロック上限。予算の 3 層 (f_10 §3) の最終防衛線 — "
                    "``create.staged.total_timeout_sec`` はこの値 − 300 秒を"
                    "上限にクランプされる (超過設定は起動時 WARNING)",
    )
    runs_keep: int = Field(
        default=20, ge=1,
        description="staged クリエイトの run レコード (f_10 §7) の保持件数",
    )
    staged: CreateStagedConfig = Field(default_factory=CreateStagedConfig)
    brief: BriefConfig = Field(default_factory=BriefConfig)
