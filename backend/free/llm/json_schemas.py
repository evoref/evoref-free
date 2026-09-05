"""補助タスク JSON 応答 purpose 別スキーマ定義

llama-server `/v1/chat/completions` の OAI 互換 ``response_format`` を用いた
制約サンプリング用に、purpose ごとの Pydantic v2 モデルを集約する。

設計方針:
- すべての schema は ``extra="forbid"`` 相当 (``additionalProperties: false``)
  を強制し、llama.cpp のサンプラ側で文法外トークンの確率を 0 化させる。
- Pydantic ``model_json_schema()`` の出力を OpenAI 互換
  ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ...,
  "strict": true}}`` 形式に変換するヘルパ ``make_response_format`` を提供。
- purpose 文字列から自動解決する ``PURPOSE_SCHEMAS`` を提供。``cogwriter``
  の plan のように content_type で分岐する purpose は呼出側で
  ``response_schema=...`` を明示する想定。

ントポリシー) の制約を維持する。``response_format`` は ``/v1/chat/completions``
の OAI 互換パラメータであり、``/v1/messages`` (Anthropic 互換) は不採用。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """``additionalProperties: false`` を強制する基底。"""

    model_config = ConfigDict(extra="forbid")


# ── 取得直後 content gate の関連性判定 (retrieval_chunk_gate) ──

class ChunkGateRelevance(_StrictModel):
    """`backend/free/rag/chunk_content_gate.py` の marginal band 関連性判定。

    クエリに関連する候補チャンクの 0 始まりインデックスのみを返す。OpenAI
    strict / llama.cpp 制約サンプリングは top-level に object が必要なため
    ``relevant_indices`` キーで配列をラップする (``list_key`` で裸配列も救済)。
    """

    relevant_indices: list[int]


# ── 失敗パターン分析 (critique_synthesis) ──

class CritiqueSynthesisResult(_StrictModel):
    """`backend/free/learning/critique_synthesizer.py` の失敗クラスタ分析。"""

    failure_patterns: list[str]
    improvement_hints: list[str]
    summary: str


# ── 長文生成プラン (long_form_planning) ──

class CodeUnitPlan(_StrictModel):
    """CogWriter コード生成 1 ユニット (function / class / config 等)。"""

    kind: Literal[
        "imports", "types", "class", "function", "config", "test"
    ]
    name: str
    file_path: str
    spec: str
    depends_on: list[str] = Field(default_factory=list)
    estimated_tokens: int = 500


class SectionUnitPlan(_StrictModel):
    """CogWriter テキスト生成 1 セクション。"""

    heading: str
    key_points: list[str] = Field(default_factory=list)
    estimated_tokens: int = 500
    # SPLIT モード (機能ごと個別ファイル出力) で LLM が埋める出力ファイル名
    # 候補 (拡張子なし、英数字 + アンダースコア)。CONTINUE/EXPAND モードでは
    # ``None``。SPLIT モードでも欠落時はオーケストレータ側で連番フォールバック。
    file_name: str | None = None


class CodePlan(_StrictModel):
    """コード生成プラン全体 (`content_type=code`)。"""

    title: str = ""
    target_length: int = 0
    global_context: str = ""
    constraints: list[str] = Field(default_factory=list)
    units: list[CodeUnitPlan]


class TextPlan(_StrictModel):
    """テキスト生成プラン全体 (`content_type=text`)。"""

    title: str = ""
    target_length: int = 0
    global_context: str = ""
    constraints: list[str] = Field(default_factory=list)
    units: list[SectionUnitPlan] = Field(default_factory=list)
    # ユーザー指示に具体的な主題が無く、関連メモリの話題を主題に流用しないと
    # 計画できない場合に True。True のとき units は空でよく、
    # clarification_question にユーザーへの確認文を書く
    # (2026-07-22 ライブ検証で判明した長文トピック混入バグの対策)。
    needs_clarification: bool = False
    clarification_question: str = ""


# ── コード設計仕様 (code_spec_synthesis) ──
#
# コード生成の「事前準備」段階で合成する、ファイル横断の共有契約。
# CodePlan を生成する前にこれを固めることで、各ユニット (小ブロック) が
# 同一のモジュール名・データモデル・公開シグネチャ・エントリポイント・
# 通信プロトコルに準拠する。f_08 §3 参照。

class CodeSpecField(_StrictModel):
    """共有データモデルの 1 フィールド。"""

    name: str
    type: str


class CodeSpecModule(_StrictModel):
    """正準なモジュール (ファイル) 構成の 1 エントリ。"""

    path: str
    purpose: str


class CodeSpecDataModel(_StrictModel):
    """ファイル横断で共有するデータモデル (dataclass / class / enum 等)。"""

    name: str
    module: str
    kind: Literal[
        "dataclass", "class", "enum", "typeddict", "pydantic", "namedtuple", "other"
    ] = "dataclass"
    fields: list[CodeSpecField] = Field(default_factory=list)


class CodeSpecInterface(_StrictModel):
    """公開する関数 / メソッドのシグネチャ契約。"""

    module: str
    signature: str


class CodeSpecEntryPoint(_StrictModel):
    """プログラムの起点。空文字列なら起点なし (ライブラリ等)。"""

    module: str = ""
    invocation: str = ""


class CodeSpec(_StrictModel):
    """コード生成の共有設計仕様 (contract)。

    `backend/free/generation/strategy_cogwriter.py` の create_plan が
    purpose="code_spec_synthesis" で合成し、CodePlan の生成と各ユニット
    プロンプトに注入する。SPEC.md として成果物にも出力する。
    """

    title: str = ""
    summary: str = ""
    modules: list[CodeSpecModule] = Field(default_factory=list)
    data_models: list[CodeSpecDataModel] = Field(default_factory=list)
    interfaces: list[CodeSpecInterface] = Field(default_factory=list)
    entry_point: CodeSpecEntryPoint = Field(default_factory=CodeSpecEntryPoint)
    protocol: str = ""
    constraints: list[str] = Field(default_factory=list)


# ── フローチャート生成 (flowchart_synthesis) ──

class FlowchartSpec(_StrictModel):
    """CodeSpec から導く mermaid フローチャート (config で任意 ON)。"""

    mermaid: str = ""


# ── staged クリエイトのタスクグラフ合成 (create_task_graph) ──

class CreateModuleUnit(_StrictModel):
    """staged クリエイトが生成する 1 モジュール (= 1 ファイル) の粗計画。

    `backend/free/loop/staged/synthesizer.py` が purpose="create_task_graph"
    で取得し、spec/code/test の task ファクト三層へ決定的に展開する。
    モジュール間の ``depends_on`` は code パス内の import 配線への参考情報で
    あり、task 依存グラフには落とさない (循環回避のため)。
    ``key_components`` は spec 工程の ``### Component:`` アンカー候補として
    プロンプトへ供給される (default 付きで旧応答とも後方互換)。
    """

    file_path: str = ""
    purpose: str = ""
    key_components: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class CreateTaskGraph(_StrictModel):
    """staged クリエイトの粗計画 (全体設計 + モジュール分割)。

    LLM には粗計画のみを返させ、三層展開・依存配線・slug 衝突回避は呼出側
    (synthesizer) が Python で決定的に行う。
    """

    summary: str = ""
    modules: list[CreateModuleUnit] = Field(default_factory=list)


# ── staged フロー構造合成 (flow_spec_synthesis) ──

class FlowSpecEdge(_StrictModel):
    """フローの 1 遷移。``condition`` は分岐ラベル (無条件遷移は空文字)。"""

    to: str = ""
    condition: str = ""


class FlowSpecStep(_StrictModel):
    """フローの 1 ステップ。``module`` は正準ファイル一覧のパス (start/end のみ空可)。"""

    id: str = ""
    module: str = ""
    label: str = ""
    kind: Literal["start", "process", "decision", "error", "end"] = "process"
    next: list[FlowSpecEdge] = Field(default_factory=list)


class FlowSpec(_StrictModel):
    """staged spec 工程のフロー構造 (`executor._synthesize_flow_steps`)。

    spec.md の ``## Processing flow`` 節と flowchart.md の mermaid を
    同一データから決定論レンダリングするための単一情報源。検証・描画は
    `backend/free/loop/staged/flow_render.py` (LLM 不使用) が担う。
    """

    steps: list[FlowSpecStep] = Field(default_factory=list)


# ── staged test 工程の spec 見直し判定 (spec_revision_judge) ──

class SpecRevisionJudgement(_StrictModel):
    """test 不合格時の spec 該当節見直し判定 (executor._spec_revision_cycle)。

    「spec 節自体の欠陥 (矛盾する型/シグネチャ・欠落した振る舞い・曖昧さ) か、
    コード/テスト側のミスか」を判定させ、欠陥なら ``## Module:`` 見出しから
    始まる改訂節全文を ``revised_section`` に返させる。改訂が全体の
    ``## Entry point`` 節と矛盾する場合のみ、その修正版全文を
    ``revised_entry_point`` に返させる (通常は空)。
    """

    spec_ok: bool = True
    reason: str = ""
    revised_section: str = ""
    revised_entry_point: str = ""


# ── 長文生成レビュー (long_form_*_review) ──

class ReviewIssueItem(_StrictModel):
    """レビュー指摘 1 件。"""

    unit_idx: int
    issue: str
    fix: str


class ReviewIssues(_StrictModel):
    """レビュー指摘リスト (code/text 共通)。"""

    issues: list[ReviewIssueItem]


# ── meta-cognitive 計画 (meta_cognitive_plan) ──

class MetaCognitivePlan(_StrictModel):
    """`backend/free/agent/meta_cognitive.py` のタスク計画。

    旧仕様は ``["task1", "task2"]`` の裸 JSON 配列だったが、OpenAI strict
    structured outputs / llama.cpp 制約サンプリングは top-level に object
    が必要なため、``tasks`` キーで配列をラップする
    """

    tasks: list[str]


# ── カートリッジ eval.json 生成 (cartridge_eval_generation) ──

class CartridgeEvalQAItem(_StrictModel):
    """カートリッジ eval.json の QA ペア 1 件

    `backend/pro/cartridge_creator.py` がドキュメント理解度評価用の QA ペアを
    補助タスクに生成させる際に使う。``tags`` は分類用の任意ラベルだが、
    OpenAI strict structured outputs では全プロパティ required のため、
    LLM 側に必ず空配列以上を返させる。
    """

    question: str
    ground_truth: str
    tags: list[str] = Field(default_factory=list)


class CartridgeEvalQAList(_StrictModel):
    """カートリッジ eval.json QA ペアリスト

    OpenAI strict structured outputs / llama.cpp 制約サンプリングは
    top-level に object が必要なため、``qa_pairs`` キーで配列をラップする
    (``MetaCognitivePlan`` と同方針)。
    """

    qa_pairs: list[CartridgeEvalQAItem]


# ── URL リコール用 自己採点 (url_relevance_score) ──

class UrlRelevanceJudgement(_StrictModel):
    """`backend/free/memory/sleep/url_curator.py` の URL 自己採点。

    sleep-time worker で fetch_url が呼ばれたターンを抽出し、
    補助タスクに「この URL は質問に対して正しく答えられたか」を
    0..1 の score で採点させる。
    """

    score: float = Field(ge=0.0, le=1.0)
    relevant: bool
    reason: str = Field(max_length=200)


# ── 型付けできなかった言明の命名 (assertion_naming) ──


class AssertionNaming(_StrictModel):
    """`backend/free/memory/sleep/assertion_curator.py` の言明命名。

    ``candidate_fact_tags`` が空を返した断定文 (日本語の「〜です」で終わる
    言明の大半) に対し、SemMem の subject に使える **ASCII slug** と
    命題化した object を補助タスクに付けさせる。

    ``subject_ns._SAFE_PART_RE`` が ASCII 英数字 / ``_`` / ``-`` しか許さない
    ため、日本語のキーワードからは決定論的に subject を導けない
    (``extractors.chat._is_usable_world_keyword`` が英字必須で弾く)。
    ``fact_attributes.yaml`` の JA→ASCII 辞書は登録済みの話題しか拾えない。
    ここだけモデルに命名させることで未登録の話題にも届かせる。

    ``is_assertion=False`` を返させる余地を残すのは、規則側の疑問形 / 依頼形
    ゲートをすり抜けた非言明を最後に落とすため。
    """

    is_assertion: bool
    slug: str = Field(max_length=40)
    object: str = Field(max_length=300)


# ── few-shot 手本の品質採点 (fewshot_quality_score) ──


class FewShotQualityJudgement(_StrictModel):
    """`backend/free/learning/fewshot_pool.py` の few-shot 手本 自己採点。

    「この Q/A を手本として提示したときモデルの振る舞いが良くなるか」を 0..1 で
    採点する。**採用可否の拒否権は持たない** (重み付けのみ) ため、
    ``UrlRelevanceJudgement`` のような bool 判定フィールドは置かない。

    拒否権を与えない理由は実測にある (2026-07-31): 稼働 aux (Qwen3.5-4B) は
    「42.195 ÷ 1.609 ≈ 26.195」(正しくは 26.2244) に満点を付けた。算術の誤りは
    ``response_arithmetic`` の決定論検算が担当し、本採点は定性面の順位付けに使う。
    """

    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)


# ── base prompt 候補の採用ゲート採点 (prompt_candidate_judge) ──


class PromptCandidateJudgement(_StrictModel):
    """`backend/free/llm/prompt_candidate_eval.py` の候補 prompt 採点。

    「失敗した実ターンの query に対し、候補 system prompt で再生成した応答は
    期待された振る舞い (ヒント) にどれだけ沿うか」を 0..1 で採点する。現行と
    候補を **同じケース・同じ judge** で採点し、差だけを採用判定に使う
    (f_04 §4.5)。絶対値には意味を持たせない。``score`` を先頭に置く (出力が
    ``max_tokens`` で切れても採点だけは取れる)。
    """

    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=80)


# ── purpose -> schema 自動解決マップ ──
#
# 呼出側が ``response_schema`` を明示しない場合、AuxClient が purpose
# 文字列から自動解決する。content_type で schema が分岐する purpose
# (``long_form_planning``) は本マップに含めず、呼出側が明示する。
PURPOSE_SCHEMAS: dict[str, type[_StrictModel]] = {
    "retrieval_chunk_gate": ChunkGateRelevance,
    "critique_synthesis": CritiqueSynthesisResult,
    "long_form_code_review": ReviewIssues,
    "long_form_text_review": ReviewIssues,
    "code_spec_synthesis": CodeSpec,
    "flowchart_synthesis": FlowchartSpec,
    "create_task_graph": CreateTaskGraph,
    "flow_spec_synthesis": FlowSpec,
    "flow_spec_part_synthesis": FlowSpec,
    "spec_revision_judge": SpecRevisionJudgement,
    "meta_cognitive_plan": MetaCognitivePlan,
    "cartridge_eval_generation": CartridgeEvalQAList,
    "url_relevance_score": UrlRelevanceJudgement,
    "assertion_naming": AssertionNaming,
    "fewshot_quality_score": FewShotQualityJudgement,
    "prompt_candidate_judge": PromptCandidateJudgement,
}


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """``$defs`` / ``definitions`` を ``$ref`` 位置にインライン展開する。

    llama.cpp の制約サンプリングは JSON Schema の ``$ref`` を完全には解決
    できないビルドが多いため、ネストモデルを再帰的にインライン化して
    flat な schema を返す。
    """
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
    if not defs:
        return schema

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                target = defs.get(key)
                if isinstance(target, dict):
                    # 再帰展開してから返す
                    return _resolve({k: v for k, v in target.items()})
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                key = ref.split("/")[-1]
                target = defs.get(key)
                if isinstance(target, dict):
                    return _resolve({k: v for k, v in target.items()})
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    return _resolve(schema)


def _enforce_additional_properties_false(node: Any) -> Any:
    """``type: object`` のノード全てに ``additionalProperties: false`` を付与する。

    Pydantic v2 の ``model_json_schema()`` は ``extra="forbid"`` でも
    ``additionalProperties: false`` を必ず明記しないことがあるため、
    schema 全体を走査して明示的に追加する。llama.cpp の strict サンプリ
    ングが「未定義キー」を許してしまわないようにするための保険。
    既に ``True`` が明示されているノード (free-form object を宣言したフィールド)
    は ``setdefault`` の挙動で上書きされない。
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _enforce_additional_properties_false(v)
    elif isinstance(node, list):
        for v in node:
            _enforce_additional_properties_false(v)
    return node


def _enforce_strict_required(node: Any) -> Any:
    """OpenAI strict structured outputs 互換: 全プロパティを required にする。

    OpenAI / llama.cpp の strict structured outputs では、``additionalProperties:
    false`` だけでは不十分で、すべての ``properties`` のキーを ``required``
    に列挙する必要がある (任意フィールドが許されない)。Pydantic v2 はデフォ
    ルト値ありフィールドを ``required`` から外すため、本ヘルパで補正する。

    ``additionalProperties: True`` (free-form object、例: ``tool_args``) は
    properties 自体を持たないことが多いため、その場合は何もしない。
    """
    if isinstance(node, dict):
        if (
            node.get("type") == "object"
            and isinstance(node.get("properties"), dict)
            and node.get("additionalProperties") is False
        ):
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _enforce_strict_required(v)
    elif isinstance(node, list):
        for v in node:
            _enforce_strict_required(v)
    return node


def make_response_format(
    schema_cls: type[BaseModel],
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """Pydantic モデルから OAI 互換 ``response_format`` dict を生成する。

    Args:
        schema_cls: Pydantic v2 BaseModel サブクラス。
        name: ``json_schema.name`` フィールド。省略時は class 名を使用。

    Returns:
        ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ...,
        "strict": true}}`` 形式の dict。``/v1/chat/completions`` ペイロード
        の ``response_format`` フィールドにそのまま埋め込める。
    """
    raw = schema_cls.model_json_schema()
    inlined = _inline_defs(raw)
    _enforce_additional_properties_false(inlined)
    _enforce_strict_required(inlined)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or schema_cls.__name__,
            "strict": True,
            "schema": inlined,
        },
    }


def resolve_response_format_for_purpose(purpose: str) -> dict[str, Any] | None:
    """purpose 文字列から ``response_format`` dict を自動解決する。

    ``PURPOSE_SCHEMAS`` に登録された purpose に対してのみ有効。``content_type``
    で分岐する purpose (``long_form_planning``) は本関数では解決できず、
    呼出側が ``response_schema`` を明示する必要がある。

    Returns:
        解決可能なら ``response_format`` dict、不明 purpose なら ``None``。
    """
    if not purpose:
        return None
    cls = PURPOSE_SCHEMAS.get(purpose)
    if cls is None:
        return None
    return make_response_format(cls, name=purpose)


__all__ = [
    "PromptCandidateJudgement",
    "RetrievalQualityJudgement",
    "RetrievalNecessityJudgement",
    "ChunkGateRelevance",
    "CritiqueSynthesisResult",
    "CodeUnitPlan",
    "SectionUnitPlan",
    "CodePlan",
    "TextPlan",
    "ReviewIssueItem",
    "ReviewIssues",
    "MetaCognitivePlan",
    "CartridgeEvalQAItem",
    "CartridgeEvalQAList",
    "UrlRelevanceJudgement",
    "SpecRevisionJudgement",
    "PURPOSE_SCHEMAS",
    "make_response_format",
    "resolve_response_format_for_purpose",
]
