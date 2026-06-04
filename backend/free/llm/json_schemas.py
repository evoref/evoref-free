"""アシストモデル JSON 応答 purpose 別スキーマ定義

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


# ── 検索品質判定 (retrieval_quality_judge) ──

class RetrievalQualityJudgement(_StrictModel):
    """`backend/free/rag/self_rag_judge.py` の閾値境界判定。"""

    quality: Literal["high", "medium", "low"]


# ── 検索必要性判定 (retrieval_necessity_judge) ──

class RetrievalNecessityJudgement(_StrictModel):
    """`backend/free/rag/self_rag_judge.py` の uncertain 救済判定。

    ルールで確定できないクエリに対して、アシストモデルが 3 択で意図を返す:

    - ``retrieve``: ローカル RAG (ドキュメント / 過去会話 / カートリッジ) を
      参照する必要がある (How-to / 既知ドキュメント質問 / 過去会話参照)。
    - ``fetch``: 外部からリアルタイムに情報を取りに行く必要がある
      (最新ニュース / 株価 / 天気 / 公式サイトの最新状態)。RAG ではなく
      ``fetch_url`` ツールに委ねる。
    - ``skip``: どちらも不要な雑談 / 自明な質問。

    新規追加。旧 ``need_rag: bool`` 2 値からの差分は、
    呼出側で ``action == "skip"`` を `"skip"`、それ以外 (``retrieve`` /
    ``fetch``) を該当値に解釈する。
    """

    action: Literal["retrieve", "fetch", "skip"]


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
    units: list[SectionUnitPlan]


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
    アシストモデルに生成させる際に使う。``tags`` は分類用の任意ラベルだが、
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
    アシストモデルに「この URL は質問に対して正しく答えられたか」を
    0..1 の score で採点させる。
    """

    score: float = Field(ge=0.0, le=1.0)
    relevant: bool
    reason: str = Field(max_length=200)


# ── 実行可能クエリのコマンド合成 (executable_command_synth) ──

class ExecutableCommandSynth(_StrictModel):
    """`backend/free/agent/tool_call_judge.py` の executable query 判定 + コマンド合成.

    「Chrome のバージョン」「現在のディスク使用量」「インストール済み Python
    パッケージ」のような環境依存事実 (Python / シェルから取得できる事実) を、
    パターン辞書 (`_EXECUTABLE_QUERY_COMMANDS`) ではなく アシストモデルで
    判定する。

    出力フィールド:

    - ``is_executable``: クエリが Python / シェルで取得できる環境依存事実か。
      知識質問 / 一般会話 / 副作用を伴う依頼の場合は ``False``。
    - ``command``: ``is_executable=True`` 時に取得用の単一行コマンド。
      ``python -c "..."`` 形式または OS ネイティブ短コマンド。30 秒以内に
      終了し、副作用 (書き込み / 削除 / ネットワーク送信 / 対話) を持たない
      ものに限定する。``is_executable=False`` のときは ``""``。
    - ``rationale``: 1 行説明 (ログ用、UI 非表示)。
    """

    is_executable: bool
    command: str = ""
    rationale: str = ""


# ── ツール呼出判定 (tool_judgment) ──

class ToolJudgmentResult(_StrictModel):
    """`backend/free/agent/tool_call_judge.py` のアシスト判定。

    既存プロンプト (`_DEFAULT_SYSTEM_PROMPT`) の出力形式
    ``{"tool": "<ツール名>" | "", "args": {...}}`` に合わせる。
    旧仕様のリテラル文字列 ``"no_tool"`` は廃止
    ツール不要時は ``tool=""`` を返す。

    ``args`` フィールドは tool ごとに任意の引数を取るため、free-form
    object (``additionalProperties: True``) として扱う。``ToolsRegistry``
    が引数バリデーションを担当するため、JSON 構文レベルの strict 化は不要。
    """

    tool: str = ""
    args: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"additionalProperties": True},
    )


# ── エディタ出力ファイル名導出 (editor_filename) ──

class EditorFilenameResult(_StrictModel):
    """`backend/free/llm/editor_filename.py` のエディタタブ名導出.

    コーディングモードで生成したコード/仕様書を Pro エディタへタブ表示する際、
    生成内容から **拡張子なしの ASCII snake_case** ファイル名 stem を 1 つ
    導出する。日本語見出しをそのまま流用するとタブ名が日本語化するため、
    アシストモデルに英語の簡潔名を生成させる (SPLIT モードの ``file_name`` と
    同思想)。

    - ``file_name``: 英小文字 + 数字 + アンダースコアのみ、拡張子なし。
      呼出側が言語に応じた拡張子を付与する。LLM が日本語/記号を返しても
      呼出側で ASCII slug 化 + 言語別フォールバックするため安全。
    """

    file_name: str = ""


# ── purpose -> schema 自動解決マップ ──
#
# 呼出側が ``response_schema`` を明示しない場合、AssistClient が purpose
# 文字列から自動解決する。content_type で schema が分岐する purpose
# (``long_form_planning``) は本マップに含めず、呼出側が明示する。
PURPOSE_SCHEMAS: dict[str, type[_StrictModel]] = {
    "retrieval_quality_judge": RetrievalQualityJudgement,
    "retrieval_necessity_judge": RetrievalNecessityJudgement,
    "retrieval_chunk_gate": ChunkGateRelevance,
    "critique_synthesis": CritiqueSynthesisResult,
    "long_form_code_review": ReviewIssues,
    "long_form_text_review": ReviewIssues,
    "tool_judgment": ToolJudgmentResult,
    "executable_command_synth": ExecutableCommandSynth,
    "meta_cognitive_plan": MetaCognitivePlan,
    "cartridge_eval_generation": CartridgeEvalQAList,
    "url_relevance_score": UrlRelevanceJudgement,
    "editor_filename": EditorFilenameResult,
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
    既に ``True`` が明示されているノード (例: ``ToolJudgmentResult.args``)
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
    "ToolJudgmentResult",
    "ExecutableCommandSynth",
    "MetaCognitivePlan",
    "CartridgeEvalQAItem",
    "CartridgeEvalQAList",
    "UrlRelevanceJudgement",
    "EditorFilenameResult",
    "PURPOSE_SCHEMAS",
    "make_response_format",
    "resolve_response_format_for_purpose",
]
