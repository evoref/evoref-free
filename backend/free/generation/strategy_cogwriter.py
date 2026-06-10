"""CogWriter 戦略（アシストモデルあり）

設計書 f_09_long_form_generation.md §5 準拠。
認知的ライティング理論に基づき、計画・レビューをアシストモデルが担当し、
メインモデルは生成に専念する。
"""

from __future__ import annotations

import ast
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from backend.exceptions import UnitGenerationError
from backend.free.generation.code_skeleton import CodeSkeleton
from backend.free.generation.models import (
    CodeUnit,
    ContentType,
    GenerationPlan,
    LongFormMode,
    SectionPlan,
    extract_target_chars,
)
from backend.free.generation.rolling_context import RollingContext
from backend.free.generation.spec_renderer import render_spec_for_prompt
from backend.free.generation.strategy_common import (
    build_code_unit_messages,
    build_text_unit_messages,
    excerpt_continuation_content,
    excerpt_for_expand,
    fallback_plan,
    finalize_plan_units,
    generate_plan_json,
    parse_plan,
    resolve_max_units,
)
from backend.free.generation.token_budget import TokenBudget
from backend.free.llm.json_schemas import CodeSpec

if TYPE_CHECKING:
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.llm.local_client import LocalClient

logger = logging.getLogger("backend.free.generation.strategy_cogwriter")


# ── プロンプトテンプレート ──

# 事前準備: コード生成の前に固める共有設計仕様 (contract)。後続の plan 生成と
# 全ユニット生成に注入され、ファイル横断のモジュール名 / データモデル /
# シグネチャ / エントリポイント / プロトコルを統一する。f_08 §3 参照。
_CODE_SPEC_PROMPT = """\
あなたは熟練ソフトウェアアーキテクトです。
以下の指示を実装するための「設計仕様(契約)」をJSON形式で出力してください。
この仕様は後続の全コードユニット生成に共有され、ファイル間の整合性を保証します。
矛盾の無い単一の整合した設計にしてください。

【ユーザー指示】{instruction}
【既存コード情報（RAG）】{rag_context}
【関連メモリ】{memory_context}

必ず次を厳密に定義してください:
- modules: 正準なファイル構成 (各 path と purpose)。後続コードはこの path 以外を import しない。\
存在しないモジュール (例: 集約用の架空 models.py) を作らない。
- data_models: ファイル横断で共有する型 (name / module / kind / fields[name,type])。\
**全ファイルがこのフィールド名・型に厳密に従う**。同じ型を別フィールドで再定義しない。
- interfaces: 公開する関数/メソッドのシグネチャ (module / signature)。送受信する側が一致させる。
- entry_point: 実行起点。CLI/アプリなら必ず設定する \
(例: module="main.py", invocation="if __name__ == '__main__': asyncio.run(main())")。\
ライブラリで起点が無い場合のみ module を空文字列にする。
- protocol: コンポーネント間のデータ/通信契約 \
(例: クライアントとサーバが送受信する JSON のキーと型)。無ければ空文字列。
- constraints: 言語/バージョン/規約などの制約。

JSON のみ出力してください。"""

_FLOWCHART_PROMPT = """\
あなたはソフトウェア設計者です。
以下の設計仕様に基づき、プログラムの制御/データフローを表す
mermaid フローチャート (flowchart TD) を生成してください。

【ユーザー指示】{instruction}
【設計仕様】
{spec}

"mermaid" キーに mermaid 記法の文字列のみ (先頭は flowchart TD)、
補足があれば "notes" キーに入れた JSON を返してください。
mermaid 値にコードフェンス (```) は含めないでください。"""

_CODE_PLAN_PROMPT = """\
あなたはソフトウェアアーキテクトです。
以下の指示と設計仕様(契約)に基づき、実装計画をJSON形式で生成してください。

【ユーザー指示】{instruction}
【設計仕様(契約) — この通りのモジュール名・型・シグネチャで計画する】
{spec}
【既存コード情報（RAG）】{rag_context}
【関連メモリ】{memory_context}

設計方針:
- **【ユーザー指示】で要求されたプログラムを正確に実装する**。【既存コード情報（RAG）】
  【関連メモリ】に別プログラム（別のゲーム等）のコードや設計が含まれていても、それを
  流用・模倣しない。参考情報がユーザー指示と矛盾する場合は **ユーザー指示を最優先** する。
- **設計仕様(契約)が与えられている場合、各 unit の file_path は契約の modules の path に一致させ、
  扱うデータモデルは契約の data_models のフィールド/型に厳密準拠する**。
- units は **単一責務ごとに細かく分割** する (1 関数 / 1 クラス / 1 型定義 = 1 unit)。
  大きな関数は補助 unit に分け各 unit を小さく保つ (ローカル LLM が 1 回の生成で
  安定して出力できる粒度にする = 品質向上)。
- 規模が大きい場合は **論理的なまとまりごとに別ファイル** へ分割し、各 unit の
  file_path を適切に振り分ける (例: 本体 / ユーティリティ / 型定義 / テスト)。
  単一ファイルで十分なら全 unit を同じ file_path にする。
- depends_on で unit 間の依存を明示し、生成順を正す。
- entry_point が契約で指定されている場合、それを実装する unit を必ず含める。

出力形式:
{{
  "content_type": "code",
  "title": "...",
  "target_length": 3000,
  "global_context": "Python 3.12+, type hints必須, async/await",
  "constraints": ["既存APIとの互換性維持"],
  "units": [
    {{
      "kind": "types",
      "name": "TokenBudget",
      "file_path": "token_budget.py",
      "spec": "context_sizeから比率ベースで予算を算出するdataclass",
      "depends_on": [],
      "estimated_tokens": 300
    }},
    {{
      "kind": "function",
      "name": "compute_budget",
      "file_path": "token_budget.py",
      "spec": "TokenBudget を生成する純関数。境界値を検証する",
      "depends_on": ["TokenBudget"],
      "estimated_tokens": 300
    }}
  ]
}}

JSON のみ出力してください。"""

_TEXT_PLAN_PROMPT = """\
あなたは文書構成の専門家です。
以下の指示に基づき、文書のアウトラインをJSON形式で生成してください。

【ユーザー指示】{instruction}
【参考情報（RAG）】{rag_context}
【関連メモリ】{memory_context}

重要（成果物の主題を取り違えないこと）:
- ユーザー指示や参考情報に含まれる「保存先パス」「ファイル名」「.docx で」\
「出力してください」「保存してください」等は **配信（どこへ出力するか）の指示** \
であって、文書の主題ではありません。これらは constraints に retain しつつ、\
title・global_context・units は **生成する文書そのもの**（例:「小説」なら物語本文）\
を対象に計画してください。
- ファイルの保存手順・パスの存在確認・タスク実行の説明のような「作業手順」を\
文書内容として計画・生成してはいけません。求められているのは成果物の実体です。
- global_context は成果物の種類と文体（例:「短編小説（物語）」）を記述し、\
「システム操作」「ファイル生成タスク」等の手続き的な分類にしないでください。

出力形式:
{{
  "content_type": "text",
  "title": "...",
  "target_length": 5000,
  "global_context": "文書の種類・文体・語り口（例: 技術文書、丁寧体）",
  "constraints": ["初心者向け"],
  "units": [
    {{
      "heading": "はじめに",
      "key_points": ["背景", "目的"],
      "estimated_tokens": 500
    }}
  ]
}}

global_context には文書の種類と文体を具体的に記述してください。
key_points には、生成すべき具体的な内容を書いてください。
「〜について書く」のようなメタ的な記述ではなく、実際に含める内容を指定してください。

JSON のみ出力してください。"""

_TEXT_PLAN_CONTINUATION_PROMPT = """\
あなたは文書構成の専門家です。
既存テキストに基づき、ユーザー指示に沿ったアウトラインをJSON形式で生成してください。

【ユーザー指示】{instruction}

【既存テキスト】
{existing_content}

【参考情報（RAG）】{rag_context}
【関連メモリ】{memory_context}

重要な制約:
- まず既存テキストの種類（小説、報告書、技術文書、議事録、マニュアル等）を判別してください。
- 既存テキストの文体（語り口、文末表現、語彙のレベル、構造的特徴）を正確に分析し、\
global_context に具体的に記述してください。
- key_points には、実際に生成すべき具体的な内容を書いてください。\
メタ的な記述（「〜について書く」「〜を描写する」等）ではなく、生成する内容そのものを指定してください。
- 既存テキストの文体・トーン・構造を維持してください。

出力形式:
{{
  "content_type": "text",
  "title": "...",
  "target_length": {target_length},
  "global_context": "既存テキストの分析結果（種類、文体、構造の特徴を具体的に記述）",
  "constraints": ["既存テキストの文体・トーンを維持", "メタ的記述を含めない"],
  "units": [
    {{
      "heading": "本文",
      "key_points": ["具体的な生成内容"],
      "estimated_tokens": 500
    }}
  ]
}}

JSON のみ出力してください。"""

# SPLIT モード: 既存テキストを元に **機能ごと個別ファイル** に分解して
# 詳細仕様書を出力する。プロンプトは EXPAND とほぼ同じだが、各 unit の
# ``file_name`` フィールドを LLM に必ず埋めさせる点が異なる。
_TEXT_PLAN_SPLIT_PROMPT = """\
あなたは技術仕様書の編集者です。
既存テキストに含まれる機能/トピックを抽出し、それぞれを **独立した個別ファイル**
として書き出す詳細仕様書のアウトラインを JSON 形式で生成してください。
各 unit が 1 個のファイルに対応します。

【ユーザー指示】{instruction}

【既存テキスト】
{existing_content}

【参考情報（RAG）】{rag_context}
【関連メモリ】{memory_context}

重要な制約:
- 既存テキストから機能/コンポーネント/トピックを 5〜12 個抽出し、\
それぞれを 1 unit (= 1 ファイル) として units 配列に列挙してください。
- 各 unit の **必ず file_name フィールドを埋めて** ください。\
file_name は英数字 + アンダースコア (`_`) のみ、32 文字以内、拡張子なし、\
スネークケースで機能を端的に表す名前 (例: `grid_management`, `rule_logic`, \
`gui_layout`, `io_handlers`, `testing_strategy`)。
- 各 unit の heading は機能名 (例: 「グリッド管理」「ルール演算」「描画」)。
- 各 unit の key_points には、その機能の **詳細仕様** を箇条書きで列挙してください。\
入力/出力、データ構造、アルゴリズム、境界条件、エラー処理、テスト観点 など、\
仕様書として必要な要素を具体的に書く。「〜について述べる」等のメタ記述は禁止。
- estimated_tokens は各 unit で {per_unit_tokens} 程度を目安に設定してください。
- 各 unit は **完全に独立して読める** 内容にしてください\
(他 unit への参照は最小限、ファイル単体で意味が通る)。

出力形式:
{{
  "content_type": "text",
  "title": "詳細仕様書 (機能別)",
  "target_length": {target_length},
  "global_context": "既存テキストの分析 (技術分野、文体、対象読者) を具体的に",
  "constraints": ["要約せず詳細化", "1 機能 = 1 ファイル", "ファイル単体で完結"],
  "units": [
    {{
      "heading": "グリッド管理",
      "file_name": "grid_management",
      "key_points": [
        "二次元 NumPy 配列でセル状態を保持し、各セルは bool で生死を表現する",
        "境界条件は周期境界 (torus topology) を採用し、上下/左右をラップする",
        "公開メソッド: get(i,j), set(i,j,value), neighbors(i,j), to_array()",
        "テスト: 周期境界のラップが正しいこと / 不正座標で IndexError"
      ],
      "estimated_tokens": {per_unit_tokens}
    }},
    {{
      "heading": "ルール演算",
      "file_name": "rule_logic",
      "key_points": [
        "コンウェイの公理 (B3/S23) を厳密実装",
        "step() メソッドが次世代グリッドを返す純粋関数として動作",
        "NumPy 配列演算で全セル並列に隣接数を計算する (シフト+加算)",
        "テスト: 静物 (ブロック) と振動子 (ブリンカー) で 1 周期検証"
      ],
      "estimated_tokens": {per_unit_tokens}
    }}
  ]
}}

JSON のみ出力してください。"""

# EXPAND モード: 既存テキストを元に **詳細仕様書化** する。
# 既存の機能/トピックを **複数の独立セクション** に分解し、各セクションに
# 詳細仕様 (入出力、データ構造、アルゴリズム、エラー条件、テスト観点) を記述する。
_TEXT_PLAN_EXPAND_PROMPT = """\
あなたは技術仕様書の編集者です。
既存テキストに含まれる機能/トピックを抽出し、それぞれを **独立した詳細セクション** に
分解した詳細仕様書のアウトラインを JSON 形式で生成してください。

【ユーザー指示】{instruction}

【既存テキスト】
{existing_content}

【参考情報（RAG）】{rag_context}
【関連メモリ】{memory_context}

重要な制約:
- 既存テキストを **要約せず、機能ごとに詳細化** してください。\
継続テキストを書くのではなく、機能仕様書として再構成してください。
- 既存テキストから機能/コンポーネント/トピックを 5〜12 個程度抽出し、\
それぞれを 1 unit (= 1 セクション) として units 配列に列挙してください。
- 各 unit の heading は機能名 (例: 「グリッド管理」「ルール演算」「描画」「I/O」「テスト戦略」)。
- 各 unit の key_points には、その機能の **詳細仕様** を箇条書きで列挙してください。\
入力/出力、データ構造、アルゴリズム、境界条件、エラー処理、テスト観点 など、\
仕様書として必要な要素を具体的に書く。「〜について述べる」等のメタ記述は禁止。
- estimated_tokens は各セクションで {per_unit_tokens} 程度を目安に設定してください。
- target_length は全 unit 合計の目標文字数として {target_length} を指定してください。
- 既存テキストの語り口・文体は維持しますが、構造は機能ごとセクション化に再編成してください。

出力形式:
{{
  "content_type": "text",
  "title": "{title_hint}",
  "target_length": {target_length},
  "global_context": "既存テキストの分析 (技術分野、文体、対象読者) を具体的に",
  "constraints": ["要約せず詳細化", "機能ごとにセクション化", "メタ記述を含めない"],
  "units": [
    {{
      "heading": "グリッド管理",
      "key_points": [
        "二次元 NumPy 配列でセル状態を保持し、各セルは bool で生死を表現する",
        "境界条件は周期境界 (torus topology) を採用し、上下/左右をラップする",
        "初期化 API: from_shape(rows, cols) / from_pattern(coords) / random(density)",
        "公開メソッド: get(i,j), set(i,j,value), neighbors(i,j), to_array()",
        "テスト: 周期境界のラップが正しいこと / 不正座標で IndexError を投げること"
      ],
      "estimated_tokens": {per_unit_tokens}
    }},
    {{
      "heading": "ルール演算",
      "key_points": [
        "コンウェイの公理 (B3/S23) を厳密実装",
        "step() メソッドが次世代グリッドを返す純粋関数として動作",
        "NumPy 配列演算で全セル並列に隣接数を計算する (シフト+加算)",
        "境界条件は Grid に委譲する設計とし、ルール演算自体は境界に非依存",
        "テスト: 静物 (ブロック/フィッシュフック) と振動子 (ブリンカー) で 1 周期検証"
      ],
      "estimated_tokens": {per_unit_tokens}
    }}
  ]
}}

JSON のみ出力してください。"""

_TEXT_UNIT_USER = """\
# 文書構成
タイトル: {title}
セクション一覧: {section_headings}

# 直前セクション末尾
{short_term}

# 現在のセクション
見出し: {heading}
含めるべき要点: {key_points}

# 参考資料
{rag_context}

本文のみを出力してください（見出し行・メタ解説・執筆意図の説明は不要）:"""

_CODE_REVIEW_PROMPT = """\
以下の生成コードをレビューしてください。

【実装計画の仕様】
{plan_specs}

【生成コードのシグネチャ】
{generated_signatures}

以下の観点でチェックし、問題があればJSON配列で返してください:
1. 計画のspecと生成コードのシグネチャの整合性
2. ユニット間の型参照の一貫性

問題がなければ空配列 [] を返してください。
出力形式:
[
  {{"unit_idx": 0, "issue": "引数の型が計画と異なる", "fix": "int → str に修正"}}
]

JSON のみ出力してください。"""

_TEXT_REVIEW_PROMPT = """\
以下の文書をレビューしてください。

【セクション計画】
{plan_sections}

【生成テキスト（各セクション冒頭）】
{generated_previews}

以下の観点でチェックし、問題があればJSON配列で返してください:
1. 各セクションが計画の要点（key_points）をカバーしているか
2. 文体・用語の一貫性

問題がなければ空配列 [] を返してください。
出力形式:
[
  {{"unit_idx": 0, "issue": "要点「背景」が欠落", "fix": "冒頭に背景説明を追加"}}
]

JSON のみ出力してください。"""

_REVISE_PROMPT = """\
以下のコードを修正指示に従って修正してください。

【元のコード】
{original}

【修正指示】
{fix_instruction}

修正後のコードのみ出力してください:"""


# ── レビュー結果 ──

class ReviewIssue:
    """レビューで発見された問題"""

    __slots__ = ("unit_idx", "issue", "fix")

    def __init__(self, unit_idx: int, issue: str, fix: str):
        self.unit_idx = unit_idx
        self.issue = issue
        self.fix = fix


# ── CogWriter 戦略本体 ──

class CogWriterStrategy:
    """CogWriter 戦略（アシストモデルあり）

    計画・レビューをアシストモデルが担当し、
    メインモデルは生成に専念する。
    """

    def __init__(
        self,
        main_client: LocalClient,
        assist_client: AssistModelClient,
        config: dict,
        debug_logger=None,
        generation_params: dict | None = None,
    ):
        self.main_client = main_client
        self.assist_client = assist_client
        self.config = config
        self._debug_logger = debug_logger
        self._lf_config = config.get("long_form", {})
        self._generation_params = generation_params or {}

    async def create_plan(
        self,
        instruction: str,
        context: dict,
        content_type: ContentType,
        budget: TokenBudget,
    ) -> GenerationPlan:
        """アシストモデルで計画を生成"""
        t0 = time.monotonic()
        rag_context = context.get("rag", "")
        memory_context = context.get("memory", "")

        existing_content = context.get("existing_content", "")

        long_form_mode: LongFormMode = context.get(
            "long_form_mode", LongFormMode.CONTINUE,
        )

        # コード生成は事前準備として設計仕様 (契約) を先に合成し、plan と
        # 全ユニット生成に注入する。合成失敗 / 無効化時は None で従来挙動。
        code_spec: CodeSpec | None = None

        if content_type == ContentType.CODE:
            code_spec = await self._synthesize_code_spec(
                instruction, rag_context, memory_context,
            )
            prompt = _CODE_PLAN_PROMPT.format(
                instruction=instruction,
                spec=render_spec_for_prompt(code_spec) or "(なし)",
                rag_context=rag_context,
                memory_context=memory_context,
            )
        elif existing_content and long_form_mode in (
            LongFormMode.EXPAND, LongFormMode.SPLIT,
        ):
            # EXPAND/SPLIT モード: 既存テキストを詳細仕様書として再構成する。
            # 抜粋を継続モードより広く取り (4000 char) 機能列挙の精度を上げる。
            excerpt = excerpt_for_expand(existing_content)
            # target_length は「ユーザー指定 > 入力長 × 1.5 > 最小 3000」
            user_target = extract_target_chars(instruction, default=0)
            input_scaled = int(len(existing_content) * 1.5)
            target_length = max(user_target, input_scaled, 3000)
            per_unit_tokens = max(600, target_length // 12)
            if long_form_mode == LongFormMode.SPLIT:
                prompt = _TEXT_PLAN_SPLIT_PROMPT.format(
                    instruction=instruction,
                    existing_content=excerpt,
                    rag_context=rag_context,
                    memory_context=memory_context,
                    target_length=target_length,
                    per_unit_tokens=per_unit_tokens,
                )
            else:
                prompt = _TEXT_PLAN_EXPAND_PROMPT.format(
                    instruction=instruction,
                    existing_content=excerpt,
                    rag_context=rag_context,
                    memory_context=memory_context,
                    target_length=target_length,
                    per_unit_tokens=per_unit_tokens,
                    title_hint="詳細仕様書",
                )
        elif existing_content:
            # 追記モード: 既存テキストを含む専用プロンプトを使用 (冒頭+末尾を抜粋)
            excerpt = excerpt_continuation_content(existing_content)
            # target_length をユーザー指示から推定（デフォルト1000字）
            target_length = extract_target_chars(instruction, default=1000)
            prompt = _TEXT_PLAN_CONTINUATION_PROMPT.format(
                instruction=instruction,
                existing_content=excerpt,
                rag_context=rag_context,
                memory_context=memory_context,
                target_length=target_length,
            )
        else:
            prompt = _TEXT_PLAN_PROMPT.format(
                instruction=instruction,
                rag_context=rag_context,
                memory_context=memory_context,
            )

        # EXPAND/SPLIT モードでは下限 8 を保証 (機能ごとセクション化のため)
        max_units = resolve_max_units(self._lf_config, long_form_mode)
        # content_type に応じた schema 選択と例外フォールバックは共通化
        data = await generate_plan_json(self.assist_client, prompt, content_type)

        if not data or "units" not in data:
            logger.warning(
                "Plan JSON parse failed, falling back to single unit"
            )
            plan = fallback_plan(instruction, content_type, code_spec=code_spec)
        else:
            plan = parse_plan(data, content_type, instruction, code_spec=code_spec)

        # ユニット数上限 + コード用依存順ソート (共通後処理)
        finalize_plan_units(plan, max_units, content_type)

        # 任意: 設計仕様から mermaid フローチャートを合成し plan に同伴する
        # (config code_flowchart_enabled=True 時のみ、既定 OFF)。SPEC.md に埋込。
        if content_type == ContentType.CODE and code_spec is not None:
            plan.code_flowchart = await self._synthesize_flowchart(
                instruction, code_spec,
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "Plan created: content_type=%s, units=%d, elapsed=%.2fs",
            content_type.value, len(plan.units), elapsed,
        )
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "plan",
                "strategy": "cogwriter",
                "content_type": content_type.value,
                "units_count": len(plan.units),
                "elapsed_sec": round(elapsed, 3),
            })

        return plan

    async def _synthesize_code_spec(
        self,
        instruction: str,
        rag_context: str,
        memory_context: str,
    ) -> CodeSpec | None:
        """コード生成の事前準備として共有設計仕様 (契約) を合成する。

        ``assist_client=None`` (degraded) / config 無効化 / 合成失敗時は
        ``None`` を返し、呼出側は仕様なしの従来挙動にフォールバックする。
        """
        if self.assist_client is None:
            return None
        if not self._lf_config.get("code_spec_enabled", True):
            return None

        prompt = _CODE_SPEC_PROMPT.format(
            instruction=instruction,
            rag_context=rag_context,
            memory_context=memory_context,
        )
        t0 = time.monotonic()
        try:
            data = await self.assist_client.generate_json(
                prompt, max_tokens=3072, temperature=0.3,
                purpose="code_spec_synthesis",
            )
            if not isinstance(data, dict):
                logger.warning("Code spec synthesis returned non-dict; skipping")
                return None
            spec = CodeSpec.model_validate(data)
        except Exception as e:  # noqa: BLE001 — 合成失敗は従来挙動に倒す
            logger.warning("Code spec synthesis failed: %s", e)
            return None

        elapsed = time.monotonic() - t0
        logger.info(
            "Code spec synthesized: modules=%d, data_models=%d, elapsed=%.2fs",
            len(spec.modules), len(spec.data_models), elapsed,
        )
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "code_spec",
                "strategy": "cogwriter",
                "modules": len(spec.modules),
                "data_models": len(spec.data_models),
                "interfaces": len(spec.interfaces),
                "elapsed_sec": round(elapsed, 3),
            })
        return spec

    async def _synthesize_flowchart(
        self,
        instruction: str,
        spec: CodeSpec,
    ) -> str:
        """設計仕様から mermaid フローチャートを合成する (config 任意・既定 OFF)。

        ``code_flowchart_enabled`` が False / ``assist_client=None`` / 合成失敗時は
        空文字列を返し、SPEC.md にフローチャート節を付けない。
        """
        if self.assist_client is None:
            return ""
        if not self._lf_config.get("code_flowchart_enabled", False):
            return ""

        prompt = _FLOWCHART_PROMPT.format(
            instruction=instruction,
            spec=render_spec_for_prompt(spec),
        )
        try:
            data = await self.assist_client.generate_json(
                prompt, max_tokens=1024, temperature=0.3,
                purpose="flowchart_synthesis",
            )
            if isinstance(data, dict):
                return str(data.get("mermaid", "") or "")
        except Exception as e:  # noqa: BLE001 — 合成失敗は無効化と同じ扱い
            logger.warning("Flowchart synthesis failed: %s", e)
        return ""

    async def generate_unit(
        self,
        unit: CodeUnit | SectionPlan,
        rolling: RollingContext,
        content_type: ContentType,
    ) -> AsyncIterator[str]:
        """メインモデルでユニットを逐次生成（ストリーミング）"""
        config_max = self._lf_config.get("unit_max_tokens", 2000)
        # estimated_tokens の 1.5 倍を上限に（余裕を持たせる）、設定値と比較して大きい方
        unit_max_tokens = max(config_max, int(unit.estimated_tokens * 1.5))

        if content_type == ContentType.CODE:
            assert isinstance(unit, CodeUnit)
            messages = build_code_unit_messages(unit, rolling)
        else:
            assert isinstance(unit, SectionPlan)
            messages = self._build_text_unit_messages(unit, rolling)

        try:
            gen_kwargs: dict = {
                "stream": True,
                "temperature": self._generation_params.get("temperature", 0.7),
                "max_tokens": unit_max_tokens,
                "id_slot": self.main_client.chat_slot,
            }
            for k in ("top_p", "top_k", "presence_penalty"):
                if k in self._generation_params:
                    gen_kwargs[k] = self._generation_params[k]
            stream = await self.main_client.generate(messages, **gen_kwargs)
            async for token in stream:
                yield token
        except Exception as e:
            logger.error("Unit generation failed: %s", e)
            raise UnitGenerationError(
                f"Failed to generate unit: {e}"
            ) from e

    async def review(
        self,
        rolling: RollingContext,
        content_type: ContentType,
    ) -> list[ReviewIssue]:
        """アシストモデルでレビュー"""
        if not rolling.generated_units:
            return []

        t0 = time.monotonic()

        if content_type == ContentType.CODE:
            issues = await self._review_code(rolling)
        else:
            issues = await self._review_text(rolling)

        elapsed = time.monotonic() - t0
        logger.info(
            "Review complete: issues=%d, elapsed=%.2fs",
            len(issues), elapsed,
        )
        if self._debug_logger:
            self._debug_logger.log_long_form_event({
                "phase": "review",
                "strategy": "cogwriter",
                "content_type": content_type.value,
                "issues_found": len(issues),
                "elapsed_sec": round(elapsed, 3),
            })

        return issues

    async def revise_unit(
        self,
        issue: ReviewIssue,
        rolling: RollingContext,
        content_type: ContentType,
    ) -> AsyncIterator[str]:
        """修正指示に基づくリライト（最大1回）"""
        if issue.unit_idx >= len(rolling.generated_units):
            return

        original = rolling.generated_units[issue.unit_idx]
        config_max = self._lf_config.get("unit_max_tokens", 2000)
        # リライト時も元のユニットのトークン数に応じた上限を設定
        from backend.utils import estimate_tokens as _est
        unit_max_tokens = max(config_max, int(_est(original) * 1.5))

        prompt = _REVISE_PROMPT.format(
            original=original,
            fix_instruction=issue.fix,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            stream = await self.main_client.generate(
                messages,
                stream=True,
                temperature=0.5,
                max_tokens=unit_max_tokens,
                id_slot=self.main_client.chat_slot,
            )
            revised_text = ""
            async for token in stream:
                revised_text += token
                yield token

            # generated_units を更新
            rolling.generated_units[issue.unit_idx] = revised_text
        except Exception as e:
            logger.warning("Revision failed for unit %d: %s", issue.unit_idx, e)

    # ── プロンプト構築 ──

    def _build_text_unit_messages(
        self,
        unit: SectionPlan,
        rolling: RollingContext,
    ) -> list[dict]:
        """テキスト生成用のメッセージを構築 (共通スケルトンに委譲)。

        CogWriter は長期要約スロットを持たないため
        ``include_long_term_summary=False``。
        """
        return build_text_unit_messages(
            unit, rolling, _TEXT_UNIT_USER, include_long_term_summary=False,
        )

    # ── レビュー実装 ──

    async def _review_code(self, rolling: RollingContext) -> list[ReviewIssue]:
        """コード用レビュー: AST検証 + アシストによるシグネチャ整合性チェック"""
        issues: list[ReviewIssue] = []
        plan = rolling.plan

        # 1. ルールベース: 各ユニットのAST検証
        for i, code in enumerate(rolling.generated_units):
            try:
                ast.parse(code)
            except SyntaxError as e:
                issues.append(ReviewIssue(
                    unit_idx=i,
                    issue=f"SyntaxError: line {e.lineno}: {e.msg}",
                    fix=f"構文エラーを修正: {e.msg}",
                ))

        # 2. アシストモデル: シグネチャ整合性チェック
        plan_specs = []
        for u in plan.units:
            if isinstance(u, CodeUnit):
                plan_specs.append(f"- {u.name} ({u.kind}): {u.spec}")

        generated_sigs = []
        for i, code in enumerate(rolling.generated_units):
            skeleton = CodeSkeleton.extract(code)
            sigs = skeleton.function_signatures + skeleton.class_outlines
            generated_sigs.append(
                f"[unit {i}] " + "; ".join(sigs[:5]) if sigs else f"[unit {i}] (empty)"
            )

        prompt = _CODE_REVIEW_PROMPT.format(
            plan_specs="\n".join(plan_specs),
            generated_signatures="\n".join(generated_sigs),
        )

        try:
            data = await self.assist_client.generate_json(
                # 複数 ReviewIssue(unit_idx/issue/fix) を配列で返すため 512 では
                # finish_reason=length で約半数が途中切断され json_repair 依存に
                # なる。1024 へ拡張して切断を抑える。
                prompt, max_tokens=1024, temperature=0.3,
                purpose="long_form_code_review",
                list_key="issues",
            )
            review_items = data.get("issues", []) if isinstance(data, dict) else []

            for item in review_items:
                if isinstance(item, dict) and "unit_idx" in item:
                    issues.append(ReviewIssue(
                        unit_idx=item["unit_idx"],
                        issue=item.get("issue", ""),
                        fix=item.get("fix", ""),
                    ))
        except Exception as e:
            logger.warning("Assist review failed: %s", e)

        return issues

    async def _review_text(self, rolling: RollingContext) -> list[ReviewIssue]:
        """テキスト用レビュー: アシストによる要点カバー率 + 文体一貫性"""
        issues: list[ReviewIssue] = []
        plan = rolling.plan

        plan_sections = []
        for u in plan.units:
            if isinstance(u, SectionPlan):
                plan_sections.append(
                    f"- {u.heading}: {', '.join(u.key_points)}"
                )

        generated_previews = []
        for i, text in enumerate(rolling.generated_units):
            preview = text[:300]
            generated_previews.append(f"[section {i}] {preview}")

        prompt = _TEXT_REVIEW_PROMPT.format(
            plan_sections="\n".join(plan_sections),
            generated_previews="\n".join(generated_previews),
        )

        try:
            data = await self.assist_client.generate_json(
                # code_review と同様、複数 issue 配列で 512 切断が起きるため 1024。
                prompt, max_tokens=1024, temperature=0.3,
                purpose="long_form_text_review",
                list_key="issues",
            )
            review_items = data.get("issues", []) if isinstance(data, dict) else []

            for item in review_items:
                if isinstance(item, dict) and "unit_idx" in item:
                    issues.append(ReviewIssue(
                        unit_idx=item["unit_idx"],
                        issue=item.get("issue", ""),
                        fix=item.get("fix", ""),
                    ))
        except Exception as e:
            logger.warning("Assist text review failed: %s", e)

        return issues
