"""文法制約 JSON によるツール選択の変換層。

``ToolsRegistry`` を ``response_format`` (json_schema) の **enum 分類問題**へ
落とし込み、応答をパースして ``(tool_name, args)`` に戻す。:mod:`native_tools`
(OAI ``tools`` 変換) の置き換えで、責務は同じく **変換とパースだけ**。mode ゲート・
引数グラウンディング・readonly 拒否は呼出側 (``ToolCallJudge._finalize``) の
既存 funnel が担う。

**なぜ OAI tool calling ではないか** (2026-08-12 実機実測):

- ``tools`` を渡しても、モデルは tool_call を出さずに **回答本文を書き始める**
  ことがある。``max_tokens`` を使い切って 15.6〜60.2 秒を捨てる
  (本番ログにも ``Native tool calling exhausted max_tokens=256 without a
  tool call`` が実在)。
- ``tool_choice: "required"`` でも強制されない — 6 件中 3 件で無視された。
- 一方 ``response_format`` の json_schema は llama-server 側の GBNF 制約なので
  **必ずスキーマに従い、出力トークン数の上限が読める**。20 ケースのベンチで
  Qwen3.5-27B が 19/20・平均 20 トークン、gemma-4-12b が 20/20・平均 26 トークン。
  decode が支配的な環境ではレイテンシの予測可能性がそのまま体感差になる。

引数は ``{"tool": <enum>, "arg": <string>}`` の 1 スロットで受け取り、ツールの
**主引数** (必須引数の先頭、無ければ最初の引数) へ割り当てる。ツールごとに
異なる構造化引数を LLM に組ませると出力トークンが伸び、壊れやすくなるため。
複数引数が要るツールは既存の決定論層 / 呼出側の補完に委ねる。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from backend.free.core.locale_patterns import select_locale_variant
from backend.free.llm.json_extract import extract_json_object
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.agent.tools_registry import ToolsRegistry

logger = get_logger("agent.grammar_tool_classifier")

#: 「ツール不要」を表す enum 値。ツール名と衝突しない語を使う。
NO_TOOL = "none"

#: 分類応答の最大トークン。``{"tool": "...", "arg": "..."}`` の典型は実測 16〜44
#: トークンだが、``arg`` にコマンドや式が入る ``run_command_readonly`` /
#: ``calculate`` はその数倍に伸びる。上限は「そこで打ち切る天井」であって
#: 生成量の目標ではないため、短い応答の遅延には影響しない。逆に低すぎると
#: JSON が途中で切れ、``parse_classifier_response`` が None を返してツール
#: 呼び出しごと黙って消える。
#:
#: 実インシデント (2026-08-14 ライブ監査 ターン23): 「正規表現を実際に検証して
#: から提示して」に対しモデルは calculate へ検証コードを渡そうとしたが、
#: ``Constrained generation hit max_tokens=96`` で JSON が途中終了し
#: (``{"tool": "calculate", "arg": "import re; pattern = ...`` で切断)、
#: 分類結果が丸ごと破棄された。検証は一度も走らず、モデルは未検証の
#: 「一致しました」を捏造した。
CLASSIFY_MAX_TOKENS = 256

#: 分類器 (層 5.9) の候補に載せないツール。``draft_document`` の出力は回答そのもの
#: (同じベースモデルが書く下書き) で、取得した事実ではない (deliberative の
#: ``_GENERATED_DRAFT_TOOLS``)。選ばれると下書きの生成 (max 900 トークン) が 1 回
#: 余分に走り、回答の段がその下書きを prefill してから書き直すので、生成が 2 回に
#: なる。2026-10-05 ライブ監査: 「旅行の持ち物リストを作ってください。」で
#: 下書き 71 秒 + 回答の prefill 1,362 トークン 64 秒、計 150 秒 (2 件とも同じ形)。
#: 回答の段は下書き無しで同じ文書を書ける (docs/c_14 §1.3)。
CLASSIFIER_EXCLUDED_TOOLS: frozenset[str] = frozenset({"draft_document"})


def available_tool_names(tools_registry: "ToolsRegistry", mode: str) -> list[str]:
    """``mode`` で実行可能な、分類器の候補にするツール名 (hidden 含む) を返す。

    ``build_oai_tools`` と同じく hidden も候補に載せる。hidden は「プロンプトの
    一覧に出さない」印であって「使わせない」印ではなく、chat の
    ``run_command_readonly`` のようにコード側が注入する前提のツールが該当する。
    :data:`CLASSIFIER_EXCLUDED_TOOLS` は載せない (メニュー・enum・応答の解釈で同じ集合)。
    """
    names = []
    for name in tools_registry.list_names():
        if name in CLASSIFIER_EXCLUDED_TOOLS:
            continue
        if not tools_registry.is_available(name, mode):
            continue
        if tools_registry.get(name) is None:
            continue
        names.append(name)
    return names


def primary_param(tools_registry: "ToolsRegistry", name: str) -> str | None:
    """ツールの主引数名。必須引数の先頭、無ければ最初の引数。

    ``parameters`` に無い必須引数 (session_id 等の非公開引数) は対象外。
    引数を取らないツールは ``None``。
    """
    tool = tools_registry.get(name)
    if tool is None:
        return None
    params = tool.parameters or {}
    if not params:
        return None
    required = [r for r in sorted(tools_registry.required_params(name)) if r in params]
    if required:
        return required[0]
    return next(iter(params), None)


def build_classifier_schema(
    tools_registry: "ToolsRegistry", mode: str,
) -> dict[str, Any] | None:
    """``response_format`` に渡す json_schema を組む。候補なしなら ``None``。"""
    names = available_tool_names(tools_registry, mode)
    if not names:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": [*names, NO_TOOL]},
                    "arg": {"type": "string"},
                },
                "required": ["tool", "arg"],
                "additionalProperties": False,
            },
        },
    }


#: メニューの locale 依存部分。役割宣言 (``_NATIVE_JUDGE_SYSTEM`` / ``_EN``) と
#: 同じ仕組み (``select_locale_variant``) で切り替える — 以前は ``none`` 行と
#: 「引数なし」の注記だけが日本語固定で、英語 UI でも混在していた。
_EMPTY_ARG_NOTE = " [arg は空文字]"
_EMPTY_ARG_NOTE_EN = " [arg is an empty string]"
_NO_TOOL_LINE = "ツールを使わず自分の知識と会話から答える"
_NO_TOOL_LINE_EN = "answer from your own knowledge and the conversation, without a tool"


def build_tool_menu(tools_registry: "ToolsRegistry", mode: str) -> str:
    """システムプロンプトへ挿す「ツール名 = 説明 (主引数)」の一覧。"""
    empty_note = select_locale_variant(_EMPTY_ARG_NOTE, _EMPTY_ARG_NOTE_EN)
    lines = []
    for name in available_tool_names(tools_registry, mode):
        tool = tools_registry.get(name)
        desc = (getattr(tool, "description", "") or "").strip().replace("\n", " ")
        param = primary_param(tools_registry, name)
        suffix = f" [arg={param}]" if param else empty_note
        lines.append(f"- {name}: {desc}{suffix}")
    lines.append(
        f"- {NO_TOOL}: {select_locale_variant(_NO_TOOL_LINE, _NO_TOOL_LINE_EN)}"
        f"{empty_note}",
    )
    return "\n".join(lines)


def parse_classifier_response(
    content: str | None,
    tools_registry: "ToolsRegistry",
    mode: str,
) -> tuple[str, dict[str, Any]] | None:
    """分類応答を ``(tool_name, args)`` に戻す。

    ``tool`` が ``none`` / 未知 / 実行不可なら ``None`` を返し、呼出側の
    既存フォールバックへ倒す。スキーマを強制しない build では非 JSON が
    返り得るため ``extract_json_object`` で救済を試みる。
    """
    if not content or not content.strip():
        return None
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        data = extract_json_object(content)
    if not isinstance(data, dict):
        logger.info("tool classifier: unparseable response: %r", content[:120])
        return None

    name = data.get("tool")
    if not isinstance(name, str) or not name or name == NO_TOOL:
        return None
    if name not in available_tool_names(tools_registry, mode):
        logger.info("tool classifier: %r is not available in mode=%s", name, mode)
        return None

    raw_arg = data.get("arg")
    arg = raw_arg.strip() if isinstance(raw_arg, str) else ""
    param = primary_param(tools_registry, name)
    args: dict[str, Any] = {}
    if param and arg:
        args[param] = arg
    return name, args


#: 式合成 (層 5.95) の応答スキーマ。分類器と違い **選択肢は無い** — 「式を
#: 書くか否か」を判断させると引き算は「不要」と判定されるため。
EXPRESSION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "arithmetic_expression",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

#: 式合成のシステムプロンプト。
#:
#: **判断させないことが要点。** 実測 (2026-08-26、Qwen3.8-27B、温度 0、各 4 回):
#:
#:   「ツールが要るか判定せよ」(calculate / no_tool) → **4/4 no_tool**
#:   「式を合成せよ」(選択肢なし)                   → **4/4 ``12-12`` で正答**
#:
#: 不足しているのは合成能力ではなく判断。分類器 (層5.9) は割り算のような
#: 「難しい」計算では calculate を選ぶが、引き算は自分で暗算できると判断し、
#: そのうえで実測 5/8 しか正答しない。
EXPRESSION_SYSTEM = (
    "あなたは数式合成器です。ユーザーの最後の質問に答えるための算術式を"
    "1 つだけ作り、JSON だけを返してください。回答本文や説明は書かないこと。\n"
    "制約:\n"
    "- expression には会話中に実際に現れた数値だけを使うこと。\n"
    "- 新しい数値を発明しないこと。\n"
    "- 式は Python の算術式として評価できる形にすること。"
)
EXPRESSION_SYSTEM_EN = (
    "You are an arithmetic expression synthesizer. Build exactly one "
    "expression that answers the user's last question and return JSON only. "
    "Do not write the reply itself.\n"
    "Constraints:\n"
    "- Use only numbers that actually appear in the conversation.\n"
    "- Never invent a number.\n"
    "- The expression must evaluate as a Python arithmetic expression."
)


#: 問いに数も指示語も無い数量の問い (被演算子をすべて会話から取る入口) にだけ足す棄権。
#: この入口は門 (取りこぼしを嫌って開く側) に頼るので、知識の問い (「東京までの新幹線は
#: いくらくらい？」) も届きうる。共通の system は変えず、指示の側に足す (docs/f_03 §3.1)。
EXPRESSION_ABSTAIN = (
    "- 会話に出た数の計算で答えられない質問 (一般知識・最新の事実・会話に無い値を問う"
    "もの) なら、expression を空文字にすること。"
)
EXPRESSION_ABSTAIN_EN = (
    "- If the question cannot be answered by computing with numbers from the "
    "conversation (general knowledge, current facts, values not in the conversation), "
    "return an empty string as the expression."
)


#: 構造の疑いで落ちた式を 1 回だけ組み直させるときの追記 (docs/f_03 §3.1)。
#: ``{issues}`` は ``expression_sanity_issues`` の文面 (正しい組み方を述べている)。
EXPRESSION_RETRY = (
    "その式は次の点で誤っている疑いがあります:\n{issues}\n"
    "指摘に従って式を 1 つだけ組み直し、JSON だけを返してください。"
    "会話中に現れた数値だけを使うこと。"
)
EXPRESSION_RETRY_EN = (
    "That expression looks wrong for these reasons:\n{issues}\n"
    "Rebuild exactly one expression following the notes and return JSON only. "
    "Use only numbers that appear in the conversation."
)


def parse_expression_response(content: str) -> str:
    """式合成の応答から ``expression`` を取り出す。取れなければ空文字。"""
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        data = extract_json_object(content)
    if not isinstance(data, dict):
        logger.info(
            "expression synthesis: unparseable response: %r", content[:120],
        )
        return ""
    expression = data.get("expression")
    return expression.strip() if isinstance(expression, str) else ""
