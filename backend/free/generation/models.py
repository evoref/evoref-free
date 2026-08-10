"""長文生成エンジンのデータモデル

設計書 f_08_long_form_generation.md §3 準拠。
CogWriter / Recurrent 両戦略で共通の計画・ユニット定義。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.free.llm.json_schemas import CodeSpec


# ── ユーティリティ ──

# ユーザー指示から目標文字数を抽出するパターン
_TARGET_CHARS_RE = re.compile(
    r"(\d[\d,]*)\s*(?:文字|字|characters?|chars?)",
    re.IGNORECASE,
)


def extract_target_chars(instruction: str, default: int = 1000) -> int:
    """ユーザー指示から目標文字数を抽出する

    「3000文字程度」「5,000字」等のパターンにマッチ。
    見つからない場合は default を返す。
    """
    m = _TARGET_CHARS_RE.search(instruction)
    if m:
        return int(m.group(1).replace(",", ""))
    return default


# 数値を伴わない「簡潔に」要求のシグナル。``_TARGET_CHARS_RE`` は「300 字以内」の
# ような数値指定しか拾えないため、これらは従来まったく計画へ伝わらなかった
# (実測 2026-07-25: 「丁寧だが冗長にならない書き方で」→ 6 ユニット 7,192 字、
#  「3 つ挙げて。箇条書きで簡潔に」→ 6 ユニット 3,264 字)。
_BREVITY_PATTERNS = (
    re.compile(r"簡潔|冗長に(?:なら|し)ない|手短|端的|要点だけ|要点のみ"),
    re.compile(r"箇条書き|箇条書|リスト形式で"),
    re.compile(r"(?:一言|ひとこと|短く|短め|コンパクト)(?:で|に)"),
    re.compile(
        r"\b(?:concise(?:ly)?|briefly|brief|succinct|terse|in\s+bullet"
        r"|bullet\s*points?|keep\s+it\s+short|tl;?dr)\b",
        re.IGNORECASE,
    ),
)

#: ブレビティ要求時に被せる目標文字数の上限。
BREVITY_TARGET_CHARS = 600

#: ブレビティ要求時、ユニット 1 個あたりに割り当てる最小文字数。ユニットは
#: 最低 200 トークン (≒330 字) 生成されるため、目標に対してユニット数が多いと
#: それ自体が冗長化の原因になる。
BREVITY_CHARS_PER_UNIT = 400


def detect_brevity_cap(instruction: str) -> int:
    """指示に数値を伴わない簡潔さ要求があれば目標文字数の上限を返す。

    無ければ ``0``。明示の文字数指定 (``extract_target_chars``) がある場合は
    そちらが優先されるため、呼出側は数値指定が無いときだけ本関数を使う。
    """
    if any(p.search(instruction) for p in _BREVITY_PATTERNS):
        return BREVITY_TARGET_CHARS
    return 0


#: ユーザーが明示した「書式」の要求。plan / unit プロンプトには本文の要点しか
#: 渡らないため、これらは従来まったく生成側へ伝わらず、散文で返っていた
#: (実測 2026-07-27: 「議事録の形式に整えて。見出しと箇条書きで」→ 見出しも
#:  箇条書きも無い 4 段落の散文)。
_BULLET_FORMAT_RE = re.compile(
    r"箇条書き|箇条書|リスト形式|リストで"
    r"|\b(?:in\s+bullet|bullet\s*points?|as\s+a\s+list)\b",
    re.IGNORECASE,
)
_TABLE_FORMAT_RE = re.compile(
    r"表形式|表にして|テーブル形式|\b(?:as\s+a\s+table|in\s+a\s+table)\b",
    re.IGNORECASE,
)

_BULLET_DIRECTIVE = (
    "ユーザーは箇条書きを指定している。本文は「- 」で始まる箇条書きを主体に構成し、"
    "説明的な段落で埋めないこと。1 項目 1 行で簡潔に書くこと。"
)
_TABLE_DIRECTIVE = (
    "ユーザーは表形式を指定している。本文は Markdown のパイプ表 "
    "(| 列 | 列 |) で構成すること。"
)


#: 「3 行で」「3 行以内」「in 3 lines」等の行数指定。文字数指定 (_TARGET_CHARS_RE)
#: にも簡潔さシグナル (_BREVITY_PATTERNS) にも掛からず、そのまま長文計画へ流れて
#: いた (実測 2026-07-27: 「案内を 3 行で書いて」→ 3 ユニットの丁寧文書)。
_LINE_LIMIT_RE = re.compile(
    r"(\d{1,2})\s*行(?:程度|以内|くらい|ぐらい)?(?:で|に|の)"
    r"|\bin\s+(\d{1,2})\s+lines?\b",
    re.IGNORECASE,
)

#: 行数指定を目標文字数へ換算する係数 (日本語 1 行 ≒ 60 字)。
LINE_LIMIT_CHARS_PER_LINE = 60


def detect_line_limit_chars(instruction: str) -> int:
    """行数指定があれば目標文字数の上限へ換算して返す。無ければ ``0``。"""
    m = _LINE_LIMIT_RE.search(instruction)
    if not m:
        return 0
    raw = m.group(1) or m.group(2)
    try:
        lines = int(raw)
    except (TypeError, ValueError):
        return 0
    if lines <= 0:
        return 0
    return lines * LINE_LIMIT_CHARS_PER_LINE


def detect_structure_directive(instruction: str) -> str:
    """指示に明示された書式要求 (箇条書き / 表) を生成側への指示文に変換する。

    要求が無ければ空文字列。表と箇条書きの両方が指定された場合は表を優先する
    (表の方が構造が強く、箇条書きは表のセル内に収まるため)。
    """
    if _TABLE_FORMAT_RE.search(instruction):
        return _TABLE_DIRECTIVE
    if _BULLET_FORMAT_RE.search(instruction):
        return _BULLET_DIRECTIVE
    return ""


def chars_to_tokens(chars: int) -> int:
    """日本語文字数からトークン数を概算する

    CJK 文字は概ね 1文字≒1トークン。安全マージンとして 0.6 倍。
    """
    return max(int(chars * 0.6), 256)


class ContentType(str, Enum):
    """生成コンテンツの種別"""

    CODE = "code"
    TEXT = "text"


class LongFormMode(str, Enum):
    """長文生成の出力モード (file_output_mode 時のサブモード)。

    :func:`backend.free.api.chat._long_form_intent.detect_long_form_mode`
    がユーザー指示から判定し、:meth:`backend.free.generation.orchestrator
    .LongFormOrchestrator.generate` の ``long_form_mode`` kwarg として
    渡される。

    - :attr:`OFF`: 長文生成パスに入らない
    - :attr:`CONTINUE`: 既存ファイルの「継続/追記」(従来動作)
    - :attr:`EXPAND`: 単一ファイル詳細化 / 機能別セクション化
    - :attr:`SPLIT`: 機能ごと個別ファイル出力
    """

    OFF = "off"
    CONTINUE = "continue"
    EXPAND = "expand"
    SPLIT = "split"


@dataclass
class CodeUnit:
    """コード生成の1単位（関数・クラス・設定ファイル等）"""

    kind: Literal["imports", "types", "class", "function", "config", "test"]
    name: str
    file_path: str
    spec: str
    depends_on: list[str]
    estimated_tokens: int
    status: Literal["pending", "generating", "done", "failed"] = "pending"


@dataclass
class SectionPlan:
    """テキスト生成の1単位"""

    heading: str
    key_points: list[str]
    estimated_tokens: int
    status: Literal["pending", "generating", "done", "failed"] = "pending"
    # SPLIT モード時に LLM が指定する出力ファイル名候補。``None`` の場合は
    # オーケストレータが連番フォールバック (例: ``unit_03``) を採用する。
    file_name: str | None = None
    # 過大ユニットの分割で生まれた続きユニットの通し番号 (0 = 親 / 分割なし)。
    # 以前は heading に「（続き2）」を埋め込んでいたため、見出し文字列が
    # プロンプトのセクション一覧経由で本文へ漏出していた (2026-07-25)。
    # 分割は構造データで持ち、heading は親と同一に保つ。
    sub_index: int = 0


@dataclass
class GenerationPlan:
    """生成計画（両戦略・両コンテンツ種別で共通の基底）"""

    content_type: ContentType
    title: str
    target_length: int
    global_context: str
    constraints: list[str]
    units: list[CodeUnit | SectionPlan] = field(default_factory=list)
    # コード生成の事前準備で合成した共有設計仕様 (contract)。CODE 経路でのみ
    # 設定され、各ユニットのプロンプトに注入 + SPEC.md として出力する。
    # 合成失敗 / TEXT 経路では None (従来挙動)。
    code_spec: CodeSpec | None = None
    # 任意の mermaid フローチャート (config code_flowchart_enabled=True 時のみ)。
    # SPEC.md に埋め込む。既定 OFF のため通常は空文字列。
    code_flowchart: str = ""
    # ユーザー指示に具体的な主題が無く、生成を進められない場合に True。
    # orchestrator.generate() はこの場合ユニット生成をスキップし、
    # clarification_question をそのまま応答として返す
    # (2026-07-22 ライブ検証で判明した長文トピック混入バグの対策)。
    needs_clarification: bool = False
    clarification_question: str = ""
