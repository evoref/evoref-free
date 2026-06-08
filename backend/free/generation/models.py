"""長文生成エンジンのデータモデル

設計書 f_09_long_form_generation.md §3 準拠。
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
