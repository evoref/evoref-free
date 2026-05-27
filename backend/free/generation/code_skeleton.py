"""CodeSkeleton — 正規表現によるコード構造抽出

設計書 f_09_long_form_generation.md §3.2, §8 準拠。
生成済みコードから型・シグネチャ・import 等のスケルトンを
LLM 不要のルールベースで抽出し、ローリングコンテキストに利用する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.utils import estimate_tokens

# ── 正規表現パターン ──

RE_IMPORT = re.compile(r"^(import .+|from .+ import .+)$", re.MULTILINE)

RE_FUNCTION_SIG = re.compile(
    r"^(async\s+)?def\s+\w+\([^)]*\)(\s*->\s*[^:]+)?:",
    re.MULTILINE,
)

RE_CLASS_DEF = re.compile(r"^class\s+\w+(?:\([^)]*\))?:", re.MULTILINE)

RE_TYPE_DEF = re.compile(
    r"^@dataclass.*\nclass\s+\w+|^class\s+\w+\(TypedDict\)",
    re.MULTILINE,
)

RE_CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]+\s*[:=]", re.MULTILINE)

RE_METHOD_SIG = re.compile(
    r"^    (async\s+)?def\s+\w+\([^)]*\)(\s*->\s*[^:]+)?:",
    re.MULTILINE,
)


def _get_indent(code: str, pos: int) -> int:
    """指定位置の行頭インデント（スペース数）を取得"""
    line_start = code.rfind("\n", 0, pos) + 1
    indent = 0
    for ch in code[line_start:pos]:
        if ch == " ":
            indent += 1
        elif ch == "\t":
            indent += 4
        else:
            break
    return indent


def _extract_class_outlines(code: str) -> list[str]:
    """クラスの概要（名前 + メソッドシグネチャ一覧）を抽出"""
    outlines: list[str] = []
    for match in RE_CLASS_DEF.finditer(code):
        class_line = match.group()
        class_start = match.end()

        # クラス本体の終端を推定（同レベル以下のインデントの行まで）
        _get_indent(code, match.start())
        body_end = len(code)
        for line_match in re.finditer(r"^(\S)", code[class_start:], re.MULTILINE):
            candidate_pos = class_start + line_match.start()
            # 空行は無視
            if code[candidate_pos:candidate_pos + 1].strip():
                body_end = candidate_pos
                break

        class_body = code[class_start:body_end]

        # クラス内のメソッドシグネチャを収集
        methods: list[str] = []
        for m in RE_METHOD_SIG.finditer(class_body):
            sig = m.group().strip()
            methods.append(sig)

        if methods:
            outline = class_line + "\n" + "\n".join(
                f"    {m}" for m in methods
            )
        else:
            outline = class_line
        outlines.append(outline)
    return outlines


def _deduplicate(items: list[str]) -> list[str]:
    """順序を保持しつつ重複を排除"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


@dataclass
class CodeSkeleton:
    """生成済みコードの構造情報（要約の代替）"""

    imports: list[str] = field(default_factory=list)
    type_definitions: list[str] = field(default_factory=list)
    function_signatures: list[str] = field(default_factory=list)
    class_outlines: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)

    @classmethod
    def extract(cls, code: str) -> CodeSkeleton:
        """正規表現でコードからスケルトンを抽出（LLM不要）"""
        imports = [m.group().strip() for m in RE_IMPORT.finditer(code)]
        type_defs = [m.group().strip() for m in RE_TYPE_DEF.finditer(code)]
        func_sigs = [m.group().strip() for m in RE_FUNCTION_SIG.finditer(code)]
        class_outlines = _extract_class_outlines(code)
        constants = [m.group().strip() for m in RE_CONSTANT.finditer(code)]

        # トップレベル関数のみ（クラスメソッドを除外）
        top_level_sigs: list[str] = []
        for sig in func_sigs:
            if not sig.startswith((" ", "\t")):
                top_level_sigs.append(sig)

        return cls(
            imports=imports,
            type_definitions=type_defs,
            function_signatures=top_level_sigs,
            class_outlines=class_outlines,
            constants=constants,
        )

    def to_prompt(self, budget: int) -> str:
        """予算内に収まるスケルトン文字列を生成

        優先順: imports > types > signatures > class_outlines > constants
        """
        sections: list[tuple[str, list[str]]] = [
            ("# Imports", self.imports),
            ("# Types", self.type_definitions),
            ("# Functions", self.function_signatures),
            ("# Classes", self.class_outlines),
            ("# Constants", self.constants),
        ]

        result_parts: list[str] = []
        remaining = budget

        for header, items in sections:
            if not items:
                continue

            section_text = header + "\n" + "\n".join(items)
            section_tokens = estimate_tokens(section_text)

            if section_tokens <= remaining:
                result_parts.append(section_text)
                remaining -= section_tokens
            else:
                # 予算に収まる分だけ追加
                partial_lines = [header]
                partial_tokens = estimate_tokens(header)
                for item in items:
                    item_tokens = estimate_tokens(item)
                    if partial_tokens + item_tokens + 1 <= remaining:
                        partial_lines.append(item)
                        partial_tokens += item_tokens + 1
                    else:
                        break
                if len(partial_lines) > 1:
                    result_parts.append("\n".join(partial_lines))
                    remaining -= partial_tokens
                break

        return "\n\n".join(result_parts)


def update_skeleton(skeleton: CodeSkeleton, generated_code: str) -> CodeSkeleton:
    """生成コードからスケルトンを更新（LLM不要、正規表現のみ）"""
    new_parts = CodeSkeleton.extract(generated_code)
    skeleton.imports = _deduplicate(skeleton.imports + new_parts.imports)
    skeleton.function_signatures = _deduplicate(
        skeleton.function_signatures + new_parts.function_signatures
    )
    skeleton.type_definitions = _deduplicate(
        skeleton.type_definitions + new_parts.type_definitions
    )
    skeleton.class_outlines = _deduplicate(
        skeleton.class_outlines + new_parts.class_outlines
    )
    skeleton.constants = _deduplicate(skeleton.constants + new_parts.constants)
    return skeleton
