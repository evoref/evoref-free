"""agent pillar 共通プロンプトユーティリティ

EvorefLoop pillar (agent) 配下に置かれる純粋関数 + dataclass 群。
以前は ``learning.fewshot_pool`` / ``optimizer.prompt_evolver`` に
散在しており、agent.prompt_manager が関数内 lazy import で参照していた。

4 pillar アーキテクチャ の依存方向 (EvorefLearn → EvorefLoop) を
素直に満たすため、Learn pillar 側 (``fewshot_pool`` / ``prompt_evolver`` /
``scheduler``) から agent.prompt_utils を import する構図に統一する。

提供するもの:
- ``FewShotExample``: Few-shot 候補 1 件を表す純粋 dataclass
- ``format_fewshot_section``: Few-shot 例をプロンプト末尾用セクションに整形
- ``extract_protected_sections`` / ``has_orphan_protected_markers`` /
  ``strip_orphan_protected_markers``: PROTECTED マーカー検出・除去
- ``dedupe_paragraphs`` / ``text_contains_sentence``: 段落重複排除・包含判定
- ``validate_protected_sections`` / ``restore_protected_sections``:
  保護セクションの検証と強制復元

いずれも副作用のない純粋関数 / dataclass であり、他 pillar (Learn / Mem /
Gen) から自由に import してよい。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ─────────────────────────────────────────────────────────────────────
# FewShotExample / format_fewshot_section
# ─────────────────────────────────────────────────────────────────────


@dataclass
class FewShotExample:
    """Few-shot 候補の 1 エントリ (純粋 dataclass)"""

    id: str = ""
    query: str = ""
    response: str = ""
    mode: str = "chat"
    fitness: float = 0.0
    added_at: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]


@runtime_checkable
class FewShotSelector(Protocol):
    """query 依存で few-shot 例を選ぶ最小 API。

    実装は EvorefLearn の ``FewShotPool``。``SystemPromptManager`` (EvorefLoop) が
    ``FewShotPool`` を直接 import すると pillar 境界 (Loop→Learn) を侵すため、
    Protocol を Loop 所有の本モジュールに置き、wire 時に実体を注入する
    (LoopWriteAPIProtocol と同様式)。``FewShotExample`` も本モジュール所有なので
    境界はクリーン。
    """

    def select_top_k(
        self, mode: str, query: str, k: int = 3,
    ) -> list[FewShotExample]: ...


def format_fewshot_section(examples: list[FewShotExample]) -> str:
    """Few-shot 例をシステムプロンプトに埋め込むテキストにフォーマットする"""
    if not examples:
        return ""
    lines = ["\n## Few-shot Examples\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}")
        lines.append(f"User: {ex.query}")
        lines.append(f"Assistant: {ex.response}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# PROTECTED セクション (マーカー定数 + 検出・除去 util)
# ─────────────────────────────────────────────────────────────────────

_PROTECTED_OPEN = "<!-- PROTECTED -->"
_PROTECTED_CLOSE = "<!-- /PROTECTED -->"
_PROTECTED_RE = re.compile(
    re.escape(_PROTECTED_OPEN) + r"\n?(.*?)\n?" + re.escape(_PROTECTED_CLOSE),
    re.DOTALL,
)
_ANY_PROTECTED_MARKER_RE = re.compile(
    re.escape(_PROTECTED_OPEN) + r"|" + re.escape(_PROTECTED_CLOSE),
)


def extract_protected_sections(text: str) -> list[str]:
    """プロンプトから保護セクションの内容を順序付きで抽出する"""
    return [m.group(1).strip() for m in _PROTECTED_RE.finditer(text)]


def has_orphan_protected_markers(text: str) -> bool:
    """ペアになっていない `<!-- PROTECTED -->` / `<!-- /PROTECTED -->` が残っているか判定"""
    paired_spans = [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]
    for marker in _ANY_PROTECTED_MARKER_RE.finditer(text):
        pos = marker.start()
        if not any(s <= pos < e for s, e in paired_spans):
            return True
    return False


def strip_orphan_protected_markers(text: str) -> str:
    """ペアを形成していない保護マーカーを除去する

    LLM 変異や交叉によって `<!-- /PROTECTED -->` だけが取り残されるなど、
    ペアが壊れたマーカーが残るとプロンプト表示が崩れるため、
    `restore_protected_sections` 適用前後のクリーンアップに使う。
    """
    if not text:
        return text
    paired_spans = [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]
    orphan_spans: list[tuple[int, int]] = []
    for marker in _ANY_PROTECTED_MARKER_RE.finditer(text):
        pos = marker.start()
        if not any(s <= pos < e for s, e in paired_spans):
            orphan_spans.append((marker.start(), marker.end()))
    if not orphan_spans:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end in orphan_spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    stripped = "".join(parts)
    # マーカー除去で生じた連続空行を折りたたむ
    return re.sub(r"\n{3,}", "\n\n", stripped)


# ─────────────────────────────────────────────────────────────────────
# 段落・文の正規化と重複検出
# ─────────────────────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    """重複判定用に空白・大小文字を正規化する"""
    return _WS_RE.sub(" ", text).strip().lower()


def dedupe_paragraphs(text: str) -> str:
    """連続/非連続の重複段落を除去する (出現順を保持)

    保護セクション (`<!-- PROTECTED --> ... <!-- /PROTECTED -->`) は段落として
    扱い、内容ベースの重複判定対象に含める (同一保護セクションの二重貼り付けも防ぐ)。
    """
    if not text:
        return text
    paragraphs = re.split(r"\n{2,}", text)
    seen: set[str] = set()
    out: list[str] = []
    for p in paragraphs:
        norm = _normalize_for_dedup(p)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(p.strip())
    return "\n\n".join(out)


def text_contains_sentence(text: str, sentence: str) -> bool:
    """text が sentence を (正規化したうえで) 部分文字列として含むか判定"""
    n_text = _normalize_for_dedup(text)
    n_sent = _normalize_for_dedup(sentence)
    if not n_sent:
        return True
    return n_sent in n_text


# ─────────────────────────────────────────────────────────────────────
# 保護セクションの検証・強制復元
# ─────────────────────────────────────────────────────────────────────


def validate_protected_sections(original: str, candidate: str) -> bool:
    """candidate が original の保護セクションを全て原文どおり含むか検証する

    孤児マーカー (ペアになっていない `<!-- PROTECTED -->` 単独 or
    `<!-- /PROTECTED -->` 単独) が candidate に残っている場合も無効とする。
    """
    orig_sections = extract_protected_sections(original)
    if not orig_sections:
        # 保護セクションなし → 元は孤児マーカーがなければ常に有効
        return not has_orphan_protected_markers(candidate)
    cand_sections = extract_protected_sections(candidate)
    if orig_sections != cand_sections:
        return False
    return not has_orphan_protected_markers(candidate)


def restore_protected_sections(original: str, candidate: str) -> str:
    """candidate 内の保護セクションを original の内容で強制復元する

    - candidate に保護マーカーが残っている場合: 中身を original の内容で上書き
    - candidate から保護マーカーが消えている場合: 末尾に追記
    - 孤児マーカー (ペア不成立の単独マーカー) は事前に除去してから処理
    """
    # まず孤児マーカーを除去して paired マッチを信頼できる状態にする
    candidate = strip_orphan_protected_markers(candidate)

    orig_sections = extract_protected_sections(original)
    if not orig_sections:
        return candidate  # 保護対象なし

    cand_matches = list(_PROTECTED_RE.finditer(candidate))

    if not cand_matches:
        # マーカーが全て消失 → 末尾に全保護セクションを追記
        blocks = "\n\n".join(
            f"{_PROTECTED_OPEN}\n{s}\n{_PROTECTED_CLOSE}" for s in orig_sections
        )
        return candidate.rstrip() + "\n\n" + blocks

    # マーカーが残っている場合: 中身を上書き (逆順で位置を壊さない)
    result = candidate
    for i, match in reversed(list(enumerate(cand_matches))):
        if i < len(orig_sections):
            replacement = f"{_PROTECTED_OPEN}\n{orig_sections[i]}\n{_PROTECTED_CLOSE}"
            result = result[:match.start()] + replacement + result[match.end():]

    # candidate のマーカー数が不足している場合: 残りを末尾に追記
    if len(cand_matches) < len(orig_sections):
        extra = orig_sections[len(cand_matches):]
        blocks = "\n\n".join(
            f"{_PROTECTED_OPEN}\n{s}\n{_PROTECTED_CLOSE}" for s in extra
        )
        result = result.rstrip() + "\n\n" + blocks

    return result


__all__ = [
    "FewShotExample",
    "FewShotSelector",
    "dedupe_paragraphs",
    "extract_protected_sections",
    "format_fewshot_section",
    "has_orphan_protected_markers",
    "restore_protected_sections",
    "strip_orphan_protected_markers",
    "text_contains_sentence",
    "validate_protected_sections",
]
