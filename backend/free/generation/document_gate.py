"""ドキュメント出力の決定論的品質ゲート。

長文生成 (:class:`backend.free.generation.orchestrator.LongFormOrchestrator`) の
レビュー後段で、生成済みセクション群を構造面から検査し ``ReviewIssue`` 列を返す。
``document_quality_enabled`` (既定 OFF) かつ TEXT のドキュメント出力先のときのみ
呼ばれる。

設計方針:
- **値の新規作成は一切しない**。検出するのは構造の欠落 (空セクション / 見出しのみ
  本文無し / 表の列ずれ / 見出し階層の飛び / 形式不適合) のみで、取得済みデータを
  LLM に捏造させないため、修正指示 (fix) も「本文に展開せよ」「列を揃えよ」等の
  構造指示に限定する。
- 解析はすべて export と同一の :class:`ContentConverter` で行う。これにより
  「組版可能性」を生成中に先取り検証し (空 .docx / 壊れた表による export 破綻の
  予防)、かつコードフェンス内の ``#`` 行を見出し誤検出する等のパーサ不一致を避ける。
"""

from __future__ import annotations

import re

from backend.export.base import ContentBlock
from backend.export.content_converter import ContentConverter
from backend.free.generation.strategy_cogwriter import ReviewIssue

# 表データを要する出力先 (表が無ければ export Writer が空ファイル化する)。
_TABLE_FORMATS = frozenset({".xlsx", ".xls", ".csv", ".ods"})
# スライド出力先 (見出し level<=2 でスライド分割される)。
_SLIDE_FORMATS = frozenset({".pptx"})
# リッチ文書出力先 (非空ブロックが必要)。
_RICH_DOC_FORMATS = frozenset({".docx", ".odt", ".odp", ".md"})
# 品質ゲート対象の全形式 (これ以外 = .txt 等は素通し)。
_DOCUMENT_FORMATS = _TABLE_FORMATS | _SLIDE_FORMATS | _RICH_DOC_FORMATS

# 見出し以外で「本文」とみなすブロック種別。
_BODY_BLOCK_TYPES = frozenset({"paragraph", "table", "list", "quote", "code"})

# 空とみなすプレースホルダ行 (句読点/装飾のみ、または TODO 等の未記入語)。
# ``ここに...`` は任意トークン吸収を避け、明示的な記入指示語のみに限定する
# (例: 「ここに来てください」のような正規の本文を誤判定しない)。
_PLACEHOLDER_RE = re.compile(
    r"^[\s\-\*_#>。、,.…|]*"
    r"(?:todo|tbd|wip|本文|ここに(?:記述|内容|本文|入力|書いてください)?|"
    r"（?続き）?|省略|\.\.\.|…)?"
    r"[\s\-\*_#>。、,.…|]*$",
    re.IGNORECASE,
)

# 1 スライドあたりの箇条書き上限 (超過は過密と判定)。
_MAX_BULLETS_PER_SLIDE = 12


def is_document_format(target_format: str | None) -> bool:
    """品質ゲートを適用すべきドキュメント出力先か判定する。"""
    return (target_format or "").lower() in _DOCUMENT_FORMATS


def evaluate_document(
    units: list[str],
    plan_headings: list[str],
    target_format: str | None,
    *,
    max_bullets_per_slide: int = _MAX_BULLETS_PER_SLIDE,
) -> list[ReviewIssue]:
    """生成済みセクション群を決定論的に検査し ``ReviewIssue`` 列を返す。

    Args:
        units: セクションごとの生成テキスト (``rolling.generated_units``)。
        plan_headings: 計画上の見出し (issue 文言と revise 指示に使う)。
        target_format: 出力先拡張子 (``.docx`` / ``.pptx`` / ``.xlsx`` 等)。
        max_bullets_per_slide: スライド過密判定の閾値。

    Returns:
        ``(unit_idx, issue, fix)`` を持つ ``ReviewIssue`` のリスト (重複除去済み)。
        改稿は呼出側 (orchestrator) が ``max_revisions`` で有界化する。
    """
    fmt = (target_format or "").lower()
    issues: list[ReviewIssue] = []
    per_unit_blocks: list[list[ContentBlock]] = []
    for i, text in enumerate(units):
        heading = plan_headings[i] if i < len(plan_headings) else ""
        if _is_placeholder(text):
            per_unit_blocks.append([])
            issues.append(ReviewIssue(
                i,
                f"セクション『{heading or i}』が空またはプレースホルダのみです",
                f"見出し『{heading}』の本文を実際に記述してください。要点を満たす"
                "具体的な内容を書き、TODO や空欄のままにしないでください。",
            ))
            continue
        blocks = ContentConverter().convert(text)
        per_unit_blocks.append(blocks)
        issues.extend(_check_heading_only(i, heading, blocks))
        issues.extend(_check_tables(i, blocks))
        if fmt in _SLIDE_FORMATS:
            issues.extend(_check_slide(i, blocks, max_bullets_per_slide))
    issues.extend(_check_heading_hierarchy(per_unit_blocks))
    issues.extend(_check_format_fit(per_unit_blocks, fmt))
    return _dedup(issues)


def _is_placeholder(text: str) -> bool:
    """全行が空/プレースホルダなら True (実内容を 1 行でも含めば False)。"""
    stripped = (text or "").strip()
    if not stripped:
        return True
    for line in stripped.splitlines():
        s = line.strip()
        if s and not _PLACEHOLDER_RE.match(s):
            return False
    return True


def _check_heading_only(
    idx: int, heading: str, blocks: list[ContentBlock],
) -> list[ReviewIssue]:
    """見出しはあるが本文ブロックが無いセクションを検出する (空 docx 予防)。"""
    if not any(b.type == "heading" for b in blocks):
        return []
    if any(
        b.type in _BODY_BLOCK_TYPES and (b.content or b.rows or b.items)
        for b in blocks
    ):
        return []
    return [ReviewIssue(
        idx,
        f"セクション『{heading or idx}』が見出しのみで本文がありません",
        f"見出し『{heading}』の下に本文 (段落 / 表 / 箇条書き) を記述してください。",
    )]


def _check_tables(idx: int, blocks: list[ContentBlock]) -> list[ReviewIssue]:
    """GFM 表の列数整合を検査する (取得データの創作はさせない)。"""
    issues: list[ReviewIssue] = []
    for block in blocks:
        if block.type != "table" or not block.rows:
            continue
        ncol = len(block.rows[0])
        for r, row in enumerate(block.rows[1:], start=2):
            if len(row) != ncol:
                issues.append(ReviewIssue(
                    idx,
                    f"表の列数が不揃いです (ヘッダ {ncol} 列、{r} 行目が {len(row)} 列)",
                    f"表の全行を {ncol} 列に揃えてください。セルの値を新規に創作せず、"
                    "欠けたセルは空欄にするか既存データの該当値で補ってください。",
                ))
                break  # 1 表につき 1 指摘で十分
    return issues


def _check_slide(
    idx: int, blocks: list[ContentBlock], max_bullets: int,
) -> list[ReviewIssue]:
    """スライド単位 (見出し level<=2 区切り) の箇条書き過密を検査する。"""
    worst = 0
    current = 0
    for b in blocks:
        if b.type == "heading" and b.level <= 2:
            worst = max(worst, current)
            current = 0
        elif b.type == "list":
            current += len(b.items)
    worst = max(worst, current)
    if worst > max_bullets:
        return [ReviewIssue(
            idx,
            f"スライドの箇条書きが過密です (最大 {worst} 項目)",
            f"1 スライドあたりの箇条書きを {max_bullets} 項目以下に絞るか、"
            "見出しで複数スライドに分割してください。内容を新規に増やさないでください。",
        )]
    return []


def _check_heading_hierarchy(
    per_unit_blocks: list[list[ContentBlock]],
) -> list[ReviewIssue]:
    """見出しレベルの飛び級 (H1 → H3 等) を検査する (フェンス内は ContentConverter
    がコードブロックとして消費するため見出し扱いされない)。"""
    issues: list[ReviewIssue] = []
    last_level = 0
    for idx, blocks in enumerate(per_unit_blocks):
        flagged = False
        for b in blocks:
            if b.type != "heading":
                continue
            level = b.level
            if last_level and level > last_level + 1 and not flagged:
                issues.append(ReviewIssue(
                    idx,
                    f"見出しレベルが飛んでいます (H{last_level} の次に H{level})",
                    f"見出し階層を 1 段ずつにしてください (H{last_level} の下は "
                    f"H{last_level + 1})。",
                ))
                flagged = True
            last_level = level
    return issues


def _check_format_fit(
    per_unit_blocks: list[list[ContentBlock]], fmt: str,
) -> list[ReviewIssue]:
    """出力先形式に必要な構造 (表 / スライド見出し / 非空本文) の有無を検査する。"""
    blocks = [b for unit in per_unit_blocks for b in unit]
    if not blocks:
        return []
    if fmt in _TABLE_FORMATS:
        if not any(b.type == "table" and b.rows for b in blocks):
            return [ReviewIssue(
                0,
                "表形式 (xlsx/csv 等) の出力先ですが表が含まれていません",
                "データを GFM 表 (| 区切り、ヘッダ + 区切り行 + データ行) で記述して"
                "ください。取得済みデータがあればそれを表に展開し、無い値は創作しない"
                "でください。",
            )]
    elif fmt in _SLIDE_FORMATS:
        if not any(b.type == "heading" and b.level <= 2 for b in blocks):
            return [ReviewIssue(
                0,
                "スライド (pptx) の出力先ですがスライド見出し (# / ##) がありません",
                "各スライドの先頭に # または ## の見出しを付けてください。",
            )]
    elif not any(
        b.type in _BODY_BLOCK_TYPES and (b.content or b.rows or b.items)
        for b in blocks
    ):
        return [ReviewIssue(
            0,
            "ドキュメント本文が空です",
            "見出しと本文を持つ実際の文書内容を生成してください。",
        )]
    return []


def _dedup(issues: list[ReviewIssue]) -> list[ReviewIssue]:
    """``(unit_idx, issue)`` で重複指摘を除去する (順序維持)。"""
    seen: set[tuple[int, str]] = set()
    out: list[ReviewIssue] = []
    for it in issues:
        key = (it.unit_idx, it.issue)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
