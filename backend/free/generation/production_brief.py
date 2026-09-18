"""制作段 (long_form / staged) へ渡す不変の文脈ブロック — ProductionBrief (f_08 §2.2)。

create モードの長時間ターンは、チャット応答パスが持つ文脈 (fact slate /
SemMem 注入 / 直前成果物 / RAG / 添付 / ProjectMap) を 1 つも継承していなかった
(2026-09-18 監査)。ProductionBrief はターン入口で **1 回だけ決定論で組み**、
ターン中は byte 不変のテキストブロックとして、以後の全 LLM 呼出のプロンプト
先頭に同じ bytes を置く。材料は呼出側が既に計算済みのもの — ここでは新しい
検索・生成はしない (ProジェクトMap の neighborhood だけは決定論で追加、ms 級)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from backend.utils import estimate_tokens

#: 節の内部キー→見出しの対応。この順で組み立て、総予算超過時は逆順
#: (code_map → attachments → references → prior_work → memory → facts) で削る。
_SECTION_HEADINGS: tuple[tuple[str, str], ...] = (
    ("facts", "Facts"),
    ("memory", "Memory"),
    ("prior_work", "Prior work"),
    ("references", "References"),
    ("attachments", "Attachments"),
    ("code_map", "Code map"),
)


@dataclass(frozen=True)
class BriefLimits:
    """ProductionBrief の予算 (f_08 §2.2)。既定値は ``create.brief`` (config.yaml)
    と揃える。呼出側 (chat.py) が cfg から組んで渡す。"""

    max_tokens: int = 2000
    facts: int = 200
    memory: int = 600
    prior_work: int = 400
    references: int = 400
    attachments: int = 400
    code_map: int = 400

    def budget(self, key: str) -> int:
        return getattr(self, key)


def _truncate_tail(text: str, budget_tokens: int) -> str:
    """先頭を残し、予算 (推定トークン数) を超える手前で末尾を切る (末尾切り)。"""
    text = (text or "").strip()
    if not text or budget_tokens <= 0:
        return ""
    if estimate_tokens(text) <= budget_tokens:
        return text
    # 二分探索で予算に収まる最大長の prefix を求める (CJK/ASCII 混在は文字数に
    # 比例しないため単純な切り詰めでは合わない)。
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip()


def _render_references(chunks: list[tuple[str, float, str]]) -> str:
    """RAG 採用チャンク (chunk_id, salience, content) を References 節の本文へ整形する。"""
    parts: list[str] = []
    for chunk_id, _score, content in chunks:
        body = (content or "").strip()
        if not body:
            continue
        parts.append(f"[{chunk_id}]\n{body}" if chunk_id else body)
    return "\n\n".join(parts)


def build_production_brief(
    *,
    fact_slate: str = "",
    semmem_block: str = "",
    prior_work: str = "",
    rag_chunks: list[tuple[str, float, str]] | None = None,
    file_block: str = "",
    code_map: str = "",
    limits: BriefLimits | None = None,
) -> str:
    """制作段への不変ブリーフを組み立てる (f_08 §2.2)。

    材料は全て呼出側が既に計算済みのもの (fact slate / SemMem 注入 / 直前
    成果物 / RAG チャンク / 添付ブロック / ProjectMap 近傍)。各節は節別予算
    (``limits``) で末尾切りし、空の節は見出しごと省く。総予算
    (``limits.max_tokens``) を超える場合は code_map → attachments →
    references → prior_work → memory → facts の順で節ごと削る。全節が空なら
    ``""``。純粋関数 (同じ入力から常に同じ bytes) — 新しい検索・生成はしない。
    """
    limits = limits or BriefLimits()
    materials = {
        "facts": fact_slate,
        "memory": semmem_block,
        "prior_work": prior_work,
        "references": _render_references(rag_chunks or []),
        "attachments": file_block,
        "code_map": code_map,
    }
    sections: dict[str, str] = {}
    for key, _heading in _SECTION_HEADINGS:
        text = _truncate_tail(materials.get(key) or "", limits.budget(key))
        if text:
            sections[key] = text

    def _total_tokens() -> int:
        return sum(
            estimate_tokens(f"## {heading}\n{sections[key]}")
            for key, heading in _SECTION_HEADINGS
            if key in sections
        )

    for key, _heading in reversed(_SECTION_HEADINGS):
        if _total_tokens() <= limits.max_tokens:
            break
        sections.pop(key, None)

    if not sections:
        return ""

    lines = ["# Project brief"]
    for key, heading in _SECTION_HEADINGS:
        if key in sections:
            lines.append(f"## {heading}\n{sections[key]}")
    lines.append("---")
    return "\n\n".join(lines)


# ── Code map (ProjectMap 近傍) ────────────────────────────────────────────

#: 要求文中のファイルパスを拾う正規表現。``backend.free.agent.tool_judge_signals.
#: _CODE_PATH_TOKEN_RE`` と同等 (EvorefGen は完全独立の pillar のため import
#: せず複製する、CLAUDE.md §3)。
_PATH_TOKEN_RE = re.compile(
    r"(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)+\.[A-Za-z0-9]{1,6}",
)
#: ASCII 識別子候補 (3 文字以上)。
_IDENT_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

#: 引く候補の最大件数 (パス優先)。
_MAX_CANDIDATES = 3
#: 1 対象あたりの neighborhood 描画予算。節全体の最終予算は
#: ``build_production_brief`` (``BriefLimits.code_map``) の末尾切りが締める。
_NEIGHBORHOOD_BUDGET_TOKENS = 400

#: 完全一致した function/method/class の本文を足す上限行数 (2026-09-18)。
#: 署名の近傍だけでは実装の規則 (区切り文字の具体値等) が伝わらない
#: (実機: chunk_evidence_id と同じ規則で、と頼んでも "\x00".join が "".join に
#: なった)。総量は build_production_brief の code_map 予算内で末尾切りされる。
_BODY_MAX_LINES = 40


def _node_map_root(reader: object) -> Path | None:
    """``reader`` (単独 ``ProjectMapReader`` / ``MultiProjectMapReader``) の
    ソース読み出し起点を返す。Multi は先頭 reader の root を使う (query.py の
    ``MultiProjectMapReader.readers`` と同じ両対応)。"""
    sub_readers = getattr(reader, "readers", None)
    if sub_readers:
        return getattr(sub_readers[0], "root", None)
    return getattr(reader, "root", None)


def _read_node_body(reader: object, node: object) -> str:
    """完全一致ノードの本文を ``line_start``〜``line_end`` (最大 40 行) で読む。

    function / method / class のみ呼ばれる想定。読めなければ (root 不明 /
    ファイル不在 / 行範囲不正) 空文字を返し、呼出側は節ごと省く。
    """
    root = _node_map_root(reader)
    if root is None:
        return ""
    path = str(getattr(node, "path", "") or "")
    line_start = int(getattr(node, "line_start", 0) or 0)
    line_end = int(getattr(node, "line_end", 0) or 0)
    if not path or line_start <= 0 or line_end < line_start:
        return ""
    try:
        text = (Path(root) / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    if line_start > len(lines):
        return ""
    end = min(line_end, line_start + _BODY_MAX_LINES - 1, len(lines))
    body = "\n".join(lines[line_start - 1:end]).rstrip()
    return f"```\n{body}\n```" if body else ""


def _code_map_candidates(query: str) -> list[str]:
    """要求文からパス優先で最大 ``_MAX_CANDIDATES`` 件の候補を抽出する。"""
    paths: list[str] = []
    for m in _PATH_TOKEN_RE.finditer(query):
        tok = m.group(0)
        if tok not in paths:
            paths.append(tok)
    masked = _PATH_TOKEN_RE.sub(" ", query)
    idents: list[str] = []
    for m in _IDENT_TOKEN_RE.finditer(masked):
        tok = m.group(0)
        if tok not in idents and tok not in paths:
            idents.append(tok)
    return (paths + idents)[:_MAX_CANDIDATES]


def build_code_map_block(query: str, reader: object) -> str:
    """要求文中のパス/識別子を ProjectMap で引き、完全一致対象の近傍を並べる。

    ``reader`` は ``state.project_map_reader_getter()`` の戻り値 (未構築 /
    無効なら ``None``)。reader 障害はチャットを止めないよう握って空文字にする
    (c_16 §4.4 と同じ方針)。完全一致 (name / qualname / path) した候補のみ採用し、
    前方一致・無関係な候補は無視する。
    """
    if reader is None:
        return ""
    blocks: list[str] = []
    seen_targets: set[str] = set()
    for candidate in _code_map_candidates(query):
        try:
            nodes = reader.lookup(candidate)
        except Exception:
            continue
        if not nodes:
            continue
        top = nodes[0]
        exact = (
            top.name == candidate or top.qualname == candidate
            or top.path == candidate or top.path.endswith("/" + candidate)
        )
        if not exact:
            continue
        target = top.path if top.node_type == "file" else (top.qualname or top.name)
        if not target or target in seen_targets:
            continue
        seen_targets.add(target)
        try:
            text = reader.neighborhood(
                target, depth=1, budget_tokens=_NEIGHBORHOOD_BUDGET_TOKENS,
            )
        except Exception:
            continue
        if text:
            blocks.append(text)
        if top.node_type in ("function", "method", "class"):
            body = _read_node_body(reader, top)
            if body:
                blocks.append(body)
    return "\n\n".join(blocks)
