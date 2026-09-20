"""構成テンプレート (`.outline.md`) の決定論パース (f_08 §3.1.1 / c_16 §4.5.2)

テンプレートのエントリが ``outline`` を持つとき、章立てを LLM に決めさせず
outline から決定論で組む。本モジュールは outline の Markdown をパースする
純粋関数と、生成後に固定文 (引用ブロック) の逐語一致を検査・補完する
純粋関数だけを持つ。計画 (:class:`~backend.free.generation.models
.GenerationPlan`) への変換と補助タスク呼出は
:mod:`backend.free.generation.strategy_common` 側 (aux 呼出登録済みファイル、
c_17 の purpose 監査対象) が持つ。

パース規則:
    - ``# `` : 文書タイトル (最初の 1 行のみ採用)
    - ``## `` : unit の見出し (出現順)
    - ``- `` / ``* `` : 直近の unit への key_point
    - ``> `` : 直近の unit へ逐語で入れる固定文 (連続する引用行は 1 つに連結)
    - コードフェンス (``` / ~~~) の内側は一切解釈しない

unit が 1 つも無ければ ``None`` を返す (呼出側が WARNING を出し、seed しない)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 空白 (改行・タブ含む) の正規化用。逐語一致検査は空白の違いだけ許容する
#: (f_08 §3.1.1)。
_WS_RE = re.compile(r"\s+")

_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True, slots=True)
class OutlineUnit:
    """outline 1 セクション分 (見出し + 要点 + 任意の固定文)。"""

    heading: str
    key_points: list[str] = field(default_factory=list)
    #: 引用ブロックから来た、このセクションに逐語で含めるべき固定文。
    verbatim: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedOutline:
    """`.outline.md` 全体のパース結果。"""

    title: str
    units: tuple[OutlineUnit, ...]


def parse_outline_markdown(text: str) -> ParsedOutline | None:
    """`.outline.md` の Markdown を :class:`ParsedOutline` に変換する (純粋関数)。

    unit (``## `` 見出し) が 1 つも見つからなければ ``None`` を返す。
    """
    title = ""
    headings: list[str] = []
    key_points: list[list[str]] = []
    verbatims: list[str | None] = []

    in_fence = False
    quote_buffer: list[str] = []

    def _flush_quote() -> None:
        if not quote_buffer or not headings:
            quote_buffer.clear()
            return
        sentence = " ".join(line.strip() for line in quote_buffer if line.strip())
        quote_buffer.clear()
        if not sentence:
            return
        idx = len(headings) - 1
        if verbatims[idx]:
            verbatims[idx] = f"{verbatims[idx]} {sentence}"
        else:
            verbatims[idx] = sentence

    for raw_line in text.splitlines():
        if _FENCE_RE.match(raw_line):
            in_fence = not in_fence
            _flush_quote()
            continue
        if in_fence:
            continue

        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("> "):
            quote_buffer.append(stripped[2:])
            continue
        # 引用ブロック以外の行が来たら、溜めた引用を確定する。
        _flush_quote()

        if not title and stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped.startswith("## "):
            headings.append(stripped[3:].strip())
            key_points.append([])
            verbatims.append(None)
            continue
        if stripped.startswith(("- ", "* ")) and headings:
            point = stripped[2:].strip()
            if point:
                key_points[-1].append(point)
            continue

    _flush_quote()

    if not headings:
        return None

    units = tuple(
        OutlineUnit(heading=h, key_points=list(kp), verbatim=v)
        for h, kp, v in zip(headings, key_points, verbatims, strict=True)
    )
    return ParsedOutline(title=title, units=units)


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub("", text.strip())


def ensure_verbatim_sentence(text: str, sentence: str) -> tuple[str, bool]:
    """``sentence`` が ``text`` に逐語 (空白の正規化だけ許容) で含まれるか検査する。

    含まれていなければ末尾へ決定論で差し込む (再生成しない、f_08 §3.1.1)。
    戻り値は ``(補完後テキスト, 差し込んだか)``。
    """
    needle = sentence.strip()
    if not needle:
        return text, False
    if _normalize_ws(needle) in _normalize_ws(text):
        return text, False
    separator = "" if not text or text.endswith("\n") else "\n\n"
    return f"{text}{separator}{needle}", True


__all__ = [
    "OutlineUnit",
    "ParsedOutline",
    "ensure_verbatim_sentence",
    "parse_outline_markdown",
]
