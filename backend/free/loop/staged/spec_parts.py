"""spec.md の構造化見出し (``## Module:`` / ``### Component:``) の決定論パーサ。

staged パイプラインの spec.md は f_10 §3 の見出し規約で構造化される:

- ``## Module: <file_path>`` — 正準ファイル一覧の file_path と一致するモジュール節
- ``### Component: <name>`` — モジュール節内のコンポーネント副見出し
  (公開 class 1 個 or 関数グループ 1 個 = 1 component)
- ``## Processing flow`` — 予約決定論見出し。flow_render が FlowSpec から
  レンダリングして挿入する節で、LLM 出力に混入した同名見出しは事前剥離する

本モジュールはこの規約に対する唯一のパーサ実装 (SSOT):

- 部分ごと生成 (``plan_file_parts``) — code 工程がコンポーネント単位で分割生成する
- spec 見直しループ (``extract_module_section`` / ``replace_module_section``) —
  test 不合格時に該当モジュール節だけを差し替える

分割・差し替えはすべて正規表現 + span 演算の決定論処理で、LLM による再計画・
fuzzy マッチは行わない (f_10 §9 禁則)。spec が規約に従っていない場合は None を
返し、呼出側が現行の単発生成 / 見直しスキップへフォールバックする。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

# 見出し規約の正規表現 (SSOT)。バッククォート囲みは任意。
_MODULE_HEADING_RE = re.compile(r"(?m)^##\s+Module:\s*`?([^`\r\n]+?)`?\s*$")
_COMPONENT_HEADING_RE = re.compile(r"(?m)^###\s+Component:\s*`?([^`\r\n]+?)`?\s*$")
_ANY_H2_RE = re.compile(r"(?m)^##\s")
_CANONICAL_HEADING_RE = re.compile(r"(?m)^##\s+Canonical file list")
_FLOW_HEADING_RE = re.compile(r"(?m)^##\s+Processing flow\b[^\r\n]*$")
_ENTRY_HEADING_RE = re.compile(r"(?m)^##\s+Entry point\b")
# synthesizer._render_module_list の行形式 (`- <path>: <purpose>`) から path を復元
_MODULE_LIST_LINE_RE = re.compile(r"(?m)^-\s+(\S+?):\s")
# 同行末尾の注釈 (`[components: ...]` / `(depends on: ...)`) を purpose から除去
_MODULE_LIST_ANNOTATION_RE = re.compile(
    r"\s*(?:\[components:[^\]]*\]|\(depends on:[^)]*\))",
)


@dataclass(frozen=True)
class ComponentGroup:
    """1 回の部分生成が担当する component 群 (通常 1 個、上限超過時は併合)。"""

    label: str
    spec_text: str  # ``### Component:`` 見出し行を含む節全文 (無改変)


@dataclass(frozen=True)
class FilePartsPlan:
    """1 ファイルの部分生成計画 (決定論分割の結果)。"""

    module_section: str  # ``## Module:`` 節全文 (preamble + 全 component、無改変)
    canonical_list: str  # spec 末尾の ``## Canonical file list`` 節 (無ければ "")
    groups: list[ComponentGroup]  # len >= 2 保証


def _basename(path: str) -> str:
    """OS 非依存の basename (生成物パスは posix 前提だが backslash も許容)。"""
    return PurePosixPath(path.replace("\\", "/")).name


def _section_end(spec: str, heading_end: int) -> int:
    """見出し行末尾以降で次の H2 見出しが始まる位置 (無ければ文字列末尾)。"""
    nxt = _ANY_H2_RE.search(spec, heading_end)
    return nxt.start() if nxt else len(spec)


def extract_module_section(
    spec: str, file_path: str,
) -> tuple[int, int, str] | None:
    """``## Module: <file_path>`` 節の (start, end, text) を返す。

    1st pass: 捕捉パスと ``file_path`` の完全一致。
    2nd pass: basename 一致 (**一意のときのみ**。複数一致は曖昧なので None)。
    節の終端は次の H2 見出しまたは文字列末尾。見つからなければ None。
    """
    matches = list(_MODULE_HEADING_RE.finditer(spec))
    if not matches:
        return None

    exact = [m for m in matches if m.group(1).strip() == file_path]
    if not exact:
        by_name = [
            m for m in matches
            if _basename(m.group(1).strip()) == _basename(file_path)
        ]
        if len(by_name) != 1:
            return None
        exact = by_name

    m = exact[0]
    start, end = m.start(), _section_end(spec, m.end())
    return start, end, spec[start:end]


def replace_module_section(
    spec: str, file_path: str, new_section: str,
) -> str | None:
    """モジュール節を ``new_section`` で span 差し替えした spec を返す。

    抽出不能 (アンカー不在・曖昧) なら None。fuzzy 置換は行わない。
    """
    found = extract_module_section(spec, file_path)
    if found is None:
        return None
    start, end, _ = found
    replacement = new_section.strip() + "\n\n"
    return spec[:start] + replacement + spec[end:].lstrip("\n")


def merge_foreign_module_sections(
    spec: str, canonical_paths: list[str],
) -> tuple[str, int]:
    """正準に無い幻覚 ``## Module:`` 節を正準モジュール節へ決定論的に移送する。

    spec 生成 LLM は「ファイルを追加するな」の指示を無視して正準に無い
    ``## Module: game.py`` のような節を発明することがあり、本来正準モジュールに
    属すべき ``### Component:`` がそちらへ置かれると部分生成が不発になる
    (2026-07-06 live: 1 モジュール構成で component が泣き別れ)。

    **正準モジュールが 1 つだけの場合のみ**、幻覚節の本文 (見出し行を除く全文)
    を正準節の末尾へ移送する (テキスト移動のみ・LLM 不使用)。複数正準時は移送先
    が曖昧なため no-op。正準節が存在しない場合も no-op (先に
    :func:`ensure_module_sections` で補完しておくこと)。

    Returns:
        (処理後 spec, 移送した幻覚節の数)。
    """
    if len(canonical_paths) != 1:
        return spec, 0
    canonical = canonical_paths[0]
    if extract_module_section(spec, canonical) is None:
        return spec, 0

    canonical_base = _basename(canonical)
    foreign_spans: list[tuple[int, int, str]] = []
    for m in _MODULE_HEADING_RE.finditer(spec):
        captured = m.group(1).strip()
        if captured == canonical or _basename(captured) == canonical_base:
            continue
        end = _section_end(spec, m.end())
        body = spec[m.end():end].strip("\n")
        foreign_spans.append((m.start(), end, body))
    if not foreign_spans:
        return spec, 0

    # 幻覚節を後ろから除去 (span がずれないように)
    stripped = spec
    for start, end, _body in reversed(foreign_spans):
        stripped = stripped[:start] + stripped[end:]

    found = extract_module_section(stripped, canonical)
    if found is None:  # 除去で正準節が壊れることはないはずだが防御的に
        return spec, 0
    _, c_end, _ = found
    moved = "\n\n".join(body for _s, _e, body in foreign_spans if body.strip())
    merged = (
        stripped[:c_end].rstrip("\n")
        + ("\n\n" + moved + "\n\n" if moved else "\n\n")
        + stripped[c_end:].lstrip("\n")
    )
    return merged, len(foreign_spans)


def ensure_module_sections(
    spec: str, module_paths: list[str],
) -> tuple[str, int]:
    """欠落しているモジュール節をプレースホルダで決定的に補完する。

    LLM の自由記述が一部モジュールの ``## Module:`` 見出しを落としても、
    アンカー総数 = 正準モジュール数を保証する (部分生成・見直しループの前提)。

    Returns:
        (補完後 spec, 元から見出しが存在したモジュール数)。
    """
    found = 0
    additions: list[str] = []
    for path in module_paths:
        if extract_module_section(spec, path) is not None:
            found += 1
        else:
            additions.append(
                f"## Module: {path}\n"
                f"(No detailed section was generated — implement per the "
                f"Overview and the canonical file list.)\n"
            )
    if additions:
        spec = spec.rstrip() + "\n\n" + "\n".join(additions)
    return spec, found


def count_component_headings(text: str) -> int:
    """``### Component:`` 見出しの個数を返す (spec 改訂節の劣化ガード用)。"""
    return len(_COMPONENT_HEADING_RE.findall(text))


def component_heading_names(text: str) -> list[str]:
    """``### Component:`` 見出し名を出現順で返す (深化/改訂ガードの名前集合用)。

    count だけの既存ガードはリネーム (数同一で名前が変わる構造劣化) を
    素通りさせるため、名前集合での照合に使う。
    """
    return [m.group(1).strip() for m in _COMPONENT_HEADING_RE.finditer(text)]


def iter_component_sections(section: str) -> list[tuple[str, str]]:
    """モジュール節内の (component 名, component 節本文) を出現順で返す。

    本文 span は ``### Component:`` 見出し行から次の component 見出し
    または節末尾まで (見出し行を含む)。宣言契約パーサ
    (``spec_contract.parse_declared_contract``) の走査単位。
    """
    matches = list(_COMPONENT_HEADING_RE.finditer(section))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(section)
        )
        out.append((m.group(1).strip(), section[m.start():end]))
    return out


def count_h2_headings(text: str) -> int:
    """H2 見出し (``## ``) の個数を返す (改訂 Entry point 節の混入ガード用)。"""
    return len(_ANY_H2_RE.findall(text))


def internal_contract_names(spec: str) -> frozenset[str]:
    """spec が宣言するプログラム内部の契約名 (Component 名 + 正準モジュール stem)。

    幻覚内部 import (`from game import Game` — Game は spec の Component だが
    game.py は正準に無い) を「外部依存の未インストール」と誤分類しないための
    判定根拠。smoke (`run_import_smoke` の internal_names) と pytest の
    env-skip 判定 (`StagedTestRunner`) が共用する。照合は使用側で
    case-insensitive に行う (class Game → module game の幻覚変換を拾うため)。
    """
    names = {m.group(1).strip() for m in _COMPONENT_HEADING_RE.finditer(spec)}
    for path in module_paths_from_list(canonical_module_list(spec)):
        stem = _basename(path).rsplit(".", 1)[0]
        names.add(stem)
    return frozenset(n for n in names if n)


def replace_entry_point_section(spec: str, new_section: str) -> str | None:
    """``## Entry point`` 節を span 差し替えした spec を返す (節が無ければ None)。

    spec 見直しループが Module 節の改訂と同時に、起動列へ残った矛盾記述
    (改訂で撤去した方式への言及等) を正すための随伴差し替え。fuzzy 置換は
    行わない。
    """
    m = _ENTRY_HEADING_RE.search(spec)
    if m is None:
        return None
    end = _section_end(spec, m.end())
    return (
        spec[:m.start()] + new_section.strip() + "\n\n" + spec[end:].lstrip("\n")
    )


def module_paths_from_list(module_list: str) -> list[str]:
    """正準モジュール一覧テキスト (``- <path>: <purpose>`` 行) から path を復元。"""
    return [m.group(1) for m in _MODULE_LIST_LINE_RE.finditer(module_list)]


def module_entries_from_list(module_list: str) -> list[tuple[str, str]]:
    """正準モジュール一覧から (path, 注釈除去済み purpose 先頭行) を復元する。

    ``flow_render.fallback_flow`` の入力。1 行目末尾の ``[components: ...]`` /
    ``(depends on: ...)`` 注釈は除去し、継続行 (複数行 purpose) は含めない。
    """
    entries: list[tuple[str, str]] = []
    for m in _MODULE_LIST_LINE_RE.finditer(module_list):
        line_end = module_list.find("\n", m.end())
        if line_end == -1:
            line_end = len(module_list)
        head = _MODULE_LIST_ANNOTATION_RE.sub(
            "", module_list[m.end():line_end],
        ).strip()
        entries.append((m.group(1), head))
    return entries


def extract_flow_section(spec: str) -> tuple[int, int, str] | None:
    """``## Processing flow`` 節の (start, end, text) を返す (無ければ None)。"""
    m = _FLOW_HEADING_RE.search(spec)
    if m is None:
        return None
    start, end = m.start(), _section_end(spec, m.end())
    return start, end, spec[start:end]


def replace_flow_section(spec: str, new_section: str) -> str:
    """``## Processing flow`` 節を差し替え / 挿入 / 除去した spec を返す。

    - 既存節あり: span 差し替え (``new_section`` が空なら節ごと除去)
    - 既存節なし + ``new_section`` あり: ``## Canonical file list`` 見出しの
      直前へ挿入 (Canonical 節が無ければ末尾へ追記)
    - 既存節なし + ``new_section`` 空: no-op
    """
    body = new_section.strip()
    found = extract_flow_section(spec)
    if found is not None:
        start, end, _ = found
        head = spec[:start].rstrip("\n")
        tail = spec[end:].lstrip("\n")
        parts = [p for p in (head, body, tail) if p]
        return "\n\n".join(parts) + "\n" if parts else ""
    if not body:
        return spec
    m = _CANONICAL_HEADING_RE.search(spec)
    if m is None:
        return spec.rstrip("\n") + "\n\n" + body + "\n"
    head = spec[:m.start()].rstrip("\n")
    prefix = head + "\n\n" if head else ""
    return prefix + body + "\n\n" + spec[m.start():]


def extract_entry_point_section(spec: str) -> str:
    """``## Entry point`` 節の全文を返す (無ければ "")。

    フロー合成入力が予算超過する際の優先注入ブロック (起動列はフロー抽出の
    最重要根拠のため、spec 先頭の切詰めから独立して常に渡す)。
    """
    m = _ENTRY_HEADING_RE.search(spec)
    if m is None:
        return ""
    return spec[m.start():_section_end(spec, m.end())]


def _extract_canonical_list(spec: str) -> str:
    """spec 末尾の ``## Canonical file list`` 節を返す (無ければ "")。"""
    m = _CANONICAL_HEADING_RE.search(spec)
    if m is None:
        return ""
    return spec[m.start():_section_end(spec, m.end())]


def canonical_module_list(spec: str) -> str:
    """``## Canonical file list`` 節の本文 (モジュール一覧行) を返す。

    フロー再合成の入力 (``_synthesize_flow_steps`` の module_list) に使う。
    節が無ければ ""。
    """
    section = _extract_canonical_list(spec)
    if not section:
        return ""
    _, _, body = section.partition("\n")
    return body.strip()


def foreign_module_headings(section: str, file_path: str) -> list[str]:
    """節テキストに含まれる「自分以外の ``## Module:`` 見出し」のパスを返す。

    spec 見直しの改訂節が他モジュールの節を巻き込んでいないかの決定論ガード。
    自モジュール (完全一致 or basename 一致) は対象外。
    """
    own = _basename(file_path)
    out: list[str] = []
    for m in _MODULE_HEADING_RE.finditer(section):
        captured = m.group(1).strip()
        if captured == file_path or _basename(captured) == own:
            continue
        out.append(captured)
    return out


def _merge_groups(
    groups: list[ComponentGroup], max_parts: int,
) -> list[ComponentGroup]:
    """spec 順を保った連続均等併合 (先頭側のグループが 1 個多い)。"""
    n = len(groups)
    base, extra = divmod(n, max_parts)
    merged: list[ComponentGroup] = []
    idx = 0
    for i in range(max_parts):
        size = base + (1 if i < extra else 0)
        chunk = groups[idx:idx + size]
        idx += size
        merged.append(ComponentGroup(
            label=", ".join(g.label for g in chunk),
            spec_text="\n\n".join(g.spec_text.rstrip() for g in chunk),
        ))
    return merged


def plan_file_parts(
    spec: str, source_path: str, *, max_parts: int,
) -> FilePartsPlan | None:
    """spec のモジュール節を component 単位の部分生成計画へ決定論分割する。

    以下の場合は None (呼出側は現行の単発生成へフォールバック):
    モジュール節が見つからない / component 見出しが 0〜1 個 / spec が空。
    ``max_parts`` 超過時は spec 順の連続グループへ決定論的に併合する。
    """
    if not spec.strip() or max_parts < 2:
        return None
    found = extract_module_section(spec, source_path)
    if found is None:
        return None
    _, _, section = found

    comp_matches = list(_COMPONENT_HEADING_RE.finditer(section))
    if len(comp_matches) < 2:
        return None

    groups: list[ComponentGroup] = []
    for i, m in enumerate(comp_matches):
        end = (
            comp_matches[i + 1].start()
            if i + 1 < len(comp_matches) else len(section)
        )
        groups.append(ComponentGroup(
            label=m.group(1).strip(),
            spec_text=section[m.start():end].rstrip(),
        ))

    if len(groups) > max_parts:
        groups = _merge_groups(groups, max_parts)

    return FilePartsPlan(
        module_section=section,
        canonical_list=_extract_canonical_list(spec),
        groups=groups,
    )
