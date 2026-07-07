"""CodeSpec のレンダリング

コード生成の事前準備で合成した共有設計仕様 (`backend.free.llm.json_schemas.CodeSpec`)
を 2 用途に整形する:

- :func:`render_spec_for_prompt` — 各ユニット (小ブロック) 生成プロンプトへ注入する
  コンパクトテキスト。全ブロックが同一のモジュール名 / データモデル / シグネチャ /
  エントリポイント / プロトコルに準拠するための契約。
- :func:`render_spec_markdown` — 成果物 ``SPEC.md`` 本文 (Markdown)。flowchart が
  ある場合は mermaid を埋め込む。

設計書 f_08 §3 参照。``spec is None`` は空文字列を返す (従来挙動フォールバック)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.i18n_helper import get_locale

if TYPE_CHECKING:
    from backend.free.llm.json_schemas import CodeSpec

# SPEC.md の固定見出し (GUI の言語設定 = i18n locale に追従)。本文の記述テキスト
# は合成プロンプト側で locale 言語を指示するため、見出しもここで揃える。
_MD_LABELS: dict[str, dict[str, str]] = {
    "ja": {
        "title_fallback": "設計仕様",
        "modules": "モジュール構成",
        "file": "ファイル",
        "role": "役割",
        "data_models": "データモデル",
        "field": "フィールド",
        "type": "型",
        "interfaces": "公開シグネチャ",
        "entry_point": "エントリポイント",
        "module": "モジュール",
        "invocation": "起動",
        "protocol": "通信/データプロトコル",
        "constraints": "制約",
        "flowchart": "フローチャート",
    },
    "en": {
        "title_fallback": "Design Specification",
        "modules": "Module structure",
        "file": "File",
        "role": "Role",
        "data_models": "Data models",
        "field": "Field",
        "type": "Type",
        "interfaces": "Public signatures",
        "entry_point": "Entry point",
        "module": "Module",
        "invocation": "Invocation",
        "protocol": "Communication/data protocol",
        "constraints": "Constraints",
        "flowchart": "Flowchart",
    },
}


def render_spec_for_prompt(spec: CodeSpec | None) -> str:
    """設計仕様を生成プロンプトへ注入するコンパクト整形を返す。

    ``spec`` が ``None`` または実質空の場合は空文字列を返し、呼出側は
    仕様注入をスキップして従来挙動にフォールバックする。
    """
    if spec is None:
        return ""

    lines: list[str] = []

    if spec.modules:
        lines.append("モジュール構成 (この正準ファイル名以外を import しない):")
        for m in spec.modules:
            lines.append(f"  - {m.path}: {m.purpose}")

    if spec.data_models:
        lines.append("共有データモデル (このフィールド名/型に厳密準拠):")
        for dm in spec.data_models:
            field_str = ", ".join(f"{f.name}: {f.type}" for f in dm.fields)
            lines.append(f"  - {dm.name} ({dm.kind} @ {dm.module}): {field_str}")

    if spec.interfaces:
        lines.append("公開シグネチャ (この通りに定義/呼出):")
        for itf in spec.interfaces:
            lines.append(f"  - [{itf.module}] {itf.signature}")

    if spec.entry_point and spec.entry_point.module:
        lines.append(
            f"エントリポイント: {spec.entry_point.module} — "
            f"{spec.entry_point.invocation}"
        )

    if spec.protocol:
        lines.append(f"通信/データプロトコル: {spec.protocol}")

    if spec.constraints:
        lines.append("制約: " + "; ".join(spec.constraints))

    return "\n".join(lines)


def render_spec_markdown(spec: CodeSpec | None, flowchart_mermaid: str = "") -> str:
    """設計仕様を ``SPEC.md`` 成果物 (Markdown) として整形する。

    ``flowchart_mermaid`` が非空ならフローチャート節を mermaid コードブロックで
    埋め込む。``spec is None`` は空文字列を返す。
    """
    if spec is None:
        return ""

    labels = _MD_LABELS.get(get_locale(), _MD_LABELS["en"])
    out: list[str] = []
    out.append(f"# {spec.title or labels['title_fallback']}")
    out.append("")
    if spec.summary:
        out.append(spec.summary)
        out.append("")

    if spec.modules:
        out.append(f"## {labels['modules']}")
        out.append("")
        out.append(f"| {labels['file']} | {labels['role']} |")
        out.append("|---|---|")
        for m in spec.modules:
            out.append(f"| `{m.path}` | {m.purpose} |")
        out.append("")

    if spec.data_models:
        out.append(f"## {labels['data_models']}")
        out.append("")
        for dm in spec.data_models:
            out.append(f"### `{dm.name}` ({dm.kind} @ `{dm.module}`)")
            out.append("")
            if dm.fields:
                out.append(f"| {labels['field']} | {labels['type']} |")
                out.append("|---|---|")
                for f in dm.fields:
                    out.append(f"| `{f.name}` | `{f.type}` |")
                out.append("")

    if spec.interfaces:
        out.append(f"## {labels['interfaces']}")
        out.append("")
        for itf in spec.interfaces:
            out.append(f"- `{itf.module}`: `{itf.signature}`")
        out.append("")

    if spec.entry_point and spec.entry_point.module:
        out.append(f"## {labels['entry_point']}")
        out.append("")
        out.append(f"- {labels['module']}: `{spec.entry_point.module}`")
        out.append(f"- {labels['invocation']}: `{spec.entry_point.invocation}`")
        out.append("")

    if spec.protocol:
        out.append(f"## {labels['protocol']}")
        out.append("")
        out.append(spec.protocol)
        out.append("")

    if spec.constraints:
        out.append(f"## {labels['constraints']}")
        out.append("")
        for c in spec.constraints:
            out.append(f"- {c}")
        out.append("")

    if flowchart_mermaid.strip():
        out.append(f"## {labels['flowchart']}")
        out.append("")
        out.append("```mermaid")
        out.append(flowchart_mermaid.strip())
        out.append("```")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
