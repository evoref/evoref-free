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

if TYPE_CHECKING:
    from backend.free.llm.json_schemas import CodeSpec


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

    out: list[str] = []
    out.append(f"# {spec.title or '設計仕様'}")
    out.append("")
    if spec.summary:
        out.append(spec.summary)
        out.append("")

    if spec.modules:
        out.append("## モジュール構成")
        out.append("")
        out.append("| ファイル | 役割 |")
        out.append("|---|---|")
        for m in spec.modules:
            out.append(f"| `{m.path}` | {m.purpose} |")
        out.append("")

    if spec.data_models:
        out.append("## データモデル")
        out.append("")
        for dm in spec.data_models:
            out.append(f"### `{dm.name}` ({dm.kind} @ `{dm.module}`)")
            out.append("")
            if dm.fields:
                out.append("| フィールド | 型 |")
                out.append("|---|---|")
                for f in dm.fields:
                    out.append(f"| `{f.name}` | `{f.type}` |")
                out.append("")

    if spec.interfaces:
        out.append("## 公開シグネチャ")
        out.append("")
        for itf in spec.interfaces:
            out.append(f"- `{itf.module}`: `{itf.signature}`")
        out.append("")

    if spec.entry_point and spec.entry_point.module:
        out.append("## エントリポイント")
        out.append("")
        out.append(f"- モジュール: `{spec.entry_point.module}`")
        out.append(f"- 起動: `{spec.entry_point.invocation}`")
        out.append("")

    if spec.protocol:
        out.append("## 通信/データプロトコル")
        out.append("")
        out.append(spec.protocol)
        out.append("")

    if spec.constraints:
        out.append("## 制約")
        out.append("")
        for c in spec.constraints:
            out.append(f"- {c}")
        out.append("")

    if flowchart_mermaid.strip():
        out.append("## フローチャート")
        out.append("")
        out.append("```mermaid")
        out.append(flowchart_mermaid.strip())
        out.append("```")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
