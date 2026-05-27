"""JSON / YAML Writer

.json: 構造化データ → JSON
.yaml: 構造化データ → YAML（PyYAML 既存依存）
"""

from __future__ import annotations

import json
from typing import override

from backend.export._writer_base import BytesWriterBase
from backend.export.base import ExportContent, ExportError


def _build_export_data(content: ExportContent) -> object:
    """ExportContent から出力対象データを構築"""
    # raw_data があればそれを使用
    if content.raw_data is not None:
        return content.raw_data

    # blocks から構造化データを構築
    if content.blocks:
        result: dict = {}
        if content.title:
            result["title"] = content.title
        if content.metadata:
            result["metadata"] = content.metadata
        sections: list[dict] = []
        for block in content.blocks:
            section: dict = {"type": block.type}
            if block.content:
                section["content"] = block.content
            if block.type == "heading":
                section["level"] = block.level
            elif block.type == "code" and block.language:
                section["language"] = block.language
            elif block.type == "table" and block.rows:
                section["rows"] = block.rows
            elif block.type == "list":
                section["ordered"] = block.ordered
                section["items"] = block.items
            sections.append(section)
        result["content"] = sections
        return result

    # raw_markdown しかない場合
    if content.raw_markdown:
        result = {}
        if content.title:
            result["title"] = content.title
        result["text"] = content.raw_markdown
        if content.metadata:
            result["metadata"] = content.metadata
        return result

    return {}


class JsonYamlWriter(BytesWriterBase):
    """JSON / YAML ファイル Writer"""

    @property
    @override
    def extensions(self) -> frozenset[str]:
        return frozenset({".json", ".yaml", ".yml"})

    @override
    def _render_bytes(self, content: ExportContent, ext: str) -> bytes:
        return self._render(content, ext).encode("utf-8")

    @staticmethod
    def _render(content: ExportContent, ext: str) -> str:
        """拡張子に応じてシリアライズ"""
        data = _build_export_data(content)

        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ExportError(
                    "missing_library",
                    "PyYAML is required for YAML export: pip install pyyaml",
                )
            return yaml.dump(
                data,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

        # .json
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
