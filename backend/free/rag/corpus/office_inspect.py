"""``templates/`` 配下の Office ファイルの中身検査 (c_16 §4.5.2)

拡張子は信用しない (``.docm`` は ``.docx`` に改名できる)。zip の中身を見て、
マクロや外部テンプレート参照を持つファイルを弾く。フレームワーク非依存
(標準ライブラリの ``zipfile`` だけ)。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

#: ``.dotx`` / ``.potx`` / ``.xltx`` → 本体形式への変換材料
#: (拡張子, ``[Content_Types].xml`` の main part 名, その語幹)。
#: python-docx / python-pptx / openpyxl は template の content-type
#: (``…template.main+xml``) を ``ValueError`` で拒否するため (2026-09-20 実測)、
#: 梱包前に本体形式の content-type (``…<word>.main+xml``) へ書き換える。
_TEMPLATE_MAIN_PARTS: dict[str, tuple[str, str, str]] = {
    ".dotx": (".docx", "word/document.xml", "document"),
    ".potx": (".pptx", "ppt/presentation.xml", "presentation"),
    ".xltx": (".xlsx", "xl/workbook.xml", "sheet"),
}


def convert_template_extension(content: bytes, ext: str) -> tuple[bytes, str]:
    """``.dotx`` 等を本体形式のバイト列へ変換する (c_16 §4.5.2)。

    対象外の拡張子はそのまま返す。``[Content_Types].xml`` の main part の
    content-type だけを書き換え、他のパートは無変更でコピーする。

    Raises:
        ValueError: zip として読めない / ``[Content_Types].xml`` が無い /
            main part の content-type が見つからない。
    """
    target = _TEMPLATE_MAIN_PARTS.get(ext.lower())
    if target is None:
        return content, ext
    target_ext, part_name, word = target

    with zipfile.ZipFile(io.BytesIO(content)) as src:
        names = src.namelist()
        if "[Content_Types].xml" not in names:
            raise ValueError("template file has no [Content_Types].xml")
        parts = {name: src.read(name) for name in names}

    ct_xml = parts["[Content_Types].xml"].decode("utf-8")
    pattern = re.compile(
        r'(PartName="/' + re.escape(part_name) + r'"\s+ContentType=")([^"]*)(")',
    )
    new_ct_xml, count = pattern.subn(
        lambda m: m.group(1) + m.group(2).replace("template.main+xml", f"{word}.main+xml") + m.group(3),
        ct_xml, count=1,
    )
    if count == 0:
        raise ValueError(f"template file has no content-type override for {part_name}")
    parts["[Content_Types].xml"] = new_ct_xml.encode("utf-8")

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in parts.items():
            dst.writestr(name, data)
    return out.getvalue(), target_ext


def inspect_office_file(path: Path | str) -> str | None:
    """Office ファイル (docx/pptx/xlsx 形式の zip) を中身で検査する。

    拒否理由の文字列を返す。安全なら ``None``。

    拒否条件 (c_16 §4.5.2):

    - zip 内に ``vbaProject.bin`` がある (VBA マクロ)
    - ``[Content_Types].xml`` にマクロ有効の content-type がある
      (``.docm`` / ``.xlsm`` / ``.pptm`` を拡張子だけ変えたもの)
    - ``.rels`` に ``attachedTemplate`` で ``TargetMode="External"`` の
      関係がある (外部テンプレートの読み込み)
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(str(path)) as zf:
            names = zf.namelist()
            if any(name.lower().endswith("vbaproject.bin") for name in names):
                return "contains a VBA macro project (vbaProject.bin)"

            content_types_name = "[Content_Types].xml"
            if content_types_name in names:
                content_types = zf.read(content_types_name).decode(
                    "utf-8", errors="replace",
                )
                if "macroEnabled" in content_types:
                    return "declares a macro-enabled content type"

            for name in names:
                if not name.endswith(".rels"):
                    continue
                rels = zf.read(name).decode("utf-8", errors="replace")
                if "attachedTemplate" in rels and 'TargetMode="External"' in rels:
                    return "references an external attached template"
    except zipfile.BadZipFile:
        return "not a valid Office (zip) file"
    return None


__all__ = ["convert_template_extension", "inspect_office_file"]
