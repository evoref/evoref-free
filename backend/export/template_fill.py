"""帳票の値決定 (f_11 §9.2 / c_16 §4.5.2、段階 B-1b)。

``fields`` (:mod:`backend.export.template_writer`) から動的な pydantic
モデルを組み、``AuxClient.generate_json`` へ **文法制約つき生成 1 回**だけ
渡す (CLAUDE.md §6 #1: チャット応答パスの補助判定は決定論層、またはベースの
文法制約 JSON で行う)。``PURPOSE_SCHEMAS`` は静的表で ``fields`` は様式ごとに
形が変わるため登録できない — 呼出側が明示 ``response_schema=`` を渡す経路
(``generate_plan_json`` / ``generate_seeded_plan_json`` と同じ先例) を使う。

json_schema が守るのは形 (全項目 nullable の文字列 / 配列) だけなので、型
(``number`` / ``date``) と ``required`` の検査は
:func:`backend.export.template_writer.validate_values` が別途行う。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, create_model

from backend.export.template_writer import FieldSpec
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.aux_client import AuxClient

logger = get_logger("export.template_fill")

_STRICT = ConfigDict(extra="forbid")


def build_fields_schema(fields: tuple[FieldSpec, ...]) -> type:
    """``fields`` (``compute`` を除く) から nullable な動的 pydantic モデルを組む。

    ``number`` は ``float | None``、それ以外は ``str | None`` (``date`` は
    ``YYYY-MM-DD`` 文字列のまま、コード側の
    :func:`~backend.export.template_writer.validate_values` が変換・検証する)。
    ``repeat`` フィールドは行オブジェクト (列も全て nullable) の配列。
    ``compute`` はモデルに含めない — 値は LLM ではなくコードが計算する。

    ``number`` を文字列にしない理由 (2026-09-20 実機確認): 全項目 ``str`` だった
    初版では、「りんご 2個 単価100円」に対して 27B が数量・単価を **空文字**で
    返した。文法制約で数値型にすれば、モデルは数か null しか出せず、単位を
    付けるか迷って空にする余地が無い。
    """
    attrs: dict[str, tuple[Any, Any]] = {}
    for f in fields:
        if f.compute is not None:
            continue
        if f.repeat:
            col_attrs = {c.name: (_json_type(c.type), None) for c in f.columns}
            row_model = create_model(
                f"TemplateRow_{f.name}", __config__=_STRICT, **col_attrs,
            )
            attrs[f.name] = (list[row_model] | None, None)  # type: ignore[valid-type]
            continue
        attrs[f.name] = (_json_type(f.type), None)
    return create_model("TemplateFieldsResult", __config__=_STRICT, **attrs)


def _json_type(field_type: str) -> Any:
    return (float | None) if field_type == "number" else (str | None)


_TYPE_HINTS = {
    "number": "数値。単位や桁区切りを付けず数字だけ",
    "date": "日付。YYYY-MM-DD",
    "string": "文字列",
    "text": "文章",
}


def _build_prompt(
    fields: tuple[FieldSpec, ...], *, instruction: str, history_text: str, file_text: str,
) -> str:
    lines = [
        "次の依頼・直近の対話・添付ファイルの内容から、帳票の項目を埋めてください。",
        "根拠が無い項目は null を返してください。推測で値を作らないでください。",
        "",
        "# 項目",
    ]
    for f in fields:
        if f.compute is not None:
            continue
        if f.repeat:
            cols = "、".join(
                f"{c.name} = {_TYPE_HINTS.get(c.type, c.type)}" for c in f.columns
            )
            lines.append(
                f"- {f.name} (繰り返し行。1 件につき 1 行、列: {cols}): {f.description}".rstrip(": "),
            )
        else:
            lines.append(
                f"- {f.name} ({_TYPE_HINTS.get(f.type, f.type)}): {f.description}".rstrip(": "),
            )
    lines += ["", "# 依頼", instruction or "", "", "# 直近の対話", history_text or "(なし)"]
    if file_text:
        lines += ["", "# 添付ファイル", file_text]
    return "\n".join(lines)


async def decide_template_values(
    aux_client: "AuxClient | None",
    fields: tuple[FieldSpec, ...],
    *,
    instruction: str,
    history_text: str = "",
    file_text: str = "",
) -> dict[str, Any]:
    """依頼文・直近の対話・添付ファイルから値を 1 回で決める (根拠が無ければ ``null``)。

    ``aux_client is None`` (degraded) / 例外時は空 dict (= 全項目未決定として
    呼出側の needs_input へ倒れる)。
    """
    if aux_client is None:
        logger.info("template_fill: aux client is not wired; skipping value decision")
        return {}
    schema = build_fields_schema(fields)
    prompt = _build_prompt(
        fields, instruction=instruction, history_text=history_text, file_text=file_text,
    )
    try:
        data = await aux_client.generate_json(
            prompt, purpose="template_fill", response_schema=schema,
        )
    except Exception as e:  # noqa: BLE001 - 縮退して needs_input へ倒す
        logger.warning("template_fill: value decision failed (degrading): %s", e)
        return {}
    return data if isinstance(data, dict) else {}


__all__ = ["build_fields_schema", "decide_template_values"]
