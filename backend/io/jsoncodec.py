"""永続化の JSON 符号化 (c_05 §0.5、G1 設計 §17.4)。

``pydantic_core`` の ``to_json`` / ``from_json`` をこの 1 箇所に閉じて使う
(stdlib ``json`` の約 2 倍速・メモリ半減。既存依存で新規依存なし)。

- 出力は区切りの空白なし・``ensure_ascii`` なし・``indent`` なし・キー順は dict の
  組み立て順 (``sort_keys`` を使わない。封筒は ``payload`` を最後に置く)。
- NaN / Infinity は JSON ``null`` で書く (stdlib は ``NaN`` を書き、他言語の
  読み手が読めない)。
- 失敗は stdlib と同じ例外階層で送出する: 書けない値は ``TypeError`` / ``ValueError``、
  壊れた JSON は ``json.JSONDecodeError`` (呼出側の ``except json.JSONDecodeError``
  をそのまま効かせる)。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_core import PydanticSerializationError, from_json, to_json


def dumps_bytes(obj: Any, *, indent: int | None = None) -> bytes:
    """``obj`` を UTF-8 の JSON バイト列にする (``indent`` は利用者が手で直すファイル用)。"""
    try:
        return to_json(obj, indent=indent, inf_nan_mode="null")
    except PydanticSerializationError as e:
        raise TypeError(str(e)) from e


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """``obj`` を JSON 文字列にする。"""
    return dumps_bytes(obj, indent=indent).decode("utf-8")


def loads(data: str | bytes | bytearray) -> Any:
    """JSON を読む。壊れていれば ``json.JSONDecodeError``。"""
    try:
        return from_json(data)
    except ValueError as e:
        text = data if isinstance(data, str) else bytes(data).decode("utf-8", "replace")
        raise json.JSONDecodeError(str(e), text, 0) from e


__all__ = ["dumps", "dumps_bytes", "loads"]
