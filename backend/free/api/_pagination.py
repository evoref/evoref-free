"""一覧 API の任意ページング (``limit`` / ``offset``、c_06 §1.6)。

従来は全件を返していた一覧へ **後から足す** ための作法。``limit`` を省けば
従来どおり全件 (``offset`` だけ渡せば先頭を飛ばした残り全部)、本文の形は
変えず、絞る前の総数は応答ヘッダ ``X-Total-Count`` で返す。範囲外の値は
FastAPI の検証で 422。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypeVar

from fastapi import Query, Response

T = TypeVar("T")

#: ``limit`` の上限。
PAGE_LIMIT_MAX = 1000

#: 絞る前の総数を返す応答ヘッダ。
TOTAL_COUNT_HEADER = "X-Total-Count"

#: ``limit: LimitParam = None`` / ``offset: OffsetParam = 0`` で受ける。
LimitParam = Annotated[int | None, Query(ge=1, le=PAGE_LIMIT_MAX)]
OffsetParam = Annotated[int, Query(ge=0)]


def paginate(
    items: Sequence[T], limit: int | None, offset: int,
    response: Response | None = None, *, header: str = TOTAL_COUNT_HEADER,
) -> list[T]:
    """``items[offset:offset+limit]`` を返し、``response`` があれば総数をヘッダに載せる。"""
    if response is not None:
        response.headers[header] = str(len(items))
    end = None if limit is None else offset + limit
    return list(items[offset:end])
