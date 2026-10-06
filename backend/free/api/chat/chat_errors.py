"""チャット応答のエラーを ``{code, 利用者向け文言, 再試行可否}`` へ分類する。

一次の根拠は例外の型 (``backend.exceptions``) と HTTP ステータス、llama-server の
エラー本文の ``error.type`` (``LocalClient`` が ``error_type`` として構造化して持つ)。
文字列照合は ``_is_context_exceeded`` の後方互換 1 箇所だけ。例外の原文は利用者へ
出さずログに残す (呼出側の責務)。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx

from backend.error_handlers import E0000, E1000, E1001, E1002, E1007, E1008
from backend.exceptions import (
    EvorefError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMProcessCrashedError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    ModelNotFoundError,
)
from backend.i18n_helper import msg

#: llama-server が文脈超過 (HTTP 400) で返す ``error.type``。
_CONTEXT_ERROR_TYPE = "exceed_context_size_error"


@dataclass(frozen=True)
class ChatErrorInfo:
    code: str
    i18n_key: str
    retryable: bool

    @property
    def message(self) -> str:
        return msg(self.i18n_key)


EMPTY_RESPONSE = ChatErrorInfo(E1008, "error.chat.empty_response", True)
#: LLM クライアント未接続 (起動前 / 起動失敗)。サービスを起こせば再試行できる。
LLM_NOT_CONNECTED = ChatErrorInfo(E1001, "cli.llm_not_connected", True)
UNKNOWN = ChatErrorInfo(E0000, "error.chat.unknown", False)


def _is_context_exceeded(status: int | None, error_type: str, text: str) -> bool:
    """文脈超過か。構造化された ``error_type`` が一次、無いときだけ本文の語で見る (後方互換)。"""
    if status != 400:
        return False
    if error_type:
        return error_type == _CONTEXT_ERROR_TYPE
    return "exceed_context_size" in text or "exceeds the available context size" in text


def _rejected(code: str, status: int | None, error_type: str, text: str) -> ChatErrorInfo:
    if _is_context_exceeded(status, error_type, text):
        return ChatErrorInfo(code, "error.chat.context_exceeded", False)
    return ChatErrorInfo(code, "error.chat.request_rejected", False)


def _http_status_error(exc: httpx.HTTPStatusError) -> ChatErrorInfo:
    """非ストリーム経路で型変換されずに届く ``HTTPStatusError`` (``_map_llama_error`` の契約)。"""
    status = exc.response.status_code
    if status >= 500:
        return ChatErrorInfo(E1000, "error.chat.server_error", True)
    try:
        text = exc.response.text
    except httpx.ResponseNotRead:
        text = ""
    error_type = ""
    try:
        err = json.loads(text).get("error")
        if isinstance(err, dict):
            error_type = str(err.get("type") or "")
    except (ValueError, AttributeError):
        pass
    return _rejected(E1007, status, error_type, text)


def classify_chat_error(exc: BaseException) -> ChatErrorInfo:
    """例外を利用者向けのエラー分類へ写す。"""
    if isinstance(exc, LLMRequestRejectedError):
        return _rejected(
            exc.code, exc.context.get("http_status"),
            str(exc.context.get("error_type") or ""), str(exc),
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return _http_status_error(exc)
    if isinstance(exc, LLMTimeoutError | asyncio.TimeoutError | httpx.TimeoutException):
        return ChatErrorInfo(E1002, "error.chat.timeout", True)
    if isinstance(exc, LLMConnectionError | LLMProcessCrashedError | httpx.TransportError):
        code = exc.code if isinstance(exc, LLMError) else E1001
        return ChatErrorInfo(code, "error.chat.connection", True)
    if isinstance(exc, ModelNotFoundError):
        return ChatErrorInfo(exc.code, "error.chat.model_unavailable", False)
    if isinstance(exc, LLMInvalidResponseError | LLMError):
        return ChatErrorInfo(exc.code, "error.chat.server_error", True)
    if isinstance(exc, EvorefError):
        return ChatErrorInfo(exc.code, exc.i18n_key, False)
    return UNKNOWN
