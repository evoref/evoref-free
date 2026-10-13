"""エラーハンドリング基盤"""

import traceback
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.exceptions import EvorefError
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("error_handlers")


def _is_debug_enabled(request: Request) -> bool:
    """リクエストコンテキストからデバッグモード有効かを判定"""
    try:
        dl = request.app.state.app_state.debug_logger
        return dl is not None and dl.enabled
    except AttributeError:
        return False


@dataclass
class ErrorResponse:
    """統一エラーレスポンス"""
    code: str
    message: str
    i18n_key: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "i18n_key": self.i18n_key,
            "context": self.context,
        }


# --- エラーコード定義 ---

# llama.cpp (E1xxx)
E1000 = "E1000"  # 汎用（5xx など型に落ちない失敗）
E1001 = "E1001"  # 接続失敗
E1002 = "E1002"  # 推論タイムアウト
E1003 = "E1003"  # 不正レスポンス
E1004 = "E1004"  # モデルファイル不在
E1005 = "E1005"  # LoRA アダプタ不在
E1006 = "E1006"  # プロセス異常終了
E1007 = "E1007"  # リクエスト拒否（コンテンツ起因）
E1008 = "E1008"  # チャット応答の本文が空（再試行しても空）

# 汎用 (E0xxx)
E0000 = "E0000"  # 分類できない例外（チャットのエラー分類の既定）

# ファイルシステム (E3xxx)
E3001 = "E3001"  # ベクトルインデックス破損 (index_q8.npy / scales.npy)
E3002 = "E3002"  # metadata.json 破損
E3005 = "E3005"  # ディスク容量不足
E3006 = "E3006"  # パス不存在

# RAG / メモリ (E4xxx)
E4002 = "E4002"  # チャンク分割失敗
E4003 = "E4003"  # 不正 ZIP
E4005 = "E4005"  # ID 重複
E4011 = "E4011"  # ベクトル次元不一致（実行時）

# CLI (E6xxx)
E6001 = "E6001"  # バックエンド未起動
E6002 = "E6002"  # UTF-8 エンコード失敗
E6003 = "E6003"  # PID ファイルロック競合
E6004 = "E6004"  # ストリーミング中の接続断


def register_exception_handlers(app: FastAPI) -> None:
    """FastAPI に例外ハンドラを登録

    全 HTTPException を設計書 §19.3 準拠の構造化形式で返す。
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException,
    ) -> JSONResponse:
        """全 HTTPException を構造化エラーレスポンスに正規化"""
        detail = exc.detail

        # 構造化済み（code フィールドを含む dict）→ そのまま使用
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail},
            )

        # 文字列・その他 → 構造化形式に変換
        status = exc.status_code
        message = str(detail) if detail else ""
        i18n_key = ""
        context: dict = {}

        if status == 404:
            i18n_key = "api.not_found"
            context = {"resource": str(request.url.path)}
            if not message:
                message = msg("api.not_found", resource=str(request.url.path))
        elif status >= 500:
            i18n_key = "api.server_error"
            logger.error("Server error %d: %s", status, detail)
            if not message:
                message = msg("api.server_error")

        return JSONResponse(
            status_code=status,
            content={
                "detail": {
                    "code": f"E0{status}",
                    "message": message,
                    "i18n_key": i18n_key,
                    "context": context,
                }
            },
        )

    @app.exception_handler(EvorefError)
    async def evoref_error_handler(
        request: Request, exc: EvorefError,  # noqa: ARG001
    ) -> JSONResponse:
        """EvorefError 派生例外を構造化形式で返す"""
        user_message = msg(exc.i18n_key, **exc.context) if exc.i18n_key else str(exc)
        logger.error("%s %s", exc.code, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": {
                    "code": exc.code,
                    "message": user_message,
                    "i18n_key": exc.i18n_key,
                    "context": exc.context,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception,
    ) -> JSONResponse:
        """未捕捉の例外を構造化形式で返す"""
        if _is_debug_enabled(request):
            tb = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__),
            )
            logger.error("Unhandled server error: %s\n%s", exc, tb)
        else:
            logger.error("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "E0500",
                    "message": msg("api.server_error"),
                    "i18n_key": "api.server_error",
                    "context": {},
                }
            },
        )
