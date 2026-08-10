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
E1001 = "E1001"  # 接続失敗
E1002 = "E1002"  # 推論タイムアウト
E1003 = "E1003"  # 不正レスポンス
E1004 = "E1004"  # モデルファイル不在
E1005 = "E1005"  # LoRA アダプタ不在
E1006 = "E1006"  # プロセス異常終了
E1007 = "E1007"  # リクエスト拒否（コンテンツ起因）
E1008 = "E1008"  # アシスト非常駐 (assist_model.residency=on_demand)

# ファイルシステム (E3xxx)
E3001 = "E3001"  # ベクトルインデックス破損 (index_q8.npy / scales.npy)
E3002 = "E3002"  # metadata.json 破損
E3003 = "E3003"  # インデックス/メタデータ不整合
E3004 = "E3004"  # メモリファイル破損
E3005 = "E3005"  # ディスク容量不足
E3006 = "E3006"  # パス不存在

# RAG / メモリ (E4xxx)
E4001 = "E4001"  # 空インデックス検索
E4002 = "E4002"  # チャンク分割失敗
E4003 = "E4003"  # 不正 ZIP
E4004 = "E4004"  # スキーマ不一致
E4005 = "E4005"  # ID 重複
E4006 = "E4006"  # 検索結果なし
E4007 = "E4007"  # STM ノート整合性エラー
E4011 = "E4011"  # ベクトル次元不一致（実行時）

# CLI (E6xxx)
E6001 = "E6001"  # バックエンド未起動
E6002 = "E6002"  # UTF-8 エンコード失敗
E6003 = "E6003"  # PID ファイルロック競合
E6004 = "E6004"  # ストリーミング中の接続断


def _make_error_response(
    code: str,
    i18n_key: str,
    status_code: int,
    log_msg: str,
    **context,
) -> JSONResponse:
    """エラーレスポンス生成の共通処理"""
    user_message = msg(i18n_key, **context)
    logger.error("%s %s", code, log_msg)
    error = ErrorResponse(
        code=code,
        message=user_message,
        i18n_key=i18n_key,
        context=context,
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": error.to_dict()},
    )


def _lookup_mapping(mapping: dict, code: str, category: str, **context) -> JSONResponse:
    """mapping からエラーコードを検索し、未知コードにはフォールバックを返す"""
    entry = mapping.get(code)
    if entry is None:
        logger.warning("Unknown %s error code: %s", category, code)
        return _make_error_response(
            code, "api.server_error", 500,
            f"Unknown {category} error code: {code}", **context,
        )
    status, i18n_key, log_template = entry
    return _make_error_response(code, i18n_key, status, log_template.format_map(context), **context)


def handle_llama_error(code: str, **context) -> JSONResponse:
    """llama.cpp 関連エラー"""
    mapping = {
        E1001: (503, "error.llama.connection_failed", "llama.cpp server connection failed: {host}:{port}"),
        E1002: (408, "error.llama.timeout", "llama.cpp inference timed out after {seconds}s"),
        E1003: (502, "error.llama.invalid_response", "llama.cpp returned invalid response: {detail}"),
        E1004: (503, "error.llama.model_not_found", "GGUF model file not found: {path}"),
        E1006: (503, "error.llama.process_crashed", "llama.cpp process exited unexpectedly"),
        E1007: (400, "error.llama.request_rejected", "llama.cpp rejected request: {detail}"),
    }
    return _lookup_mapping(mapping, code, "llama", **context)


def handle_fs_error(code: str, **context) -> JSONResponse:
    """ファイルシステム関連エラー"""
    mapping = {
        E3001: (500, "error.fs.index_corrupted", "Vector index corrupted: {path}"),
        E3002: (500, "error.fs.metadata_corrupted", "Metadata file corrupted: {path}"),
        E3003: (500, "error.fs.index_metadata_mismatch", "Index/metadata mismatch: index={idx_count}, metadata={meta_count}"),
        E3004: (500, "error.fs.memory_corrupted", "Memory file corrupted: {path}"),
        E3005: (500, "error.fs.disk_full", "Disk space insufficient: {path}"),
    }
    return _lookup_mapping(mapping, code, "fs", **context)


def handle_rag_error(code: str, **context) -> JSONResponse:
    """RAG 関連エラー"""
    mapping = {
        E4002: (400, "error.rag.chunk_failed", "Failed to chunk file: {filename}"),
        E4003: (400, "error.cartridge.invalid_zip", "Invalid cartridge ZIP file: {filename}"),
        E4004: (422, "error.cartridge.schema_mismatch", "Cartridge schema validation failed: {detail}"),
        E4005: (409, "error.cartridge.duplicate_id", "Cartridge ID already exists: {id}"),
    }
    return _lookup_mapping(mapping, code, "rag", **context)


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
