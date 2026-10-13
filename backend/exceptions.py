"""evoref カスタム例外階層

エラーコード体系（error_handlers.py の E コード）と対応する
例外クラスを定義する。全カスタム例外は EvorefError を継承し、
FastAPI 例外ハンドラで自動マッピングされる。
"""


class EvorefError(Exception):
    """全カスタム例外の基底クラス"""

    code: str = "E0000"
    status_code: int = 500
    i18n_key: str = "api.server_error"

    def __init__(self, message: str = "", **context):
        super().__init__(message)
        self.context = context


# ── E1xxx: llama.cpp 関連 ──


class LLMError(EvorefError):
    """llama.cpp 関連エラーの基底"""

    code = "E1000"
    status_code = 500
    i18n_key = "error.llama.generic"


class LLMConnectionError(LLMError):
    """E1001: llama-server 接続失敗"""

    code = "E1001"
    status_code = 503
    i18n_key = "error.llama.connection_failed"


class LLMTimeoutError(LLMError):
    """E1002: 推論タイムアウト"""

    code = "E1002"
    status_code = 408
    i18n_key = "error.llama.timeout"


class LLMInvalidResponseError(LLMError):
    """E1003: 不正レスポンス"""

    code = "E1003"
    status_code = 502
    i18n_key = "error.llama.invalid_response"


class ModelNotFoundError(LLMError):
    """E1004: モデルファイル不在"""

    code = "E1004"
    status_code = 503
    i18n_key = "error.llama.model_not_found"


class LLMProcessCrashedError(LLMError):
    """E1006: プロセス異常終了"""

    code = "E1006"
    status_code = 503
    i18n_key = "error.llama.process_crashed"


class LLMRequestRejectedError(LLMError):
    """E1007: サーバーは健全だがリクエストを内容起因で拒否 (例: context 長超過)"""

    code = "E1007"
    status_code = 400
    i18n_key = "error.llama.request_rejected"


# ── E4xxx: RAG / メモリ関連 ──


class RAGError(EvorefError):
    """RAG / メモリ関連エラーの基底"""

    code = "E4000"
    status_code = 500
    i18n_key = "error.rag.generic"


class VectorDimensionMismatchError(RAGError):
    """E4011: ベクトル次元不一致（実行時）

    クエリ／追加対象ベクトルの次元と、保存済みインデックスの
    次元が一致しないときに発生する。`evoref reindex` の実行が必要。
    （E4009 は import 時の次元不一致用）
    """

    code = "E4011"
    status_code = 409
    i18n_key = "error.rag.vector_dim_mismatch"


# ── E9xxx: 長文生成エンジン関連 ──


class LongFormError(EvorefError):
    """長文生成エンジン関連エラーの基底"""

    code = "E9000"
    status_code = 500
    i18n_key = "error.long_form.generic"


class InsufficientContextError(LongFormError):
    """E9010: コンテキストサイズ不足"""

    code = "E9010"
    status_code = 400
    i18n_key = "error.long_form.insufficient_context"


class UnitGenerationError(LongFormError):
    """E9012: ユニット生成失敗"""

    code = "E9012"
    status_code = 500
    i18n_key = "error.long_form.unit_failed"
