"""evoref カスタム例外階層

エラーコード体系（error_handlers.py E1xxx〜E6xxx）と対応する
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


# ── E3xxx: ファイルシステム関連 ──


class StorageError(EvorefError):
    """ファイルシステム関連エラーの基底"""

    code = "E3000"
    status_code = 500
    i18n_key = "error.fs.generic"


class IndexCorruptedError(StorageError):
    """E3001: ベクトルインデックス破損"""

    code = "E3001"
    i18n_key = "error.fs.index_corrupted"


class MetadataCorruptedError(StorageError):
    """E3002: メタデータファイル破損"""

    code = "E3002"
    i18n_key = "error.fs.metadata_corrupted"


class IndexMetadataMismatchError(StorageError):
    """E3003: インデックスとメタデータの不整合"""

    code = "E3003"
    i18n_key = "error.fs.index_metadata_mismatch"


class MemoryFileCorruptedError(StorageError):
    """E3004: メモリファイル破損"""

    code = "E3004"
    i18n_key = "error.fs.memory_corrupted"


class DiskFullError(StorageError):
    """E3005: ディスク容量不足"""

    code = "E3005"
    i18n_key = "error.fs.disk_full"


class PathNotFoundError(StorageError):
    """E3006: パス不存在"""

    code = "E3006"
    i18n_key = "error.fs.path_not_found"


# ── E4xxx: RAG / メモリ関連 ──


class RAGError(EvorefError):
    """RAG / メモリ関連エラーの基底"""

    code = "E4000"
    status_code = 500
    i18n_key = "error.rag.generic"


class EmptyIndexError(RAGError):
    """E4001: 空インデックス検索"""

    code = "E4001"
    status_code = 400
    i18n_key = "error.rag.empty_index"


class ChunkError(RAGError):
    """E4002: チャンク分割失敗"""

    code = "E4002"
    status_code = 400
    i18n_key = "error.rag.chunk_failed"


class CartridgeInvalidZipError(RAGError):
    """E4003: 不正 ZIP"""

    code = "E4003"
    status_code = 400
    i18n_key = "error.cartridge.invalid_zip"


class CartridgeSchemaMismatchError(RAGError):
    """E4004: スキーマ不一致"""

    code = "E4004"
    status_code = 422
    i18n_key = "error.cartridge.schema_mismatch"


class CartridgeDuplicateIdError(RAGError):
    """E4005: ID 重複"""

    code = "E4005"
    status_code = 409
    i18n_key = "error.cartridge.duplicate_id"


class SearchNoResultError(RAGError):
    """E4006: 検索結果なし"""

    code = "E4006"
    status_code = 200
    i18n_key = "error.rag.no_results"


class STMIntegrityError(RAGError):
    """E4007: STM ノート整合性エラー"""

    code = "E4007"
    i18n_key = "error.rag.stm_integrity"


class VectorDimensionMismatchError(RAGError):
    """E4011: ベクトル次元不一致（実行時）

    クエリ／追加対象ベクトルの次元と、保存済みインデックスの
    次元が一致しないときに発生する。`evoref reindex` の実行が必要。
    （E4009 は import 時の次元不一致用）
    """

    code = "E4011"
    status_code = 409
    i18n_key = "error.rag.vector_dim_mismatch"


# ── E6xxx: CLI 関連 ──


class CLIError(EvorefError):
    """CLI 関連エラーの基底"""

    code = "E6000"
    status_code = 500
    i18n_key = "error.cli.generic"


class BackendNotRunningError(CLIError):
    """E6001: バックエンド未起動"""

    code = "E6001"
    status_code = 503
    i18n_key = "error.cli.backend_not_running"


class EncodingError(CLIError):
    """E6002: UTF-8 エンコード失敗"""

    code = "E6002"
    i18n_key = "error.cli.encoding_failed"


class PidLockError(CLIError):
    """E6003: PID ファイルロック競合"""

    code = "E6003"
    i18n_key = "error.cli.pid_lock"


class StreamInterruptedError(CLIError):
    """E6004: ストリーミング中の接続断"""

    code = "E6004"
    i18n_key = "error.cli.stream_interrupted"


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


class PlanGenerationError(LongFormError):
    """E9011: 計画生成失敗"""

    code = "E9011"
    status_code = 500
    i18n_key = "error.long_form.plan_failed"


class UnitGenerationError(LongFormError):
    """E9012: ユニット生成失敗"""

    code = "E9012"
    status_code = 500
    i18n_key = "error.long_form.unit_failed"
