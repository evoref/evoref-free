"""チャット機能の定数定義

マジックナンバーを排除し、意味のある名前を付与する。
config.yaml のデフォルト値として使われるものと、
コード内のハードコード値の両方を集約する。
"""

# ---------------------------------------------------------------------------
# コンテキスト・トークン関連のデフォルト値
# ---------------------------------------------------------------------------

#: config.yaml llama.context_size のフォールバック値
DEFAULT_CONTEXT_SIZE: int = 4096

#: config.yaml llama.max_tokens のフォールバック値
DEFAULT_MAX_TOKENS: int = 1024

#: build_messages の generation_reserve デフォルト
DEFAULT_GENERATION_RESERVE: int = 512

#: build_messages_for_loop の working_max_tokens デフォルト
DEFAULT_WORKING_MAX_TOKENS: int = 2048

# ---------------------------------------------------------------------------
# ストリーミング関連
# ---------------------------------------------------------------------------

#: SSE キープアライブ送信間隔（秒）
DEFAULT_KEEPALIVE_INTERVAL_SEC: float = 15.0

# ---------------------------------------------------------------------------
# Deliberative 層
# ---------------------------------------------------------------------------

#: ツール実行タイムアウト（秒）
TOOL_EXECUTION_TIMEOUT_SEC: float = 30.0

#: ツール結果の最大文字数（超過分は先頭/末尾のみ残す）
TOOL_RESULT_MAX_CHARS: int = 4096

#: コンテンツ生成用 max_tokens の下限
CONTENT_MAX_TOKENS_MIN: int = 1024

#: コンテンツ生成用 max_tokens 計算時の system プロンプト予約
CONTENT_SYSTEM_RESERVE: int = 512

#: ツール結果切り詰め時の先頭比率
TOOL_RESULT_HEAD_RATIO: float = 0.6

#: ツール結果切り詰め時の省略メッセージ文字数予約
TOOL_RESULT_OMISSION_CHARS: int = 60

# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------

#: メッセージ長の上限（文字数）
MAX_MESSAGE_LENGTH: int = 100_000

#: file_contexts のチャンク数上限（全ファイル合計）
MAX_FILE_CONTEXT_TOTAL_CHUNKS: int = 100

#: file_contexts の合計文字数上限
MAX_FILE_CONTEXT_TOTAL_CHARS: int = 500_000

#: session_id のフォーマット（UUID hex 8-64文字）
SESSION_ID_MIN_LENGTH: int = 8
SESSION_ID_MAX_LENGTH: int = 64

# ---------------------------------------------------------------------------
# step_queue サイズ制限（BUG-10 対策）
# ---------------------------------------------------------------------------

#: step_queue の最大サイズ（これを超えると古いイベントを破棄）
MAX_STEP_QUEUE_SIZE: int = 200
