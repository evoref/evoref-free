"""チャット機能の定数定義

マジックナンバーを排除し、意味のある名前を付与する。
config.yaml のデフォルト値として使われるものと、
コード内のハードコード値の両方を集約する。
"""

# ---------------------------------------------------------------------------
# コンテキスト・トークン関連のデフォルト値
# ---------------------------------------------------------------------------

#: config.yaml llama.context_size のフォールバック値。
#: SSOT は ``backend.config._CONTEXT_SIZE_FALLBACK`` (config 明示も arch
#: プロファイル宣言も無いときの ``-c``)。値を揃えないと、ここを既定に取る
#: 経路 (``inference.build_messages`` の既定引数) だけが半分の窓で組み立てる。
DEFAULT_CONTEXT_SIZE: int = 8192

#: config.yaml llama.max_tokens のフォールバック値
DEFAULT_MAX_TOKENS: int = 1024

#: build_messages の generation_reserve デフォルト
DEFAULT_GENERATION_RESERVE: int = 512

#: build_messages_for_loop の working_max_tokens デフォルト
#: (backend/schemas/memory.py の既定値と同期させること)
DEFAULT_WORKING_MAX_TOKENS: int = 4096

#: config.yaml memory.history_min_tokens のフォールバック値
#: (backend/schemas/memory.py の既定値と同期させること)
DEFAULT_HISTORY_MIN_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# ストリーミング関連
# ---------------------------------------------------------------------------

#: SSE キープアライブ送信間隔（秒）
DEFAULT_KEEPALIVE_INTERVAL_SEC: float = 15.0

# ---------------------------------------------------------------------------
# Reactive 軽量パス層
# ---------------------------------------------------------------------------

#: 軽量パスへエスカレーション判定を掛ける履歴長の目安 (メッセージ数)。
#:
#: かつては軽量パスへ渡す履歴を ``history[-N:]`` で切る値だった。末尾スライド窓は
#: 会話が 1 ターン伸びるたびに接頭辞 KV キャッシュを捨てるため、履歴の切り出しは
#: ``build_chat_messages`` (``_trim_history`` + ``_quantize_history_drop``) へ
#: 移した。現在この値は ``_gate_reactive_light`` が「会話が単発でない」ことを
#: 見る目安としてのみ使う。
REACTIVE_LIGHT_HISTORY_TURNS: int = 6

#: 軽量パスの max_tokens 上限。reactive 分類は短文応答前提で、reasoning 暴走の上限も兼ねる
REACTIVE_LIGHT_MAX_TOKENS: int = 512

# ---------------------------------------------------------------------------
# Deliberative 層
# ---------------------------------------------------------------------------

#: ツール実行タイムアウト（秒）
TOOL_EXECUTION_TIMEOUT_SEC: float = 30.0

#: 内部で LLM 生成を行うツール (summarize / translate / draft_document) の
#: 実行タイムアウト（秒）。既定 30 秒は生成系には短く、低速な環境では
#: 常に失敗する (実測 2026-07-26: draft_document が会議テンプレート生成で
#: 30 秒に達し、「ツール実行がタイムアウトしたため完了していません」だけが
#: 回答として返った)。iGPU の decode 実測 7〜13 tok/s で数百トークンの
#: 生成に 30〜60 秒かかるため、その 3 倍程度を確保する。
LLM_TOOL_EXECUTION_TIMEOUT_SEC: float = 180.0

#: 生成系ツール (summarize / translate / draft_document) の ``max_tokens``。
#:
#: **未指定にしてはいけない。** ``LocalClient.generate`` は ``max_tokens=None``
#: のとき payload からキーごと落とすので、llama-server は n_ctx を使い切るまで
#: 生成し続ける。上の実行タイムアウト (180 秒) と組み合わさると、遅い環境では
#: **必ず** タイムアウトする — 待ち時間だけ掛かって根拠枠にはエラー文字列が載る。
#:
#: 実インシデント (2026-08-31 ライブ監査 T07#1): 「目次案を10章分作って」で
#: ``draft_document`` が選ばれ ``max_tokens=None`` で発行 → 180 秒で
#: ``Tool execution timed out`` → その後ベースモデルが自力で答えた。
#: **タイムアウト分 (180 秒) がまるごと無駄** になった。
#:
#: 値は「タイムアウト内に必ず終わる」ことを優先して決める。実測の decode は
#: 遅い環境で 5 tok/s 程度なので、180 秒なら 900 トークンが上限の目安。
LLM_TOOL_MAX_TOKENS: int = 900

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

#: ツール結果に基づく接地回答（grounded QA）の生成温度。
#: 接地回答は創作不要で決定性優先。chat 既定 0.7 のままだと weak base が
#: 非決定的に拒否/混同しやすい（実機: ニュースで 0.7→~25%拒否、0.2→安定）。
#: ツール使用ターンのみ本値へ下げる。
TOOL_GROUNDED_TEMPERATURE: float = 0.2

#: 検索した記憶に基づく接地回答の生成温度。
#:
#: ツール結果ほど厳密ではないが、同じ理屈が効く: 参考情報が付いたターンの仕事は
#: 「与えられた材料から答えを組む」ことで、創作ではない。既定 0.7 のままだと
#: weak base が材料を無視したり話題を混同しやすい。
#:
#: ツール接地 (0.2) より高くしてあるのは、ツール結果が実測値そのものなのに対し、
#: 記憶は関連しているだけで**答えそのものとは限らない**ため。低くしすぎると
#: 参考情報を逐語で写す方向に倒れる。
#:
#: 発火条件は「関連度ゲートを通ったチャンクが 1 件以上ある」= ``rag_used``。
#: 注入自体が較正済みの棒で絞られているので、これが立つ = 材料があるターン。
CONTEXT_GROUNDED_TEMPERATURE: float = 0.4

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
