/** フロントエンド共通定数 */

/** モード再起動ステータス表示の自動クリア時間 (ms) */
export const MODE_RESTART_STATUS_TIMEOUT_MS = 3000;

/** コピー通知の表示時間 (ms) */
export const COPY_NOTIFICATION_TIMEOUT_MS = 1500;

/** SSE ストリーミングのチャンク間タイムアウト (ms) — サーバー無応答検知
 *
 * ベース LLM のプリフィルが長文コンテキスト時に 30〜50 秒に達するため、
 * 初回トークン到達前の誤検知を抑制する目的で設定。
 * バックエンド側 SSE keepalive (15 秒間隔) と組み合わせて、実 idle 時には
 * ≤15s で検知される。
 *
 * **60 秒では短すぎた。** 2026-09-03 ライブ監査で、バックエンドが
 * `tokens=96, elapsed=138.08s` と**完走していた**応答が捨てられ
 * `Error: stream_timeout` になった (制約検証の修復生成中に keepalive が
 * 途切れる穴があった)。その穴は塞いだが、この値自体もバックエンドが
 * 単発で持ちうる最長の同期処理 (`LLM_TOOL_EXECUTION_TIMEOUT_SEC` = 180 秒)
 * を下回っていてはならない。keepalive が生きている限り 15 秒間隔でフレームが
 * 届くので、この上限が効くのは**本当にバックエンドが固まったとき**だけ。
 */
export const STREAM_CHUNK_TIMEOUT_MS = 180_000;

/** 添付ファイルの最大サイズ (bytes) — 10MB */
export const FILE_MAX_SIZE_BYTES = 10 * 1024 * 1024;

/** 添付ファイルの最大サイズ表示用 (MB) */
export const FILE_MAX_SIZE_MB = 10;
