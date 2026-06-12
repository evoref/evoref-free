/** フロントエンド共通定数 */

/** モード再起動ステータス表示の自動クリア時間 (ms) */
export const MODE_RESTART_STATUS_TIMEOUT_MS = 3000;

/** コピー通知の表示時間 (ms) */
export const COPY_NOTIFICATION_TIMEOUT_MS = 1500;

/** SSE ストリーミングのチャンク間タイムアウト (ms) — サーバー無応答検知
 *
 * ベース LLM のプリフィルが長文コンテキスト時に 30〜50 秒に達するため、
 * 初回トークン到達前の誤検知を抑制する目的で 60 秒に設定。
 * バックエンド側 SSE keepalive (15 秒間隔) と組み合わせて、実 idle 時には
 * ≤15s で検知される。
 */
export const STREAM_CHUNK_TIMEOUT_MS = 60_000;

/** 添付ファイルの最大サイズ (bytes) — 10MB */
export const FILE_MAX_SIZE_BYTES = 10 * 1024 * 1024;

/** 添付ファイルの最大サイズ表示用 (MB) */
export const FILE_MAX_SIZE_MB = 10;
