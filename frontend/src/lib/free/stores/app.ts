import { writable } from 'svelte/store';

/** アプリケーション全体で共有するインスタンス名 */
export const instanceName = writable('evoref');

/** アプリケーションバージョン
 *
 * frontend は独自の版番号を持たない。version の SSOT は backend
 * (`backend/free/__version__.py`) で、起動後に `/api/status` レスポンスの
 * `version` で上書きされる。取得前は空文字 (UI は版を表示しない)。
 */
export const appVersion = writable('');
