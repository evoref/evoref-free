/**
 * VRAM 使用量ストア
 *
 * ``GET /api/system/vram_status`` を定期ポーリングして 3 モデル
 * (base / embed) の推定値 + 実測値を保持する。
 *
 * ポーリング間隔はサーバ負荷を考慮して 10 秒既定 (issue の非スコープに
 * 明示記載: "WebSocket push 更新は不要。polling で 5 秒間隔程度で十分。
 * 負荷を考慮して 10 秒でも可")。バックエンドが落ちているときはエラー時
 * に ``null`` に戻し、UI は "利用不可" を表示する。
 */

import { writable } from 'svelte/store';
import type { VramStatusResponse } from '$lib/free/api';
import { getVramStatus } from '$lib/free/api';
import { handleApiCall } from '$lib/free/utils/error';
import { createPoller } from '$lib/free/stores/_polling';

export const vramStatus = writable<VramStatusResponse | null>(null);

/** 1 回だけ取得して store を更新 */
export async function refreshVramStatus(): Promise<void> {
	const status = await handleApiCall(() => getVramStatus(), {
		silent: true,
		fallbackKey: 'error.status_failed'
	});
	if (status) {
		vramStatus.set(status);
	} else {
		vramStatus.set(null);
	}
}

const _poller = createPoller(refreshVramStatus, 10_000);

/** ポーリング開始 (既定 10 秒) */
export function startVramPolling(intervalMs?: number): void {
	_poller.start(intervalMs);
}

/** ポーリング停止 */
export function stopVramPolling(): void {
	_poller.stop();
}
