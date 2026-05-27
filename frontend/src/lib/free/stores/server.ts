import { writable } from 'svelte/store';
import type { ComponentStatus, DebugStatusInfo, MemoryStats, ServerName } from '$lib/free/api';
import { getStatus } from '$lib/free/api';
import { handleApiCall } from '$lib/free/utils/error';
import { instanceName, appVersion } from '$lib/free/stores/app';

export interface ServerState {
	/** バックエンドに到達可能か */
	backendOnline: boolean;
	/** 各コンポーネント (base / assist / embed / reranker) */
	components: ComponentStatus[];
	/** デバッグ情報 */
	debug: DebugStatusInfo | null;
	/** メモリ統計 */
	memory: MemoryStats | null;
}

const initial: ServerState = {
	backendOnline: false,
	components: [],
	debug: null,
	memory: null
};

export const serverState = writable<ServerState>(initial);

/**
 * 現在 start/stop 操作中のサーバー名 (ServerStatus コンポーネントのスピナー表示用)
 *
 * コンポーネントローカルの $state にすると、Sidebar の collapse / ルート遷移で
 * ServerStatus が再マウントされた瞬間にスピナーが消えてしまうため、グローバル
 * ストアに保持して再マウントを跨いでも維持する。
 */
export const busyServer = writable<ServerName | null>(null);

let intervalId: ReturnType<typeof setInterval> | null = null;

/** ステータスを1回取得して store を更新 */
export async function refreshServerStatus(): Promise<void> {
	const status = await handleApiCall(() => getStatus(), {
		silent: true,
		fallbackKey: 'error.status_failed'
	});
	if (status) {
		instanceName.set(status.instance_name ?? 'evoref');
		if (status.version) appVersion.set(status.version);
		serverState.set({
			backendOnline: true,
			components: status.components ?? [],
			debug: status.debug ?? null,
			memory: status.memory ?? null
		});
	} else {
		serverState.set({ backendOnline: false, components: [], debug: null, memory: null });
	}
}

/** ポーリング開始（デフォルト30秒間隔） */
export function startPolling(intervalMs = 30_000): void {
	stopPolling();
	refreshServerStatus();
	intervalId = setInterval(refreshServerStatus, intervalMs);
}

/** ポーリング停止 */
export function stopPolling(): void {
	if (intervalId !== null) {
		clearInterval(intervalId);
		intervalId = null;
	}
}
