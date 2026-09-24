import { writable } from 'svelte/store';
import { isPro } from '$lib/edition';
import type {
	ComponentStatus,
	DataHealthInfo,
	DebugStatusInfo,
	MemoryStats,
	ServerName
} from '$lib/free/api';
import { getStatus } from '$lib/free/api';
import { handleApiCall } from '$lib/free/utils/error';
import { instanceName, appVersion } from '$lib/free/stores/app';
import { createPoller } from '$lib/free/stores/_polling';
import { addToast } from '$lib/free/stores/toast';

export interface ServerState {
	/** バックエンドに到達可能か */
	backendOnline: boolean;
	/** 各コンポーネント (base / embed) */
	components: ComponentStatus[];
	/** デバッグ情報 */
	debug: DebugStatusInfo | null;
	/** メモリ統計 */
	memory: MemoryStats | null;
	/** データ根の状態 (readonly なら記憶・学習・履歴・設定を保存しない) */
	dataHealth: DataHealthInfo | null;
}

const initial: ServerState = {
	backendOnline: false,
	components: [],
	debug: null,
	memory: null,
	dataHealth: null
};

/** readonly の通知はセッションで 1 回だけ出す (入力欄の表示は常時) */
let readonlyNotified = false;
/** エディション切替・ビルドとの食い違いの通知もセッションで 1 回だけ */
let editionSwitchNotified = false;
let editionMismatchNotified = false;

export const serverState = writable<ServerState>(initial);

/**
 * バックエンドが報告するエディション (/api/status の edition)。未取得なら null。
 * ビルド時の isPro が Pro でもバックエンドが free なら Pro の導線を出さない。
 */
export const backendEdition = writable<string | null>(null);

/**
 * 現在 start/stop 操作中のサーバー名 (ModelServerControl のスピナー表示用)
 *
 * コンポーネントローカルの $state にすると、設定タブ切替やルート遷移で
 * ModelServerControl が再マウントされた瞬間にスピナーが消えてしまうため、
 * グローバルストアに保持して再マウントを跨いでも維持する。
 */
export const busyServer = writable<ServerName | null>(null);

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
			memory: status.memory ?? null,
			dataHealth: status.data_health ?? null
		});
		backendEdition.set(status.edition ?? null);
		if (status.data_health?.readonly && !readonlyNotified) {
			readonlyNotified = true;
			addToast({ type: 'error', i18nKey: 'chat.data_readonly_toast', duration: 0 });
		}
		const switchedTo = status.data_health?.edition_switched_to;
		if (switchedTo && !editionSwitchNotified) {
			editionSwitchNotified = true;
			addToast({
				type: 'info',
				i18nKey:
					switchedTo === 'free' ? 'chat.edition_switched_to_free' : 'chat.edition_switched_to_pro',
				duration: 0
			});
		}
		if (isPro && status.edition === 'free' && !editionMismatchNotified) {
			editionMismatchNotified = true;
			addToast({ type: 'warning', i18nKey: 'chat.edition_build_mismatch', duration: 0 });
		}
	} else {
		serverState.set({
			backendOnline: false,
			components: [],
			debug: null,
			memory: null,
			dataHealth: null
		});
	}
}

const _poller = createPoller(refreshServerStatus, 30_000);

/** ポーリング開始（デフォルト30秒間隔） */
export function startPolling(intervalMs?: number): void {
	_poller.start(intervalMs);
}

/** ポーリング停止 */
export function stopPolling(): void {
	_poller.stop();
}
