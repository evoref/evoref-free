/**
 * ラルフループ制御 store
 *
 * Loop Console の制御バー / タスクパネルが共有する project_id とループ状態、
 * および backend `/api/loop/*` を叩く薄いアクションを提供する。
 * 観測 store ([loop_events.ts]) とは独立 (あちらは SSE 観測、こちらは制御)。
 */
import { get, writable } from 'svelte/store';
import {
	startLoop,
	stopLoop,
	pauseLoop,
	resumeLoop,
	getLoopStatus,
	type LoopStateInfo
} from '$lib/free/api';
import { ApiError } from '$lib/free/api';
import { addToast } from '$lib/free/stores/toast';

/** 操作対象の project_id (制御バー入力・タスクパネル共有) */
export const projectId = writable<string>('');

/** 直近に取得したループ状態 (未取得時 null) */
export const loopState = writable<LoopStateInfo | null>(null);

/** API 呼び出し中フラグ (ボタン二重押下防止) */
export const busy = writable<boolean>(false);

/** ApiError なら i18n キー優先、無ければ message を detail に載せて error toast */
function notifyError(e: unknown): void {
	if (e instanceof ApiError && e.i18nKey) {
		addToast({ type: 'error', i18nKey: e.i18nKey, params: e.context as Record<string, string | number> });
		return;
	}
	const detail = e instanceof Error ? e.message : String(e);
	addToast({ type: 'error', i18nKey: 'loop.control.action_failed', params: { detail } });
}

/** ループ状態を取得して store に反映する */
export async function refreshStatus(): Promise<void> {
	try {
		const res = await getLoopStatus();
		loopState.set(res.state);
	} catch (e) {
		notifyError(e);
	}
}

/** ループを起動する (project_id 空なら警告して no-op) */
export async function start(): Promise<void> {
	const pid = get(projectId).trim();
	if (!pid) {
		addToast({ type: 'warning', i18nKey: 'loop.control.need_project' });
		return;
	}
	busy.set(true);
	try {
		const res = await startLoop(pid);
		loopState.set(res.state);
	} catch (e) {
		notifyError(e);
	} finally {
		busy.set(false);
	}
}

/** ループを停止する */
export async function stop(): Promise<void> {
	busy.set(true);
	try {
		const res = await stopLoop();
		loopState.set(res.state);
	} catch (e) {
		notifyError(e);
	} finally {
		busy.set(false);
	}
}

/** ループを一時停止する */
export async function pause(): Promise<void> {
	busy.set(true);
	try {
		const res = await pauseLoop();
		loopState.set(res.state);
	} catch (e) {
		notifyError(e);
	} finally {
		busy.set(false);
	}
}

/** ループを再開する */
export async function resume(): Promise<void> {
	busy.set(true);
	try {
		const res = await resumeLoop();
		loopState.set(res.state);
	} catch (e) {
		notifyError(e);
	} finally {
		busy.set(false);
	}
}
