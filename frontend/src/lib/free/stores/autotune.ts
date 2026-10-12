/** 環境調整 (auto-tune) の実行・確認の共有ロジック (バナーとシステムタブが共用) */

import { writable } from 'svelte/store';
import {
	ApiError,
	decideAutoTune,
	getAutoTune,
	runAutoTune,
	type AutoTuneResponse
} from '$lib/free/api';
import { handleApiCall } from '$lib/free/utils/error';
import { refreshServerStatus } from './server';
import { addToast } from './toast';

/** GET /api/system/auto-tune の最新値 (未取得なら null) */
export const autoTuneDetail = writable<AutoTuneResponse | null>(null);

/** 進捗のポーリング間隔 (ms) */
export const AUTO_TUNE_POLL_MS = 1000;

/** 変わった軸がホスト名だけか (「変更なし」を選べる唯一の場合) */
export function isHostnameOnly(axes: string[] | undefined): boolean {
	return !!axes && axes.length === 1 && axes[0] === 'hostname';
}

export async function refreshAutoTune(): Promise<AutoTuneResponse | undefined> {
	const r = await handleApiCall(() => getAutoTune(), {
		silent: true,
		fallbackKey: 'autotune.load_failed'
	});
	if (r) autoTuneDetail.set(r);
	return r;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

let watching: Promise<void> | null = null;

/** 取得が続けて失敗したときに見届けを諦めるまでの回数 */
export const AUTO_TUNE_MAX_POLL_FAILURES = 5;

/**
 * state が running でなくなるまで間隔を置いて取得し、終わったら /api/status も更新する。
 *
 * 一度の取得失敗では止めない (止めると楽観的に立てた running のまま固まる)。続けて
 * 失敗したら諦め、手元の running を下ろして再実行できるようにする (走っていれば 409 で見届けに戻る)。
 */
export function watchAutoTune(intervalMs: number = AUTO_TUNE_POLL_MS): Promise<void> {
	if (watching) return watching;
	watching = (async () => {
		try {
			let failures = 0;
			for (;;) {
				const r = await refreshAutoTune();
				if (!r) {
					failures += 1;
					if (failures >= AUTO_TUNE_MAX_POLL_FAILURES) {
						autoTuneDetail.update((d) => (d && d.state === 'running' ? { ...d, state: 'idle' } : d));
						addToast({ type: 'error', i18nKey: 'autotune.load_failed' });
						break;
					}
				} else {
					failures = 0;
					if (r.state !== 'running') break;
				}
				await sleep(intervalMs);
			}
			await refreshServerStatus();
		} finally {
			watching = null;
		}
	})();
	return watching;
}

/**
 * 環境調整を始めて完了まで見届ける。始められた (または既に実行中だった) なら true。
 * 実行中 (409) は失敗ではなく、走っている実行をそのまま見届ける。
 * 既定は force=false (保存済みの結果がまだ有効な項目は測り直さない)。確認待ちの移行は先に
 * decision=accepted を送ってから呼ぶ (force で確認を飛ばさない)。
 */
export async function startAutoTune(
	opts: { only?: string[] | null; force?: boolean } = {}
): Promise<boolean> {
	const started = await handleApiCall(
		async () => {
			try {
				await runAutoTune({ only: opts.only ?? null, force: opts.force ?? false });
			} catch (e) {
				if (!(e instanceof ApiError && e.code === 'E0409')) throw e;
			}
			return true;
		},
		{ fallbackKey: 'autotune.run_failed' }
	);
	if (!started) return false;
	autoTuneDetail.update((d) => (d ? { ...d, state: 'running' } : d));
	await watchAutoTune();
	return true;
}

/** 確認に答え、/api/status と詳細を更新する。送れたら true */
export async function decideAndRefresh(
	decision: 'accepted' | 'declined' | 'unchanged'
): Promise<boolean> {
	const sent = await handleApiCall(
		async () => {
			await decideAutoTune(decision);
			return true;
		},
		{ fallbackKey: 'autotune.decision_failed' }
	);
	if (!sent) return false;
	await Promise.all([refreshServerStatus(), refreshAutoTune()]);
	return true;
}
