/** エラーハンドリングユーティリティ */

import { addToast } from '$lib/free/stores/toast';
import { ApiError } from '$lib/free/api';

/**
 * API 呼び出しをラップし、エラー時にトースト通知 + コンソールログを行う
 *
 * @param fn - 実行する非同期関数
 * @param options.silent - true の場合トースト通知を抑制（コンソールログは出力）
 * @param options.fallbackKey - ApiError に i18n_key がない場合のフォールバックキー
 * @param options.fallback - エラー時に返すフォールバック値
 */
export async function handleApiCall<T>(
	fn: () => Promise<T>,
	options?: { silent?: boolean; fallbackKey?: string; fallback?: T }
): Promise<T | undefined> {
	try {
		return await fn();
	} catch (e) {
		const i18nKey =
			e instanceof ApiError && e.i18nKey
				? e.i18nKey
				: (options?.fallbackKey ?? 'common.error');

		if (options?.silent) {
			console.warn('[API Warning]', e);
		} else {
			addToast({ type: 'error', i18nKey });
			console.error('[API Error]', e);
		}
		return options?.fallback;
	}
}
