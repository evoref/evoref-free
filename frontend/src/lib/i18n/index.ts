import { writable, derived } from 'svelte/store';
import ja from './ja.json';
import en from './en.json';

const messages: Record<string, Record<string, unknown>> = { ja, en };

/** 現在のロケール */
export const locale = writable<string>('ja');

/**
 * ネストされたキーでメッセージを取得する関数を返すストア
 * 使用例: $t('chat.send') → "送信"
 * 補間:   $t('chat.rag_hits', { count: 3 }) → "RAG検索: 3件ヒット"
 */
export const t = derived(locale, ($locale) => {
	return (key: string, params?: Record<string, string | number>): string => {
		const keys = key.split('.');
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		let value: any = messages[$locale] ?? messages['ja'];
		for (const k of keys) {
			value = value?.[k];
		}
		if (typeof value !== 'string') {
			// フォールバック
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			value = keys.reduce((obj: any, k) => obj?.[k], messages['ja']);
		}
		if (typeof value !== 'string') return key;
		if (params) {
			for (const [k, v] of Object.entries(params)) {
				value = value.replace(`{${k}}`, String(v));
			}
		}
		return value;
	};
});

/** ロケール変更（config.yaml への保存APIも呼ぶ） */
export async function setLocale(newLocale: string): Promise<void> {
	locale.set(newLocale);
	await fetch('/api/config/locale', {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ locale: newLocale })
	});
}

/** 利用可能なロケール一覧 */
export const availableLocales = Object.keys(messages);
