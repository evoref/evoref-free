import { writable, derived, get } from 'svelte/store';
import ja from './ja.json';
import en from './en.json';
import { getLocales } from '$lib/free/api';
import { performPromptLocaleSwitch } from '$lib/free/services/configService';
import { addToast } from '$lib/free/stores/toast';

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

/**
 * チャット応答言語 (i18n.prompt_locale) の永続化済み値のキャッシュ。
 * switchLocale() の冪等性判定 (無駄な再学習トリガー回避) に使う。
 */
export const promptLocale = writable<string>('ja');

/** prompt_locale 切替の進行中フラグ (Sidebar の select disable / スピナー用) */
export const promptLocaleSwitching = writable<boolean>(false);

/**
 * 起動時にバックエンドの実際のロケール値でストアを初期化する。
 * 失敗しても既定値 (ja) のまま動作を継続する (initTheme() と同型パターン)。
 */
export async function initLocale(): Promise<void> {
	try {
		const data = await getLocales();
		locale.set(data.current);
		promptLocale.set(data.prompt_locale);
	} catch (e) {
		console.warn('[i18n] Failed to initialize locale from backend, using defaults', e);
	}
}

/**
 * サイドバー専用のロケール切替。UI表示言語 (setLocale) に加え、
 * チャット応答言語 (prompt_locale) が変わる場合のみ POST /api/prompts/locale
 * を呼んで連動させる (既に同じ prompt_locale なら無駄な再学習トリガーを避ける)。
 *
 * prompt_locale の切替はプロンプトのアーカイブ + 再学習スケジュールを伴う
 * 破壊的操作のため、GeneralSettings.svelte の handlePromptLocaleChange と
 * 同じ確認ダイアログを挟む (2026-07-22 監査で判明: サイドバー経由だと
 * UI表示言語を変えるだけのつもりでも無警告でこの破壊的操作が実行されていた)。
 * UI表示言語自体 (setLocale) は非破壊的なので確認無しで即時反映する。
 */
export async function switchLocale(newLocale: string): Promise<void> {
	await setLocale(newLocale);
	if (get(promptLocale) === newLocale) return;
	if (!confirm(get(t)('settings.i18n.prompt_locale_confirm'))) return;

	promptLocaleSwitching.set(true);
	try {
		const { relearning } = await performPromptLocaleSwitch(newLocale);
		promptLocale.set(newLocale);
		addToast({
			type: 'success',
			i18nKey: relearning
				? 'settings.i18n.prompt_locale_switched_with_relearn'
				: 'settings.i18n.prompt_locale_switched'
		});
	} catch (e) {
		addToast({ type: 'error', i18nKey: 'settings.i18n.prompt_locale_failed' });
		console.error('[i18n] prompt locale switch failed', e);
	} finally {
		promptLocaleSwitching.set(false);
	}
}
