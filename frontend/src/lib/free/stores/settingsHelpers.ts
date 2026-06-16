/**
 * 設定ストアの型安全アクセスヘルパー
 *
 * configData ストアから各セクションを型付きで取得するユーティリティ。
 * 各設定コンポーネントの $derived パターンを統一する。
 */

import type { ConfigData } from '$lib/types/settings';
import { updateField, updateNestedField, updateDeepNestedField } from './settings';

/**
 * configData から指定セクションを型安全に取得する
 *
 * @example
 * let llama = $derived(configSection($configData, 'llama'));
 * // llama は LlamaConfig 型
 */
export function configSection<K extends keyof ConfigData>(
	config: Record<string, Record<string, unknown>>,
	section: K
): ConfigData[K] {
	return (config[section] ?? {}) as ConfigData[K];
}

/** unknown を string に変換（フォールバック付き） */
export function asString(value: unknown, fallback = ''): string {
	if (typeof value === 'string') return value;
	if (value == null) return fallback;
	return String(value);
}

/** unknown を number に変換（フォールバック付き） */
export function asNumber(value: unknown, fallback = 0): number {
	if (typeof value === 'number' && !isNaN(value)) return value;
	if (typeof value === 'string') {
		const parsed = Number(value);
		if (!isNaN(parsed)) return parsed;
	}
	return fallback;
}

/** unknown を boolean に変換（フォールバック付き） */
export function asBoolean(value: unknown, fallback = false): boolean {
	if (typeof value === 'boolean') return value;
	return fallback;
}

// ── フィールドバインディングヘルパー ──

/** セクションとキーから onchange コールバックを生成 */
export function fieldUpdater(section: string, key: string) {
	return (value: unknown) => updateField(section, key, value);
}

/** ネストされたセクションの onchange コールバックを生成 */
export function nestedFieldUpdater(section: string, parent: string, key: string) {
	return (value: unknown) => updateNestedField(section, parent, key, value);
}

/** 2 段ネスト (section.parent.child.key) の onchange コールバックを生成 */
export function deepNestedFieldUpdater(
	section: string,
	parent: string,
	child: string,
	key: string
) {
	return (value: unknown) => updateDeepNestedField(section, parent, child, key, value);
}

// ── ビジネスロジックヘルパー ──

/** FadeMem の重み合計を検証する */
export function validateFadeWeights(
	alpha: number,
	beta: number,
	gamma: number
): { sum: number; valid: boolean } {
	const sum = alpha + beta + gamma;
	return { sum, valid: Math.abs(sum - 1.0) < 0.01 };
}

/** 配列の要素をトグルする（イミュータブル） */
export function toggleArrayItem<T>(array: T[], item: T, include: boolean): T[] {
	const result = [...array];
	if (include && !result.includes(item)) {
		result.push(item);
	} else if (!include) {
		const idx = result.indexOf(item);
		if (idx >= 0) result.splice(idx, 1);
	}
	return result;
}
