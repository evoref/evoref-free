/** 環境調整 (auto-tune) の項目表示用の整形 (SystemSettings) */

import type { AutoTuneItem } from '$lib/free/api';

/** 値の行に出さないキー (basis = 再計算の指紋、model_key = 内部の識別子)。manual は別枠のバッジ */
const HIDDEN_KEYS = new Set(['basis', 'model_key']);

export interface ValueRow {
	label: string;
	text: string;
	/** 0 = 項目直下、1 = 入れ子 (1 段だけ展開) */
	depth: 0 | 1;
}

export interface FormattedValue {
	/** スカラ値 (dict でないとき) */
	scalar: string | null;
	rows: ValueRow[];
	/** 利用者が config に明示したキー */
	manual: string[];
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
	return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** 数値は桁区切り + 小数 2 桁まで。その他は文字列化 */
export function formatScalar(v: unknown): string {
	if (v === null || v === undefined) return '—';
	if (typeof v === 'number') {
		return Number.isFinite(v)
			? v.toLocaleString('en-US', { maximumFractionDigits: 2 })
			: String(v);
	}
	if (typeof v === 'boolean') return v ? 'true' : 'false';
	return String(v);
}

/** 長い配列・深い入れ子の要約 (3 件までのスカラ配列はそのまま) */
function summarize(v: unknown): string {
	if (Array.isArray(v)) {
		return v.length <= 3 && v.every((x) => !isPlainObject(x) && !Array.isArray(x))
			? v.map(formatScalar).join(', ')
			: `[${v.length}]`;
	}
	if (isPlainObject(v)) return `{${Object.keys(v).length}}`;
	return formatScalar(v);
}

/** 項目の value を、スカラ or `キー: 値` の行 + 手動設定キーへ分解する */
export function formatItemValue(value: unknown): FormattedValue {
	if (!isPlainObject(value)) {
		return { scalar: Array.isArray(value) ? summarize(value) : formatScalar(value), rows: [], manual: [] };
	}
	const manual = Array.isArray(value.manual) ? value.manual.map(String) : [];
	const rows: ValueRow[] = [];
	for (const [k, v] of Object.entries(value)) {
		if (k === 'manual' || HIDDEN_KEYS.has(k)) continue;
		if (isPlainObject(v)) {
			rows.push({ label: k, text: '', depth: 0 });
			for (const [k2, v2] of Object.entries(v)) {
				if (HIDDEN_KEYS.has(k2)) continue;
				rows.push({ label: k2, text: summarize(v2), depth: 1 });
			}
		} else {
			rows.push({ label: k, text: summarize(v), depth: 0 });
		}
	}
	return { scalar: null, rows, manual };
}

export type StatusKind =
	| 'scheduled'
	| 'requires_stop'
	| 'requires_explicit'
	| 'no_samples'
	| 'manual'
	| 'failed'
	| 'not_auto'
	| 'restart'
	| 'applied';

/** 状態の分類 (reason を source / applied より優先する。唯一の判定点) */
export function classifyItem(item: Pick<AutoTuneItem, 'reason' | 'source' | 'applied' | 'requires_restart'>): StatusKind {
	switch (item.reason) {
		case 'scheduled':
		case 'requires_stop':
		case 'requires_explicit':
		case 'no_samples':
		case 'manual':
			return item.reason;
	}
	if (item.source === 'failed') return 'failed';
	if (!item.applied) return 'not_auto';
	return item.requires_restart ? 'restart' : 'applied';
}

/**
 * 環境調整が決めるキー (`auto` / null = 自動) の数値欄の表示値。自動なら null (欄に「auto」を出す)。
 * `missingIsAuto` はキーが無いときも自動か (スキーマの既定が auto / null のキー)、`fallback` は無いときの既定値。
 */
export function autoNumber(v: unknown, missingIsAuto: boolean, fallback: number): number | null {
	if (v === null || v === undefined) return missingIsAuto ? null : fallback;
	const n = Number(v); // 'auto' は NaN
	return Number.isFinite(n) ? n : null;
}
