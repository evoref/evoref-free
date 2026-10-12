import { writable, get } from 'svelte/store';
import type { ThemeActivateResponse, ThemesListResponse, ThemeListItem } from '$lib/free/api';

export type ColorMode = 'light' | 'dark';

/** ThemeListItem の再エクスポート（後方互換エイリアス） */
export type ThemeInfo = ThemeListItem;

/** ColorMode の型ガード */
export function isColorMode(v: string): v is ColorMode {
	return v === 'light' || v === 'dark';
}

export interface LayoutConfig {
	sidebar: {
		position: 'left' | 'right' | 'hidden';
		width: number;
		collapsible: boolean;
		default_collapsed: boolean;
	};
	chat: {
		message_style: 'bubble' | 'flat' | 'compact';
		max_width: number;
		show_timestamps: boolean;
		show_agentic_steps: 'expanded' | 'collapsed' | 'hidden';
	};
	dashboard: {
		panels: string[];
		grid_columns: number;
	};
	global: {
		font_size_base: number;
		border_radius: number;
		spacing_unit: number;
		animation: boolean;
	};
}

const defaultLayout: LayoutConfig = {
	sidebar: {
		position: 'left',
		width: 180,
		collapsible: true,
		default_collapsed: false
	},
	chat: {
		message_style: 'bubble',
		max_width: 800,
		show_timestamps: true,
		show_agentic_steps: 'collapsed'
	},
	dashboard: {
		panels: ['learning_status', 'rag_stats'],
		grid_columns: 2
	},
	global: {
		font_size_base: 14,
		border_radius: 8,
		spacing_unit: 4,
		animation: true
	}
};

/** カラーモード (light/dark) */
export const colorMode = writable<ColorMode>('dark');

/** レイアウト設定 */
export const layout = writable<LayoutConfig>(defaultLayout);

/** サイドバーの折りたたみ状態 */
export const sidebarCollapsed = writable<boolean>(false);

/** テーマID（空文字列ならテーマ未適用） */
export const themeId = writable<string>('');

/** テーマ一覧 */
export const availableThemes = writable<ThemeInfo[]>([]);

/** カラーモードの切り替え（現在のテーマを���しいカラーモードで再���用） */
export async function toggleColorMode(): Promise<void> {
	const newMode: ColorMode = get(colorMode) === 'dark' ? 'light' : 'dark';
	const currentThemeId = get(themeId);
	if (currentThemeId) {
		await activateTheme(currentThemeId, newMode);
	} else {
		// テーマなし状態: app.css のカラーモードのみ切替
		colorMode.set(newMode);
	}
}

/** レイアウト CSS 変数を DOM に設定する（副作用のみ） */
export function setLayoutCssVariables(config: LayoutConfig): void {
	const root = document.documentElement;
	root.style.setProperty('--sidebar-width', config.sidebar.width + 'px');
	root.style.setProperty('--font-size-base', config.global.font_size_base + 'px');
	root.style.setProperty('--border-radius', config.global.border_radius + 'px');
	root.style.setProperty('--spacing-unit', config.global.spacing_unit + 'px');
}

function isRecord(v: unknown): v is Record<string, unknown> {
	return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** gui-layout.json (一部のキーだけでもよい) を既定値と節ごとに深くマージする */
export function mergeLayout(partial: unknown): LayoutConfig {
	const src = isRecord(partial) ? partial : {};
	const section = <K extends keyof LayoutConfig>(key: K): LayoutConfig[K] => {
		const value = src[key];
		return { ...defaultLayout[key], ...(isRecord(value) ? value : {}) } as LayoutConfig[K];
	};
	return {
		...src,
		sidebar: section('sidebar'),
		chat: section('chat'),
		dashboard: section('dashboard'),
		global: section('global')
	} as LayoutConfig;
}

/** CSS のエスケープ (`\69` / `\i` / 行継続) を文字に戻す。判定用 */
function decodeCssEscapes(text: string): string {
	return text.replace(/\\(?:([0-9a-fA-F]{1,6})\s?|(\r\n|[\n\r\f])|([\s\S]))/g, (_m, hex, nl, ch) => {
		if (hex) {
			const cp = parseInt(hex, 16);
			return cp > 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : '�';
		}
		return nl ? '' : ch;
	});
}

/** 文 / 宣言 1 つ分の CSS が外部を読む参照 (@import・data: 以外の url() / image-set()) を含むか */
function hasExternalReference(segment: string): boolean {
	const text = decodeCssEscapes(segment);
	if (/@import/i.test(text)) return true;
	for (const m of text.matchAll(/url\(\s*(['"]?)\s*([^'")]*)/gi)) {
		if (!/^data:/i.test(m[2].trim())) return true;
	}
	if (/image-set\(/i.test(text)) {
		for (const m of text.matchAll(/(['"])([^'"]*)\1/g)) {
			if (!/^data:/i.test(m[2].trim())) return true;
		}
	}
	return false;
}

/**
 * テーマの CSS から外部を読む参照を落とす (@import と data: 以外の url() / image-set())。
 * テーマはすべて同じ扱いで、例外は設けない (docs/c_11 §1)。
 * `;` `{` `}` で区切った 1 片ずつ判定し、該当する片だけを除く (区切りは残す)。
 */
export function sanitizeThemeCss(cssText: string): string {
	return cssText
		.split(/([;{}])/)
		.map((part) => (part === ';' || part === '{' || part === '}' || !hasExternalReference(part) ? part : ''))
		.join('');
}

/** レイアウト設定をDOMに反映しストアを更新する */
export function applyLayout(config: LayoutConfig): void {
	setLayoutCssVariables(config);
	layout.set(config);
}

/** テーマ CSS を動的注入 */
export function applyThemeColors(cssText: string): void {
	const STYLE_ID = 'evoref-theme-colors';
	let styleEl = document.getElementById(STYLE_ID) as HTMLStyleElement | null;
	if (!styleEl) {
		styleEl = document.createElement('style');
		styleEl.id = STYLE_ID;
	}
	styleEl.textContent = cssText;
	// 常に <head> 末尾に配置して app.css より後に来るようにする
	// （appendChild は既存要素を末尾に移動する）
	document.head.appendChild(styleEl);
}

/** テーマ CSS の動的注入を除去（テーマなし状態で app.css のデフォルトに戻す） */
export function clearThemeColors(): void {
	const STYLE_ID = 'evoref-theme-colors';
	const styleEl = document.getElementById(STYLE_ID);
	if (styleEl) {
		styleEl.remove();
	}
}

/**
 * テーマをアクティベートし、CSS / レイアウトを一括適用
 */
export async function activateTheme(
	targetThemeId: string,
	targetColorMode?: string
): Promise<void> {
	const { activateThemeApi } = await import('$lib/free/api');
	const result: ThemeActivateResponse = await activateThemeApi(targetThemeId, targetColorMode);

	// CSS 適用 (外部へ通信する参照は常に落とす)
	applyThemeColors(sanitizeThemeCss(result.colors));

	// レイアウト適用（欠けたキーは既定値で埋める）
	if (result.gui_layout) {
		applyLayout(mergeLayout(result.gui_layout));
	}

	// ストア更新
	themeId.set(result.theme_id);
	colorMode.set(result.color_mode);
}

/**
 * アプリ起動時にバックエンドからテーマ状態を取得して適用
 */
export async function initTheme(): Promise<void> {
	try {
		const { getThemes } = await import('$lib/free/api');
		const data: ThemesListResponse = await getThemes();
		availableThemes.set(data.themes);

		const activeId = data.active_theme_id ?? '';
		const initialMode = data.color_mode ?? 'dark';
		colorMode.set(isColorMode(initialMode) ? initialMode : 'dark');

		// active_theme_id は config の `theme.active` をそのまま返すため、該当テーマが
		// 未インストールでも空にならない (テーマ 0 件でも既定の "fallback" が返る)。
		// 存在確認をせずに activateTheme を呼ぶと 404 で throw し、catch へ抜けて
		// 下の「テーマなし」処理ごとスキップされる (= 旧テーマの CSS が残る)。
		const installed = data.themes.some((t) => t.theme_id === activeId);
		if (activeId && installed) {
			// テーマがアクティブな場合は CSS / layout を適用
			await activateTheme(activeId, isColorMode(initialMode) ? initialMode : 'dark');
		} else {
			// テーマなし状態: 動的CSS注入を除去し app.css のデフォルトで表示
			clearThemeColors();
			themeId.set('');
		}
	} catch (e) {
		// Backend not running — use defaults
		console.warn('[Theme] Failed to initialize theme from backend, using defaults', e);
	}
}
