import { writable, derived, get, type Writable } from 'svelte/store';
import type { Component } from 'svelte';
import type { ThemeActivateResponse, ThemesListResponse, ThemeListItem } from '$lib/free/api';

export type ColorMode = 'light' | 'dark';

/** ThemeListItem の再エクスポート（後方互換エイリアス） */
export type ThemeInfo = ThemeListItem;

/** ColorMode の型ガード */
export function isColorMode(v: string): v is ColorMode {
	return v === 'light' || v === 'dark';
}

/** スロット名の型 */
export type SlotName = 'chat_header' | 'message_prefix' | 'input_suffix' | 'sidebar_widget';

/** スロットコンポーネントの状態 */
export type ThemeSlots = Record<SlotName, Component | null>;

export interface EditorConfig {
	show_line_numbers: boolean;
	word_wrap: boolean;
	tab_size: number;
	font_family: string;
	font_size: number;
	line_height: number;
	show_toolbar: boolean;
	show_active_line: boolean;
	/** 新規タブ / ファイル読込時の既定文字コード (config.editor 由来、テーマは持たない) */
	default_encoding: string;
	/** 新規タブの既定改行コード (config.editor 由来、テーマは持たない) */
	default_line_ending: 'lf' | 'crlf' | 'cr';
	/** シンタックスハイライトを有効にする言語 (config.editor 由来、テーマは持たない) */
	highlight_languages: string[];
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
	create: {
		pane_ratio: [number, number];
		pane_direction: 'horizontal' | 'vertical';
		editor: EditorConfig;
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

export const defaultEditorConfig: EditorConfig = {
	show_line_numbers: true,
	word_wrap: false,
	tab_size: 4,
	font_family: 'monospace',
	font_size: 14,
	line_height: 1.5,
	show_toolbar: true,
	show_active_line: true,
	default_encoding: 'utf-8',
	default_line_ending: 'lf',
	highlight_languages: [
		'markdown', 'python', 'javascript', 'typescript',
		'json', 'html', 'css', 'yaml', 'xml', 'sql', 'php'
	]
};

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
	create: {
		pane_ratio: [50, 50],
		pane_direction: 'horizontal',
		editor: { ...defaultEditorConfig }
	},
	dashboard: {
		panels: ['learning_status', 'lora_versions', 'performance_chart', 'rag_stats'],
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

/** サイドバー幅（CSS用） */
export const sidebarWidth = derived(
	[layout, sidebarCollapsed],
	([$layout, $collapsed]) => {
		if ($layout.sidebar.position === 'hidden' || $collapsed) return 0;
		return $layout.sidebar.width;
	}
);

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

	const ed = config.create.editor;
	root.style.setProperty('--editor-font-family', ed.font_family);
	root.style.setProperty('--editor-font-size', ed.font_size + 'px');
	root.style.setProperty('--editor-line-height', String(ed.line_height));
	root.style.setProperty('--editor-tab-size', String(ed.tab_size));
}

/** レイアウト設定をDOMに反映しストアを更新する */
export function applyLayout(config: LayoutConfig): void {
	// editor が省略されている場合はデフォルト値で補完
	if (!config.create.editor) {
		config.create.editor = { ...defaultEditorConfig };
	}
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

// ── スロットコンポーネント管理 ──

const defaultSlots: ThemeSlots = {
	chat_header: null,
	message_prefix: null,
	input_suffix: null,
	sidebar_widget: null
};

/** テーマスロットコンポーネント */
export const themeSlots: Writable<ThemeSlots> = writable({ ...defaultSlots });

/** 全スロットをクリア */
export function clearSlotComponents(): void {
	themeSlots.set({ ...defaultSlots });
}

/**
 * テーマのスロットコンポーネントを動的ロード
 * trusted でない場合は全スロットを null にする
 */
export async function loadSlotComponents(
	currentThemeId: string,
	slotsConfig: Record<string, string | null>,
	trusted: boolean
): Promise<void> {
	if (!trusted) {
		clearSlotComponents();
		return;
	}

	const loaded: ThemeSlots = { ...defaultSlots };
	const slotNames: SlotName[] = ['chat_header', 'message_prefix', 'input_suffix', 'sidebar_widget'];

	for (const slot of slotNames) {
		const componentFile = slotsConfig[slot];
		if (!componentFile) continue;

		// .svelte → .js に変換（プリコンパイル済み規約）
		const jsFile = componentFile.replace(/\.svelte$/, '.js');
		try {
			const module = await import(
				/* @vite-ignore */ `/api/themes/${currentThemeId}/components/${jsFile}`
			);
			if (module.default) {
				loaded[slot] = module.default;
			}
		} catch (e) {
			console.warn(`Failed to load slot component: ${slot} (${jsFile})`, e);
		}
	}

	themeSlots.set(loaded);
}

/**
 * テーマをアクティベートし、CSS / レイアウト / スロットコンポーネントを一括適用
 * @returns アクティベート結果。信頼確認が必要な場合は呼び出し元で処理する
 */
export async function activateTheme(
	targetThemeId: string,
	targetColorMode?: string
): Promise<void> {
	const { activateThemeApi } = await import('$lib/free/api');
	const result: ThemeActivateResponse = await activateThemeApi(targetThemeId, targetColorMode);

	// CSS 適用
	applyThemeColors(result.colors);

	// レイアウト適用（slots キーを除外して LayoutConfig に変換）
	if (result.gui_layout) {
		const { slots: _slots, ...layoutFields } = result.gui_layout;
		applyLayout({
			...layoutFields,
			create: {
				...layoutFields.create,
				// テーマは config.editor 由来のフィールド (encoding / line ending /
				// highlight_languages) を持たないため、既定値を土台に重ねる。
				editor: { ...defaultEditorConfig, ...layoutFields.create.editor },
			},
		});
	}

	// ストア更新
	themeId.set(result.theme_id);
	colorMode.set(result.color_mode);

	// スロットコンポーネントロード
	if (result.slots) {
		await loadSlotComponents(result.theme_id, result.slots, true);
	} else {
		clearSlotComponents();
	}
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
		// 下の「テーマなし」処理ごとスキップされる (= 旧テーマの CSS/スロットが残る)。
		const installed = data.themes.some((t) => t.theme_id === activeId);
		if (activeId && installed) {
			// テーマがアクティブな場合は CSS / layout / スロットを適用
			await activateTheme(activeId, isColorMode(initialMode) ? initialMode : 'dark');
		} else {
			// テーマなし状態: 動的CSS注入を除去し app.css のデフォルトで表示
			clearThemeColors();
			clearSlotComponents();
			themeId.set('');
		}
	} catch (e) {
		// Backend not running — use defaults
		console.warn('[Theme] Failed to initialize theme from backend, using defaults', e);
	}
}
