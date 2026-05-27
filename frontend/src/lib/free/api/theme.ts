/** テーマ API */

import { request, requestVoid, requestFormData } from './_client';

/** テーマ一覧レスポンス */
export interface ThemesListResponse {
	themes: ThemeListItem[];
	active_theme_id: string;
	color_mode: string;
}

export interface ThemeListItem {
	theme_id: string;
	name: string;
	version: string;
	author: string;
	description: string;
	active: boolean;
	trusted: boolean;
	has_components: boolean;
	component_count: number;
	builtin: boolean;
	has_preview: boolean;
	has_cli_theme: boolean;
	has_cli_modules: boolean;
	cli_module_count: number;
	features: Record<string, boolean> | null;
}

/** GUI レイアウト + スロット情報（activate レスポンス用） */
export interface GuiLayoutResponse {
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
	coding: {
		pane_ratio: [number, number];
		pane_direction: 'horizontal' | 'vertical';
		editor?: {
			show_line_numbers: boolean;
			word_wrap: boolean;
			tab_size: number;
			font_family: string;
			font_size: number;
			line_height: number;
			show_toolbar: boolean;
			show_active_line: boolean;
		};
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
	slots?: Record<string, string | null>;
}

/** CLI テーマ設定（cli-theme.json の構造） */
export interface CliThemeConfig {
	prompt?: { marker?: string; step_indent?: number };
	colors?: Record<string, string>;
	status?: { show?: boolean; format?: string };
	welcome?: { show?: boolean };
}

/** テーマアクティベートレスポンス */
export interface ThemeActivateResponse {
	theme_id: string;
	name: string;
	color_mode: 'light' | 'dark';
	colors: string;
	gui_layout: GuiLayoutResponse | null;
	cli_theme: CliThemeConfig | null;
	slots: Record<string, string | null> | null;
	cli_modules: Record<string, string> | null;
	trusted: boolean;
	features: Record<string, boolean> | null;
}

/** テーマインストールレスポンス */
export interface ThemeInstallResponse {
	theme_id: string;
	name: string;
	version: string;
	author: string;
	trusted: boolean;
	widget_manifest?: { required_apis: string[] };
}

/** テーマ一覧取得 */
export async function getThemes(): Promise<ThemesListResponse> {
	return request<ThemesListResponse>('GET', '/themes');
}

/** テーマアクティベート */
export async function activateThemeApi(
	themeId: string,
	colorMode?: string
): Promise<ThemeActivateResponse> {
	return request<ThemeActivateResponse>('POST', '/themes/activate', {
		theme_id: themeId,
		color_mode: colorMode
	});
}

/** テーマを信頼済みとしてマーク */
export async function trustThemeApi(
	themeId: string
): Promise<{ theme_id: string; trusted: boolean }> {
	return request<{ theme_id: string; trusted: boolean }>('POST', `/themes/${themeId}/trust`);
}

/** テーマインストール (ZIP) */
export async function installThemeApi(file: File): Promise<ThemeInstallResponse> {
	const formData = new FormData();
	formData.append('file', file);
	return requestFormData<ThemeInstallResponse>('/themes/install', formData);
}

/** テーマアンインストール */
export async function uninstallThemeApi(themeId: string): Promise<void> {
	return requestVoid('DELETE', `/themes/${themeId}`);
}
