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
	has_preview: boolean;
	has_cli_theme: boolean;
}

/** GUI レイアウト（activate レスポンス用） */
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
}

/** テーマインストールレスポンス */
export interface ThemeInstallResponse {
	theme_id: string;
	name: string;
	version: string;
	author: string;
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
