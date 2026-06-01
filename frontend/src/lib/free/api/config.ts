/** 設定 API */

import { request } from './_client';

export interface ConfigFullResponse {
	config: Record<string, Record<string, unknown>>;
	sections: string[];
	edition: string;
}

export interface ConfigUpdateResponse {
	section: string;
	updated: boolean;
}

export interface ConfigValidateResponse {
	section: string;
	valid: boolean;
	errors: string[];
}

export interface PresetInfo {
	id: string;
}

export interface PresetListResponse {
	presets: PresetInfo[];
	current: string | null;
}

export interface PresetApplyResult {
	applied: string;
	changed_sections: string[];
	/** 起動引数変更で再起動が必要な llama-server 名 (base / assist / embed) */
	restart_servers: string[];
}

/** 全設定取得 */
export async function getConfig(): Promise<ConfigFullResponse> {
	return request<ConfigFullResponse>('GET', '/config');
}

/** 設定セクション更新 */
export async function updateConfigSection(
	section: string,
	data: Record<string, unknown>
): Promise<ConfigUpdateResponse> {
	return request<ConfigUpdateResponse>('PUT', `/config/${section}`, { data });
}

/** 設定セクションバリデーション */
export async function validateConfigSection(
	section: string,
	data: Record<string, unknown>
): Promise<ConfigValidateResponse> {
	return request<ConfigValidateResponse>('POST', `/config/${section}/validate`, { data });
}

/** パフォーマンスプリセット一覧 + 現在一致するプリセット */
export async function getPresets(): Promise<PresetListResponse> {
	return request<PresetListResponse>('GET', '/config/presets');
}

/** パフォーマンスプリセットを適用 */
export async function applyConfigPreset(id: string): Promise<PresetApplyResult> {
	return request<PresetApplyResult>('POST', `/config/presets/${id}/apply`);
}
