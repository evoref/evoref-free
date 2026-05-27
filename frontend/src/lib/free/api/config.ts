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
