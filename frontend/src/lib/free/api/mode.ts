/** モード切替 API */

import { request } from './_client';

/** モード切替レスポンス */
export interface ModeSwitchResponse {
	mode: string;
	model_changed: boolean;
	restart_initiated: boolean;
	message: string;
}

/** モード切替 API */
export async function switchModeApi(mode: string): Promise<ModeSwitchResponse> {
	return request<ModeSwitchResponse>('POST', '/mode/switch', { mode });
}
