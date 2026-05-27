/**
 * 設定 API サービス層
 *
 * settings ストアから API 呼び出しロジックを分離し、
 * ストアが状態管理に専念できるようにする。
 */

import {
	getConfig,
	getStatus,
	updateConfigSection,
	validateConfigSection,
	switchPromptLocale as switchPromptLocaleApi,
	type ConfigFullResponse,
	type ConfigUpdateResponse,
	type ConfigValidateResponse,
	type StatusResponse
} from '$lib/free/api';

/** 設定全体を取得 */
export async function fetchConfig(): Promise<ConfigFullResponse> {
	return getConfig();
}

/** セクション単位で保存 */
export async function saveSection(
	section: string,
	data: Record<string, unknown>
): Promise<ConfigUpdateResponse> {
	return updateConfigSection(section, data);
}

/** セクション検証（保存なし） */
export async function validateSection(
	section: string,
	data: Record<string, unknown>
): Promise<ConfigValidateResponse> {
	return validateConfigSection(section, data);
}

/** サーバーステータスを取得 */
export async function fetchStatus(): Promise<StatusResponse> {
	return getStatus();
}

/**
 * プロンプトの言語を切り替える
 *
 * API 呼び出しを実行し、結果（再学習トリガーの有無）を返す。
 * UI 側の confirm ダイアログやトースト表示はコンポーネント側の責務。
 */
export async function performPromptLocaleSwitch(
	newLocale: string
): Promise<{ relearning: boolean }> {
	const result = await switchPromptLocaleApi(newLocale);
	return { relearning: result.relearning_triggered ?? false };
}
