/** プロンプト管理 API (システムプロンプト + 補助タスクプロンプト) */

import { request, requestVoid } from './_client';

/** プロンプト一覧アイテム */
export interface PromptListItem {
	mode: string;
	version: number;
	source: string;
	updated_at: string;
	content_preview: string;
}

/** プロンプト詳細 */
export interface PromptDetail {
	mode: string;
	version: number;
	source: string;
	updated_at: string;
	content: string;
}

/** プロンプト更新レスポンス */
export interface PromptUpdateResponse {
	status: string;
	version: number;
}

/** プロンプト履歴アイテム */
export interface PromptHistoryItem {
	version: number;
	file: string;
}

/** システムプロンプト一覧取得 */
export async function getPrompts(): Promise<PromptListItem[]> {
	return request<PromptListItem[]>('GET', '/prompts');
}

/** システムプロンプト詳細取得 */
export async function getPromptDetail(mode: string): Promise<PromptDetail> {
	return request<PromptDetail>('GET', `/prompts/${mode}`);
}

/** システムプロンプト更新 */
export async function updatePrompt(
	mode: string,
	content: string
): Promise<PromptUpdateResponse> {
	return request<PromptUpdateResponse>('PUT', `/prompts/${mode}`, { content });
}

/** システムプロンプト再読込み */
export async function reloadPrompt(mode: string): Promise<void> {
	return requestVoid('POST', `/prompts/${mode}/reload`);
}

/** システムプロンプト履歴取得 */
export async function getPromptHistory(mode: string): Promise<PromptHistoryItem[]> {
	return request<PromptHistoryItem[]>('GET', `/prompts/${mode}/history`);
}

/** システムプロンプトロールバック */
export async function rollbackPrompt(
	mode: string,
	version: number
): Promise<PromptUpdateResponse> {
	return request<PromptUpdateResponse>('POST', `/prompts/${mode}/rollback`, { version });
}

/** プロンプト言語切替レスポンス */
export interface PromptLocaleSwitchResponse {
	status: string;
	locale: string;
	versions: Record<string, number>;
	relearning_triggered: boolean;
}

/** プロンプト言語を切替 */
export async function switchPromptLocale(locale: string): Promise<PromptLocaleSwitchResponse> {
	return request<PromptLocaleSwitchResponse>('POST', '/prompts/locale', { locale });
}

// ── 補助タスクプロンプト管理 API ──

/** 補助タスクプロンプト一覧アイテム */
export interface AuxPromptListItem {
	task: string;
	version: number;
	source: string;
	updated_at: string;
	fitness_score: number;
	content_preview: string;
}

/** 補助タスクプロンプト詳細 */
export interface AuxPromptDetail {
	task: string;
	version: number;
	source: string;
	updated_at: string;
	fitness_score: number;
	content: string;
}

/** 補助タスクプロンプト一覧取得 */
export async function getAuxPrompts(): Promise<AuxPromptListItem[]> {
	return request<AuxPromptListItem[]>('GET', '/aux-prompts');
}

/** 補助タスクプロンプト詳細取得 */
export async function getAuxPromptDetail(task: string): Promise<AuxPromptDetail> {
	return request<AuxPromptDetail>('GET', `/aux-prompts/${task}`);
}

/** 補助タスクプロンプト更新 */
export async function updateAuxPrompt(
	task: string,
	content: string
): Promise<PromptUpdateResponse> {
	return request<PromptUpdateResponse>('PUT', `/aux-prompts/${task}`, { content });
}

/** 補助タスクプロンプト履歴取得 */
export async function getAuxPromptHistory(task: string): Promise<PromptHistoryItem[]> {
	return request<PromptHistoryItem[]>('GET', `/aux-prompts/${task}/history`);
}

/** 補助タスクプロンプトロールバック */
export async function rollbackAuxPrompt(
	task: string,
	version: number
): Promise<PromptUpdateResponse> {
	return request<PromptUpdateResponse>('POST', `/aux-prompts/${task}/rollback`, {
		version
	});
}
