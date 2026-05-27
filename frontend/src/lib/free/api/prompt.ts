/** プロンプト管理 API (システムプロンプト + アシストプロンプト) */

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

// ── アシストプロンプト管理 API ──

/** アシストプロンプト一覧アイテム */
export interface AssistPromptListItem {
	task: string;
	version: number;
	source: string;
	updated_at: string;
	fitness_score: number;
	content_preview: string;
}

/** アシストプロンプト詳細 */
export interface AssistPromptDetail {
	task: string;
	version: number;
	source: string;
	updated_at: string;
	fitness_score: number;
	content: string;
}

/** アシストプロンプト一覧取得 */
export async function getAssistPrompts(): Promise<AssistPromptListItem[]> {
	return request<AssistPromptListItem[]>('GET', '/assist-prompts');
}

/** アシストプロンプト詳細取得 */
export async function getAssistPromptDetail(task: string): Promise<AssistPromptDetail> {
	return request<AssistPromptDetail>('GET', `/assist-prompts/${task}`);
}

/** アシストプロンプト更新 */
export async function updateAssistPrompt(
	task: string,
	content: string
): Promise<PromptUpdateResponse> {
	return request<PromptUpdateResponse>('PUT', `/assist-prompts/${task}`, { content });
}

/** アシストプロンプト履歴取得 */
export async function getAssistPromptHistory(task: string): Promise<PromptHistoryItem[]> {
	return request<PromptHistoryItem[]>('GET', `/assist-prompts/${task}/history`);
}

/** アシストプロンプトロールバック */
export async function rollbackAssistPrompt(
	task: string,
	version: number
): Promise<PromptUpdateResponse> {
	return request<PromptUpdateResponse>('POST', `/assist-prompts/${task}/rollback`, {
		version
	});
}
