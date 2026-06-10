/**
 * Model migration API client
 *
 * - base モデル切替 (migrate / rollback / reload / process restart)
 * - assist / embedding のコンポーネント切替・ロールバック
 */

import { request } from './_client';

// ── ベースモデル migrate ──

export interface BaseMigrateRequest {
	new_model_path: string;
	try_lora?: boolean;
	regenerate_context?: boolean;
	dry_run?: boolean;
}

export interface MigrateDataSummary {
	memory_notes: number;
	experience_entries: number;
	perplexity_reset: number;
	rag_chunks: number;
	cartridges: number;
	prompts_modes: string[];
}

export interface BaseMigrateResponse {
	dry_run: boolean;
	old_model: string;
	new_model: string;
	lora_action: string;
	data_summary: MigrateDataSummary;
	calibration: { eval_questions: number; baseline_ppl_avg: number } | null;
	recommendations: string[];
}

export interface BaseRollbackResponse {
	rolled_back_to: string;
	lora_restored: boolean;
}

export interface ReloadResponse {
	reloaded: boolean;
	model_id: string;
	chat_template: string;
	has_system_role: boolean;
}

export interface ProcessRestartResponse {
	component: string;
	restarted: boolean;
	host?: string;
	port?: number;
	pid?: number;
}

/** ベースモデルの migrate を実行する (dry_run=true でプレビュー) */
export async function migrateBaseModel(
	body: BaseMigrateRequest
): Promise<BaseMigrateResponse> {
	return request<BaseMigrateResponse>('POST', '/model/migrate', body);
}

/** ベースモデルの migrate をロールバックする */
export async function rollbackBaseModel(
	target_model?: string
): Promise<BaseRollbackResponse> {
	return request<BaseRollbackResponse>(
		'POST',
		'/model/rollback',
		target_model ? { target_model } : {}
	);
}

/** migrate 後に llama-server へ再接続し model_state を同期する */
export async function reloadModel(): Promise<ReloadResponse> {
	return request<ReloadResponse>('POST', '/model/reload', {});
}

/** base llama-server プロセスを再起動する (LlamaProcessManager 管理下のみ) */
export async function restartBaseProcess(): Promise<ProcessRestartResponse> {
	return request<ProcessRestartResponse>('POST', '/model/process/base/restart', {});
}

export type ModelComponent = 'assist' | 'embedding';

export interface ComponentMigrateRequest {
	new_model_path: string;
	dry_run?: boolean;
	auto_restart?: boolean;
}

export interface ComponentMigrateResponse {
	component: ModelComponent;
	dry_run: boolean;
	old_model: string;
	new_model: string;
	restarted: boolean;
	recommendations: string[];
}

export interface ComponentRollbackResponse {
	component: ModelComponent;
	rolled_back_to: string;
}

export async function migrateComponent(
	component: ModelComponent,
	body: ComponentMigrateRequest
): Promise<ComponentMigrateResponse> {
	return request<ComponentMigrateResponse>(
		'POST',
		`/model/${component}/migrate`,
		body
	);
}

export async function rollbackComponent(
	component: ModelComponent,
	target_model?: string
): Promise<ComponentRollbackResponse> {
	return request<ComponentRollbackResponse>(
		'POST',
		`/model/${component}/rollback`,
		target_model ? { target_model } : {}
	);
}
