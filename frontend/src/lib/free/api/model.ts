/**
 * Model migration API client
 *
 * - assist / embedding / reranker のコンポーネント切替・ロールバック
 */

import { request } from './_client';

export type ModelComponent = 'assist' | 'embedding' | 'reranker';

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
