/** ラルフループ制御 API (start / stop / pause / resume / status / tasks) */

import { request } from './_client';

/** LoopDriver の状態 DTO (backend: LoopStateInfo) */
export interface LoopStateInfo {
	running: boolean;
	project_id: string | null;
	started_at: number | null;
	iteration: number;
	stop_requested: boolean;
	last_picked_fact_id: string | null;
	last_picked_task_id: string | null;
	pause_requested: boolean;
	paused: boolean;
}

/** /api/loop/{start,stop,pause,resume,status} レスポンス */
export interface LoopStateResponse {
	state: LoopStateInfo;
}

/** task ファクトの一覧表示用 DTO (backend: TaskInfo) */
export interface TaskInfo {
	fact_id: string;
	task_id: string;
	title: string;
	description: string;
	depends_on: string[];
	salience: number;
	status: string;
	source_path: string | null;
	project_id: string;
	created_at: number;
	accessed_at: number;
}

/** /api/loop/tasks レスポンス */
export interface TaskListResponse {
	project_id: string;
	total: number;
	tasks: TaskInfo[];
	next_task_id: string | null;
}

/** /api/loop/tasks/import レスポンス */
export interface TaskImportResponse {
	project_id: string;
	imported: number;
	fact_ids: string[];
}

/** ループを起動して周回を開始する */
export async function startLoop(projectId: string): Promise<LoopStateResponse> {
	return request<LoopStateResponse>('POST', '/loop/start', { project_id: projectId });
}

/** ループを停止する (冪等) */
export async function stopLoop(): Promise<LoopStateResponse> {
	return request<LoopStateResponse>('POST', '/loop/stop');
}

/** 現サイクル完了後に一時停止する */
export async function pauseLoop(): Promise<LoopStateResponse> {
	return request<LoopStateResponse>('POST', '/loop/pause');
}

/** 一時停止から再開する */
export async function resumeLoop(): Promise<LoopStateResponse> {
	return request<LoopStateResponse>('POST', '/loop/resume');
}

/** ループの現在状態を取得する */
export async function getLoopStatus(): Promise<LoopStateResponse> {
	return request<LoopStateResponse>('GET', '/loop/status');
}

/** プロジェクトの task 一覧を取得する */
export async function listTasks(
	projectId: string,
	status?: string
): Promise<TaskListResponse> {
	const params = new URLSearchParams({ project_id: projectId });
	if (status) params.set('status', status);
	return request<TaskListResponse>('GET', `/loop/tasks?${params.toString()}`);
}

/** PRD JSON (文字列) を task ファクトとして投入する */
export async function importTasks(
	projectId: string,
	prdJson: string
): Promise<TaskImportResponse> {
	return request<TaskImportResponse>('POST', '/loop/tasks/import', {
		project_id: projectId,
		prd_json: prdJson
	});
}
