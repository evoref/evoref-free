/** ProjectMap API (Free)
 *
 * 既存プロジェクトの code グラフ (c_16 §4.4) の状態表示と手動更新。
 * CLI の `evoref projectmap status` / `update` と同じ内容。
 */

import { request } from './_client';

/** root ごとの現在版の状態 (バックエンド `_root_status` 準拠) */
export interface ProjectMapRootStatus {
	root: string;
	package_id: string;
	/** 現在版 (まだ一度も作られていなければ null) */
	version: string | null;
	update_kind: string | null;
	languages: Record<string, number>;
	nodes: number;
	edges: number;
	/** 現在版の書き込み時刻 (ISO 8601 UTC)。版が無ければ null */
	written_at: string | null;
}

/** GET /api/rag/project_map */
export interface ProjectMapStatus {
	enabled: boolean;
	roots: ProjectMapRootStatus[];
}

/** POST /api/rag/project_map/update */
export interface ProjectMapUpdateResult {
	/** 更新した root の数 */
	updated_roots: number;
	running: boolean;
}

/** ProjectMap の状態を取得 */
export async function getProjectMapStatus(): Promise<ProjectMapStatus> {
	return request<ProjectMapStatus>('GET', '/rag/project_map');
}

/** ProjectMap を手動で更新する (数分掛かりうる。実行中なら 409) */
export async function updateProjectMap(): Promise<ProjectMapUpdateResult> {
	return request<ProjectMapUpdateResult>('POST', '/rag/project_map/update');
}
