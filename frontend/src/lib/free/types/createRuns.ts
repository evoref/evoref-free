/** staged クリエイト run の型定義 (再接続 API、f_05 §4.5 / f_10 §7) */

/** 表示状態 (`derive_run_status` が読み出し時に導出。永続はしない) */
export type CreateRunStatus = 'working' | 'needs_input' | 'done' | 'failed' | 'cancelled' | 'timeout';

/** `GET /api/create/runs` / `GET /api/create/runs/{run_id}` の 1 件 */
export interface CreateRunSummary {
	run_id: string;
	session_id: string;
	request_id: string;
	mode: string;
	query: string;
	output_target: string;
	started_at: string;
	ended_at: string | null;
	activity_state: string;
	exit_kind: string | null;
	last_event_seq: number;
	status: CreateRunStatus;
	/** `status === 'needs_input'` のときの問い */
	question?: string;
	/** 前回の `blocked` run を引き継いで再開した場合の元 run_id */
	resume_of?: string;
}

/** `events.jsonl` の 1 レコード (`GET .../events?after=` の要素) */
export interface CreateRunEvent {
	seq: number;
	at: string;
	kind: string;
	payload: Record<string, unknown>;
}

/** `GET /api/create/runs/{run_id}/events?after=` のレスポンス */
export interface CreateRunEventsResponse {
	run_id: string;
	after: number;
	events: CreateRunEvent[];
	last_seq: number;
}

/** `GET /api/create/runs/{run_id}/artifacts` の 1 件 */
export interface CreateRunArtifact {
	path: string;
	language: string;
	content: string;
}

/** `GET /api/create/runs/{run_id}/artifacts` のレスポンス */
export interface CreateRunArtifactsResponse {
	run_id: string;
	artifacts: CreateRunArtifact[];
}
