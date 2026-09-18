/**
 * staged クリエイトの run 再接続 API (`GET /api/create/runs*`)
 *
 * リロード / タブ閉じで SSE が切れても、backend の run.json / events.jsonl に
 * 事実が残っているため、UI は SSE を再購読せずこの API 群をポーリングして
 * 表示を復元する (f_05 §4.5 / f_10 §7)。
 */

import { request } from './_client';
import type {
	CreateRunArtifactsResponse,
	CreateRunEventsResponse,
	CreateRunSummary
} from '$lib/free/types/createRuns';

/** セッションの run 一覧 (`started_at` 降順) */
export async function listCreateRuns(sessionId: string): Promise<CreateRunSummary[]> {
	const res = await request<{ runs: CreateRunSummary[] }>(
		'GET',
		`/create/runs?session_id=${encodeURIComponent(sessionId)}`
	);
	return res.runs;
}

/** run レコード 1 件 + 導出状態 */
export async function getCreateRun(runId: string): Promise<CreateRunSummary> {
	return request<CreateRunSummary>('GET', `/create/runs/${encodeURIComponent(runId)}`);
}

/** 追記イベントログの watermark 付き読み (再接続の口) */
export async function getCreateRunEvents(
	runId: string,
	after: number
): Promise<CreateRunEventsResponse> {
	return request<CreateRunEventsResponse>(
		'GET',
		`/create/runs/${encodeURIComponent(runId)}/events?after=${after}`
	);
}

/** run の成果物 (workspace の `src/**` + `SPEC.md` + `flowchart.md`) */
export async function getCreateRunArtifacts(runId: string): Promise<CreateRunArtifactsResponse> {
	return request<CreateRunArtifactsResponse>(
		'GET',
		`/create/runs/${encodeURIComponent(runId)}/artifacts`
	);
}
