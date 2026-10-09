/** 会話履歴 API
 *
 * バックエンドの `/api/history/*` エンドポイントに対応する取得関数を提供する。
 * 旧 `routes/history/+page.svelte` 内の直接 fetch 呼び出しを集約
 *
 * - 一覧 (日付範囲つき) / 詳細取得 / 削除 / 統計 / 圧縮 / 一括削除を対象とする (`/search` は未使用)
 * - レースコンディション対策の AbortSignal 受け渡しに対応
 */

import { request, requestVoid } from './_client';
import type { SessionDetailData, SessionSummary } from '$lib/free/types/history';

/** 一覧 API のクエリパラメータ */
export interface ListHistoryQuery {
	limit?: number;
	offset?: number;
	mode?: string;
	/** 日付範囲の下限 (YYYY-MM-DD。その日の 0 時から) */
	from?: string;
	/** 日付範囲の上限 (YYYY-MM-DD。その日の終わりまで含む) */
	to?: string;
	q?: string;
}

/** 統計 API のレスポンス */
export interface HistoryStats {
	total_sessions: number;
	total_turns: number;
	total_size_mb: number;
	max_storage_mb: number;
	mode_counts: Record<string, number>;
	summary_generated: number;
}

/** 圧縮 API のレスポンス */
export interface HistoryCompactResult {
	compressed: number;
	summarized: number;
	deleted: number;
	freed_mb: number;
}

/** バックエンドの一覧 API の limit 上限 */
const HISTORY_LIST_MAX_LIMIT = 100;

/** 一覧 API のレスポンス */
export interface HistoryListResponse {
	total: number;
	sessions: SessionSummary[];
}

function buildQuery(params: Record<string, unknown>): string {
	const sp = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) {
		if (v === undefined || v === null || v === '') continue;
		sp.set(k, String(v));
	}
	const qs = sp.toString();
	return qs ? `?${qs}` : '';
}

/** 会話履歴一覧を取得 */
export async function listHistory(
	query: ListHistoryQuery = {},
	signal?: AbortSignal
): Promise<HistoryListResponse> {
	return request<HistoryListResponse>(
		'GET',
		`/history${buildQuery(query as Record<string, unknown>)}`,
		undefined,
		{ signal }
	);
}

/** 特定セッションの詳細を取得 */
export async function getHistoryDetail(sessionId: string): Promise<SessionDetailData> {
	return request<SessionDetailData>('GET', `/history/${sessionId}`);
}

/** セッションを削除 */
export async function deleteHistorySession(sessionId: string): Promise<void> {
	return requestVoid('DELETE', `/history/${sessionId}`);
}

/** 統計情報を取得 */
export async function getHistoryStats(): Promise<HistoryStats> {
	return request<HistoryStats>('GET', '/history/stats');
}

/** 保持ポリシーに基づく圧縮を実行 (古いセッションの圧縮・要約化・削除) */
export async function compactHistory(): Promise<HistoryCompactResult> {
	return request<HistoryCompactResult>('POST', '/history/compact');
}

/** 複数セッションを一括削除し、削除できた件数を返す */
export async function batchDeleteHistory(sessionIds: string[]): Promise<number> {
	const res = await request<{ deleted: number }>('DELETE', '/history', {
		session_ids: sessionIds
	});
	return res.deleted;
}

/** 絞り込み条件に合う全セッションの ID を集める (一覧 API を上限幅でたどる) */
export async function listHistorySessionIds(
	query: Omit<ListHistoryQuery, 'limit' | 'offset'> = {}
): Promise<string[]> {
	const ids: string[] = [];
	for (;;) {
		const page = await listHistory({
			...query,
			limit: HISTORY_LIST_MAX_LIMIT,
			offset: ids.length
		});
		ids.push(...page.sessions.map((s) => s.session_id));
		if (page.sessions.length === 0 || ids.length >= page.total) return ids;
	}
}
