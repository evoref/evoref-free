/** 会話履歴 API
 *
 * バックエンドの `/api/history/*` エンドポイントに対応する取得関数を提供する。
 * 旧 `routes/history/+page.svelte` 内の直接 fetch 呼び出しを集約
 *
 * - 一覧 / 詳細取得 / 削除のみを対象とする (検索 / 統計 / バッチ削除等は別途追加)
 * - レースコンディション対策の AbortSignal 受け渡しに対応
 */

import { request, requestVoid } from './_client';
import type { SessionDetailData, SessionSummary } from '$lib/free/types/history';

/** 一覧 API のクエリパラメータ */
export interface ListHistoryQuery {
	limit?: number;
	offset?: number;
	mode?: string;
	q?: string;
}

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
