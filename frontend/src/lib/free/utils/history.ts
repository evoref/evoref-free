/** 会話履歴ヘルパー関数 */

import type { SessionSummary, DateGroup } from '$lib/free/types/history';

const MS_PER_DAY = 86_400_000;

/**
 * セッション一覧を日付グループに分類する。
 * 「今日」「昨日」「それ以前（日付文字列）」の3区分。
 */
export function groupByDate(items: SessionSummary[]): DateGroup[] {
	const now = new Date();
	const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
	const yesterday = new Date(today.getTime() - MS_PER_DAY);

	const groups = new Map<string, SessionSummary[]>();

	for (const s of items) {
		const d = new Date(s.started_at);
		const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
		let label: string;
		if (day.getTime() === today.getTime()) {
			label = 'today';
		} else if (day.getTime() === yesterday.getTime()) {
			label = 'yesterday';
		} else {
			label = d.toLocaleDateString();
		}
		if (!groups.has(label)) groups.set(label, []);
		groups.get(label)!.push(s);
	}

	return Array.from(groups.entries()).map(([label, sessions]) => ({ label, sessions }));
}

/** 秒を分に変換（最小1分） */
export function formatDuration(sec: number): string {
	return String(Math.max(1, Math.round(sec / 60)));
}

/** セッションの表示テキストを返す（サマリ → 最初のユーザ発話 → フォールバック） */
export function sessionDisplayText(s: SessionSummary, fallback: string): string {
	return s.summary || s.first_user_preview || fallback;
}

/**
 * 検索マッチプレビューを返す。
 * クエリが空、matched_previewが無い、またはsummaryにクエリが含まれる場合はnull。
 */
export function sessionMatchedPreview(s: SessionSummary, query: string): string | null {
	if (!query || !s.matched_preview) return null;
	if (s.summary && s.summary.toLowerCase().includes(query.toLowerCase())) return null;
	return s.matched_preview;
}
