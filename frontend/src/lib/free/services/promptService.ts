/**
 * プロンプト API サービス層
 *
 * system / aux プロンプトのカテゴリ分岐をアダプタパターンで一本化し、
 * prompts ストアから分岐ロジックを除去する。
 */

import {
	getPromptDetail,
	updatePrompt,
	reloadPrompt,
	getPromptHistory,
	rollbackPrompt,
	getAuxPromptDetail,
	updateAuxPrompt,
	getAuxPromptHistory,
	rollbackAuxPrompt,
	type PromptDetail,
	type PromptHistoryItem
} from '$lib/free/api';

/** fitness_score を含むプロンプト詳細（system / aux 統一型） */
export interface PromptDetailWithScore extends PromptDetail {
	fitness_score?: number;
}

/** プロンプト種別 */
export type PromptCategory = 'system' | 'aux';

/** カテゴリ別 API アダプタ */
export interface PromptAdapter {
	fetchDetail(id: string): Promise<PromptDetailWithScore>;
	save(id: string, content: string): Promise<void>;
	reload(id: string): Promise<void>;
	fetchHistory(id: string): Promise<PromptHistoryItem[]>;
	rollback(id: string, version: number): Promise<void>;
}

/** システムプロンプト用アダプタ */
const systemAdapter: PromptAdapter = {
	async fetchDetail(id: string): Promise<PromptDetailWithScore> {
		return getPromptDetail(id);
	},

	async save(id: string, content: string): Promise<void> {
		await updatePrompt(id, content);
	},

	async reload(id: string): Promise<void> {
		await reloadPrompt(id);
	},

	async fetchHistory(id: string): Promise<PromptHistoryItem[]> {
		return getPromptHistory(id);
	},

	async rollback(id: string, version: number): Promise<void> {
		await rollbackPrompt(id, version);
	}
};

/** 補助タスクプロンプト用アダプタ */
const auxAdapter: PromptAdapter = {
	async fetchDetail(id: string): Promise<PromptDetailWithScore> {
		const detail = await getAuxPromptDetail(id);
		return {
			mode: detail.task,
			version: detail.version,
			source: detail.source,
			updated_at: detail.updated_at,
			content: detail.content,
			fitness_score: detail.fitness_score
		};
	},

	async save(id: string, content: string): Promise<void> {
		await updateAuxPrompt(id, content);
	},

	async reload(_id: string): Promise<void> {
		// 補助タスクプロンプトは reload API なし（一覧再取得で対応）
	},

	async fetchHistory(id: string): Promise<PromptHistoryItem[]> {
		return getAuxPromptHistory(id);
	},

	async rollback(id: string, version: number): Promise<void> {
		await rollbackAuxPrompt(id, version);
	}
};

/** カテゴリに応じた API アダプタを返す */
export function getPromptAdapter(category: PromptCategory): PromptAdapter {
	return category === 'system' ? systemAdapter : auxAdapter;
}
