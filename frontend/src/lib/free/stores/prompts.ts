/** プロンプト管理の状態管理 */

import { writable, get } from 'svelte/store';
import {
	getPrompts,
	getAssistPrompts,
	type PromptListItem,
	type AssistPromptListItem,
	type PromptHistoryItem
} from '$lib/free/api';
import {
	getPromptAdapter,
	type PromptCategory,
	type PromptDetailWithScore
} from '$lib/free/services/promptService';
import { addToast } from './toast';

/** 選択中のプロンプト識別子 */
export interface PromptSelection {
	category: PromptCategory;
	id: string; // mode (chat/coding) or task (rag_judge/...)
}

// PromptCategory と PromptDetailWithScore を re-export
export type { PromptCategory, PromptDetailWithScore };

/** システムプロンプト一覧 */
export const systemPrompts = writable<PromptListItem[]>([]);

/** アシストプロンプト一覧 */
export const assistPrompts = writable<AssistPromptListItem[]>([]);

/** 選択中のプロンプト */
export const selectedPrompt = writable<PromptSelection | null>(null);

/** 編集中の内容 */
export const editContent = writable<string>('');

/** 現在のプロンプト詳細 */
export const currentDetail = writable<PromptDetailWithScore | null>(null);

/** 履歴 */
export const promptHistory = writable<PromptHistoryItem[]>([]);

/** 読込み中フラグ */
export const promptsLoading = writable(false);

/** 保存中フラグ */
export const promptSaving = writable(false);

/** dirty フラグ（編集中の変更あり） */
export const promptDirty = writable(false);

/** 履歴パネルの表示フラグ */
export const historyOpen = writable(false);

/** 元の内容（dirty 検出用） */
let originalContent = '';

/** アシストプロンプトのデフォルトタスク一覧 */
const DEFAULT_ASSIST_TASKS = ['rag_judge', 'query_expand', 'tool_call', 'note_evolve'];

/** API未取得時のフォールバック用アシストプロンプト一覧 */
function defaultAssistPrompts(): AssistPromptListItem[] {
	return DEFAULT_ASSIST_TASKS.map((task) => ({
		task,
		version: 0,
		source: 'default',
		updated_at: '',
		fitness_score: 0,
		content_preview: ''
	}));
}

/** プロンプト一覧を読み込む */
export async function loadPromptList(): Promise<void> {
	promptsLoading.set(true);
	try {
		const [sys, assist] = await Promise.all([
			getPrompts(),
			getAssistPrompts().catch(() => defaultAssistPrompts())
		]);
		systemPrompts.set(sys);
		assistPrompts.set(assist.length > 0 ? assist : defaultAssistPrompts());
	} catch {
		addToast({ type: 'error', i18nKey: 'settings.prompts.load_failed' });
	} finally {
		promptsLoading.set(false);
	}
}

/** プロンプトを選択して詳細を読み込む */
export async function selectPrompt(selection: PromptSelection): Promise<void> {
	selectedPrompt.set(selection);
	promptsLoading.set(true);
	historyOpen.set(false);

	try {
		const adapter = getPromptAdapter(selection.category);
		const detail = await adapter.fetchDetail(selection.id);
		currentDetail.set(detail);
		editContent.set(detail.content);
		originalContent = detail.content;
		promptDirty.set(false);
	} catch {
		addToast({ type: 'error', i18nKey: 'settings.prompts.load_failed' });
	} finally {
		promptsLoading.set(false);
	}
}

/** 編集内容を更新 */
export function updateEditContent(content: string): void {
	editContent.set(content);
	promptDirty.set(content !== originalContent);
}

/** プロンプトを保存 */
export async function savePrompt(): Promise<boolean> {
	const sel = get(selectedPrompt);
	const content = get(editContent);
	if (!sel) return false;

	promptSaving.set(true);
	try {
		const adapter = getPromptAdapter(sel.category);
		await adapter.save(sel.id, content);
		originalContent = content;
		promptDirty.set(false);
		addToast({ type: 'success', i18nKey: 'settings.prompts.saved' });

		// 一覧と詳細を再取得
		await Promise.all([loadPromptList(), selectPrompt(sel)]);
		return true;
	} catch {
		addToast({ type: 'error', i18nKey: 'settings.prompts.save_failed' });
		return false;
	} finally {
		promptSaving.set(false);
	}
}

/** ディスクから再読込み */
export async function reloadFromDisk(): Promise<void> {
	const sel = get(selectedPrompt);
	if (!sel) return;

	try {
		const adapter = getPromptAdapter(sel.category);
		await adapter.reload(sel.id);
		addToast({ type: 'success', i18nKey: 'settings.prompts.reloaded' });
		await Promise.all([loadPromptList(), selectPrompt(sel)]);
	} catch {
		addToast({ type: 'error', i18nKey: 'settings.prompts.reload_failed' });
	}
}

/** 履歴を読み込む */
export async function loadHistory(): Promise<void> {
	const sel = get(selectedPrompt);
	if (!sel) return;

	try {
		const adapter = getPromptAdapter(sel.category);
		const history = await adapter.fetchHistory(sel.id);
		promptHistory.set(history);
		historyOpen.set(true);
	} catch {
		promptHistory.set([]);
	}
}

/** ロールバック実行 */
export async function executeRollback(version: number): Promise<boolean> {
	const sel = get(selectedPrompt);
	if (!sel) return false;

	promptSaving.set(true);
	try {
		const adapter = getPromptAdapter(sel.category);
		await adapter.rollback(sel.id, version);
		addToast({ type: 'success', i18nKey: 'settings.prompts.rolled_back', params: { version } });
		await Promise.all([loadPromptList(), selectPrompt(sel)]);
		historyOpen.set(false);
		return true;
	} catch {
		addToast({ type: 'error', i18nKey: 'settings.prompts.rollback_failed' });
		return false;
	} finally {
		promptSaving.set(false);
	}
}

/** 編集をリセット */
export function resetEdit(): void {
	editContent.set(originalContent);
	promptDirty.set(false);
}
