import { writable, derived, get } from 'svelte/store';
import type { TokenInfo, RagDebugInfo } from '$lib/free/api';
import { switchModeApi } from '$lib/free/api';
import { MODE_RESTART_STATUS_TIMEOUT_MS } from '$lib/free/constants';

export interface AgenticStep {
	type: string;
	detail: string;
	status?: string;
	elapsed_ms?: number;
}

let _msgId = 0;

/** メッセージごとの一意 ID を生成 */
export function nextMessageId(): number {
	return ++_msgId;
}

export interface StepResult {
	detail: string;
	status: string;
}

export interface ChatMessage {
	id: number;
	role: 'user' | 'assistant';
	content: string;
	timestamp: number;
	files?: string[];
	agentic_steps?: AgenticStep[];
	step_results?: StepResult[];
	rag_debug?: RagDebugInfo;
	/** コーディングモードでの生成コード出力先 ('editor' 既定 / 'chat' 明示指示時) */
	editor_route?: 'editor' | 'chat';
}

/** モード別メッセージバッファ */
const modeMessages: Record<string, ChatMessage[]> = {
	chat: [],
	coding: []
};

/** モード別セッションID */
const modeSessions: Record<string, string> = {
	chat: crypto.randomUUID(),
	coding: crypto.randomUUID()
};

/** メッセージ配列 */
export const messages = writable<ChatMessage[]>([]);

/** 現在のモード */
export const currentMode = writable<string>('chat');

/** セッションID */
export const sessionId = writable<string>(modeSessions.chat);

/** トークン使用量 */
export const tokenInfo = writable<TokenInfo>({ used: 0, limit: 4096, pct: 0, instance_name: 'evoref' });

/** ストリーミング中フラグ */
export const isStreaming = writable<boolean>(false);

/** トークン生成速度 (tok/s)。ストリーミング中のみ更新、終了後は最終値を保持 */
export const tokenSpeed = writable<number>(0);

/** 添付ファイル一覧 */
export const attachedFiles = writable<File[]>([]);

/** トークン使用率 (%) */
export const tokenPct = derived(tokenInfo, ($info) =>
	Math.min(100, Math.round(($info.used / $info.limit) * 100))
);

/** メッセージ追加 */
export function addMessage(msg: ChatMessage): void {
	messages.update((msgs) => [...msgs, msg]);
}

/** 最後のアシスタントメッセージにトークンを追記 */
export function appendToLastAssistant(token: string): void {
	messages.update((msgs) => {
		const last = msgs[msgs.length - 1];
		if (last?.role === 'assistant') {
			return [...msgs.slice(0, -1), { ...last, content: last.content + token }];
		}
		return msgs;
	});
}

/** 最後のアシスタントメッセージにステップを追加 */
export function addStepToLastAssistant(step: AgenticStep): void {
	messages.update((msgs) => {
		const last = msgs[msgs.length - 1];
		if (last?.role === 'assistant') {
			const steps = [...(last.agentic_steps ?? []), step];
			return [...msgs.slice(0, -1), { ...last, agentic_steps: steps }];
		}
		return msgs;
	});
}

/** 最後のアシスタントメッセージに RAG デバッグ情報をセット */
export function setRagDebugToLastAssistant(ragDebug: RagDebugInfo): void {
	messages.update((msgs) => {
		const last = msgs[msgs.length - 1];
		if (last?.role === 'assistant') {
			return [...msgs.slice(0, -1), { ...last, rag_debug: ragDebug }];
		}
		return msgs;
	});
}

/** 最後のアシスタントメッセージに生成コードの出力先をセット */
export function setEditorRouteToLastAssistant(target: 'editor' | 'chat'): void {
	messages.update((msgs) => {
		const last = msgs[msgs.length - 1];
		if (last?.role === 'assistant') {
			return [...msgs.slice(0, -1), { ...last, editor_route: target }];
		}
		return msgs;
	});
}

/** 最後のアシスタントメッセージにステップ結果を追加 */
export function addStepResultToLastAssistant(detail: string, status: string = 'done'): void {
	messages.update((msgs) => {
		const last = msgs[msgs.length - 1];
		if (last?.role === 'assistant') {
			const results = [...(last.step_results ?? []), { detail, status }];
			return [...msgs.slice(0, -1), { ...last, step_results: results }];
		}
		return msgs;
	});
}

/** モード再起動ステータス */
export type ModeRestartStatus = 'idle' | 'restarting' | 'ready' | 'failed';
export const modeRestartStatus = writable<ModeRestartStatus>('idle');

/** モード切替（API成功後にUI更新、失敗時はロールバック） */
export async function switchMode(newMode: string): Promise<void> {
	const current = get(currentMode);
	if (current === newMode) return;
	if (get(modeRestartStatus) === 'restarting') return;

	// 現在のモードのメッセージを保存（API 呼び出し前にバックアップ）
	modeMessages[current] = get(messages);

	// バックエンド API 呼び出し（UI更新前）
	modeRestartStatus.set('restarting');
	try {
		const result = await switchModeApi(newMode);

		// API 成功: UI 状態を更新
		currentMode.set(newMode);
		messages.set(modeMessages[newMode] ?? []);
		sessionId.set(modeSessions[newMode] ?? crypto.randomUUID());

		if (result.model_changed && !result.restart_initiated) {
			modeRestartStatus.set('failed');
		} else if (result.model_changed && result.restart_initiated) {
			modeRestartStatus.set('ready');
			setTimeout(() => modeRestartStatus.set('idle'), MODE_RESTART_STATUS_TIMEOUT_MS);
		} else {
			modeRestartStatus.set('idle');
		}
	} catch {
		// API 失敗: UI はそのまま（ロールバック不要 — まだ更新していない）
		modeRestartStatus.set('failed');
	}
}

/** 会話履歴クリア（現在のモードのみ） */
export function clearMessages(): void {
	const current = get(currentMode);
	messages.set([]);
	modeMessages[current] = [];
	const newSessionId = crypto.randomUUID();
	modeSessions[current] = newSessionId;
	tokenInfo.set({ used: 0, limit: 4096, pct: 0, instance_name: 'evoref' });
	sessionId.set(newSessionId);
}

/** 添付ファイル追加 */
export function addFile(file: File): void {
	attachedFiles.update((files) => [...files, file]);
}

/** 添付ファイル削除 */
export function removeFile(index: number): void {
	attachedFiles.update((files) => files.filter((_, i) => i !== index));
}

/** 添付ファイルクリア */
export function clearFiles(): void {
	attachedFiles.set([]);
}
