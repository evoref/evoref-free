import { writable, derived, get } from 'svelte/store';
import type { TokenInfo, RagDebugInfo, EditorCodeArtifact } from '$lib/free/api';
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

/** long_form 生成の進捗 (ユニット i/total と現在のユニット名)。チャットに常時表示する */
export interface LongFormProgress {
	current: number;
	total: number;
	label: string;
	done: boolean;
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
	/** long_form 生成の進捗 (ユニット i/total)。生成中のみセットされ常時表示される */
	long_form_progress?: LongFormProgress;
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

/**
 * 出力先パス未指定時にエディタペインへ流す生成コード片の受け皿。
 *
 * バックエンドの `editor_code` SSE フレームを ChatInput が push し、Pro の
 * EditorPanel が購読して `loadGeneratedCode` へ渡す (Free→Pro 越境を作らない
 * ための共有ストア。`messages`/`isStreaming` を Pro が読む構図と同型)。
 * Free ビルドでは購読側が存在せず no-op。
 */
export const generatedEditorCode = writable<EditorCodeArtifact[]>([]);

/** 生成コード片を 1 件追加する (editor_code SSE フレーム受信時) */
export function pushGeneratedEditorCode(artifact: EditorCodeArtifact): void {
	generatedEditorCode.update((list) => [...list, artifact]);
}

/** 生成コード片の受け皿をクリアする (消費後 / 新ターン開始時) */
export function clearGeneratedEditorCode(): void {
	generatedEditorCode.set([]);
}

/**
 * long_form 生成の逐次更新コード (ユニット完了ごとの累積本文)。
 *
 * `editor_code` SSE フレームの `partial=true` を ChatInput がここへ set し、Pro の
 * EditorPanel が購読して同一タブを上書き更新する (生成途中の経過を可視化)。
 * 終端の確定本文 (`partial=false`) も同経路で流し、最後に EditorPanel が clear する。
 * `generatedEditorCode` (確定本文を新規タブへ流す既存経路) とは独立。Free ビルドでは
 * 購読側が存在せず no-op。
 */
export const streamingEditorCode = writable<EditorCodeArtifact | null>(null);

/** 逐次更新コードを set する (partial / final 共通) */
export function setStreamingEditorCode(artifact: EditorCodeArtifact): void {
	streamingEditorCode.set(artifact);
}

/** 逐次更新コードの受け皿をクリアする (確定反映後 / 新ターン開始時) */
export function clearStreamingEditorCode(): void {
	streamingEditorCode.set(null);
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

/** 最後のアシスタントメッセージに long_form 進捗をセット (生成中の常時表示用) */
export function setLongFormProgressToLastAssistant(progress: LongFormProgress): void {
	messages.update((msgs) => {
		const last = msgs[msgs.length - 1];
		if (last?.role === 'assistant') {
			return [...msgs.slice(0, -1), { ...last, long_form_progress: progress }];
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
		// API 失敗 (接続エラー等の一過性): UI はそのまま（ロールバック不要 —
		// まだ currentMode を更新していない）。'ready' と対称に一定時間後
		// 'idle' へ自動復帰させ、失敗表示が出っぱなしになるのを防ぐ。
		// (model_changed && !restart_initiated の 'failed' は実モデル未起動を
		//  反映するため sticky のまま据え置く)
		modeRestartStatus.set('failed');
		setTimeout(() => {
			if (get(modeRestartStatus) === 'failed') modeRestartStatus.set('idle');
		}, MODE_RESTART_STATUS_TIMEOUT_MS);
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
