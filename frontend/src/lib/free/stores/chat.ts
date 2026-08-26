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
	/** クリエイトモードでの生成コード出力先 ('editor' 既定 / 'chat' 明示指示時) */
	editor_route?: 'editor' | 'chat';
	/** long_form 生成の進捗 (ユニット i/total)。生成中のみセットされ常時表示される */
	long_form_progress?: LongFormProgress;
	/**
	 * この user 発言が長さ制限で切り詰められて送られた場合の内訳。
	 * backend の system 注記だけではベースモデルが従わず、ユーザーには何も
	 * 見えなかったため (2026-07-26 実測)、発言バブル自体に表示する。
	 */
	input_truncated?: { original_chars: number; sent_chars: number };
	/**
	 * この応答が出力トークン上限に達して途中終了した場合の内訳。
	 * 注記文字列を本文へ連結すると履歴に保存され、次ターンでモデルが
	 * 逐語復唱する (2026-08-25 実測)。本文の外に持って表示だけに使う。
	 */
	output_truncated?: { tokens_generated: number; max_tokens: number | null };
}

/** モード別メッセージバッファ */
const modeMessages: Record<string, ChatMessage[]> = {
	chat: [],
	create: []
};

/** モード別セッションID */
const modeSessions: Record<string, string> = {
	chat: crypto.randomUUID(),
	create: crypto.randomUUID()
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

/**
 * 直近の user メッセージに「長さ制限で切り詰められた」内訳を付ける。
 *
 * 切り詰めの通知はアシスタント応答のストリーム冒頭で届くため、その時点で
 * 末尾は空のアシスタントメッセージ。対象は 1 つ前の user メッセージになる。
 */
export function markLastUserTruncated(info: {
	original_chars: number;
	sent_chars: number;
}): void {
	messages.update((msgs) => {
		for (let i = msgs.length - 1; i >= 0; i--) {
			if (msgs[i].role === 'user') {
				const updated = [...msgs];
				updated[i] = { ...msgs[i], input_truncated: info };
				return updated;
			}
		}
		return msgs;
	});
}

/**
 * 直近のアシスタントメッセージに「出力上限で途中終了した」内訳を付ける。
 *
 * 通知は応答ストリームの終端 (token_info の直前) で届くため、対象は
 * 末尾のアシスタントメッセージ。
 */
export function markLastAssistantTruncated(info: {
	tokens_generated: number;
	max_tokens: number | null;
}): void {
	messages.update((msgs) => {
		for (let i = msgs.length - 1; i >= 0; i--) {
			if (msgs[i].role === 'assistant') {
				const updated = [...msgs];
				updated[i] = { ...msgs[i], output_truncated: info };
				return updated;
			}
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

	// バックエンド API 呼び出し（UI更新前）
	modeRestartStatus.set('restarting');
	try {
		const result = await switchModeApi(newMode);

		// 現在のモードのメッセージを保存。await の「後」に取ること。
		// llama-server のモデル入替を伴う切替は数十秒かかり、その間に
		// ユーザーがメッセージを送れてしまう。await 前にスナップショットを
		// 取ると、待機中に追加されたターンが下の messages.set で消えたまま
		// バックアップにも残らず永久に失われる
		// (実インシデント 2026-07-27 ライブ検証: クリエイトへ切替直後に
		//  送信したメッセージが自分の吹き出しごと消え、応答も破棄された)。
		modeMessages[current] = get(messages);

		// API 成功: UI 状態を更新
		currentMode.set(newMode);
		messages.set(modeMessages[newMode] ?? []);
		sessionId.set(modeSessions[newMode] ?? crypto.randomUUID());

		// base の再起動が必要だったが完了しなかった場合は failed。
		// 両方とも変更なし、または変更があった分は全て再起動完了していれば ready。
		const baseOk = !result.model_changed || result.restart_initiated;
		const anyChanged = result.model_changed;

		if (anyChanged && !baseOk) {
			modeRestartStatus.set('failed');
		} else if (anyChanged) {
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
