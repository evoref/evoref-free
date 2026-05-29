/** チャット API (SSE ストリーミング + キャンセル) */

import { STREAM_CHUNK_TIMEOUT_MS } from '$lib/free/constants';
import { loggedFetch as fetch, devLog, IS_DEV } from '$lib/devlog';
import { BASE_URL, parseApiError } from './_client';

export interface TokenInfo {
	used: number;
	limit: number;
	pct: number;
	instance_name: string;
}

export interface ChatStreamStep {
	type: string;
	detail: string;
	status: string;
	elapsed_ms?: number;
}

export interface RagDebugChunk {
	source: string;
	score: number;
	preview: string;
}

export interface RagDebugInfo {
	chunks: RagDebugChunk[];
	search_time_ms: number;
}

/** 出力先パス未指定時にエディタペインへ直接流す生成コード片 */
export interface EditorCodeArtifact {
	content: string;
	language: string;
	filename: string | null;
	/** long_form 生成途中のユニット完了ごとの逐次更新フレームか (終端の確定本文は false) */
	partial?: boolean;
}

export interface ChatStreamEvent {
	type: 'token' | 'token_info' | 'done' | 'error' | 'step' | 'rag_debug' | 'editor_route' | 'editor_code';
	token?: string;
	token_info?: TokenInfo;
	error?: string;
	step?: ChatStreamStep;
	rag_debug?: RagDebugInfo;
	editor_route?: { target: 'editor' | 'chat' };
	editor_code?: EditorCodeArtifact;
}

/** SSE ストリーミングチャット */
export async function* chatStream(
	message: string,
	mode: string,
	sessionId?: string,
	files?: string[],
	signal?: AbortSignal
): AsyncGenerator<ChatStreamEvent> {
	const streamStart = IS_DEV ? performance.now() : 0;
	const eventCounts: Record<string, number> = IS_DEV
		? { token: 0, token_info: 0, step: 0, rag_debug: 0, error: 0, done: 0 }
		: {};
	let connectMs = 0;
	let firstByteMs = 0;
	let firstByteRecorded = false;

	let res: Response;
	try {
		res = await fetch(`${BASE_URL}/chat`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				message,
				mode,
				session_id: sessionId,
				files
			}),
			signal
		});
		if (IS_DEV) {
			connectMs = performance.now() - streamStart;
			devLog('SSE:connect', `/chat connected in ${connectMs.toFixed(1)}ms`);
		}
	} catch (e) {
		if (e instanceof DOMException && e.name === 'AbortError') return;
		throw e;
	}

	if (!res.ok) {
		const err = await parseApiError(res);
		if (IS_DEV) eventCounts.error++;
		yield { type: 'error', error: err.message };
		return;
	}

	const reader = res.body?.getReader();
	if (!reader) {
		if (IS_DEV) eventCounts.error++;
		yield { type: 'error', error: 'No response body' };
		return;
	}

	const decoder = new TextDecoder();
	let buffer = '';
	let parseErrorCount = 0;

	try {
		while (true) {
			const { done, value } = await Promise.race([
				reader.read(),
				new Promise<never>((_, reject) =>
					setTimeout(() => reject(new Error('Stream chunk timeout')), STREAM_CHUNK_TIMEOUT_MS)
				)
			]);
			if (done) break;
			if (IS_DEV && !firstByteRecorded) {
				firstByteMs = performance.now() - streamStart;
				firstByteRecorded = true;
				devLog('SSE:first-byte', `/chat first byte in ${firstByteMs.toFixed(1)}ms`);
			}

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() ?? '';

			for (const line of lines) {
				if (!line.startsWith('data: ')) continue;
				const data = line.slice(6).trim();
				if (data === '[DONE]') {
					if (IS_DEV) {
						eventCounts.done++;
						const totalMs = performance.now() - streamStart;
						devLog(
							'SSE:done',
							`/chat completed in ${totalMs.toFixed(1)}ms`,
							{ connectMs, firstByteMs, totalMs, eventCounts, parseErrorCount }
						);
					}
					yield { type: 'done' };
					return;
				}
				try {
					const parsed = JSON.parse(data);
					if (parsed.token !== undefined) {
						if (IS_DEV) eventCounts.token++;
						yield { type: 'token', token: parsed.token };
					}
					if (parsed.token_info) {
						if (IS_DEV) eventCounts.token_info++;
						yield { type: 'token_info', token_info: parsed.token_info };
					}
					if (parsed.error) {
						if (IS_DEV) eventCounts.error++;
						yield { type: 'error', error: parsed.error };
					}
					if (parsed.step) {
						if (IS_DEV) eventCounts.step++;
						yield { type: 'step', step: parsed.step };
					}
					if (parsed.rag_debug) {
						if (IS_DEV) eventCounts.rag_debug++;
						yield { type: 'rag_debug', rag_debug: parsed.rag_debug };
					}
					if (parsed.editor_route) {
						yield { type: 'editor_route', editor_route: parsed.editor_route };
					}
					if (parsed.editor_code) {
						yield { type: 'editor_code', editor_code: parsed.editor_code };
					}
				} catch (e) {
					parseErrorCount++;
					console.warn(`[SSE Parse Error] count=${parseErrorCount} data="${data.slice(0, 200)}"`, e);
				}
			}
		}
		if (IS_DEV) {
			const totalMs = performance.now() - streamStart;
			devLog(
				'SSE:end',
				`/chat stream ended without [DONE] in ${totalMs.toFixed(1)}ms`,
				{ connectMs, firstByteMs, totalMs, eventCounts, parseErrorCount }
			);
		}
	} catch (e) {
		if (e instanceof DOMException && e.name === 'AbortError') {
			if (IS_DEV) {
				const totalMs = performance.now() - streamStart;
				devLog('SSE:abort', `/chat aborted after ${totalMs.toFixed(1)}ms`, { eventCounts });
			}
			return;
		}
		if (e instanceof Error && e.message === 'Stream chunk timeout') {
			if (IS_DEV) eventCounts.error++;
			yield { type: 'error', error: 'stream_timeout' };
		} else {
			throw e;
		}
	} finally {
		reader.releaseLock();
	}
}

/** チャットストリーミングをキャンセル */
export async function cancelChat(sessionId: string): Promise<boolean> {
	try {
		const res = await fetch(`${BASE_URL}/chat/cancel`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ session_id: sessionId })
		});
		if (!res.ok) return false;
		const data = await res.json();
		return data.cancelled ?? false;
	} catch {
		return false;
	}
}
