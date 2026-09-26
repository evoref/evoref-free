/** チャット API (SSE ストリーミング + キャンセル) */

import { STREAM_CHUNK_TIMEOUT_MS } from '$lib/free/constants';
import { loggedFetch as fetch, devLog, IS_DEV } from '$lib/devlog';
import { BASE_URL, cancelStreamingOperation, parseApiError } from './_client';
import { readSseFrames, SSE_FRAME_TYPES, type SSEFrameType } from './sse_frames';

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

/** 出典 1 件 (backend `sources` フレーム、c_06 §2.1) */
export interface SourceItem {
	/** `<store>:<evidence_id>` (SearchResult.evidence_ids と同じ) */
	id: string;
	store: 'corpus' | 'episodic';
	package_id: string;
	package_name: string;
	doc_id: string;
	heading: string;
	score: number;
	preview: string;
}

/** この応答の [参考情報] に注入した根拠。本文には出典を書かせないので UI が出す */
export interface SourcesInfo {
	items: SourceItem[];
}

/** 出力先パス未指定時にエディタペインへ直接流す生成コード片 */
export interface EditorCodeArtifact {
	content: string;
	language: string;
	filename: string | null;
	/** long_form 生成途中のユニット完了ごとの逐次更新フレームか (終端の確定本文は false) */
	partial?: boolean;
}

/** ユーザー発言が長さ制限で切り詰められた旨の通知 */
export interface InputTruncatedInfo {
	original_chars: number;
	sent_chars: number;
}

/** 応答が出力トークン上限に達して途中終了した旨の通知 */
export interface OutputTruncatedInfo {
	tokens_generated: number;
	max_tokens: number | null;
}

/** `template_hint` フレーム 1 件分の様式候補 */
export interface TemplateHintEntry {
	/** `<package_id>:<entry_id>` (`selectedTemplate` / `POST /api/chat` の `template` にそのまま渡す) */
	key: string;
	doc_type: string;
	has_base: boolean;
	has_outline: boolean;
	has_fields: boolean;
}

/**
 * 「使える様式がある」通知 (依頼が様式を指名したが自動適用しなかった場合の
 * 副チャネル)。`candidate` = 単一の様式を指名 (「テンプレートで」等が無かった
 * ため自動適用しなかった)。`ambiguous` = 複数の様式が該当し決められなかった。
 */
export interface TemplateHint {
	kind: 'candidate' | 'ambiguous';
	templates: TemplateHintEntry[];
}

export interface ChatStreamEvent {
	type:
		| 'token'
		| 'token_info'
		| 'done'
		| 'error'
		| 'step'
		| 'agent_layer'
		| 'rag_debug'
		| 'sources'
		| 'editor_route'
		| 'editor_code'
		| 'input_truncated'
		| 'output_truncated'
		| 'create_run'
		| 'template_hint';
	token?: string;
	token_info?: TokenInfo;
	error?: string;
	/** 構造化エラー (`error_with_code`) のコード。`E0409` は制作中で受け付けなかったターン */
	error_code?: string;
	/** 構造化エラーの付帯情報 (`E0409` なら走っている側の `session_id` / `run_id`) */
	error_context?: Record<string, unknown>;
	/** 応答元レイヤー (ストリーム冒頭で 1 度) */
	agent_layer?: string;
	/** このターンの識別子。`cancelChat` に添えてこのリクエストだけを止める */
	request_id?: string;
	step?: ChatStreamStep;
	rag_debug?: RagDebugInfo;
	sources?: SourcesInfo;
	editor_route?: { target: 'editor' | 'chat' };
	editor_code?: EditorCodeArtifact;
	input_truncated?: InputTruncatedInfo;
	output_truncated?: OutputTruncatedInfo;
	/** staged クリエイトの run 識別子 (再接続用、f_05 §4.5)。run 開始直後に 1 回 */
	run_id?: string;
	session_id?: string;
	template_hint?: TemplateHint;
}

/** 文書 (corpus パッケージ) の参加モード。auto = 問いと較正で決める / on = 問い側の抑止を掛けない / off = このターンは引かない */
export type CorpusMode = 'auto' | 'on' | 'off';

/** SSE ストリーミングチャット */
export async function* chatStream(
	message: string,
	mode: string,
	sessionId?: string,
	files?: string[],
	signal?: AbortSignal,
	corpusMode: CorpusMode = 'auto',
	/** 選択中の文書テンプレート鍵 (`<package_id>:<entry_id>`)。このターンだけ効く */
	template: string | null = null
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
				files,
				corpus_mode: corpusMode,
				template
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

	if (!res.body) {
		if (IS_DEV) eventCounts.error++;
		yield { type: 'error', error: 'No response body' };
		return;
	}

	let parseErrorCount = 0;

	try {
		for await (const frame of readSseFrames(res.body!, {
			chunkTimeoutMs: STREAM_CHUNK_TIMEOUT_MS,
			onParseError: (raw, e) => {
				parseErrorCount++;
				console.warn(`[SSE Parse Error] count=${parseErrorCount} data="${raw.slice(0, 200)}"`, e);
			}
		})) {
			if (IS_DEV && !firstByteRecorded) {
				firstByteMs = performance.now() - streamStart;
				firstByteRecorded = true;
				devLog('SSE:first-byte', `/chat first byte in ${firstByteMs.toFixed(1)}ms`);
			}
			if (frame.done) {
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
			const event = toChatStreamEvent(frame.data);
			if (!event) continue;
			if (IS_DEV) eventCounts[event.type] = (eventCounts[event.type] ?? 0) + 1;
			yield event;
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
	}
}

/**
 * 1 フレームの JSON をチャットイベントへ写す。
 *
 * `type` (バックエンドが付ける) を優先し、無ければ本体キーで判定する (旧バックエンド互換)。
 * 未知の `type` は無視する — フレームを足しても消費側が落ちない。
 */
export function toChatStreamEvent(parsed: Record<string, unknown>): ChatStreamEvent | null {
	const kind = (typeof parsed.type === 'string' ? parsed.type : inferFrameKind(parsed)) as
		| SSEFrameType
		| undefined;
	switch (kind) {
		case 'token':
			return { type: 'token', token: String(parsed.token ?? '') };
		case 'token_info':
			return { type: 'token_info', token_info: parsed.token_info as TokenInfo };
		case 'agent_layer':
			return {
				type: 'agent_layer',
				agent_layer: String(parsed.agent_layer ?? ''),
				request_id: typeof parsed.request_id === 'string' ? parsed.request_id : undefined
			};
		case 'error': {
			const err = parsed.error;
			const message =
				typeof err === 'string'
					? err
					: String((err as { message?: string } | undefined)?.message ?? 'error');
			if (err && typeof err === 'object') {
				const { code, context } = err as { code?: unknown; context?: unknown };
				return {
					type: 'error',
					error: message,
					error_code: typeof code === 'string' ? code : undefined,
					error_context:
						context && typeof context === 'object'
							? (context as Record<string, unknown>)
							: undefined
				};
			}
			return { type: 'error', error: message };
		}
		case 'step':
			return { type: 'step', step: parsed.step as ChatStreamStep };
		case 'rag_debug':
			return { type: 'rag_debug', rag_debug: parsed.rag_debug as RagDebugInfo };
		case 'sources':
			return { type: 'sources', sources: parsed.sources as SourcesInfo };
		case 'input_truncated':
			return { type: 'input_truncated', input_truncated: parsed.input_truncated as InputTruncatedInfo };
		case 'output_truncated':
			return {
				type: 'output_truncated',
				output_truncated: parsed.output_truncated as OutputTruncatedInfo
			};
		case 'editor_route':
			return { type: 'editor_route', editor_route: parsed.editor_route as { target: 'editor' | 'chat' } };
		case 'editor_code':
			return { type: 'editor_code', editor_code: parsed.editor_code as EditorCodeArtifact };
		case 'create_run':
			return {
				type: 'create_run',
				run_id: typeof parsed.run_id === 'string' ? parsed.run_id : undefined,
				session_id: typeof parsed.session_id === 'string' ? parsed.session_id : undefined
			};
		case 'template_hint': {
			const hint = parseTemplateHint(parsed.template_hint);
			return hint ? { type: 'template_hint', template_hint: hint } : null;
		}
		default:
			return null;
	}
}

/** `template_hint` の本体を検証する。形が壊れていれば `null` (フレームごと無視) */
function parseTemplateHint(value: unknown): TemplateHint | null {
	if (!value || typeof value !== 'object') return null;
	const kind = (value as { kind?: unknown }).kind;
	if (kind !== 'candidate' && kind !== 'ambiguous') return null;
	const rawTemplates = (value as { templates?: unknown }).templates;
	if (!Array.isArray(rawTemplates)) return null;
	const templates = rawTemplates.filter(isTemplateHintEntry);
	if (templates.length === 0) return null;
	return { kind, templates };
}

function isTemplateHintEntry(entry: unknown): entry is TemplateHintEntry {
	if (!entry || typeof entry !== 'object') return false;
	const e = entry as Record<string, unknown>;
	return (
		typeof e.key === 'string' &&
		typeof e.doc_type === 'string' &&
		typeof e.has_base === 'boolean' &&
		typeof e.has_outline === 'boolean' &&
		typeof e.has_fields === 'boolean'
	);
}

/** `type` を持たない旧フレームの種別を本体キーから推定する */
function inferFrameKind(parsed: Record<string, unknown>): string | undefined {
	for (const k of SSE_FRAME_TYPES) {
		if (parsed[k] !== undefined) return k;
	}
	return undefined;
}

/** 応答への明示評価 (👎 / 👍 / 取り消し)。query はその応答を生んだユーザー発話 */
export type TurnFeedbackVerdict = 'negative' | 'positive' | 'clear';

export async function sendTurnFeedback(
	sessionId: string,
	verdict: TurnFeedbackVerdict,
	query?: string,
	note = ''
): Promise<{ recorded: boolean; entry_id?: string | null }> {
	try {
		const res = await fetch(`${BASE_URL}/learning/feedback`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ session_id: sessionId, verdict, query, note })
		});
		if (!res.ok) return { recorded: false };
		return (await res.json()) as { recorded: boolean; entry_id?: string | null };
	} catch {
		return { recorded: false };
	}
}

/**
 * チャットストリーミングをキャンセル (best-effort。通信失敗も false)。
 *
 * `requestId` (`agent_layer` フレームの `request_id`) があればそのリクエストだけを止める。
 * 無ければセッションの進行中リクエスト全部 (最初のフレームが届く前のキャンセル)。
 */
export async function cancelChat(sessionId: string, requestId?: string): Promise<boolean> {
	try {
		const data = await cancelStreamingOperation(
			'/chat/cancel',
			sessionId,
			requestId ? { request_id: requestId } : undefined
		);
		return data.cancelled ?? false;
	} catch {
		return false;
	}
}
