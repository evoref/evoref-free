/**
 * SSE 進捗ストリーミング用ユーティリティ
 *
 * `fetch()` で POST + multipart/form-data リクエストを送り、
 * `text/event-stream` レスポンスを `data: <json>\n\n` 単位でパースして
 * `AsyncGenerator<SSEProgressEvent>` を返す。
 *
 * バックエンドのフレームスキーマ (`backend/free/core/sse.py`):
 * - `{"step": {"phase":..., "status":..., "current"?, "total"?, "detail"?}}`
 * - `{"result": {...payload}}`
 * - `{"error": {"code":..., "message":..., "context":...}}` または `{"error": "..."}`
 * - `data: [DONE]`
 */

import { loggedFetch as fetch } from '$lib/devlog';
import { BASE_URL } from './_client';
import { readSseFrames } from './sse_frames';

/** SSE 進捗イベントの型 */
export interface SSEProgressEvent {
	type: 'step' | 'result' | 'error' | 'done';
	step?: {
		phase: string;
		status: string;
		current?: number;
		total?: number;
		detail?: string;
	};
	result?: unknown;
	error?: { code?: string; message: string };
}

/**
 * BASE_URL とパスを結合する
 *
 * 引数が `/api/...` で始まる場合はそのまま、`/cartridges/...` のように
 * `/api` プレフィックスがない場合は `BASE_URL` を前置する。
 */
function _resolveUrl(pathOrUrl: string): string {
	if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
		return pathOrUrl;
	}
	if (pathOrUrl.startsWith('/api/')) {
		return pathOrUrl;
	}
	return `${BASE_URL}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`;
}

/**
 * FormData 本体で SSE エンドポイントを叩き、進捗イベントを yield する
 *
 * @param pathOrUrl エンドポイントパス (例: `/cartridges/install/stream`)
 * @param formData アップロード用 FormData (file / session_id / その他のフォームフィールド)
 * @param signal AbortSignal — abort されると fetch がキャンセルされる
 */
export async function* streamSSEFormData(
	pathOrUrl: string,
	formData: FormData,
	signal?: AbortSignal
): AsyncGenerator<SSEProgressEvent> {
	const url = _resolveUrl(pathOrUrl);

	let res: Response;
	try {
		res = await fetch(url, {
			method: 'POST',
			body: formData,
			signal
		});
	} catch (e) {
		if (e instanceof DOMException && e.name === 'AbortError') {
			return;
		}
		throw e;
	}

	if (!res.ok) {
		// HTTP エラー時はボディから構造化エラーを抽出
		let message = `HTTP ${res.status}`;
		try {
			const body = await res.json();
			if (body?.detail?.message) {
				message = body.detail.message;
			} else if (typeof body?.detail === 'string') {
				message = body.detail;
			}
		} catch {
			// JSON parse error はそのまま fallback メッセージを使う
		}
		yield { type: 'error', error: { message } };
		return;
	}

	if (!res.body) {
		yield { type: 'error', error: { message: 'No response body' } };
		return;
	}

	// フレームの切り出しは chat と共通 (`readSseFrames`)。ここは step / result /
	// error の 3 種への写像だけ。1 フレームのパース失敗はスキップ (既定)。
	for await (const frame of readSseFrames(res.body)) {
		if (frame.done) {
			yield { type: 'done' };
			return;
		}
		const parsed = frame.data as {
			type?: string;
			step?: SSEProgressEvent['step'];
			result?: unknown;
			error?: string | { code?: string; message: string };
		};
		const kind =
			parsed.type ??
			(parsed.step !== undefined
				? 'step'
				: parsed.result !== undefined
					? 'result'
					: parsed.error !== undefined
						? 'error'
						: undefined);
		if (kind === 'step' && parsed.step !== undefined) {
			yield { type: 'step', step: parsed.step };
		} else if (kind === 'result') {
			yield { type: 'result', result: parsed.result };
		} else if (kind === 'error' && parsed.error !== undefined) {
			const err =
				typeof parsed.error === 'string'
					? { message: parsed.error }
					: { code: parsed.error.code, message: parsed.error.message };
			yield { type: 'error', error: err };
		}
	}
}

/** staged クリエイト run の永続イベント → 表示用ステップの写像結果 (`AgenticStep` と構造互換) */
export interface CreateRunStep {
	type: string;
	detail: string;
	status?: string;
}

/**
 * `events.jsonl` の 1 レコード (`kind` / `payload`) を表示用ステップへ写す。
 *
 * `kind` / `payload` はライブストリーミングの SSE `step` フレームと同じ材料
 * (f_10 §7 の「永続 → 配信」) なので、写像はここに 1 本だけ持つ (2 本書かない)。
 *
 * - `stage_progress` (payload: `{stage, detail, status, task_id}`) → `task_progress` ステップ
 * - `finalize_*` / `cancel` / `timeout` (payload が既に `{type, detail, status}` を持つ) →
 *   そのまま写す
 * - それ以外 (`task_picked` / `iteration_ended` / `gate_result` / `success` / `failure` /
 *   `design_drift` / `disconnect` 等) は表示に写せないので `null`
 */
export function eventToStep(kind: string, payload: Record<string, unknown>): CreateRunStep | null {
	if (kind === 'stage_progress') {
		const detail = typeof payload.detail === 'string' ? payload.detail : '';
		if (!detail) return null;
		return {
			type: 'task_progress',
			detail,
			status: typeof payload.status === 'string' ? payload.status : 'running'
		};
	}
	if (kind.startsWith('finalize_') || kind === 'cancel' || kind === 'timeout') {
		const detail = typeof payload.detail === 'string' ? payload.detail : '';
		if (!detail) return null;
		return {
			type: typeof payload.type === 'string' ? payload.type : kind,
			detail,
			status: typeof payload.status === 'string' ? payload.status : 'done'
		};
	}
	return null;
}
