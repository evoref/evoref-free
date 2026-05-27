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

	const reader = res.body?.getReader();
	if (!reader) {
		yield { type: 'error', error: { message: 'No response body' } };
		return;
	}

	const decoder = new TextDecoder();
	let buffer = '';

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });

			const lines = buffer.split('\n');
			buffer = lines.pop() ?? '';

			for (const line of lines) {
				if (!line.startsWith('data:')) continue;
				const data = line.slice(5).trim();
				if (data === '[DONE]') {
					yield { type: 'done' };
					return;
				}
				if (!data) continue;
				try {
					const parsed = JSON.parse(data);
					if (parsed.step !== undefined) {
						yield { type: 'step', step: parsed.step };
					} else if (parsed.result !== undefined) {
						yield { type: 'result', result: parsed.result };
					} else if (parsed.error !== undefined) {
						const err =
							typeof parsed.error === 'string'
								? { message: parsed.error }
								: { code: parsed.error.code, message: parsed.error.message };
						yield { type: 'error', error: err };
					}
				} catch {
					// 1 フレームのパース失敗はスキップ
				}
			}
		}
	} finally {
		try {
			reader.releaseLock();
		} catch {
			// 既に解放済み
		}
	}
}
