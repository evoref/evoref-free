/**
 * SSE フレームの共通読み手 (`data: <json>\n\n` 行の切り出し)。
 *
 * バックエンド (`backend/free/core/sse.py` の `SSEFrameBuilder`) は全 JSON フレームに
 * トップレベル `type` を付ける。ここは「バイト列 → 1 フレーム 1 JSON」の層だけを担い、
 * 意味付け (どのイベントに写すか) は消費側 (`chat.ts` / `sse_progress.ts`) に残す。
 * 以前は 2 本のパーサが `data:` の空白有無 / `else if` の有無 / chunk timeout の有無で
 * 食い違っていた。
 *
 * `SSE_FRAME_TYPES` はバックエンドの `FRAME_TYPES` と契約テスト
 * (`backend/free/core/tests/test_sse_frame_types.py`) で突き合わせる。
 */

/** バックエンドが出す JSON フレームの `type` 一覧 (`done` / keepalive は JSON ではない) */
export const SSE_FRAME_TYPES = [
	'token',
	'step',
	'agent_layer',
	'editor_route',
	'editor_code',
	'token_info',
	'input_truncated',
	'output_truncated',
	'rag_debug',
	'sources',
	'error',
	'result',
	'create_run'
] as const;

export type SSEFrameType = (typeof SSE_FRAME_TYPES)[number];

/** 1 フレーム。`[DONE]` は `{ done: true }` として返す */
export type SSEFrame = { done: true } | { done?: false; data: Record<string, unknown> };

export interface ReadSseOptions {
	/** 1 チャンクの読み取り上限 (ms)。超えると `'Stream chunk timeout'` を投げる。省略時は無制限 */
	chunkTimeoutMs?: number;
	/** JSON にならない `data:` 行を受けたときの通知 (既定: 無視) */
	onParseError?: (raw: string, error: unknown) => void;
}

/**
 * レスポンス本文を `data:` 行単位で読み、JSON フレームを順に返す。
 *
 * - `: keepalive` 等のコメント行と空行は読み飛ばす
 * - `data: [DONE]` で `{ done: true }` を返して終了する
 * - 途中離脱 (timeout / 消費側の早期 return) では `reader.cancel()` で接続を閉じる
 *   (`releaseLock()` だけだとバックエンドは生成を続けたまま接続が残る)
 * - read ごとに作る chunk timeout のタイマーは必ず解除する
 */
export async function* readSseFrames(
	body: ReadableStream<Uint8Array>,
	opts: ReadSseOptions = {}
): AsyncGenerator<SSEFrame> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let finished = false;

	async function readChunk(): Promise<ReadableStreamReadResult<Uint8Array>> {
		if (!opts.chunkTimeoutMs) return reader.read();
		let timer: ReturnType<typeof setTimeout> | undefined;
		try {
			return await Promise.race([
				reader.read(),
				new Promise<never>((_, reject) => {
					timer = setTimeout(() => reject(new Error('Stream chunk timeout')), opts.chunkTimeoutMs);
				})
			]);
		} finally {
			clearTimeout(timer);
		}
	}

	try {
		while (true) {
			const { done, value } = await readChunk();
			if (done) {
				finished = true;
				break;
			}
			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() ?? '';
			for (const line of lines) {
				if (!line.startsWith('data:')) continue;
				const raw = line.slice(5).trim();
				if (!raw) continue;
				if (raw === '[DONE]') {
					finished = true;
					yield { done: true };
					return;
				}
				let parsed: unknown;
				try {
					parsed = JSON.parse(raw);
				} catch (e) {
					opts.onParseError?.(raw, e);
					continue;
				}
				if (parsed && typeof parsed === 'object') {
					yield { data: parsed as Record<string, unknown> };
				}
			}
		}
	} finally {
		if (!finished) {
			try {
				await reader.cancel();
			} catch {
				// 既に閉じている / abort 済み
			}
		}
		try {
			reader.releaseLock();
		} catch {
			// 既に解放済み
		}
	}
}
