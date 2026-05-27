/**
 * フロントエンド DEV モード用ロガー
 *
 * `import.meta.env.DEV` が真のときのみコンソールへ出力する。
 * 本番ビルドではガード条件が定数畳み込みされ、呼び出しごと
 * バンドラー側で除去されることを期待する。
 */

/** DEV ビルド判定（テスト環境を含む） */
export const IS_DEV: boolean = (() => {
	try {
		return Boolean(import.meta.env?.DEV);
	} catch {
		return false;
	}
})();

/** デバッグログ（DEV のみ） */
export function devLog(category: string, ...args: unknown[]): void {
	if (!IS_DEV) return;
	console.debug(`[${category}]`, ...args);
}

/** 警告ログ（DEV のみ） */
export function devWarn(category: string, ...args: unknown[]): void {
	if (!IS_DEV) return;
	console.warn(`[${category}]`, ...args);
}

/** API リクエストの timing 計測結果 */
export interface FetchTiming {
	method: string;
	url: string;
	status: number;
	ok: boolean;
	elapsedMs: number;
}

/**
 * fetch をラップして DEV モード時に request/response のタイミングを記録する。
 *
 * - 非 DEV ビルドではオーバーヘッドゼロ（生 fetch をそのまま呼ぶ）
 * - エラー時は失敗ログを出して再 throw
 * - グローバル fetch を参照するため、テスト側で `vi.stubGlobal('fetch', ...)`
 *   をしている既存テストがそのまま動作する
 */
export async function loggedFetch(
	input: RequestInfo | URL,
	init?: RequestInit
): Promise<Response> {
	if (!IS_DEV) {
		return fetch(input, init);
	}

	const method = (init?.method ?? 'GET').toUpperCase();
	const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
	const start = performance.now();
	devLog('API:req', method, url);

	try {
		const res = await fetch(input, init);
		const elapsedMs = performance.now() - start;
		const timing: FetchTiming = {
			method,
			url,
			status: res.status,
			ok: res.ok,
			elapsedMs
		};
		devLog('API:res', `${method} ${url} → ${res.status} (${elapsedMs.toFixed(1)}ms)`, timing);
		return res;
	} catch (e) {
		const elapsedMs = performance.now() - start;
		// AbortError は通常フローのキャンセル — エラー扱いしない
		if (e instanceof DOMException && e.name === 'AbortError') {
			devLog('API:abort', `${method} ${url} (${elapsedMs.toFixed(1)}ms)`);
		} else {
			devWarn('API:err', `${method} ${url} failed (${elapsedMs.toFixed(1)}ms)`, e);
		}
		throw e;
	}
}
