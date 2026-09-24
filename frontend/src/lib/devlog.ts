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
/** backend のアクセス制御 (c_06 §1.5) が状態変更リクエストに要求するヘッダ */
export const CLIENT_HEADER = 'X-Evoref-Client';
/** `server.allow_remote: true` の構成で全リクエストに要るトークンのヘッダ */
export const TOKEN_HEADER = 'X-Evoref-Token';
const TOKEN_STORAGE_KEY = 'evoref.api_token';
const TOKEN_QUERY_PARAM = 'evoref_token';

/**
 * LAN から使う構成のトークン。初回だけ URL の `?evoref_token=` から受け取って
 * sessionStorage に移し、URL からは消す (履歴やスクリーンショットに残さない)。
 */
function apiToken(): string | null {
	if (typeof window === 'undefined') return null;
	try {
		const url = new URL(window.location.href);
		const fromQuery = url.searchParams.get(TOKEN_QUERY_PARAM);
		if (fromQuery) {
			window.sessionStorage.setItem(TOKEN_STORAGE_KEY, fromQuery);
			url.searchParams.delete(TOKEN_QUERY_PARAM);
			window.history.replaceState(window.history.state, '', url.toString());
			return fromQuery;
		}
		return window.sessionStorage.getItem(TOKEN_STORAGE_KEY);
	} catch {
		return null;
	}
}

function isApiRequest(input: RequestInfo | URL): boolean {
	const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
	if (raw.startsWith('/api')) return true;
	if (typeof window === 'undefined') return false;
	try {
		const url = new URL(raw, window.location.href);
		return url.origin === window.location.origin && url.pathname.startsWith('/api');
	} catch {
		return false;
	}
}

/** `/api` への要求にクライアントヘッダ (と、あればトークン) を付ける。呼び出し側の init は変更しない。 */
export function withClientHeaders(input: RequestInfo | URL, init?: RequestInit): RequestInit | undefined {
	if (!isApiRequest(input)) return init;
	const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
	headers.set(CLIENT_HEADER, '1');
	const token = apiToken();
	if (token) headers.set(TOKEN_HEADER, token);
	return { ...init, headers };
}

export async function loggedFetch(
	input: RequestInfo | URL,
	init?: RequestInit
): Promise<Response> {
	init = withClientHeaders(input, init);
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
