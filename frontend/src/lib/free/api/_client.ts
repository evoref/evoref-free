/**
 * API クライアント共通基盤
 *
 * すべての `lib/free/api/*.ts` モジュールが利用する共通ユーティリティ:
 * - `BASE_URL` 定数
 * - `ApiError` / `ApiErrorDetail` 型
 * - `parseApiError` / `throwIfNotOk` ヘルパー
 * - `request<T>()` / `requestVoid()` / `requestFormData<T>()` ラッパー
 *
 * このファイルは `lib/free/api` 内部からのみ import すること。
 * 外部 (コンポーネント / ストア / サービス) は `$lib/free/api` 経由で利用する。
 */

import { loggedFetch as fetch } from '$lib/devlog';

/** API のベース URL */
export const BASE_URL = '/api';

/** 構造化エラーレスポンスの型（設計書 §19.3 準拠） */
export interface ApiErrorDetail {
	code: string;
	message: string;
	i18n_key: string;
	context: Record<string, unknown>;
}

/** 構造化 API エラー — catch ブロックで instanceof 判定可能 */
export class ApiError extends Error {
	code: string;
	i18nKey: string;
	context: Record<string, unknown>;

	constructor(detail: ApiErrorDetail) {
		super(detail.message);
		this.name = 'ApiError';
		this.code = detail.code;
		this.i18nKey = detail.i18n_key;
		this.context = detail.context;
	}
}

/** API エラーレスポンスから構造化エラーを抽出 */
export async function parseApiError(res: Response): Promise<ApiErrorDetail> {
	try {
		const body = await res.json();
		if (body?.detail && typeof body.detail === 'object' && body.detail.code) {
			return body.detail as ApiErrorDetail;
		}
	} catch {
		// JSON parse failed
	}
	return {
		code: `E0${res.status}`,
		message: res.statusText || `HTTP ${res.status}`,
		i18n_key: '',
		context: {}
	};
}

/** レスポンスが !ok の場合 ApiError を throw する共通処理 */
export async function throwIfNotOk(res: Response): Promise<void> {
	if (!res.ok) {
		const detail = await parseApiError(res);
		throw new ApiError(detail);
	}
}

/** サポートする HTTP メソッド */
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';

/** 共通リクエストの追加オプション */
export interface RequestOptions {
	/** AbortController.signal — レースコンディション対策 (連続フィルタ変更等) */
	signal?: AbortSignal;
}

/**
 * JSON ボディを送信し、JSON レスポンスを受け取る共通リクエスト
 *
 * - body 未指定なら Content-Type ヘッダを付けない (GET/DELETE 想定)
 * - 非 ok レスポンスは ApiError を throw
 * - レスポンスボディが空 (204 等) の場合は呼び出し側で `requestVoid` を使うこと
 * - `options.signal` で AbortController キャンセル対応
 */
export async function request<T>(
	method: HttpMethod,
	path: string,
	body?: unknown,
	options?: RequestOptions
): Promise<T> {
	const init: RequestInit = { method };
	if (body !== undefined) {
		init.headers = { 'Content-Type': 'application/json' };
		init.body = JSON.stringify(body);
	}
	if (options?.signal) {
		init.signal = options.signal;
	}
	const res = await fetch(`${BASE_URL}${path}`, init);
	await throwIfNotOk(res);
	return res.json() as Promise<T>;
}

/**
 * JSON ボディを送信するが、レスポンスを読まない共通リクエスト
 *
 * 204 No Content / 単に成功フラグだけ確認したいケースで使う。
 */
export async function requestVoid(
	method: HttpMethod,
	path: string,
	body?: unknown,
	options?: RequestOptions
): Promise<void> {
	const init: RequestInit = { method };
	if (body !== undefined) {
		init.headers = { 'Content-Type': 'application/json' };
		init.body = JSON.stringify(body);
	}
	if (options?.signal) {
		init.signal = options.signal;
	}
	const res = await fetch(`${BASE_URL}${path}`, init);
	await throwIfNotOk(res);
}

/**
 * multipart/form-data リクエスト共通ヘルパー
 *
 * ファイルアップロード系 (themes/install, cartridges/install, datasets/upload 等) で使う。
 * Content-Type は指定しない (ブラウザが boundary 付きで自動生成する)。
 */
export async function requestFormData<T>(
	path: string,
	formData: FormData
): Promise<T> {
	const res = await fetch(`${BASE_URL}${path}`, {
		method: 'POST',
		body: formData
	});
	await throwIfNotOk(res);
	return res.json() as Promise<T>;
}

/** multipart/form-data でレスポンスボディを読まないバリエーション */
export async function requestFormDataVoid(
	path: string,
	formData: FormData
): Promise<void> {
	const res = await fetch(`${BASE_URL}${path}`, {
		method: 'POST',
		body: formData
	});
	await throwIfNotOk(res);
}
