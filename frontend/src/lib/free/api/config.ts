/** 設定 API */

import { request } from './_client';

export interface ConfigFullResponse {
	config: Record<string, Record<string, unknown>>;
	sections: string[];
	edition: string;
}

export interface ConfigUpdateResponse {
	section: string;
	updated: boolean;
}

export interface ConfigValidateResponse {
	section: string;
	valid: boolean;
	errors: string[];
}

export interface LocalesResponse {
	locales: string[];
	current: string;
	prompt_locale: string;
}

/** 全設定取得 */
export async function getConfig(): Promise<ConfigFullResponse> {
	return request<ConfigFullResponse>('GET', '/config');
}

/** 利用可能ロケール一覧 + 現在値 (UI locale / チャット応答言語 prompt_locale) */
export async function getLocales(): Promise<LocalesResponse> {
	return request<LocalesResponse>('GET', '/config/locales');
}

/** 設定セクション更新 */
export async function updateConfigSection(
	section: string,
	data: Record<string, unknown>
): Promise<ConfigUpdateResponse> {
	return request<ConfigUpdateResponse>('PUT', `/config/${section}`, { data });
}

/** 設定セクションバリデーション */
export async function validateConfigSection(
	section: string,
	data: Record<string, unknown>
): Promise<ConfigValidateResponse> {
	return request<ConfigValidateResponse>('POST', `/config/${section}/validate`, { data });
}

/** create の実行環境 1 件の状態 (`create.runtimes.<name>`、f_10 §12.4) */
export interface RuntimeInfo {
	name: string;
	/** config.yaml の設定値 (空なら PATH から探す) */
	configured: string;
	/** 解決された実行ファイルの絶対パス (見つからなければ null) */
	resolved: string | null;
	source: 'configured' | 'path' | null;
	/** `--version` の 1 行目 (取れなければ空文字列) */
	version: string;
	/** 設定値が使えないときの検証理由 (`not_absolute` 等) */
	error: string | null;
}

/** 実行環境の一覧 (設定値・解決先・版) */
export async function getRuntimes(): Promise<RuntimeInfo[]> {
	return request<RuntimeInfo[]>('GET', '/config/runtimes');
}

/**
 * 実行環境のパスを保存する (サーバ側で検証。空文字列は設定を消して PATH から探す)。
 * 汎用の `PUT /config/create` からは書けない能力キーのため専用 API を使う (c_06 §1.5)。
 */
export async function setRuntimePath(name: string, path: string): Promise<RuntimeInfo> {
	return request<RuntimeInfo>('PUT', `/config/runtimes/${encodeURIComponent(name)}`, { path });
}
