/** LoRA バージョン管理 API
 *
 * バックエンドの `/api/lora/*` エンドポイントに対応する取得関数を提供する。
 * 旧 `routes/dashboard/+page.svelte` 内の直接 fetch 呼び出しを集約
 *
 * 注意: バックエンド側 (`backend/pro/api/lora.py`) は Pro エディションのみで
 * ルータが登録されるため、Free エディションでは呼び出すと 404 になる。
 * routes/dashboard 自体が isPro ガードで保護されている前提。
 */

import { request } from './_client';
import type { ImprovementScore } from './dashboard';

/** LoRA バージョン情報 (バックエンド VersionInfoResponse 準拠) */
export interface LoraVersionInfo {
	version: number;
	eval_score: number;
	created_at: string;
	metadata: Record<string, unknown>;
}

/** バージョン一覧 API のレスポンス */
export interface LoraVersionsResponse {
	versions: LoraVersionInfo[];
	active_adapter_exists: boolean;
	latest_version: number;
}

/** ロールバックリクエスト */
export interface LoraRollbackRequest {
	version: number;
}

/** ロールバック API のレスポンス */
export interface LoraRollbackResponse {
	rolled_back_to: number;
	adapter_path: string;
}

/**
 * dashboard ストアが扱う緩い型 (一部のフィールドのみ参照)。
 * 旧 `safeFetch<Record<string, unknown>>('/api/lora/versions')` 互換。
 */
export interface LoraVersionsLooseResponse {
	versions?: ImprovementScore[];
	latest_version?: number;
}

/** LoRA バージョン一覧を取得 */
export async function getLoraVersions(): Promise<LoraVersionsResponse> {
	return request<LoraVersionsResponse>('GET', '/lora/versions');
}

/** 指定バージョンへロールバック */
export async function rollbackLora(version: number): Promise<LoraRollbackResponse> {
	return request<LoraRollbackResponse>('POST', '/lora/rollback', { version });
}
