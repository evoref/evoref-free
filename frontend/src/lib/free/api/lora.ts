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

/** LoRA 系列 (base / assist) */
export type LoraTarget = 'base' | 'assist';

/** LoRA バージョン情報 (バックエンド VersionInfoResponse 準拠) */
export interface LoraVersionInfo {
	version: number;
	eval_score: number;
	created_at: string;
	metadata: Record<string, unknown>;
}

/** base / assist 1 系列分のバージョン一覧 (バックエンド TargetVersionsResponse 準拠) */
export interface LoraTargetVersions {
	versions: LoraVersionInfo[];
	active_adapter_exists: boolean;
	latest_version: number;
	model: string | null;
}

/** バージョン一覧 API のレスポンス (base / assist 2 系列) */
export interface LoraVersionsResponse {
	base: LoraTargetVersions;
	assist: LoraTargetVersions;
}

/** ロールバックリクエスト */
export interface LoraRollbackRequest {
	version: number;
	target?: LoraTarget;
}

/** ロールバック API のレスポンス */
export interface LoraRollbackResponse {
	rolled_back_to: number;
	adapter_path: string;
	target: LoraTarget;
}

/** LoRA バージョン一覧を取得 */
export async function getLoraVersions(): Promise<LoraVersionsResponse> {
	return request<LoraVersionsResponse>('GET', '/lora/versions');
}

/** 指定系列の指定バージョンへロールバック */
export async function rollbackLora(
	version: number,
	target: LoraTarget = 'base'
): Promise<LoraRollbackResponse> {
	return request<LoraRollbackResponse>('POST', '/lora/rollback', { version, target });
}
