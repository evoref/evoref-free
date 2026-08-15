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

/** LoRA 系列 */
export type LoraTarget = 'base';

/** モード (chat / create)。level2_adapter_partition=="model_mode" のときのみ意味を持つ。 */
export type LoraMode = 'chat' | 'create';

/** LoRA バージョン情報 (バックエンド VersionInfoResponse 準拠) */
export interface LoraVersionInfo {
	version: number;
	eval_score: number;
	created_at: string;
	metadata: Record<string, unknown>;
}

/** 1 系列分のバージョン一覧 (バックエンド TargetVersionsResponse 準拠) */
export interface LoraTargetVersions {
	versions: LoraVersionInfo[];
	active_adapter_exists: boolean;
	latest_version: number;
	model: string | null;
	/** アダプタが当該モードのモデルへ実際に適用されるか (null = アダプタ不在 / 判定不能)。
	 *  active_adapter_exists はファイル存在しか見ないため、存在しても起動時に
	 *  arch/形状/系統の不一致で --lora が付かないことがある。 */
	adapter_applicable?: boolean | null;
	/** adapter_applicable が false のときの理由 (英語短文) */
	adapter_incompatible_reason?: string | null;
}

/** LoRA アダプタのパーティションスキーム (learning.level2_adapter_partition) */
export type LoraAdapterPartition = 'model' | 'model_mode';

/** バージョン一覧 API のレスポンス */
export interface LoraVersionsResponse {
	base: LoraTargetVersions;
	/** 'model' のときアダプタはモード非依存で chat/create が同一実体を共有する */
	adapter_partition?: LoraAdapterPartition;
}

/** ロールバックリクエスト */
export interface LoraRollbackRequest {
	version: number;
	target?: LoraTarget;
	mode?: LoraMode;
}

/** ロールバック API のレスポンス */
export interface LoraRollbackResponse {
	rolled_back_to: number;
	adapter_path: string;
	target: LoraTarget;
	/** ロールバック対象モードが現在サービング中のモードと一致する場合のみ true。
	 *  自動再起動はされないため true の場合は再起動 (またはモード再切替) が必要。 */
	restart_required: boolean;
}

/**
 * LoRA バージョン一覧を取得
 *
 * @param mode `learning.level2_adapter_partition=="model_mode"` のときのみ有効。
 *   省略時はバックエンド側で現在のアクティブモードが使われる ("model" スキームでは無関係)。
 */
export async function getLoraVersions(mode?: LoraMode): Promise<LoraVersionsResponse> {
	const query = mode ? `?mode=${mode}` : '';
	return request<LoraVersionsResponse>('GET', `/lora/versions${query}`);
}

/** 指定系列の指定バージョンへロールバック */
export async function rollbackLora(
	version: number,
	target: LoraTarget = 'base',
	mode?: LoraMode
): Promise<LoraRollbackResponse> {
	const body: LoraRollbackRequest = { version, target };
	if (mode) {
		body.mode = mode;
	}
	return request<LoraRollbackResponse>('POST', '/lora/rollback', body);
}
