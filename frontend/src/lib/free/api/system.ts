/**
 * システム情報 API クライアント
 *
 * VRAM 使用量スナップショットを取得する。``nvidia-smi`` 不在 / PID 追跡不能
 * な環境でも 200 で推定値を返すため、呼び出し側はレスポンスが取れれば
 * ``measurement_available`` / ``source`` で UI 表示を分岐する。
 */

import { request } from './_client';

export type VramSource = 'estimate' | 'actual' | 'mixed';
export type VramPlacement = 'GPU' | 'CPU' | 'none';

export interface VramModelInfo {
	name: string; // "base" | "assist" | "embed"
	present: boolean;
	vram_mb: number;
	gpu_layers: number;
	model_mb: number | null;
	source: Exclude<VramSource, 'mixed'>;
	placement: VramPlacement;
	pid: number | null;
}

export interface VramStatusResponse {
	source: VramSource;
	measurement_available: boolean;
	nvidia_smi_available: boolean;
	process_manager_enabled: boolean;
	models: VramModelInfo[];
	total_mb: number;
	budget_mb: number | null;
	over_budget: boolean;
}

/** VRAM 使用量スナップショットを取得 */
export async function getVramStatus(): Promise<VramStatusResponse> {
	return request<VramStatusResponse>('GET', '/system/vram_status');
}
