/** 環境調整 (auto-tune) API (docs/c_16 §7.2.3) */

import { request } from './_client';

export type AutoTuneRunState = 'idle' | 'running' | 'done' | 'failed';
export type AutoTuneDecision = 'auto' | 'accepted' | 'declined' | 'pending' | 'unchanged';

export interface AutoTuneProgress {
	phase: string;
	current: number;
	total: number;
}

export interface AutoTunePc {
	hostname: string;
	cpu: string;
	logical_cores: number;
	memory_gb: number;
	gpus: string[];
}

/** 1 項目の結果。value は提案値 (手動設定のため未適用 = applied false の項目でも入る) */
export interface AutoTuneItem {
	key: string;
	value: unknown;
	source: 'measured' | 'estimated' | string;
	/** manual (明示値のため未適用) / not_auto (自動指定ではない) ほか識別子 */
	reason: string;
	config_key: string;
	applied: boolean;
	requires_restart: boolean;
	measured_at: string;
}

export interface AutoTuneResponse {
	state: AutoTuneRunState;
	progress: AutoTuneProgress | null;
	pc: AutoTunePc;
	items: AutoTuneItem[];
	decision: AutoTuneDecision;
	changed_axes: string[];
	error: string | null;
	restart_required: boolean;
}

export interface AutoTuneRunRequest {
	only: string[] | null;
	force: boolean;
}

/** 環境調整の詳細 (PC・項目・進捗) を取得する */
export async function getAutoTune(): Promise<AutoTuneResponse> {
	return request<AutoTuneResponse>('GET', '/system/auto-tune');
}

/** 環境調整を始める (202。実行中は 409 E0409) */
export async function runAutoTune(body: AutoTuneRunRequest): Promise<{ state: string }> {
	return request<{ state: string }>('POST', '/system/auto-tune/run', body);
}

/** 環境移行の確認に答える */
export async function decideAutoTune(
	decision: 'accepted' | 'declined' | 'unchanged'
): Promise<unknown> {
	return request<unknown>('POST', '/system/auto-tune/decision', { decision });
}
