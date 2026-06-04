/** ステータス API */

import { request } from './_client';

export interface LlamaServerInfo {
	connected: boolean;
	host: string;
	port: number;
}

export interface ModelInfo {
	name?: string;
	chat_template?: string;
	has_system_role: boolean;
	context_size: number;
}

export interface MemoryStats {
	working_turns: number;
	short_term_notes: number;
	long_term_chunks: number;
}

export interface ComponentStatus {
	name: string;
	connected: boolean;
}

export interface LearningBriefStatus {
	running: boolean;
	experience_count: number;
	conditions_met: boolean;
}

export interface DebugStatusInfo {
	enabled: boolean;
	log_dir: string;
	disk_usage_mb: number;
	recent_errors_count: number;
	cache_hit_rate: number;
	last_ttft_ms: number | null;
	last_tok_per_sec: number | null;
	learning: LearningBriefStatus;
}

/** モデル能力プローブ結果 (docs/c_15)。未プローブ時は probed=false。 */
export interface CapabilityInfo {
	slot: string; // "base" | "assist"
	model_id: string;
	probed: boolean;
	effective_reasoning_mode: string | null;
	reasoning_separated: boolean | null;
	emits_think_tags: boolean | null;
	closes_think_tags: boolean | null;
	json_schema_enforced: boolean | null;
	needs_lenient_json: boolean;
	probe_divergence: string[];
	probed_at: string;
}

export interface StatusResponse {
	status: string;
	edition: string;
	instance_name: string;
	version: string;
	uptime_seconds: number;
	llama_server: LlamaServerInfo;
	model?: ModelInfo;
	components: ComponentStatus[];
	memory: MemoryStats;
	cartridges_loaded: number;
	debug: DebugStatusInfo;
	capabilities?: CapabilityInfo[];
}

/** ステータス取得 */
export async function getStatus(): Promise<StatusResponse> {
	return request<StatusResponse>('GET', '/status');
}
