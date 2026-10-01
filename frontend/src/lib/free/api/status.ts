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
	slot: string; // "base"
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

/**
 * 効果の死活監視で立っている警告 (docs/c_07 §7.1)。
 * kind: starved (以前届いていた段が届かない) / stalled (入力があるのに効果ゼロが続く) /
 * degenerate (判定点の発火率が 0% か 100%) / failing (連続失敗) / emptied (ストアが空になった)。
 * detail は英語 (ログと同じ文)。
 */
export interface LivenessAlert {
	stage: string;
	kind: 'starved' | 'stalled' | 'degenerate' | 'failing' | 'emptied' | string;
	since: string;
	detail: string;
}

/**
 * データ根の状態 (docs/c_05 §0.9)。readonly の間は記憶・学習・履歴・設定を保存しない。
 * reason / warnings は英語 (ログと同じ文)。
 * reembed_pending は埋め込みモデルが変わって再埋め込みの確認待ちのストア (確認までは記憶の検索が限られる)。
 * served_model_mismatch は llama-server が実際に載せているモデルが config と違うこと (docs/c_05 §0.5.7)。
 * degraded はこの起動中にチャット経路の保存に失敗した形式 (format_id)。
 * formats は読み手が開いたときに current でなかった形式だけ (健全なら空。全件の照合は evoref doctor)。
 */
export interface DataHealthInfo {
	data_root: string;
	readonly: boolean;
	reason: string | null;
	warnings: string[];
	reembed_pending: string[];
	served_model_mismatch: boolean;
	served_model: string;
	expected_model: string;
	degraded: string[];
	/** 前回と違うエディションで起動した (from → to)。切替が無ければ null */
	edition_switched_from: string | null;
	edition_switched_to: string | null;
	formats: Record<string, FormatHealthInfo>;
}

/** 読み手が current として読めなかった形式 1 つ。state は newer / foreign / unmigratable / corrupt / readonly、reason は英語 */
export interface FormatHealthInfo {
	state: string;
	reason: string;
}

/** 再順位段 (リランカー) の状態 (c_16 §7.2.1)。reason は無効の理由の識別子 */
export interface RerankStatusInfo {
	mode: string;
	enabled: boolean;
	placement: string;
	ms_per_doc: number | null;
	candidates: number;
	reason: string;
	tested_at: string | null;
	/** 自己テストの後にリランカーのモデルが変わった (再テストはしない) */
	model_changed_since_selftest: boolean;
}

/** 埋め込みサーバの配置 GPU / CPU (c_16 §7.2.2)。setting は auto / explicit / default、reason は判別の理由の識別子 */
export interface EmbedPlacementInfo {
	setting: string;
	placement: string;
	gpu_layers: number;
	reason: string;
	/** 判別の時刻 (ISO 8601 UTC)。明示の設定・未判別なら null */
	decided_at: string | null;
	cpu_p50_ms: number | null;
	gpu_p50_ms: number | null;
	/** 同じ文の CPU と GPU のベクトルの cosine の最小 */
	cosine_min: number | null;
}

export interface StatusResponse {
	status: string;
	edition: string;
	instance_name: string;
	version: string;
	free_version?: string;
	pro_version?: string | null;
	data_generation?: number;
	uptime_seconds: number;
	llama_server: LlamaServerInfo;
	model?: ModelInfo;
	components: ComponentStatus[];
	memory: MemoryStats;
	cartridges_loaded: number;
	debug: DebugStatusInfo;
	capabilities?: CapabilityInfo[];
	/** 効果の死活監視の警告。空なら異常なし */
	liveness?: LivenessAlert[];
	/** データ根の状態 (readonly なら入力欄に常時表示) */
	data_health?: DataHealthInfo;
	/** 再順位段 (リランカー) の状態 */
	rerank?: RerankStatusInfo;
	/** 埋め込みサーバの配置 */
	embed_placement?: EmbedPlacementInfo;
}

/** ステータス取得 */
export async function getStatus(): Promise<StatusResponse> {
	return request<StatusResponse>('GET', '/status');
}
