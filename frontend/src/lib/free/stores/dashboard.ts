/**
 * ダッシュボード store (Free) — RAG 統計・学習状態 (Level 0/1) のデータ取得
 */
import type { DashboardLearningData, DashboardRagStats } from '$lib/free/api';

// ── デフォルト値 ──

export const DEFAULT_LEARNING_DATA: DashboardLearningData = {
	running: false,
	experience_count: 0,
	new_experience_count: 0,
	min_experiences: 5,
	conditions_met: false,
	last_level1_run: null,
	last_level2_run: null,
	running_target: null,
	level2: null,
	lora_version: 0,
	lora_adapter_exists: false,
	eval_cases_count: 0,
	eval_pass_threshold: 0.0,
	last_level0_record: null,
	experience_by_mode: { chat: 0, create: 0 },
	correction_rate: 0,
	rag_usage_rate: 0,
	prev_correction_rate: null,
	prev_rag_usage_rate: null,
	level1_run_count: 0,
	last_level1_results: {},
	executed_phases: [],
	fitness_history: {},
	policy_evolver_status: {},
	priority_queue: [],
	active_session: null
};

export const DEFAULT_RAG_STATS: DashboardRagStats = {
	chunkCount: 0,
	vectorCount: 0,
	indexSizeMb: 0,
	sourceCount: 0,
	chunkingStrategy: 'semantic',
	hybridSearch: false,
	fusionMethod: 'rrf',
	embeddingDimMismatch: false,
	embeddingDimStored: null,
	embeddingDimCurrent: 0,
	createdAt: null,
	lastReindexAt: null,
	embeddingModel: null,
	embeddingBackend: null
};

// ── データ取得 ──

export async function safeFetch<T>(url: string): Promise<T | null> {
	try {
		const resp = await fetch(url);
		return resp.ok ? await resp.json() : null;
	} catch {
		return null;
	}
}

export function buildLearningData(data: Record<string, unknown>): DashboardLearningData {
	const sched = (data.scheduler_status as Record<string, unknown>) ?? {};
	const byMode = (sched.experience_by_mode as Record<string, number>) ?? {};
	return {
		running: (sched.running as boolean) ?? false,
		experience_count: (sched.experience_count as number) ?? 0,
		new_experience_count: (sched.new_experience_count as number) ?? 0,
		min_experiences: (sched.min_experiences as number) ?? 5,
		conditions_met: (sched.conditions_met as boolean) ?? false,
		last_level1_run: (sched.last_level1_run as string) ?? null,
		last_level2_run: (sched.last_level2_run as string) ?? null,
		running_target: (sched.running_target as string) ?? null,
		level2: (sched.level2 as import('$lib/free/api').Level2Status) ?? null,
		lora_version: (data.lora_version as number) ?? 0,
		lora_adapter_exists: (data.lora_adapter_exists as boolean) ?? false,
		eval_cases_count: (data.eval_cases_count as number) ?? 0,
		eval_pass_threshold: (data.eval_pass_threshold as number) ?? 0.0,
		last_level0_record: (sched.last_level0_record as string) ?? null,
		experience_by_mode: { chat: byMode.chat ?? 0, create: byMode.create ?? 0 },
		correction_rate: (sched.correction_rate as number) ?? 0,
		rag_usage_rate: (sched.rag_usage_rate as number) ?? 0,
		prev_correction_rate: (sched.prev_correction_rate as number) ?? null,
		prev_rag_usage_rate: (sched.prev_rag_usage_rate as number) ?? null,
		level1_run_count: (sched.level1_run_count as number) ?? 0,
		last_level1_results: (sched.last_level1_results as Record<string, import('$lib/free/api').Level1ResultEntry>) ?? {},
		executed_phases: (sched.executed_phases as string[]) ?? [],
		fitness_history: (sched.fitness_history as Record<string, import('$lib/free/api').FitnessPoint[]>) ?? {},
		policy_evolver_status: (sched.policy_evolver_status as Record<string, import('$lib/free/api').PolicyEvolverDomainStatus>) ?? {},
		priority_queue: (sched.priority_queue as import('$lib/free/api').PriorityRequestEntry[]) ?? [],
		active_session: (sched.active_session as import('$lib/free/api').ActiveSessionInfo | null) ?? null
	};
}

export interface FreeDashboardData {
	learningData: DashboardLearningData;
	ragStats: DashboardRagStats;
	hasError: boolean;
}

/**
 * ダッシュボード表示に必要な Free 側 2 API (RAG 統計 / 学習状態) を並列 fetch する
 */
export async function fetchFreeDashboardData(): Promise<FreeDashboardData> {
	const results = await Promise.all([
		safeFetch<Record<string, unknown>>('/api/rag/stats'),
		safeFetch<Record<string, unknown>>('/api/learning/status')
	]);
	const [ragData, learningStatusData] = results;
	const hasError = results.some((r) => r === null);

	const ragStats: DashboardRagStats = ragData
		? {
				chunkCount: (ragData.total_chunks as number) ?? 0,
				vectorCount: (ragData.total_vectors as number) ?? 0,
				indexSizeMb: (ragData.index_size_mb as number) ?? 0,
				sourceCount: (ragData.total_sources as number) ?? 0,
				chunkingStrategy: (ragData.chunking_strategy as string) ?? 'semantic',
				hybridSearch: (ragData.hybrid_search as boolean) ?? false,
				fusionMethod: (ragData.fusion_method as string) ?? 'rrf',
				embeddingDimMismatch: (ragData.embedding_dim_mismatch as boolean) ?? false,
				embeddingDimStored: (ragData.embedding_dim_stored as number | null) ?? null,
				embeddingDimCurrent: (ragData.embedding_dim as number) ?? 0,
				createdAt: (ragData.created_at as string | null) ?? null,
				lastReindexAt: (ragData.last_reindex_at as string | null) ?? null,
				embeddingModel: (ragData.embedding_model as string | null) ?? null,
				embeddingBackend: (ragData.embedding_backend as string | null) ?? null
			}
		: { ...DEFAULT_RAG_STATS };

	const learningData: DashboardLearningData = learningStatusData
		? buildLearningData(learningStatusData)
		: { ...DEFAULT_LEARNING_DATA };

	return { learningData, ragStats, hasError };
}
