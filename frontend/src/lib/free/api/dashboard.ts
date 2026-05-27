/** ダッシュボード関連の型定義 (関数なし)
 *
 * 学習スケジューラ・LoRA・改善カーブ等のフロントエンド状態用型を集約。
 * これらの型は対応する API エンドポイント (status / learning) のレスポンスや
 * dashboard ストアで使用される。
 */

/** モード別経験数 */
export interface ExperienceByMode {
	chat: number;
	coding: number;
}

/** Level 1 各フェーズの結果 */
export interface Level1ResultEntry {
	improved: boolean;
	fitness_before: number | null;
	fitness_after: number | null;
}

/** ポリシー進化ドメインの状態（Pro） */
export interface PolicyEvolverDomainStatus {
	current_fitness: number | null;
	best_fitness: number;
	decline_count: number;
	sigma: number;
	phase: string;
}

/** fitness 履歴の1ポイント */
export interface FitnessPoint {
	run: number;
	fitness: number;
}

/** 優先キュー要求のスナップショット */
export interface PriorityRequestEntry {
	reason: string;
	requested_at: number;
	relax_ratio: number;
	payload: Record<string, unknown> | null;
}

/** SUSPENDED な Level1Session の情報 */
export interface ActiveSessionInfo {
	session_id: string;
	started_at: number;
	reason: string;
	completed_phases: string[];
	yield_count: number;
	cartridge_snapshot: string[];
	experience_count: number;
}

/** スケジューラ状態（バックエンド SchedulerStatusModel 準拠） */
export interface SchedulerStatus {
	running: boolean;
	experience_count: number;
	new_experience_count: number;
	min_experiences: number;
	conditions_met: boolean;
	last_level1_run: string | null;
	last_level2_run: string | null;
	last_level0_record: string | null;
	experience_by_mode: ExperienceByMode;
	correction_rate: number;
	rag_usage_rate: number;
	prev_correction_rate: number | null;
	prev_rag_usage_rate: number | null;
	level1_run_count: number;
	last_level1_results: Record<string, Level1ResultEntry>;
	executed_phases: string[];
	fitness_history: Record<string, FitnessPoint[]>;
	policy_evolver_status: Record<string, PolicyEvolverDomainStatus>;
	/** 優先キュー */
	priority_queue: PriorityRequestEntry[];
	/** SUSPENDED な Level1Session（無ければ null） */
	active_session: ActiveSessionInfo | null;
}

/** ダッシュボード学習データ（フロントエンド状態用） */
export interface DashboardLearningData {
	running: boolean;
	experience_count: number;
	new_experience_count: number;
	min_experiences: number;
	conditions_met: boolean;
	last_level1_run: string | null;
	last_level2_run: string | null;
	lora_version: number;
	lora_adapter_exists: boolean;
	eval_cases_count: number;
	eval_pass_threshold: number;
	last_level0_record: string | null;
	experience_by_mode: ExperienceByMode;
	correction_rate: number;
	rag_usage_rate: number;
	prev_correction_rate: number | null;
	prev_rag_usage_rate: number | null;
	level1_run_count: number;
	last_level1_results: Record<string, Level1ResultEntry>;
	executed_phases: string[];
	fitness_history: Record<string, FitnessPoint[]>;
	policy_evolver_status: Record<string, PolicyEvolverDomainStatus>;
	/** 優先キュー / SUSPENDED session */
	priority_queue: PriorityRequestEntry[];
	active_session: ActiveSessionInfo | null;
}

/** ダッシュボード RAG 統計（フロントエンド状態用） */
export interface DashboardRagStats {
	chunkCount: number;
	vectorCount: number;
	indexSizeMb: number;
	/** 埋め込み次元と保存済みストアの次元が不一致 */
	embeddingDimMismatch: boolean;
	embeddingDimStored: number | null;
	embeddingDimCurrent: number;
}

/** 改善カーブのポイント */
export interface ImprovementScore {
	version: number;
	eval_score: number;
	created_at: string;
}

/** LoRA バージョン情報 */
export interface LoRAVersion {
	version: number;
	name: string;
	date: string;
	current: boolean;
}
