/** ダッシュボード関連の型定義 (関数なし)
 *
 * 学習スケジューラ等のフロントエンド状態用型を集約。
 * これらの型は対応する API エンドポイント (status / learning) のレスポンスや
 * dashboard ストアで使用される。
 */

/** モード別経験数 */
export interface ExperienceByMode {
	chat: number;
	create: number;
}

/** Level 1 各フェーズの結果 */
export interface Level1ResultEntry {
	improved: boolean;
	fitness_before: number | null;
	fitness_after: number | null;
	/** 対象が無く実行していない (経験不足 / 選択圧なし / シグナル未配線) */
	skipped?: boolean;
	reason?: string | null;
	/** base prompt 採用ゲートの実測値 (現行 / 候補の judge 平均) */
	measured_before?: number | null;
	measured_after?: number | null;
	/** 採用ゲートの種類 (pairwise のとき measured_* は null で、勝敗が下に載る) */
	gate?: string | null;
	wins?: number | null;
	losses?: number | null;
	ties?: number | null;
	/** 現行 vs 現行 (カナリア) の |純勝ち| */
	noise_floor?: number | null;
	/** 一対比較で採用候補になったときだけ測る標本 (成功ターン) の勝敗 */
	sample_wins?: number | null;
	sample_losses?: number | null;
	sample_ties?: number | null;
	/** 失敗の証拠があったのに評価ケースにしなかった件数 (除外理由 → 件数) */
	excluded?: Record<string, number> | null;
	/** 標本に足した回帰台帳の guard の合否 */
	guards_passed?: number | null;
	guards_failed?: number | null;
	/** ゲートを通ったが自動採用が効かず提案にした回の提案 ID (reason === 'proposed') */
	proposal_id?: string | null;
}

/** Level 2 (成長) の要約 (`/api/learning/status` の scheduler_status.growth) */
export interface GrowthSummary {
	auto_adopt_configured: boolean;
	auto_adopt_effective: boolean;
	auto_adopt_reason: string;
	pending_proposals: number;
	guards: number;
	pressure: number;
	rollback_suggested: number;
}

/** ポリシー進化ドメインの状態（Pro） */
export interface PolicyEvolverDomainStatus {
	current_fitness: number | null;
	best_fitness: number;
	decline_count: number;
	sigma: number;
	phase: string;
	/** 恒真 fitness (選択圧ゼロ) と判定され凍結中か */
	degenerate: boolean;
	fitness_history_len: number;
	last_evolved_at: string | null;
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
	/** `--no-learning` 起動中かどうか (true の間は Level 1/2 が no-op) */
	is_disabled?: boolean;
	single_slot?: boolean;
	experience_count: number;
	new_experience_count: number;
	min_experiences: number;
	/**
	 * 経験件数だけを見た表示値。Level 1 の実ゲートにはアイドル時間・
	 * ユーザー活動・LLM クライアント配線も含まれるため、true でも走らない
	 * ことは正常状態として起こる。実際の詰まりは level1_blocked_reason を見る。
	 */
	conditions_met: boolean;
	/** Level 1 が今走れない理由 (走れる状態なら null) */
	level1_blocked_reason: string | null;
	/** waiting_for_idle のとき、アイドル成立までの残り秒数 */
	level1_seconds_until_idle: number | null;
	/** Level 1 を延期している理由 (serving_another_partition = create_model 等が載っている) */
	deferred_reason?: string | null;
	/** 延期が始まった時刻 (ISO 8601 UTC)。延期していなければ null */
	deferred_since?: string | null;
	last_level1_run: string | null;
	last_level0_record: string | null;
	experience_by_mode: ExperienceByMode;
	correction_rate: number;
	rag_usage_rate: number;
	prev_correction_rate: number | null;
	prev_rag_usage_rate: number | null;
	rag_gated_rate?: number;
	rag_pseudo_derived_rate?: number;
	rag_abstain_rate?: number;
	/** phase3 (embed_instruction) 部分集合の件数 */
	rag_score_experience_count?: number;
	/** phase4 (token_budget) 部分集合の件数 */
	long_form_experience_count?: number;
	/** 上 2 つの部分集合条件のしきい値 */
	phase_subset_min_experiences?: number;
	level1_run_count: number;
	last_level1_results: Record<string, Level1ResultEntry>;
	executed_phases: string[];
	fitness_history: Record<string, FitnessPoint[]>;
	policy_evolver_status: Record<string, PolicyEvolverDomainStatus>;
	/** 優先キュー */
	priority_queue: PriorityRequestEntry[];
	/** SUSPENDED な Level1Session（無ければ null） */
	active_session: ActiveSessionInfo | null;
	/** Level 2 (成長) の要約 (読めなければ null) */
	growth?: GrowthSummary | null;
}

/** ダッシュボード学習データ（フロントエンド状態用） */
export interface DashboardLearningData {
	running: boolean;
	experience_count: number;
	new_experience_count: number;
	min_experiences: number;
	/**
	 * 経験件数だけを見た表示値。Level 1 の実ゲートにはアイドル時間・
	 * ユーザー活動・LLM クライアント配線も含まれるため、true でも走らない
	 * ことは正常状態として起こる。実際の詰まりは level1_blocked_reason を見る。
	 */
	conditions_met: boolean;
	/** Level 1 が今走れない理由 (走れる状態なら null) */
	level1_blocked_reason: string | null;
	/** waiting_for_idle のとき、アイドル成立までの残り秒数 */
	level1_seconds_until_idle: number | null;
	last_level1_run: string | null;
	/** `--no-learning` 起動中 (true の間は学習が no-op) */
	is_disabled: boolean;
	last_level0_record: string | null;
	experience_by_mode: ExperienceByMode;
	correction_rate: number;
	rag_usage_rate: number;
	prev_correction_rate: number | null;
	prev_rag_usage_rate: number | null;
	rag_gated_rate?: number;
	rag_pseudo_derived_rate?: number;
	rag_abstain_rate?: number;
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
	/** 取り込み元ドキュメント（distinct source）数 */
	sourceCount: number;
	/** チャンク分割戦略（semantic 等） */
	chunkingStrategy: string;
	/** 埋め込み次元と保存済みストアの次元が不一致 */
	embeddingDimMismatch: boolean;
	embeddingDimStored: number | null;
	embeddingDimCurrent: number;
	/** 運用情報（store_info 由来。バックエンド未提供時は null） */
	createdAt: string | null;
	lastReindexAt: string | null;
	embeddingModel: string | null;
	embeddingBackend: string | null;
}
