/** ダッシュボード関連の型定義 (関数なし)
 *
 * 学習スケジューラ・LoRA・改善カーブ等のフロントエンド状態用型を集約。
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
}

/** ポリシー進化ドメインの状態（Pro） */
export interface PolicyEvolverDomainStatus {
	current_fitness: number | null;
	best_fitness: number;
	decline_count: number;
	sigma: number;
	phase: string;
	/** 恒真 fitness (選択圧ゼロ) と判定され凍結中か */
	degenerate?: boolean;
	fitness_history_len?: number;
	last_evolved_at?: string | null;
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

/** Level 2 の状態 + 発火条件（Pro） */
export interface Level2TargetStatus {
	method: string;
	bootstrap_enabled: boolean;
	adapter_exists: boolean;
	version: number;
	/** 蓄積中の発火データ数（失敗数） */
	experiences_current: number;
	bootstrap_min: number;
	spsa_min: number;
	cvector_min: number;
	/** 発火しない理由コード（"" = 発火可能）。表示は i18n でラベル化する */
	block_reason: string;
}

/** Level 2 自動発火の共通タイミングゲート（Pro） */
export interface Level2Gates {
	active_minutes: number;
	overdue_hours: number;
	recheck_interval_sec: number;
	/**
	 * target ごとの **実効** クールダウン時間（h）。連続無改善が stale_streak に
	 * 達した target は延長クールダウンへ落ちるため overdue_hours とは一致しない。
	 */
	cooldown_hours?: Record<string, number>;
	/** target ごとの連続無改善回数（採用成功で 0 に戻る） */
	no_improve_streak?: Record<string, number>;
	/** 延長クールダウンへ落ちる連続無改善回数の閾値 */
	stale_streak?: number;
	/**
	 * target ごとの「クールダウンを過ぎているか」。データ充足（block_reason==""）
	 * と AND を取って初めて実際に発火する。未提供（旧バックエンド）は true 扱い。
	 */
	overdue?: Record<string, boolean>;
	/** target ごとの前回実行からの経過秒。未実行は null */
	seconds_since_run?: Record<string, number | null>;
}

/** Level 2 (LoRA) の状態（Pro のみ非 null） */
export interface Level2Status {
	running_target: string | null;
	next_target: string;
	base: Level2TargetStatus;
	gates: Level2Gates;
}

/** スケジューラ状態（バックエンド SchedulerStatusModel 準拠） */
export interface SchedulerStatus {
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
	last_level2_run: string | null;
	/** 実行中の Level 2 対象（"base"/null） */
	running_target: string | null;
	/** Level 2 (LoRA) の状態（Pro のみ非 null） */
	level2: Level2Status | null;
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
	last_level2_run: string | null;
	/** 実行中の Level 2 対象（"base"/null） */
	running_target: string | null;
	/** Level 2 (LoRA) の状態（Pro のみ非 null） */
	level2: Level2Status | null;
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
	/** 取り込み元ドキュメント（distinct source）数 */
	sourceCount: number;
	/** チャンク分割戦略（semantic 等） */
	chunkingStrategy: string;
	/** BM25 + ベクトルのハイブリッド検索が有効か */
	hybridSearch: boolean;
	/** スコア融合手法（rrf 等） */
	fusionMethod: string;
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
	/** スナップショットの実バイト数 (不明時は null)。日時と併せて表示する */
	sizeBytes?: number | null;
}
