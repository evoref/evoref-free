/**
 * config.yaml に対応するフロントエンド型定義
 *
 * backend/schemas/ 配下の Pydantic モデルと 1:1 対応。
 * configSection() ヘルパーと組み合わせて型安全なアクセスを提供する。
 */

// ── セクション別型定義 ──────────────────────────────────────

export interface InstanceConfig {
	name: string;
}

export interface ServerConfig {
	host: string;
	port: number;
	frontend_port: number;
	timeout: number;
}

export interface LlamaConfig {
	host: string;
	port: number;
	context_size: number;
	gpu_layers: number;
	threads: number;
	batch_size: number;
	flash_attn: boolean;
	mlock: boolean;
	cache_prompt: boolean;
	slots: number;
	cache_type_k: string;
	cache_type_v: string;
	max_tokens: number;
	lora_target: string;
	enable_thinking: boolean | null;
	extra_args: string[];
}

export interface ThemeConfig {
	active: string;
	color_mode: 'dark' | 'light';
	trusted: string[];
	cli_layout_mode: 'auto' | 'split' | 'sequential';
}

export interface ModelPathsConfig {
	base_model: string;
	coding_model?: string | null;
	[key: string]: unknown;
}

export interface LocalPathsConfig {
	lora_adapter: string;
	lora_versions_dir: string;
	assist_lora_adapter: string;
	assist_lora_versions_dir: string;
	experience_assist_file: string;
	eval_assist_file: string;
	lora_archive_dir: string;
	embed_lora_adapter: string;
	embed_lora_versions_dir: string;
	vectors_dir: string;
	knowledge_dir: string;
	experience_file: string;
	eval_core_file: string;
	model_state_file: string;
	local_state_file: string;
	memory_dir: string;
	prompts_dir: string;
	cartridges_dir: string;
	history_dir: string;
	learned_patterns_file: string;
	themes_dir: string;
	migration_archive_dir: string;
	triggers_dir: string;
	[key: string]: unknown;
}

export interface HistoryConfig {
	auto_save: boolean;
	checkpoint_interval: number;
	retention_full_days: number;
	retention_compressed_days: number;
	max_storage_mb: number;
	summary_batch_size: number;
}

export interface RAGConfig {
	chunk_size: number;
	chunk_overlap: number;
	chunking_strategy: 'semantic' | 'fixed';
	semantic_min_chunk: number;
	semantic_max_chunk: number;
	top_k: number;
	hybrid_search: boolean;
	bm25_weight: number;
	vector_weight: number;
	fusion_method: 'rrf' | 'weighted';
	rrf_k: number;
	embedding_dim: number;
	contextual_retrieval: boolean;
	contextual_prefix_max_tokens: number;
	contextual_max_doc_chars: number;
	contextual_batch_size: number;
	quantization: 'none' | 'int8';
	rescore_candidates: number;
	memmap_threshold: number;
	relevance_threshold: number;
	support_threshold: number;
	confidence_threshold: number;
	hysteresis_band: number;
	self_rag: {
		assist_judge: {
			enabled: boolean;
			max_per_session: number;
			max_per_query: number;
			only_when_quality: string[];
		};
	};
}

export interface EmbeddingConfig {
	backend: 'llama-cpp';
	llama_host: string;
	llama_port: number;
	dim: number;
	timeout: number;
	model_path: string;
	max_length: number;
	model_name: string;
	cache_enabled: boolean;
	cache_max_mb: number;
	cache_dir: string;
}

export interface MemoryConfig {
	working_max_turns: number;
	working_max_tokens: number;
	short_term_max_notes: number;
	lightmem_decay_days: number;
	fade_alpha: number;
	fade_beta: number;
	fade_gamma: number;
	fade_threshold: number;
	conflict_similarity_threshold: number;
	conflict_batch_size: number;
	note_evolution_enabled: boolean;
	note_evolution_batch: number;
	note_evolution_context_k: number;
	llm_call_base_interval: number;
}

export interface LearningConfig {
	level1_min_experiences: number;
	level1_generations: number;
	level1_population_size: number;
	level2_min_failures: number;
	full_idle_minutes: number;
	level1_idle_minutes: number;
	level1_recheck_interval_sec: number;
	priority_threshold_ratio: number;
	active_minutes: number;
	level2_spsa_iterations: number;
	level2_sparse_params: number;
	level2_schedule_hour: number;
	pattern_max_patterns: number;
	pattern_initial_weight: number;
	pattern_decay_rate: number;
	pattern_boost_amount: number;
	pattern_min_weight: number;
	pattern_match_threshold: number;
	[key: string]: unknown;
}

export interface AgentConfig {
	step_compaction_enabled: boolean;
	step_compaction_rag_lines: number;
	step_compaction_command_head_tail: number;
	reminders_enabled: boolean;
	max_reminders_per_turn: number;
	dangerous_command_block: boolean;
	tool_judge_enabled: boolean;
	meta_cognitive_enabled: boolean;
	meta_cognitive_min_budget: number;
}

export interface ToolsConfig {
	fetch_url_enabled: boolean;
	fetch_url_timeout: number;
}

export interface WidgetApiConfig {
	name: string;
	base_url: string;
	api_key_env: string;
	api_key_header: string;
	api_key_param: string;
	rate_limit: string;
	allowed_paths: string[];
}

export interface WidgetProxyConfig {
	enabled: boolean;
	global_rate_limit: string;
	request_timeout_sec: number;
	max_response_size_kb: number;
	cache_ttl_sec: number;
	apis: WidgetApiConfig[];
}

export interface ExternalApiConfig {
	enabled: boolean;
	provider: 'anthropic' | 'openai';
	api_key: string;
	model: string;
	max_tokens: number;
	timeout: number;
}

export interface I18nConfig {
	locale: 'ja' | 'en';
	fallback: 'ja' | 'en';
	prompt_locale: 'ja' | 'en';
}

export interface DebugConfig {
	enabled: boolean;
	log_dir: string;
	log_level: 'DEBUG' | 'INFO' | 'WARNING';
	log_requests: boolean;
	log_rag: boolean;
	log_memory: boolean;
	log_learning: boolean;
	log_long_form: boolean;
	max_log_mb: number;
}

export interface AssistModelLocalConfig {
	host: string;
	port: number;
	context_size: number;
	[key: string]: unknown;
}

export interface AssistModelConcurrencyConfig {
	realtime: number;
	background: number;
	learning: number;
	[key: string]: number;
}

export interface AssistModelConfig {
	local: AssistModelLocalConfig | null;
	model_path: string;
	timeout: number;
	concurrency: AssistModelConcurrencyConfig;
	[key: string]: unknown;
}

export interface LongFormConfig {
	max_units: number;
	unit_max_tokens: number;
	rolling_short_term_chars: number;
	review_enabled: boolean;
	max_revisions: number;
	rag_per_unit: boolean;
	rag_top_k_per_unit: number;
}

export interface ChatModeConfig {
	temperature: number;
	top_p: number;
	top_k: number;
	presence_penalty: number;
}

export type CodingModeConfig = ChatModeConfig;

export interface ModesConfig {
	chat: ChatModeConfig;
	coding: CodingModeConfig;
}

export interface EditorSettingsConfig {
	font_family: string;
	font_size: number;
	line_height: number;
	tab_size: number;
	word_wrap: boolean;
	show_line_numbers: boolean;
	show_active_line: boolean;
	show_toolbar: boolean;
	default_encoding: string;
	default_line_ending: 'lf' | 'crlf' | 'cr';
	highlight_languages: string[];
}

// ── トップレベル設定型 ──────────────────────────────────────

/** config.yaml の全セクションに対応する型 */
export interface ConfigData {
	instance: InstanceConfig;
	server: ServerConfig;
	llama: LlamaConfig;
	theme: ThemeConfig;
	model_paths: ModelPathsConfig;
	local_paths: LocalPathsConfig;
	history: HistoryConfig;
	rag: RAGConfig;
	embedding: EmbeddingConfig;
	memory: MemoryConfig;
	learning: LearningConfig;
	agent: AgentConfig;
	tools: ToolsConfig;
	widget_proxy: WidgetProxyConfig;
	external_api: ExternalApiConfig;
	i18n: I18nConfig;
	debug: DebugConfig;
	assist_model: AssistModelConfig;
	modes: ModesConfig;
	editor: EditorSettingsConfig;
	long_form: LongFormConfig;
}
