/**
 *
 * バックエンドの `/api/memory/*` エンドポイントに対応する型と
 * 取得関数を定義する。`memoryService.ts` から利用される薄いラッパー層。
 */

import { request } from './_client';

// ──────────────────────────────────────────────────────────────────────────
// 型定義
// ──────────────────────────────────────────────────────────────────────────

export interface FactInfo {
	id: string;
	subject: string;
	predicate: string;
	object: string;
	type: string;
	scope: string;
	confidence: number;
	pinned: boolean;
	pin_locked_until: number | null;
	mode_origin: string;
	created_at: number;
	accessed_at: number;
	access_count: number;
	superseded_by: string | null;
	supersedes: string[];
	auto_evolved: boolean;
	failure_signature: string | null;
	eval_metric: Record<string, number> | null;
	trace_id: string | null;
	private: boolean;
	requires_user_review: boolean;
	review_status: string;
}

export interface FactsResponse {
	scope: string;
	total: number;
	facts: FactInfo[];
}

export interface TaskFactInfo {
	fact_id: string;
	task_id: string;
	project_id: string;
	title: string;
	description: string;
	depends_on: string[];
	salience: number;
	status: string;
	source_path: string | null;
	created_at: number;
	accessed_at: number;
	access_count: number;
}

export interface TaskFactsResponse {
	scope: string;
	total: number;
	tasks: TaskFactInfo[];
}

export interface SemanticMemoryScopeStats {
	scope: string;
	total: number;
	active: number;
	superseded: number;
	pinned: number;
	by_type: Record<string, number>;
	by_mode_origin: Record<string, number>;
}

export interface SemanticMemoryStats {
	scopes: SemanticMemoryScopeStats[];
	total_facts: number;
	total_pinned: number;
}

export interface WorkingMemoryStatsDetail {
	turns: number;
	max_turns: number;
	tokens_used: number;
	max_tokens: number;
	session_id: string;
}

export interface ShortTermMemoryStatsDetail {
	notes: number;
	max_notes: number;
	pending_embeddings: number;
	pending_evolution: number;
	avg_lightmem_score: number;
}

export interface LongTermMemoryStatsDetail {
	chunks: number;
	index_size_mb: number;
	sources: number;
}

export interface FadeMemStatsDetail {
	alpha: number;
	beta: number;
	gamma: number;
	threshold: number;
}

export interface MemoryDetailedStats {
	working: WorkingMemoryStatsDetail;
	short_term: ShortTermMemoryStatsDetail;
	long_term: LongTermMemoryStatsDetail;
	fadem: FadeMemStatsDetail;
	semantic: SemanticMemoryStats;
	current_mode: string;
}

// ──────────────────────────────────────────────────────────────────────────
// クエリパラメータ
// ──────────────────────────────────────────────────────────────────────────

export interface FactsQuery {
	scope?: string;
	type?: string;
	mode?: 'chat' | 'coding';
	pinned?: boolean;
	include_superseded?: boolean;
	limit?: number;
	offset?: number;
	sort?: 'created' | 'accessed' | 'confidence';
}

export interface PolicyQuery {
	scope?: string;
	include_superseded?: boolean;
	only_active?: boolean;
	limit?: number;
	offset?: number;
}

export interface TasksQuery {
	scope: string;
	status?: 'open' | 'in_progress' | 'done' | 'failed';
	limit?: number;
	offset?: number;
}

// ── Conflict Resolution ─────────────────────────

export interface ConflictGroupInfo {
	scope: string;
	subject: string;
	predicate: string;
	type: string;
	facts: FactInfo[];
	detected_at: number | null;
	decision: string | null;
	reason: string | null;
	winner_id: string | null;
	loser_ids: string[];
}

export interface ConflictsResponse {
	scope: string;
	pending: ConflictGroupInfo[];
	auto_resolved_history: ConflictGroupInfo[];
}

export interface ConflictsQuery {
	scope?: string;
	history_limit?: number;
}

export type ConflictAction = 'keep_old' | 'keep_new' | 'merge';

export interface ResolveConflictRequest {
	scope: string;
	winner_id: string;
	loser_ids: string[];
	action: ConflictAction;
	merged_object?: string;
}

export interface ResolveConflictResponse {
	scope: string;
	action: string;
	winner_id: string;
	superseded_ids: string[];
	new_fact_id: string | null;
}

// ──────────────────────────────────────────────────────────────────────────
// 関数
// ──────────────────────────────────────────────────────────────────────────

function buildQuery(params: Record<string, unknown>): string {
	const sp = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) {
		if (v === undefined || v === null) continue;
		sp.set(k, String(v));
	}
	const qs = sp.toString();
	return qs ? `?${qs}` : '';
}

export async function getMemoryStats(): Promise<MemoryDetailedStats> {
	return request<MemoryDetailedStats>('GET', '/memory/stats');
}

export async function listFacts(query: FactsQuery = {}): Promise<FactsResponse> {
	return request<FactsResponse>(
		'GET',
		`/memory/facts${buildQuery(query as Record<string, unknown>)}`
	);
}

export async function listPolicyFacts(query: PolicyQuery = {}): Promise<FactsResponse> {
	return request<FactsResponse>(
		'GET',
		`/memory/policy${buildQuery(query as Record<string, unknown>)}`
	);
}

export async function listTaskFacts(query: TasksQuery): Promise<TaskFactsResponse> {
	return request<TaskFactsResponse>(
		'GET',
		`/memory/tasks${buildQuery(query as unknown as Record<string, unknown>)}`
	);
}

export async function listConflicts(
	query: ConflictsQuery = {}
): Promise<ConflictsResponse> {
	return request<ConflictsResponse>(
		'GET',
		`/memory/conflicts${buildQuery(query as Record<string, unknown>)}`
	);
}

export async function resolveConflict(
	body: ResolveConflictRequest
): Promise<ResolveConflictResponse> {
	return request<ResolveConflictResponse>(
		'POST',
		'/memory/conflicts/resolve',
		body
	);
}
