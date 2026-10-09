/** 記憶 API (統計 / ノート / ピン留め) */

import { request } from './_client';

export interface WorkingMemoryStats {
	turns: number;
	max_turns: number;
	tokens_used: number;
	max_tokens: number;
	session_id: string;
}

export interface ShortTermMemoryStats {
	notes: number;
	max_notes: number;
	pending_embeddings: number;
	pending_evolution: number;
}

export interface LongTermMemoryStats {
	chunks: number;
	index_size_mb: number;
	sources: number;
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

export interface MemoryDetailedStats {
	working: WorkingMemoryStats;
	short_term: ShortTermMemoryStats;
	long_term: LongTermMemoryStats;
	semantic: SemanticMemoryStats;
	current_mode: string;
}

export interface NoteInfo {
	id: string;
	content: string;
	keywords: string[];
	tags: string[];
	created_at: number;
	accessed_at: number;
	session_id: string;
	context_description: string;
	evolution_pending: boolean;
	has_embedding: boolean;
}

export interface MemoryNotesResponse {
	total: number;
	notes: NoteInfo[];
}

export interface PinnedFactInfo {
	id: string;
	subject: string;
	predicate: string;
	object: string;
	type: string;
	scope: string;
	confidence: number;
	pinned: boolean;
	/** ロック解除の UNIX 秒。未ロックは null (ロック中は force なしで解除できない) */
	pin_locked_until: number | null;
	mode_origin: string;
	created_at: number;
	accessed_at: number;
}

export interface PinnedFactsResponse {
	scope: string;
	total: number;
	facts: PinnedFactInfo[];
}

export interface PinFactResponse {
	fact: PinnedFactInfo;
}

export interface UnpinFactRequest {
	fact_id: string;
	scope?: string;
	force?: boolean;
}

/** 記憶システム全体の統計 */
export async function getMemoryStats(): Promise<MemoryDetailedStats> {
	return request<MemoryDetailedStats>('GET', '/memory/stats');
}

/** 短期記憶のノート一覧 */
export async function getMemoryNotes(limit = 20, offset = 0): Promise<MemoryNotesResponse> {
	return request<MemoryNotesResponse>('GET', `/memory/notes?limit=${limit}&offset=${offset}`);
}

/** ピン留めされたファクトの一覧 */
export async function listPinnedFacts(scope = 'global'): Promise<PinnedFactsResponse> {
	return request<PinnedFactsResponse>('GET', `/memory/pinned?scope=${encodeURIComponent(scope)}`);
}

/** 文章を新規ファクトとしてピン留めする */
export async function pinMemoryContent(
	content: string,
	scope = 'global'
): Promise<PinFactResponse> {
	return request<PinFactResponse>('POST', '/memory/pin', { content, scope });
}

/** ピン留めを解除する (ロック中は 409。force=true で上書き) */
export async function unpinMemoryFact(req: UnpinFactRequest): Promise<PinFactResponse> {
	return request<PinFactResponse>('POST', '/memory/unpin', req);
}
