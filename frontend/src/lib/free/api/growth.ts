/** Level 2 (成長) API — 提案の採用 / 却下・巻き戻し・回帰ケースの除外 (docs/f_04 §6.8)
 *
 * `backend/free/api/learning/growth.py` に対応する。型はバックエンドの
 * `backend/free/api/learning/_growth_schemas.py` と揃える (`test_frontend_type_parity`)。
 */

import { request } from './_client';

/** 自動採用の設定と、いま効いているか (効かなければ満たさなかった最初の条件) */
export interface AutoAdoptStatus {
	configured: boolean;
	effective: boolean;
	reason: string;
}

/** 採用ゲートの結果 (測っていない値は null) */
export interface GrowthGate {
	name: string;
	wins: number | null;
	losses: number | null;
	ties: number | null;
	noise_floor: number | null;
	sample_wins: number | null;
	sample_losses: number | null;
	sample_ties: number | null;
	guards_passed: number | null;
	guards_failed: number | null;
	case_digest: string;
}

/** 未決の提案の要約 (候補の本文は詳細でだけ返る) */
export interface GrowthProposalSummary {
	proposal_id: string;
	asset_kind: string;
	mode: string;
	parent_version: string;
	current_version: string;
	/** 提案の後に現行の版が変わった (採用すると 409 stale_parent) */
	stale: boolean;
	created_at: string;
	expires_at: string;
	measure: GrowthGate | null;
	candidate_deltas: Record<string, number>;
	regression_case_ids: string[];
	guard_ids: string[];
}

/** 提案の詳細 (差分表示用に候補と現行の本文を持つ) */
export interface GrowthProposalDetail extends GrowthProposalSummary {
	candidate_text: string;
	current_text: string;
	current_deltas: Record<string, number>;
}

/** 成長の台帳の 1 行 (ID だけ) */
export interface GrowthLedgerRow {
	row_id: string;
	written_at: string;
	proposal_id: string;
	op: string;
	asset_kind: string;
	mode: string;
	parent_version: string;
	candidate_digest: string;
	decided_by: string | null;
	gate: GrowthGate | null;
	reason: string;
	adoption_id: string | null;
	regression_case_ids: string[];
	experience_ids: string[];
	level1_session_id: string | null;
	model_key: string;
}

/** 回帰台帳の件数 */
export interface RegressionSummary {
	active: number;
	guards: number;
	pressure: number;
	flaky: number;
	quarantined: number;
	retired: number;
	by_source: Record<string, number>;
	by_mode: Record<string, Record<string, number>>;
	last_run_at: string | null;
}

/** 採用した版の記録 (guard 監視と巻き戻しの提案) */
export interface GrowthAdoption {
	asset_kind: string;
	mode: string;
	decided_by: string;
	proposal_id: string;
	adopted_version: string;
	adopted_at: string | null;
	monitoring: boolean;
	/** 利用者が採用した版で回帰が起きた (自動では巻き戻さない) */
	rollback_suggested: boolean;
	suggestion_reason: string;
}

/** GET /api/learning/growth */
export interface GrowthResponse {
	auto_adopt: AutoAdoptStatus;
	proposals: GrowthProposalSummary[];
	recent: GrowthLedgerRow[];
	regression: RegressionSummary;
	adoptions: GrowthAdoption[];
}

/** 採用・却下・巻き戻しの結果 */
export interface GrowthActionResponse {
	asset_kind: string;
	mode: string;
	proposal_id: string | null;
	version: string | null;
	row_id: string | null;
	adoption_id: string | null;
}

/** 回帰ケースの除外 (retire) / 除外の取り消し (unretire) の結果 */
export interface RegressionRetireResponse {
	case_id: string;
	/** 操作の後の台帳の状態 */
	status: string;
	changed: boolean;
	/** 操作の前の台帳の状態 */
	previous_status: string;
}

/** 回帰ケースの一覧の 1 行 (問いは先頭の要約だけ。本文・期待値は返らない) */
export interface RegressionCaseItem {
	case_id: string;
	mode: string;
	source: string;
	/** 台帳の状態 (active / quarantined / retired) */
	status: string;
	/** active のケースの役割 (guard / pressure / flaky / unknown)。それ以外は空 */
	role: string;
	/** 同梱の種ケース (retire できない) */
	seed: boolean;
	query_preview: string;
	query_chars: number;
	written_at: string;
	last_result: string | null;
	last_run_at: string | null;
}

/** GET /api/learning/growth/regression */
export interface RegressionCaseListResponse {
	cases: RegressionCaseItem[];
	/** 絞り込んだ後の件数 (ページングの前) */
	total: number;
	limit: number;
	offset: number;
}

export type GrowthAssetKind = 'prompt' | 'generation_params';

/** 自動採用の状態・未決の提案・直近の台帳・回帰台帳の件数 */
export async function getGrowth(): Promise<GrowthResponse> {
	return request<GrowthResponse>('GET', '/learning/growth');
}

/** 提案の詳細 (差分表示用) */
export async function getGrowthProposal(id: string): Promise<GrowthProposalDetail> {
	return request<GrowthProposalDetail>(
		'GET',
		`/learning/growth/proposals/${encodeURIComponent(id)}`
	);
}

/** 提案を採用する (現行の版が変わっていれば / 学習の実行中は 409) */
export async function adoptGrowthProposal(id: string): Promise<GrowthActionResponse> {
	return request<GrowthActionResponse>(
		'POST',
		`/learning/growth/proposals/${encodeURIComponent(id)}/adopt`
	);
}

/** 提案を却下する */
export async function rejectGrowthProposal(id: string): Promise<GrowthActionResponse> {
	return request<GrowthActionResponse>(
		'POST',
		`/learning/growth/proposals/${encodeURIComponent(id)}/reject`
	);
}

/** 資産を直前の版へ戻す */
export async function rollbackGrowth(
	assetKind: GrowthAssetKind,
	mode: string
): Promise<GrowthActionResponse> {
	return request<GrowthActionResponse>('POST', '/learning/growth/rollback', {
		asset_kind: assetKind,
		mode
	});
}
