/** 学習済み few-shot 手本の型定義 (f_04 §3.2.2) */

/** few-shot 手本の状態 */
export type FewShotState = 'active' | 'stale' | 'archived';

/** 手本に対する操作 */
export type FewShotAction = 'pin' | 'unpin' | 'archive' | 'restore';

/** バックエンド `FewShotExampleModel` に対応する 1 件分の手本 */
export interface FewShotExample {
	id: string;
	mode: string;
	query: string;
	response: string;
	fitness: number;
	quality_score: number | null;
	added_at: string;
	state: FewShotState;
	state_since: string;
	pinned: boolean;
	use_count: number;
	last_used_at: string;
	helpful: number;
	harmful: number;
	source_experience_id: string;
	lang: string;
}

/** `GET /api/learning/fewshot` のレスポンス */
export interface FewShotListResponse {
	examples: FewShotExample[];
	archived: FewShotExample[];
	pool_size: number;
	writeback: string;
}

/** `POST /api/learning/fewshot/{id}/{action}` のレスポンス */
export interface FewShotActionResponse {
	id: string;
	action: FewShotAction;
	ok: boolean;
}
