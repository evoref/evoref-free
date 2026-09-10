/** 学習済み few-shot 手本 API
 *
 * `backend/free/api/learning/fewshot.py` に対応する取得 / 操作関数を提供する。
 * 一覧 (プール内 + 退避済み) と pin / unpin / archive / restore の手動操作のみ。
 */

import { request } from './_client';
import type {
	FewShotAction,
	FewShotActionResponse,
	FewShotListResponse
} from '$lib/free/types/fewshot';

export type {
	FewShotAction,
	FewShotActionResponse,
	FewShotExample,
	FewShotListResponse,
	FewShotState
} from '$lib/free/types/fewshot';

/**
 * few-shot 手本一覧を取得する
 *
 * @param mode 'chat' | 'create' 等でフィルタ。省略時は全モード。
 */
export async function listFewshot(mode?: string): Promise<FewShotListResponse> {
	const query = mode ? `?mode=${encodeURIComponent(mode)}` : '';
	return request<FewShotListResponse>('GET', `/learning/fewshot${query}`);
}

/** 指定した手本に pin / unpin / archive / restore を実行する */
export async function actOnFewshot(
	id: string,
	action: FewShotAction
): Promise<FewShotActionResponse> {
	return request<FewShotActionResponse>(
		'POST',
		`/learning/fewshot/${encodeURIComponent(id)}/${action}`
	);
}
