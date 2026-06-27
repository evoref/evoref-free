/** RAG API (Free)
 *
 * ベクトルインデックスの再構築 (reindex) と、埋め込みモデル切替後の
 * 検索閾値ヒューリスティック較正 (calibrate) を提供する。
 */

import { request } from './_client';

/** reindex レスポンス (dry_run / 実行で一部フィールドが異なる) */
export interface ReindexResponse {
	dry_run: boolean;
	rag_chunks: number;
	cartridge_chunks: number;
	/** dry_run 時のみ: カートリッジ別件数 */
	cartridges?: { id: string; chunks: number }[];
	/** dry_run 時のみ: 対象メモリノート数 */
	memory_notes?: number;
	/** 実行時のみ: 再構築したカートリッジ ID */
	cartridges_rebuilt?: string[];
	/** 実行時のみ: リセットしたメモリノート数 */
	memory_notes_reset?: number;
	/** 実行時のみ: 所要秒 */
	elapsed_sec?: number;
	/** 実行時のみ: 再構築後も次元不一致が残るか */
	embedding_dim_mismatch?: boolean;
}

/** 閾値較正の提案値 (rag セクション) */
export interface ThresholdSuggestions {
	relevance_threshold: number;
	support_threshold: number;
	confidence_threshold: number;
}

/** スコア分布統計 (UI 表示用) */
export interface ThresholdDistribution {
	match_top1_p25: number;
	match_top1_p50: number;
	match_top1_p75: number;
	match_top3_p25: number;
	match_top3_p50: number;
	background_p50: number;
	background_p95: number;
}

/** 閾値較正レスポンス */
export interface CalibrateThresholdsResponse {
	ok: boolean;
	/** ok=false の理由 (no_vectors / insufficient_vectors) */
	reason?: string;
	n_vectors: number;
	sampled?: number;
	distribution?: ThresholdDistribution;
	suggestions?: ThresholdSuggestions;
}

/** ベクトルインデックスを現在の埋め込みモデルで再構築する。
 *
 * - `dry_run`: 対象件数だけ返して実行しない (プレビュー用)
 * - `cartridge`: 指定するとそのカートリッジのみ再構築
 * reindex は QUERY パラメータ方式 (JSON ボディ無し)。
 */
export async function reindexVectors(
	opts: { dry_run?: boolean; cartridge?: string } = {}
): Promise<ReindexResponse> {
	const params = new URLSearchParams();
	params.set('dry_run', String(opts.dry_run ?? false));
	if (opts.cartridge) params.set('cartridge', opts.cartridge);
	return request<ReindexResponse>('POST', `/rag/reindex?${params.toString()}`);
}

/** 再構築済みベクトルのスコア分布から rag.* 閾値の推奨値を取得する (提案のみ)。 */
export async function calibrateThresholds(): Promise<CalibrateThresholdsResponse> {
	return request<CalibrateThresholdsResponse>('POST', '/rag/calibrate-thresholds');
}
