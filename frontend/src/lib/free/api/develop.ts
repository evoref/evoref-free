/** Develop エディション専用 API
 *
 * Develop 版でのみ mount される ``/api/develop/*`` を叩くクライアント。
 * Free / Pro ビルドでは対応エンドポイントが存在せず 404 になるため、
 * 呼び出し側 (DeveloperSettings) が ``isDevelop`` でガードすること。
 */

import { request } from './_client';

/** local/ 初期化レスポンス */
export interface ResetLocalDataResponse {
	/** "restarting" (デタッチヘルパー起動済み) */
	status: string;
	/** 人間向けの状況説明 */
	message: string;
	/** 起動したリセットヘルパーの PID (取得できた場合) */
	helper_pid: number | null;
}

/**
 * local/ データを setup.bat 直後の空スケルトンへ初期化し、サービスを再起動する。
 *
 * バックエンドはデタッチヘルパーを起動して即座に 202 を返す。その後ヘルパーが
 * backend を含む全サービスを停止 → local/ を wipe → 再起動するため、本リクエスト
 * 完了後ほどなく backend への接続は一時的に切れる (呼び出し側で再起動待ち UI を出す)。
 */
export async function resetLocalData(confirm: boolean): Promise<ResetLocalDataResponse> {
	return request<ResetLocalDataResponse>('POST', '/develop/reset-local-data', {
		confirm
	});
}
