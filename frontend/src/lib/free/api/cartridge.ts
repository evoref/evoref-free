/** カートリッジ API (Free)
 *
 * インストール / 一覧 / ロード / アンロード / 削除 / 再構築
 * の Free エディション機能のみを提供する。カートリッジ作成
 * (POST /api/pro/cartridges/create) は Pro 専用機能のため
 * `$lib/pro/api/cartridge.ts` 側に分離している。
 */

import {
	BASE_URL,
	request,
	requestVoid
} from './_client';
import { streamSSEFormData, type SSEProgressEvent } from './sse_progress';

/** カートリッジ情報（一覧用） */
export interface Cartridge {
	id: string;
	name: string;
	version: string;
	description: string;
	status: string;
	chunks: number;
	size_mb: number;
}

/** カートリッジ詳細情報（全フィールド） */
export interface CartridgeDetail {
	id: string;
	name: string;
	version: string;
	author: string;
	description: string;
	tags: string[];
	language: string;
	chunks: number;
	doc_count: number;
	size_mb: number;
	status: string;
	priority: number;
	installed_at: string;
	compatibility: string;
}

/** カートリッジ再構築結果 */
export interface CartridgeRebuildResult {
	id: string;
	name: string;
	chunks: number;
	status: string;
	size_mb: number;
	rebuild_time_sec: number;
	embedder_used: string;
}

/** カートリッジ一覧取得 */
export async function getCartridges(): Promise<Cartridge[]> {
	const data = await request<{ cartridges: Cartridge[] }>('GET', '/cartridges');
	return data.cartridges;
}

/** カートリッジ読込み */
export async function loadCartridge(id: string): Promise<void> {
	return requestVoid('POST', `/cartridges/${id}/load`);
}

/** カートリッジ取外し */
export async function unloadCartridge(id: string): Promise<void> {
	return requestVoid('POST', `/cartridges/${id}/unload`);
}

/** カートリッジ削除 */
export async function deleteCartridge(id: string): Promise<void> {
	return requestVoid('DELETE', `/cartridges/${id}`);
}

/** カートリッジ詳細取得 */
export async function getCartridgeDetail(id: string): Promise<CartridgeDetail> {
	return request<CartridgeDetail>('GET', `/cartridges/${id}`);
}

/** カートリッジ再構築 */
export async function rebuildCartridge(id: string): Promise<CartridgeRebuildResult> {
	return request<CartridgeRebuildResult>('POST', `/cartridges/${id}/rebuild`);
}

/** SSE ストリーミング版インストール (進捗・キャンセル対応)
 *
 * `for await` で SSEProgressEvent を受け取り、`step` / `result` / `error` /
 * `done` の遷移を UI に反映する想定。中止は `signal.abort()` または
 * `cancelCartridgeInstall(sessionId)` を呼ぶ。
 */
export async function* installCartridgeStreaming(
	file: File,
	sessionId: string,
	signal?: AbortSignal
): AsyncGenerator<SSEProgressEvent> {
	const fd = new FormData();
	fd.append('file', file);
	fd.append('session_id', sessionId);
	yield* streamSSEFormData(`${BASE_URL}/cartridges/install/stream`, fd, signal);
}

/** ストリーミング install のキャンセル要求 */
export async function cancelCartridgeInstall(
	sessionId: string
): Promise<{ cancelled: boolean }> {
	const res = await fetch(`${BASE_URL}/cartridges/install/cancel`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ session_id: sessionId })
	});
	if (!res.ok) {
		return { cancelled: false };
	}
	return res.json();
}
