/** カートリッジ API (Free)
 *
 * インストール / 一覧 / ロード / アンロード / 削除 / 再構築
 * の Free エディション機能のみを提供する。カートリッジ作成
 * (POST /api/pro/cartridges/create) は Pro 専用機能のため
 * `$lib/pro/api/cartridge.ts` 側に分離している。
 */

import {
	BASE_URL,
	cancelStreamingOperation,
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
	/** 運ぶセクション (`docs` / `templates` / `language`)。`docs` が無ければ索引の操作は出さない */
	provides?: string[];
}

/** 言語パック (c_16 §4.5.3) 1 エントリの有効/無効と理由 */
export interface LanguagePackEntry {
	id: string;
	grammar: string;
	extensions: string[];
	enabled: boolean;
	/** 無効時の理由 (バックエンドの技術的な英語メッセージ。i18n されない) */
	reason: string;
}

/** カートリッジ詳細情報（全フィールド）
 *
 * `priority` は廃止 (docs/c_16_evidence_store.md §8)。順位付けの
 * `store_prior` へ置き換わり、PC 固有の上書きはバックエンドの
 * `corpus/manifest.json` が持つので UI からは触らない。
 */
export interface CartridgeDetail {
	id: string;
	name: string;
	version: string;
	/** 現在有効なパッケージ版 (`version` と同じ値) */
	active_version: string;
	author: string;
	license: string;
	description: string;
	tags: string[];
	language: string;
	chunks: number;
	doc_count: number;
	size_mb: number;
	status: string;
	installed_at: string;
	compatibility: string;
	/** docs/ の内容ダイジェスト (再現性の鍵) */
	content_digest: string;
	embedding_model_id: string;
	embedding_dim: number;
	/** パッケージの所有と生まれ方 ("package" | "project_map"、c_16 §4.3) */
	kind: string;
	/** 運ぶセクション名の列 ("docs" / "templates" / "language") */
	provides: string[];
	/** "language/" セクションの各エントリの有効/無効と理由 (c_16 §4.5.3) */
	language_pack: LanguagePackEntry[];
}

/** カートリッジ再構築結果 */
export interface CartridgeRebuildResult {
	id: string;
	name: string;
	version: string;
	chunks: number;
	status: string;
	size_mb: number;
	embedding_model_id: string;
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
	return cancelStreamingOperation('/cartridges/install/cancel', sessionId);
}
