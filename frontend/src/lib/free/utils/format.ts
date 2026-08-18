/**
 * 共通フォーマットヘルパ
 *
 * 重複していた `formatDate` / `formatSize` / `formatTime` / `formatDateTime`
 * を集約する
 *
 * 各ヘルパは純粋関数。Svelte コンポーネントからは
 *   import { formatSize, formatDate } from '$lib/free/utils/format';
 * のように使用する。
 */

/**
 * MB 単位のサイズを人間可読文字列に変換する。
 * 1 MB 以上は "1.5 MB"、1 MB 未満は "512 KB" のように表示する。
 *
 * 元実装: CartridgeManager.svelte / CartridgeDetailDialog.svelte
 */
export function formatSize(mb: number): string {
	if (mb >= 1) return `${mb.toFixed(1)} MB`;
	return `${(mb * 1024).toFixed(0)} KB`;
}

/**
 * バイト数を人間可読文字列に変換する ("36.4 MB" / "512 KB" / "800 B")。
 * null / undefined / 負値は '-' を返す (サイズ不明を数値として見せない)。
 *
 * `formatSize` は MB 入力を前提としており、API がバイトで返す値
 * (LoRA スナップショットのファイルサイズ等) には使えないため別に持つ。
 */
export function formatBytes(bytes: number | null | undefined): string {
	if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) {
		return '-';
	}
	const mb = bytes / (1024 * 1024);
	if (mb >= 1) return `${mb.toFixed(1)} MB`;
	const kb = bytes / 1024;
	if (kb >= 1) return `${kb.toFixed(0)} KB`;
	return `${bytes} B`;
}

/**
 * ISO 文字列を日時文字列 (`toLocaleString()`) に変換する。
 * 空文字列やパース失敗時は '-' を返す。
 */
export function formatDate(iso: string | null | undefined): string {
	if (!iso) return '-';
	try {
		return new Date(iso).toLocaleString();
	} catch {
		return iso;
	}
}

/**
 * タイムスタンプを時:分形式 (例 "14:32") に変換する。
 * `value` は ISO 文字列または ms epoch 数値のどちらでも受け付ける。
 *
 * 元実装: MessageBubble.svelte (number) / utils/history.ts (string)
 */
export function formatTime(value: string | number): string {
	const d = new Date(value);
	return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * ISO 文字列を「月 日 時:分」形式に変換する (例 "Apr 7, 14:32")。
 *
 * 元実装: utils/history.ts
 */
export function formatDateTime(iso: string): string {
	const d = new Date(iso);
	return d.toLocaleString([], {
		month: 'short',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit',
	});
}
