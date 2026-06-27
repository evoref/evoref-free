<script lang="ts">
	/**
	 * ベクトルインデックス再構築ボタン
	 *
	 * 埋め込みモデル切替後に、現在の埋め込みモデルで全ベクトル (RAG / カートリッジ /
	 * メモリ) を再構築する。`cartridge` を渡すとそのカートリッジのみ再構築する。
	 *
	 * フロー: まず dry-run で対象件数をプレビュー → ユーザー確認 → 本実行。
	 * reindex はブロッキング (進捗 SSE 無し) なので、件数を先に見せて安心させる。
	 */
	import { t } from '$lib/i18n';
	import { reindexVectors, ApiError, type ReindexResponse } from '$lib/free/api';

	type Props = {
		/** 指定するとそのカートリッジのみ再構築 (省略時は全体) */
		cartridge?: string;
		/** 再構築成功後に config / 一覧を再取得させるコールバック */
		onReindexed?: () => void;
	};

	let { cartridge, onReindexed }: Props = $props();

	let busy = $state(false);
	let preview = $state<ReindexResponse | null>(null);
	let result = $state<ReindexResponse | null>(null);
	let error = $state<string | null>(null);

	function toError(e: unknown): string {
		if (e instanceof ApiError) return e.message;
		if (e instanceof Error) return e.message;
		return String(e);
	}

	async function doPreview() {
		if (busy) return;
		busy = true;
		error = null;
		result = null;
		preview = null;
		try {
			preview = await reindexVectors({ dry_run: true, cartridge });
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function doExecute() {
		if (busy) return;
		busy = true;
		error = null;
		try {
			result = await reindexVectors({ dry_run: false, cartridge });
			preview = null;
			onReindexed?.();
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="reindex" class:compact={!!cartridge}>
	{#if !preview && !result}
		<button type="button" class="preview-btn" disabled={busy} onclick={doPreview}>
			{busy ? $t('settings.reindex.checking') : $t('settings.reindex.preview')}
		</button>
	{/if}

	{#if preview}
		<div class="result info">
			<div>
				{$t('settings.reindex.targets', {
					rag: preview.rag_chunks,
					cart: preview.cartridge_chunks,
					mem: preview.memory_notes ?? 0
				})}
			</div>
			<div class="actions">
				<button type="button" class="run-btn" disabled={busy} onclick={doExecute}>
					{busy ? $t('settings.reindex.running') : $t('settings.reindex.run')}
				</button>
				<button
					type="button"
					class="cancel-btn"
					disabled={busy}
					onclick={() => (preview = null)}
				>
					{$t('settings.reindex.cancel')}
				</button>
			</div>
		</div>
	{/if}

	{#if result}
		<div class="result success">
			{$t('settings.reindex.done', {
				rag: result.rag_chunks,
				cart: result.cartridge_chunks,
				mem: result.memory_notes_reset ?? 0,
				sec: (result.elapsed_sec ?? 0).toFixed(1)
			})}
			{#if result.embedding_dim_mismatch}
				<span class="badge warn">{$t('settings.reindex.still_mismatch')}</span>
			{/if}
		</div>
	{/if}

	{#if error}
		<div class="result error">{error}</div>
	{/if}
</div>

<style>
	.reindex {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-top: 0.5rem;
	}
	.reindex.compact {
		margin-top: 0;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.4rem;
	}
	button {
		padding: 0.4rem 0.8rem;
		border-radius: 4px;
		border: 1px solid var(--border-color, #999);
		background: var(--bg-secondary, #f5f5f5);
		color: var(--text-primary, #222);
		cursor: pointer;
		font-size: 0.9rem;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	button.run-btn,
	button.preview-btn {
		background: var(--accent-bg, #2563eb);
		color: var(--accent-text, #fff);
		border-color: var(--accent-bg, #2563eb);
	}
	.result {
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.85rem;
	}
	.result.info {
		background: var(--bg-info-soft, #eff6ff);
		color: var(--text-info, #1e40af);
		border: 1px solid var(--border-info, #93c5fd);
	}
	.result.success {
		background: var(--bg-success-soft, #ecfdf5);
		color: var(--text-success, #065f46);
		border: 1px solid var(--border-success, #6ee7b7);
	}
	.result.error {
		background: var(--bg-error-soft, #fef2f2);
		color: var(--text-error, #991b1b);
		border: 1px solid var(--border-error, #fca5a5);
	}
	.badge {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		font-size: 0.75rem;
	}
	.badge.warn {
		background: var(--bg-warning-soft, #fef3c7);
		color: var(--text-warning, #92400e);
	}
</style>
