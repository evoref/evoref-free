<script lang="ts">
	/**
	 * 埋め込みモデル切替後の再構築ボタン (RAG reindex + SemMem fact reembed)
	 *
	 * RAG/カートリッジ/メモリのベクトル (reindexVectors) と、URL/コマンドリコール用の
	 * SemMem fact 埋め込み (reembedFacts) は別ストアのため、埋め込みモデル切替後は
	 * 両方の再構築が必要。このボタンは1回の preview→confirm→run で両方をまとめて実行する。
	 *
	 * フロー: dry-run で両方の対象件数をプレビュー → ユーザー確認 → 本実行。
	 * 実行は reindex → reembed の順に逐次実行し、それぞれの成否は独立に扱う
	 * (互いに独立したストアのため、片方が失敗してももう片方は実行する)。
	 */
	import { t } from '$lib/i18n';
	import {
		reindexVectors,
		reembedFacts,
		ApiError,
		type ReindexResponse,
		type ReembedFactsResponse
	} from '$lib/free/api';

	type Props = {
		/** 両方の実行試行後に config / 一覧を再取得させるコールバック */
		onRebuilt?: () => void;
	};

	let { onRebuilt }: Props = $props();

	let busy = $state(false);
	let reindexPreview = $state<ReindexResponse | null>(null);
	let reembedPreview = $state<ReembedFactsResponse | null>(null);
	let reindexResult = $state<ReindexResponse | null>(null);
	let reembedResult = $state<ReembedFactsResponse | null>(null);
	let reindexError = $state<string | null>(null);
	let reembedError = $state<string | null>(null);

	function toError(e: unknown): string {
		if (e instanceof ApiError) return e.message;
		if (e instanceof Error) return e.message;
		return String(e);
	}

	function resetState() {
		reindexPreview = null;
		reembedPreview = null;
		reindexResult = null;
		reembedResult = null;
		reindexError = null;
		reembedError = null;
	}

	async function doPreview() {
		if (busy) return;
		busy = true;
		resetState();
		try {
			const [reindex, reembed] = await Promise.all([
				reindexVectors({ dry_run: true }),
				reembedFacts({ dry_run: true })
			]);
			reindexPreview = reindex;
			reembedPreview = reembed;
		} catch (e) {
			reindexError = toError(e);
		} finally {
			busy = false;
		}
	}

	async function doExecute() {
		if (busy) return;
		busy = true;
		reindexError = null;
		reembedError = null;
		try {
			reindexResult = await reindexVectors({ dry_run: false });
		} catch (e) {
			reindexError = toError(e);
		}
		try {
			reembedResult = await reembedFacts({ dry_run: false });
		} catch (e) {
			reembedError = toError(e);
		}
		reindexPreview = null;
		reembedPreview = null;
		busy = false;
		onRebuilt?.();
	}
</script>

<div class="rebuild">
	{#if !reindexPreview && !reembedPreview && !reindexResult && !reembedResult}
		<button type="button" class="preview-btn" disabled={busy} onclick={doPreview}>
			{busy ? $t('settings.reindex.checking') : $t('settings.reindex.preview')}
		</button>
	{/if}

	{#if reindexPreview || reembedPreview}
		<div class="result info">
			{#if reindexPreview}
				<div>
					{$t('settings.reindex.targets', {
						rag: reindexPreview.rag_chunks,
						cart: reindexPreview.cartridge_chunks,
						mem: reindexPreview.memory_notes ?? 0
					})}
				</div>
			{/if}
			{#if reembedPreview}
				<div>{$t('settings.reembed.targets', { n: reembedPreview.fact_count })}</div>
			{/if}
			<div class="actions">
				<button type="button" class="run-btn" disabled={busy} onclick={doExecute}>
					{busy ? $t('settings.reindex.running') : $t('settings.reindex.run')}
				</button>
				<button
					type="button"
					class="cancel-btn"
					disabled={busy}
					onclick={() => {
						reindexPreview = null;
						reembedPreview = null;
					}}
				>
					{$t('settings.reindex.cancel')}
				</button>
			</div>
		</div>
	{/if}

	{#if reindexResult}
		<div
			class="result"
			class:success={!reindexResult.cartridges_failed?.length}
			class:error={!!reindexResult.cartridges_failed?.length}
		>
			{$t('settings.reindex.done', {
				rag: reindexResult.rag_chunks,
				cart: reindexResult.cartridge_chunks,
				mem: reindexResult.memory_notes_reset ?? 0,
				sec: (reindexResult.elapsed_sec ?? 0).toFixed(1)
			})}
			{#if reindexResult.cartridges_failed?.length}
				<span class="badge warn">
					{$t('settings.reindex.cart_failed', {
						ids: reindexResult.cartridges_failed.join(', ')
					})}
				</span>
			{/if}
			{#if reindexResult.embedding_dim_mismatch}
				<span class="badge warn">{$t('settings.reindex.still_mismatch')}</span>
			{/if}
		</div>
	{/if}
	{#if reindexError}
		<div class="result error">{reindexError}</div>
	{/if}

	{#if reembedResult}
		<div class="result success">
			{$t('settings.reembed.done', {
				n: reembedResult.reembedded ?? 0,
				sec: (reembedResult.elapsed_sec ?? 0).toFixed(1)
			})}
		</div>
	{/if}
	{#if reembedResult?.restart_required}
		<div class="result warn">{$t('settings.reembed.restart_note')}</div>
	{/if}
	{#if reembedError}
		<div class="result error">{reembedError}</div>
	{/if}
</div>

<style>
	.rebuild {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.4rem;
	}
	button {
		padding: 0.4rem 0.8rem;
		border-radius: 4px;
		border: 1px solid var(--input-border);
		background: var(--bg-secondary);
		color: var(--text-primary);
		cursor: pointer;
		font-size: 0.9rem;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	button.run-btn,
	button.preview-btn {
		background: var(--accent);
		color: var(--text-on-accent);
		border-color: var(--accent);
	}
	.result {
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.85rem;
	}
	.result.info {
		background: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}
	.result.success {
		background: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--color-success);
	}
	.result.warn {
		background: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--color-warning);
	}
	.result.error {
		background: var(--bg-secondary);
		color: var(--color-error);
		border: 1px solid var(--color-error);
	}
	.badge {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		font-size: 0.75rem;
	}
	.badge.warn {
		background: var(--bg-secondary);
		color: var(--text-primary);
	}
</style>
