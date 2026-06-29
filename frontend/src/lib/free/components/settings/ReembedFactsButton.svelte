<script lang="ts">
	/**
	 * SemMem fact 再 embed ボタン
	 *
	 * 埋め込みモデル切替後、URL/コマンドリコール用の SemMem fact ベクトルを現在の
	 * 埋め込みモデルで再構築する。RAG の ReindexButton は SemMem fact を対象外のため、
	 * 切替後はこのボタンも併せて押す。in-process 実行で稼働中の store を直接更新する
	 * ため **再起動不要**。
	 *
	 * フロー: dry-run で対象 fact 数をプレビュー → ユーザー確認 → 本実行。
	 */
	import { t } from '$lib/i18n';
	import { reembedFacts, ApiError, type ReembedFactsResponse } from '$lib/free/api';

	type Props = {
		/** 再構築成功後に config / 一覧を再取得させるコールバック */
		onReembedded?: () => void;
	};

	let { onReembedded }: Props = $props();

	let busy = $state(false);
	let preview = $state<ReembedFactsResponse | null>(null);
	let result = $state<ReembedFactsResponse | null>(null);
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
			preview = await reembedFacts({ dry_run: true });
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
			result = await reembedFacts({ dry_run: false });
			preview = null;
			onReembedded?.();
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="reembed">
	{#if !preview && !result}
		<button type="button" class="preview-btn" disabled={busy} onclick={doPreview}>
			{busy ? $t('settings.reembed.checking') : $t('settings.reembed.preview')}
		</button>
	{/if}

	{#if preview}
		<div class="result info">
			<div>{$t('settings.reembed.targets', { n: preview.fact_count })}</div>
			<div class="actions">
				<button type="button" class="run-btn" disabled={busy} onclick={doExecute}>
					{busy ? $t('settings.reembed.running') : $t('settings.reembed.run')}
				</button>
				<button
					type="button"
					class="cancel-btn"
					disabled={busy}
					onclick={() => (preview = null)}
				>
					{$t('settings.reembed.cancel')}
				</button>
			</div>
		</div>
	{/if}

	{#if result}
		<div class="result success">
			{$t('settings.reembed.done', {
				n: result.reembedded ?? 0,
				sec: (result.elapsed_sec ?? 0).toFixed(1)
			})}
		</div>
	{/if}

	{#if error}
		<div class="result error">{error}</div>
	{/if}
</div>

<style>
	.reembed {
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
</style>
