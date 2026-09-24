<script lang="ts">
	import { t } from '$lib/i18n';
	import DialogShell from './DialogShell.svelte';

	interface Props {
		themeId: string;
		themeName: string;
		themeAuthor: string;
		themeVersion: string;
		componentCount: number;
		/** カスタムコードなし (色とレイアウトだけ) で適用する */
		onApplyWithoutCode: () => void;
		onCancel: () => void;
	}

	let {
		themeId,
		themeName,
		themeAuthor,
		themeVersion,
		componentCount,
		onApplyWithoutCode,
		onCancel
	}: Props = $props();

	// 信頼の付与は端末からだけ (docs/c_11 §1)。ここでは手順を案内する。
	let command = $derived(`evoref theme trust ${themeId}`);
</script>

<DialogShell
	ariaLabel={$t('theme_manager.trust_title')}
	onClose={onCancel}
	minWidth="360px"
	maxWidth="480px"
>
	<h3 class="dialog-title">{$t('theme_manager.trust_title')}</h3>

	<div class="theme-info">
		<div class="info-row">
			<span class="label">{$t('theme_manager.trust_theme_name')}</span>
			<span class="value">{themeName}</span>
		</div>
		<div class="info-row">
			<span class="label">{$t('theme_manager.trust_author')}</span>
			<span class="value">{themeAuthor || '—'}</span>
		</div>
		<div class="info-row">
			<span class="label">{$t('theme_manager.trust_version')}</span>
			<span class="value">v{themeVersion}</span>
		</div>
		<div class="info-row">
			<span class="label">{$t('theme_manager.trust_components')}</span>
			<span class="value">{componentCount}</span>
		</div>
	</div>

	<p class="confirm-message">{$t('theme_manager.trust_cli_required')}</p>
	<pre class="trust-command"><code>{command}</code></pre>
	<p class="confirm-message">{$t('theme_manager.trust_cli_restart')}</p>

	<div class="dialog-actions">
		<button class="btn btn-cancel" onclick={onCancel}>
			{$t('common.cancel')}
		</button>
		<button class="btn btn-trust" onclick={onApplyWithoutCode}>
			{$t('theme_manager.trust_apply_without_code')}
		</button>
	</div>
</DialogShell>

<style>
	.dialog-title {
		margin: 0 0 16px;
		font-size: 1.1rem;
		color: var(--text-primary);
	}
	.theme-info {
		background-color: var(--bg-secondary);
		border-radius: var(--border-radius);
		padding: 12px;
		margin-bottom: 16px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.info-row {
		display: flex;
		justify-content: space-between;
		font-size: 0.95rem;
	}
	.label {
		color: var(--text-secondary);
	}
	.value {
		color: var(--text-primary);
		font-weight: 500;
	}
	.confirm-message {
		color: var(--text-primary);
		font-size: 1rem;
		line-height: 1.5;
		margin: 0 0 20px;
	}
	.trust-command {
		background-color: var(--bg-secondary);
		border-radius: var(--border-radius);
		padding: 8px 12px;
		margin: 0 0 12px;
		font-size: 0.9rem;
		user-select: all;
		overflow-x: auto;
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
	}
	.btn {
		padding: 8px 16px;
		border-radius: var(--border-radius);
		font-size: 0.95rem;
		cursor: pointer;
		border: none;
	}
	.btn-cancel {
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}
	.btn-trust {
		background-color: var(--accent);
		color: var(--text-on-accent);
	}
	.btn:hover {
		opacity: 0.9;
	}
</style>
