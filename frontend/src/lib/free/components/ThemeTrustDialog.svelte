<script lang="ts">
	import { t } from '$lib/i18n';
	import DialogShell from './DialogShell.svelte';

	interface Props {
		themeName: string;
		themeAuthor: string;
		themeVersion: string;
		componentCount: number;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { themeName, themeAuthor, themeVersion, componentCount, onConfirm, onCancel }: Props =
		$props();
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

	<p class="confirm-message">{$t('theme_manager.trust_confirm')}</p>

	<div class="dialog-actions">
		<button class="btn btn-cancel" onclick={onCancel}>
			{$t('common.cancel')}
		</button>
		<button class="btn btn-trust" onclick={onConfirm}>
			{$t('theme_manager.trust_accept')}
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
