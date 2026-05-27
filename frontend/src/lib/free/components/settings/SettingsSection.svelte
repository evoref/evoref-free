<script lang="ts">
	import type { Snippet } from 'svelte';
	import { t } from '$lib/i18n';
	import { dirtyTabs, savingTabs, tabErrors, applySection, resetSection } from '$lib/free/stores/settings';

	interface Props {
		tabId: string;
		children: Snippet;
	}

	let { tabId, children }: Props = $props();

	let dirty = $derived($dirtyTabs[tabId] ?? false);
	let saving = $derived($savingTabs[tabId] ?? false);
	let errors = $derived($tabErrors[tabId] ?? []);
</script>

<div class="section">
	<div class="section-body">
		{@render children()}
	</div>

	{#if errors.length > 0}
		<div class="section-errors">
			{#each errors as err}
				<p class="error-line">{err}</p>
			{/each}
		</div>
	{/if}

	<div class="section-footer">
		<button
			type="button"
			class="btn-reset"
			disabled={!dirty || saving}
			onclick={() => resetSection(tabId)}
		>
			{$t('settings.reset')}
		</button>
		<button
			type="button"
			class="btn-apply"
			disabled={!dirty || saving}
			onclick={() => applySection(tabId)}
		>
			{#if saving}
				{$t('settings.applying')}
			{:else}
				{$t('settings.apply')}
			{/if}
		</button>
	</div>
</div>

<style>
	.section {
		display: flex;
		flex-direction: column;
		height: 100%;
	}

	.section-body {
		flex: 1;
		overflow-y: auto;
		padding: 16px 24px;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
		align-content: start;
	}

	.section-errors {
		padding: 8px 24px;
		background: color-mix(in srgb, var(--color-error) 10%, transparent);
		border-top: 0.5px solid var(--color-error);
	}

	.error-line {
		margin: 2px 0;
		font-size: 12px;
		color: var(--color-error);
	}

	.section-footer {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
		padding: 12px 24px;
		border-top: 0.5px solid var(--border);
		background: var(--bg-secondary);
	}

	.btn-apply {
		padding: 8px 20px;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
		transition: opacity 0.15s;
	}

	.btn-apply:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn-reset {
		padding: 8px 16px;
		background: none;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
	}

	.btn-reset:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	@media (max-width: 900px) {
		.section-body {
			grid-template-columns: 1fr;
		}
	}

	@media (max-width: 767px) {
		.section-body {
			padding: 12px 16px;
		}
		.section-footer {
			padding: 10px 16px;
		}
	}
</style>
