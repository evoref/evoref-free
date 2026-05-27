<script lang="ts">
	import type { Snippet } from 'svelte';
	import { t } from '$lib/i18n';

	interface Props {
		label: string;
		description?: string;
		fullWidth?: boolean;
		columns?: 1 | 2;
		children: Snippet;
	}

	let { label, description = '', fullWidth = false, columns = 1, children }: Props = $props();
</script>

<fieldset class="field-group" class:full-width={fullWidth}>
	<legend class="group-legend">{$t(label)}</legend>
	{#if description}
		<p class="group-desc">{$t(description)}</p>
	{/if}
	<div class="group-content" class:two-columns={columns === 2}>
		{@render children()}
	</div>
</fieldset>

<style>
	.field-group {
		background: var(--card-bg);
		border: 0.5px solid var(--card-border, var(--border));
		border-radius: 10px;
		padding: 14px 16px;
		margin: 0;
		backdrop-filter: blur(6px);
		-webkit-backdrop-filter: blur(6px);
	}
	.full-width {
		grid-column: 1 / -1;
	}
	.group-legend {
		font-size: 14.5px;
		font-weight: 600;
		color: var(--text-secondary);
		padding: 0 6px;
	}
	.group-desc {
		font-size: 11px;
		color: var(--text-muted);
		margin: 0 0 8px 0;
	}
	.group-content {
		display: flex;
		flex-direction: column;
	}
	.group-content.two-columns {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0 24px;
	}
	@media (max-width: 900px) {
		.group-content.two-columns {
			grid-template-columns: 1fr;
		}
	}
</style>
