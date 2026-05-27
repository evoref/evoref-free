<script lang="ts">
	import type { Snippet } from 'svelte';

	let { title, children }: { title: string; children: Snippet } = $props();
</script>

<fieldset class="card">
	<legend class="card-legend">{title}</legend>
	{@render children()}
</fieldset>

<style>
	.card {
		background: var(--card-bg);
		border: 0.5px solid var(--card-border);
		border-radius: 10px;
		padding: 14px 16px;
		margin: 0;
		backdrop-filter: blur(6px);
		-webkit-backdrop-filter: blur(6px);
	}
	.card-legend {
		font-size: 14.5px;
		font-weight: 600;
		color: var(--text-secondary);
		padding: 0 6px;
	}
</style>
