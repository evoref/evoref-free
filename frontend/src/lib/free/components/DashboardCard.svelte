<script lang="ts">
	import type { Snippet } from 'svelte';

	let { title, children }: { title: string; children: Snippet } = $props();
</script>

<section class="card">
	<h2 class="card-title">{title}</h2>
	{@render children()}
</section>

<style>
	.card {
		background: var(--card-bg);
		border: 1px solid color-mix(in srgb, var(--card-border) 55%, transparent);
		border-radius: var(--radius-xl);
		box-shadow: var(--elev-1);
		padding: var(--space-4) var(--space-5);
		margin: 0;
		min-width: 0;
	}
	.card-title {
		margin: 0 0 var(--space-3);
		font-size: var(--font-size-sm);
		font-weight: 600;
		letter-spacing: 0.01em;
		color: var(--text-primary);
	}
</style>
