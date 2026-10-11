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
		border: 1px solid var(--card-border);
		border-radius: var(--border-radius);
		padding: 14px 16px;
		margin: 0;
		min-width: 0;
	}
	.card-title {
		margin: 0 0 10px;
		font-size: 0.9375rem;
		font-weight: 600;
		color: var(--text-primary);
	}
</style>
