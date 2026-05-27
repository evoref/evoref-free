<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title = '',
		actions,
		children,
		fullHeight = false
	}: {
		title?: string;
		actions?: Snippet;
		children: Snippet;
		fullHeight?: boolean;
	} = $props();
</script>

<div class="page" class:full-height={fullHeight}>
	{#if title || actions}
		<div class="page-header">
			{#if title}
				<h1 class="page-title">{title}</h1>
			{/if}
			{#if actions}
				<div class="page-actions">
					{@render actions()}
				</div>
			{/if}
		</div>
	{/if}
	<div class="page-body" class:full-height={fullHeight}>
		{@render children()}
	</div>
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		padding: 20px 24px;
	}
	.page.full-height {
		height: 100%;
		padding: 0;
	}
	.page-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 16px;
		flex-shrink: 0;
	}
	.full-height > .page-header {
		padding: 20px 24px 0;
	}
	.page-title {
		font-size: 18px;
		font-weight: 500;
		margin: 0;
		color: var(--text-primary);
	}
	.page-actions {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.page-body {
		flex: 1;
		min-height: 0;
	}
</style>
