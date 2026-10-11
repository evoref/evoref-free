<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title = '',
		description,
		actions,
		children,
		fullHeight = false
	}: {
		title?: string;
		/** 見出しの下に置く 1〜2 行の説明 (この画面が何のためにあるか) */
		description?: Snippet;
		actions?: Snippet;
		children: Snippet;
		fullHeight?: boolean;
	} = $props();
</script>

<div class="page" class:full-height={fullHeight}>
	{#if title || actions}
		<div class="page-header">
			<div class="page-heading">
				{#if title}
					<h1 class="page-title">{title}</h1>
				{/if}
				{#if description}
					<p class="page-description">{@render description()}</p>
				{/if}
			</div>
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
		padding: var(--space-5);
	}
	.page.full-height {
		height: 100%;
		padding: 0;
	}
	.page-header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: var(--space-3);
		margin-bottom: var(--space-4);
		flex-shrink: 0;
	}
	.full-height > .page-header {
		padding: var(--space-5) var(--space-5) 0;
	}
	.page-heading {
		min-width: 0;
		flex: 1 1 280px;
	}
	.page-title {
		font-size: var(--font-size-xl);
		font-weight: 600;
		margin: 0;
		color: var(--text-primary);
	}
	.page-description {
		margin: var(--space-1) 0 0;
		color: var(--text-secondary);
		font-size: var(--font-size-sm);
		line-height: 1.6;
	}
	.page-actions {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--space-2);
	}
	.page-body {
		flex: 1;
		min-height: 0;
	}
</style>
