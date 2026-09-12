<script lang="ts">
	import { t } from '$lib/i18n';
	import type { SourcesInfo, SourceItem } from '$lib/free/api';

	let { sources }: { sources: SourcesInfo } = $props();
	let expanded = $state(false);

	function label(item: SourceItem): string {
		if (item.store !== 'corpus') return $t('sources.memory');
		const doc = item.doc_id || item.package_name || item.package_id;
		return doc ? `${item.package_name || item.package_id} / ${doc}` : $t('sources.corpus');
	}
</script>

{#if sources.items.length > 0}
	<div class="sources-panel">
		<button class="toggle" onclick={() => (expanded = !expanded)}>
			<span class="arrow" class:open={expanded}>&#9654;</span>
			{$t('sources.title')} ({sources.items.length})
		</button>
		{#if expanded}
			<ol class="items">
				{#each sources.items as item, i (item.id + i)}
					<li class="item">
						<div class="head">
							<span class="index">#{i + 1}</span>
							<span class="doc" title={item.id}>{label(item)}</span>
							{#if item.store === 'corpus'}
								<span class="heading">{item.heading || $t('sources.untitled')}</span>
							{/if}
						</div>
						<div class="preview">{item.preview}</div>
					</li>
				{/each}
			</ol>
		{/if}
	</div>
{/if}

<style>
	.sources-panel {
		margin-top: 4px;
		padding: 0 14px 4px;
		font-size: 0.78rem;
		color: var(--text-secondary, #666);
	}
	.toggle {
		background: none;
		border: none;
		color: inherit;
		cursor: pointer;
		padding: 2px 0;
		font: inherit;
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.arrow {
		display: inline-block;
		font-size: 0.6rem;
		transition: transform 0.15s;
	}
	.arrow.open {
		transform: rotate(90deg);
	}
	.items {
		margin: 4px 0 0;
		padding-left: 0;
		list-style: none;
	}
	.item {
		padding: 4px 0;
		border-top: 1px solid var(--border-color, #e0e0e0);
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: baseline;
	}
	.index {
		opacity: 0.7;
	}
	.doc {
		font-weight: 600;
	}
	.heading {
		opacity: 0.85;
	}
	.preview {
		margin-top: 2px;
		white-space: pre-wrap;
		word-break: break-word;
		opacity: 0.85;
	}
</style>
