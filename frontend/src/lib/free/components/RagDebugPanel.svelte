<script lang="ts">
	import { t } from '$lib/i18n';
	import type { RagDebugInfo } from '$lib/free/api';

	let { ragDebug }: { ragDebug: RagDebugInfo } = $props();
	let expanded = $state(false);
</script>

{#if ragDebug.chunks.length > 0}
	<div class="rag-debug-panel">
		<button class="toggle" onclick={() => (expanded = !expanded)}>
			<span class="arrow" class:open={expanded}>&#9654;</span>
			{$t('rag_debug.title')} ({ragDebug.chunks.length})
			<span class="search-time">— {$t('rag_debug.search_time', { ms: ragDebug.search_time_ms.toFixed(1) })}</span>
		</button>
		{#if expanded}
			<div class="chunks-list">
				{#each ragDebug.chunks as chunk, i}
					<div class="chunk-item">
						<div class="chunk-header">
							<span class="chunk-index">#{i + 1}</span>
							<span class="chunk-source" title={chunk.source}>{chunk.source}</span>
							<span class="chunk-score">
								<span class="score-bar" style="width: {Math.round(chunk.score * 100)}%"></span>
								<span class="score-text">{(chunk.score * 100).toFixed(1)}%</span>
							</span>
						</div>
						<div class="chunk-preview">{chunk.preview}{chunk.preview.length >= 100 ? '...' : ''}</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<style>
	.rag-debug-panel {
		margin-top: 4px;
		padding: 0 14px 4px;
		font-size: 0.78rem;
	}
	.toggle {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		padding: 2px 0;
		font-size: 0.78rem;
		display: flex;
		align-items: center;
		gap: 4px;
		opacity: 0.6;
	}
	.toggle:hover {
		opacity: 0.85;
		color: var(--accent);
	}
	.arrow {
		display: inline-block;
		font-size: 0.45rem;
		transition: transform 0.2s;
	}
	.arrow.open {
		transform: rotate(90deg);
	}
	.search-time {
		opacity: 0.7;
		font-weight: normal;
	}
	.chunks-list {
		padding: 4px 0 0 16px;
	}
	.chunk-item {
		padding: 4px 0;
		border-bottom: 1px solid color-mix(in srgb, var(--border) 40%, transparent);
	}
	.chunk-item:last-child {
		border-bottom: none;
	}
	.chunk-header {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.chunk-index {
		color: var(--text-secondary);
		opacity: 0.5;
		font-size: 0.72rem;
		flex-shrink: 0;
	}
	.chunk-source {
		font-weight: 600;
		color: var(--accent);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 200px;
	}
	.chunk-score {
		margin-left: auto;
		display: flex;
		align-items: center;
		gap: 4px;
		flex-shrink: 0;
	}
	.score-bar {
		display: inline-block;
		height: 4px;
		min-width: 4px;
		max-width: 60px;
		background-color: var(--accent);
		border-radius: 2px;
		opacity: 0.7;
	}
	.score-text {
		font-size: 0.72rem;
		opacity: 0.6;
		white-space: nowrap;
		font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
	}
	.chunk-preview {
		margin-top: 2px;
		color: var(--text-secondary);
		opacity: 0.55;
		font-size: 0.72rem;
		line-height: 1.4;
		overflow: hidden;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
	}
</style>
