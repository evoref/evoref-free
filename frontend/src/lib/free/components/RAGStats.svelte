<script lang="ts">
	import { t } from '$lib/i18n';
	import type { DashboardRagStats } from '$lib/free/api';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';

	let { stats }: { stats: DashboardRagStats } = $props();

	let safeChunkCount = $derived(Number.isFinite(stats.chunkCount) ? stats.chunkCount : 0);
	let safeVectorCount = $derived(Number.isFinite(stats.vectorCount) ? stats.vectorCount : 0);
	let safeSourceCount = $derived(Number.isFinite(stats.sourceCount) ? stats.sourceCount : 0);
	let safeIndexSizeMb = $derived(Number.isFinite(stats.indexSizeMb) ? stats.indexSizeMb : 0);

	// 保存済み次元が現在次元と相違する場合のみ併記（完全な不一致警告は dim-warning が担当）
	let dimDisplay = $derived(
		stats.embeddingDimStored != null && stats.embeddingDimStored !== stats.embeddingDimCurrent
			? `${stats.embeddingDimStored} / ${stats.embeddingDimCurrent}`
			: `${stats.embeddingDimCurrent}`
	);

	function formatTs(iso: string | null): string {
		if (!iso) return '—';
		const d = new Date(iso);
		return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString();
	}
</script>

<DashboardCard title={$t('dashboard.rag_stats')}>
	{#if stats.embeddingDimMismatch}
		<div class="dim-warning" role="alert">
			{$t('dashboard.embedding_dim_mismatch', {
				stored: stats.embeddingDimStored ?? '?',
				current: stats.embeddingDimCurrent
			})}
		</div>
	{/if}
	<div class="stats-grid">
		<div class="stat">
			<span class="stat-value">{safeChunkCount.toLocaleString()}</span>
			<span class="stat-label">{$t('dashboard.chunks')}</span>
		</div>
		<div class="stat">
			<span class="stat-value">{safeVectorCount.toLocaleString()}</span>
			<span class="stat-label">{$t('dashboard.vectors')}</span>
		</div>
		<div class="stat">
			<span class="stat-value">{safeSourceCount.toLocaleString()}</span>
			<span class="stat-label">{$t('dashboard.sources')}</span>
		</div>
		<div class="stat">
			<span class="stat-value">{safeIndexSizeMb.toFixed(1)} MB</span>
			<span class="stat-label">{$t('dashboard.index_size')}</span>
		</div>
	</div>

	<dl class="info-section">
		<div class="info-row">
			<dt>{$t('dashboard.chunking_strategy')}</dt>
			<dd>{stats.chunkingStrategy}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.hybrid_search')}</dt>
			<dd>{stats.hybridSearch ? $t('dashboard.enabled') : $t('dashboard.disabled')}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.fusion_method')}</dt>
			<dd>{stats.fusionMethod}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.embedding_dim')}</dt>
			<dd>{dimDisplay}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.embedding_model')}</dt>
			<dd>{stats.embeddingModel ?? '—'}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.embedding_backend')}</dt>
			<dd>{stats.embeddingBackend ?? '—'}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.created_at')}</dt>
			<dd>{formatTs(stats.createdAt)}</dd>
		</div>
		<div class="info-row">
			<dt>{$t('dashboard.last_reindex_at')}</dt>
			<dd>{formatTs(stats.lastReindexAt)}</dd>
		</div>
	</dl>
</DashboardCard>

<style>
	.dim-warning {
		background: var(--warning-bg, #422);
		color: var(--warning-fg, #fdd);
		border: 1px solid var(--warning-border, #f88);
		padding: 8px 12px;
		border-radius: 4px;
		margin-bottom: 12px;
		font-size: 0.85rem;
	}
	.stats-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(72px, 1fr));
		gap: 12px;
	}
	.stat {
		text-align: center;
	}
	.stat-value {
		display: block;
		font-size: 1.4rem;
		font-weight: 700;
		color: var(--accent);
	}
	.stat-label {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}
	.info-section {
		margin: 14px 0 0;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.info-row {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font-size: 0.85rem;
	}
	.info-row dt {
		color: var(--text-secondary);
	}
	.info-row dd {
		margin: 0;
		text-align: right;
		overflow-wrap: anywhere;
	}
</style>
