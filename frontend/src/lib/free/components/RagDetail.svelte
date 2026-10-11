<script lang="ts">
	/**
	 * 検索 (RAG) の詳細 (Pro のダッシュボードの「詳細」): ベクトル数・分割戦略・埋め込みの
	 * 次元/モデル/バックエンド・作成と再構築の日時。件数とサイズは DocumentSummary が出す。
	 */
	import { t } from '$lib/i18n';
	import type { DashboardRagStats } from '$lib/free/api';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';
	import { formatTimestamp } from '$lib/free/components/DocumentSummary.svelte';

	let { stats }: { stats: DashboardRagStats } = $props();

	let safeVectorCount = $derived(Number.isFinite(stats.vectorCount) ? stats.vectorCount : 0);

	// 保存済み次元が現在次元と相違する場合のみ併記 (不一致の警告は全体の状態の欄が出す)
	let dimDisplay = $derived(
		stats.embeddingDimStored != null && stats.embeddingDimStored !== stats.embeddingDimCurrent
			? `${stats.embeddingDimStored} / ${stats.embeddingDimCurrent}`
			: `${stats.embeddingDimCurrent}`
	);
</script>

<DashboardCard title={$t('dashboard.rag_stats')}>
	<StatusRow label={$t('dashboard.vectors')} value={safeVectorCount.toLocaleString()} />
	<StatusRow label={$t('dashboard.chunking_strategy')} value={stats.chunkingStrategy} />
	<StatusRow label={$t('dashboard.embedding_dim')} value={dimDisplay} />
	<StatusRow label={$t('dashboard.embedding_model')} value={stats.embeddingModel ?? '—'} />
	<StatusRow label={$t('dashboard.embedding_backend')} value={stats.embeddingBackend ?? '—'} />
	<StatusRow label={$t('dashboard.created_at')} value={formatTimestamp(stats.createdAt)} />
	<StatusRow label={$t('dashboard.last_reindex_at')} value={formatTimestamp(stats.lastReindexAt)} />
</DashboardCard>
