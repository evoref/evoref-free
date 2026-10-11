<script lang="ts" module>
	/** ISO 時刻をローカルの日時に (無い・読めなければ —) */
	export function formatTimestamp(iso: string | null): string {
		if (!iso) return '—';
		const d = new Date(iso);
		return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString();
	}
</script>

<script lang="ts">
	/**
	 * ダッシュボードの文書カード (一般向け): 取り込んだ文書の数・チャンク数・索引のサイズ・最終更新。
	 * 埋め込みや分割の内訳は RagDetail (Pro) が出す。
	 */
	import { t } from '$lib/i18n';
	import type { DashboardRagStats } from '$lib/free/api';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';

	let { stats }: { stats: DashboardRagStats } = $props();

	function safe(n: number): number {
		return Number.isFinite(n) ? n : 0;
	}

	let empty = $derived(safe(stats.sourceCount) === 0 && safe(stats.chunkCount) === 0);
</script>

<DashboardCard title={$t('dashboard.documents_title')}>
	{#if empty}
		<p class="hint-text">{$t('dashboard.documents_empty')}</p>
	{:else}
		<StatusRow label={$t('dashboard.documents_sources')} value={safe(stats.sourceCount).toLocaleString()} />
		<StatusRow label={$t('dashboard.documents_chunks')} value={safe(stats.chunkCount).toLocaleString()} />
		<StatusRow label={$t('dashboard.index_size')} value="{safe(stats.indexSizeMb).toFixed(1)} MB" />
		<StatusRow
			label={$t('dashboard.documents_updated')}
			value={formatTimestamp(stats.lastReindexAt ?? stats.createdAt)}
		/>
	{/if}
</DashboardCard>

<style>
	.hint-text {
		margin: 0;
		font-size: 0.8125rem;
		color: var(--text-secondary);
	}
</style>
