<script lang="ts">
	/** 履歴の統計 (件数・容量) と、圧縮 / 絞り込み結果の一括削除ボタン */
	import { t } from '$lib/i18n';
	import type { HistoryStats } from '$lib/free/api';
	import Button from '../ui/Button.svelte';

	interface Props {
		stats: HistoryStats;
		/** 現在の絞り込みに合う件数 (一括削除の対象数) */
		matchedTotal: number;
		oncompact: () => void;
		onbulkdelete: () => void;
	}

	let { stats, matchedTotal, oncompact, onbulkdelete }: Props = $props();
</script>

<div class="stats-bar">
	<span class="stats-text">
		{$t('history_page.stats_summary', {
			sessions: stats.total_sessions,
			turns: stats.total_turns,
			size: stats.total_size_mb,
			max: stats.max_storage_mb
		})}
	</span>
	<div class="stats-actions">
		<Button size="sm" onclick={oncompact}>{$t('history_page.compact')}</Button>
		<Button size="sm" variant="danger" onclick={onbulkdelete} disabled={matchedTotal === 0}>
			{$t('history_page.bulk_delete', { count: matchedTotal })}
		</Button>
	</div>
</div>

<style>
	.stats-bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-5);
		font-size: var(--font-size-xs);
		color: var(--text-secondary);
		border-bottom: 0.5px solid var(--border);
	}
	.stats-actions {
		display: flex;
		gap: var(--space-2);
	}
</style>
