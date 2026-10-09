<script lang="ts">
	/** 履歴の統計 (件数・容量) と、圧縮 / 絞り込み結果の一括削除ボタン */
	import { t } from '$lib/i18n';
	import type { HistoryStats } from '$lib/free/api';

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
		<button class="stats-btn" onclick={oncompact}>{$t('history_page.compact')}</button>
		<button class="stats-btn danger" onclick={onbulkdelete} disabled={matchedTotal === 0}>
			{$t('history_page.bulk_delete', { count: matchedTotal })}
		</button>
	</div>
</div>

<style>
	.stats-bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		padding: 6px 16px;
		font-size: 12px;
		color: var(--text-secondary);
		border-bottom: 0.5px solid var(--border);
	}
	.stats-actions {
		display: flex;
		gap: 8px;
	}
	.stats-btn {
		padding: 4px 10px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 12px;
		cursor: pointer;
	}
	.stats-btn.danger {
		color: var(--error, #c0392b);
	}
	.stats-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
