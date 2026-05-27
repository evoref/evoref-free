<script lang="ts">
	import { t } from '$lib/i18n';
	import { formatDuration } from '$lib/free/utils/history';
	import { formatDateTime } from '$lib/free/utils/format';
	import type { SessionDetailData } from '$lib/free/types/history';
	import TurnItem from './TurnItem.svelte';

	interface Props {
		selectedId: string | null;
		detail: SessionDetailData | null;
		detailLoading: boolean;
		detailError: string;
		onresume: () => void;
	}

	let { selectedId, detail, detailLoading, detailError, onresume }: Props = $props();
</script>

<div class="detail-pane">
	{#if !selectedId}
		<div class="center-msg">{$t('history_page.select_session')}</div>
	{:else if detailLoading}
		<div class="center-msg">{$t('history_page.detail_loading')}</div>
	{:else if detailError}
		<div class="center-msg error">{$t('history_page.detail_load_failed')}</div>
	{:else if detail}
		<div class="detail-header">
			<div class="detail-meta">
				<span class="session-mode mode-{detail.mode}">{detail.mode}</span>
				<span class="detail-info">{$t('history_page.started_at')}: {formatDateTime(detail.started_at)}</span>
				<span class="detail-info">{$t('history_page.turns', { count: detail.turn_count })}</span>
				<span class="detail-info">{$t('history_page.duration', { minutes: formatDuration(detail.duration_sec) })}</span>
				{#if detail.base_model}
					<span class="detail-info">{$t('history_page.model')}: {detail.base_model}</span>
				{/if}
			</div>
			<div class="detail-actions">
				{#if detail.topics.length > 0}
					<div class="detail-topics">
						{#each detail.topics as topic}
							<span class="topic-tag">{topic}</span>
						{/each}
					</div>
				{/if}
				<button class="resume-btn" onclick={onresume}>{$t('history_page.resume')}</button>
			</div>
		</div>

		<div class="turn-list">
			{#if detail.turns.length === 0}
				<div class="center-msg">{$t('history_page.no_turns')}</div>
			{:else}
				{#each detail.turns as turn}
					<TurnItem {turn} instanceName={detail.instance_name} />
				{/each}
			{/if}
		</div>
	{/if}
</div>

<style>
	.detail-pane {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: var(--bg-primary);
	}
	.center-msg {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 40px 16px;
		color: var(--text-secondary);
		font-size: 14px;
		height: 100%;
	}
	.center-msg.error {
		color: var(--error);
	}
	.detail-header {
		padding: 12px 16px;
		border-bottom: 0.5px solid var(--sidebar-border);
		flex-shrink: 0;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
	}
	.detail-meta {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	.session-mode {
		font-size: 10px;
		padding: 1px 5px;
		border-radius: 3px;
		font-weight: 500;
		flex-shrink: 0;
	}
	.session-mode.mode-chat {
		background: color-mix(in srgb, var(--accent) 15%, transparent);
		color: var(--accent);
	}
	.session-mode.mode-coding {
		background: color-mix(in srgb, var(--color-success) 15%, transparent);
		color: var(--color-success);
	}
	.detail-info {
		font-size: 12px;
		color: var(--text-secondary);
	}
	.detail-actions {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.detail-topics {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}
	.topic-tag {
		font-size: 11px;
		padding: 1px 6px;
		border-radius: 3px;
		background: color-mix(in srgb, var(--accent) 10%, transparent);
		color: var(--text-secondary);
	}
	.resume-btn {
		padding: 5px 14px;
		border: none;
		border-radius: 6px;
		background: color-mix(in srgb, var(--accent) 85%, transparent);
		color: var(--text-on-accent);
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
		transition: background 0.15s;
		white-space: nowrap;
	}
	.resume-btn:hover {
		background: var(--accent);
	}
	.turn-list {
		flex: 1;
		overflow-y: auto;
		padding: 12px 16px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
</style>
