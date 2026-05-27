<script lang="ts">
	import { t } from '$lib/i18n';
	import type { DateGroup } from '$lib/free/types/history';
	import SessionCard from './SessionCard.svelte';

	interface Props {
		dateGroups: DateGroup[];
		loading: boolean;
		error: string;
		sessionsEmpty: boolean;
		hasMore: boolean;
		selectedId: string | null;
		query: string;
		onselect: (sessionId: string) => void;
		ondelete: (e: MouseEvent, sessionId: string) => void;
		onloadmore: () => void;
	}

	let {
		dateGroups, loading, error, sessionsEmpty,
		hasMore, selectedId, query,
		onselect, ondelete, onloadmore,
	}: Props = $props();

	function displayLabel(label: string): string {
		if (label === 'today') return $t('history_page.today');
		if (label === 'yesterday') return $t('history_page.yesterday');
		return label;
	}
</script>

<div class="session-list-pane">
	{#if loading && sessionsEmpty}
		<div class="center-msg">{$t('history_page.loading')}</div>
	{:else if error}
		<div class="center-msg error">{$t('history_page.load_failed')}</div>
	{:else if sessionsEmpty}
		<div class="center-msg">{$t('history_page.no_sessions')}</div>
	{:else}
		<div class="session-list">
			{#each dateGroups as group}
				<div class="date-group">
					<h3 class="date-label"><span>{displayLabel(group.label)}</span></h3>
					{#each group.sessions as session}
						<SessionCard
							{session}
							selected={selectedId === session.session_id}
							{query}
							{onselect}
							{ondelete}
						/>
					{/each}
				</div>
			{/each}
			{#if hasMore}
				<button class="load-more-btn" onclick={onloadmore} disabled={loading}>
					{loading ? $t('history_page.loading') : $t('history_page.load_more')}
				</button>
			{/if}
		</div>
	{/if}
</div>

<style>
	.session-list-pane {
		width: 340px;
		min-width: 280px;
		flex-shrink: 0;
		overflow-y: auto;
		background: var(--bg-primary);
		padding: 8px;
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
	.session-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.date-group {
		margin-bottom: 8px;
	}
	.date-label {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		font-weight: 500;
		color: var(--text-secondary);
		margin: 0 0 4px;
		padding: 0 4px;
	}
	.date-label::after {
		content: '';
		flex: 1;
		height: 1px;
		background: var(--border-primary, var(--text-secondary));
		opacity: 0.3;
	}
	.load-more-btn {
		display: block;
		margin: 8px auto;
		padding: 6px 16px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
		transition: background 0.15s;
	}
	.load-more-btn:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.load-more-btn:disabled {
		opacity: 0.5;
		cursor: default;
	}
</style>
