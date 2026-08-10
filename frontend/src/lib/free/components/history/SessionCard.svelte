<script lang="ts">
	import { t } from '$lib/i18n';
	import { formatDuration, sessionDisplayText, sessionMatchedPreview } from '$lib/free/utils/history';
	import { formatTime } from '$lib/free/utils/format';
	import type { SessionSummary } from '$lib/free/types/history';

	interface Props {
		session: SessionSummary;
		selected: boolean;
		query: string;
		onselect: (sessionId: string) => void;
		ondelete: (e: MouseEvent, sessionId: string) => void;
	}

	let { session, selected, query, onselect, ondelete }: Props = $props();

	let matchedPreview = $derived(sessionMatchedPreview(session, query));
	let displayText = $derived(sessionDisplayText(session, $t('history_page.no_summary')));

	let ariaLabel = $derived(
		`${session.mode} - ${displayText} - ${$t('history_page.turns', { count: session.turn_count })}`
	);
</script>

<div
	class="session-card"
	class:selected
	onclick={() => onselect(session.session_id)}
	onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onselect(session.session_id); } }}
	role="button"
	tabindex="0"
	aria-label={ariaLabel}
	aria-pressed={selected}
>
	<div class="session-main">
		<span class="session-mode mode-{session.mode}">{session.mode}</span>
		<span class="session-summary">{displayText}</span>
	</div>
	{#if matchedPreview}
		<div class="session-match-preview">{matchedPreview}</div>
	{/if}
	<div class="session-meta">
		<span class="session-time">{formatTime(session.started_at)}</span>
		<span class="session-turns">{$t('history_page.turns', { count: session.turn_count })}</span>
		<span class="session-duration">{$t('history_page.duration', { minutes: formatDuration(session.duration_sec) })}</span>
		<button
			class="delete-btn"
			onclick={(e) => ondelete(e, session.session_id)}
			aria-label={$t('history_page.delete_confirm')}
		>
			{$t('common.delete')}
		</button>
	</div>
</div>

<style>
	.session-card {
		display: block;
		width: 100%;
		text-align: left;
		padding: 8px 10px;
		border-radius: 6px;
		background: transparent;
		border: none;
		cursor: pointer;
		transition: background 0.15s;
		font-family: inherit;
		color: inherit;
	}
	.session-card:hover {
		background: color-mix(in srgb, var(--accent) 6%, transparent);
	}
	.session-card.selected {
		background: color-mix(in srgb, var(--accent) 22%, transparent);
	}
	:global([data-theme='dark']) .session-card.selected {
		background: color-mix(in srgb, var(--accent) 32%, transparent);
	}
	.session-main {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-bottom: 3px;
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
	.session-mode.mode-create {
		background: color-mix(in srgb, var(--color-success) 15%, transparent);
		color: var(--color-success);
	}
	.session-summary {
		font-size: 14px;
		color: var(--text-primary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.session-match-preview {
		font-size: 11px;
		color: var(--text-secondary);
		margin-top: 2px;
		padding-left: 2px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		opacity: 0.8;
	}
	.session-meta {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 11px;
		color: var(--text-secondary);
		padding-left: 2px;
	}
	.delete-btn {
		margin-left: auto;
		font-size: 11px;
		color: var(--text-secondary);
		background: none;
		border: none;
		cursor: pointer;
		padding: 1px 4px;
		border-radius: 3px;
		opacity: 0;
		transition: opacity 0.15s, color 0.15s;
	}
	.session-card:hover .delete-btn,
	.delete-btn:focus-visible {
		opacity: 1;
	}
	.delete-btn:hover {
		color: var(--color-error);
	}
</style>
