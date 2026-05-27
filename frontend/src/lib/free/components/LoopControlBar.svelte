<script lang="ts">
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n';
	import { events } from '$lib/free/stores/loop_events';
	import {
		projectId,
		loopState,
		busy,
		refreshStatus,
		start,
		stop,
		pause,
		resume
	} from '$lib/free/stores/loop_control';

	let running = $derived($loopState?.running ?? false);
	let paused = $derived($loopState?.paused ?? false);

	let statusText = $derived(
		!running
			? $t('loop.control.stopped')
			: paused
				? $t('loop.control.paused')
				: $t('loop.control.running')
	);

	onMount(() => {
		refreshStatus();
	});

	// SSE のループライフサイクルイベントを検知して状態を再取得 (外部 CLI 起動も反映)。
	let lastSeenId = '';
	$effect(() => {
		const list = $events;
		const latest = list[list.length - 1];
		if (!latest || latest.clientId === lastSeenId) return;
		lastSeenId = latest.clientId;
		if (
			latest.event === 'loop_started' ||
			latest.event === 'loop_stopped' ||
			latest.event === 'loop_paused' ||
			latest.event === 'loop_resumed'
		) {
			refreshStatus();
		}
	});
</script>

<div class="loop-control" role="group" aria-label={$t('loop.control.title')}>
	<span class="control-title">{$t('loop.control.title')}</span>

	<label class="project-field">
		<span class="field-label">{$t('loop.control.project_label')}</span>
		<input
			type="text"
			bind:value={$projectId}
			placeholder={$t('loop.control.project_placeholder')}
			disabled={running}
		/>
	</label>

	<div class="control-buttons">
		<button type="button" class="btn btn-start" onclick={start} disabled={$busy || running}>
			{$t('loop.control.start')}
		</button>
		<button type="button" class="btn" onclick={pause} disabled={$busy || !running || paused}>
			{$t('loop.control.pause')}
		</button>
		<button type="button" class="btn" onclick={resume} disabled={$busy || !running || !paused}>
			{$t('loop.control.resume')}
		</button>
		<button type="button" class="btn btn-stop" onclick={stop} disabled={$busy || !running}>
			{$t('loop.control.stop')}
		</button>
	</div>

	<div class="control-status" aria-live="polite">
		<span class="status-badge" class:running class:paused>{statusText}</span>
		{#if running}
			<span class="iter">{$t('loop.control.iteration', { iteration: $loopState?.iteration ?? 0 })}</span>
		{/if}
	</div>
</div>

<style>
	.loop-control {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px;
		padding: 10px 16px;
		border-bottom: 0.5px solid var(--input-border);
		background: var(--bg-secondary, var(--bg-primary));
	}
	.control-title {
		font-weight: 700;
		font-size: 0.95rem;
		color: var(--text-primary);
	}
	.project-field {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.field-label {
		font-size: 0.8rem;
		color: var(--text-secondary);
	}
	.project-field input {
		background: var(--input-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		padding: 4px 8px;
		font-size: 0.85rem;
		font-family: inherit;
		min-width: 160px;
		outline: none;
	}
	.project-field input:focus {
		border-color: var(--accent);
	}
	.project-field input:disabled {
		opacity: 0.6;
	}
	.control-buttons {
		display: flex;
		gap: 6px;
	}
	.btn {
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		padding: 4px 12px;
		font-size: 0.85rem;
		cursor: pointer;
		font-family: inherit;
		transition: background 0.15s;
	}
	.btn:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
		border-color: var(--accent);
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.btn-start:not(:disabled) {
		border-color: var(--success, var(--accent));
		color: var(--success, var(--accent));
	}
	.btn-stop:not(:disabled) {
		border-color: var(--error, var(--text-secondary));
		color: var(--error, var(--text-secondary));
	}
	.control-status {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-left: auto;
	}
	.status-badge {
		display: inline-flex;
		align-items: center;
		padding: 2px 10px;
		font-size: 0.78rem;
		font-weight: 600;
		border-radius: 4px;
		border: 1px solid var(--input-border);
		background: var(--control-bg);
		color: var(--text-secondary);
	}
	.status-badge.running {
		color: var(--success, var(--accent));
		border-color: var(--success, var(--accent));
	}
	.status-badge.paused {
		color: var(--warning, var(--accent));
		border-color: var(--warning, var(--accent));
	}
	.iter {
		font-size: 0.78rem;
		color: var(--text-secondary);
	}
</style>
