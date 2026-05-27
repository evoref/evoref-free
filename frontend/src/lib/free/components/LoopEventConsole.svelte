<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { t } from '$lib/i18n';
	import {
		events,
		connectionStatus,
		reconnectAttempt,
		LOOP_EVENT_KINDS,
		type LoopEvent,
		type LoopEventKind,
		connect,
		disconnect,
		clearEvents,
		pause,
		resume,
		reconnectNow
	} from '$lib/free/stores/loop_events';

	const ALL_KINDS_SET: Set<LoopEventKind> = new Set(LOOP_EVENT_KINDS);

	let selectedKinds = $state<Set<LoopEventKind>>(new Set(LOOP_EVENT_KINDS));
	let expandedIds = $state<Set<string>>(new Set());
	let autoScroll = $state<boolean>(true);
	let isPaused = $state<boolean>(false);

	let scrollContainer: HTMLDivElement | undefined = $state();
	// ユーザーが上方向にスクロールしたら autoScroll を切るための判定。
	// プログラム的スクロールと区別するためのフラグ。
	let lastScrollHeight = 0;
	let suppressNextScroll = false;

	let filteredEvents = $derived(
		$events.filter((e) => selectedKinds.has(e.event))
	);

	const allKindsSelected = $derived(
		selectedKinds.size === LOOP_EVENT_KINDS.length
	);

	function shortTrace(trace: string): string {
		if (!trace) return '-';
		return trace.length <= 12 ? trace : trace.slice(0, 12);
	}

	function formatTimestamp(ts: number): string {
		if (!Number.isFinite(ts) || ts <= 0) return '-';
		const d = new Date(ts * 1000);
		const hh = String(d.getHours()).padStart(2, '0');
		const mm = String(d.getMinutes()).padStart(2, '0');
		const ss = String(d.getSeconds()).padStart(2, '0');
		const ms = String(d.getMilliseconds()).padStart(3, '0');
		return `${hh}:${mm}:${ss}.${ms}`;
	}

	function toggleKind(kind: LoopEventKind): void {
		const next = new Set(selectedKinds);
		if (next.has(kind)) {
			next.delete(kind);
		} else {
			next.add(kind);
		}
		selectedKinds = next;
	}

	function toggleAllKinds(): void {
		selectedKinds = allKindsSelected ? new Set() : new Set(ALL_KINDS_SET);
	}

	function toggleExpanded(clientId: string): void {
		const next = new Set(expandedIds);
		if (next.has(clientId)) {
			next.delete(clientId);
		} else {
			next.add(clientId);
		}
		expandedIds = next;
	}

	function handlePauseToggle(): void {
		if (isPaused) {
			resume();
			isPaused = false;
		} else {
			pause();
			isPaused = true;
		}
	}

	function handleClear(): void {
		clearEvents();
		expandedIds = new Set();
	}

	function handleReconnect(): void {
		reconnectNow();
	}

	function handleScroll(): void {
		if (!scrollContainer) return;
		if (suppressNextScroll) {
			suppressNextScroll = false;
			return;
		}
		const el = scrollContainer;
		// 末尾から 24px 以内なら autoScroll を維持、それ以外は停止
		const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
		autoScroll = distanceFromBottom < 24;
	}

	function scrollToBottom(): void {
		if (!scrollContainer) return;
		suppressNextScroll = true;
		scrollContainer.scrollTop = scrollContainer.scrollHeight;
		lastScrollHeight = scrollContainer.scrollHeight;
	}

	$effect(() => {
		// filteredEvents.length を読むことで再評価をトリガする
		void filteredEvents.length;
		if (!autoScroll) return;
		const rafId = requestAnimationFrame(() => {
			if (scrollContainer) {
				const newHeight = scrollContainer.scrollHeight;
				if (newHeight !== lastScrollHeight) {
					scrollToBottom();
				}
			}
		});
		return () => cancelAnimationFrame(rafId);
	});

	onMount(() => {
		connect();
	});

	onDestroy(() => {
		// ルート離脱時は接続を切る (再入時に connect が走る)
		disconnect();
	});

	function statusKey(s: string): string {
		switch (s) {
			case 'connecting':
				return 'loop.console.status.connecting';
			case 'open':
				return 'loop.console.status.open';
			case 'reconnecting':
				return 'loop.console.status.reconnecting';
			case 'closed':
				return 'loop.console.status.closed';
			case 'paused':
				return 'loop.console.status.paused';
			default:
				return 'loop.console.status.idle';
		}
	}

	function eventKindKey(kind: LoopEventKind): string {
		return `loop.console.filter.${kind}`;
	}

	function renderData(evt: LoopEvent): string {
		try {
			return JSON.stringify(evt.data, null, 2);
		} catch {
			return String(evt.data);
		}
	}
</script>

<section class="loop-console" aria-label={$t('loop.console.title')}>
	<header class="loop-console-header">
		<h1 class="title">{$t('loop.console.title')}</h1>
		<div class="status-row">
			<span
				class="status-badge"
				data-status={$connectionStatus}
				aria-live="polite"
			>
				{$t(statusKey($connectionStatus))}
				{#if $connectionStatus === 'reconnecting' && $reconnectAttempt > 0}
					<span class="status-attempt">#{$reconnectAttempt}</span>
				{/if}
			</span>
			<span class="counter" aria-live="polite">
				{$t('loop.console.counter', {
					shown: filteredEvents.length,
					total: $events.length
				})}
			</span>
		</div>
		<div class="actions">
			<button type="button" class="btn" onclick={handlePauseToggle}>
				{isPaused
					? $t('loop.console.actions.resume')
					: $t('loop.console.actions.pause')}
			</button>
			<button type="button" class="btn" onclick={handleClear}>
				{$t('loop.console.actions.clear')}
			</button>
			<button type="button" class="btn" onclick={handleReconnect}>
				{$t('loop.console.actions.reconnect')}
			</button>
		</div>
	</header>

	<div class="filter-row" role="group" aria-label={$t('loop.console.filter_label')}>
		<button
			type="button"
			class="filter-chip filter-all"
			class:active={allKindsSelected}
			onclick={toggleAllKinds}
		>
			{$t('loop.console.filter.all')}
		</button>
		{#each LOOP_EVENT_KINDS as kind (kind)}
			<button
				type="button"
				class="filter-chip"
				data-kind={kind}
				class:active={selectedKinds.has(kind)}
				onclick={() => toggleKind(kind)}
			>
				{$t(eventKindKey(kind))}
			</button>
		{/each}
	</div>

	<div
		class="event-stream"
		role="log"
		aria-live="polite"
		aria-label={$t('loop.console.stream_label')}
		bind:this={scrollContainer}
		onscroll={handleScroll}
	>
		{#if filteredEvents.length === 0}
			<div class="empty">
				{#if isPaused}
					{$t('loop.console.paused')}
				{:else}
					{$t('loop.console.empty')}
				{/if}
			</div>
		{:else}
			{#each filteredEvents as evt (evt.clientId)}
				{@const expanded = expandedIds.has(evt.clientId)}
				<div class="event-row-container" data-kind={evt.event}>
					<button
						type="button"
						class="event-row"
						data-kind={evt.event}
						aria-expanded={expanded}
						onclick={() => toggleExpanded(evt.clientId)}
					>
						<span class="event-time">{formatTimestamp(evt.timestamp)}</span>
						<span class="event-kind" data-kind={evt.event}>
							{$t(eventKindKey(evt.event))}
						</span>
						<span class="event-iter">
							{$t('loop.console.iteration_label', { iteration: evt.iteration })}
						</span>
						<span class="event-trace" title={evt.trace_id}>
							{$t('loop.console.trace_id_label')}
							<span class="trace-value">{shortTrace(evt.trace_id)}</span>
						</span>
						{#if evt.project_id}
							<span class="event-project" title={evt.project_id}>
								{evt.project_id}
							</span>
						{/if}
					</button>
					{#if expanded}
						<pre class="event-data">{renderData(evt)}</pre>
					{/if}
				</div>
			{/each}
		{/if}
	</div>
</section>

<style>
	.loop-console {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
		background: var(--bg-primary);
		color: var(--text-primary);
	}
	.loop-console-header {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px;
		padding: 12px 16px;
		border-bottom: 0.5px solid var(--input-border);
		background: var(--bg-secondary, var(--bg-primary));
	}
	.title {
		margin: 0;
		font-size: 1.05rem;
		font-weight: 700;
		color: var(--text-primary);
	}
	.status-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.status-badge {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 2px 8px;
		font-size: 0.78rem;
		font-weight: 600;
		border-radius: 4px;
		border: 1px solid var(--input-border);
		background: var(--control-bg);
		color: var(--text-secondary);
	}
	.status-badge[data-status='open'] {
		color: var(--success, var(--accent));
		border-color: var(--success, var(--accent));
	}
	.status-badge[data-status='connecting'],
	.status-badge[data-status='reconnecting'] {
		color: var(--warning, var(--accent));
		border-color: var(--warning, var(--accent));
	}
	.status-badge[data-status='closed'] {
		color: var(--error, var(--text-secondary));
		border-color: var(--error, var(--text-secondary));
	}
	.status-badge[data-status='paused'] {
		color: var(--text-secondary);
	}
	.status-attempt {
		font-weight: 500;
		opacity: 0.8;
	}
	.counter {
		font-size: 0.78rem;
		color: var(--text-secondary);
	}
	.actions {
		display: flex;
		gap: 6px;
		margin-left: auto;
	}
	.btn {
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		padding: 4px 10px;
		font-size: 0.85rem;
		cursor: pointer;
		font-family: inherit;
		transition: background 0.15s;
	}
	.btn:hover {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
		border-color: var(--accent);
	}
	.filter-row {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		padding: 8px 16px;
		border-bottom: 0.5px solid var(--input-border);
	}
	.filter-chip {
		background: var(--control-bg);
		color: var(--text-secondary);
		border: 0.5px solid var(--input-border);
		border-radius: 999px;
		padding: 2px 10px;
		font-size: 0.75rem;
		cursor: pointer;
		font-family: inherit;
		transition: all 0.15s;
	}
	.filter-chip.active {
		background: color-mix(in srgb, var(--accent) 25%, transparent);
		color: var(--text-primary);
		border-color: var(--accent);
	}
	.filter-chip.filter-all {
		font-weight: 600;
	}
	.event-stream {
		flex: 1;
		overflow-y: auto;
		padding: 8px 0;
		font-family: var(--font-monospace, monospace);
		font-size: 0.82rem;
		min-height: 0;
	}
	.empty {
		padding: 24px;
		color: var(--text-secondary);
		text-align: center;
		font-style: italic;
	}
	.event-row-container {
		border-bottom: 0.5px dashed
			color-mix(in srgb, var(--input-border) 50%, transparent);
	}
	.event-row {
		display: grid;
		grid-template-columns: max-content max-content max-content max-content 1fr;
		grid-template-areas: 'time kind iter trace project';
		gap: 8px;
		align-items: baseline;
		width: 100%;
		text-align: left;
		padding: 4px 16px;
		background: transparent;
		border: none;
		color: var(--text-primary);
		font-family: inherit;
		font-size: inherit;
		cursor: pointer;
	}
	.event-row:hover {
		background: color-mix(in srgb, var(--accent) 6%, transparent);
	}
	.event-row[aria-expanded='true'] {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.event-time {
		grid-area: time;
		color: var(--text-secondary);
	}
	.event-kind {
		grid-area: kind;
		font-weight: 600;
		padding: 0 6px;
		border-radius: 3px;
		border: 0.5px solid currentColor;
	}
	/* イベント種別ごとの色分け — 全て CSS 変数経由 */
	.event-kind[data-kind='loop_started'],
	.event-kind[data-kind='loop_resumed'],
	.event-kind[data-kind='iteration_started'] {
		color: var(--accent);
	}
	.event-kind[data-kind='loop_paused'],
	.event-kind[data-kind='loop_stopped'],
	.event-kind[data-kind='iteration_ended'] {
		color: var(--text-secondary);
	}
	.event-kind[data-kind='task_picked'],
	.event-kind[data-kind='action_executed'] {
		color: var(--success, var(--accent));
	}
	.event-kind[data-kind='gate_result'] {
		color: var(--warning, var(--accent));
	}
	.event-kind[data-kind='fact_written'] {
		color: var(--info, var(--accent));
	}
	.event-iter {
		grid-area: iter;
		color: var(--text-secondary);
		font-size: 0.78rem;
	}
	.event-trace {
		grid-area: trace;
		color: var(--text-secondary);
		font-size: 0.78rem;
	}
	.trace-value {
		color: var(--text-primary);
		font-family: var(--font-monospace, monospace);
	}
	.event-project {
		grid-area: project;
		color: var(--text-secondary);
		font-size: 0.78rem;
		max-width: 200px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.event-data {
		margin: 0 16px 8px;
		padding: 8px;
		background: var(--control-bg);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		color: var(--text-primary);
		white-space: pre-wrap;
		word-break: break-all;
		max-height: 320px;
		overflow-y: auto;
	}
</style>
