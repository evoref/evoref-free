<script lang="ts">
	import { t } from '$lib/i18n';
	import type { MemoryDetailedStats } from '$lib/free/api';
	import { serverState } from '$lib/free/stores/server';

	let { stats }: { stats: MemoryDetailedStats } = $props();

	const debug = $derived($serverState.debug);
	const debugEnabled = $derived(debug?.enabled === true);
</script>

<div class="stats-panel">
	<section class="stats-section">
		<h3>Working / STM / LTM</h3>
		<dl class="stats-grid">
			<dt>WM turns</dt>
			<dd>{stats.working.turns} / {stats.working.max_turns}</dd>
			<dt>WM tokens</dt>
			<dd>{stats.working.tokens_used} / {stats.working.max_tokens}</dd>
			<dt>STM notes</dt>
			<dd>{stats.short_term.notes} / {stats.short_term.max_notes}</dd>
			<dt>LTM chunks</dt>
			<dd>{stats.long_term.chunks}</dd>
			<dt>LTM index</dt>
			<dd>{stats.long_term.index_size_mb.toFixed(2)} MB</dd>
		</dl>
	</section>

	<section class="stats-section">
		<h3>SemMem</h3>
		<dl class="stats-grid">
			<dt>{$t('memory_inspector.stats_total_facts')}</dt>
			<dd>{stats.semantic.total_facts}</dd>
			<dt>{$t('memory_inspector.stats_total_pinned')}</dt>
			<dd>{stats.semantic.total_pinned}</dd>
		</dl>

		{#each stats.semantic.scopes as scope (scope.scope)}
			<div class="scope-block">
				<h4 class="scope-title">{scope.scope}</h4>
				<dl class="stats-grid">
					<dt>{$t('memory_inspector.stats_active')}</dt>
					<dd>{scope.active}</dd>
					<dt>{$t('memory_inspector.stats_superseded')}</dt>
					<dd>{scope.superseded}</dd>
					<dt>{$t('memory_inspector.stats_total_pinned')}</dt>
					<dd>{scope.pinned}</dd>
				</dl>
				{#if Object.keys(scope.by_type).length > 0}
					<div class="by-group">
						<span class="by-label">{$t('memory_inspector.stats_by_type')}:</span>
						{#each Object.entries(scope.by_type) as [k, v] (k)}
							<span class="chip">{k}: {v}</span>
						{/each}
					</div>
				{/if}
				{#if Object.keys(scope.by_mode_origin).length > 0}
					<div class="by-group">
						<span class="by-label">{$t('memory_inspector.stats_by_mode_origin')}:</span>
						{#each Object.entries(scope.by_mode_origin) as [k, v] (k)}
							<span class="chip">{k}: {v}</span>
						{/each}
					</div>
				{/if}
			</div>
		{/each}
	</section>

	{#if debugEnabled}
		<section class="stats-section debug-section">
			<h3>{$t('memory_inspector.stats_performance')}</h3>
			<dl class="stats-grid">
				<dt>{$t('debug_overlay.ttft')}</dt>
				<dd>{debug?.last_ttft_ms != null ? `${debug.last_ttft_ms.toFixed(0)} ms` : $t('debug_overlay.no_data')}</dd>
				<dt>{$t('debug_overlay.tok_per_sec')}</dt>
				<dd>{debug?.last_tok_per_sec != null ? debug.last_tok_per_sec.toFixed(1) : $t('debug_overlay.no_data')}</dd>
				<dt>{$t('debug_overlay.cache_hit_rate')}</dt>
				<dd>{debug ? `${(debug.cache_hit_rate * 100).toFixed(1)}%` : '—'}</dd>
			</dl>
		</section>

		<section class="stats-section debug-section">
			<h3>{$t('memory_inspector.stats_learning')}</h3>
			<dl class="stats-grid">
				<dt>{$t('memory_inspector.stats_learning_status')}</dt>
				<dd class:running={debug?.learning?.running}>
					{debug?.learning?.running ? $t('debug_overlay.running') : $t('debug_overlay.idle')}
				</dd>
				<dt>{$t('debug_overlay.experiences')}</dt>
				<dd>{debug?.learning?.experience_count ?? 0}</dd>
				<dt>{$t('debug_overlay.conditions_met')}</dt>
				<dd class:met={debug?.learning?.conditions_met}>
					{debug?.learning?.conditions_met ? $t('debug_overlay.met') : $t('debug_overlay.not_met')}
				</dd>
			</dl>
		</section>
	{/if}
</div>

<style>
	.stats-panel {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}
	.stats-section {
		background: var(--bg-secondary);
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		padding: 12px 16px;
	}
	.stats-section h3 {
		margin: 0 0 8px 0;
		font-size: 14px;
		color: var(--text-primary);
	}
	.stats-grid {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: 4px 16px;
		margin: 0;
		font-size: 13px;
	}
	.stats-grid dt {
		color: var(--text-secondary);
	}
	.stats-grid dd {
		margin: 0;
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
	}
	.scope-block {
		margin-top: 12px;
		padding-top: 8px;
		border-top: 0.5px dashed var(--input-border);
	}
	.scope-title {
		margin: 0 0 6px 0;
		font-size: 13px;
		font-family: ui-monospace, monospace;
		color: var(--text-primary);
	}
	.by-group {
		margin-top: 6px;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		font-size: 12px;
	}
	.by-label {
		color: var(--text-secondary);
	}
	.chip {
		padding: 2px 8px;
		border-radius: 10px;
		background: color-mix(in srgb, var(--accent) 18%, transparent);
		color: var(--text-primary);
		font-size: 11px;
	}
	.debug-section {
		border-left: 3px solid var(--warning, #f59e0b);
	}
	.debug-section h3::after {
		content: 'DEBUG';
		margin-left: 8px;
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.5px;
		padding: 1px 6px;
		border-radius: 3px;
		vertical-align: middle;
	}
	:global([data-theme='light']) .debug-section h3::after {
		background: var(--warning, #f59e0b);
		color: var(--text-primary);
		border: 1px solid var(--warning, #f59e0b);
	}
	:global([data-theme='dark']) .debug-section h3::after {
		background: transparent;
		color: var(--warning, #f59e0b);
		border: 1px solid var(--warning, #f59e0b);
	}
	dd.running {
		color: var(--warning, #f59e0b);
	}
	dd.met {
		color: var(--success, #22c55e);
	}
</style>
