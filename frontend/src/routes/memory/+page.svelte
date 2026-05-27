<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { t } from '$lib/i18n';
	import { isPro } from '$lib/edition';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import FactTable from '$lib/free/components/memory/FactTable.svelte';
	import TaskTable from '$lib/free/components/memory/TaskTable.svelte';
	import StatsPanel from '$lib/free/components/memory/StatsPanel.svelte';
	import ConflictsPanel from '$lib/free/components/memory/ConflictsPanel.svelte';
	import {
		getMemoryStats,
		listFacts,
		listPolicyFacts,
		listTaskFacts,
		listConflicts,
		resolveConflict,
		type ConflictAction,
		type ConflictGroupInfo,
		type FactInfo,
		type FactsResponse,
		type MemoryDetailedStats,
		type TaskFactInfo
	} from '$lib/free/api';

	type Tab =
		| 'chat'
		| 'coding'
		| 'pinned'
		| 'policy'
		| 'failure'
		| 'task'
		| 'conflicts'
		| 'stats';

	let activeTab = $state<Tab>('chat');
	let scope = $state<string>('global');
	let onlyActivePolicy = $state(false);
	let includeSuperseded = $state(false);

	let stats = $state<MemoryDetailedStats | null>(null);
	let facts = $state<FactInfo[]>([]);
	let factsTotal = $state(0);
	let tasks = $state<TaskFactInfo[]>([]);
	let tasksTotal = $state(0);
	let conflictsPending = $state<ConflictGroupInfo[]>([]);
	let conflictsHistory = $state<ConflictGroupInfo[]>([]);

	let loading = $state(false);
	let error = $state('');

	let fetchAbort: AbortController | null = null;

	async function loadStats() {
		try {
			stats = await getMemoryStats();
		} catch (e) {
			console.error('[Memory] stats fetch failed', e);
		}
	}

	async function loadCurrentTab() {
		fetchAbort?.abort();
		fetchAbort = new AbortController();
		loading = true;
		error = '';
		try {
			if (activeTab === 'chat') {
				const r: FactsResponse = await listFacts({ scope, mode: 'chat', include_superseded: includeSuperseded });
				facts = r.facts;
				factsTotal = r.total;
			} else if (activeTab === 'coding') {
				const r: FactsResponse = await listFacts({ scope, mode: 'coding', include_superseded: includeSuperseded });
				facts = r.facts;
				factsTotal = r.total;
			} else if (activeTab === 'pinned') {
				const r: FactsResponse = await listFacts({ scope, pinned: true, include_superseded: includeSuperseded });
				facts = r.facts;
				factsTotal = r.total;
			} else if (activeTab === 'policy') {
				const r: FactsResponse = await listPolicyFacts({
					scope,
					only_active: onlyActivePolicy,
					include_superseded: includeSuperseded
				});
				facts = r.facts;
				factsTotal = r.total;
			} else if (activeTab === 'failure') {
				const r: FactsResponse = await listFacts({
					scope,
					type: 'failure_pattern',
					include_superseded: includeSuperseded
				});
				facts = r.facts;
				factsTotal = r.total;
			} else if (activeTab === 'task') {
				if (!scope.startsWith('project:')) {
					tasks = [];
					tasksTotal = 0;
					error = 'task_scope_required';
				} else {
					const r = await listTaskFacts({ scope });
					tasks = r.tasks;
					tasksTotal = r.total;
				}
			} else if (activeTab === 'conflicts') {
				const r = await listConflicts({ scope });
				conflictsPending = r.pending;
				conflictsHistory = r.auto_resolved_history;
			} else if (activeTab === 'stats') {
				await loadStats();
			}
		} catch (e) {
			console.error('[Memory] tab fetch failed', e);
			error = 'load_failed';
		} finally {
			loading = false;
		}
	}

	function setTab(tab: Tab) {
		activeTab = tab;
		loadCurrentTab();
	}

	function refresh() {
		loadStats();
		loadCurrentTab();
	}

	onMount(() => {
		if (!isPro) {
			goto('/');
			return;
		}
		loadStats();
		loadCurrentTab();
	});

	const tabs: { id: Tab; labelKey: string }[] = [
		{ id: 'chat', labelKey: 'memory_inspector.tab_chat' },
		{ id: 'coding', labelKey: 'memory_inspector.tab_coding' },
		{ id: 'pinned', labelKey: 'memory_inspector.tab_pinned' },
		{ id: 'policy', labelKey: 'memory_inspector.tab_policy' },
		{ id: 'failure', labelKey: 'memory_inspector.tab_failure' },
		{ id: 'task', labelKey: 'memory_inspector.tab_task' },
		{ id: 'conflicts', labelKey: 'memory_inspector.tab_conflicts' },
		{ id: 'stats', labelKey: 'memory_inspector.tab_stats' }
	];

	async function handleResolve(
		group: ConflictGroupInfo,
		action: ConflictAction,
		mergedObject?: string
	) {
		if (!group.winner_id) return;
		try {
			await resolveConflict({
				scope: group.scope,
				winner_id: group.winner_id,
				loser_ids: group.loser_ids,
				action,
				merged_object: mergedObject
			});
			await loadCurrentTab();
			await loadStats();
		} catch (e) {
			console.error('[Memory] resolve conflict failed', e);
			error = 'load_failed';
		}
	}
</script>

<PageLayout title={$t('memory_inspector.title')}>
	<div class="memory-toolbar">
		<label class="field">
			<span class="field-label">{$t('memory_inspector.scope_label')}</span>
			<input
				type="text"
				class="scope-input"
				placeholder={$t('memory_inspector.scope_input_placeholder')}
				bind:value={scope}
				onkeydown={(e) => {
					if (e.key === 'Enter') refresh();
				}}
			/>
		</label>
		{#if stats}
			<span class="mode-pill">
				{$t('memory_inspector.current_mode')}: <strong>{stats.current_mode}</strong>
			</span>
		{/if}
		<button class="btn-refresh" type="button" onclick={refresh}>
			{$t('memory_inspector.refresh')}
		</button>
	</div>

	<nav class="tab-bar" aria-label="memory inspector tabs">
		{#each tabs as tab (tab.id)}
			<button
				type="button"
				class="tab"
				class:active={activeTab === tab.id}
				onclick={() => setTab(tab.id)}
			>
				{$t(tab.labelKey)}
			</button>
		{/each}
	</nav>

	<div class="filter-row">
		{#if activeTab === 'policy'}
			<label class="checkbox">
				<input type="checkbox" bind:checked={onlyActivePolicy} onchange={loadCurrentTab} />
				{$t('memory_inspector.only_active_label')}
			</label>
		{/if}
		{#if activeTab !== 'task' && activeTab !== 'stats'}
			<label class="checkbox">
				<input type="checkbox" bind:checked={includeSuperseded} onchange={loadCurrentTab} />
				{$t('memory_inspector.include_superseded')}
			</label>
		{/if}
	</div>

	{#if loading}
		<p class="status-line">{$t('memory_inspector.loading')}</p>
	{:else if error === 'load_failed'}
		<p class="status-line error">{$t('memory_inspector.load_failed')}</p>
	{/if}

	<div class="content">
		{#if activeTab === 'stats'}
			{#if stats}
				<StatsPanel {stats} />
			{/if}
		{:else if activeTab === 'task'}
			<TaskTable {tasks} />
			<p class="count-line">
				{$t('memory_inspector.showing_count', { shown: tasks.length, total: tasksTotal })}
			</p>
		{:else if activeTab === 'conflicts'}
			<ConflictsPanel
				pending={conflictsPending}
				history={conflictsHistory}
				onResolve={handleResolve}
			/>
		{:else}
			<FactTable
				{facts}
				showFailureSig={activeTab === 'failure'}
				showEvalMetric={activeTab === 'policy'}
			/>
			<p class="count-line">
				{$t('memory_inspector.showing_count', { shown: facts.length, total: factsTotal })}
			</p>
		{/if}
	</div>
</PageLayout>

<style>
	.memory-toolbar {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.field {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.field-label {
		font-size: 13px;
		color: var(--text-secondary);
	}
	.scope-input {
		padding: 6px 10px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 13px;
		min-width: 220px;
		font-family: inherit;
	}
	.mode-pill {
		font-size: 12px;
		color: var(--text-secondary);
		padding: 4px 10px;
		border-radius: 12px;
		background: color-mix(in srgb, var(--accent) 15%, transparent);
	}
	.btn-refresh {
		padding: 6px 12px;
		border-radius: 6px;
		background: var(--accent);
		color: var(--text-on-accent, #fff);
		border: 0.5px solid var(--accent);
		cursor: pointer;
		font-size: 13px;
	}
	.tab-bar {
		display: flex;
		gap: 4px;
		border-bottom: 0.5px solid var(--input-border);
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.tab {
		padding: 8px 14px;
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 13px;
		border-bottom: 2px solid transparent;
		transition: color 0.15s, border-color 0.15s;
	}
	.tab:hover {
		color: var(--text-primary);
	}
	.tab.active {
		color: var(--text-primary);
		border-bottom-color: var(--accent);
	}
	.filter-row {
		display: flex;
		gap: 12px;
		margin-bottom: 8px;
		min-height: 22px;
	}
	.checkbox {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: var(--text-secondary);
		cursor: pointer;
	}
	.status-line {
		padding: 8px 0;
		color: var(--text-secondary);
		font-size: 13px;
	}
	.status-line.error {
		color: var(--color-error, #ef4444);
	}
	.content {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.count-line {
		margin: 4px 0 0;
		font-size: 12px;
		color: var(--text-secondary);
		text-align: right;
	}
</style>
