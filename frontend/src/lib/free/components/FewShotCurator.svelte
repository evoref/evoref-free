<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import { listFewshot, actOnFewshot, ApiError } from '$lib/free/api';
	import type { FewShotAction, FewShotExample } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatDate } from '$lib/free/utils/format';

	type ModeFilter = 'all' | 'chat' | 'create';

	let examples = $state<FewShotExample[]>([]);
	let archived = $state<FewShotExample[]>([]);
	let poolSize = $state(0);
	let writeback = $state('');
	let modeFilter = $state<ModeFilter>('all');
	let loading = $state(true);
	let errorMessage = $state('');
	let delegatedMessage = $state('');
	let expandedId = $state<string | null>(null);
	let actingId = $state<string | null>(null);

	onMount(() => {
		refresh();
	});

	async function refresh() {
		loading = true;
		errorMessage = '';
		delegatedMessage = '';
		try {
			const mode = modeFilter === 'all' ? undefined : modeFilter;
			const result = await listFewshot(mode);
			examples = result.examples;
			archived = result.archived;
			poolSize = result.pool_size;
			writeback = result.writeback;
		} catch (e) {
			if (e instanceof ApiError && e.code === 'E0409') {
				delegatedMessage = e.i18nKey ? $t(e.i18nKey) : e.message;
				examples = [];
				archived = [];
			} else {
				errorMessage = e instanceof ApiError && e.i18nKey ? $t(e.i18nKey) : $t('fewshot.list_failed');
			}
		} finally {
			loading = false;
		}
	}

	async function handleFilterChange(next: ModeFilter) {
		modeFilter = next;
		await refresh();
	}

	async function handleAction(id: string, action: FewShotAction) {
		if (actingId) return;
		actingId = id;
		try {
			await handleApiCall(() => actOnFewshot(id, action), {
				fallbackKey: 'fewshot.action_failed'
			});
			await refresh();
		} finally {
			actingId = null;
		}
	}

	function toggleExpand(id: string) {
		expandedId = expandedId === id ? null : id;
	}

	function truncate(text: string, max = 80): string {
		if (text.length <= max) return text;
		return `${text.slice(0, max)}...`;
	}

	function formatLastUsed(iso: string): string {
		if (!iso) return $t('fewshot.never_used');
		return formatDate(iso);
	}
</script>

<div class="fewshot-curator">
	<div class="header">
		<div class="filters">
			<button
				class="filter-btn"
				class:active={modeFilter === 'all'}
				onclick={() => handleFilterChange('all')}
			>
				{$t('fewshot.filter_all')}
			</button>
			<button
				class="filter-btn"
				class:active={modeFilter === 'chat'}
				onclick={() => handleFilterChange('chat')}
			>
				{$t('fewshot.filter_chat')}
			</button>
			<button
				class="filter-btn"
				class:active={modeFilter === 'create'}
				onclick={() => handleFilterChange('create')}
			>
				{$t('fewshot.filter_create')}
			</button>
		</div>
		{#if !loading && !errorMessage && !delegatedMessage}
			<span class="pool-summary">
				{$t('fewshot.pool_summary', { pool_size: poolSize, writeback })}
			</span>
		{/if}
	</div>

	{#if loading}
		<p class="hint">{$t('fewshot.loading')}</p>
	{:else if delegatedMessage}
		<p class="hint delegated">{delegatedMessage}</p>
	{:else if errorMessage}
		<p class="error">{errorMessage}</p>
	{:else}
		<section class="fewshot-section">
			<h2 class="section-title">{$t('fewshot.section_pool')}</h2>
			{#if examples.length === 0}
				<p class="empty">{$t('fewshot.list_empty')}</p>
			{:else}
				<ul class="fewshot-list">
					{#each examples as ex (ex.id)}
						<li class="fewshot-item">
							<div class="row-main">
								<div class="badges">
									<span class="badge state-{ex.state}">
										{ex.state === 'active' ? $t('fewshot.state_active') : $t('fewshot.state_stale')}
									</span>
									{#if ex.pinned}
										<span class="badge pinned">{$t('fewshot.pinned')}</span>
									{/if}
									<span class="badge mode">{ex.mode}</span>
								</div>
								<button class="query" onclick={() => toggleExpand(ex.id)}>
									{expandedId === ex.id ? ex.query : truncate(ex.query)}
								</button>
								<div class="metrics">
									<span>{$t('fewshot.fitness')}: {ex.fitness.toFixed(2)}</span>
									<span
										>{$t('fewshot.quality')}: {ex.quality_score !== null
											? ex.quality_score.toFixed(2)
											: '-'}</span
									>
									<span>{$t('fewshot.usage', { use_count: ex.use_count, helpful: ex.helpful, harmful: ex.harmful })}</span>
									<span>{$t('fewshot.last_used')}: {formatLastUsed(ex.last_used_at)}</span>
								</div>
							</div>
							{#if expandedId === ex.id}
								<div class="response-block">
									<span class="response-label">{$t('fewshot.response_label')}</span>
									<p class="response-text">{ex.response}</p>
								</div>
							{/if}
							<div class="actions">
								<button
									class="action-btn"
									disabled={actingId === ex.id}
									onclick={() => handleAction(ex.id, ex.pinned ? 'unpin' : 'pin')}
								>
									{ex.pinned ? $t('fewshot.unpin') : $t('fewshot.pin')}
								</button>
								<button
									class="action-btn danger"
									disabled={actingId === ex.id}
									onclick={() => handleAction(ex.id, 'archive')}
								>
									{$t('fewshot.archive')}
								</button>
							</div>
						</li>
					{/each}
				</ul>
			{/if}
		</section>

		<section class="fewshot-section">
			<h2 class="section-title">{$t('fewshot.section_archived')}</h2>
			{#if archived.length === 0}
				<p class="empty">{$t('fewshot.archived_empty')}</p>
			{:else}
				<ul class="fewshot-list">
					{#each archived as ex (ex.id)}
						<li class="fewshot-item">
							<div class="row-main">
								<div class="badges">
									<span class="badge state-archived">{$t('fewshot.state_archived')}</span>
									<span class="badge mode">{ex.mode}</span>
								</div>
								<button class="query" onclick={() => toggleExpand(ex.id)}>
									{expandedId === ex.id ? ex.query : truncate(ex.query)}
								</button>
								<div class="metrics">
									<span>{$t('fewshot.fitness')}: {ex.fitness.toFixed(2)}</span>
									<span>{$t('fewshot.last_used')}: {formatLastUsed(ex.last_used_at)}</span>
								</div>
							</div>
							{#if expandedId === ex.id}
								<div class="response-block">
									<span class="response-label">{$t('fewshot.response_label')}</span>
									<p class="response-text">{ex.response}</p>
								</div>
							{/if}
							<div class="actions">
								<button
									class="action-btn"
									disabled={actingId === ex.id}
									onclick={() => handleAction(ex.id, 'restore')}
								>
									{$t('fewshot.restore')}
								</button>
							</div>
						</li>
					{/each}
				</ul>
			{/if}
		</section>
	{/if}
</div>

<style>
	.fewshot-curator {
		padding: 16px;
	}
	.header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 16px;
		flex-wrap: wrap;
	}
	.filters {
		display: flex;
		gap: 6px;
	}
	.filter-btn {
		padding: 4px 12px;
		font-size: 0.9rem;
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		color: var(--text-primary);
		cursor: pointer;
	}
	.filter-btn.active {
		background-color: var(--accent);
		color: var(--text-on-accent);
		border-color: var(--accent);
	}
	.pool-summary {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}
	.hint {
		color: var(--text-secondary);
	}
	.hint.delegated {
		background-color: var(--bg-secondary);
		border: 0.5px solid var(--card-border);
		border-radius: var(--border-radius);
		padding: 10px 12px;
	}
	.error {
		color: var(--color-error);
		font-size: 0.95rem;
	}
	.empty {
		color: var(--text-secondary);
		font-style: italic;
	}
	.fewshot-section {
		margin-bottom: 20px;
	}
	.section-title {
		font-size: 1rem;
		font-weight: 600;
		color: var(--text-primary);
		margin: 0 0 8px;
	}
	.fewshot-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.fewshot-item {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 10px 12px;
		background-color: var(--bg-secondary);
		border: 0.5px solid var(--card-border);
		border-radius: 10px;
	}
	.row-main {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.badges {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	.badge {
		font-size: 0.75rem;
		padding: 1px 8px;
		border-radius: 10px;
		border: 1px solid var(--border);
		color: var(--text-secondary);
	}
	.badge.state-active {
		color: var(--success, #22c55e);
		border-color: var(--success, #22c55e);
	}
	.badge.state-stale,
	.badge.state-archived {
		color: var(--warning, #f59e0b);
		border-color: var(--warning, #f59e0b);
	}
	.badge.pinned {
		color: var(--accent);
		border-color: var(--accent);
	}
	.badge.mode {
		opacity: 0.8;
	}
	.query {
		background: none;
		border: none;
		padding: 0;
		text-align: left;
		cursor: pointer;
		font-family: inherit;
		font-size: 0.95rem;
		color: var(--text-primary);
	}
	.query:hover {
		text-decoration: underline;
	}
	.metrics {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		font-size: 0.8rem;
		color: var(--text-secondary);
		opacity: 0.8;
	}
	.response-block {
		background-color: var(--bg-primary);
		border-radius: var(--border-radius);
		padding: 8px 10px;
	}
	.response-label {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--text-secondary);
	}
	.response-text {
		margin: 4px 0 0;
		font-size: 0.9rem;
		color: var(--text-primary);
		white-space: pre-wrap;
	}
	.actions {
		display: flex;
		gap: 6px;
	}
	.action-btn {
		padding: 4px 10px;
		font-size: 0.85rem;
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		color: var(--text-primary);
		cursor: pointer;
	}
	.action-btn:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.action-btn.danger {
		background-color: var(--color-error);
		color: var(--text-on-accent);
		border: 1px solid var(--color-error);
	}
	.action-btn:not(:disabled):hover {
		opacity: 0.85;
	}
	@media (max-width: 700px) {
		.metrics {
			flex-direction: column;
			gap: 2px;
		}
	}
</style>
