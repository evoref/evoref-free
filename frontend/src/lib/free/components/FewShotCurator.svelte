<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import { listFewshot, actOnFewshot, ApiError } from '$lib/free/api';
	import type { FewShotAction, FewShotExample } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatDate } from '$lib/free/utils/format';
	import { proActive } from '$lib/free/stores/server';
	import Badge from './ui/Badge.svelte';
	import Button from './ui/Button.svelte';
	import Disclosure from './ui/Disclosure.svelte';
	import EmptyState from './ui/EmptyState.svelte';
	import Loading from './ui/Loading.svelte';

	type ModeFilter = 'all' | 'chat' | 'create';

	const FILTERS: [ModeFilter, string][] = [
		['all', 'fewshot.filter_all'],
		['chat', 'fewshot.filter_chat'],
		['create', 'fewshot.filter_create']
	];

	/** 一覧で見せる質問の要約の最大文字数 */
	const QUERY_PREVIEW_CHARS = 80;

	let examples = $state<FewShotExample[]>([]);
	let archived = $state<FewShotExample[]>([]);
	let modeFilter = $state<ModeFilter>('all');
	let loading = $state(true);
	let errorMessage = $state('');
	let expandedId = $state<string | null>(null);
	let actingId = $state<string | null>(null);

	onMount(() => {
		refresh();
	});

	async function refresh() {
		loading = true;
		errorMessage = '';
		try {
			const mode = modeFilter === 'all' ? undefined : modeFilter;
			const result = await listFewshot(mode);
			examples = result.examples;
			archived = result.archived;
		} catch (e) {
			errorMessage = e instanceof ApiError && e.i18nKey ? $t(e.i18nKey) : $t('fewshot.list_failed');
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

	function truncate(text: string, max = QUERY_PREVIEW_CHARS): string {
		if (text.length <= max) return text;
		return `${text.slice(0, max)}...`;
	}

	function formatLastUsed(iso: string): string {
		if (!iso) return $t('fewshot.never_used');
		return formatDate(iso);
	}
</script>

<!-- 1 件分の行。要約と状態を主に、数値の指標は展開した詳細にだけ出す -->
{#snippet row(ex: FewShotExample, isArchived: boolean)}
	{@const open = expandedId === ex.id}
	<li class="fewshot-item">
		<div class="row-main">
			<button
				class="summary"
				aria-expanded={open}
				aria-controls="fewshot-detail-{ex.id}"
				onclick={() => toggleExpand(ex.id)}
			>
				<span class="query">{truncate(ex.query)}</span>
				<span class="toggle-label">{open ? $t('fewshot.hide_full') : $t('fewshot.show_full')}</span>
			</button>
			<span class="badges">
				{#if isArchived}
					<Badge tone="muted">{$t('fewshot.state_archived')}</Badge>
				{:else}
					{#if ex.state === 'stale'}
						<Badge tone="muted">{$t('fewshot.state_stale')}</Badge>
					{:else}
						<Badge tone="strong">{$t('fewshot.state_active')}</Badge>
					{/if}
					{#if ex.pinned}
						<Badge>{$t('fewshot.pinned')}</Badge>
					{/if}
				{/if}
			</span>
			<span class="row-actions">
				{#if isArchived}
					<Button size="sm" disabled={actingId === ex.id} onclick={() => handleAction(ex.id, 'restore')}>
						{$t('fewshot.restore')}
					</Button>
				{:else}
					<Button
						size="sm"
						disabled={actingId === ex.id}
						onclick={() => handleAction(ex.id, ex.pinned ? 'unpin' : 'pin')}
					>
						{ex.pinned ? $t('fewshot.unpin') : $t('fewshot.pin')}
					</Button>
					<Button
						variant="ghost"
						size="sm"
						disabled={actingId === ex.id}
						onclick={() => handleAction(ex.id, 'archive')}
					>
						{$t('fewshot.archive')}
					</Button>
				{/if}
			</span>
		</div>
		{#if open}
			<dl class="detail" id="fewshot-detail-{ex.id}">
				<dt>{$t('fewshot.query_label')}</dt>
				<dd class="pre">{ex.query}</dd>
				<dt>{$t('fewshot.response_label')}</dt>
				<dd class="pre">{ex.response}</dd>
				<dt>{$t('fewshot.mode_label')}</dt>
				<dd>{ex.mode}</dd>
				<dt>{$t('fewshot.fitness')}</dt>
				<dd>{ex.fitness.toFixed(2)}</dd>
				<dt>{$t('fewshot.quality')}</dt>
				<dd>{ex.quality_score !== null ? ex.quality_score.toFixed(2) : '-'}</dd>
				<dt>{$t('fewshot.usage_label')}</dt>
				<dd>
					{$t('fewshot.usage', { use_count: ex.use_count, helpful: ex.helpful, harmful: ex.harmful })}
				</dd>
				<dt>{$t('fewshot.last_used')}</dt>
				<dd>{formatLastUsed(ex.last_used_at)}</dd>
			</dl>
		{/if}
	</li>
{/snippet}

<div class="fewshot-curator">
	{#if $proActive}
		<!-- チャット / クリエイトの絞り込みは、クリエイトモードがある Pro だけで意味を持つ -->
		<div class="filters" role="group" aria-label={$t('fewshot.filter_label')}>
			{#each FILTERS as [value, key] (value)}
				<button
					class="filter-btn"
					class:active={modeFilter === value}
					aria-pressed={modeFilter === value}
					onclick={() => handleFilterChange(value)}
				>
					{$t(key)}
				</button>
			{/each}
		</div>
	{/if}

	{#if loading}
		<Loading />
	{:else if errorMessage}
		<p class="error" role="alert">{errorMessage}</p>
	{:else}
		<section class="fewshot-section">
			<h2 class="section-title">{$t('fewshot.section_pool', { count: examples.length })}</h2>
			{#if examples.length === 0}
				<EmptyState message={$t('fewshot.list_empty')} />
			{:else}
				<ul class="fewshot-list">
					{#each examples as ex (ex.id)}
						{@render row(ex, false)}
					{/each}
				</ul>
			{/if}
		</section>

		<Disclosure label={$t('fewshot.section_archived', { count: archived.length })}>
			{#if archived.length === 0}
				<p class="muted">{$t('fewshot.archived_empty')}</p>
			{:else}
				<ul class="fewshot-list">
					{#each archived as ex (ex.id)}
						{@render row(ex, true)}
					{/each}
				</ul>
			{/if}
		</Disclosure>
	{/if}
</div>

<style>
	.filters {
		display: inline-flex;
		margin-bottom: var(--space-4);
		border: 1px solid var(--input-border);
		border-radius: var(--border-radius);
		overflow: hidden;
	}
	.filter-btn {
		padding: 4px 14px;
		font-size: var(--font-size-sm);
		font-family: inherit;
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
	}
	.filter-btn + .filter-btn {
		border-left: 1px solid var(--input-border);
	}
	.filter-btn.active {
		background-color: var(--accent);
		color: var(--text-on-accent);
		font-weight: 600;
	}
	.error {
		color: var(--color-error);
		font-weight: 600;
	}
	.muted {
		color: var(--text-secondary);
		font-size: var(--font-size-sm);
	}
	.fewshot-section {
		margin-bottom: var(--space-5);
	}
	.section-title {
		font-size: var(--font-size-base);
		font-weight: 600;
		color: var(--text-primary);
		margin: 0 0 var(--space-3);
	}
	.fewshot-list {
		list-style: none;
		padding: 0;
		margin: 0;
		border-top: 1px solid var(--border);
	}
	.fewshot-item {
		padding: var(--space-3) var(--space-1);
		border-bottom: 1px solid var(--border);
	}
	.row-main {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		flex-wrap: wrap;
	}
	.summary {
		flex: 1 1 280px;
		min-width: 0;
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		background: none;
		border: none;
		padding: 0;
		text-align: left;
		cursor: pointer;
		font-family: inherit;
		color: var(--text-primary);
	}
	.query {
		min-width: 0;
	}
	.summary:hover .query {
		text-decoration: underline;
	}
	.toggle-label {
		flex-shrink: 0;
		font-size: var(--font-size-xs);
		color: var(--text-muted);
	}
	.badges,
	.row-actions {
		display: inline-flex;
		align-items: center;
		gap: var(--space-1);
	}
	.detail {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: var(--space-1) var(--space-4);
		margin: var(--space-3) 0 0;
		padding: var(--space-3);
		border-left: 2px solid var(--border);
		font-size: var(--font-size-sm);
	}
	dt {
		color: var(--text-secondary);
	}
	dd {
		margin: 0;
		color: var(--text-primary);
	}
	.pre {
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
</style>
