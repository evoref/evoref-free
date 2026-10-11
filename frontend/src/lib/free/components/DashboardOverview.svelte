<script lang="ts">
	/**
	 * ダッシュボード最上段の全体の状態: 正常 / 注意 / 異常 / 確認中を文言と記号で示し
	 * (色だけに頼らない)、注意・異常があれば何が起きていてどうなるかを平文で並べる。
	 */
	import { t } from '$lib/i18n';
	import { serverState, systemHealth } from '$lib/free/stores/server';
	import type { DashboardRagStats } from '$lib/free/api';

	interface DashboardOverviewProps {
		ragStats: DashboardRagStats;
		/** ダッシュボードの API の一部が取れなかった */
		fetchError?: boolean;
	}

	let { ragStats, fetchError = false }: DashboardOverviewProps = $props();

	type Level = 'ok' | 'warn' | 'error' | 'unknown';
	interface Issue {
		level: 'warn' | 'error';
		text: string;
		detail?: string;
	}

	const MARKS: Record<Level, string> = { ok: '✓', warn: '!', error: '✕', unknown: '…' };

	let data = $derived($serverState.dataHealth);
	let health = $derived($systemHealth);

	let issues = $derived.by(() => {
		const list: Issue[] = [];
		if (data?.readonly) {
			list.push({
				level: 'error',
				text: $t('dashboard.overview.readonly'),
				detail: data.reason ?? undefined
			});
		}
		if (fetchError) list.push({ level: 'error', text: $t('dashboard.fetch_error') });
		if (ragStats.embeddingDimMismatch) {
			list.push({
				level: 'warn',
				text: $t('dashboard.embedding_dim_mismatch', {
					stored: ragStats.embeddingDimStored ?? '?',
					current: ragStats.embeddingDimCurrent
				})
			});
		}
		const notes = (data?.warnings.length ?? 0) + (data?.degraded.length ?? 0);
		if (notes > 0) {
			list.push({ level: 'warn', text: $t('dashboard.overview.data_warnings', { count: notes }) });
		}
		if ((health?.liveness.length ?? 0) > 0) {
			list.push({
				level: 'warn',
				text: $t('dashboard.overview.liveness', { count: health?.liveness.length ?? 0 })
			});
		}
		return list;
	});

	let level: Level = $derived(
		issues.some((i) => i.level === 'error')
			? 'error'
			: issues.length > 0
				? 'warn'
				: !data && !health
					? 'unknown'
					: 'ok'
	);
</script>

<section class="overview" data-level={level} aria-labelledby="dashboard-overview-title">
	<h2 id="dashboard-overview-title" class="visually-hidden">{$t('dashboard.overview.title')}</h2>
	<p class="headline" role="status">
		<span class="mark" aria-hidden="true">{MARKS[level]}</span>
		<span class="level">{$t(`dashboard.overview.${level}`)}</span>
		{#if level === 'ok' || level === 'unknown'}
			<span class="hint">{$t(`dashboard.overview.${level}_hint`)}</span>
		{/if}
	</p>
	{#if issues.length > 0}
		<ul class="issues">
			{#each issues as issue, i (i)}
				<li class="issue" data-level={issue.level}>
					<span class="issue-mark" aria-hidden="true">{MARKS[issue.level]}</span>
					<span>
						<span class="visually-hidden">{`${$t(`dashboard.overview.${issue.level}`)}: `}</span>{issue.text}
						{#if issue.detail}
							<span class="detail">{issue.detail}</span>
						{/if}
					</span>
				</li>
			{/each}
		</ul>
	{/if}
</section>

<style>
	.overview {
		border: 1px solid var(--card-border);
		border-radius: var(--border-radius);
		background: var(--card-bg);
		padding: 12px 16px;
		margin-bottom: 12px;
	}
	.overview[data-level='warn'] {
		border-color: var(--color-warning);
	}
	.overview[data-level='error'] {
		border-color: var(--color-error);
		border-width: 2px;
	}
	.headline {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 8px;
		margin: 0;
	}
	.mark {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.4em;
		height: 1.4em;
		border: 1px solid currentColor;
		border-radius: 50%;
		font-size: 0.875rem;
		font-weight: 700;
		color: var(--text-primary);
	}
	.level {
		font-size: 1rem;
		font-weight: 700;
		color: var(--text-primary);
	}
	.hint {
		font-size: 0.8125rem;
		color: var(--text-secondary);
	}
	.issues {
		list-style: none;
		margin: 10px 0 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.issue {
		display: flex;
		gap: 8px;
		font-size: 0.875rem;
		color: var(--text-primary);
	}
	.issue-mark {
		flex-shrink: 0;
		width: 1em;
		font-weight: 700;
		text-align: center;
	}
	.issue[data-level='error'] .issue-mark {
		color: var(--color-error);
	}
	.detail {
		display: block;
		margin-top: 2px;
		font-size: 0.75rem;
		color: var(--text-secondary);
		overflow-wrap: anywhere;
	}
	.visually-hidden {
		position: absolute;
		width: 1px;
		height: 1px;
		margin: -1px;
		padding: 0;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
		border: 0;
	}
</style>
