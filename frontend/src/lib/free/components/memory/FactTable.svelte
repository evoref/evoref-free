<script lang="ts">
	import { t } from '$lib/i18n';
	import type { FactInfo } from '$lib/free/api';

	let { facts, showFailureSig = false, showEvalMetric = false }: {
		facts: FactInfo[];
		showFailureSig?: boolean;
		showEvalMetric?: boolean;
	} = $props();

	function fmtTs(ts: number): string {
		if (!ts) return '-';
		try {
			return new Date(ts * 1000).toLocaleString();
		} catch {
			return '-';
		}
	}

	function fmtConfidence(c: number): string {
		return c.toFixed(2);
	}

	function fmtEval(metric: Record<string, number> | null): string {
		if (!metric) return '-';
		return Object.entries(metric)
			.map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
			.join(', ');
	}
</script>

{#if facts.length === 0}
	<p class="empty">{$t('memory_inspector.empty_facts')}</p>
{:else}
	<div class="table-wrap">
		<table class="fact-table">
			<thead>
				<tr>
					<th>{$t('memory_inspector.col_subject')}</th>
					<th>{$t('memory_inspector.col_predicate')}</th>
					<th>{$t('memory_inspector.col_object')}</th>
					<th>{$t('memory_inspector.col_type')}</th>
					<th>{$t('memory_inspector.col_mode')}</th>
					<th>{$t('memory_inspector.col_confidence')}</th>
					<th>{$t('memory_inspector.col_pinned')}</th>
					{#if showFailureSig}
						<th>{$t('memory_inspector.col_failure_signature')}</th>
					{/if}
					{#if showEvalMetric}
						<th>{$t('memory_inspector.col_eval_metric')}</th>
						<th>{$t('memory_inspector.col_auto_evolved')}</th>
					{/if}
					<th>{$t('memory_inspector.col_created')}</th>
					<th>{$t('memory_inspector.col_access_count')}</th>
				</tr>
			</thead>
			<tbody>
				{#each facts as fact (fact.id)}
					<tr class:superseded={fact.superseded_by !== null}>
						<td class="cell-subject" title={fact.subject}>{fact.subject}</td>
						<td class="cell-predicate">{fact.predicate}</td>
						<td class="cell-object" title={fact.object}>{fact.object}</td>
						<td>{fact.type}</td>
						<td>{fact.mode_origin}</td>
						<td class="num">{fmtConfidence(fact.confidence)}</td>
						<td>{fact.pinned ? '★' : ''}</td>
						{#if showFailureSig}
							<td class="cell-mono">{fact.failure_signature ?? '-'}</td>
						{/if}
						{#if showEvalMetric}
							<td class="cell-mono">{fmtEval(fact.eval_metric)}</td>
							<td>{fact.auto_evolved ? '✓' : ''}</td>
						{/if}
						<td class="cell-ts">{fmtTs(fact.created_at)}</td>
						<td class="num">{fact.access_count}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}

<style>
	.empty {
		padding: 12px;
		color: var(--text-secondary);
		text-align: center;
	}
	.table-wrap {
		overflow-x: auto;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--bg-secondary);
	}
	.fact-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}
	.fact-table th,
	.fact-table td {
		padding: 6px 10px;
		border-bottom: 0.5px solid var(--input-border);
		text-align: left;
		vertical-align: top;
	}
	.fact-table th {
		background: var(--bg-tertiary, var(--bg-secondary));
		color: var(--text-secondary);
		font-weight: 600;
		position: sticky;
		top: 0;
	}
	.cell-subject,
	.cell-object {
		max-width: 320px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.cell-mono {
		font-family: ui-monospace, monospace;
		font-size: 12px;
	}
	.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.cell-ts {
		white-space: nowrap;
		color: var(--text-secondary);
		font-size: 12px;
	}
	tr.superseded {
		opacity: 0.55;
	}
</style>
