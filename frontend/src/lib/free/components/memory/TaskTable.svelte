<script lang="ts">
	import { t } from '$lib/i18n';
	import type { TaskFactInfo } from '$lib/free/api';

	let { tasks }: { tasks: TaskFactInfo[] } = $props();

	function statusLabel(status: string): string {
		switch (status) {
			case 'open':
				return $t('memory_inspector.task_status_open');
			case 'in_progress':
				return $t('memory_inspector.task_status_in_progress');
			case 'done':
				return $t('memory_inspector.task_status_done');
			case 'failed':
				return $t('memory_inspector.task_status_failed');
			default:
				return status;
		}
	}

	function fmtTs(ts: number): string {
		if (!ts) return '-';
		try {
			return new Date(ts * 1000).toLocaleString();
		} catch {
			return '-';
		}
	}
</script>

{#if tasks.length === 0}
	<p class="empty">{$t('memory_inspector.empty_tasks')}</p>
{:else}
	<div class="table-wrap">
		<table class="task-table">
			<thead>
				<tr>
					<th>{$t('memory_inspector.col_task_id')}</th>
					<th>{$t('memory_inspector.col_title')}</th>
					<th>{$t('memory_inspector.col_status')}</th>
					<th>{$t('memory_inspector.col_salience')}</th>
					<th>{$t('memory_inspector.col_depends_on')}</th>
					<th>{$t('memory_inspector.col_created')}</th>
				</tr>
			</thead>
			<tbody>
				{#each tasks as task (task.fact_id)}
					<tr class={`status-${task.status}`}>
						<td class="cell-mono">{task.task_id}</td>
						<td class="cell-title" title={task.description}>{task.title}</td>
						<td>
							<span class={`status-badge status-badge-${task.status}`}>
								{statusLabel(task.status)}
							</span>
						</td>
						<td class="num">{task.salience.toFixed(2)}</td>
						<td class="cell-deps">{task.depends_on.join(', ') || '-'}</td>
						<td class="cell-ts">{fmtTs(task.created_at)}</td>
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
	.task-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}
	.task-table th,
	.task-table td {
		padding: 6px 10px;
		border-bottom: 0.5px solid var(--input-border);
		text-align: left;
		vertical-align: top;
	}
	.task-table th {
		background: var(--bg-tertiary, var(--bg-secondary));
		color: var(--text-secondary);
		font-weight: 600;
	}
	.cell-mono {
		font-family: ui-monospace, monospace;
		font-size: 12px;
	}
	.cell-title {
		max-width: 360px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.cell-deps {
		max-width: 200px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 12px;
		color: var(--text-secondary);
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
	.status-badge {
		display: inline-block;
		padding: 2px 8px;
		border-radius: 12px;
		font-size: 11px;
		font-weight: 600;
	}
	.status-badge-open {
		background: color-mix(in srgb, var(--accent) 25%, transparent);
		color: var(--text-primary);
	}
	.status-badge-in_progress {
		background: color-mix(in srgb, #f59e0b 30%, transparent);
		color: var(--text-primary);
	}
	.status-badge-done {
		background: color-mix(in srgb, #22c55e 30%, transparent);
		color: var(--text-primary);
	}
	.status-badge-failed {
		background: color-mix(in srgb, #ef4444 30%, transparent);
		color: var(--text-primary);
	}
</style>
