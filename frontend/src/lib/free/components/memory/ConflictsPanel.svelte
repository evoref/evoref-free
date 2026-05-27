<script lang="ts">
	import { t } from '$lib/i18n';
	import type {
		ConflictGroupInfo,
		ConflictAction,
		FactInfo
	} from '$lib/free/api';

	let {
		pending = [],
		history = [],
		onResolve
	}: {
		pending?: ConflictGroupInfo[];
		history?: ConflictGroupInfo[];
		onResolve: (
			group: ConflictGroupInfo,
			action: ConflictAction,
			mergedObject?: string
		) => Promise<void>;
	} = $props();

	let mergeInputs = $state<Record<string, string>>({});
	let busyKey = $state<string | null>(null);

	function groupKey(g: ConflictGroupInfo): string {
		return `${g.scope}|${g.subject}|${g.predicate}|${g.type}`;
	}

	function fmtTs(ts: number | null | undefined): string {
		if (!ts) return '-';
		try {
			return new Date(ts * 1000).toLocaleString();
		} catch {
			return '-';
		}
	}

	function findFact(group: ConflictGroupInfo, id: string | null): FactInfo | null {
		if (!id) return null;
		return group.facts.find((f) => f.id === id) ?? null;
	}

	function olderId(group: ConflictGroupInfo): string | null {
		if (group.facts.length === 0) return null;
		const sorted = [...group.facts].sort((a, b) => a.created_at - b.created_at);
		return sorted[0].id;
	}

	function newerId(group: ConflictGroupInfo): string | null {
		return group.winner_id;
	}

	async function handleAction(group: ConflictGroupInfo, action: ConflictAction) {
		const key = groupKey(group);
		if (busyKey) return;
		busyKey = key;
		try {
			if (action === 'merge') {
				const merged = (mergeInputs[key] ?? '').trim();
				if (!merged) {
					busyKey = null;
					return;
				}
				await onResolve(group, action, merged);
			} else if (action === 'keep_old') {
				// keep_old: 古い fact を winner にする
				const old = olderId(group);
				const others = group.facts
					.map((f) => f.id)
					.filter((id) => id !== old);
				await onResolve(
					{ ...group, winner_id: old, loser_ids: others },
					action
				);
			} else {
				// keep_new: 既定の winner_id をそのまま使う
				await onResolve(group, action);
			}
		} finally {
			busyKey = null;
		}
	}
</script>

<section class="conflicts">
	<h2 class="section-heading">{$t('memory_inspector.conflicts_pending_heading')}</h2>
	{#if pending.length === 0}
		<p class="empty">{$t('memory_inspector.conflicts_no_pending')}</p>
	{:else}
		{#each pending as group (groupKey(group))}
			{@const key = groupKey(group)}
			{@const oldFact = findFact(group, olderId(group))}
			{@const newFact = findFact(group, newerId(group))}
			<article class="conflict-card">
				<header class="card-head">
					<div class="head-line">
						<span class="badge">{group.type}</span>
						<span class="subject">{group.subject}</span>
						<span class="predicate">{group.predicate}</span>
					</div>
					<div class="head-meta">
						{$t('memory_inspector.conflicts_facts_count', { count: group.facts.length })}
					</div>
				</header>

				<div class="fact-pair">
					<div class="fact-col" class:dim={false}>
						<div class="fact-label">
							{$t('memory_inspector.conflicts_old_label')}
						</div>
						<div class="fact-object">{oldFact?.object ?? '-'}</div>
						<div class="fact-meta">
							<span>conf {oldFact?.confidence.toFixed(2) ?? '-'}</span>
							<span>{fmtTs(oldFact?.created_at ?? null)}</span>
							{#if oldFact?.pinned}<span class="pin">★</span>{/if}
						</div>
					</div>
					<div class="fact-col">
						<div class="fact-label">
							{$t('memory_inspector.conflicts_new_label')}
						</div>
						<div class="fact-object">{newFact?.object ?? '-'}</div>
						<div class="fact-meta">
							<span>conf {newFact?.confidence.toFixed(2) ?? '-'}</span>
							<span>{fmtTs(newFact?.created_at ?? null)}</span>
							{#if newFact?.pinned}<span class="pin">★</span>{/if}
						</div>
					</div>
				</div>

				<div class="merge-row">
					<input
						type="text"
						class="merge-input"
						placeholder={$t('memory_inspector.conflicts_merge_placeholder')}
						bind:value={mergeInputs[key]}
					/>
				</div>

				<div class="action-row">
					<button
						type="button"
						class="btn"
						disabled={busyKey === key}
						onclick={() => handleAction(group, 'keep_old')}
					>
						{$t('memory_inspector.conflicts_action_keep_old')}
					</button>
					<button
						type="button"
						class="btn"
						disabled={busyKey === key}
						onclick={() => handleAction(group, 'keep_new')}
					>
						{$t('memory_inspector.conflicts_action_keep_new')}
					</button>
					<button
						type="button"
						class="btn btn-primary"
						disabled={busyKey === key || !(mergeInputs[key] ?? '').trim()}
						onclick={() => handleAction(group, 'merge')}
					>
						{$t('memory_inspector.conflicts_action_merge')}
					</button>
				</div>
			</article>
		{/each}
	{/if}

	<h2 class="section-heading timeline-heading">
		{$t('memory_inspector.conflicts_timeline_heading')}
	</h2>
	{#if history.length === 0}
		<p class="empty">{$t('memory_inspector.conflicts_no_history')}</p>
	{:else}
		<ol class="timeline">
			{#each history as entry, i (entry.detected_at ?? i)}
				<li class="timeline-entry">
					<div class="timeline-when">{fmtTs(entry.detected_at ?? null)}</div>
					<div class="timeline-body">
						<div class="timeline-head">
							<span class="badge">{entry.type}</span>
							<span class="subject">{entry.subject}</span>
							<span class="predicate">{entry.predicate}</span>
							{#if entry.reason}
								<span class="reason">{entry.reason}</span>
							{/if}
						</div>
						{#if entry.facts.length > 0}
							<div class="timeline-facts">
								{#each entry.facts as f (f.id)}
									<span
										class="chip"
										class:winner={f.id === entry.winner_id}
										class:auto={f.auto_evolved}
										title={`${f.id} conf=${f.confidence.toFixed(2)}`}
									>
										{f.object}
									</span>
								{/each}
							</div>
						{/if}
					</div>
				</li>
			{/each}
		</ol>
	{/if}
</section>

<style>
	.conflicts {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.section-heading {
		font-size: 14px;
		font-weight: 600;
		color: var(--text-secondary);
		margin: 0;
	}
	.timeline-heading {
		margin-top: 16px;
	}
	.empty {
		padding: 12px;
		color: var(--text-secondary);
		text-align: center;
		border: 0.5px dashed var(--input-border);
		border-radius: 6px;
		margin: 0;
	}
	.conflict-card {
		border: 0.5px solid var(--input-border);
		border-radius: 8px;
		background: var(--bg-secondary);
		padding: 12px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.card-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
	}
	.head-line {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	.head-meta {
		font-size: 12px;
		color: var(--text-secondary);
	}
	.badge {
		padding: 2px 8px;
		border-radius: 10px;
		background: color-mix(in srgb, var(--accent) 18%, transparent);
		color: var(--text-primary);
		font-size: 11px;
		font-weight: 600;
	}
	.subject {
		font-weight: 600;
		color: var(--text-primary);
		font-size: 13px;
	}
	.predicate {
		color: var(--text-secondary);
		font-size: 12px;
	}
	.fact-pair {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
	}
	.fact-col {
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		padding: 8px 10px;
		background: var(--control-bg);
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.fact-label {
		font-size: 11px;
		text-transform: uppercase;
		color: var(--text-secondary);
		letter-spacing: 0.04em;
	}
	.fact-object {
		font-size: 13px;
		color: var(--text-primary);
		word-break: break-word;
	}
	.fact-meta {
		display: flex;
		gap: 8px;
		font-size: 11px;
		color: var(--text-secondary);
	}
	.pin {
		color: #f59e0b;
	}
	.merge-row {
		display: flex;
	}
	.merge-input {
		flex: 1;
		padding: 6px 10px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 13px;
		font-family: inherit;
	}
	.action-row {
		display: flex;
		gap: 8px;
		justify-content: flex-end;
		flex-wrap: wrap;
	}
	.btn {
		padding: 6px 12px;
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		cursor: pointer;
		font-size: 12px;
	}
	.btn:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 12%, var(--control-bg));
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn-primary {
		background: var(--accent);
		color: var(--text-on-accent, #fff);
		border-color: var(--accent);
	}
	.timeline {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.timeline-entry {
		display: grid;
		grid-template-columns: 160px 1fr;
		gap: 12px;
		padding: 8px 10px;
		border: 0.5px solid var(--input-border);
		border-left: 3px solid var(--accent);
		border-radius: 6px;
		background: var(--bg-secondary);
	}
	.timeline-when {
		font-size: 11px;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
	}
	.timeline-body {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 0;
	}
	.timeline-head {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.reason {
		font-size: 11px;
		color: var(--text-secondary);
		font-family: ui-monospace, monospace;
	}
	.timeline-facts {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	.chip {
		padding: 2px 8px;
		border-radius: 10px;
		background: var(--control-bg);
		color: var(--text-secondary);
		font-size: 11px;
		border: 0.5px solid var(--input-border);
		max-width: 280px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.chip.winner {
		color: var(--text-primary);
		border-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 12%, var(--control-bg));
	}
	.chip.auto::before {
		content: '⚙ ';
	}
</style>
