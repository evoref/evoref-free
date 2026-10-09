<script lang="ts">
	import { t } from '$lib/i18n';
	import { serverState, systemHealth } from '$lib/free/stores/server';
	import DashboardCard from './DashboardCard.svelte';
	import StatusRow from './StatusRow.svelte';

	let data = $derived($serverState.dataHealth);
	let health = $derived($systemHealth);

	let dataNotes = $derived([...(data?.warnings ?? []), ...(data?.degraded ?? [])]);
	let dataValue = $derived(
		!data
			? '-'
			: data.readonly
			? $t('dashboard.health.data_readonly')
			: dataNotes.length > 0
				? $t('dashboard.health.data_warnings', { count: dataNotes.length })
				: $t('dashboard.health.ok')
	);
	let livenessValue = $derived(
		!health
			? '-'
			: health.liveness.length > 0
				? $t('dashboard.health.liveness_alerts', { count: health.liveness.length })
				: $t('dashboard.health.ok')
	);
	let rerankValue = $derived(
		health?.rerank
			? health.rerank.enabled
				? $t('dashboard.health.rerank_on')
				: `${health.rerank.mode} (${health.rerank.reason || '-'})`
			: '-'
	);
	let embedValue = $derived(
		health?.embedPlacement
			? `${health.embedPlacement.placement} (${health.embedPlacement.setting})`
			: '-'
	);
	let learningValue = $derived(health?.learningHealth?.verdict ?? '-');
</script>

<DashboardCard title={$t('dashboard.health.title')}>
	<div class="health-rows">
		<StatusRow
			active={!!data && !data.readonly && dataNotes.length === 0}
			label={$t('dashboard.health.data')}
			value={dataValue}
			ariaLabel={dataValue}
		/>
		{#if data?.readonly && data.reason}
			<p class="note" role="alert">{data.reason}</p>
		{/if}
		{#each dataNotes as note, i (i)}
			<p class="note">{note}</p>
		{/each}
		<StatusRow
			active={!!health && health.liveness.length === 0}
			label={$t('dashboard.health.liveness')}
			value={livenessValue}
			ariaLabel={livenessValue}
		/>
		{#each health?.liveness ?? [] as alert (alert.stage + alert.kind)}
			<p class="note">{alert.stage}: {alert.kind}</p>
		{/each}
		<StatusRow
			active={!!health?.rerank?.enabled}
			label={$t('dashboard.health.rerank')}
			value={rerankValue}
			ariaLabel={rerankValue}
		/>
		<StatusRow
			active={!!health?.embedPlacement}
			label={$t('dashboard.health.embed_placement')}
			value={embedValue}
			ariaLabel={embedValue}
		/>
		<StatusRow
			active={health?.learningHealth?.verdict === 'continue'}
			label={$t('dashboard.health.learning_verdict')}
			value={learningValue}
			ariaLabel={learningValue}
		/>
	</div>
</DashboardCard>

<style>
	.health-rows {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.note {
		margin: 0 0 0 20px;
		font-size: 12px;
		color: var(--text-secondary);
		overflow-wrap: anywhere;
	}
</style>
