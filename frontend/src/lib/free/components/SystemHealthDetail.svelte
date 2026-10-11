<script lang="ts">
	/**
	 * システムの詳細 (Pro のダッシュボードの「詳細」): データ点検の注記・効果の死活監視の内訳・
	 * リランカー・埋め込みの配置・学習の判定。正常/注意/異常の要約は DashboardOverview が出す。
	 */
	import { t } from '$lib/i18n';
	import { serverState, systemHealth } from '$lib/free/stores/server';
	import DashboardCard from './DashboardCard.svelte';
	import StatusRow from './StatusRow.svelte';
	import StatusSection from './StatusSection.svelte';

	let data = $derived($serverState.dataHealth);
	let health = $derived($systemHealth);

	let dataNotes = $derived([...(data?.warnings ?? []), ...(data?.degraded ?? [])]);
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
	<div class="health">
		<div>
			<StatusRow label={$t('dashboard.health.rerank')} value={rerankValue} />
			<StatusRow label={$t('dashboard.health.embed_placement')} value={embedValue} />
			<StatusRow label={$t('dashboard.health.learning_verdict')} value={learningValue} />
		</div>
		{#if dataNotes.length > 0}
			<StatusSection label={$t('dashboard.health.data')}>
				{#each dataNotes as note, i (i)}
					<p class="note">{note}</p>
				{/each}
			</StatusSection>
		{/if}
		{#if (health?.liveness.length ?? 0) > 0}
			<StatusSection label={$t('dashboard.health.liveness')}>
				{#each health?.liveness ?? [] as alert (alert.stage + alert.kind)}
					<p class="note">{alert.stage}: {alert.kind}</p>
				{/each}
			</StatusSection>
		{/if}
	</div>
</DashboardCard>

<style>
	.health {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.note {
		margin: 0;
		font-size: 0.75rem;
		color: var(--text-secondary);
		overflow-wrap: anywhere;
	}
</style>
