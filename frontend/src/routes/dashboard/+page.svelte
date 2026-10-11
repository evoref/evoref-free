<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import LearningStatus from '$lib/free/components/LearningStatus.svelte';
	import RAGStats from '$lib/free/components/RAGStats.svelte';
	import SystemHealthCard from '$lib/free/components/SystemHealthCard.svelte';
	import { onMount, onDestroy } from 'svelte';
	import type { Component } from 'svelte';
	import { isPro } from '$lib/edition';
	import { proActive } from '$lib/free/stores/server';
	import { getGrowth } from '$lib/free/api';
	import type { DashboardLearningData, DashboardRagStats, GrowthResponse } from '$lib/free/api';
	import {
		DEFAULT_LEARNING_DATA,
		DEFAULT_RAG_STATS,
		fetchFreeDashboardData
	} from '$lib/free/stores/dashboard';

	const REFRESH_INTERVAL_MS = 30_000;

	// エディション境界: route 層 (ここ) が Pro の組織ポリシーパックの状態パネルを
	// `import.meta.glob` で動的ロードする (Free の配下は $lib/pro を参照できない、docs/p_09 §8)。
	const policyLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/OrgPolicyPanel.svelte'
	);
	let OrgPolicyPanel: Component | null = $state(null);
	// モデル適合ラボの最後の評価 (docs/p_11 §7)。読むだけで、実行は CLI (`evoref modelfit run`)
	const modelFitLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/ModelFitPanel.svelte'
	);
	let ModelFitPanel: Component | null = $state(null);

	let learningData: DashboardLearningData = $state({ ...DEFAULT_LEARNING_DATA });
	let ragStats: DashboardRagStats = $state({ ...DEFAULT_RAG_STATS });
	let fetchError = $state(false);
	let growth: GrowthResponse | null = $state(null);
	let loaded = $state(false);

	let gridCols = $derived($layout.dashboard.grid_columns);

	async function refreshGrowth() {
		// Level 2 (成長) の欄。取れなければ欄を出さない (学習状態の表示は止めない)
		growth = await getGrowth().catch(() => null);
	}

	async function refreshDashboardData() {
		const [freeData] = await Promise.all([fetchFreeDashboardData(), refreshGrowth()]);
		learningData = freeData.learningData;
		ragStats = freeData.ragStats;
		fetchError = freeData.hasError;
	}

	let refreshTimer: ReturnType<typeof setInterval> | undefined;

	onMount(async () => {
		if (isPro) {
			const loaders = Object.values(policyLoaders);
			if (loaders.length > 0) OrgPolicyPanel = (await loaders[0]()).default;
			const fit = Object.values(modelFitLoaders);
			if (fit.length > 0) ModelFitPanel = (await fit[0]()).default;
		}
		await refreshDashboardData();
		loaded = true;
		refreshTimer = setInterval(refreshDashboardData, REFRESH_INTERVAL_MS);
	});

	onDestroy(() => {
		if (refreshTimer) clearInterval(refreshTimer);
	});
</script>

<PageLayout title={$t('sidebar.dashboard')}>
	{#if loaded}
		{#if fetchError}
			<p class="fetch-error">{$t('dashboard.fetch_error')}</p>
		{/if}
		<div class="dashboard-grid" style="grid-template-columns: repeat({gridCols}, 1fr)">
			<LearningStatus
				running={learningData.running}
				experienceCount={learningData.experience_count}
				newExperienceCount={learningData.new_experience_count}
				minExperiences={learningData.min_experiences}
				conditionsMet={learningData.conditions_met}
				level1BlockedReason={learningData.level1_blocked_reason}
				level1SecondsUntilIdle={learningData.level1_seconds_until_idle}
				lastLevel1Run={learningData.last_level1_run}
				isDisabled={learningData.is_disabled}
				lastLevel0Record={learningData.last_level0_record}
				experienceByMode={learningData.experience_by_mode}
				correctionRate={learningData.correction_rate}
				ragUsageRate={learningData.rag_usage_rate}
				prevCorrectionRate={learningData.prev_correction_rate}
				prevRagUsageRate={learningData.prev_rag_usage_rate}
				level1RunCount={learningData.level1_run_count}
				lastLevel1Results={learningData.last_level1_results}
				executedPhases={learningData.executed_phases}
				fitnessHistory={learningData.fitness_history}
				policyEvolverStatus={learningData.policy_evolver_status}
				{growth}
				onGrowthChange={refreshDashboardData}
			/>
			<RAGStats stats={ragStats} />
			<SystemHealthCard />
		</div>
		{#if $proActive && OrgPolicyPanel}
			{@const PolicyPanel = OrgPolicyPanel}
			<PolicyPanel />
		{/if}
		{#if $proActive && ModelFitPanel}
			{@const FitPanel = ModelFitPanel}
			<FitPanel />
		{/if}
	{:else}
		<div class="loading">{$t('common.loading')}</div>
	{/if}
</PageLayout>

<style>
	.dashboard-grid {
		display: grid;
		gap: 12px;
	}
	@media (max-width: 900px) {
		.dashboard-grid {
			grid-template-columns: 1fr !important;
		}
	}
	.fetch-error {
		color: var(--color-error, #ef4444);
		font-size: 0.875rem;
		padding: 8px 12px;
		margin-bottom: 8px;
		background-color: var(--bg-secondary);
		border-radius: var(--border-radius);
	}
	.loading {
		display: flex;
		justify-content: center;
		align-items: center;
		padding: 2rem;
		color: var(--text-secondary);
	}
</style>
