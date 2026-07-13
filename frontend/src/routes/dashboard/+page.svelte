<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import { isPro } from '$lib/edition';
	import { goto } from '$app/navigation';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import type { LoraTarget, DashboardLearningData, DashboardRagStats } from '$lib/free/api';
	import { getLoraVersions, rollbackLora } from '$lib/free/api';
	import {
		type ProComponentMap,
		type DashboardLoraTargets,
		type ImprovementSeries,
		DEFAULT_LEARNING_DATA,
		DEFAULT_RAG_STATS,
		loadProComponents,
		fetchDashboardData,
		mapLoraTarget,
		emptyLoraSeries
	} from '$lib/pro/stores/dashboard';

	// Pro ガード: Free 版ではトップにリダイレクト
	if (!isPro) {
		goto('/');
	}

	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/{LearningStatus,LoRAVersions,PerformanceChart,RAGStats}.svelte'
	);

	let components: ProComponentMap | null = $state(null);
	let learningData: DashboardLearningData = $state({ ...DEFAULT_LEARNING_DATA });
	let loraTargets = $state<DashboardLoraTargets>({
		base: emptyLoraSeries(),
		assist: emptyLoraSeries()
	});
	let improvement = $state<ImprovementSeries>({ base: [], assist: [] });
	let ragStats: DashboardRagStats = $state({ ...DEFAULT_RAG_STATS });
	let fetchError = $state(false);

	let gridCols = $derived($layout.dashboard.grid_columns);

	onMount(async () => {
		if (!isPro) return;

		components = await loadProComponents(proLoaders);
		if (!components) {
			goto('/');
			return;
		}

		const data = await fetchDashboardData();
		learningData = data.learningData;
		loraTargets = data.loraTargets;
		improvement = data.improvement;
		ragStats = data.ragStats;
		fetchError = data.hasError;
	});

	async function handleRollback(target: LoraTarget, version: number) {
		const label =
			target === 'base' ? $t('dashboard.level2_base') : $t('dashboard.level2_assist');
		if (!confirm($t('dashboard.rollback_confirm', { target: label, version }))) return;

		try {
			await rollbackLora(version, target);
		} catch {
			return;
		}

		try {
			const updated = await getLoraVersions();
			loraTargets = {
				base: mapLoraTarget(updated.base),
				assist: mapLoraTarget(updated.assist)
			};
		} catch {
			// silent — 既存挙動踏襲
		}
	}
</script>

<PageLayout title={$t('sidebar.dashboard')}>
	{#if components?.LearningStatus && components?.LoRAVersions && components?.PerformanceChart && components?.RAGStats}
		{#if fetchError}
			<p class="fetch-error">{$t('dashboard.fetch_error')}</p>
		{/if}
		<div class="dashboard-grid" style="grid-template-columns: repeat({gridCols}, 1fr)">
			<components.LearningStatus
				running={learningData.running}
				experienceCount={learningData.experience_count}
				newExperienceCount={learningData.new_experience_count}
				minExperiences={learningData.min_experiences}
				conditionsMet={learningData.conditions_met}
				lastLevel1Run={learningData.last_level1_run}
				lastLevel2Run={learningData.last_level2_run}
				runningTarget={learningData.running_target}
				level2={learningData.level2}
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
			/>
			<components.LoRAVersions
				base={loraTargets.base}
				assist={loraTargets.assist}
				evalCasesCount={learningData.eval_cases_count}
				evalPassThreshold={learningData.eval_pass_threshold}
				onrollback={handleRollback}
			/>
			<components.PerformanceChart
				baseScores={improvement.base}
				assistScores={improvement.assist}
				baseLabel={loraTargets.base.label}
				assistLabel={loraTargets.assist.label}
			/>
			<components.RAGStats stats={ragStats} />
		</div>
	{:else if isPro}
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
