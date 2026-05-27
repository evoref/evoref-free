<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import { isPro } from '$lib/edition';
	import { goto } from '$app/navigation';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import type { LoRAVersion, DashboardLearningData, DashboardRagStats, ImprovementScore } from '$lib/free/api';
	import { getLoraVersions, rollbackLora } from '$lib/free/api';
	import {
		type ProComponentMap,
		DEFAULT_LEARNING_DATA,
		DEFAULT_RAG_STATS,
		loadProComponents,
		fetchDashboardData,
		mapLoraVersions
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
	let loraVersions = $state<LoRAVersion[]>([]);
	let improvementScores = $state<ImprovementScore[]>([]);
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
		loraVersions = data.loraVersions;
		improvementScores = data.improvementScores;
		ragStats = data.ragStats;
		fetchError = data.hasError;
	});

	async function handleRollback(version: number) {
		try {
			await rollbackLora(version);
		} catch {
			return;
		}

		try {
			const updated = await getLoraVersions();
			loraVersions = mapLoraVersions(updated.versions, updated.latest_version ?? 0);
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
				versions={loraVersions}
				loraVersion={learningData.lora_version}
				loraAdapterExists={learningData.lora_adapter_exists}
				evalCasesCount={learningData.eval_cases_count}
				evalPassThreshold={learningData.eval_pass_threshold}
				onrollback={handleRollback}
			/>
			<components.PerformanceChart scores={improvementScores} />
			<components.RAGStats
				chunkCount={ragStats.chunkCount}
				vectorCount={ragStats.vectorCount}
				indexSizeMb={ragStats.indexSizeMb}
			/>
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
