<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import { isPro } from '$lib/edition';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import LearningStatus from '$lib/free/components/LearningStatus.svelte';
	import RAGStats from '$lib/free/components/RAGStats.svelte';
	import { onMount, onDestroy } from 'svelte';
	import type { Component } from 'svelte';
	import type { LoraTarget, LoraMode, DashboardLearningData, DashboardRagStats } from '$lib/free/api';
	import { getLoraVersions, rollbackLora } from '$lib/free/api';
	import {
		DEFAULT_LEARNING_DATA,
		DEFAULT_RAG_STATS,
		fetchFreeDashboardData
	} from '$lib/free/stores/dashboard';
	import {
		type ProComponentMap,
		type DashboardLoraTargets,
		type ImprovementSeries,
		loadProComponents,
		fetchProDashboardData,
		mapLoraTarget,
		emptyLoraSeries
	} from '$lib/pro/stores/dashboard';

	const REFRESH_INTERVAL_MS = 30_000;

	// Pro のみ動的ロード対象 (LoRA バージョン / 改善カーブは Pro 専用データのため)
	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/{LoRAVersions,PerformanceChart}.svelte'
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
	let loaded = $state(false);
	// learning.level2_adapter_partition=="model_mode" のときのみ意味を持つ表示対象
	// モード ("model" スキームでは chat/coding どちらを選んでも同じ内容が返る)。
	let loraMode = $state<LoraMode>('chat');

	let gridCols = $derived($layout.dashboard.grid_columns);

	async function refreshDashboardData() {
		if (isPro) {
			const [freeData, proData] = await Promise.all([
				fetchFreeDashboardData(),
				fetchProDashboardData(loraMode)
			]);
			learningData = freeData.learningData;
			ragStats = freeData.ragStats;
			loraTargets = proData.loraTargets;
			improvement = proData.improvement;
			fetchError = freeData.hasError || proData.hasError;
		} else {
			const freeData = await fetchFreeDashboardData();
			learningData = freeData.learningData;
			ragStats = freeData.ragStats;
			fetchError = freeData.hasError;
		}
	}

	let refreshTimer: ReturnType<typeof setInterval> | undefined;

	onMount(async () => {
		if (isPro) {
			const [loadedComponents] = await Promise.all([
				loadProComponents(proLoaders),
				refreshDashboardData()
			]);
			components = loadedComponents;
		} else {
			await refreshDashboardData();
		}
		loaded = true;
		refreshTimer = setInterval(refreshDashboardData, REFRESH_INTERVAL_MS);
	});

	onDestroy(() => {
		if (refreshTimer) clearInterval(refreshTimer);
	});

	async function handleModeChange(newMode: LoraMode) {
		loraMode = newMode;
		try {
			const updated = await getLoraVersions(loraMode);
			loraTargets = {
				base: mapLoraTarget(updated.base),
				assist: mapLoraTarget(updated.assist)
			};
		} catch {
			// silent — 次回の定期リフレッシュで再試行される
		}
	}

	async function handleRollback(target: LoraTarget, version: number) {
		const label =
			target === 'base' ? $t('dashboard.level2_base') : $t('dashboard.level2_assist');
		if (!confirm($t('dashboard.rollback_confirm', { target: label, version }))) return;

		try {
			const result = await rollbackLora(version, target, loraMode);
			if (result.restart_required) {
				alert($t('dashboard.rollback_restart_required'));
			}
		} catch {
			return;
		}

		try {
			const updated = await getLoraVersions(loraMode);
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
			{#if isPro && components?.LoRAVersions}
				<components.LoRAVersions
					base={loraTargets.base}
					assist={loraTargets.assist}
					evalCasesCount={learningData.eval_cases_count}
					evalPassThreshold={learningData.eval_pass_threshold}
					onrollback={handleRollback}
					mode={loraMode}
					onmodechange={handleModeChange}
				/>
			{/if}
			{#if isPro && components?.PerformanceChart}
				<components.PerformanceChart
					baseScores={improvement.base}
					assistScores={improvement.assist}
					baseLabel={loraTargets.base.label}
					assistLabel={loraTargets.assist.label}
				/>
			{/if}
			<RAGStats stats={ragStats} />
		</div>
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
