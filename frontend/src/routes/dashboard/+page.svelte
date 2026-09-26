<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import { isPro } from '$lib/edition';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import LearningStatus from '$lib/free/components/LearningStatus.svelte';
	import RAGStats from '$lib/free/components/RAGStats.svelte';
	import { onMount, onDestroy } from 'svelte';
	import type { Component } from 'svelte';
	import type { DashboardLearningData, DashboardRagStats } from '$lib/free/api';
	import {
		DEFAULT_LEARNING_DATA,
		DEFAULT_RAG_STATS,
		fetchFreeDashboardData
	} from '$lib/free/stores/dashboard';

	const REFRESH_INTERVAL_MS = 30_000;

	// エディション境界: Free 配布物には $lib/pro が無いので、静的 import すると
	// Free のビルドが失敗する。Pro のパネル (LoRA バージョン / 改善カーブ) は
	// `import.meta.glob` で動的ロードする (Free では空になる)。
	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/DashboardProPanels.svelte'
	);

	let DashboardProPanels: Component | null = $state(null);
	let learningData: DashboardLearningData = $state({ ...DEFAULT_LEARNING_DATA });
	let ragStats: DashboardRagStats = $state({ ...DEFAULT_RAG_STATS });
	let freeFetchError = $state(false);
	let proFetchError = $state(false);
	let fetchError = $derived(freeFetchError || proFetchError);
	let loaded = $state(false);

	let gridCols = $derived($layout.dashboard.grid_columns);

	async function refreshDashboardData() {
		const freeData = await fetchFreeDashboardData();
		learningData = freeData.learningData;
		ragStats = freeData.ragStats;
		freeFetchError = freeData.hasError;
	}

	async function loadProPanels() {
		const loaders = Object.values(proLoaders);
		if (loaders.length > 0) {
			DashboardProPanels = (await loaders[0]()).default;
		}
	}

	let refreshTimer: ReturnType<typeof setInterval> | undefined;

	onMount(async () => {
		await Promise.all([refreshDashboardData(), isPro ? loadProPanels() : Promise.resolve()]);
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
			{#if isPro && DashboardProPanels}
				{@const ProPanels = DashboardProPanels}
				<ProPanels
					evalCasesCount={learningData.eval_cases_count}
					evalPassThreshold={learningData.eval_pass_threshold}
					refreshIntervalMs={REFRESH_INTERVAL_MS}
					onfetched={(hasError: boolean) => (proFetchError = hasError)}
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
