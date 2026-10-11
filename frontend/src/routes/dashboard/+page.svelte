<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import DashboardOverview from '$lib/free/components/DashboardOverview.svelte';
	import LearningSummary from '$lib/free/components/LearningSummary.svelte';
	import DocumentSummary from '$lib/free/components/DocumentSummary.svelte';
	import LearningDetail from '$lib/free/components/LearningDetail.svelte';
	import RagDetail from '$lib/free/components/RagDetail.svelte';
	import SystemHealthDetail from '$lib/free/components/SystemHealthDetail.svelte';
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

<!--
	Free (一般向け) は全体の状態・学習・文書だけを出す。Pro (プロ向け) はその下に「詳細」
	(学習効果・成長のゲートと回帰・RAG の内訳・システムの診断・組織ポリシー・モデル適合) を足す。
-->
<PageLayout title={$t('sidebar.dashboard')}>
	{#if loaded}
		<DashboardOverview {ragStats} {fetchError} />
		<div class="summary-grid" style="grid-template-columns: repeat({gridCols}, minmax(0, 1fr))">
			<LearningSummary
				data={learningData}
				{growth}
				showMeasure={$proActive}
				onGrowthChange={refreshDashboardData}
			/>
			<DocumentSummary stats={ragStats} />
		</div>
		{#if $proActive}
			<details class="details" open>
				<summary>{$t('dashboard.details_title')}</summary>
				<div class="details-grid">
					<LearningDetail data={learningData} {growth} />
					<RagDetail stats={ragStats} />
					<SystemHealthDetail />
				</div>
				{#if OrgPolicyPanel}
					{@const PolicyPanel = OrgPolicyPanel}
					<PolicyPanel />
				{/if}
				{#if ModelFitPanel}
					{@const FitPanel = ModelFitPanel}
					<FitPanel />
				{/if}
			</details>
		{/if}
	{:else}
		<div class="loading">{$t('common.loading')}</div>
	{/if}
</PageLayout>

<style>
	.summary-grid {
		display: grid;
		gap: 12px;
		align-items: start;
	}
	.details {
		margin-top: 20px;
	}
	.details > summary {
		cursor: pointer;
		font-size: 0.9375rem;
		font-weight: 600;
		color: var(--text-secondary);
		margin-bottom: 12px;
	}
	.details-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: 12px;
		align-items: start;
	}
	@media (max-width: 900px) {
		.summary-grid {
			grid-template-columns: minmax(0, 1fr) !important;
		}
		.details-grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.loading {
		display: flex;
		justify-content: center;
		align-items: center;
		padding: 2rem;
		color: var(--text-secondary);
	}
</style>
