<script lang="ts">
	/**
	 * 学習の詳細 (Pro のダッシュボードの「詳細」): 学習効果 (fitness の推移)・Level 2 (成長) の
	 * 自動採用と回帰台帳・経験の質・パラメータ最適化。一般向けの状態は LearningSummary が出す。
	 */
	import { t } from '$lib/i18n';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';
	import StatusSection from '$lib/free/components/StatusSection.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';
	import GrowthSection from '$lib/free/components/GrowthSection.svelte';
	import type {
		DashboardLearningData,
		GrowthResponse,
		PolicyEvolverDomainStatus
	} from '$lib/free/api';

	interface LearningDetailProps {
		data: DashboardLearningData;
		/** Level 2 (成長) の状態 (取れなければ欄を出さない) */
		growth?: GrowthResponse | null;
	}

	let { data, growth = null }: LearningDetailProps = $props();

	function formatPercent(rate: number): string {
		return `${Math.round(rate * 100)}%`;
	}

	function formatFitness(val: number | null): string {
		if (val === null || val === undefined) return '-';
		return val.toFixed(4);
	}

	/** トレンド矢印を計算 */
	function trendArrow(current: number, prev: number | null): string {
		if (prev === null) return '';
		const diff = current - prev;
		if (Math.abs(diff) < 0.005) return '';
		return diff > 0 ? '↑' : '↓';
	}

	/** 探索フェーズのラベル */
	function phaseLabel(phase: string): string {
		const map: Record<string, string> = {
			explore: $t('dashboard.phase_explore'),
			exploit: $t('dashboard.phase_exploit'),
			transition: $t('dashboard.phase_transition'),
			explore_reset: $t('dashboard.phase_explore_reset')
		};
		return map[phase] ?? phase;
	}

	let hasFitnessHistory = $derived(Object.keys(data.fitness_history).length > 0);

	/** モード別の累積改善情報を構築 */
	let fitnessImpact = $derived.by(() => {
		const items: { mode: string; first: number; last: number; changePercent: number }[] = [];
		for (const [mode, points] of Object.entries(data.fitness_history)) {
			if (!points || points.length === 0) continue;
			const first = points[0].fitness;
			const last = points[points.length - 1].fitness;
			const changePercent = first > 0 ? ((last - first) / first) * 100 : 0;
			items.push({ mode, first, last, changePercent });
		}
		return items;
	});

	/** 変化があったフェーズのみ抽出 */
	let changedPhases = $derived.by(() => {
		const lines: string[] = [];
		for (const [key, entry] of Object.entries(data.last_level1_results)) {
			if (key.startsWith('_')) continue;
			if (entry.improved) {
				lines.push(`${key}: ${formatFitness(entry.fitness_before)}→${formatFitness(entry.fitness_after)}`);
			}
		}
		return lines;
	});

	let hasPolicyEvolver = $derived(Object.keys(data.policy_evolver_status).length > 0);

	/** σ を 0〜100% の探索度に変換（σ範囲: 0.01〜0.15） */
	function sigmaToPercent(sigma: number): number {
		return Math.min(100, Math.max(0, ((sigma - 0.01) / (0.15 - 0.01)) * 100));
	}

	/** ドメインの表示ラベル（安定 / 探索中 / 改善中 など） */
	function domainLabel(status: PolicyEvolverDomainStatus): string {
		if (status.phase === 'exploit' && status.decline_count === 0) {
			return $t('dashboard.stable');
		}
		return phaseLabel(status.phase);
	}
</script>

<DashboardCard title={$t('dashboard.learning_status')}>
	<div class="status-grid">
		<StatusSection label={$t('dashboard.learning_impact')}>
			{#if data.level1_run_count > 0}
				<StatusRow
					label={$t('dashboard.level1_run_count')}
					value={$t('dashboard.level1_run_count_value', { count: data.level1_run_count })}
				/>
			{/if}
			{#if hasFitnessHistory}
				{#each fitnessImpact as item}
					<div class="fitness-row">
						<span class="fitness-mode">{item.mode}</span>
						<span class="fitness-values">
							{item.first.toFixed(2)} <span class="arrow">{'→'}</span> {item.last.toFixed(2)}
						</span>
						<span class="fitness-change">
							{#if item.changePercent > 0}
								{'▲'}+{item.changePercent.toFixed(0)}%
							{:else if item.changePercent < 0}
								{'▼'}{item.changePercent.toFixed(0)}%
							{:else}
								{$t('dashboard.stable')}
							{/if}
						</span>
					</div>
				{/each}
				{#if changedPhases.length > 0}
					<div class="changed-phases">
						<span class="sub-label">{$t('dashboard.changed_phases')}</span>
						{#each changedPhases as line}
							<span class="phase-line">{line}</span>
						{/each}
					</div>
				{:else if data.level1_run_count > 0}
					<span class="hint-text">{$t('dashboard.stable')} - {$t('dashboard.stable_hint')}</span>
				{/if}
			{:else if data.level1_run_count > 0}
				<span class="hint-text">{$t('dashboard.stable')} - {$t('dashboard.stable_hint')}</span>
			{:else}
				<span class="hint-text">{$t('dashboard.no_history')}</span>
			{/if}
		</StatusSection>

		{#if growth}
			<GrowthSection {growth} />
		{/if}

		<StatusSection label={$t('dashboard.experience_quality')}>
			<StatusRow
				label={$t('dashboard.experience_by_mode')}
				value="{$t('dashboard.mode_chat')}: {data.experience_by_mode.chat} / {$t('dashboard.mode_create')}: {data.experience_by_mode.create}"
			/>
			<StatusRow
				label={$t('dashboard.correction_rate')}
				value="{formatPercent(data.correction_rate)} {trendArrow(data.correction_rate, data.prev_correction_rate)}"
			/>
			<StatusRow
				label={$t('dashboard.rag_usage_rate')}
				value="{formatPercent(data.rag_usage_rate)} {trendArrow(data.rag_usage_rate, data.prev_rag_usage_rate)}"
			/>
		</StatusSection>

		{#if hasPolicyEvolver}
			<StatusSection label={$t('dashboard.optimization_status')}>
				{#each Object.entries(data.policy_evolver_status) as [domain, status]}
					<div class="optimization-row">
						<span class="domain-label">{domain}</span>
						<span class="domain-status">{domainLabel(status)}</span>
						<div class="exploration-bar">
							<div class="exploration-fill" style="width: {sigmaToPercent(status.sigma)}%"></div>
						</div>
						<span class="sigma-value">{Math.round(sigmaToPercent(status.sigma))}%</span>
						{#if status.decline_count >= 3}
							<span class="rollback-badge">rollback</span>
						{/if}
					</div>
				{/each}
			</StatusSection>
		{/if}
	</div>
</DashboardCard>

<style>
	.status-grid {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.hint-text {
		font-size: 0.8125rem;
		color: var(--text-secondary);
	}
	.fitness-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.8125rem;
	}
	.fitness-mode {
		min-width: 60px;
		color: var(--text-primary);
		font-weight: 500;
	}
	.fitness-values,
	.fitness-values .arrow {
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
	}
	.fitness-change {
		font-weight: 600;
		font-size: 0.75rem;
		color: var(--text-primary);
	}
	.changed-phases {
		display: flex;
		flex-direction: column;
		gap: 2px;
		font-size: 0.75rem;
	}
	.sub-label {
		color: var(--text-secondary);
	}
	.phase-line {
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
	}
	.optimization-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.8125rem;
	}
	.domain-label {
		min-width: 70px;
		color: var(--text-primary);
	}
	.domain-status {
		min-width: 50px;
		font-size: 0.75rem;
		color: var(--text-secondary);
	}
	.exploration-bar {
		flex: 1;
		max-width: 60px;
		height: 4px;
		background-color: var(--bg-secondary);
		border-radius: 2px;
		overflow: hidden;
	}
	.exploration-fill {
		height: 100%;
		background-color: var(--text-secondary);
		border-radius: 2px;
	}
	.sigma-value {
		font-size: 0.75rem;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
		min-width: 30px;
		text-align: right;
	}
	.rollback-badge {
		font-size: 0.6875rem;
		color: var(--text-primary);
		border: 1px solid var(--color-error);
		padding: 0 4px;
		border-radius: 3px;
	}
</style>
