<script lang="ts">
	import { t } from '$lib/i18n';
	import { isPro } from '$lib/edition';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';
	import StatusSection from '$lib/free/components/StatusSection.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';
	import GrowthSection from '$lib/free/components/GrowthSection.svelte';
	import type {
		ExperienceByMode,
		Level1ResultEntry,
		FitnessPoint,
		GrowthResponse,
		PolicyEvolverDomainStatus
	} from '$lib/free/api';

	interface LearningStatusProps {
		running?: boolean;
		experienceCount?: number;
		newExperienceCount?: number;
		minExperiences?: number;
		conditionsMet?: boolean;
		/** Level 1 が今走れない理由 (走れる状態なら null) */
		level1BlockedReason?: string | null;
		/** waiting_for_idle のとき、アイドル成立までの残り秒数 */
		level1SecondsUntilIdle?: number | null;
		lastLevel1Run?: string | null;
		lastLevel0Record?: string | null;
		experienceByMode?: ExperienceByMode;
		correctionRate?: number;
		ragUsageRate?: number;
		prevCorrectionRate?: number | null;
		prevRagUsageRate?: number | null;
		level1RunCount?: number;
		lastLevel1Results?: Record<string, Level1ResultEntry>;
		executedPhases?: string[];
		fitnessHistory?: Record<string, FitnessPoint[]>;
		policyEvolverStatus?: Record<string, PolicyEvolverDomainStatus>;
		/** `--no-learning` 起動中 (常時バナーを出す) */
		isDisabled?: boolean;
		/** Level 2 (成長) の状態 (`GET /api/learning/growth`。取れなければ欄を出さない) */
		growth?: GrowthResponse | null;
		/** 提案の採用・却下・巻き戻しの後に呼ぶ (親が再取得する) */
		onGrowthChange?: () => void;
	}

	let {
		running = false,
		newExperienceCount = 0,
		minExperiences = 5,
		conditionsMet = false,
		level1BlockedReason = null,
		level1SecondsUntilIdle = null,
		lastLevel1Run = null,
		experienceByMode = { chat: 0, create: 0 },
		correctionRate = 0,
		ragUsageRate = 0,
		prevCorrectionRate = null,
		prevRagUsageRate = null,
		level1RunCount = 0,
		lastLevel1Results = {},
		fitnessHistory = {},
		policyEvolverStatus = {},
		isDisabled = false,
		growth = null,
		onGrowthChange,
		..._ // experienceCount, lastLevel0Record, executedPhases: API互換のため受容
	}: LearningStatusProps = $props();

	/** 相対時間表示（例: "2時間前"） */
	function formatRelativeTime(iso: string | null): string {
		if (!iso) return '-';
		try {
			const d = new Date(iso);
			const now = Date.now();
			const diffMs = now - d.getTime();
			if (diffMs < 0) return '-';
			const diffMin = Math.floor(diffMs / 60000);
			if (diffMin < 1) return $t('dashboard.last_run_ago', { time: '<1min' });
			if (diffMin < 60) return $t('dashboard.last_run_ago', { time: `${diffMin}min` });
			const diffH = Math.floor(diffMin / 60);
			if (diffH < 24) return $t('dashboard.last_run_ago', { time: `${diffH}h` });
			const diffD = Math.floor(diffH / 24);
			return $t('dashboard.last_run_ago', { time: `${diffD}d` });
		} catch {
			return '-';
		}
	}

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
		return diff > 0 ? '\u2191' : '\u2193';
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

	/** Level 1 が止まっている理由の表示文 (詰まっていなければ空) */
	let blockedHint = $derived.by(() => {
		if (!level1BlockedReason) return '';
		if (level1BlockedReason === 'waiting_for_idle') {
			const minutes = Math.max(1, Math.ceil((level1SecondsUntilIdle ?? 0) / 60));
			return $t('dashboard.level1_blocked_waiting_for_idle', { minutes });
		}
		const known = [
			'user_active',
			'no_llm_client',
			'loop_not_started',
			'already_running',
			'learning_disabled',
			'single_slot',
			'deferred_by_full_cycle',
			'serving_another_partition',
			'insufficient_experiences'
		];
		if (known.includes(level1BlockedReason)) {
			return $t(`dashboard.level1_blocked_${level1BlockedReason}`);
		}
		// 未知の理由コードは握り潰さず、そのまま見せる
		return level1BlockedReason;
	});

	// ── 次の学習セクション ──
	let remaining = $derived(Math.max(0, minExperiences - newExperienceCount));
	let newExpProgress = $derived(
		minExperiences > 0 ? Math.min(newExperienceCount / minExperiences, 1) : 0
	);

	// ── 学習効果セクション ──
	let hasFitnessHistory = $derived(Object.keys(fitnessHistory).length > 0);

	/** モード別の累積改善情報を構築 */
	let fitnessImpact = $derived.by(() => {
		const items: { mode: string; first: number; last: number; changePercent: number }[] = [];
		for (const [mode, points] of Object.entries(fitnessHistory)) {
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
		for (const [key, entry] of Object.entries(lastLevel1Results)) {
			if (key.startsWith('_')) continue;
			if (entry.improved) {
				lines.push(`${key}: ${formatFitness(entry.fitness_before)}\u2192${formatFitness(entry.fitness_after)}`);
			}
		}
		return lines;
	});

	let hasPolicyEvolver = $derived(Object.keys(policyEvolverStatus).length > 0);

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
	{#if isDisabled}
		<p class="disabled-banner">{$t('dashboard.learning_disabled_banner')}</p>
	{/if}
	<div class="status-grid">
		<!-- セクション 1: 次の学習 -->
		<StatusSection label={$t('dashboard.next_learning')}>
			{#if running}
				<div class="running-indicator" role="status">
					<span class="pulse"></span>
					<span class="running-text">{$t('dashboard.learning_running')}</span>
				</div>
			{:else}
				<div class="new-exp-row">
					<span class="new-exp-label">{$t('dashboard.new_experiences')}</span>
					<div class="progress-bar">
						<div
							class="progress-fill"
							class:full={newExpProgress >= 1}
							style="width: {newExpProgress * 100}%"
						></div>
					</div>
					<span class="new-exp-value">{newExperienceCount} / {minExperiences}</span>
				</div>
				{#if conditionsMet}
					<!-- conditionsMet は経験件数だけの表示値。実ゲート (アイドル /
					     ユーザー活動 / LLM 配線) で止まっている場合はその理由を出す。
					     出さないと「条件を満たしているのに走らない」が原因不明に見える。 -->
					<span class="hint-text">{blockedHint || $t('dashboard.ready_hint')}</span>
				{:else}
					<span class="hint-text">{$t('dashboard.remaining_hint', { count: remaining })}</span>
				{/if}
			{/if}
			{#if lastLevel1Run}
				<StatusRow
					active={true}
					label={$t('dashboard.last_level1')}
					value={formatRelativeTime(lastLevel1Run)}
				/>
			{/if}
			{#if level1RunCount > 0}
				<StatusRow
					active={true}
					label={$t('dashboard.level1_run_count')}
					value={$t('dashboard.level1_run_count_value', { count: level1RunCount })}
				/>
			{/if}
		</StatusSection>

		<!-- セクション 2: 学習効果 -->
		<StatusSection label={$t('dashboard.learning_impact')}>
			{#if hasFitnessHistory}
				{#each fitnessImpact as item}
					<div class="fitness-row">
						<span class="fitness-mode">{item.mode}</span>
						<span class="fitness-values">
							{item.first.toFixed(2)} <span class="arrow">{'\u2192'}</span> {item.last.toFixed(2)}
						</span>
						<span
							class="fitness-change"
							class:positive={item.changePercent > 0}
							class:negative={item.changePercent < 0}
						>
							{#if item.changePercent > 0}
								{'\u25B2'}+{item.changePercent.toFixed(0)}%
							{:else if item.changePercent < 0}
								{'\u25BC'}{item.changePercent.toFixed(0)}%
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
				{:else if level1RunCount > 0}
					<span class="stable-text">{$t('dashboard.stable')} - {$t('dashboard.stable_hint')}</span>
				{/if}
			{:else if level1RunCount > 0}
				<span class="stable-text">{$t('dashboard.stable')} - {$t('dashboard.stable_hint')}</span>
			{:else}
				<StatusRow active={false} label="" value={$t('dashboard.no_history')} />
			{/if}
		</StatusSection>

		<!-- Level 2 (成長): 自動採用の状態・提案・回帰台帳・巻き戻しの提案 (f_04 §6.8) -->
		{#if growth}
			<GrowthSection {growth} onchange={onGrowthChange} />
		{/if}

		<!-- セクション 3: 経験の質 -->
		<StatusSection label={$t('dashboard.experience_quality')}>
			<StatusRow
				active={experienceByMode.chat > 0 || experienceByMode.create > 0}
				label={$t('dashboard.experience_by_mode')}
				value="{$t('dashboard.mode_chat')}: {experienceByMode.chat} / {$t('dashboard.mode_create')}: {experienceByMode.create}"
			/>
			<StatusRow
				active={correctionRate > 0}
				label={$t('dashboard.correction_rate')}
				value="{formatPercent(correctionRate)} {trendArrow(correctionRate, prevCorrectionRate)}"
			/>
			<StatusRow
				active={ragUsageRate > 0}
				label={$t('dashboard.rag_usage_rate')}
				value="{formatPercent(ragUsageRate)} {trendArrow(ragUsageRate, prevRagUsageRate)}"
			/>
		</StatusSection>

		<!-- 探索/活用フェーズ（Pro のみ、データがある場合） -->
		{#if isPro && hasPolicyEvolver}
			<StatusSection label={$t('dashboard.optimization_status')}>
				{#each Object.entries(policyEvolverStatus) as [domain, status]}
					<div class="optimization-row">
						<span class="domain-label">{domain}</span>
						<span class="domain-status">{domainLabel(status)}</span>
						<div class="exploration-bar">
							<div
								class="exploration-fill"
								style="width: {sigmaToPercent(status.sigma)}%"
							></div>
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
	.disabled-banner {
		margin: 0 0 12px;
		padding: 8px 12px;
		border-radius: 6px;
		font-size: 0.8125rem;
		color: var(--color-warning, var(--text-secondary));
		border: 1px solid var(--color-warning, var(--border-color, currentColor));
	}

	.status-grid {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	/* ── 次の学習: 実行中インジケータ ── */
	.running-indicator {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 4px 0;
	}
	.pulse {
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background-color: var(--color-info, var(--accent));
		animation: pulse-anim 1.5s ease-in-out infinite;
	}
	@keyframes pulse-anim {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.3; }
	}
	.running-text {
		font-size: 0.8125rem;
		color: var(--color-info, var(--accent));
		font-weight: 500;
	}

	/* ── 次の学習: 新規経験プログレス ── */
	.new-exp-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 2px 0;
	}
	.new-exp-label {
		font-size: 0.75rem;
		color: var(--text-secondary);
		white-space: nowrap;
	}
	.new-exp-value {
		font-size: 0.75rem;
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.hint-text {
		font-size: 0.6875rem;
		color: var(--text-secondary);
		padding-left: 2px;
	}

	.progress-bar {
		flex: 1;
		height: 4px;
		background-color: var(--bg-secondary);
		border-radius: 2px;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background-color: var(--color-info, var(--accent));
		border-radius: 2px;
		transition: width 0.3s ease;
	}
	.progress-fill.full {
		background-color: var(--color-success, var(--accent));
	}

	/* ── 学習効果: fitness 表示 ── */
	.fitness-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 2px 16px;
		font-size: 0.8125rem;
	}
	.fitness-mode {
		min-width: 60px;
		color: var(--text-primary);
		font-weight: 500;
	}
	.fitness-values {
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
	}
	.fitness-values .arrow {
		color: var(--text-secondary);
	}
	.fitness-change {
		font-weight: 600;
		font-size: 0.75rem;
	}
	.fitness-change.positive {
		color: var(--color-success, #22c55e);
	}
	.fitness-change.negative {
		color: var(--color-error, #ef4444);
	}

	.changed-phases {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding-left: 16px;
		font-size: 0.75rem;
	}
	.changed-phases .sub-label {
		color: var(--text-secondary);
		font-size: 0.6875rem;
	}
	.phase-line {
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
	}

	.stable-text {
		font-size: 0.75rem;
		color: var(--text-secondary);
		padding-left: 16px;
	}

	/* ── 探索/活用: パラメータ最適化 ── */
	.optimization-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.8125rem;
		padding-left: 16px;
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
		background-color: var(--color-info, var(--accent));
		border-radius: 2px;
		transition: width 0.3s ease;
	}
	.sigma-value {
		font-size: 0.75rem;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
		min-width: 30px;
		text-align: right;
	}
	.rollback-badge {
		font-size: 0.625rem;
		color: var(--color-error, #ef4444);
		background-color: color-mix(in srgb, var(--color-error, #ef4444) 15%, transparent);
		padding: 0 4px;
		border-radius: 3px;
	}
</style>
