<script lang="ts">
	import { t } from '$lib/i18n';
	import { isPro } from '$lib/edition';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';
	import StatusSection from '$lib/free/components/StatusSection.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';
	import type {
		ExperienceByMode,
		Level1ResultEntry,
		FitnessPoint,
		PolicyEvolverDomainStatus,
		Level2Status,
		Level2TargetStatus
	} from '$lib/free/api';

	interface LearningStatusProps {
		running?: boolean;
		experienceCount?: number;
		newExperienceCount?: number;
		minExperiences?: number;
		conditionsMet?: boolean;
		lastLevel1Run?: string | null;
		lastLevel2Run?: string | null;
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
		runningTarget?: string | null;
		level2?: Level2Status | null;
	}

	let {
		running = false,
		newExperienceCount = 0,
		minExperiences = 5,
		conditionsMet = false,
		lastLevel1Run = null,
		lastLevel2Run = null,
		experienceByMode = { chat: 0, coding: 0 },
		correctionRate = 0,
		ragUsageRate = 0,
		prevCorrectionRate = null,
		prevRagUsageRate = null,
		level1RunCount = 0,
		lastLevel1Results = {},
		fitnessHistory = {},
		policyEvolverStatus = {},
		runningTarget = null,
		level2 = null,
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

	// ── Level 2 (base/assist) 個別状態 ──
	interface Level2TargetView {
		key: 'base' | 'assist';
		label: string;
		isRunning: boolean;
		method: string;
		versionLabel: string;
		current: number;
		required: number | null; // null = no-op (発火しない方式)
		statusLabel: string;
		reasonLabel: string; // 発火しない理由 (空文字なら非表示)
		progress: number;
	}

	/** base/assist それぞれの状態 + 発火閾値を表示用に整形する */
	function buildTargetView(key: 'base' | 'assist', tStat: Level2TargetStatus): Level2TargetView {
		const isRunning = runningTarget === key;
		// 発火に必要な閾値: adapter 未生成なら bootstrap、生成済みは方式で分岐。
		// base=lora(adapter有) と assist=none は no-op skip のため required=null。
		let required: number | null;
		if (!tStat.adapter_exists) {
			required = tStat.bootstrap_min;
		} else if (key === 'base' && tStat.method === 'cvector') {
			required = tStat.cvector_min;
		} else if (key === 'assist' && tStat.method === 'spsa-real-eval') {
			required = tStat.spsa_min;
		} else {
			required = null;
		}
		const current = tStat.experiences_current;
		let statusLabel: string;
		if (isRunning) statusLabel = $t('dashboard.level2_running');
		else if (required === null) statusLabel = $t('dashboard.level2_noop');
		else if (current >= required) statusLabel = $t('dashboard.level2_ready');
		else statusLabel = $t('dashboard.level2_accumulating');
		// backend が算出した発火不可理由 ("" = 発火可能) を i18n ラベル化する。
		// 表示と実トリガの乖離 (例: ready なのに発火しない) を可視化する単一ソース。
		const reasonLabel = tStat.block_reason
			? $t(`dashboard.level2_reason_${tStat.block_reason}`)
			: '';
		return {
			key,
			label: key === 'base' ? $t('dashboard.level2_base') : $t('dashboard.level2_assist'),
			isRunning,
			method: tStat.method || '-',
			versionLabel: tStat.adapter_exists
				? `v${tStat.version}`
				: $t('dashboard.level2_adapter_none'),
			current,
			required,
			statusLabel,
			reasonLabel,
			progress: required && required > 0 ? Math.min(current / required, 1) : 0
		};
	}

	let level2Targets = $derived(
		level2
			? [buildTargetView('base', level2.base), buildTargetView('assist', level2.assist)]
			: []
	);
</script>

<DashboardCard title={$t('dashboard.learning_status')}>
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
					<span class="hint-text">{$t('dashboard.ready_hint')}</span>
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
			{#if isPro && lastLevel2Run}
				<StatusRow
					active={true}
					label={$t('dashboard.last_level2')}
					value={formatRelativeTime(lastLevel2Run)}
				/>
			{/if}
		</StatusSection>

		<!-- セクション 2.5: Level 2 (base/assist) 個別状態 + 発火条件 -->
		{#if isPro && level2}
			<StatusSection label={$t('dashboard.level2_section')}>
				{#each level2Targets as tv (tv.key)}
					<div class="level2-row">
						<span class="l2-dot" class:running={tv.isRunning}></span>
						<span class="l2-name">{tv.label}</span>
						<span class="l2-status" class:running={tv.isRunning}>{tv.statusLabel}</span>
						<span class="l2-meta">{tv.method} · {tv.versionLabel}</span>
					</div>
					{#if tv.reasonLabel}
						<div class="level2-reason-row">
							<span class="l2-reason">{tv.reasonLabel}</span>
						</div>
					{/if}
					<div class="level2-progress-row">
						{#if tv.required !== null && tv.progress >= 1}
							<span class="l2-count l2-saturated">
								{$t('dashboard.level2_sufficient_data', { count: tv.current })}
							</span>
						{:else if tv.required !== null}
							<div class="progress-bar">
								<div class="progress-fill" style="width: {tv.progress * 100}%"></div>
							</div>
							<span class="l2-count">{tv.current} / {tv.required}</span>
						{:else}
							<span class="l2-count l2-noop">{tv.current} / —</span>
						{/if}
					</div>
				{/each}
				<div class="level2-gates">
					<span class="sub-label">{$t('dashboard.level2_firing_conditions')}</span>
					<span class="gate-line">
						{$t('dashboard.level2_gate_idle', { min: level2.gates.active_minutes })} ・
						{$t('dashboard.level2_gate_overdue', { hours: level2.gates.overdue_hours })} ・
						{$t('dashboard.level2_gate_interval', { sec: level2.gates.recheck_interval_sec })}
					</span>
				</div>
			</StatusSection>
		{/if}

		<!-- セクション 3: 経験の質 -->
		<StatusSection label={$t('dashboard.experience_quality')}>
			<StatusRow
				active={experienceByMode.chat > 0 || experienceByMode.coding > 0}
				label={$t('dashboard.experience_by_mode')}
				value="{$t('dashboard.mode_chat')}: {experienceByMode.chat} / {$t('dashboard.mode_coding')}: {experienceByMode.coding}"
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

	/* ── Level 2: base/assist 個別状態 ── */
	.level2-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 2px 0;
		font-size: 0.8125rem;
	}
	.l2-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		border: 1.5px solid var(--text-secondary);
		flex-shrink: 0;
	}
	.l2-dot.running {
		background-color: var(--color-info, var(--accent));
		border-color: var(--color-info, var(--accent));
		animation: pulse-anim 1.5s ease-in-out infinite;
	}
	.l2-name {
		min-width: 52px;
		font-weight: 500;
		color: var(--text-primary);
	}
	.l2-status {
		font-size: 0.75rem;
		color: var(--text-secondary);
	}
	.l2-status.running {
		color: var(--color-info, var(--accent));
		font-weight: 500;
	}
	.l2-meta {
		margin-left: auto;
		font-size: 0.6875rem;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
	}
	.level2-reason-row {
		padding: 0 0 2px 16px;
	}
	.l2-reason {
		font-size: 0.6875rem;
		color: var(--text-warning, var(--text-secondary));
		opacity: 0.85;
	}
	.level2-progress-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 0 0 4px 16px;
	}
	.l2-count {
		font-size: 0.6875rem;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.l2-count.l2-noop {
		opacity: 0.6;
	}
	.l2-count.l2-saturated {
		color: var(--color-success, var(--accent));
		font-weight: 500;
	}
	.level2-gates {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding-left: 16px;
	}
	.level2-gates .sub-label {
		font-size: 0.6875rem;
		color: var(--text-secondary);
	}
	.gate-line {
		font-size: 0.6875rem;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
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
