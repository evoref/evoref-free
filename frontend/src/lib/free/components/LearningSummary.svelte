<script lang="ts">
	/**
	 * ダッシュボードの学習カード (一般向け): 学習が動いているか、次はいつ進むか、
	 * 直近の更新、承認待ちの提案 (採用・却下)。専門の数値は LearningDetail (Pro) が出す。
	 */
	import { t } from '$lib/i18n';
	import DashboardCard from '$lib/free/components/DashboardCard.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';
	import StatusSection from '$lib/free/components/StatusSection.svelte';
	import GrowthProposals from '$lib/free/components/GrowthProposals.svelte';
	import type { DashboardLearningData, GrowthResponse } from '$lib/free/api';

	interface LearningSummaryProps {
		data: DashboardLearningData;
		/** Level 2 (成長) の状態。取れなければ提案の欄を出さない */
		growth?: GrowthResponse | null;
		/** 提案の比較の内訳を出す (Pro) */
		showMeasure?: boolean;
		onGrowthChange?: () => void;
	}

	let { data, growth = null, showMeasure = false, onGrowthChange }: LearningSummaryProps = $props();

	/** 経験が貯まっても進まない理由のうち、待っても解けないもの (設定・起動の問題) */
	const HARD_STOPS = ['learning_disabled', 'single_slot', 'no_llm_client', 'loop_not_started'];
	const KNOWN_REASONS = [
		...HARD_STOPS,
		'user_active',
		'already_running',
		'deferred_by_full_cycle',
		'serving_another_partition',
		'insufficient_experiences'
	];

	let blockedHint = $derived.by(() => {
		const reason = data.level1_blocked_reason;
		if (!reason) return '';
		if (reason === 'waiting_for_idle') {
			const minutes = Math.max(1, Math.ceil((data.level1_seconds_until_idle ?? 0) / 60));
			return $t('dashboard.level1_blocked_waiting_for_idle', { minutes });
		}
		// 未知の理由コードは握り潰さず、そのまま見せる
		return KNOWN_REASONS.includes(reason) ? $t(`dashboard.level1_blocked_${reason}`) : reason;
	});

	let stopped = $derived(
		!data.is_disabled && !data.running && HARD_STOPS.includes(data.level1_blocked_reason ?? '')
	);

	let stateText = $derived(
		data.is_disabled
			? $t('dashboard.learning_state_off')
			: data.running
				? $t('dashboard.learning_state_running')
				: stopped
					? $t('dashboard.learning_state_stopped')
					: $t('dashboard.learning_state_idle')
	);

	let nextText = $derived.by(() => {
		if (data.is_disabled) return $t('dashboard.learning_next_off');
		if (data.running) return $t('dashboard.learning_next_running');
		if (stopped || data.conditions_met) return blockedHint || $t('dashboard.ready_hint');
		return $t('dashboard.remaining_hint', {
			count: Math.max(0, data.min_experiences - data.new_experience_count)
		});
	});

	function formatRelativeTime(iso: string | null): string {
		if (!iso) return $t('dashboard.learning_last_none');
		const diffMs = Date.now() - new Date(iso).getTime();
		if (!Number.isFinite(diffMs) || diffMs < 0) return '-';
		const diffMin = Math.floor(diffMs / 60000);
		if (diffMin < 1) return $t('dashboard.last_run_ago', { time: '<1min' });
		if (diffMin < 60) return $t('dashboard.last_run_ago', { time: `${diffMin}min` });
		const diffH = Math.floor(diffMin / 60);
		if (diffH < 24) return $t('dashboard.last_run_ago', { time: `${diffH}h` });
		return $t('dashboard.last_run_ago', { time: `${Math.floor(diffH / 24)}d` });
	}
</script>

<DashboardCard title={$t('dashboard.learning_title')}>
	<div class="rows">
		<StatusRow label={$t('dashboard.learning_state')} value={stateText} />
		{#if !data.is_disabled && !data.running && !stopped}
			<StatusRow
				label={$t('dashboard.learning_progress')}
				value={$t('dashboard.learning_progress_value', {
					count: data.new_experience_count,
					min: data.min_experiences
				})}
			/>
		{/if}
		<StatusRow label={$t('dashboard.learning_last')} value={formatRelativeTime(data.last_level1_run)} />
		<p class="next" data-testid="learning-next">{nextText}</p>
	</div>
	{#if growth}
		<div class="proposals">
			<StatusSection label={$t('dashboard.learning_proposals')}>
				<GrowthProposals {growth} {showMeasure} onchange={onGrowthChange} />
			</StatusSection>
		</div>
	{/if}
</DashboardCard>

<style>
	.rows {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.next {
		margin: 6px 0 0;
		font-size: 0.8125rem;
		color: var(--text-secondary);
	}
	.proposals {
		margin-top: 14px;
		padding-top: 10px;
		border-top: 1px solid var(--border);
	}
</style>
