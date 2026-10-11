<script lang="ts">
	/**
	 * Level 2 (成長) の詳細 (docs/f_04 §6.8): 自動採用の状態と理由、回帰台帳の件数、
	 * 直近の成長の台帳。提案の採用・却下は GrowthProposals (学習のサマリ側) が受け持つ。
	 */
	import { t } from '$lib/i18n';
	import StatusSection from '$lib/free/components/StatusSection.svelte';
	import StatusRow from '$lib/free/components/StatusRow.svelte';
	import type { GrowthResponse } from '$lib/free/api';

	let { growth }: { growth: GrowthResponse } = $props();

	const RECENT_ROWS = 5;
	const KNOWN_AUTO_REASONS = [
		'auto_adopt_off',
		'growth_disabled',
		'health_hold',
		'health_stop',
		'health_unavailable',
		'single_slot',
		'insufficient_guards',
		'candidate_guard_failed'
	];

	let autoText = $derived.by(() => {
		const auto = growth.auto_adopt;
		if (auto.effective) return $t('dashboard.growth_auto_effective');
		const reason = KNOWN_AUTO_REASONS.includes(auto.reason)
			? $t(`dashboard.growth_reason_${auto.reason}`)
			: auto.reason;
		return $t(auto.configured ? 'dashboard.growth_auto_waiting' : 'dashboard.growth_auto_off', {
			reason
		});
	});

	let recent = $derived(growth.recent.slice(-RECENT_ROWS).reverse());

	function kindLabel(kind: string): string {
		return kind === 'prompt' || kind === 'generation_params'
			? $t(`dashboard.growth_kind_${kind}`)
			: kind;
	}
</script>

<StatusSection label={$t('dashboard.growth_title')}>
	<StatusRow label={$t('dashboard.growth_auto_adopt')} value={autoText} />
	<StatusRow
		label={$t('dashboard.growth_regression')}
		value={$t('dashboard.growth_regression_value', {
			active: growth.regression.active,
			guards: growth.regression.guards,
			pressure: growth.regression.pressure,
			flaky: growth.regression.flaky
		})}
	/>
	{#if recent.length > 0}
		<div class="recent">
			<span class="sub-label">{$t('dashboard.growth_recent')}</span>
			{#each recent as row (row.row_id)}
				<span class="recent-line">
					{row.written_at.slice(0, 16).replace('T', ' ')} · {row.op} · {kindLabel(row.asset_kind)}
					· {row.mode}{row.decided_by ? ` · ${row.decided_by}` : ''}{row.reason
						? ` · ${row.reason}`
						: ''}
				</span>
			{/each}
		</div>
	{/if}
</StatusSection>

<style>
	.recent {
		display: flex;
		flex-direction: column;
		gap: 2px;
		font-size: 0.75rem;
	}
	.sub-label {
		color: var(--text-secondary);
	}
	.recent-line {
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
		overflow-wrap: anywhere;
	}
</style>
