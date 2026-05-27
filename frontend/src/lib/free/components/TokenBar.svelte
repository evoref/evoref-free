<script lang="ts">
	import { t } from '$lib/i18n';

	let { used = 0, limit = 4096, speed = 0 }: { used?: number; limit?: number; speed?: number } = $props();
	let pct = $derived(Math.min(100, Math.round((used / limit) * 100)));
	let color = $derived(pct >= 95 ? 'critical' : pct >= 80 ? 'warning' : 'normal');
</script>

<div class="token-bar" title={$t('token_bar.warning_tooltip')}>
	{#if speed > 0}
		<span class="info">{speed.toFixed(1)} token/s : </span>
	{/if}
	<span class="info">{$t('token_bar.usage', { used, limit })}</span>
	<div
		class="bar-track"
		role="progressbar"
		aria-label={$t('token_bar.progress_label')}
		aria-valuenow={used}
		aria-valuemin={0}
		aria-valuemax={limit}
	>
		<div class="bar-fill {color}" style="width: {pct}%"></div>
	</div>
	<span class="pct {color}">{pct}%</span>
</div>

<style>
	.token-bar {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: 6px;
		font-size: 12px;
		color: var(--text-secondary);
	}
	.info {
		white-space: nowrap;
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
	}
	.bar-track {
		flex: 1;
		height: 5px;
		background: var(--border);
		border-radius: 3px;
		overflow: hidden;
	}
	.bar-fill {
		height: 100%;
		border-radius: 3px;
		transition: width 0.3s ease;
	}
	.bar-fill.normal {
		background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 65%, #fff));
	}
	.bar-fill.warning {
		background-color: var(--color-warning);
	}
	.bar-fill.critical {
		background-color: var(--color-error);
	}
	.pct {
		min-width: 3em;
		text-align: right;
		color: var(--accent);
	}
	.pct.warning {
		color: var(--color-warning);
	}
	.pct.critical {
		color: var(--color-error);
	}
</style>
