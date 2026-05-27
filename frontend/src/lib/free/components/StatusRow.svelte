<script lang="ts">
	interface StatusRowProps {
		active?: boolean;
		label: string;
		value: string;
		ariaLabel?: string;
	}

	let { active = false, label, value, ariaLabel }: StatusRowProps = $props();
</script>

<div class="status-row">
	<span
		class="indicator"
		class:active
		role="status"
		aria-label={ariaLabel}
	></span>
	<span class="label">{label}</span>
	<span class="value">{value}</span>
</div>

<style>
	.status-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.875rem;
		color: var(--text-primary);
	}
	.label {
		flex: 1;
	}
	.value {
		font-variant-numeric: tabular-nums;
		color: var(--text-secondary);
	}
	.indicator {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background-color: var(--text-secondary);
		opacity: 0.3;
		flex-shrink: 0;
	}
	.indicator.active {
		background-color: var(--color-success);
		opacity: 1;
	}
</style>
