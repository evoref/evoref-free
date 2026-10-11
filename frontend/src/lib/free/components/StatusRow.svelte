<script lang="ts">
	interface StatusRowProps {
		label: string;
		value: string;
	}

	let { label, value }: StatusRowProps = $props();
</script>

<div class="status-row">
	<span class="label">{label}</span>
	<span class="value">{value}</span>
</div>

<style>
	.status-row {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 12px;
		font-size: 0.875rem;
		padding: 2px 0;
	}
	.label {
		color: var(--text-secondary);
		flex-shrink: 0;
	}
	.value {
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
		text-align: right;
		overflow-wrap: anywhere;
	}
</style>
