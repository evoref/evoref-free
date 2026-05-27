<script lang="ts">
	import type { Snippet } from 'svelte';

	interface StatusSectionProps {
		label: string;
		children: Snippet;
	}

	let { label, children }: StatusSectionProps = $props();
</script>

<div class="status-section">
	<div class="section-label">{label}</div>
	{@render children()}
</div>

<style>
	.status-section {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.section-label {
		font-size: 0.75rem;
		font-weight: 600;
		color: var(--text-secondary);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
</style>
