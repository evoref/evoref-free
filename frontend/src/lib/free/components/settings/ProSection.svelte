<script lang="ts">
	import type { Snippet } from 'svelte';
	import { isPro } from '$lib/edition';

	interface Props {
		children: Snippet;
		columns?: 1 | 2;
	}

	let { children, columns = 2 }: Props = $props();
</script>

{#if isPro}
	<div class="pro-section">
		<div class="pro-divider">
			<span class="pro-label">Pro</span>
		</div>
		<div class="pro-content" class:single={columns === 1}>
			{@render children()}
		</div>
	</div>
{/if}

<style>
	.pro-section {
		grid-column: 1 / -1;
	}

	.pro-divider {
		display: flex;
		align-items: center;
		gap: 10px;
		margin: 16px 0 8px;
	}

	.pro-divider::before,
	.pro-divider::after {
		content: '';
		flex: 1;
		height: 0.5px;
		background: var(--accent);
		opacity: 0.4;
	}

	.pro-label {
		font-size: 11px;
		font-weight: 700;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.pro-content {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}

	.pro-content.single {
		grid-template-columns: 1fr;
	}

	@media (max-width: 900px) {
		.pro-content {
			grid-template-columns: 1fr;
		}
	}
</style>
