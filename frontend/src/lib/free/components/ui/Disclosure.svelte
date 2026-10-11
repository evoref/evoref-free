<script lang="ts">
	/** 折りたたみ。一般の利用者に要らない詳細を既定で閉じておく (<details> そのもの)。 */
	import type { Snippet } from 'svelte';

	let {
		label,
		open = $bindable(false),
		children
	}: {
		label: string;
		open?: boolean;
		children: Snippet;
	} = $props();
</script>

<details class="disclosure" bind:open>
	<summary>{label}</summary>
	<div class="body">{@render children()}</div>
</details>

<style>
	.disclosure {
		border-top: 1px solid var(--border);
	}
	summary {
		padding: var(--space-2) 0;
		color: var(--text-secondary);
		font-size: var(--font-size-sm);
		cursor: pointer;
		user-select: none;
	}
	summary:hover {
		color: var(--text-primary);
	}
	.disclosure[open] > summary {
		color: var(--text-primary);
		font-weight: 600;
	}
	.body {
		padding: 0 0 var(--space-3);
	}
</style>
