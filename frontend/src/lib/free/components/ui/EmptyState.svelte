<script lang="ts">
	/** 一覧が空のときの案内。次の一手 (ボタン等) があれば children で渡す。 */
	import type { Snippet } from 'svelte';

	let { message, children }: { message: string; children?: Snippet } = $props();
</script>

<div class="empty-state">
	<p class="message">{message}</p>
	{#if children}
		<div class="next">{@render children()}</div>
	{/if}
</div>

<style>
	.empty-state {
		padding: var(--space-5) var(--space-4);
		border: 1px dashed var(--input-border);
		border-radius: var(--border-radius);
		text-align: center;
	}
	.message {
		margin: 0;
		color: var(--text-secondary);
	}
	.next {
		margin-top: var(--space-3);
	}
</style>
