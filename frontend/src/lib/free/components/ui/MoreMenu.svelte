<script lang="ts">
	/**
	 * 行内の「その他の操作」。頻度の低い操作・危険な操作をここへ下げ、行には主操作だけを残す。
	 * 中の項目 (children) を押すと閉じる。外側のクリックと Escape でも閉じる。
	 */
	import type { Snippet } from 'svelte';
	import { t } from '$lib/i18n';

	let { label, children }: { label?: string; children: Snippet } = $props();

	let open = $state(false);
	let root: HTMLDivElement | undefined = $state();

	function onWindowClick(e: MouseEvent) {
		if (open && root && !root.contains(e.target as Node)) open = false;
	}
</script>

<svelte:window onclick={onWindowClick} />

<div class="more-menu" bind:this={root}>
	<button
		type="button"
		class="trigger"
		aria-label={label ?? $t('common.more_actions')}
		title={label ?? $t('common.more_actions')}
		aria-expanded={open}
		onclick={() => (open = !open)}
	>
		<span aria-hidden="true">⋯</span>
	</button>
	{#if open}
		<!-- 項目の押下で閉じる。キーボードの決定もボタンの click として届く -->
		<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
		<div
			class="panel"
			onclick={() => (open = false)}
			onkeydown={(e) => {
				if (e.key === 'Escape') open = false;
			}}
		>
			{@render children()}
		</div>
	{/if}
</div>

<style>
	.more-menu {
		position: relative;
		display: inline-block;
	}
	.trigger {
		width: 30px;
		height: 28px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: 1px solid transparent;
		border-radius: var(--border-radius);
		color: var(--text-secondary);
		font-size: var(--font-size-lg);
		cursor: pointer;
	}
	.trigger:hover,
	.trigger[aria-expanded='true'] {
		border-color: var(--input-border);
		color: var(--text-primary);
	}
	.panel {
		position: absolute;
		right: 0;
		top: calc(100% + 4px);
		z-index: 20;
		min-width: 160px;
		display: flex;
		flex-direction: column;
		padding: var(--space-1);
		background-color: var(--bg-primary);
		border: 1px solid var(--input-border);
		border-radius: var(--border-radius);
		box-shadow: 0 4px 16px var(--shadow-dialog);
	}
	/* 項目は呼び出し側の Button (ghost) を左寄せの 1 列に並べる */
	.panel :global(.btn) {
		justify-content: flex-start;
		width: 100%;
	}
</style>
