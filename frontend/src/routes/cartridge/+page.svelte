<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import CartridgeManager from '$lib/free/components/CartridgeManager.svelte';
	import { isPro } from '$lib/edition';

	// エディション境界: Free 配下のコンポーネントは $lib/pro を参照できない。
	// route 層 (ここ) が edition-aware composition 層として Pro コンポーネントを
	// `import.meta.glob` で動的ロードし、snippet 経由で子に注入する。
	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/CartridgeCreateDialog.svelte'
	);

	let CartridgeCreateDialog: Component | null = $state(null);
	let showCreateDialog = $state(false);

	onMount(async () => {
		if (!isPro) return;
		const entries = Object.values(proLoaders);
		if (entries.length === 0) return;
		const mod = await entries[0]();
		CartridgeCreateDialog = mod.default;
	});
</script>

<PageLayout title={$t('sidebar.cartridges')}>
	<CartridgeManager>
		{#snippet headerExtra({ refresh }: { refresh: () => Promise<void> })}
			{#if isPro && CartridgeCreateDialog}
				<button class="create-btn" onclick={() => (showCreateDialog = true)}>
					{$t('cartridge.create')}
				</button>
				{#if showCreateDialog}
					{@const Dialog = CartridgeCreateDialog}
					<Dialog
						onClose={() => (showCreateDialog = false)}
						onCreated={refresh}
					/>
				{/if}
			{/if}
		{/snippet}
	</CartridgeManager>
</PageLayout>

<style>
	.create-btn {
		padding: 6px 14px;
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		cursor: pointer;
		font-size: 0.95rem;
	}
	.create-btn:hover {
		opacity: 0.85;
	}
</style>
