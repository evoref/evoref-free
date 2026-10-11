<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import ProjectMapStatus from '$lib/free/components/ProjectMapStatus.svelte';
	import CartridgeManager from '$lib/free/components/CartridgeManager.svelte';
	import CartridgeInstallButton from '$lib/free/components/CartridgeInstallButton.svelte';
	import Button from '$lib/free/components/ui/Button.svelte';
	import { isPro } from '$lib/edition';
	import { proActive } from '$lib/free/stores/server';

	// エディション境界: Free 配下のコンポーネントは $lib/pro を参照できない。
	// route 層 (ここ) が edition-aware composition 層として Pro コンポーネントを
	// `import.meta.glob` で動的ロードし、snippet 経由で子に注入する。
	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/CartridgeCreateDialog.svelte'
	);
	const trustLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/CartridgeTrustPanel.svelte'
	);
	// チーム知識ハブの状態 (docs/p_10 §8)。読むだけで、操作は CLI (`evoref hub ...`)
	const hubLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/TeamHubPanel.svelte'
	);

	let CartridgeCreateDialog: Component | null = $state(null);
	let CartridgeTrustPanel: Component | null = $state(null);
	let TeamHubPanel: Component | null = $state(null);
	let showCreateDialog = $state(false);
	let manager: ReturnType<typeof CartridgeManager> | undefined = $state();

	onMount(async () => {
		if (!isPro) return;
		const entries = Object.values(proLoaders);
		if (entries.length > 0) {
			const mod = await entries[0]();
			CartridgeCreateDialog = mod.default;
		}
		const trust = Object.values(trustLoaders);
		if (trust.length > 0) {
			CartridgeTrustPanel = (await trust[0]()).default;
		}
		const hub = Object.values(hubLoaders);
		if (hub.length > 0) {
			TeamHubPanel = (await hub[0]()).default;
		}
	});
</script>

<PageLayout title={$t('sidebar.cartridges')}>
	{#snippet description()}{$t('cartridge.page_description')}{/snippet}
	{#snippet actions()}
		<CartridgeInstallButton onInstalled={() => manager?.refresh()} />
	{/snippet}
	<CartridgeManager bind:this={manager}>
		{#snippet headerExtra({ refresh }: { refresh: () => Promise<void> })}
			{#if $proActive && CartridgeCreateDialog}
				<Button size="sm" onclick={() => (showCreateDialog = true)}>
					{$t('cartridge.create')}
				</Button>
				{#if showCreateDialog}
					{@const Dialog = CartridgeCreateDialog}
					<Dialog onClose={() => (showCreateDialog = false)} onCreated={refresh} />
				{/if}
			{/if}
		{/snippet}
	</CartridgeManager>
	{#if $proActive && CartridgeTrustPanel}
		{@const TrustPanel = CartridgeTrustPanel}
		<TrustPanel />
	{/if}
	{#if $proActive && TeamHubPanel}
		{@const HubPanel = TeamHubPanel}
		<HubPanel />
	{/if}
	<ProjectMapStatus />
</PageLayout>
