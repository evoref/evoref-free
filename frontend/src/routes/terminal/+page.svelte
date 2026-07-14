<script lang="ts">
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { isPro } from '$lib/edition';
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';

	// Pro ガード: Free 版ではトップにリダイレクト (Pro 専用機能)。
	// goto は SSR 中に呼ぶと 500 になるため browser ガード必須。
	if (browser && !isPro) {
		goto('/');
	}

	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/Terminal/Terminal.svelte'
	);

	let TerminalComponent: Component | null = $state(null);

	onMount(async () => {
		if (!isPro) return;
		const entry = Object.values(proLoaders)[0];
		if (!entry) {
			goto('/');
			return;
		}
		const mod = await entry();
		TerminalComponent = mod.default;
	});
</script>

<PageLayout title={$t('terminal.title')}>
	<div class="terminal-container">
		{#if TerminalComponent}
			<TerminalComponent />
		{:else if isPro}
			<div class="loading">{$t('common.loading')}</div>
		{/if}
	</div>
</PageLayout>

<style>
	.terminal-container {
		height: calc(100vh - 120px);
		min-height: 320px;
	}
	.loading {
		display: flex;
		justify-content: center;
		align-items: center;
		padding: 2rem;
		color: var(--text-secondary);
	}
</style>
