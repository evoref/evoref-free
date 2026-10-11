<script lang="ts">
	/**
	 * Pro 専用ページの枠。Free 版 (またはバックエンドが Free) ではトップへ戻す。
	 *
	 * エディション境界: このコンポーネントは $lib/pro を知らない。Pro の本体は route 層が
	 * `import.meta.glob` で見つけたローダを `loader` で渡す (見つからなければ undefined)。
	 */
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';
	import { goto } from '$app/navigation';
	import { isPro } from '$lib/edition';
	import { proActive } from '$lib/free/stores/server';
	import PageLayout from './PageLayout.svelte';
	import Loading from './ui/Loading.svelte';

	let {
		title,
		loader
	}: {
		title: string;
		loader: (() => Promise<{ default: Component }>) | undefined;
	} = $props();

	let Body: Component | null = $state(null);

	// $effect はブラウザでだけ走る (SSR 中の goto は 500 になるので onMount / $effect に置く)
	$effect(() => {
		if (!$proActive) goto('/');
	});

	onMount(async () => {
		if (!isPro) return;
		if (!loader) {
			goto('/');
			return;
		}
		Body = (await loader()).default;
	});
</script>

<PageLayout {title}>
	{#if Body && $proActive}
		<Body />
	{:else if isPro}
		<Loading />
	{/if}
</PageLayout>
