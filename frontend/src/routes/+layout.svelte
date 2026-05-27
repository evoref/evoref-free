<script lang="ts">
	import '../app.css';
	import 'uplot/dist/uPlot.min.css';
	import Sidebar from '$lib/free/components/Sidebar.svelte';
	import Toast from '$lib/free/components/Toast.svelte';
	import { colorMode, sidebarCollapsed, initTheme } from '$lib/free/stores/theme';
	import { t } from '$lib/i18n';
	import { onMount, onDestroy } from 'svelte';
	import { startPolling, stopPolling } from '$lib/free/stores/server';
	import { startVramPolling, stopVramPolling } from '$lib/free/stores/vram';
	import { instanceName } from '$lib/free/stores/app';

	let { children } = $props();

	onMount(() => {
		// 狭い画面ではサイドバーをデフォルトで折りたたむ
		if (window.innerWidth <= 900) {
			sidebarCollapsed.set(true);
		}

		// サーバーステータスのポーリング開始（初回即時取得 + 10秒間隔）
		// VRAM 側 (10秒) と揃え、緑ランプとメモリ表示の反映ラグを無くす
		startPolling(10_000);

		// VRAM ステータスのポーリング開始
		startVramPolling();

		// テーマ初期化（バックエンドからアクティブテーマを取得して適用）
		initTheme();
	});

	onDestroy(() => {
		stopPolling();
		stopVramPolling();
	});

	$effect(() => {
		document.documentElement.setAttribute('data-theme', $colorMode);
	});
</script>

<svelte:head>
	<title>{$instanceName} - evoref</title>
</svelte:head>

<div class="app-layout">
	<div class="app-bg"></div>
	<Sidebar instanceName={$instanceName} />
	<main class="main-content">
		{@render children()}
	</main>
</div>

<Toast />

<style>
	.app-layout {
		display: flex;
		height: 100vh;
		overflow: hidden;
		position: relative;
	}
	.app-bg {
		position: absolute;
		inset: 0;
		background: var(--app-bg);
		z-index: 0;
		pointer-events: none;
	}
	.main-content {
		flex: 1;
		overflow-y: auto;
		position: relative;
		z-index: 1;
	}
</style>
