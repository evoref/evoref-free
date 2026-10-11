<script lang="ts">
	import type { Component } from 'svelte';
	import { t } from '$lib/i18n';
	import ProOnlyPage from '$lib/free/components/ProOnlyPage.svelte';

	// Pro 専用 (Free 版・Free バックエンドではトップへ戻す)。本体は route 層で見つけて渡す
	const loader = Object.values(
		import.meta.glob<{ default: Component }>('/src/lib/pro/components/KnowledgeSources.svelte')
	)[0];
</script>

<ProOnlyPage title={$t('sidebar.knowledge_sources')} {loader} />
