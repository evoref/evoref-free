<script lang="ts">
	import MessageList from '$lib/free/components/MessageList.svelte';
	import ChatInput from '$lib/free/components/ChatInput.svelte';
	import { messages, currentMode } from '$lib/free/stores/chat';
	import { themeSlots, layout } from '$lib/free/stores/theme';
	import { t } from '$lib/i18n';
	import { instanceName } from '$lib/free/stores/app';
	import { isPro } from '$lib/edition';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import { onMount } from 'svelte';
	import type { Component } from 'svelte';

	// エディション境界: Free 配下のコンポーネントは $lib/pro を参照できない。
	// route 層 (ここ) が edition-aware composition 層として Pro 専用 CodeMirror UI を
	// `import.meta.glob` で動的ロードする。Free ビルドでは Pro チャンクが entry に含まれない。
	const proLoaders = import.meta.glob<{ default: Component }>(
		'/src/lib/pro/components/{EditorPanel,SplitPane}.svelte'
	);

	let SplitPane: Component | null = $state(null);
	let EditorPanel: Component | null = $state(null);

	onMount(async () => {
		if (!isPro) return;
		const entries = Object.entries(proLoaders);
		for (const [path, loader] of entries) {
			const mod = await loader();
			if (path.endsWith('/SplitPane.svelte')) SplitPane = mod.default;
			else if (path.endsWith('/EditorPanel.svelte')) EditorPanel = mod.default;
		}
	});
</script>

{#snippet chatContent()}
	<!-- chat_header スロット -->
	{#if $themeSlots.chat_header}
		{@const ChatHeader = $themeSlots.chat_header}
		<div class="slot-chat-header">
			<ChatHeader />
		</div>
	{/if}

	{#if $messages.length === 0}
		<div class="empty-state">
			<p>{$t('chat.input_placeholder')}</p>
		</div>
	{:else}
		<MessageList instanceName={$instanceName} />
	{/if}
	<ChatInput />
{/snippet}

<PageLayout fullHeight>
	{#if $currentMode === 'coding' && isPro && SplitPane && EditorPanel}
		{@const SP = SplitPane}
		{@const EP = EditorPanel}
		<SP
			ratio={$layout.coding.pane_ratio}
			direction={$layout.coding.pane_direction}
			onResize={(newRatio: [number, number]) => {
				layout.update(l => ({ ...l, coding: { ...l.coding, pane_ratio: newRatio } }));
			}}
		>
			{#snippet left()}
				<div class="chat-page">
					{@render chatContent()}
				</div>
			{/snippet}
			{#snippet right()}
				<EP />
			{/snippet}
		</SP>
	{:else}
		<div class="chat-page">
			{@render chatContent()}
		</div>
	{/if}
</PageLayout>

<style>
	.chat-page {
		display: flex;
		flex-direction: column;
		height: 100%;
	}
	.empty-state {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-secondary);
		font-size: 14px;
	}
	.slot-chat-header {
		flex-shrink: 0;
	}
</style>
