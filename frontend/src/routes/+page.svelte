<script lang="ts">
	import MessageList from '$lib/free/components/MessageList.svelte';
	import ChatInput from '$lib/free/components/ChatInput.svelte';
	import { messages, currentMode, switchMode } from '$lib/free/stores/chat';
	import { activeCreateRun } from '$lib/free/stores/createRun';
	import { get } from 'svelte/store';
	import { t } from '$lib/i18n';
	import { instanceName } from '$lib/free/stores/app';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import { onMount } from 'svelte';
	import { reattachCreateRun } from '$lib/free/services/createReattach';

	onMount(async () => {
		// クリエイト run の再接続 (f_05 §4.5)。
		// リロード後は既定の chat モードで開くので、保存された run があれば
		// 先に create へ切り替える (バックエンドが既に create ならモデル入替は無い)。
		if (get(activeCreateRun)) {
			if (get(currentMode) !== 'create') {
				await switchMode('create');
			}
			if (get(currentMode) === 'create') {
				reattachCreateRun();
			}
		}
	});
</script>

{#snippet chatContent()}
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
	<div class="chat-page">
		{@render chatContent()}
	</div>
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
		color: var(--text-muted);
		font-size: calc(var(--font-size-base) + 6px);
		font-weight: 500;
		letter-spacing: -0.01em;
	}
</style>
