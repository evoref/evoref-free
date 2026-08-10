<script lang="ts">
	import { t } from '$lib/i18n';
	import MessageBubble from './MessageBubble.svelte';
	import { messages, isStreaming, currentMode } from '$lib/free/stores/chat';
	import { themeSlots } from '$lib/free/stores/theme';

	let { instanceName = 'evoref' }: { instanceName?: string } = $props();
	let container: HTMLDivElement | undefined = $state();

	$effect(() => {
		// メッセージ変更時に自動スクロール
		if ($messages.length && container) {
			const rafId = requestAnimationFrame(() => {
				if (container) {
					container.scrollTop = container.scrollHeight;
				}
			});
			return () => cancelAnimationFrame(rafId);
		}
	});
</script>

<div
	class="message-list"
	class:constrained={$currentMode !== 'create'}
	role="log"
	aria-live="polite"
	aria-label={$t('chat.message_list')}
	bind:this={container}
>
	{#each $messages as message (message.id)}
		<!-- message_prefix スロット -->
		{#if $themeSlots.message_prefix}
			{@const MessagePrefix = $themeSlots.message_prefix}
			<MessagePrefix {message} />
		{/if}
		<MessageBubble {message} {instanceName} streaming={$isStreaming && message === $messages[$messages.length - 1]} mode={$currentMode} />
	{/each}
</div>

<style>
	.message-list {
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		padding: 16px;
		gap: 4px;
		width: 100%;
	}
	.message-list.constrained {
		max-width: 960px;
		margin-left: auto;
		margin-right: auto;
	}
</style>
