<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import type { ChatMessage } from '$lib/free/stores/chat';
	import AgenticSteps from './AgenticSteps.svelte';
	import RagDebugPanel from './RagDebugPanel.svelte';
	import SourcesPanel from './SourcesPanel.svelte';
	import MarkdownRenderer from './MarkdownRenderer.svelte';
	import { formatTime } from '$lib/free/utils/format';
	import { sendTurnFeedback } from '$lib/free/api';
	import { sessionId, messages } from '$lib/free/stores/chat';
	import { get } from 'svelte/store';

	let {
		message,
		instanceName = 'evoref',
		streaming = false,
		mode = 'chat',
		prevUserQuery = ''
	}: {
		message: ChatMessage;
		instanceName?: string;
		streaming?: boolean;
		mode?: string;
		prevUserQuery?: string;
	} = $props();
	// 明示評価 (f_04 §3.2.3): 👎 は本人が失敗と言った唯一の信号で、Level 1 の
	// 採用ゲートのケースに最優先で使われる。👍 は記録のみ。
	let noteOpen = $state(false);
	let note = $state('');
	let sending = $state(false);
	let feedbackFailed = $state(false);

	async function submitFeedback(verdict: 'negative' | 'positive') {
		if (sending) return;
		sending = true;
		feedbackFailed = false;
		const res = await sendTurnFeedback(get(sessionId), verdict, prevUserQuery || undefined, note);
		sending = false;
		if (!res.recorded) {
			feedbackFailed = true;
			return;
		}
		messages.update((list) =>
			list.map((m) => (m.id === message.id ? { ...m, feedback: verdict } : m))
		);
		noteOpen = false;
		note = '';
	}
	let isUser = $derived(message.role === 'user');
	let isCreate = $derived(mode === 'create');
	// クリエイトモードではコードをエディタへ流すため、チャット側は
	// プレースホルダに置換する。'chat' 明示指示時のみコードを表示。
	let suppressCode = $derived(isCreate && message.editor_route !== 'chat');
	let showTimestamp = $derived($layout.chat.show_timestamps);
	let showSpinner = $derived(!isUser && streaming && message.content === '');
</script>

<div class="message-bubble" class:user={isUser} class:assistant={!isUser} class:create={isCreate}>
	{#if !isCreate}
		<div class="message-header" class:header-right={isUser}>
			<span class="role-label">{isUser ? $t('chat.you') : instanceName}</span>
			{#if showTimestamp}
				<span class="timestamp">&nbsp;{formatTime(message.timestamp)}</span>
			{/if}
		</div>
	{/if}

	{#if !isUser && (message.agentic_steps?.length || message.step_results?.length || message.long_form_progress || showSpinner)}
		<AgenticSteps steps={message.agentic_steps} results={message.step_results} progress={message.long_form_progress} {showSpinner} {streaming} />
	{/if}

	{#if !showSpinner}
		<div class="message-content">
			{#if isUser}
				<p class="whitespace-pre-wrap">{message.content}</p>
			{:else}
				<MarkdownRenderer content={message.content} {suppressCode} />
			{/if}
		</div>
	{/if}

	{#if isUser && message.input_truncated}
		<p class="truncation-notice">
			{$t('chat.input_truncated', {
				original: message.input_truncated.original_chars,
				sent: message.input_truncated.sent_chars
			})}
		</p>
	{/if}

	{#if !isUser && message.output_truncated}
		<p class="truncation-notice">
			{$t('chat.output_truncated')}
		</p>
	{/if}

	{#if !isUser && message.sources && message.sources.items.length > 0}
		<SourcesPanel sources={message.sources} />
	{/if}

	{#if !isUser && message.rag_debug}
		<RagDebugPanel ragDebug={message.rag_debug} />
	{/if}

	{#if !isUser && !streaming && message.content}
		<div class="feedback-row">
			{#if message.feedback}
				<span class="feedback-sent">{message.feedback === 'negative' ? $t('chat.feedback_sent_bad') : $t('chat.feedback_sent_good')}</span>
			{:else}
				<button
					type="button"
					class="feedback-btn"
					title={$t('chat.feedback_good')}
					aria-label={$t('chat.feedback_good')}
					disabled={sending}
					onclick={() => submitFeedback('positive')}
				>👍</button>
				<button
					type="button"
					class="feedback-btn feedback-bad"
					title={$t('chat.feedback_bad')}
					aria-label={$t('chat.feedback_bad')}
					disabled={sending}
					onclick={() => (noteOpen = !noteOpen)}
				>👎</button>
				{#if noteOpen}
					<input
						class="feedback-note"
						type="text"
						maxlength="400"
						placeholder={$t('chat.feedback_note_placeholder')}
						bind:value={note}
						onkeydown={(e) => {
							if (e.key === 'Enter') submitFeedback('negative');
						}}
					/>
					<button
						type="button"
						class="feedback-btn feedback-send"
						disabled={sending}
						onclick={() => submitFeedback('negative')}
					>{$t('chat.feedback_send')}</button>
				{/if}
				{#if feedbackFailed}
					<span class="feedback-sent">{$t('chat.feedback_failed')}</span>
				{/if}
			{/if}
		</div>
	{/if}

</div>

<style>
	.feedback-row {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 6px;
		font-size: 0.8rem;
		color: var(--text-secondary, #888);
	}
	.feedback-btn {
		background: transparent;
		border: 1px solid var(--border, #ccc);
		border-radius: 4px;
		padding: 1px 6px;
		cursor: pointer;
		opacity: 0.7;
	}
	.feedback-btn:hover:not(:disabled) {
		opacity: 1;
	}
	.feedback-note {
		flex: 1;
		min-width: 120px;
		padding: 2px 6px;
		border: 1px solid var(--border, #ccc);
		border-radius: 4px;
		background: var(--bg-primary, transparent);
		color: var(--text-primary);
	}
	.message-bubble {
		max-width: 85%;
		padding: 10px 14px;
		border-radius: var(--border-radius);
		margin-bottom: 12px;
	}
	/* 切り詰め注記 (入力側は該当 user 発言、出力側は該当 assistant 応答に付く) */
	.truncation-notice {
		margin-top: 6px;
		padding-top: 6px;
		border-top: 1px dashed color-mix(in srgb, var(--text-primary) 25%, transparent);
		font-size: 12px;
		opacity: 0.85;
	}
	/* --- チャットモード --- */
	.message-bubble.user:not(.create) {
		align-self: flex-end;
		background-color: color-mix(in srgb, var(--accent) 18%, var(--bg-secondary));
		color: var(--text-primary);
		border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--border));
		border-bottom-right-radius: 2px;
	}
	.message-bubble.assistant:not(.create) {
		align-self: flex-start;
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border-bottom-left-radius: 2px;
	}
	/* --- クリエイトモード: 横幅いっぱい --- */
	.message-bubble.create {
		max-width: 100%;
		width: 100%;
		margin-bottom: 2px;
	}
	.message-bubble.create.user {
		align-self: flex-start;
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
		border-bottom-left-radius: 2px;
	}
	.message-bubble.create.assistant {
		background-color: transparent;
		border: none;
		padding: 0 0 10px;
	}
	.message-header {
		display: flex;
		align-items: baseline;
		margin-bottom: 4px;
		font-size: 0.85rem;
	}
	.message-header.header-right {
		justify-content: flex-end;
	}
	.role-label {
		font-weight: 600;
		opacity: 0.8;
	}
	.timestamp {
		opacity: 0.6;
		font-size: 0.8rem;
	}
	.message-content {
		line-height: 1.5;
		word-break: break-word;
	}
	/* user メッセージは plain text の <p> のみ。
	 * assistant メッセージの Markdown 系スタイル (h1-h6 / pre / code / table 等) は
	 * MarkdownRenderer.svelte 側で完結させる。 */
	.message-content :global(p) {
		margin: 0 0 8px;
	}
	.message-content :global(p:last-child) {
		margin-bottom: 0;
	}
</style>
