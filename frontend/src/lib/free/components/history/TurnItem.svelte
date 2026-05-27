<script lang="ts">
	import { t } from '$lib/i18n';
	import { formatTime } from '$lib/free/utils/format';
	import type { TurnData } from '$lib/free/types/history';
	import { COPY_NOTIFICATION_TIMEOUT_MS } from '$lib/free/constants';

	interface Props {
		turn: TurnData;
		instanceName: string;
	}

	let { turn, instanceName }: Props = $props();

	let copyStatus = $state<'idle' | 'copied' | 'failed'>('idle');

	async function copyContent() {
		try {
			await navigator.clipboard.writeText(turn.content);
			copyStatus = 'copied';
		} catch {
			copyStatus = 'failed';
		}
		setTimeout(() => { copyStatus = 'idle'; }, COPY_NOTIFICATION_TIMEOUT_MS);
	}

	let copyLabel = $derived(
		copyStatus === 'copied' ? $t('history_page.copied')
		: copyStatus === 'failed' ? $t('history_page.copy_failed')
		: $t('history_page.copy_turn')
	);

	let roleLabel = $derived(
		turn.role === 'user'
			? $t('history_page.user_label')
			: (instanceName || $t('history_page.assistant_label'))
	);
</script>

<div class="turn" class:turn-user={turn.role === 'user'} class:turn-assistant={turn.role === 'assistant'}>
	<div class="turn-header">
		<span class="turn-role" class:role-user={turn.role === 'user'} class:role-assistant={turn.role === 'assistant'}>
			{roleLabel}
		</span>
		{#if turn.timestamp}
			<span class="turn-time">{formatTime(turn.timestamp)}</span>
		{/if}
		<button
			class="copy-btn"
			onclick={copyContent}
			aria-label={$t('history_page.copy_turn')}
		>
			{copyLabel}
		</button>
	</div>
	<div class="turn-content">{turn.content}</div>
</div>

<style>
	.turn {
		padding: 10px 12px;
		border-radius: 6px;
	}
	.turn-user {
		background: color-mix(in srgb, var(--accent) 22%, transparent);
	}
	:global([data-theme='dark']) .turn-user {
		background: color-mix(in srgb, var(--accent) 32%, transparent);
	}
	.turn-assistant {
		background: transparent;
	}
	.turn-header {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 4px;
	}
	.turn-role {
		font-size: 14px;
		font-weight: 700;
	}
	.role-user {
		color: color-mix(in srgb, var(--accent) 90%, transparent);
	}
	.role-assistant {
		color: var(--text-secondary);
	}
	.turn-time {
		font-size: 11px;
		color: var(--text-secondary);
		opacity: 0.7;
	}
	.copy-btn {
		margin-left: auto;
		font-size: 11px;
		color: var(--text-secondary);
		background: none;
		border: none;
		cursor: pointer;
		padding: 1px 4px;
		border-radius: 3px;
		opacity: 0;
		transition: opacity 0.15s;
	}
	.turn:hover .copy-btn,
	.copy-btn:focus-visible {
		opacity: 1;
	}
	.turn-content {
		font-size: 14px;
		color: var(--text-primary);
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
