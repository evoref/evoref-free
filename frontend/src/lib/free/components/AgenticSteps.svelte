<script lang="ts">
	import { t } from '$lib/i18n';
	import { layout } from '$lib/free/stores/theme';
	import { isStreaming } from '$lib/free/stores/chat';
	import type { AgenticStep, StepResult } from '$lib/free/stores/chat';

	let { steps = [], results = [], showSpinner = false }: { steps?: AgenticStep[]; results?: StepResult[]; showSpinner?: boolean } = $props();
	let showMode = $derived($layout.chat.show_agentic_steps);
	let expanded = $state(false);

	/** 全ステップの合計経過時間 (秒) */
	let totalElapsed = $derived(
		steps.reduce((sum, s) => sum + (s.elapsed_ms ?? 0), 0) / 1000
	);

	/** 全ステップが完了しているか */
	let allDone = $derived(
		!$isStreaming && steps.length > 0 && steps.every((s) => s.status === 'done')
	);

	function formatElapsed(ms?: number): string {
		if (ms == null) return '';
		return (ms / 1000).toFixed(1);
	}

	$effect(() => {
		expanded = showMode === 'expanded';
	});
</script>

{#if showMode !== 'hidden' && (steps.length > 0 || results.length > 0 || showSpinner)}
	<div class="agentic-steps">
		{#if showSpinner && !steps.length}
			<div class="thinking-spinner" aria-label={$t('chat.thinking')}>
				<span class="spinner-dot"></span>
				<span class="spinner-dot"></span>
				<span class="spinner-dot"></span>
			</div>
		{/if}
		{#if steps.length > 0}
			<button class="toggle" onclick={() => (expanded = !expanded)}>
				<span class="arrow" class:open={expanded}>&#9654;</span>
				{$t('chat.agentic_steps')} ({steps.length})
				{#if allDone}
					<span class="done-label">— {$t('chat.agentic_steps_done', { elapsed: totalElapsed.toFixed(1) + 's' })}</span>
				{/if}
			</button>
			{#if expanded}
				<ul class="steps-list">
					{#each steps as step}
						<li class="step-item">
							<span class="step-marker">{step.status === 'done' ? $t('chat.step_marker_done') : $t('chat.step_marker_active')}</span>
							<span class="step-type">{step.type}</span>
							<span class="step-detail">{step.detail}</span>
							{#if step.elapsed_ms != null}
								<span class="step-elapsed">{$t('chat.agentic_step_elapsed', { elapsed: formatElapsed(step.elapsed_ms) })}</span>
							{/if}
						</li>
					{/each}
				</ul>
			{/if}
			{#if showSpinner}
				<div class="thinking-spinner" aria-label={$t('chat.thinking')}>
					<span class="spinner-dot"></span>
					<span class="spinner-dot"></span>
					<span class="spinner-dot"></span>
				</div>
			{/if}
		{/if}
		{#if results.length > 0}
			<div class="results-list">
				{#each results as result}
					<div class="result-item">
						<span class="result-status" class:failed={result.status === 'failed'}>{result.status === 'failed' ? $t('chat.step_marker_failed') : $t('chat.step_marker_done')}</span>
						<span class="result-text">{result.detail}</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<style>
	.agentic-steps {
		margin-bottom: 0;
		padding: 0 14px 14px;
		font-size: 0.78rem;
		background-color: transparent;
		border-radius: var(--border-radius);
		border-bottom-left-radius: 2px;
	}
	.toggle {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		padding: 2px 0;
		font-size: 0.78rem;
		display: flex;
		align-items: center;
		gap: 4px;
		opacity: 0.6;
	}
	.toggle:hover {
		opacity: 0.85;
		color: var(--accent);
	}
	.arrow {
		display: inline-block;
		font-size: 0.45rem;
		transition: transform 0.2s;
	}
	.arrow.open {
		transform: rotate(90deg);
	}
	.steps-list {
		list-style: none;
		padding: 4px 0 0 16px;
		margin: 0;
	}
	.step-item {
		display: flex;
		align-items: baseline;
		gap: 6px;
		padding: 2px 0;
		color: var(--text-secondary);
		opacity: 0.55;
	}
	.step-marker {
		color: var(--accent);
		font-size: 0.6rem;
		flex-shrink: 0;
	}
	.step-type {
		font-weight: 600;
	}
	.step-detail {
		opacity: 0.8;
	}
	.step-elapsed {
		margin-left: auto;
		opacity: 0.5;
		font-size: 0.72rem;
		white-space: nowrap;
	}
	.done-label {
		opacity: 0.7;
		font-weight: normal;
	}
	.results-list {
		padding-top: 4px;
	}
	.result-item {
		color: var(--text-secondary);
		font-size: 0.82rem;
		padding: 2px 0;
		line-height: 1.5;
		display: flex;
		align-items: baseline;
		gap: 4px;
	}
	.result-status {
		color: #22c55e;
		font-weight: 700;
		flex-shrink: 0;
	}
	.result-status.failed {
		color: #ef4444;
	}
	.result-text {
		opacity: 0.85;
	}
	.thinking-spinner {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 4px 0 2px;
	}
	.spinner-dot {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background-color: var(--text-secondary);
		opacity: 0.5;
		animation: dot-pulse 1.4s ease-in-out infinite;
	}
	.spinner-dot:nth-child(2) {
		animation-delay: 0.2s;
	}
	.spinner-dot:nth-child(3) {
		animation-delay: 0.4s;
	}
	@keyframes dot-pulse {
		0%, 80%, 100% {
			opacity: 0.3;
			transform: scale(0.8);
		}
		40% {
			opacity: 1;
			transform: scale(1);
		}
	}
</style>
