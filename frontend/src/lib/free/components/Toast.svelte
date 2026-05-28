<script lang="ts">
	import { t } from '$lib/i18n';
	import { toasts, removeToast } from '$lib/free/stores/toast';
</script>

{#if $toasts.length > 0}
	<div class="toast-container" role="status" aria-live="polite">
		{#each $toasts as toast (toast.id)}
			<div class="toast toast--{toast.type}">
				<span class="toast-message">{$t(toast.i18nKey, toast.params)}</span>
				<button
					class="toast-close"
					onclick={() => removeToast(toast.id)}
					aria-label={$t('common.close')}
				>
					&times;
				</button>
			</div>
		{/each}
	</div>
{/if}

<style>
	.toast-container {
		position: fixed;
		top: 16px;
		right: 16px;
		z-index: 9999;
		display: flex;
		flex-direction: column;
		gap: 8px;
		max-width: 400px;
	}
	.toast {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 14px;
		border-radius: var(--border-radius, 8px);
		font-size: 0.95rem;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
		animation: toast-slide-in 0.2s ease-out;
	}
	.toast--error {
		background-color: var(--toast-error-bg, #d32f2f);
		color: var(--toast-error-fg, #fff);
	}
	.toast--success {
		background-color: var(--toast-success-bg, #43a047);
		color: var(--toast-success-fg, #fff);
	}
	.toast--warning {
		background-color: var(--toast-warning-bg, #fb8c00);
		color: var(--toast-warning-fg, #fff);
	}
	.toast--info {
		background-color: var(--toast-info-bg, #4fc3f7);
		color: var(--toast-info-fg, #fff);
	}
	.toast-message {
		flex: 1;
		line-height: 1.4;
	}
	.toast-close {
		background: none;
		border: none;
		color: inherit;
		font-size: 1.2rem;
		cursor: pointer;
		padding: 0 2px;
		opacity: 0.8;
		line-height: 1;
	}
	.toast-close:hover {
		opacity: 1;
	}
	@keyframes toast-slide-in {
		from {
			opacity: 0;
			transform: translateX(20px);
		}
		to {
			opacity: 1;
			transform: translateX(0);
		}
	}
</style>
