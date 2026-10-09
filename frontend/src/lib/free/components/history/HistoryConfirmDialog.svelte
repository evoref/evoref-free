<script lang="ts">
	/**
	 * 履歴の破壊操作 (一括削除 / 圧縮) の確認ダイアログ。
	 *
	 * @param title - 見出し (i18n 解決済み)
	 * @param message - 影響の説明 (i18n 解決済み)
	 * @param confirmLabel - 実行ボタンの文言 (i18n 解決済み)
	 * @param busy - 実行中。ボタンを無効にし、外側クリックでは閉じない
	 */
	import { t } from '$lib/i18n';
	import DialogShell from '../DialogShell.svelte';

	interface Props {
		title: string;
		message: string;
		confirmLabel: string;
		busy?: boolean;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { title, message, confirmLabel, busy = false, onConfirm, onCancel }: Props = $props();
</script>

<DialogShell ariaLabel={title} onClose={onCancel} canCloseOnOverlayClick={!busy}>
	<h3 class="dialog-title">{title}</h3>
	<p class="dialog-message">{message}</p>
	<div class="dialog-actions">
		<button class="btn btn-cancel" onclick={onCancel} disabled={busy}>
			{$t('common.cancel')}
		</button>
		<button class="btn btn-danger" onclick={onConfirm} disabled={busy}>
			{confirmLabel}
		</button>
	</div>
</DialogShell>

<style>
	.dialog-title {
		margin: 0 0 12px;
		font-size: 1.1rem;
		color: var(--text-primary);
	}
	.dialog-message {
		margin: 0 0 20px;
		color: var(--text-primary);
		line-height: 1.5;
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
	}
	.btn {
		padding: 8px 16px;
		border-radius: var(--border-radius);
		font-size: 0.95rem;
		cursor: pointer;
		border: none;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn-cancel {
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}
	.btn-danger {
		background-color: var(--error, #c0392b);
		color: #fff;
	}
</style>
