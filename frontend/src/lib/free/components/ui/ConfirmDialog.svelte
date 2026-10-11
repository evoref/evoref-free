<script lang="ts">
	/**
	 * 取り消せない操作 (削除・アンインストール・一括削除・圧縮) の確認ダイアログ。
	 * 初期フォーカスはキャンセル側 (DialogShell は最初のボタンへ寄せる)。
	 *
	 * @param title - 見出し (i18n 解決済み)
	 * @param message - 影響の説明 (i18n 解決済み)
	 * @param confirmLabel - 実行ボタンの文言 (i18n 解決済み)
	 * @param busy - 実行中。ボタンを無効にし、外側クリックでは閉じない
	 */
	import { t } from '$lib/i18n';
	import DialogShell from '../DialogShell.svelte';
	import Button from './Button.svelte';

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
		<Button onclick={onCancel} disabled={busy}>{$t('common.cancel')}</Button>
		<Button variant="danger" onclick={onConfirm} disabled={busy}>{confirmLabel}</Button>
	</div>
</DialogShell>

<style>
	.dialog-title {
		margin: 0 0 var(--space-3);
		font-size: var(--font-size-lg);
		color: var(--text-primary);
	}
	.dialog-message {
		margin: 0 0 var(--space-5);
		color: var(--text-primary);
		line-height: 1.5;
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: var(--space-2);
	}
</style>
