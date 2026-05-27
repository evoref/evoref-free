<script lang="ts">
	/**
	 * モーダルダイアログの共通シェル
	 *
	 * CartridgeDetailDialog / CartridgeCreateDialog (pro) / ThemeTrustDialog で
	 * 重複していた以下のロジックを集約する:
	 *
	 * - overlay (`<div class="overlay" role="dialog" aria-modal="true">`) と
	 *   外側クリックでの close
	 * - Escape キーでの close
	 * - Tab / Shift+Tab でのフォーカストラップ (focusable 要素を循環)
	 * - 初期表示時のフォーカス (button or input)
	 * - dialog 本体の box (background / border / padding / shadow / scroll)
	 *
	 * 中身 (タイトル・本文・ボタン等) は `children` snippet で渡す。
	 *
	 * @param ariaLabel - aria-label に渡すテキスト (i18n 解決済み)
	 * @param onClose - close ハンドラ。Escape / overlay click / 内部の close ボタンから呼ぶ
	 * @param minWidth - dialog の最小幅 (デフォルト "360px")
	 * @param maxWidth - dialog の最大幅 (デフォルト "520px")
	 * @param canCloseOnOverlayClick - false にすると overlay click を無視 (ボタン操作中など)。
	 *                                 Escape は常に有効 (既存挙動と同一)
	 * @param initialFocus - 初期フォーカスする要素 ("button" | "input")
	 */
	import type { Snippet } from 'svelte';

	interface Props {
		ariaLabel: string;
		onClose: () => void;
		minWidth?: string;
		maxWidth?: string;
		canCloseOnOverlayClick?: boolean;
		initialFocus?: 'button' | 'input';
		children: Snippet;
	}

	let {
		ariaLabel,
		onClose,
		minWidth = '360px',
		maxWidth = '520px',
		canCloseOnOverlayClick = true,
		initialFocus = 'button',
		children
	}: Props = $props();

	let dialogEl: HTMLDivElement | undefined = $state();

	function handleOverlayKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			onClose();
			return;
		}
		if (e.key === 'Tab' && dialogEl) {
			const focusable = dialogEl.querySelectorAll<HTMLElement>(
				'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
			);
			if (focusable.length === 0) return;
			const first = focusable[0];
			const last = focusable[focusable.length - 1];
			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
	}

	$effect(() => {
		if (dialogEl) {
			const selector = initialFocus === 'input' ? 'input' : 'button';
			const el = dialogEl.querySelector<HTMLElement>(selector);
			el?.focus();
		}
	});
</script>

<div
	class="overlay"
	role="dialog"
	aria-modal="true"
	aria-label={ariaLabel}
	tabindex="-1"
	onclick={(e) => {
		if (e.target === e.currentTarget && canCloseOnOverlayClick) onClose();
	}}
	onkeydown={handleOverlayKeydown}
>
	<div
		class="dialog"
		bind:this={dialogEl}
		style="--dialog-min-width: {minWidth}; --dialog-max-width: {maxWidth};"
	>
		{@render children()}
	</div>
</div>

<style>
	.overlay {
		position: fixed;
		inset: 0;
		background-color: var(--overlay-bg);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
	}
	.dialog {
		background-color: var(--bg-primary);
		border: 1px solid var(--border);
		border-radius: calc(var(--border-radius) * 2);
		padding: 24px;
		min-width: var(--dialog-min-width);
		max-width: var(--dialog-max-width);
		max-height: 85vh;
		overflow-y: auto;
		box-shadow: 0 8px 32px var(--shadow-dialog);
	}
</style>
