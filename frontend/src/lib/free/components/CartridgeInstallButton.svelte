<script lang="ts">
	/**
	 * カートリッジ画面の主操作「インストール」。ファイル選択 → 進捗ダイアログ (キャンセル可) →
	 * 成功で onInstalled。失敗はトーストで知らせる。
	 */
	import { t } from '$lib/i18n';
	import { installCartridgeStreaming, cancelCartridgeInstall } from '$lib/free/api';
	import { addToast } from '$lib/free/stores/toast';
	import CartridgeProgressView, {
		type PhaseSpec,
		type ProgressState
	} from './CartridgeProgressView.svelte';
	import DialogShell from './DialogShell.svelte';
	import Button from './ui/Button.svelte';

	let { onInstalled }: { onInstalled: () => void | Promise<void> } = $props();

	const INSTALL_PHASES: PhaseSpec[] = [
		{ id: 'extract', labelKey: 'cartridge.progress.phase_extract' },
		{ id: 'chunk_embed', labelKey: 'cartridge.progress.phase_chunk_embed' },
		{ id: 'index', labelKey: 'cartridge.progress.phase_index' }
	];

	let fileInput: HTMLInputElement | undefined = $state();
	let progressDialogOpen = $state(false);
	let progressState = $state<ProgressState>(emptyProgress());
	let installSessionId = $state('');
	let installAbort: AbortController | null = null;
	let cancelling = $state(false);

	function emptyProgress(): ProgressState {
		return { currentPhase: null, completedPhases: new Set(), current: 0, total: 0, detail: '' };
	}

	function reportFailure(detail: string) {
		addToast({
			type: 'error',
			i18nKey: 'cartridge.install_failed_detail',
			params: { detail }
		});
	}

	async function handleInstall(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		progressState = emptyProgress();
		cancelling = false;
		installSessionId =
			typeof crypto !== 'undefined' && 'randomUUID' in crypto
				? crypto.randomUUID()
				: `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
		installAbort = new AbortController();
		progressDialogOpen = true;

		try {
			let installSucceeded = false;
			let errorOccurred = false;
			for await (const ev of installCartridgeStreaming(
				file,
				installSessionId,
				installAbort.signal
			)) {
				if (ev.type === 'step' && ev.step) {
					const { phase, status, current, total, detail } = ev.step;
					if (status === 'running') {
						progressState = {
							currentPhase: phase,
							completedPhases: progressState.completedPhases,
							current: current ?? 0,
							total: total ?? 0,
							detail: detail ?? ''
						};
					} else if (status === 'done') {
						const completed = new Set(progressState.completedPhases);
						completed.add(phase);
						progressState = {
							currentPhase: null,
							completedPhases: completed,
							current: total ?? progressState.current,
							total: total ?? progressState.total,
							detail: ''
						};
					}
				} else if (ev.type === 'result') {
					installSucceeded = true;
				} else if (ev.type === 'error') {
					errorOccurred = true;
					reportFailure(ev.error?.message ?? $t('error.cartridge_install_failed'));
				}
			}
			if (installSucceeded && !errorOccurred) {
				await onInstalled();
			}
		} catch (e) {
			if (!(e instanceof DOMException && e.name === 'AbortError')) {
				reportFailure(e instanceof Error ? e.message : $t('error.cartridge_install_failed'));
			}
		} finally {
			progressDialogOpen = false;
			installAbort = null;
			installSessionId = '';
			cancelling = false;
			input.value = '';
		}
	}

	async function handleCancelInstall() {
		if (!installSessionId || cancelling) return;
		cancelling = true;
		await cancelCartridgeInstall(installSessionId);
		// fetch 自体も中止 (ストリーム side だけ閉じても残ってしまうケースに備えて)
		installAbort?.abort();
	}
</script>

<Button variant="primary" onclick={() => fileInput?.click()} disabled={progressDialogOpen}>
	{$t('cartridge.install')}
</Button>
<input
	bind:this={fileInput}
	type="file"
	accept=".evocart,.zip"
	class="hidden"
	aria-label={$t('cartridge.install')}
	onchange={handleInstall}
/>

{#if progressDialogOpen}
	<DialogShell
		ariaLabel={$t('cartridge.progress.install_title')}
		onClose={() => {
			/* 進捗中はオーバーレイ / Escape からは閉じない */
		}}
		minWidth="420px"
		maxWidth="520px"
		canCloseOnOverlayClick={false}
	>
		<h3 class="progress-title">{$t('cartridge.progress.install_title')}</h3>
		<CartridgeProgressView
			phases={INSTALL_PHASES}
			state={progressState}
			{cancelling}
			onCancel={handleCancelInstall}
		/>
	</DialogShell>
{/if}

<style>
	.hidden {
		display: none;
	}
	.progress-title {
		margin: 0 0 14px;
		font-size: var(--font-size-lg);
		color: var(--text-primary);
	}
</style>
