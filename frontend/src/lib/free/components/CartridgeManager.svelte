<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import type { Snippet } from 'svelte';
	import type { Cartridge, CartridgeDetail } from '$lib/free/api';
	import {
		getCartridges,
		installCartridgeStreaming,
		cancelCartridgeInstall,
		loadCartridge,
		unloadCartridge,
		deleteCartridge,
		getCartridgeDetail
	} from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatSize } from '$lib/free/utils/format';
	import CartridgeDetailDialog from './CartridgeDetailDialog.svelte';
	import CartridgeProgressView, {
		type PhaseSpec,
		type ProgressState
	} from './CartridgeProgressView.svelte';
	import DialogShell from './DialogShell.svelte';

	// エディション固有 UI (Pro 等) は親 route から snippet として注入する。
	// Free 配下のコンポーネントが $lib/pro を直接参照すると境界違反になるため、
	// ここでは Pro を一切知らない。
	type HeaderExtraArgs = { refresh: () => Promise<void> };
	type Props = {
		headerExtra?: Snippet<[HeaderExtraArgs]>;
	};
	let { headerExtra }: Props = $props();

	let cartridges = $state<Cartridge[]>([]);
	let fileInput: HTMLInputElement | undefined = $state();
	let errorMessage = $state('');
	let selectedDetail = $state<CartridgeDetail | null>(null);

	// インストール進捗状態
	const INSTALL_PHASES: PhaseSpec[] = [
		{ id: 'extract', labelKey: 'cartridge.progress.phase_extract' },
		{ id: 'chunk_embed', labelKey: 'cartridge.progress.phase_chunk_embed' },
		{ id: 'index', labelKey: 'cartridge.progress.phase_index' }
	];
	let progressDialogOpen = $state(false);
	let progressState = $state<ProgressState>({
		currentPhase: null,
		completedPhases: new Set(),
		current: 0,
		total: 0,
		detail: ''
	});
	let installSessionId = $state('');
	let installAbort: AbortController | null = null;
	let cancelling = $state(false);

	onMount(() => {
		refreshCartridges();
	});

	async function refreshCartridges() {
		const result = await handleApiCall(() => getCartridges(), {
			silent: true,
			fallback: []
		});
		cartridges = result ?? [];
	}

	function _resetProgress() {
		progressState = {
			currentPhase: null,
			completedPhases: new Set(),
			current: 0,
			total: 0,
			detail: ''
		};
		cancelling = false;
	}

	async function handleInstall(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) {
			return;
		}

		_resetProgress();
		errorMessage = '';
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
					errorMessage = ev.error?.message ?? $t('error.cartridge_install_failed');
				}
			}
			if (installSucceeded && !errorOccurred) {
				await refreshCartridges();
			}
		} catch (e) {
			if (!(e instanceof DOMException && e.name === 'AbortError')) {
				errorMessage =
					e instanceof Error ? e.message : $t('error.cartridge_install_failed');
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

	async function handleLoad(id: string) {
		errorMessage = '';
		await handleApiCall(() => loadCartridge(id), {
			fallbackKey: 'cartridge.load_failed'
		});
		await refreshCartridges();
	}

	async function handleUnload(id: string) {
		errorMessage = '';
		await handleApiCall(() => unloadCartridge(id), {
			fallbackKey: 'cartridge.unload_failed'
		});
		await refreshCartridges();
	}

	async function handleDelete(id: string) {
		errorMessage = '';
		await handleApiCall(() => deleteCartridge(id), {
			fallbackKey: 'cartridge.delete_failed'
		});
		await refreshCartridges();
	}

	async function handleShowDetail(id: string) {
		const detail = await handleApiCall(() => getCartridgeDetail(id), {
			fallbackKey: 'cartridge.detail_failed'
		});
		if (detail) {
			selectedDetail = detail;
		}
	}

	function handleDetailClose() {
		selectedDetail = null;
	}

	async function handleDetailUpdated() {
		await refreshCartridges();
		// 詳細情報も再取得
		if (selectedDetail) {
			const updated = await handleApiCall(() => getCartridgeDetail(selectedDetail!.id), {
				silent: true
			});
			if (updated) {
				selectedDetail = updated;
			}
		}
	}

</script>

<div class="cartridge-manager">
	<div class="header">
		{#if headerExtra}
			{@render headerExtra({ refresh: refreshCartridges })}
		{/if}
		<button class="install-btn" onclick={() => fileInput?.click()} disabled={progressDialogOpen}>
			{$t('cartridge.install')}
		</button>
		<input
			bind:this={fileInput}
			type="file"
			accept=".zip"
			class="hidden"
			onchange={handleInstall}
		/>
	</div>

	{#if errorMessage}
		<p class="error">{errorMessage}</p>
	{/if}

	{#if cartridges.length === 0}
		<p class="empty">{$t('cartridge.list_empty')}</p>
	{:else}
		<ul class="cartridge-list">
			{#each cartridges as cart}
				<li class="cartridge-item" class:loaded={cart.status === 'loaded'}>
					<button class="info" onclick={() => handleShowDetail(cart.id)}>
						<span class="name">{cart.name}</span>
						<span class="desc">{cart.description}</span>
						<span class="meta">
							v{cart.version} / {cart.chunks} chunks / {formatSize(cart.size_mb)}
						</span>
					</button>
					<div class="actions">
						{#if cart.status === 'loaded'}
							<button class="action-btn unload" onclick={() => handleUnload(cart.id)}>
								{$t('cartridge.unload')}
							</button>
						{:else}
							<button class="action-btn primary" onclick={() => handleLoad(cart.id)}>
								{$t('cartridge.load')}
							</button>
						{/if}
						<button class="action-btn danger" onclick={() => handleDelete(cart.id)}>
							{$t('cartridge.uninstall')}
						</button>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</div>

{#if selectedDetail}
	<CartridgeDetailDialog
		detail={selectedDetail}
		onClose={handleDetailClose}
		onUpdated={handleDetailUpdated}
	/>
{/if}

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
	.cartridge-manager {
		padding: 16px;
	}
	.header {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: 12px;
		margin-bottom: 16px;
	}
	.install-btn {
		padding: 6px 14px;
		background-color: var(--accent);
		color: var(--text-on-accent);
		border: none;
		border-radius: var(--border-radius);
		cursor: pointer;
		font-size: 0.95rem;
	}
	.install-btn:disabled {
		opacity: 0.5;
	}
	.progress-title {
		margin: 0 0 14px;
		font-size: 1.1rem;
		color: var(--text-primary);
	}
	.hidden {
		display: none;
	}
	.error {
		color: var(--color-error);
		font-size: 0.95rem;
		margin-bottom: 8px;
	}
	.empty {
		color: var(--text-secondary);
		font-style: italic;
	}
	.cartridge-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: grid;
		grid-template-columns: 1fr;
		gap: 8px;
	}
	.cartridge-item {
		display: flex;
		align-items: flex-start;
		padding: 10px 12px;
		background-color: var(--bg-secondary);
		border: 0.5px solid var(--card-border);
		border-radius: 10px;
	}
	.cartridge-item.loaded {
		background-color: var(--bg-primary);
		border-width: 2px;
	}
	@media (max-width: 900px) {
		.cartridge-list {
			grid-template-columns: 1fr;
		}
		.cartridge-item {
			flex-direction: column;
			align-items: flex-start;
			gap: 8px;
		}
		.actions {
			flex-direction: column;
			width: 100%;
		}
		.action-btn {
			width: 100%;
			text-align: center;
		}
	}
	.info {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 2px;
		background: none;
		border: none;
		padding: 0;
		text-align: left;
		cursor: pointer;
		font-family: inherit;
	}
	.info:hover .name {
		text-decoration: underline;
	}
	.name {
		font-weight: 600;
		color: var(--text-primary);
	}
	.cartridge-item:not(.loaded) .name {
		color: var(--text-secondary);
		opacity: 0.6;
	}
	.desc {
		font-size: 0.9rem;
		color: var(--text-secondary);
	}
	.meta {
		font-size: 0.8rem;
		color: var(--text-secondary);
		opacity: 0.7;
		margin-top: 2px;
	}
	.cartridge-item:not(.loaded) .desc {
		opacity: 0.6;
	}
	.actions {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.action-btn {
		padding: 4px 10px;
		font-size: 0.9rem;
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		color: var(--text-primary);
		cursor: pointer;
	}
	.action-btn.primary {
		background: none;
		color: var(--text-primary);
		border: 1.5px solid var(--border);
	}
	.action-btn.unload {
		background-color: var(--control-bg);
		color: var(--text-primary);
		border: 1.5px solid var(--border);
	}
	.action-btn.danger {
		background-color: var(--color-error);
		color: var(--text-on-accent);
		border: 1px solid var(--color-error);
	}
	.action-btn:hover {
		opacity: 0.85;
	}
</style>
