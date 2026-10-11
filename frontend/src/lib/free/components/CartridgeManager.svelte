<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import type { Snippet } from 'svelte';
	import type { Cartridge, CartridgeDetail } from '$lib/free/api';
	import {
		getCartridges,
		loadCartridge,
		unloadCartridge,
		deleteCartridge,
		rebuildCartridge,
		getCartridgeDetail
	} from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatSize } from '$lib/free/utils/format';
	import { templatesRevision } from '$lib/free/stores/chat';
	import CartridgeDetailDialog from './CartridgeDetailDialog.svelte';
	import RegisterTemplateDialog from './RegisterTemplateDialog.svelte';
	import Badge from './ui/Badge.svelte';
	import Button from './ui/Button.svelte';
	import ConfirmDialog from './ui/ConfirmDialog.svelte';
	import EmptyState from './ui/EmptyState.svelte';
	import MoreMenu from './ui/MoreMenu.svelte';

	// エディション固有 UI (Pro 等) は親 route から snippet として注入する。
	// Free 配下のコンポーネントが $lib/pro を直接参照すると境界違反になるため、
	// ここでは Pro を一切知らない。主操作のインストールは route が PageLayout の actions に置く。
	type HeaderExtraArgs = { refresh: () => Promise<void> };
	type Props = {
		headerExtra?: Snippet<[HeaderExtraArgs]>;
	};
	let { headerExtra }: Props = $props();

	let cartridges = $state<Cartridge[]>([]);
	let selectedDetail = $state<CartridgeDetail | null>(null);
	let showRegisterTemplate = $state(false);
	let pendingDelete = $state<Cartridge | null>(null);
	let deleting = $state(false);
	// 埋め込みモデル切替後にカートリッジ単位で再構築 (再埋め込み)。
	let rebuildingId = $state<string | null>(null);

	onMount(() => {
		refresh();
	});

	/** 一覧を取り直す (インストール完了時に route からも呼ぶ) */
	export async function refresh() {
		const result = await handleApiCall(() => getCartridges(), {
			silent: true,
			fallback: []
		});
		cartridges = result ?? [];
		// 一覧を取り直すのは install / 削除 / 様式の登録の後 — どれも選べる様式の
		// 集合を変えうるので、チャット入力の選択肢にも取り直しを知らせる。
		templatesRevision.update((n) => n + 1);
	}

	/** 索引 (読込み / 取外し / 再構築) は docs セクションの操作。provides が無い旧応答は docs 扱い */
	function hasDocs(cart: Cartridge): boolean {
		return !cart.provides || cart.provides.includes('docs');
	}

	function providesLabel(cart: Cartridge): string {
		return (cart.provides ?? []).map((p) => $t(`cartridge.provides_${p}`)).join(' / ');
	}

	async function handleLoad(id: string) {
		await handleApiCall(() => loadCartridge(id), {
			fallbackKey: 'cartridge.load_failed'
		});
		await refresh();
	}

	async function handleUnload(id: string) {
		await handleApiCall(() => unloadCartridge(id), {
			fallbackKey: 'cartridge.unload_failed'
		});
		await refresh();
	}

	async function confirmDelete() {
		if (!pendingDelete) return;
		const id = pendingDelete.id;
		deleting = true;
		try {
			await handleApiCall(() => deleteCartridge(id), {
				fallbackKey: 'cartridge.delete_failed'
			});
			await refresh();
		} finally {
			deleting = false;
			pendingDelete = null;
		}
	}

	async function handleRebuild(id: string) {
		if (rebuildingId) return;
		rebuildingId = id;
		try {
			await handleApiCall(() => rebuildCartridge(id), {
				fallbackKey: 'cartridge.rebuild_failed'
			});
			await refresh();
		} finally {
			rebuildingId = null;
		}
	}

	async function handleShowDetail(id: string) {
		const detail = await handleApiCall(() => getCartridgeDetail(id), {
			fallbackKey: 'cartridge.detail_failed'
		});
		if (detail) {
			selectedDetail = detail;
		}
	}

	async function handleDetailUpdated() {
		await refresh();
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

<section class="cartridge-manager">
	<div class="list-head">
		<h2 class="list-title">{$t('cartridge.installed_heading', { count: cartridges.length })}</h2>
		<div class="list-tools">
			{#if headerExtra}
				{@render headerExtra({ refresh })}
			{/if}
			<Button size="sm" onclick={() => (showRegisterTemplate = true)}>
				{$t('cartridge.template_register')}
			</Button>
		</div>
	</div>

	{#if cartridges.length === 0}
		<EmptyState message={$t('cartridge.list_empty')}>
			<span class="empty-hint">{$t('cartridge.list_empty_hint')}</span>
		</EmptyState>
	{:else}
		<ul class="cartridge-list">
			{#each cartridges as cart (cart.id)}
				{@const docs = hasDocs(cart)}
				{@const loaded = docs && cart.status === 'loaded'}
				<li class="cartridge-item">
					<button class="info" onclick={() => handleShowDetail(cart.id)}>
						<span class="title-line">
							<span class="name">{cart.name}</span>
							{#if docs}
								<Badge tone={loaded ? 'strong' : 'muted'}>
									{loaded ? $t('cartridge.status_loaded') : $t('cartridge.status_not_loaded')}
								</Badge>
							{/if}
						</span>
						{#if cart.description}
							<span class="desc">{cart.description}</span>
						{/if}
						<span class="meta">
							v{cart.version} · {docs ? formatSize(cart.size_mb) : providesLabel(cart)}
						</span>
					</button>
					<div class="row-actions">
						{#if docs}
							{#if loaded}
								<Button size="sm" onclick={() => handleUnload(cart.id)}>
									{$t('cartridge.unload')}
								</Button>
							{:else}
								<Button size="sm" onclick={() => handleLoad(cart.id)}>
									{$t('cartridge.load')}
								</Button>
							{/if}
						{/if}
						<MoreMenu>
							<Button variant="ghost" size="sm" onclick={() => handleShowDetail(cart.id)}>
								{$t('cartridge.show_detail')}
							</Button>
							{#if docs}
								<Button
									variant="ghost"
									size="sm"
									disabled={rebuildingId !== null}
									onclick={() => handleRebuild(cart.id)}
								>
									{rebuildingId === cart.id ? $t('cartridge.rebuilding') : $t('cartridge.rebuild')}
								</Button>
							{/if}
							<Button variant="danger" size="sm" onclick={() => (pendingDelete = cart)}>
								{$t('cartridge.uninstall')}
							</Button>
						</MoreMenu>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</section>

{#if selectedDetail}
	<CartridgeDetailDialog
		detail={selectedDetail}
		onClose={() => (selectedDetail = null)}
		onUpdated={handleDetailUpdated}
	/>
{/if}

{#if showRegisterTemplate}
	<RegisterTemplateDialog
		onClose={() => (showRegisterTemplate = false)}
		onRegistered={refresh}
	/>
{/if}

{#if pendingDelete}
	<ConfirmDialog
		title={$t('cartridge.uninstall_confirm_title')}
		message={$t('cartridge.uninstall_confirm_message', { name: pendingDelete.name })}
		confirmLabel={$t('cartridge.uninstall')}
		busy={deleting}
		onConfirm={confirmDelete}
		onCancel={() => (pendingDelete = null)}
	/>
{/if}

<style>
	.list-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: var(--space-2);
		margin-bottom: var(--space-3);
	}
	.list-title {
		margin: 0;
		font-size: var(--font-size-base);
		font-weight: 600;
		color: var(--text-primary);
	}
	.list-tools {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
	.empty-hint {
		color: var(--text-muted);
		font-size: var(--font-size-sm);
	}
	.cartridge-list {
		list-style: none;
		padding: 0;
		margin: 0;
		border-top: 1px solid var(--border);
	}
	.cartridge-item {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-3) var(--space-1);
		border-bottom: 1px solid var(--border);
	}
	.info {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
		background: none;
		border: none;
		padding: 0;
		text-align: left;
		cursor: pointer;
		font-family: inherit;
		color: inherit;
	}
	.info:hover .name {
		text-decoration: underline;
	}
	.title-line {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		flex-wrap: wrap;
	}
	.name {
		font-weight: 600;
		color: var(--text-primary);
	}
	.desc {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.meta {
		font-size: var(--font-size-xs);
		color: var(--text-muted);
	}
	.row-actions {
		display: flex;
		align-items: center;
		gap: var(--space-1);
		flex-shrink: 0;
	}
</style>
