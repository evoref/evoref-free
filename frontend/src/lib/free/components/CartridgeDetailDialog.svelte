<script lang="ts">
	import { t } from '$lib/i18n';
	import type { CartridgeDetail } from '$lib/free/api';
	import { rebuildCartridge } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatDate, formatSize } from '$lib/free/utils/format';
	import { addToast } from '$lib/free/stores/toast';
	import DialogShell from './DialogShell.svelte';
	import Badge from './ui/Badge.svelte';
	import Button from './ui/Button.svelte';
	import Disclosure from './ui/Disclosure.svelte';

	interface Props {
		detail: CartridgeDetail;
		onClose: () => void;
		onUpdated: () => void;
	}

	let { detail, onClose, onUpdated }: Props = $props();

	let rebuilding = $state(false);

	async function handleRebuild() {
		rebuilding = true;
		try {
			const result = await handleApiCall(() => rebuildCartridge(detail.id), {
				fallbackKey: 'cartridge.rebuild_failed'
			});
			if (result) {
				addToast({
					type: 'success',
					i18nKey: 'cartridge.rebuild_success',
					params: {
						chunks: String(result.chunks),
						time: String(result.rebuild_time_sec)
					}
				});
				onUpdated();
			}
		} finally {
			rebuilding = false;
		}
	}
</script>

<DialogShell
	ariaLabel={$t('cartridge.detail_title')}
	{onClose}
	minWidth="400px"
	maxWidth="560px"
>
	<div class="dialog-header">
		<h3 class="dialog-title">{detail.name}</h3>
		<Badge tone={detail.status === 'loaded' ? 'strong' : 'muted'}>
			{detail.status === 'loaded'
				? $t('cartridge.status_loaded')
				: $t('cartridge.status_not_loaded')}
		</Badge>
	</div>

	{#if detail.description}
		<p class="description">{detail.description}</p>
	{/if}

	<!-- 一般の利用者に要る項目だけ。索引の中身・互換性・ダイジェストは「詳細情報」へ -->
	<dl class="detail-grid">
		<dt>{$t('cartridge.version')}</dt>
		<dd>v{detail.version}</dd>
		<dt>{$t('cartridge.author')}</dt>
		<dd>{detail.author || '—'}</dd>
		<dt>{$t('cartridge.size_mb')}</dt>
		<dd>{formatSize(detail.size_mb)}</dd>
		{#if detail.provides.length > 0}
			<dt>{$t('cartridge.provides')}</dt>
			<dd class="tags">
				{#each detail.provides as section (section)}
					<Badge>{$t(`cartridge.provides_${section}`)}</Badge>
				{/each}
			</dd>
		{/if}
	</dl>

	{#if detail.language_pack.length > 0}
		<div class="language-pack">
			<h4 class="language-pack-title">{$t('cartridge.language_pack_title')}</h4>
			<ul class="language-pack-list">
				{#each detail.language_pack as entry (entry.id)}
					<li class="language-pack-item">
						<span class="lang-id">{entry.id}</span>
						<Badge tone={entry.enabled ? 'default' : 'muted'}>
							{entry.enabled
								? $t('cartridge.language_pack_enabled')
								: $t('cartridge.language_pack_disabled')}
						</Badge>
						{#if !entry.enabled && entry.reason}
							<span class="lang-reason">{entry.reason}</span>
						{/if}
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	<Disclosure label={$t('cartridge.more_info')}>
		<dl class="detail-grid technical">
			<dt>{$t('cartridge.active_version')}</dt>
			<dd>v{detail.active_version || detail.version}</dd>
			<dt>{$t('cartridge.language')}</dt>
			<dd>{detail.language}</dd>
			<dt>{$t('cartridge.doc_count')}</dt>
			<dd>{detail.doc_count}</dd>
			<dt>{$t('cartridge.chunks')}</dt>
			<dd>{detail.chunks}</dd>
			<dt>{$t('cartridge.license')}</dt>
			<dd>{detail.license || '—'}</dd>
			<dt>{$t('cartridge.embedding_model')}</dt>
			<dd>{detail.embedding_model_id || '—'}</dd>
			<dt>{$t('cartridge.content_digest')}</dt>
			<dd class="mono">{detail.content_digest.slice(0, 16) || '—'}</dd>
			<dt>{$t('cartridge.compatibility')}</dt>
			<dd>{detail.compatibility}</dd>
			<dt>{$t('cartridge.installed_at')}</dt>
			<dd>{detail.installed_at ? formatDate(detail.installed_at) : '—'}</dd>
			{#if detail.tags.length > 0}
				<dt>{$t('cartridge.tags')}</dt>
				<dd class="tags">
					{#each detail.tags as tag (tag)}
						<Badge>{tag}</Badge>
					{/each}
				</dd>
			{/if}
		</dl>
	</Disclosure>

	<div class="dialog-actions">
		<!-- 再構築は docs セクションの索引の操作。docs が無ければ出さない -->
		{#if !detail.provides || detail.provides.includes('docs')}
			<Button onclick={handleRebuild} disabled={rebuilding}>
				{rebuilding ? $t('cartridge.rebuilding') : $t('cartridge.rebuild')}
			</Button>
		{/if}
		<Button variant="primary" onclick={onClose}>{$t('cartridge.close')}</Button>
	</div>
</DialogShell>

<style>
	.dialog-header {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin-bottom: var(--space-3);
	}
	.dialog-title {
		margin: 0;
		font-size: var(--font-size-lg);
		color: var(--text-primary);
		flex: 1;
	}
	.description {
		color: var(--text-secondary);
		line-height: 1.5;
		margin: 0 0 var(--space-4);
	}
	.detail-grid {
		display: grid;
		grid-template-columns: max-content 1fr;
		gap: var(--space-2) var(--space-4);
		margin: 0 0 var(--space-4);
	}
	.detail-grid.technical {
		margin: var(--space-2) 0 0;
		font-size: var(--font-size-sm);
	}
	dt {
		color: var(--text-secondary);
	}
	dd {
		margin: 0;
		color: var(--text-primary);
		overflow-wrap: anywhere;
	}
	.tags {
		display: flex;
		gap: var(--space-1);
		flex-wrap: wrap;
	}
	.mono {
		font-family: monospace;
	}
	.language-pack {
		margin-bottom: var(--space-4);
	}
	.language-pack-title {
		margin: 0 0 var(--space-2);
		font-size: var(--font-size-base);
		color: var(--text-primary);
	}
	.language-pack-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}
	.language-pack-item {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		flex-wrap: wrap;
	}
	.lang-id {
		font-weight: 500;
		color: var(--text-primary);
	}
	.lang-reason {
		color: var(--text-muted);
		font-size: var(--font-size-xs);
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: var(--space-2);
		margin-top: var(--space-4);
	}
</style>
