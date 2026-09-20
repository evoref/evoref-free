<script lang="ts">
	import { t } from '$lib/i18n';
	import type { CartridgeDetail } from '$lib/free/api';
	import { rebuildCartridge } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatSize } from '$lib/free/utils/format';
	import { addToast } from '$lib/free/stores/toast';
	import DialogShell from './DialogShell.svelte';

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
	maxWidth="520px"
>
	<div class="dialog-header">
			<h3 class="dialog-title">{detail.name}</h3>
			<span class="status-badge" class:loaded={detail.status === 'loaded'}>
				{detail.status === 'loaded' ? $t('cartridge.status_loaded') : $t('cartridge.status_installed')}
			</span>
		</div>

		{#if detail.description}
			<p class="description">{detail.description}</p>
		{/if}

		<div class="detail-grid">
			<div class="info-row">
				<span class="label">{$t('cartridge.version')}</span>
				<span class="value">v{detail.version}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.active_version')}</span>
				<span class="value">v{detail.active_version || detail.version}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.author')}</span>
				<span class="value">{detail.author || '—'}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.language')}</span>
				<span class="value">{detail.language}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.doc_count')}</span>
				<span class="value">{detail.doc_count}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.chunks')}</span>
				<span class="value">{detail.chunks}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.size_mb')}</span>
				<span class="value">{formatSize(detail.size_mb)}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.license')}</span>
				<span class="value">{detail.license || '—'}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.embedding_model')}</span>
				<span class="value">{detail.embedding_model_id || '—'}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.content_digest')}</span>
				<span class="value digest">{detail.content_digest.slice(0, 16) || '—'}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.compatibility')}</span>
				<span class="value">{detail.compatibility}</span>
			</div>
			<div class="info-row">
				<span class="label">{$t('cartridge.installed_at')}</span>
				<span class="value">{detail.installed_at || '—'}</span>
			</div>
			{#if detail.tags.length > 0}
				<div class="info-row tags-row">
					<span class="label">{$t('cartridge.tags')}</span>
					<span class="value tags">
						{#each detail.tags as tag}
							<span class="tag">{tag}</span>
						{/each}
					</span>
				</div>
			{/if}
			{#if detail.provides.length > 0}
				<div class="info-row tags-row">
					<span class="label">{$t('cartridge.provides')}</span>
					<span class="value tags">
						{#each detail.provides as section}
							<span class="tag">{$t(`cartridge.provides_${section}`)}</span>
						{/each}
					</span>
				</div>
			{/if}
		</div>

		{#if detail.language_pack.length > 0}
			<div class="language-pack">
				<h4 class="language-pack-title">{$t('cartridge.language_pack_title')}</h4>
				<ul class="language-pack-list">
					{#each detail.language_pack as entry}
						<li class="language-pack-item">
							<span class="lang-id" class:disabled={!entry.enabled}>{entry.id}</span>
							<span class="lang-status" class:enabled={entry.enabled}>
								{entry.enabled
									? $t('cartridge.language_pack_enabled')
									: $t('cartridge.language_pack_disabled')}
							</span>
							{#if !entry.enabled && entry.reason}
								<span class="lang-reason">{entry.reason}</span>
							{/if}
						</li>
					{/each}
				</ul>
			</div>
		{/if}

		<!-- 操作ボタン (再構築は docs セクションの索引の操作。docs が無ければ出さない) -->
		{#if !detail.provides || detail.provides.includes('docs')}
			<div class="action-section">
				<button
					class="btn btn-action"
					onclick={handleRebuild}
					disabled={rebuilding}
				>
					{rebuilding ? $t('cartridge.rebuilding') : $t('cartridge.rebuild')}
				</button>
			</div>
		{/if}

	<div class="dialog-actions">
		<button class="btn btn-close" onclick={onClose}>
			{$t('cartridge.close')}
		</button>
	</div>
</DialogShell>

<style>
	.dialog-header {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 12px;
	}
	.dialog-title {
		margin: 0;
		font-size: 1.15rem;
		color: var(--text-primary);
		flex: 1;
	}
	.status-badge {
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 0.8rem;
		background-color: var(--bg-secondary);
		color: var(--text-secondary);
		border: 1px solid var(--border);
	}
	.status-badge.loaded {
		background-color: var(--control-bg);
		color: var(--text-primary);
		border-color: var(--border);
	}
	.description {
		color: var(--text-secondary);
		font-size: 0.95rem;
		line-height: 1.5;
		margin: 0 0 16px;
	}
	.detail-grid {
		background-color: var(--bg-secondary);
		border-radius: var(--border-radius);
		padding: 12px;
		margin-bottom: 16px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.info-row {
		display: flex;
		justify-content: space-between;
		font-size: 0.95rem;
	}
	.label {
		color: var(--text-secondary);
	}
	.value {
		color: var(--text-primary);
		font-weight: 500;
	}
	.tags-row {
		flex-wrap: wrap;
	}
	.tags {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
		justify-content: flex-end;
	}
	.tag {
		padding: 1px 6px;
		background-color: var(--bg-primary);
		border: 1px solid var(--border);
		border-radius: 6px;
		font-size: 0.8rem;
		font-weight: 400;
	}
	.language-pack {
		margin-bottom: 16px;
	}
	.language-pack-title {
		margin: 0 0 8px;
		font-size: 0.95rem;
		color: var(--text-primary);
	}
	.language-pack-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.language-pack-item {
		display: flex;
		align-items: baseline;
		gap: 8px;
		font-size: 0.9rem;
		flex-wrap: wrap;
	}
	.lang-id {
		font-weight: 500;
		color: var(--text-primary);
	}
	.lang-id.disabled {
		opacity: 0.6;
	}
	.lang-status {
		color: var(--text-secondary);
	}
	.lang-status.enabled {
		color: var(--text-primary);
	}
	.lang-reason {
		color: var(--text-secondary);
		font-size: 0.8rem;
		opacity: 0.8;
	}
	.action-section {
		display: flex;
		gap: 8px;
		margin-bottom: 16px;
	}
	.btn {
		padding: 8px 16px;
		border-radius: var(--border-radius);
		font-size: 0.95rem;
		cursor: pointer;
		border: none;
	}
	.btn-action {
		flex: 1;
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}
	.btn-action:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn-action:hover:not(:disabled) {
		opacity: 0.85;
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
	}
	.btn-close {
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}
	.btn-close:hover {
		opacity: 0.9;
	}
</style>
