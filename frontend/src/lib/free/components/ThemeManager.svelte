<script lang="ts">
	import { t } from '$lib/i18n';
	import {
		colorMode,
		themeId,
		availableThemes,
		activateTheme,
		clearThemeColors,
		type ThemeInfo
	} from '$lib/free/stores/theme';
	import { getThemes, uninstallThemeApi } from '$lib/free/api';
	import { get } from 'svelte/store';
	import { onMount } from 'svelte';
	import { handleApiCall } from '$lib/free/utils/error';
	import Badge from './ui/Badge.svelte';
	import Button from './ui/Button.svelte';
	import ConfirmDialog from './ui/ConfirmDialog.svelte';
	import EmptyState from './ui/EmptyState.svelte';

	let pendingDelete: ThemeInfo | null = $state(null);
	let deleting = $state(false);

	let activeTheme = $derived($availableThemes.find((th) => th.active) ?? null);

	onMount(() => {
		refreshThemes();
	});

	async function refreshThemes() {
		const data = await handleApiCall(() => getThemes(), {
			silent: true,
			fallbackKey: 'error.themes_failed'
		});
		if (data) {
			availableThemes.set(data.themes ?? []);
		}
	}

	async function doActivate(id: string) {
		await handleApiCall(
			async () => {
				await activateTheme(id, get(colorMode));
				await refreshThemes();
			},
			{ fallbackKey: 'theme_manager.activate_failed' }
		);
	}

	async function confirmDelete() {
		if (!pendingDelete) return;
		const id = pendingDelete.theme_id;
		const wasActive = get(themeId) === id;
		deleting = true;
		try {
			await handleApiCall(
				async () => {
					await uninstallThemeApi(id);
					const data = await getThemes();
					availableThemes.set(data.themes ?? []);

					if (wasActive) {
						if (data.active_theme_id) {
							await activateTheme(data.active_theme_id, get(colorMode));
						} else {
							// テーマなし状態: CSS除去、app.css のデフォルトで表示
							clearThemeColors();
							themeId.set('');
						}
					}
				},
				{ fallbackKey: 'theme_manager.uninstall_failed' }
			);
		} finally {
			deleting = false;
			pendingDelete = null;
		}
	}
</script>

{#if pendingDelete}
	<ConfirmDialog
		title={$t('theme_manager.uninstall_confirm_title')}
		message={$t('theme_manager.uninstall_confirm_message', { name: pendingDelete.name })}
		confirmLabel={$t('theme_manager.uninstall')}
		busy={deleting}
		onConfirm={confirmDelete}
		onCancel={() => (pendingDelete = null)}
	/>
{/if}

<div class="theme-manager">
	<p class="current" data-testid="current-theme">
		<span class="current-label">{$t('theme_manager.current')}</span>
		<strong>{activeTheme ? activeTheme.name : $t('theme_manager.none')}</strong>
	</p>

	<h2 class="list-title">
		{$t('theme_manager.installed_heading', { count: $availableThemes.length })}
	</h2>
	{#if $availableThemes.length === 0}
		<EmptyState message={$t('theme_manager.no_themes')} />
	{:else}
		<ul class="theme-list">
			{#each $availableThemes as theme (theme.theme_id)}
				<li class="theme-item">
					{#if theme.has_preview}
						<img
							class="preview"
							src="/api/themes/{theme.theme_id}/preview"
							alt={$t('theme_manager.preview_alt', { name: theme.name })}
							loading="lazy"
						/>
					{:else}
						<div class="preview preview-placeholder">{$t('theme_manager.no_preview')}</div>
					{/if}
					<div class="info">
						<span class="title-line">
							<span class="name">{theme.name}</span>
							{#if theme.active}
								<Badge tone="strong">{$t('theme_manager.active')}</Badge>
							{/if}
						</span>
						<span class="meta">
							{theme.author ? `${theme.author} · ` : ''}v{theme.version}
						</span>
						{#if theme.description}
							<span class="desc">{theme.description}</span>
						{/if}
					</div>
					<div class="row-actions">
						{#if !theme.active}
							<Button size="sm" onclick={() => doActivate(theme.theme_id)}>
								{$t('theme_manager.activate')}
							</Button>
						{/if}
						<Button variant="ghost" size="sm" onclick={() => (pendingDelete = theme)}>
							{$t('theme_manager.uninstall')}
						</Button>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.current {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		margin: 0 0 var(--space-5);
		padding: var(--space-3) var(--space-4);
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		color: var(--text-primary);
	}
	.current-label {
		color: var(--text-secondary);
		font-size: var(--font-size-sm);
	}
	.list-title {
		margin: 0 0 var(--space-3);
		font-size: var(--font-size-base);
		font-weight: 600;
		color: var(--text-primary);
	}
	.theme-list {
		list-style: none;
		padding: 0;
		margin: 0;
		border-top: 1px solid var(--border);
	}
	.theme-item {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-3) var(--space-1);
		border-bottom: 1px solid var(--border);
	}
	.preview {
		width: 96px;
		height: 60px;
		object-fit: cover;
		border-radius: var(--radius-sm);
		border: 1px solid var(--border);
		flex-shrink: 0;
	}
	.preview-placeholder {
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: var(--font-size-xs);
		color: var(--text-muted);
		text-align: center;
	}
	.info {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
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
	.meta {
		font-size: var(--font-size-xs);
		color: var(--text-muted);
	}
	.desc {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
	}
	.row-actions {
		display: flex;
		align-items: center;
		gap: var(--space-1);
		flex-shrink: 0;
	}
	@media (max-width: 600px) {
		.theme-item {
			flex-wrap: wrap;
		}
	}
</style>
