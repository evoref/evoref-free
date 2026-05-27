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
	import { getThemes, trustThemeApi, installThemeApi, uninstallThemeApi } from '$lib/free/api';
	import { get } from 'svelte/store';
	import { onMount } from 'svelte';
	import ThemeTrustDialog from './ThemeTrustDialog.svelte';
	import { handleApiCall } from '$lib/free/utils/error';

	let fileInput: HTMLInputElement | undefined = $state();
	let installing = $state(false);

	// 信頼確認ダイアログの状態
	let trustDialogTheme: ThemeInfo | null = $state(null);

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

	async function handleInstall(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		installing = true;
		const result = await handleApiCall(() => installThemeApi(file), {
			fallbackKey: 'theme_manager.install_failed'
		});
		if (result) {
			await refreshThemes();
		}
		installing = false;
		input.value = '';
	}

	async function handleActivate(theme: ThemeInfo) {
		// 外部テーマでコンポーネントを含み、未信頼 → 信頼確認ダイアログを表示
		if (theme.has_components && !theme.trusted && !theme.builtin) {
			trustDialogTheme = theme;
			return;
		}
		await doActivate(theme.theme_id);
	}

	async function handleTrustConfirm() {
		if (!trustDialogTheme) return;
		const themeToTrust = trustDialogTheme;
		trustDialogTheme = null;

		const result = await handleApiCall(() => trustThemeApi(themeToTrust.theme_id), {
			fallbackKey: 'theme_manager.activate_failed'
		});
		if (!result) return;

		await doActivate(themeToTrust.theme_id);
	}

	function handleTrustCancel() {
		const themeToSkip = trustDialogTheme;
		trustDialogTheme = null;

		// 信頼なしでアクティベート（colors + layout のみ、スロット無効）
		if (themeToSkip) {
			doActivate(themeToSkip.theme_id);
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

	async function handleDelete(id: string) {
		const wasActive = get(themeId) === id;
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
	}
</script>

{#if trustDialogTheme}
	<ThemeTrustDialog
		themeName={trustDialogTheme.name}
		themeAuthor={trustDialogTheme.author}
		themeVersion={trustDialogTheme.version}
		componentCount={trustDialogTheme.component_count}
		onConfirm={handleTrustConfirm}
		onCancel={handleTrustCancel}
	/>
{/if}

<div class="theme-manager">
	<div class="header">
		<button class="install-btn" onclick={() => fileInput?.click()} disabled={installing}>
			{installing ? $t('theme_manager.installing') : $t('theme_manager.install')}
		</button>
		<input
			bind:this={fileInput}
			type="file"
			accept=".zip"
			class="hidden"
			onchange={handleInstall}
		/>
	</div>

	{#if $availableThemes.length === 0}
		<p class="empty">{$t('theme_manager.no_themes')}</p>
	{:else}
		<ul class="theme-list">
			{#each $availableThemes as theme}
				<li class="theme-item" class:active={theme.active}>
					{#if theme.has_preview}
						<img
							class="preview-img"
							src="/api/themes/{theme.theme_id}/preview"
							alt={$t('theme_manager.preview_alt', { name: theme.name })}
							loading="lazy"
						/>
					{:else}
						<div class="preview-placeholder">
							<span class="preview-placeholder-text">{$t('theme_manager.no_preview')}</span>
						</div>
					{/if}
					<div class="info">
						<span class="name">
							{theme.name}
							{#if theme.active}
								<span class="badge active-badge">{$t('theme_manager.active')}</span>
							{/if}
							{#if theme.has_components && theme.trusted}
								<span class="badge trusted">{$t('theme_manager.trusted')}</span>
							{/if}
						</span>
						<span class="meta">
							{theme.author}
							{#if theme.description}
								— {theme.description}
							{/if}
							<span class="version">v{theme.version}</span>
						</span>
					</div>
					<div class="actions">
						{#if !theme.active}
							<button
								class="action-btn primary"
								onclick={() => handleActivate(theme)}
							>
								{$t('theme_manager.activate')}
							</button>
						{/if}
						<button
							class="action-btn danger"
							onclick={() => handleDelete(theme.theme_id)}
						>
							{$t('theme_manager.uninstall')}
						</button>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.theme-manager {
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
	.hidden {
		display: none;
	}
	.empty {
		color: var(--text-secondary);
	}
	.theme-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: grid;
		grid-template-columns: 1fr;
		gap: 8px;
	}
	.theme-item {
		display: flex;
		align-items: flex-start;
		gap: 12px;
		padding: 10px 12px;
		background-color: var(--bg-secondary);
		border: 0.5px solid var(--card-border);
		border-radius: 10px;
	}
	.theme-item.active {
		background-color: var(--bg-primary);
		border-width: 2px;
	}
	.preview-img {
		width: 160px;
		height: 100px;
		object-fit: cover;
		border-radius: 6px;
		border: 1px solid var(--border);
		flex-shrink: 0;
	}
	.preview-placeholder {
		width: 160px;
		height: 100px;
		border-radius: 6px;
		border: 1px solid var(--border);
		background-color: var(--bg-primary);
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
	}
	.preview-placeholder-text {
		font-size: 0.75rem;
		color: var(--text-secondary);
		text-align: center;
		line-height: 1.2;
	}
	.info {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.name {
		font-weight: 600;
		color: var(--text-primary);
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.theme-item:not(.active) .name {
		color: var(--text-secondary);
		opacity: 0.6;
	}
	.theme-item:not(.active) .meta {
		opacity: 0.6;
	}
	.badge {
		font-size: 0.8rem;
		padding: 1px 6px;
		background-color: var(--accent);
		color: var(--text-on-accent);
		border-radius: 4px;
		font-weight: 400;
	}
	.badge.trusted {
		background-color: var(--color-success);
	}
	.badge.active-badge {
		background-color: var(--color-success);
		color: var(--text-on-accent);
	}
	.meta {
		font-size: 0.9rem;
		color: var(--text-secondary);
	}
	.version {
		margin-left: 4px;
		opacity: 0.7;
	}
	.actions {
		display: flex;
		flex-direction: column;
		gap: 6px;
		align-self: flex-start;
		flex-shrink: 0;
		width: 140px;
	}
	.action-btn {
		padding: 4px 10px;
		width: 100%;
		text-align: center;
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
	.action-btn.danger {
		background-color: var(--color-error);
		color: var(--text-on-accent);
		border: 1px solid var(--color-error);
	}
	.action-btn:hover {
		opacity: 0.85;
	}
	@media (max-width: 900px) {
		.theme-list {
			grid-template-columns: 1fr;
		}
		.theme-item {
			flex-direction: column;
			align-items: flex-start;
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
</style>
