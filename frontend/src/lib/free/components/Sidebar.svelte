<script lang="ts">
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { t } from '$lib/i18n';
	import { colorMode, toggleColorMode, sidebarCollapsed, layout, themeSlots } from '$lib/free/stores/theme';
	import { locale, availableLocales, switchLocale, promptLocaleSwitching } from '$lib/i18n';
	import { isPro, isDevelop, edition } from '$lib/edition';
	import { appVersion } from '$lib/free/stores/app';
	import { currentMode, clearMessages, switchMode, modeRestartStatus, isStreaming } from '$lib/free/stores/chat';
	import { addToast } from '$lib/free/stores/toast';
	import { serverState, backendEdition } from '$lib/free/stores/server';

	let { instanceName = 'evoref' }: { instanceName?: string } = $props();
	let position = $derived($layout.sidebar.position);
	let collapsed = $derived($sidebarCollapsed);
	let debugEnabled = $derived($serverState.debug?.enabled === true);
	// Pro ビルドでもバックエンドが Free なら Pro の導線を出さない (Pro の API が無い)
	let showPro = $derived(isPro && $backendEdition !== 'free');

	// 上から使う頻度の順。Pro 専用の画面はバックエンドも Pro のときだけ出す
	let navItems = $derived([
		{ href: '/history', labelKey: 'sidebar.history' },
		{ href: '/dashboard', labelKey: 'sidebar.dashboard' },
		{ href: '/learning', labelKey: 'sidebar.fewshot' },
		{ href: '/cartridge', labelKey: 'sidebar.cartridges' },
		...(showPro ? [{ href: '/knowledge', labelKey: 'sidebar.knowledge_sources' }] : []),
		{ href: '/themes', labelKey: 'sidebar.themes' },
		...(showPro ? [{ href: '/audit', labelKey: 'sidebar.audit_trail' }] : []),
		{ href: '/settings', labelKey: 'sidebar.settings' }
	]);

	function isActive(href: string): boolean {
		if (href === '/') return page.url.pathname === '/';
		return page.url.pathname.startsWith(href);
	}

	function handleNewChat(e: MouseEvent) {
		e.preventDefault();
		clearMessages();
		if (page.url.pathname !== '/') {
			goto('/');
		}
	}
</script>

{#if position !== 'hidden'}
	<aside
		class="sidebar"
		class:collapsed
		class:right={position === 'right'}
	>
		<div class="sidebar-header">
			<a href="/" class="instance-name">{instanceName}</a>
			<button
				class="collapse-btn"
				onclick={() => sidebarCollapsed.update((v) => !v)}
				aria-label={$t('sidebar.toggle')}
				aria-expanded={!collapsed}
			>
				{collapsed ? '›' : '‹'}
			</button>
		</div>

		{#if !collapsed}
			<nav class="sidebar-nav" aria-label={$t('sidebar.nav_label')}>
				<div class="nav-chat-row">
					<a href="/" class="nav-item nav-chat-link" class:active={isActive('/')} aria-current={isActive('/') ? 'page' : undefined}>{$t('sidebar.chat')}</a>
					<button class="new-chat-btn" onclick={handleNewChat} aria-label={$t('sidebar.new_chat')} title={$t('sidebar.new_chat')}>+</button>
				</div>
				{#each navItems as item (item.href)}
					<a
						href={item.href}
						class="nav-item"
						class:active={isActive(item.href)}
						aria-current={isActive(item.href) ? 'page' : undefined}>{$t(item.labelKey)}</a
					>
				{/each}
				{#if debugEnabled}
					<span class="debug-badge">{$t('sidebar.debug_mode')}</span>
				{/if}
			</nav>


			<!-- sidebar_widget スロット -->
			{#if $themeSlots.sidebar_widget}
				{@const SidebarWidget = $themeSlots.sidebar_widget}
				<div class="sidebar-widget-slot">
					<SidebarWidget />
				</div>
			{/if}

			<div class="sidebar-footer">
				<div class="footer-row">
					<span class="footer-label">{$t('settings.light')}/{$t('settings.dark')}</span>
					<button
						class="toggle-switch"
						class:on={$colorMode === 'dark'}
						onclick={toggleColorMode}
						aria-label={$t('settings.color_mode')}
					>
						<span class="knob"></span>
					</button>
				</div>
				<div class="footer-row">
					<span class="footer-label">{$t('settings.language')}</span>
					<select
						class="footer-select"
						value={$locale}
						disabled={$promptLocaleSwitching}
						onchange={(e) => {
							// switchLocale() は performPromptLocaleSwitch() 失敗を内部の
							// try/catch で自前捕捉しトーストも出す (例外は再送出しない)。
							// このため以下の .catch() が実際に発火するのは setLocale()
							// (UI 表示言語の PUT /api/config/locale、prompt_locale とは
							// 無関係) が失敗した場合のみ。誤って prompt_locale_failed
							// (チャット応答言語の切替失敗) を出さないよう、UI 表示言語
							// 専用の文言を使う (2026-07-22 監査で判明)。
							switchLocale(e.currentTarget.value).catch((err: unknown) => {
								addToast({ type: 'error', i18nKey: 'settings.i18n.locale_failed' });
								console.error('[Locale Switch]', err);
							});
						}}
					>
						{#each availableLocales as loc}
							<option value={loc}>{loc.toUpperCase()}</option>
						{/each}
					</select>
				</div>
				{#if showPro}
					<div class="footer-row">
						<span class="footer-label">{$t('sidebar.mode')}</span>
						<div class="mode-select-wrapper">
							<select
								class="footer-select"
								value={$currentMode}
								disabled={$modeRestartStatus === 'restarting' || $isStreaming}
								title={$currentMode === 'create' ? $t('sidebar.mode_create_desc') : $t('sidebar.mode_chat_desc')}
								onchange={(e) => {
								switchMode(e.currentTarget.value).catch((err: unknown) => {
									addToast({ type: 'error', i18nKey: 'sidebar.mode_restart_failed' });
									console.error('[Mode Switch]', err);
								});
							}}
							>
								<option value="chat" title={$t('sidebar.mode_chat_desc')}>{$t('sidebar.mode_chat')}</option>
								<option value="create" title={$t('sidebar.mode_create_desc')}>{$t('sidebar.mode_create')}</option>
							</select>
						</div>
					</div>
					{#if $modeRestartStatus === 'restarting'}
						<div class="mode-status mode-restarting">
							<span class="spinner"></span>
							{$t('sidebar.mode_restarting')}
						</div>
					{:else if $modeRestartStatus === 'ready'}
						<div class="mode-status mode-ready">
							{$t('sidebar.mode_restart_ready')}
						</div>
					{:else if $modeRestartStatus === 'failed'}
						<div class="mode-status mode-failed">
							{$t('sidebar.mode_restart_failed')}
						</div>
					{/if}
				{/if}
			</div>

			<div class="sidebar-version">
				{isDevelop
					? 'evoref develop'
					: $appVersion
						? `evoref v${$appVersion} ${edition}`
						: `evoref ${edition}`}
			</div>
		{/if}
	</aside>
{/if}

<style>
	.sidebar {
		width: var(--sidebar-width);
		display: flex;
		flex-direction: column;
		transition: width 0.2s ease;
		overflow: hidden;
		position: relative;
		z-index: 1;
		background: var(--sidebar-bg-start);
		border-right: 1px solid color-mix(in srgb, var(--sidebar-border) 60%, transparent);
		backdrop-filter: blur(20px) saturate(1.2);
		-webkit-backdrop-filter: blur(20px) saturate(1.2);
		flex-shrink: 0;
	}
	.sidebar.right {
		order: 1;
		border-right: none;
		border-left: 0.5px solid var(--sidebar-border);
	}
	.sidebar.collapsed {
		width: 48px;
	}
	.sidebar-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 18px 14px 14px;
	}
	.instance-name {
		font-size: 15px;
		font-weight: 650;
		letter-spacing: -0.02em;
		color: var(--text-primary);
		text-decoration: none;
		white-space: nowrap;
		overflow: hidden;
	}
	.collapse-btn {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 15px;
		padding: 2px 6px;
		opacity: 0.6;
		transition: opacity 0.15s;
	}
	.collapse-btn:hover {
		opacity: 1;
	}
	.sidebar-nav {
		flex: 1;
		display: flex;
		flex-direction: column;
		padding: 0 8px;
		gap: 2px;
	}
	.nav-chat-row {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.nav-chat-link {
		flex: 1;
		min-width: 0;
	}
	.new-chat-btn {
		flex-shrink: 0;
		width: 28px;
		height: 28px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: none;
		border: 1px solid color-mix(in srgb, var(--input-border) 60%, transparent);
		border-radius: var(--radius-lg);
		color: var(--text-secondary);
		font-size: 16px;
		cursor: pointer;
		transition: all 0.15s var(--ease);
		line-height: 1;
	}
	.new-chat-btn:hover {
		background: color-mix(in srgb, var(--accent) 15%, transparent);
		color: var(--text-primary);
		border-color: var(--accent);
	}
	/* テキスト中心の縦リスト。現在地は面の濃淡 + 太字で示す (色相に頼らない) */
	.nav-item {
		display: block;
		padding: 7px 12px;
		border-radius: var(--radius-lg);
		color: var(--text-secondary);
		text-decoration: none;
		font-size: var(--font-size-base);
		transition:
			background-color 0.15s,
			color 0.15s;
	}

	.nav-item:hover {
		color: var(--text-primary);
		background: color-mix(in srgb, var(--text-primary) 6%, transparent);
	}
	.nav-item.active {
		background: color-mix(in srgb, var(--text-primary) 9%, transparent);
		color: var(--text-primary);
		font-weight: 600;
	}
	.debug-badge {
		display: block;
		margin: 6px 10px 2px;
		padding: 3px 10px;
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.5px;
		text-align: center;
		border-radius: 4px;
		opacity: 0.85;
	}
	.debug-badge {
		color: var(--color-warning);
		border: 1.5px solid var(--color-warning);
	}
	.sidebar-widget-slot {
		padding: 8px 6px;
		border-top: 1px solid color-mix(in srgb, var(--sidebar-border) 50%, transparent);
	}
	.sidebar-footer {
		border-top: 1px solid color-mix(in srgb, var(--sidebar-border) 50%, transparent);
		padding: 12px 8px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.footer-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 4px;
	}
	.footer-label {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
	}
	.footer-select {
		background: var(--control-bg);
		color: var(--text-primary);
		border: 1px solid color-mix(in srgb, var(--input-border) 60%, transparent);
		border-radius: var(--radius-sm);
		font-size: var(--font-size-sm);
		padding: 3px 6px;
		outline: none;
		min-width: 56px;
		font-family: inherit;
	}
	.footer-select:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
	}

	/* トグルスイッチ */
	.toggle-switch {
		position: relative;
		width: 32px;
		height: 16px;
		background: var(--border);
		border-radius: 8px;
		border: 1px solid var(--accent);
		cursor: pointer;
		flex-shrink: 0;
		padding: 0;
		transition: background 0.2s, border-color 0.2s;
	}
	.toggle-switch.on {
		background: var(--accent);
		border-color: var(--accent);
	}
	.toggle-switch .knob {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 10px;
		height: 10px;
		background: var(--text-secondary);
		border-radius: 50%;
		transition: transform 0.2s;
		display: block;
	}
	.toggle-switch.on .knob {
		transform: translateX(15px);
		background: var(--text-on-accent);
	}
	.mode-select-wrapper {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.mode-status {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 2px 4px;
		font-size: 12px;
		border-radius: 4px;
	}
	.mode-restarting {
		color: var(--text-secondary);
	}
	.mode-ready {
		color: var(--color-success);
	}
	.mode-failed {
		color: var(--color-error);
		font-weight: 600;
	}
	.spinner {
		display: inline-block;
		width: 12px;
		height: 12px;
		border: 2px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
	.sidebar-version {
		padding: 12px 10px 16px;
		font-size: var(--font-size-xs);
		color: var(--text-muted);
	}
</style>
