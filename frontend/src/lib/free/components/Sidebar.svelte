<script lang="ts">
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { t } from '$lib/i18n';
	import { colorMode, toggleColorMode, sidebarCollapsed, layout, themeSlots } from '$lib/free/stores/theme';
	import { locale, availableLocales, setLocale } from '$lib/i18n';
	import { isPro, isDevelop, edition } from '$lib/edition';
	import { appVersion } from '$lib/free/stores/app';
	import { currentMode, clearMessages, switchMode, modeRestartStatus } from '$lib/free/stores/chat';
	import { addToast } from '$lib/free/stores/toast';
	import { serverState } from '$lib/free/stores/server';

	let { instanceName = 'evoref' }: { instanceName?: string } = $props();
	let position = $derived($layout.sidebar.position);
	let collapsed = $derived($sidebarCollapsed);
	let debugEnabled = $derived($serverState.debug?.enabled === true);

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
					<a href="/" class="nav-item nav-chat-link" class:active={isActive('/')}>{$t('sidebar.chat')}</a>
					<button class="new-chat-btn" onclick={handleNewChat} aria-label={$t('sidebar.new_chat')} title={$t('sidebar.new_chat')}>+</button>
				</div>
				<a href="/history" class="nav-item" class:active={isActive('/history')}>{$t('sidebar.history')}</a>
				{#if isPro}
					<a href="/memory" class="nav-item" class:active={isActive('/memory')}>{$t('sidebar.memory')}</a>
				{/if}
				<a href="/cartridge" class="nav-item" class:active={isActive('/cartridge')}>{$t('sidebar.cartridges')}</a>
				<a href="/themes" class="nav-item" class:active={isActive('/themes')}>{$t('sidebar.themes')}</a>
				<a href="/loop" class="nav-item" class:active={isActive('/loop')}>{$t('sidebar.loop')}</a>
				{#if isPro}
					<a href="/dashboard" class="nav-item" class:active={isActive('/dashboard')}>{$t('sidebar.dashboard')}</a>
					<a href="/terminal" class="nav-item" class:active={isActive('/terminal')}>{$t('sidebar.terminal')}</a>
				{/if}
				<a href="/settings" class="nav-item" class:active={isActive('/settings')}>{$t('sidebar.settings')}</a>
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
						onchange={(e) => setLocale(e.currentTarget.value)}
					>
						{#each availableLocales as loc}
							<option value={loc}>{loc.toUpperCase()}</option>
						{/each}
					</select>
				</div>
				{#if isPro}
					<div class="footer-row">
						<span class="footer-label">{$t('sidebar.mode')}</span>
						<div class="mode-select-wrapper">
							<select
								class="footer-select"
								value={$currentMode}
								disabled={$modeRestartStatus === 'restarting'}
								onchange={(e) => {
								switchMode(e.currentTarget.value).catch((err: unknown) => {
									addToast({ type: 'error', i18nKey: 'sidebar.mode_restart_failed' });
									console.error('[Mode Switch]', err);
								});
							}}
							>
								<option value="chat">{$t('sidebar.mode_chat')}</option>
								<option value="coding">{$t('sidebar.mode_coding')}</option>
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
		background: linear-gradient(180deg, var(--sidebar-bg-start) 0%, var(--sidebar-bg-end) 100%);
		border-right: 0.5px solid var(--sidebar-border);
		backdrop-filter: blur(8px);
		-webkit-backdrop-filter: blur(8px);
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
		padding: 16px 10px;
	}
	.instance-name {
		font-size: 16px;
		font-weight: 700;
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
		padding: 0 6px;
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
		border: 1px solid var(--input-border);
		border-radius: 6px;
		color: var(--text-secondary);
		font-size: 16px;
		cursor: pointer;
		transition: all 0.15s;
		line-height: 1;
	}
	.new-chat-btn:hover {
		background: color-mix(in srgb, var(--accent) 15%, transparent);
		color: var(--text-primary);
		border-color: var(--accent);
	}
	.nav-item {
		display: block;
		padding: 7px 10px;
		border-radius: 6px;
		color: var(--text-primary);
		text-decoration: none;
		font-size: 14px;
		transition: all 0.15s;
	}

	.nav-item:hover {
		color: var(--text-primary);
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.nav-item.active {
		background: color-mix(in srgb, var(--accent) 35%, transparent);
		color: var(--text-primary);
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
	:global([data-theme='light']) .debug-badge {
		background: var(--warning, #f59e0b);
		color: var(--text-primary);
		border: 1px solid var(--warning, #f59e0b);
	}
	:global([data-theme='dark']) .debug-badge {
		background: transparent;
		color: var(--warning, #f59e0b);
		border: 1px solid var(--warning, #f59e0b);
	}
	.sidebar-widget-slot {
		padding: 8px 6px;
		border-top: 0.5px solid var(--sidebar-border);
	}
	.sidebar-footer {
		border-top: 0.5px solid var(--sidebar-border);
		padding: 10px 6px;
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
		font-size: 14px;
		color: var(--text-primary);
	}
	.footer-select {
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 14px;
		padding: 3px 4px;
		outline: none;
		min-width: 56px;
		font-family: inherit;
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
		background: #fff;
		border-radius: 50%;
		transition: transform 0.2s;
		display: block;
	}
	.toggle-switch.on .knob {
		transform: translateX(15px);
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
		color: var(--success, #22c55e);
	}
	.mode-failed {
		color: var(--error, #ef4444);
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
		font-size: 14px;
		color: var(--text-secondary);
		opacity: 0.7;
	}
</style>
