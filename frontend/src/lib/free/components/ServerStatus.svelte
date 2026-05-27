<script lang="ts">
	import { t } from '$lib/i18n';
	import { serverState, refreshServerStatus, busyServer } from '$lib/free/stores/server';
	import { vramStatus, refreshVramStatus } from '$lib/free/stores/vram';
	import { startServer, stopServer } from '$lib/free/api';
	import type { ServerName, VramModelInfo } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';

	const COMPONENT_LABELS = [
		'sidebar.component_base',
		'sidebar.component_assist',
		'sidebar.component_embed',
		'sidebar.component_reranker'
	];

	const SERVER_NAMES: ServerName[] = ['base', 'assist', 'embed', 'reranker'];

	// サイドバーで起動状態と VRAM を見せる対象は base / assist のみに絞る
	// (embed / reranker は CPU 既定で 0 MB 表示になり混乱を招くため非表示)
	const VISIBLE_NAMES: readonly ServerName[] = ['base', 'assist'];

	let refreshing = $state(false);

	async function handleReconnect() {
		refreshing = true;
		await Promise.all([refreshServerStatus(), refreshVramStatus()]);
		refreshing = false;
	}

	// VRAM 表示は「実際に接続中か (connected)」で gate する。停止中は推定値があっても
	// "—" にし、緑ランプ (= connected) と常に一致させる。
	function vramInline(
		m: VramModelInfo | undefined,
		connected: boolean,
	): { text: string; cls: string } {
		if (!connected) return { text: '—', cls: 'absent' };
		if (!$vramStatus || !m) return { text: '—', cls: 'absent' };
		if (!m.present) return { text: '—', cls: 'absent' };
		if (m.placement === 'CPU') return { text: $t('vram.placement_cpu'), cls: 'cpu' };
		if (m.placement === 'none') return { text: '—', cls: 'absent' };
		return { text: `${m.vram_mb} MB`, cls: 'gpu' };
	}

	/** 前回の起動失敗を記録（force restart 提案用） */
	let lastStartFailed = $state<ServerName | null>(null);

	async function handleToggle(i: number, connected: boolean) {
		const name = SERVER_NAMES[i];
		busyServer.set(name);
		try {
			if (connected) {
				await handleApiCall(() => stopServer(name), {
					fallbackKey: 'sidebar.stop_failed'
				});
				lastStartFailed = null;
			} else {
				const result = await handleApiCall(() => startServer(name), {
					fallbackKey: 'sidebar.start_failed'
				});
				if (result && !result.success) {
					lastStartFailed = name;
				} else {
					lastStartFailed = null;
				}
			}
			await Promise.all([refreshServerStatus(), refreshVramStatus()]);
		} finally {
			busyServer.set(null);
		}
	}

	async function handleForceRestart(i: number) {
		const name = SERVER_NAMES[i];
		busyServer.set(name);
		try {
			await handleApiCall(() => startServer(name, true), {
				fallbackKey: 'sidebar.start_failed'
			});
			lastStartFailed = null;
			await Promise.all([refreshServerStatus(), refreshVramStatus()]);
		} finally {
			busyServer.set(null);
		}
	}
</script>

<div class="server-status">
	<div class="status-header">
		<span class="footer-label">{$t('sidebar.server_status')}</span>
		<button
			class="reconnect-btn"
			onclick={handleReconnect}
			disabled={refreshing}
			aria-label={$t('sidebar.reconnect')}
			title={$t('sidebar.reconnect')}
		>
			<span class="reconnect-icon" class:spinning={refreshing}>⟳</span>
		</button>
	</div>
	{#if !$serverState.backendOnline}
		<div class="status-offline">{$t('sidebar.backend_offline')}</div>
	{:else}
		<div class="component-list">
			{#each $serverState.components as comp, i}
				{#if VISIBLE_NAMES.includes(SERVER_NAMES[i])}
				{@const busy = $busyServer === SERVER_NAMES[i]}
				{@const failed = lastStartFailed === SERVER_NAMES[i]}
				{@const m = $vramStatus?.models?.find((x) => x.name === SERVER_NAMES[i])}
				{@const v = vramInline(m, comp.connected)}
				<div class="component-row">
					<span class="status-dot" class:connected={comp.connected} class:failed={failed && !comp.connected}></span>
					<span class="component-name" title={comp.name || $t(COMPONENT_LABELS[i])}>
						{$t(COMPONENT_LABELS[i])}
					</span>
					<span class="vram-cell {v.cls}">{v.text}</span>
					<div class="action-group">
						{#if failed && !comp.connected}
							<button
								class="action-btn force"
								disabled={busy}
								onclick={() => handleForceRestart(i)}
								title={$t('sidebar.force_restart')}
							>
								{#if busy}
									<span class="action-icon spinning">⟳</span>
								{:else}
									⟳!
								{/if}
							</button>
						{/if}
						<button
							class="action-btn"
							class:stop={comp.connected}
							disabled={busy}
							onclick={() => handleToggle(i, comp.connected)}
							title={busy
								? (comp.connected ? $t('sidebar.stopping') : $t('sidebar.starting'))
								: (comp.connected ? $t('sidebar.stop') : $t('sidebar.start'))}
						>
							{#if busy}
								<span class="action-icon spinning">⟳</span>
							{:else if comp.connected}
								■
							{:else}
								▶
							{/if}
						</button>
					</div>
				</div>
				{/if}
			{/each}
		</div>
	{/if}
</div>

<style>
	.server-status {
		padding: 0 4px;
		width: 100%;
	}
	.status-header {
		display: flex;
		align-items: center;
		gap: 4px;
		margin-bottom: 4px;
	}
	.status-header .footer-label {
		flex: 1;
		min-width: 0;
	}
	.reconnect-btn {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 16px;
		padding: 0 2px;
		line-height: 1;
		opacity: 0.6;
		transition: opacity 0.15s;
	}
	.reconnect-btn:hover {
		opacity: 1;
	}
	.reconnect-btn:disabled {
		cursor: default;
		opacity: 0.3;
	}
	.reconnect-icon {
		display: inline-block;
	}
	.reconnect-icon.spinning,
	.action-icon.spinning {
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
	}
	.status-offline {
		font-size: 12px;
		color: var(--text-secondary);
		opacity: 0.7;
		padding: 2px 0;
	}
	.component-list {
		display: flex;
		flex-direction: column;
		gap: 3px;
	}
	.component-row {
		display: grid;
		grid-template-columns: 7px minmax(0, 1fr) auto auto;
		align-items: center;
		gap: 5px;
		min-width: 0;
	}
	.status-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		flex-shrink: 0;
		background: var(--text-secondary);
		opacity: 0.4;
	}
	.status-dot.connected {
		background: #22c55e;
		opacity: 1;
	}
	.status-dot.failed {
		background: #ef4444;
		opacity: 1;
	}
	.component-name {
		font-size: 12px;
		color: var(--text-secondary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		min-width: 0;
	}
	.vram-cell {
		font-size: 11px;
		color: var(--text-secondary);
		text-align: right;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		min-width: 48px;
		padding: 0 2px;
		opacity: 0.85;
	}
	.vram-cell.cpu {
		opacity: 0.6;
	}
	.vram-cell.absent {
		opacity: 0.4;
	}
	.action-group {
		display: flex;
		align-items: center;
		gap: 3px;
		flex-shrink: 0;
	}
	.action-btn {
		background: none;
		border: 1px solid var(--input-border);
		border-radius: 3px;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 9px;
		padding: 1px 4px;
		line-height: 1.2;
		opacity: 0.7;
		transition: opacity 0.15s, color 0.15s, border-color 0.15s;
		flex-shrink: 0;
	}
	.action-btn:hover {
		opacity: 1;
		color: var(--accent);
		border-color: var(--accent);
	}
	.action-btn.stop:hover {
		color: #ef4444;
		border-color: #ef4444;
	}
	.action-btn.force {
		color: #f59e0b;
		border-color: #f59e0b;
		opacity: 1;
	}
	.action-btn.force:hover {
		color: #d97706;
		border-color: #d97706;
	}
	.action-btn:disabled {
		cursor: default;
		opacity: 0.3;
	}
	.action-icon {
		display: inline-block;
		font-size: 11px;
	}
</style>
