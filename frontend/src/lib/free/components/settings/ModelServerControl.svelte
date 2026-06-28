<script lang="ts">
	/**
	 * モデル別サーバー制御 (単一 llama-server)
	 *
	 * モデルページの各モデル種別 (base / assist / embed) の直下に置き、対応する
	 * llama-server プロセスの起動・停止・強制再起動と接続状態 / VRAM を表示する。
	 * モデル切替 (ComponentMigrateButton) 後に「必要なサーバの再起動」をその場で
	 * 行えるようにするのが目的 (旧: 開発ページの ServerStatus を種別ごとに分割移設)。
	 *
	 * 状態・操作中スピナーは ServerStatus と同じグローバルストア
	 * (serverState / vramStatus / busyServer) を共有し、ポーリングは +layout 側で駆動する。
	 * embed サーバーの ServerName は "embed" (埋め込みモデルの migrate 種別 "embedding"
	 * とは別語) なので、埋め込みモデル欄では server="embed" を渡す。
	 */
	import { t } from '$lib/i18n';
	import { serverState, refreshServerStatus, busyServer } from '$lib/free/stores/server';
	import { vramStatus, refreshVramStatus } from '$lib/free/stores/vram';
	import { startServer, stopServer } from '$lib/free/api';
	import type { ServerName, VramModelInfo } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';

	type Props = { server: ServerName };
	let { server }: Props = $props();

	// /api/status の ComponentStatus.name はモデル識別子 (model_id / ファイル stem /
	// model_name) で、ロール名 (base/assist/embed) は持たない。一方 status.py の
	// _collect_component_statuses は必ず base → assist → embed の固定順で append するため、
	// ロールは name 一致ではなく index で引く (旧 ServerStatus も同じ前提)。
	// VRAM 側 (vram_monitor MODEL_NAMES) はロール名キーなので vramInline の find は name 一致で正しい。
	const SERVER_INDEX: Record<ServerName, number> = { base: 0, assist: 1, embed: 2 };
	let comp = $derived($serverState.components[SERVER_INDEX[server]]);
	let connected = $derived(comp?.connected ?? false);
	let busy = $derived($busyServer === server);

	/** 直前の起動失敗を記録 (force restart 提案用) */
	let startFailed = $state(false);

	// VRAM 表示は「実際に接続中か」で gate する。停止中は推定値があっても "—" にし、
	// 緑ランプ (= connected) と常に一致させる (ServerStatus と同方針)。
	function vramInline(
		m: VramModelInfo | undefined,
		isConnected: boolean,
	): { text: string; cls: string } {
		if (!isConnected) return { text: '—', cls: 'absent' };
		if (!$vramStatus || !m) return { text: '—', cls: 'absent' };
		if (!m.present) return { text: '—', cls: 'absent' };
		if (m.placement === 'CPU') return { text: $t('vram.placement_cpu'), cls: 'cpu' };
		if (m.placement === 'none') return { text: '—', cls: 'absent' };
		return { text: `${m.vram_mb} MB`, cls: 'gpu' };
	}

	let vram = $derived(
		vramInline($vramStatus?.models?.find((m) => m.name === server), connected),
	);

	async function refresh() {
		await Promise.all([refreshServerStatus(), refreshVramStatus()]);
	}

	async function handleStart(force: boolean) {
		busyServer.set(server);
		try {
			const result = await handleApiCall(() => startServer(server, force), {
				fallbackKey: 'sidebar.start_failed',
			});
			startFailed = Boolean(result && !result.success);
			await refresh();
		} finally {
			busyServer.set(null);
		}
	}

	async function handleStop() {
		busyServer.set(server);
		try {
			await handleApiCall(() => stopServer(server), {
				fallbackKey: 'sidebar.stop_failed',
			});
			startFailed = false;
			await refresh();
		} finally {
			busyServer.set(null);
		}
	}
</script>

<div class="model-server" data-testid="model-server-{server}">
	<span class="srv-label">{$t('sidebar.server_status')}</span>
	{#if !$serverState.backendOnline}
		<span class="srv-offline">{$t('sidebar.backend_offline')}</span>
	{:else}
		<span class="status-dot" class:connected class:failed={startFailed && !connected}></span>
		<span class="vram-cell {vram.cls}">{vram.text}</span>
		<div class="actions">
			{#if connected}
				<button
					type="button"
					class="srv-btn"
					disabled={busy}
					onclick={() => handleStart(true)}
					title={$t('sidebar.force_restart')}
					aria-label={$t('sidebar.restart')}
				>
					{#if busy}<span class="ic spinning">⟳</span>{:else}{$t('sidebar.restart')}{/if}
				</button>
				<button
					type="button"
					class="srv-btn stop"
					disabled={busy}
					onclick={handleStop}
					title={$t('sidebar.stop')}
					aria-label={$t('sidebar.stop')}
				>
					{#if busy}<span class="ic spinning">⟳</span>{:else}{$t('sidebar.stop')}{/if}
				</button>
			{:else}
				{#if startFailed}
					<button
						type="button"
						class="srv-btn force"
						disabled={busy}
						onclick={() => handleStart(true)}
						title={$t('sidebar.force_restart')}
						aria-label={$t('sidebar.force_restart')}
					>
						{#if busy}<span class="ic spinning">⟳</span>{:else}{$t('sidebar.restart')}{/if}
					</button>
				{/if}
				<button
					type="button"
					class="srv-btn"
					disabled={busy}
					onclick={() => handleStart(false)}
					title={$t('sidebar.start')}
					aria-label={$t('sidebar.start')}
				>
					{#if busy}<span class="ic spinning">⟳</span>{:else}{$t('sidebar.start')}{/if}
				</button>
			{/if}
		</div>
	{/if}
</div>

<style>
	.model-server {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 0.5rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--border-color, var(--border));
	}
	.srv-label {
		font-size: 0.85rem;
		color: var(--text-secondary, #555);
	}
	.srv-offline {
		font-size: 0.8rem;
		color: var(--text-secondary, #555);
		opacity: 0.7;
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
	.vram-cell {
		font-size: 11px;
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		opacity: 0.85;
	}
	.vram-cell.cpu {
		opacity: 0.6;
	}
	.vram-cell.absent {
		opacity: 0.4;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-left: auto;
	}
	.srv-btn {
		background: var(--bg-secondary, #f5f5f5);
		border: 1px solid var(--input-border, var(--border-color, #999));
		border-radius: 4px;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 0.85rem;
		font-family: inherit;
		padding: 0.3rem 0.7rem;
		line-height: 1.2;
		transition: opacity 0.15s, color 0.15s, border-color 0.15s;
	}
	.srv-btn:hover:not(:disabled) {
		color: var(--accent);
		border-color: var(--accent);
	}
	.srv-btn.stop:hover:not(:disabled) {
		color: #ef4444;
		border-color: #ef4444;
	}
	.srv-btn.force {
		color: #f59e0b;
		border-color: #f59e0b;
	}
	.srv-btn.force:hover:not(:disabled) {
		color: #d97706;
		border-color: #d97706;
	}
	.srv-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.ic {
		display: inline-block;
		font-size: 14px;
	}
	.ic.spinning {
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
