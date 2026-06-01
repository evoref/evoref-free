<script lang="ts">
	/**
	 * Develop 設定ページ — local/ データ初期化 (Danger Zone)
	 *
	 * Develop エディションでのみ表示される (SettingsTabs の DEVELOP_ONLY_TABS)。
	 * 「ローカルデータを初期化」ボタンで確認ダイアログを出し、確定すると
	 * backend のデタッチヘルパーが全サービス停止 → local/ wipe → 再起動を行う。
	 * その間 UI は「再起動中」オーバーレイで /api/status をポーリングし、復帰したら
	 * 自動リロードする。
	 */
	import { t } from '$lib/i18n';
	import { isDevelop } from '$lib/edition';
	import DialogShell from '$lib/free/components/DialogShell.svelte';
	import ServerStatus from '$lib/free/components/ServerStatus.svelte';
	import { resetLocalData, getStatus, ApiError } from '$lib/free/api';

	type Phase = 'idle' | 'confirming' | 'resetting' | 'restarting' | 'error';

	let phase = $state<Phase>('idle');
	let errorMessage = $state<string | null>(null);
	let restartTimedOut = $state(false);

	const RESTART_POLL_INTERVAL_MS = 2000;
	const RESTART_TIMEOUT_MS = 180_000;

	function openConfirm() {
		errorMessage = null;
		phase = 'confirming';
	}

	function cancelConfirm() {
		if (phase === 'confirming') phase = 'idle';
	}

	async function confirmReset() {
		phase = 'resetting';
		errorMessage = null;
		try {
			await resetLocalData(true);
			// 202 受領 = ヘルパー起動済み。以降 backend は停止に向かう。
			phase = 'restarting';
			pollForRestart();
		} catch (e) {
			// backend が応答前に落ちた場合もここに来うるが、その時は実際には
			// リセットが進行している可能性が高いので restart 監視へ倒す。
			if (e instanceof ApiError) {
				errorMessage = e.message;
				phase = 'error';
			} else {
				phase = 'restarting';
				pollForRestart();
			}
		}
	}

	async function pollForRestart() {
		const deadline = Date.now() + RESTART_TIMEOUT_MS;
		// 一旦 backend が確実に落ちる猶予を取ってからポーリング開始する
		// (停止前の生きた /api/status を「復帰」と誤検知しないため)。
		await sleep(RESTART_POLL_INTERVAL_MS * 3);
		while (Date.now() < deadline) {
			try {
				await getStatus();
				// 応答が返った = backend 復帰。少し待ってからフルリロード。
				await sleep(1000);
				location.reload();
				return;
			} catch {
				// まだ落ちている / 再起動途中。待って再試行。
			}
			await sleep(RESTART_POLL_INTERVAL_MS);
		}
		restartTimedOut = true;
	}

	function sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}
</script>

{#if !isDevelop}
	<div class="not-available">
		<p>{$t('settings.develop.unavailable')}</p>
	</div>
{:else}
	<div class="develop-settings">
		<section class="server-section">
			<h2 class="section-title">{$t('settings.develop.server_title')}</h2>
			<p class="section-hint">{$t('settings.develop.server_hint')}</p>
			<ServerStatus />
		</section>

		<section class="danger-zone">
			<h2 class="zone-title">{$t('settings.develop.title')}</h2>
			<p class="zone-desc">{$t('settings.develop.description')}</p>

			<div class="action-row">
				<div class="action-text">
					<div class="action-name">{$t('settings.develop.reset_title')}</div>
					<div class="action-hint">{$t('settings.develop.reset_hint')}</div>
				</div>
				<button class="btn-danger" onclick={openConfirm} disabled={phase !== 'idle'}>
					{$t('settings.develop.reset_button')}
				</button>
			</div>

			{#if phase === 'error' && errorMessage}
				<div class="error-box">{errorMessage}</div>
			{/if}
		</section>
	</div>
{/if}

{#if phase === 'confirming'}
	<DialogShell
		ariaLabel={$t('settings.develop.confirm_title')}
		onClose={cancelConfirm}
		minWidth="360px"
		maxWidth="480px"
	>
		<h3 class="dialog-title">{$t('settings.develop.confirm_title')}</h3>
		<div class="warning-box">{$t('settings.develop.warning')}</div>
		<p class="confirm-body">{$t('settings.develop.confirm_body')}</p>
		<div class="dialog-actions">
			<button class="btn-cancel" onclick={cancelConfirm}>{$t('common.cancel')}</button>
			<button class="btn-danger" onclick={confirmReset}>
				{$t('settings.develop.reset_button')}
			</button>
		</div>
	</DialogShell>
{/if}

{#if phase === 'resetting' || phase === 'restarting'}
	<div class="overlay" role="alertdialog" aria-modal="true" aria-label={$t('settings.develop.restart_wait')}>
		<div class="overlay-card">
			{#if !restartTimedOut}
				<div class="spinner" aria-hidden="true"></div>
				<p class="overlay-title">{$t('settings.develop.restart_wait')}</p>
				<p class="overlay-sub">{$t('settings.develop.restart_sub')}</p>
			{:else}
				<p class="overlay-title">{$t('settings.develop.restart_timeout')}</p>
				<button class="btn-cancel" onclick={() => location.reload()}>
					{$t('settings.develop.reload_now')}
				</button>
			{/if}
		</div>
	</div>
{/if}

<style>
	.develop-settings {
		padding: 24px;
		max-width: 720px;
	}
	.server-section {
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 20px;
		margin-bottom: 24px;
	}
	.section-title {
		margin: 0 0 6px;
		font-size: 1.05rem;
		color: var(--text-primary);
		font-weight: 600;
	}
	.section-hint {
		margin: 0 0 18px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.6;
	}
	.danger-zone {
		border: 1px solid var(--color-error);
		border-radius: 8px;
		padding: 20px;
		background: color-mix(in srgb, var(--color-error) 5%, transparent);
	}
	.zone-title {
		margin: 0 0 6px;
		font-size: 1.05rem;
		color: var(--color-error);
		font-weight: 600;
	}
	.zone-desc {
		margin: 0 0 18px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.6;
	}
	.action-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
	}
	.action-name {
		font-size: 14px;
		font-weight: 500;
		color: var(--text-primary);
	}
	.action-hint {
		font-size: 12px;
		color: var(--text-muted);
		margin-top: 2px;
	}
	.btn-danger {
		background: var(--color-error);
		color: var(--text-on-accent);
		border: none;
		border-radius: 6px;
		padding: 8px 16px;
		font-size: 13px;
		font-family: inherit;
		font-weight: 500;
		cursor: pointer;
		white-space: nowrap;
	}
	.btn-danger:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn-cancel {
		background: none;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
		border-radius: 6px;
		padding: 8px 16px;
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
	}
	.error-box {
		margin-top: 14px;
		padding: 8px 12px;
		border-radius: 4px;
		background: color-mix(in srgb, var(--color-error) 12%, transparent);
		color: var(--color-error);
		font-size: 13px;
	}
	.not-available {
		padding: 24px;
		color: var(--text-muted);
		font-size: 14px;
	}
	.dialog-title {
		margin: 0 0 16px;
		font-size: 1.1rem;
		color: var(--text-primary);
	}
	.warning-box {
		background: color-mix(in srgb, var(--color-error) 12%, transparent);
		border: 1px solid var(--color-error);
		color: var(--color-error);
		padding: 12px;
		border-radius: 4px;
		margin-bottom: 14px;
		font-size: 13px;
		font-weight: 500;
	}
	.confirm-body {
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.6;
		margin: 0 0 20px;
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: 10px;
	}
	.overlay {
		position: fixed;
		inset: 0;
		background-color: var(--overlay-bg);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1100;
	}
	.overlay-card {
		background: var(--bg-primary);
		border: 1px solid var(--border);
		border-radius: 12px;
		padding: 32px 40px;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
		box-shadow: 0 8px 32px var(--shadow-dialog);
	}
	.overlay-title {
		margin: 0;
		font-size: 15px;
		font-weight: 600;
		color: var(--text-primary);
	}
	.overlay-sub {
		margin: 0;
		font-size: 12px;
		color: var(--text-muted);
	}
	.spinner {
		width: 32px;
		height: 32px;
		border: 3px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
