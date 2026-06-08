<script lang="ts">
	/**
	 * ベースモデル切替ボタン (migrate 専用 UI)
	 *
	 * config 直書きは禁止 (バックエンドが 403)。base モデルの切替は必ず
	 * `POST /api/model/migrate` 経由で行い、config.yaml と model_state.json を
	 * 原子的に同期する。重い副作用 (LoRA アーカイブ / perplexity リセット /
	 * プロンプトメタリセット) を伴うため、dry_run プレビューを挟む。
	 *
	 * フロー: パス入力 → プレビュー(dry_run) → 切替(migrate) →
	 *         ベースサーバー再起動 → 再接続(reload)。
	 */
	import { t } from '$lib/i18n';
	import {
		migrateBaseModel,
		rollbackBaseModel,
		reloadModel,
		restartBaseProcess,
		ApiError,
		type BaseMigrateResponse
	} from '$lib/free/api';

	type Props = {
		currentModel: string;
		/** migrate/rollback 成功後に config を再取得させるコールバック */
		onMigrated?: () => void;
	};

	let { currentModel, onMigrated }: Props = $props();

	let newPath = $state('');
	let tryLora = $state(false);
	let busy = $state(false);
	let preview = $state<BaseMigrateResponse | null>(null);
	let result = $state<BaseMigrateResponse | null>(null);
	let restarted = $state(false);
	let reloaded = $state(false);
	let error = $state<string | null>(null);

	let canRollback = $derived(result !== null && !result.dry_run);

	function toError(e: unknown): string {
		if (e instanceof ApiError) return e.message;
		if (e instanceof Error) return e.message;
		return String(e);
	}

	async function doPreview() {
		if (!newPath || busy) return;
		busy = true;
		error = null;
		preview = null;
		result = null;
		try {
			preview = await migrateBaseModel({
				new_model_path: newPath,
				try_lora: tryLora,
				dry_run: true
			});
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function doMigrate() {
		if (!newPath || busy) return;
		busy = true;
		error = null;
		restarted = false;
		reloaded = false;
		try {
			result = await migrateBaseModel({
				new_model_path: newPath,
				try_lora: tryLora,
				dry_run: false
			});
			preview = null;
			onMigrated?.();
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function doRestart() {
		if (busy) return;
		busy = true;
		error = null;
		try {
			const r = await restartBaseProcess();
			restarted = r.restarted;
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function doReload() {
		if (busy) return;
		busy = true;
		error = null;
		try {
			const r = await reloadModel();
			reloaded = r.reloaded;
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function doRollback() {
		if (busy) return;
		busy = true;
		error = null;
		try {
			const r = await rollbackBaseModel();
			result = {
				dry_run: false,
				old_model: result?.new_model ?? '',
				new_model: r.rolled_back_to,
				lora_action: r.lora_restored ? 'restored' : '',
				data_summary: {
					memory_notes: 0,
					experience_entries: 0,
					perplexity_reset: 0,
					rag_chunks: 0,
					cartridges: 0,
					prompts_modes: []
				},
				calibration: null,
				recommendations: [$t('settings.model_migrate.rollback_done')]
			};
			onMigrated?.();
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="base-migrate">
	<div class="current" data-testid="base-current">
		{$t('settings.model_migrate.current_model', { model: currentModel || '—' })}
	</div>

	<input
		type="text"
		class="path-input"
		placeholder={$t('settings.model_migrate.new_model_path')}
		bind:value={newPath}
		disabled={busy}
	/>

	<label class="try-lora">
		<input type="checkbox" bind:checked={tryLora} disabled={busy} />
		{$t('settings.model_migrate.try_lora')}
	</label>

	<p class="notice">{$t('settings.model_migrate.base_notice')}</p>

	<div class="actions">
		<button type="button" disabled={busy || !newPath} onclick={doPreview}>
			{busy ? $t('settings.model_migrate.previewing') : $t('settings.model_migrate.preview')}
		</button>
		<button
			type="button"
			class="apply-btn"
			disabled={busy || !newPath}
			onclick={doMigrate}
		>
			{busy ? $t('settings.model_migrate.applying') : $t('settings.model_migrate.base_apply')}
		</button>
		{#if canRollback}
			<button type="button" class="rollback-btn" disabled={busy} onclick={doRollback}>
				{$t('settings.model_migrate.rollback')}
			</button>
		{/if}
	</div>

	{#if preview}
		<div class="result preview" data-testid="base-preview">
			{$t('settings.model_migrate.dry_run_summary', {
				experience: preview.data_summary.experience_entries,
				perplexity: preview.data_summary.perplexity_reset,
				lora: preview.lora_action
			})}
		</div>
	{/if}

	{#if result}
		<div class="result success" data-testid="base-result">
			<div><strong>{result.old_model}</strong> → <strong>{result.new_model}</strong></div>
			<p class="restart-hint">{$t('settings.model_migrate.base_restart_needed')}</p>
			<div class="actions">
				<button type="button" disabled={busy} onclick={doRestart}>
					{busy ? $t('settings.model_migrate.restarting') : $t('settings.model_migrate.restart_base')}
				</button>
				<button type="button" disabled={busy} onclick={doReload}>
					{busy ? $t('settings.model_migrate.reloading') : $t('settings.model_migrate.reload')}
				</button>
			</div>
			{#if restarted}
				<span class="badge ok">{$t('settings.model_migrate.restarted')}</span>
			{/if}
			{#if reloaded}
				<span class="badge ok">{$t('settings.model_migrate.reloaded')}</span>
			{/if}
		</div>
	{/if}

	{#if error}
		<div class="result error" data-testid="base-error">{error}</div>
	{/if}
</div>

<style>
	.base-migrate {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-top: 0.5rem;
	}
	.current {
		font-size: 0.85rem;
		color: var(--text-secondary, #555);
	}
	.path-input {
		padding: 0.4rem 0.6rem;
		border-radius: 4px;
		border: 1px solid var(--border-color, #999);
		background: var(--bg-primary, #fff);
		color: var(--text-primary, #222);
		font-size: 0.9rem;
	}
	.try-lora {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		font-size: 0.85rem;
	}
	.notice {
		margin: 0;
		font-size: 0.8rem;
		color: var(--text-warning, #92400e);
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	button {
		padding: 0.4rem 0.8rem;
		border-radius: 4px;
		border: 1px solid var(--border-color, #999);
		background: var(--bg-secondary, #f5f5f5);
		color: var(--text-primary, #222);
		cursor: pointer;
		font-size: 0.9rem;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	button.apply-btn {
		background: var(--accent-bg, #2563eb);
		color: var(--accent-text, #fff);
		border-color: var(--accent-bg, #2563eb);
	}
	.result {
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.85rem;
	}
	.result.preview {
		background: var(--bg-info-soft, #eff6ff);
		color: var(--text-info, #1e40af);
		border: 1px solid var(--border-info, #93c5fd);
	}
	.result.success {
		background: var(--bg-success-soft, #ecfdf5);
		color: var(--text-success, #065f46);
		border: 1px solid var(--border-success, #6ee7b7);
	}
	.result.error {
		background: var(--bg-error-soft, #fef2f2);
		color: var(--text-error, #991b1b);
		border: 1px solid var(--border-error, #fca5a5);
	}
	.restart-hint {
		margin: 0.25rem 0;
		color: var(--text-warning, #92400e);
	}
	.badge {
		display: inline-block;
		margin-right: 0.5rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		font-size: 0.75rem;
		background: var(--accent-bg, #2563eb);
		color: var(--accent-text, #fff);
	}
</style>
