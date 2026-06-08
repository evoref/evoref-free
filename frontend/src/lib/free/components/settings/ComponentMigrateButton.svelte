<script lang="ts">
	/**
	 * コンポーネントモデル切替ボタン
	 *
	 * assist / embedding / reranker のいずれかについて、入力済みの新モデルパスで
	 * `POST /api/model/{component}/migrate` を実行する。LlamaProcessManager 管理下なら
	 * 自動再起動 + クライアント差し替えが行われる (バックエンドで rebind 失敗時は
	 * 旧モデルへ自動ロールバック)。
	 *
	 * ボタンは disabled / loading / 結果表示 / ロールバックボタンの 4 状態を持つ。
	 */
	import { t } from '$lib/i18n';
	import {
		migrateComponent,
		rollbackComponent,
		ApiError,
		type ModelComponent,
		type ComponentMigrateResponse
	} from '$lib/free/api';

	type Props = {
		component: ModelComponent;
		currentModel?: string;
		/** migrate/rollback 成功後に config を再取得させるコールバック */
		onMigrated?: () => void;
	};

	let { component, currentModel = '', onMigrated }: Props = $props();

	// 新モデルパスは migrate ボタン専用の入力 (config の TextField とは独立)。
	// これにより model_paths の追跡キーは config 直保存経路に乗らず desync しない。
	let newPath = $state('');
	let busy = $state(false);
	let result = $state<ComponentMigrateResponse | null>(null);
	let error = $state<string | null>(null);
	let canRollback = $derived(result !== null && !result.dry_run);

	async function apply() {
		if (!newPath || busy) return;
		busy = true;
		error = null;
		result = null;
		try {
			result = await migrateComponent(component, {
				new_model_path: newPath,
				auto_restart: true
			});
			onMigrated?.();
		} catch (e) {
			if (e instanceof ApiError) {
				error = e.message;
			} else if (e instanceof Error) {
				error = e.message;
			} else {
				error = String(e);
			}
		} finally {
			busy = false;
		}
	}

	async function doRollback() {
		if (busy) return;
		busy = true;
		error = null;
		try {
			const r = await rollbackComponent(component);
			result = {
				component,
				dry_run: false,
				old_model: result?.new_model ?? '',
				new_model: r.rolled_back_to,
				restarted: false,
				recommendations: [$t('settings.model_migrate.rollback_done')]
			};
			onMigrated?.();
		} catch (e) {
			if (e instanceof ApiError) {
				error = e.message;
			} else if (e instanceof Error) {
				error = e.message;
			} else {
				error = String(e);
			}
		} finally {
			busy = false;
		}
	}
</script>

<div class="component-migrate">
	<div class="current" data-testid="component-current">
		{$t('settings.model_migrate.current_model', { model: currentModel || '—' })}
	</div>
	<input
		type="text"
		class="path-input"
		placeholder={$t('settings.model_migrate.new_model_path')}
		bind:value={newPath}
		disabled={busy}
	/>
	<div class="actions">
		<button
			type="button"
			class="apply-btn"
			disabled={busy || !newPath}
			onclick={apply}
		>
			{busy ? $t('settings.model_migrate.applying') : $t('settings.model_migrate.apply')}
		</button>
		{#if canRollback}
			<button
				type="button"
				class="rollback-btn"
				disabled={busy}
				onclick={doRollback}
			>
				{$t('settings.model_migrate.rollback')}
			</button>
		{/if}
	</div>

	{#if result}
		<div class="result success">
			<div>
				<strong>{result.old_model}</strong> → <strong>{result.new_model}</strong>
				{#if result.restarted}
					<span class="badge restarted">{$t('settings.model_migrate.restarted')}</span>
				{:else}
					<span class="badge manual">{$t('settings.model_migrate.manual_restart_needed')}</span>
				{/if}
			</div>
			{#if result.recommendations.length > 0}
				<ul>
					{#each result.recommendations as rec}
						<li>{rec}</li>
					{/each}
				</ul>
			{/if}
		</div>
	{/if}
	{#if error}
		<div class="result error">
			{error}
		</div>
	{/if}
</div>

<style>
	.component-migrate {
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
	.actions {
		display: flex;
		gap: 0.5rem;
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
	.badge {
		display: inline-block;
		margin-left: 0.5rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		font-size: 0.75rem;
	}
	.badge.restarted {
		background: var(--accent-bg, #2563eb);
		color: var(--accent-text, #fff);
	}
	.badge.manual {
		background: var(--bg-warning-soft, #fef3c7);
		color: var(--text-warning, #92400e);
	}
	ul {
		margin: 0.25rem 0 0 1rem;
		padding: 0;
	}
</style>
