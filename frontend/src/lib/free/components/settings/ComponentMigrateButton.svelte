<script lang="ts">
	/**
	 * モデル切替ボタン (全モデル種別共通 I/F)
	 *
	 * base / assist / embedding / coding のいずれかについて、入力済みの新モデルパスで
	 * 切替を実行する。見た目・操作系は共通だが、種別ごとに正しいバックエンドへ振り分ける:
	 *   - assist / embedding: `POST /api/model/{component}/migrate` (auto_restart で自動再起動。
	 *     rebind 失敗時はバックエンドが旧モデルへ自動ロールバック)。
	 *   - base: `POST /api/model/migrate` (LoRA アーカイブ等の付帯処理あり)。auto_restart 相当は
	 *     無く、再起動は別途手動で行うため結果に「手動で再起動してください」バッジを出す。
	 *   - coding: migrate API 無し (model_state 非追跡)。`onApply` 経由で config を即保存する。
	 *
	 * UI は apply ボタン (disabled / loading) → 結果表示 (成功 / エラー) の一発適用で、
	 * ロールバックボタン・再起動ボタンは持たない (#198 で UI から廃止)。再起動状態バッジは
	 * 非 coding のみ表示 (assist/embedding=自動再起動済 / base=手動再起動が必要)。
	 */
	import type { Snippet } from 'svelte';
	import { t } from '$lib/i18n';
	import {
		migrateComponent,
		migrateBaseModel,
		ApiError,
		type ModelComponent
	} from '$lib/free/api';

	/** UI レベルの種別。API の ModelComponent (assist/embedding) を base/coding へ拡張 */
	type ModelKind = ModelComponent | 'base' | 'coding';

	/** base/assist/embedding/coding のレスポンス差を吸収した共通結果形 */
	type MigrateResult = {
		old_model: string;
		new_model: string;
		restarted: boolean;
		recommendations: string[];
	};

	type Props = {
		component: ModelKind;
		currentModel?: string;
		/** migrate/rollback 成功後に config を再取得させるコールバック */
		onMigrated?: () => void;
		/** coding 切替時の即時保存処理 (migrate API を持たない種別用に親が注入) */
		onApply?: (newPath: string) => Promise<void>;
		/** apply ボタンと同じ行の右隣に並べる追加アクション (例: reindex ボタン) */
		actionsTrailing?: Snippet;
	};

	let { component, currentModel = '', onMigrated, onApply, actionsTrailing }: Props = $props();

	// 新モデルパスは migrate ボタン専用の入力 (config の TextField とは独立)。
	// これにより model_paths の追跡キーは config 直保存経路に乗らず desync しない。
	let newPath = $state('');
	let busy = $state(false);
	let result = $state<MigrateResult | null>(null);
	let error = $state<string | null>(null);

	// coding は config 保存のため、再起動バッジを持たない。
	let showRestartBadge = $derived(component !== 'coding');

	function toError(e: unknown): string {
		if (e instanceof ApiError) return e.message;
		if (e instanceof Error) return e.message;
		return String(e);
	}

	async function apply() {
		if (!newPath || busy) return;
		busy = true;
		error = null;
		result = null;
		try {
			if (component === 'base') {
				const r = await migrateBaseModel({ new_model_path: newPath, dry_run: false });
				result = {
					old_model: r.old_model,
					new_model: r.new_model,
					restarted: false,
					recommendations: r.recommendations
				};
			} else if (component === 'coding') {
				await onApply?.(newPath);
				result = {
					old_model: currentModel || '—',
					new_model: newPath,
					restarted: false,
					recommendations: []
				};
			} else {
				const r = await migrateComponent(component, {
					new_model_path: newPath,
					auto_restart: true
				});
				result = {
					old_model: r.old_model,
					new_model: r.new_model,
					restarted: r.restarted,
					recommendations: r.recommendations
				};
			}
			onMigrated?.();
		} catch (e) {
			error = toError(e);
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
		{@render actionsTrailing?.()}
	</div>

	{#if result}
		<div class="result success">
			<div>
				<strong>{result.old_model}</strong> → <strong>{result.new_model}</strong>
				{#if showRestartBadge}
					{#if result.restarted}
						<span class="badge restarted">{$t('settings.model_migrate.restarted')}</span>
					{:else}
						<span class="badge manual">{$t('settings.model_migrate.manual_restart_needed')}</span>
					{/if}
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
		flex-wrap: wrap;
		align-items: flex-start;
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
