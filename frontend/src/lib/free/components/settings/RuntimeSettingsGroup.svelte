<script lang="ts">
	/**
	 * create の実行環境 (`create.runtimes.php` / `node`) のパス設定 (f_10 §12.4)
	 *
	 * 能力キーなのでタブ共通の「適用」(汎用 `PUT /config/<section>`) では書けない (403)。
	 * 実行環境ごとの「確認」ボタンで専用 API (`PUT /config/runtimes/{name}`) を呼び、
	 * サーバ側の検証 (絶対パス / 実在 / .exe / ファイル名 / 保護領域の外) を通ったものだけ保存する。
	 */
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n';
	import { getRuntimes, setRuntimePath, ApiError, type RuntimeInfo } from '$lib/free/api';
	import FieldGroup from './fields/FieldGroup.svelte';

	let items = $state<RuntimeInfo[]>([]);
	let drafts = $state<Record<string, string>>({});
	let busy = $state<Record<string, boolean>>({});
	let saveErrors = $state<Record<string, string>>({});
	let loadError = $state('');

	function applyItem(item: RuntimeInfo) {
		const i = items.findIndex((x) => x.name === item.name);
		if (i >= 0) items[i] = item;
		else items.push(item);
		drafts[item.name] = item.configured;
	}

	function errorText(e: unknown): string {
		if (e instanceof ApiError && e.i18nKey) {
			return $t(e.i18nKey, e.context as Record<string, string | number>);
		}
		return e instanceof Error ? e.message : String(e);
	}

	async function load() {
		loadError = '';
		try {
			const list = await getRuntimes();
			for (const item of list) applyItem(item);
		} catch (e) {
			loadError = errorText(e);
		}
	}

	async function save(name: string) {
		if (busy[name]) return;
		busy[name] = true;
		saveErrors[name] = '';
		try {
			applyItem(await setRuntimePath(name, drafts[name] ?? ''));
		} catch (e) {
			saveErrors[name] = errorText(e);
		} finally {
			busy[name] = false;
		}
	}

	function statusText(item: RuntimeInfo): string {
		if (item.resolved) {
			return $t('settings.runtimes.resolved', {
				path: item.resolved,
				source: $t(`settings.runtimes.source_${item.source}`),
				version: item.version || $t('settings.runtimes.version_unknown')
			});
		}
		if (item.error) {
			return $t(`api.runtime_path_invalid.${item.error}`, { name: item.name });
		}
		return $t('settings.runtimes.not_found');
	}

	onMount(load);
</script>

<FieldGroup label="settings.group_runtimes" description="settings.runtimes.desc">
	{#if loadError}
		<p class="runtime-error" role="alert">{loadError}</p>
	{/if}
	{#each items as item (item.name)}
		{@const inputId = `runtime-${item.name}`}
		<div class="runtime" data-testid={`runtime-${item.name}`}>
			<label class="runtime-label" for={inputId}>{$t(`settings.runtimes.${item.name}`)}</label>
			<div class="runtime-row">
				<input
					id={inputId}
					type="text"
					class="runtime-input"
					class:has-error={!!saveErrors[item.name]}
					value={drafts[item.name] ?? ''}
					placeholder={$t('settings.runtimes.placeholder')}
					disabled={busy[item.name]}
					oninput={(e) => (drafts[item.name] = e.currentTarget.value)}
				/>
				<button
					type="button"
					class="runtime-apply"
					disabled={busy[item.name]}
					onclick={() => save(item.name)}
				>
					{busy[item.name] ? $t('settings.runtimes.checking') : $t('settings.runtimes.apply')}
				</button>
			</div>
			<span class="runtime-status" class:warn={!item.resolved && !!item.error}>{statusText(item)}</span>
			{#if saveErrors[item.name]}
				<span class="runtime-error" role="alert">{saveErrors[item.name]}</span>
			{/if}
		</div>
	{/each}
</FieldGroup>

<style>
	.runtime {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 6px 0;
	}
	.runtime-label {
		font-size: 13px;
		color: var(--text-secondary);
		font-weight: 500;
	}
	.runtime-row {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.runtime-input {
		flex: 1;
		min-width: 0;
		padding: 6px 10px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
	}
	.runtime-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	.runtime-input.has-error {
		border-color: var(--color-error);
	}
	.runtime-apply {
		padding: 6px 12px;
		background: var(--control-bg);
		color: var(--text-secondary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 12px;
		cursor: pointer;
	}
	.runtime-apply:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.runtime-status {
		font-size: 11px;
		color: var(--text-muted);
		word-break: break-all;
	}
	.runtime-status.warn,
	.runtime-error {
		font-size: 11px;
		color: var(--color-error);
		margin: 0;
	}
</style>
