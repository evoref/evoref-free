<script lang="ts">
	/**
	 * llama-server の起動オプション (launch_args) の入力欄 (c_16 §7.2.4)
	 *
	 * モデル画面の各カードに置く。「保存」でその場で config.yaml へ書く (タブの「適用」には乗らない)。
	 * 反映は該当サーバーの再起動から。書けないフラグ (ファイル・ネットワーク) はバックエンドが 403 で返し、
	 * そのメッセージを欄の下に出す。
	 */
	import { t } from '$lib/i18n';
	import { saveLaunchArgs, type LaunchArgsTarget } from '$lib/free/stores/settings';
	import { ApiError } from '$lib/free/api';

	type Props = { target: LaunchArgsTarget; value: string };
	let { target, value }: Props = $props();

	const fieldId = `la-${Math.random().toString(36).slice(2, 10)}`;

	/** 入力中の値 (null = 編集していない → 保存済みの値を出す) */
	let edited = $state<string | null>(null);
	/** この欄から保存した値 (設定の再読込を待たずに基準にする) */
	let lastSaved = $state<string | null>(null);
	let busy = $state(false);
	let error = $state<string | null>(null);
	let justSaved = $state(false);

	let current = $derived(lastSaved ?? value);
	let draft = $derived(edited ?? current);
	let changed = $derived(draft.trim() !== current.trim());

	/** API の拒否は i18n キーで訳す (訳が無ければ英語のメッセージのまま) */
	function toMessage(e: unknown): string {
		if (e instanceof ApiError && e.i18nKey) {
			const params = Object.fromEntries(
				Object.entries(e.context ?? {}).map(([k, v]) => [k, String(v)])
			);
			const text = $t(e.i18nKey, params);
			if (text !== e.i18nKey) return text;
		}
		return e instanceof Error ? e.message : String(e);
	}

	async function save() {
		if (!changed || busy) return;
		busy = true;
		error = null;
		justSaved = false;
		const next = draft.trim();
		try {
			await saveLaunchArgs(target, next);
			lastSaved = next;
			edited = null;
			justSaved = true;
		} catch (e) {
			error = toMessage(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="launch-args">
	<label class="la-label" for={fieldId}>{$t('settings.model_card.launch_args')}</label>
	<textarea
		id={fieldId}
		class="la-input"
		class:la-error={!!error}
		rows="3"
		spellcheck="false"
		autocomplete="off"
		placeholder={$t('settings.model_card.launch_args_placeholder')}
		value={draft}
		disabled={busy}
		oninput={(e) => {
			edited = e.currentTarget.value;
			justSaved = false;
		}}
	></textarea>
	<span class="la-desc">{$t('settings.model_card.launch_args_desc')}</span>
	<div class="la-actions">
		<button type="button" class="la-save" disabled={!changed || busy} onclick={save}>
			{busy ? $t('settings.applying') : $t('settings.model_card.save')}
		</button>
		{#if justSaved}
			<span class="la-note" data-testid="launch-args-saved">{$t('settings.model_card.saved_restart')}</span>
		{/if}
	</div>
	{#if error}
		<span class="la-error-msg" data-testid="launch-args-error">{error}</span>
	{/if}
</div>

<style>
	.launch-args {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.la-label {
		font-size: 13px;
		color: var(--text-secondary);
		font-weight: 500;
	}
	.la-input {
		width: 100%;
		box-sizing: border-box;
		padding: 6px 10px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 12.5px;
		font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
		line-height: 1.5;
		resize: vertical;
	}
	.la-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	.la-input:disabled {
		opacity: 0.5;
	}
	.la-input.la-error {
		border-color: var(--color-error);
	}
	.la-desc {
		font-size: 11px;
		color: var(--text-muted);
	}
	.la-actions {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-top: 2px;
	}
	.la-save {
		padding: 5px 14px;
		background: none;
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 12.5px;
		font-family: inherit;
		cursor: pointer;
	}
	.la-save:hover:not(:disabled) {
		border-color: var(--accent);
	}
	.la-save:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.la-note {
		font-size: 11px;
		color: var(--text-muted);
	}
	.la-error-msg {
		font-size: 11px;
		color: var(--color-error);
	}
</style>
