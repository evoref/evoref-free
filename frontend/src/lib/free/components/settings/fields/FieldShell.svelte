<script lang="ts">
	/**
	 * 設定フィールドの共通シェル
	 *
	 * NumberField / TextField / SelectField / SliderField / TagListField / ToggleField で
	 * 重複していた `<div class="field">` ラッパ + label + description + error 行を集約する。
	 *
	 * サブコンポーネント (実際のコントロール) は `children` snippet で渡す。
	 *
	 * @param label - i18n キー
	 * @param description - 補足説明 (i18n キー、空文字なら非表示)
	 * @param error - エラーメッセージ (空文字なら非表示)
	 * @param forId - <label for> に渡す ID。指定なしなら <span> でレンダリングする
	 *                (ToggleField のように label が input と紐づかないケース用)
	 */
	import type { Snippet } from 'svelte';
	import { t } from '$lib/i18n';

	interface Props {
		label: string;
		description?: string;
		error?: string;
		forId?: string;
		children: Snippet;
	}

	let { label, description = '', error = '', forId = '', children }: Props = $props();
</script>

<div class="field">
	{#if forId}
		<label class="field-label" for={forId}>{$t(label)}</label>
	{:else}
		<span class="field-label">{$t(label)}</span>
	{/if}
	{@render children()}
	{#if description}
		<span class="field-desc">{$t(description)}</span>
	{/if}
	{#if error}
		<span class="field-error-msg">{error}</span>
	{/if}
</div>

<style>
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 6px 0;
	}
	.field-label {
		font-size: 13px;
		color: var(--text-secondary);
		font-weight: 500;
	}
	.field-desc {
		font-size: 11px;
		color: var(--text-muted);
	}
	.field-error-msg {
		font-size: 11px;
		color: var(--color-error);
	}
</style>
