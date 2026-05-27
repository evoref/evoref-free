<script lang="ts">
	import { t } from '$lib/i18n';
	import FieldShell from './FieldShell.svelte';

	interface Props {
		label: string;
		value: string;
		description?: string;
		disabled?: boolean;
		error?: string;
		type?: 'text' | 'password';
		placeholder?: string;
		onchange: (value: string) => void;
	}

	let {
		label,
		value,
		description = '',
		disabled = false,
		error = '',
		type = 'text',
		placeholder = '',
		onchange
	}: Props = $props();

	let showPassword = $state(false);
	let inputType = $derived(type === 'password' && showPassword ? 'text' : type);

	const fieldId = `tf-${Math.random().toString(36).slice(2, 10)}`;
</script>

<FieldShell {label} {description} {error} forId={fieldId}>
	<div class="field-input-row">
		<input
			id={fieldId}
			type={inputType}
			{value}
			{disabled}
			{placeholder}
			oninput={(e) => onchange(e.currentTarget.value)}
			class="field-input"
			class:field-error={!!error}
		/>
		{#if type === 'password'}
			<button
				type="button"
				class="password-toggle"
				onclick={() => (showPassword = !showPassword)}
			>
				{showPassword ? $t('settings.hide') : $t('settings.show')}
			</button>
		{/if}
	</div>
</FieldShell>

<style>
	.field-input-row {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.field-input {
		flex: 1;
		padding: 6px 10px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
	}
	.field-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	.field-input:disabled {
		opacity: 0.5;
	}
	.field-input.field-error {
		border-color: var(--color-error);
	}
	.password-toggle {
		padding: 6px 10px;
		background: var(--control-bg);
		color: var(--text-secondary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 12px;
		cursor: pointer;
	}
</style>
