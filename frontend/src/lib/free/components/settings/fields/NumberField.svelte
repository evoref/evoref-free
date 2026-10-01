<script lang="ts">
	import FieldShell from './FieldShell.svelte';

	interface Props {
		label: string;
		/** null は自動 (環境調整が決める値)。欄を空にして `autoLabel` を出す */
		value: number | null;
		autoLabel?: string;
		description?: string;
		disabled?: boolean;
		error?: string;
		min?: number;
		max?: number;
		step?: number;
		onchange: (value: number) => void;
	}

	let {
		label,
		value,
		autoLabel = 'auto',
		description = '',
		disabled = false,
		error = '',
		min = undefined,
		max = undefined,
		step = 1,
		onchange
	}: Props = $props();

	const fieldId = `nf-${Math.random().toString(36).slice(2, 10)}`;
</script>

<FieldShell {label} {description} {error} forId={fieldId}>
	<input
		id={fieldId}
		type="number"
		value={value ?? ''}
		placeholder={value === null ? autoLabel : undefined}
		{disabled}
		{min}
		{max}
		{step}
		oninput={(e) => {
			const raw = parseFloat(e.currentTarget.value);
			const v = step === 1 ? Math.round(raw) : raw;
			if (!isNaN(v)) onchange(v);
		}}
		class="field-input"
		class:field-error={!!error}
	/>
</FieldShell>

<style>
	.field-input {
		max-width: 200px;
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
</style>
