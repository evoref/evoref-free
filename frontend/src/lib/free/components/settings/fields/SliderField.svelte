<script lang="ts">
	import FieldShell from './FieldShell.svelte';

	interface Props {
		label: string;
		value: number;
		min?: number;
		max?: number;
		step?: number;
		description?: string;
		disabled?: boolean;
		error?: string;
		onchange: (value: number) => void;
	}

	let {
		label,
		value,
		min = 0,
		max = 1,
		step = 0.01,
		description = '',
		disabled = false,
		error = '',
		onchange
	}: Props = $props();

	const fieldId = `sl-${Math.random().toString(36).slice(2, 10)}`;

	function handleInput(e: Event) {
		const v = parseFloat((e.currentTarget as HTMLInputElement).value);
		if (!isNaN(v)) onchange(v);
	}
</script>

<FieldShell {label} {description} {error} forId={fieldId}>
	<div class="slider-row">
		<input
			id={fieldId}
			type="range"
			{value}
			{min}
			{max}
			{step}
			{disabled}
			oninput={handleInput}
			class="slider-input"
		/>
		<input
			type="number"
			{value}
			{min}
			{max}
			{step}
			{disabled}
			oninput={handleInput}
			class="number-input"
			class:field-error={!!error}
		/>
	</div>
</FieldShell>

<style>
	.slider-row {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.slider-input {
		flex: 1;
		max-width: 240px;
		height: 4px;
		accent-color: var(--accent);
		cursor: pointer;
	}
	.slider-input:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.number-input {
		width: 72px;
		padding: 4px 8px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
		text-align: right;
	}
	.number-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	.number-input:disabled {
		opacity: 0.5;
	}
	.number-input.field-error {
		border-color: var(--color-error);
	}
</style>
