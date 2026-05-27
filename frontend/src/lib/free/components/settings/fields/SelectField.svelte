<script lang="ts">
	import { t } from '$lib/i18n';
	import FieldShell from './FieldShell.svelte';

	interface Props {
		label: string;
		value: string | boolean | null;
		options: { value: string | boolean | null; label: string; i18nLabel?: string }[];
		description?: string;
		disabled?: boolean;
		error?: string;
		onchange: (value: string | boolean | null) => void;
	}

	let {
		label,
		value,
		options,
		description = '',
		disabled = false,
		error = '',
		onchange
	}: Props = $props();

	const fieldId = `sf-${Math.random().toString(36).slice(2, 10)}`;

	function handleChange(e: Event) {
		const raw = (e.currentTarget as HTMLSelectElement).value;
		if (raw === 'true') onchange(true);
		else if (raw === 'false') onchange(false);
		else if (raw === 'null') onchange(null);
		else onchange(raw);
	}
</script>

<FieldShell {label} {description} {error} forId={fieldId}>
	<select
		id={fieldId}
		value={String(value)}
		{disabled}
		onchange={handleChange}
		class="field-select"
		class:field-error={!!error}
	>
		{#each options as opt}
			<option value={String(opt.value)}>{opt.i18nLabel ? $t(opt.i18nLabel) : opt.label}</option>
		{/each}
	</select>
</FieldShell>

<style>
	.field-select {
		max-width: 250px;
		padding: 6px 10px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
	}
	.field-select:focus {
		outline: none;
		border-color: var(--accent);
	}
	.field-select:disabled {
		opacity: 0.5;
	}
	.field-select.field-error {
		border-color: var(--color-error);
	}
</style>
