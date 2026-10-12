<script lang="ts">
	import { t } from '$lib/i18n';
	import FieldShell from './FieldShell.svelte';
	import { errorAria } from './fieldA11y';

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

	/** 入力中の値が範囲外のときのエラー (確定 (change) で範囲内へ丸めて消す) */
	let rangeError = $state('');
	let shownError = $derived(error || rangeError);

	function parse(text: string): number {
		const raw = parseFloat(text);
		return step === 1 ? Math.round(raw) : raw;
	}

	function inRange(v: number): boolean {
		return (min === undefined || v >= min) && (max === undefined || v <= max);
	}

	function clamp(v: number): number {
		if (min !== undefined && v < min) return min;
		if (max !== undefined && v > max) return max;
		return v;
	}

	function rangeMessage(): string {
		if (min !== undefined && max !== undefined) return $t('settings.field.range_between', { min, max });
		if (min !== undefined) return $t('settings.field.range_min', { min });
		return $t('settings.field.range_max', { max: max ?? '' });
	}

	/** 入力中: 範囲内の数値だけを反映する (空欄・範囲外は dirty にしない) */
	function handleInput(e: Event & { currentTarget: HTMLInputElement }) {
		const v = parse(e.currentTarget.value);
		if (isNaN(v)) {
			rangeError = '';
			return;
		}
		if (!inRange(v)) {
			rangeError = rangeMessage();
			return;
		}
		rangeError = '';
		onchange(v);
	}

	/** 確定: 空欄は元の値へ戻し、範囲外は範囲内へ丸めて反映する */
	function handleCommit(e: Event & { currentTarget: HTMLInputElement }) {
		const input = e.currentTarget;
		const v = parse(input.value);
		rangeError = '';
		if (isNaN(v)) {
			input.value = value === null ? '' : String(value);
			return;
		}
		if (!inRange(v)) {
			const clamped = clamp(v);
			input.value = String(clamped);
			onchange(clamped);
		}
	}
</script>

<FieldShell {label} {description} error={shownError} forId={fieldId}>
	<input
		id={fieldId}
		type="number"
		value={value ?? ''}
		placeholder={value === null ? autoLabel : undefined}
		{disabled}
		{min}
		{max}
		{step}
		oninput={handleInput}
		onchange={handleCommit}
		{...errorAria(fieldId, shownError)}
		class="field-input"
		class:field-error={!!shownError}
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
