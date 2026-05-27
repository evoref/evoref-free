<script lang="ts">
	import FieldShell from './FieldShell.svelte';

	interface Props {
		label: string;
		value: string[];
		description?: string;
		disabled?: boolean;
		error?: string;
		placeholder?: string;
		onchange: (value: string[]) => void;
	}

	let {
		label,
		value,
		description = '',
		disabled = false,
		error = '',
		placeholder = '',
		onchange
	}: Props = $props();

	let inputValue = $state('');

	function addTag() {
		const trimmed = inputValue.trim();
		if (trimmed && !value.includes(trimmed)) {
			onchange([...value, trimmed]);
			inputValue = '';
		}
	}

	function removeTag(index: number) {
		onchange(value.filter((_, i) => i !== index));
	}

	const fieldId = `tl-${Math.random().toString(36).slice(2, 10)}`;

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			addTag();
		}
	}
</script>

<FieldShell {label} {description} {error} forId={fieldId}>
	<div class="tags-container">
		{#each value as tag, i}
			<span class="tag">
				{tag}
				{#if !disabled}
					<button type="button" class="tag-remove" onclick={() => removeTag(i)}>x</button>
				{/if}
			</span>
		{/each}
	</div>
	{#if !disabled}
		<div class="tag-input-row">
			<input
				id={fieldId}
				type="text"
				bind:value={inputValue}
				{placeholder}
				onkeydown={handleKeydown}
				class="field-input"
				class:field-error={!!error}
			/>
			<button type="button" class="tag-add-btn" onclick={addTag}>+</button>
		</div>
	{/if}
</FieldShell>

<style>
	.tags-container {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.tag {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 2px 8px;
		background: var(--control-bg);
		border: 0.5px solid var(--border);
		border-radius: 12px;
		font-size: 12px;
		color: var(--text-primary);
	}
	.tag-remove {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		font-size: 11px;
		padding: 0 2px;
		line-height: 1;
	}
	.tag-remove:hover {
		color: var(--color-error);
	}
	.tag-input-row {
		display: flex;
		gap: 4px;
	}
	.field-input {
		flex: 1;
		max-width: 300px;
		padding: 4px 8px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
	}
	.field-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	.field-input.field-error {
		border-color: var(--color-error);
	}
	.tag-add-btn {
		padding: 4px 10px;
		background: var(--control-bg);
		color: var(--text-secondary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		cursor: pointer;
		font-size: 14px;
	}
</style>
