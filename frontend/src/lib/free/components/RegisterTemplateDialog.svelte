<script lang="ts">
	import { t, availableLocales } from '$lib/i18n';
	import { registerTemplate } from '$lib/free/api';
	import DialogShell from './DialogShell.svelte';

	interface Props {
		onClose: () => void;
		onRegistered: () => void;
	}

	let { onClose, onRegistered }: Props = $props();

	let file = $state<File | null>(null);
	let docType = $state('');
	let aliasesText = $state('');
	let lang = $state('ja');
	let registering = $state(false);
	let errorMessage = $state('');

	let canSubmit = $derived(file !== null && docType.trim().length > 0 && !registering);

	function handleFileSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		file = input.files?.[0] ?? null;
	}

	async function handleSubmit() {
		if (!file || !docType.trim() || registering) return;
		registering = true;
		errorMessage = '';
		const aliases = aliasesText
			.split(',')
			.map((a) => a.trim())
			.filter((a) => a.length > 0);
		try {
			await registerTemplate(file, docType.trim(), aliases, lang);
			onRegistered();
			onClose();
		} catch (e) {
			errorMessage = e instanceof Error ? e.message : $t('cartridge.template_register_failed');
		} finally {
			registering = false;
		}
	}
</script>

<DialogShell
	ariaLabel={$t('cartridge.template_register_title')}
	{onClose}
	minWidth="360px"
	maxWidth="480px"
	canCloseOnOverlayClick={!registering}
>
	<h3 class="dialog-title">{$t('cartridge.template_register_title')}</h3>

	{#if errorMessage}
		<p class="error">{errorMessage}</p>
	{/if}

	<div class="form-grid">
		<label class="field">
			<span class="field-label">{$t('cartridge.template_file')}</span>
			<input
				type="file"
				accept=".docx,.pptx,.xlsx,.dotx,.potx,.xltx"
				onchange={handleFileSelect}
				disabled={registering}
			/>
		</label>
		<label class="field">
			<span class="field-label">{$t('cartridge.template_doc_type')}</span>
			<input
				type="text"
				bind:value={docType}
				placeholder={$t('cartridge.template_doc_type_placeholder')}
				disabled={registering}
			/>
		</label>
		<label class="field">
			<span class="field-label">{$t('cartridge.template_aliases')}</span>
			<input
				type="text"
				bind:value={aliasesText}
				placeholder={$t('cartridge.template_aliases_hint')}
				disabled={registering}
			/>
		</label>
		<label class="field">
			<span class="field-label">{$t('cartridge.template_lang')}</span>
			<select bind:value={lang} disabled={registering}>
				{#each availableLocales as loc}
					<option value={loc}>{loc.toUpperCase()}</option>
				{/each}
			</select>
		</label>
	</div>

	<div class="dialog-actions">
		<button class="btn btn-close" onclick={onClose} disabled={registering}>
			{$t('cartridge.close')}
		</button>
		<button class="btn btn-primary" onclick={handleSubmit} disabled={!canSubmit}>
			{registering ? $t('cartridge.template_registering') : $t('cartridge.register')}
		</button>
	</div>
</DialogShell>

<style>
	.dialog-title {
		margin: 0 0 12px;
		font-size: 1.1rem;
		color: var(--text-primary);
	}
	.error {
		color: var(--color-error);
		font-size: 0.9rem;
		margin: 0 0 12px;
	}
	.form-grid {
		display: flex;
		flex-direction: column;
		gap: 12px;
		margin-bottom: 16px;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.field-label {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}
	.field input[type='text'],
	.field select {
		padding: 6px 8px;
		border: 0.5px solid var(--input-border);
		border-radius: var(--border-radius);
		background: var(--input-bg);
		color: var(--text-primary);
		font-size: 0.95rem;
	}
	.dialog-actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
	}
	.btn {
		padding: 8px 16px;
		border-radius: var(--border-radius);
		font-size: 0.95rem;
		cursor: pointer;
		border: none;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn-close {
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}
	.btn-primary {
		background-color: var(--accent);
		color: var(--text-on-accent);
	}
</style>
