<script lang="ts">
	import { t } from '$lib/i18n';
	import { attachedFiles, addFile } from '$lib/free/stores/chat';
	import { addToast } from '$lib/free/stores/toast';
	import { get } from 'svelte/store';
	import { FILE_MAX_SIZE_BYTES, FILE_MAX_SIZE_MB } from '$lib/free/constants';

	let dragOver = $state(false);
	let fileInput: HTMLInputElement | undefined = $state();

	function validateAndAdd(file: File): void {
		if (file.size > FILE_MAX_SIZE_BYTES) {
			addToast({ type: 'warning', i18nKey: 'file.size_exceeded', params: { name: file.name, limit: FILE_MAX_SIZE_MB } });
			return;
		}
		const existing = get(attachedFiles);
		if (existing.some((f) => f.name === file.name)) {
			addToast({ type: 'warning', i18nKey: 'file.duplicate', params: { name: file.name } });
			return;
		}
		addFile(file);
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		if (e.dataTransfer?.files) {
			for (const file of e.dataTransfer.files) {
				validateAndAdd(file);
			}
		}
	}

	function handleFileSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files) {
			for (const file of input.files) {
				validateAndAdd(file);
			}
			input.value = '';
		}
	}
</script>

<div
	class="file-upload"
	class:drag-over={dragOver}
	role="button"
	tabindex="0"
	aria-label={$t('file.upload_area')}
	ondragover={(e) => {
		e.preventDefault();
		dragOver = true;
	}}
	ondragleave={() => (dragOver = false)}
	ondrop={handleDrop}
	onclick={() => fileInput?.click()}
	onkeydown={(e) => {
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			fileInput?.click();
		}
	}}
>
	<input
		bind:this={fileInput}
		type="file"
		multiple
		class="hidden"
		onchange={handleFileSelect}
	/>
	<svg
		class="upload-icon"
		xmlns="http://www.w3.org/2000/svg"
		width="18"
		height="18"
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		stroke-width="2"
		stroke-linecap="round"
		stroke-linejoin="round"
		aria-hidden="true"
	>
		<title>{$t('file.attach')}</title>
		<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
	</svg>
</div>

<style>
	.file-upload {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 6px;
		border: 1px dashed var(--border);
		border-radius: var(--border-radius);
		cursor: pointer;
		color: var(--text-secondary);
		transition: border-color 0.2s, background-color 0.2s;
	}
	.file-upload:hover,
	.file-upload.drag-over {
		border-color: var(--accent);
		background-color: var(--bg-secondary);
		color: var(--accent);
	}
	.upload-icon {
		display: block;
		flex-shrink: 0;
	}
	.hidden {
		display: none;
	}
</style>
