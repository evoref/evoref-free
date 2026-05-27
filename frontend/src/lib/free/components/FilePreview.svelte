<script lang="ts">
	import { attachedFiles, removeFile } from '$lib/free/stores/chat';
</script>

{#if $attachedFiles.length > 0}
	<div class="file-preview">
		{#each $attachedFiles as file, i}
			<span class="file-badge">
				{file.name}
				<button class="remove-btn" onclick={() => removeFile(i)} aria-label="Remove {file.name}">
					&times;
				</button>
			</span>
		{/each}
	</div>
{/if}

<style>
	.file-preview {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		padding: 4px 12px;
	}
	.file-badge {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 2px 8px;
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		font-size: 0.9rem;
		color: var(--text-primary);
	}
	.remove-btn {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 1rem;
		padding: 0 2px;
		line-height: 1;
	}
	.remove-btn:hover {
		color: var(--color-error);
	}
</style>
