<script lang="ts">
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n';
	import { isPro } from '$lib/edition';
	import {
		systemPrompts,
		auxPrompts,
		selectedPrompt,
		editContent,
		currentDetail,
		promptHistory,
		promptsLoading,
		promptSaving,
		promptDirty,
		historyOpen,
		loadPromptList,
		selectPrompt,
		updateEditContent,
		savePrompt,
		reloadFromDisk,
		loadHistory,
		executeRollback,
		resetEdit
	} from '$lib/free/stores/prompts';
	import { formatDate } from '$lib/free/utils/format';

	/** ソース表示ラベル */
	function sourceLabel(source: string): string {
		const key = `settings.prompts.source_${source}`;
		return $t(key);
	}

	/** プロンプト切替前の未保存チェック */
	async function handleSelect(category: 'system' | 'aux', id: string): Promise<void> {
		if ($promptDirty) {
			const ok = confirm($t('settings.prompts.unsaved_changes'));
			if (!ok) return;
		}
		await selectPrompt({ category, id });
	}

	/** ロールバック実行 */
	async function handleRollback(version: number): Promise<void> {
		const ok = confirm($t('settings.prompts.rollback_confirm', { version }));
		if (!ok) return;
		await executeRollback(version);
	}

	onMount(() => {
		loadPromptList();
	});
</script>

<div class="prompt-settings">
	<!-- 左: プロンプト選択リスト -->
	<div class="prompt-sidebar">
		{#each $systemPrompts as p}
			{#if isPro || p.mode !== 'create'}
				<button
					type="button"
					class="prompt-item"
					class:prompt-item-active={$selectedPrompt?.category === 'system' && $selectedPrompt?.id === p.mode}
					onclick={() => handleSelect('system', p.mode)}
				>
					<span class="prompt-name">{$t(`settings.prompts.mode_${p.mode}`)}</span>
				</button>
			{/if}
		{/each}
		{#each $auxPrompts as p}
			<button
				type="button"
				class="prompt-item"
				class:prompt-item-active={$selectedPrompt?.category === 'aux' && $selectedPrompt?.id === p.task}
				onclick={() => handleSelect('aux', p.task)}
			>
				<span class="prompt-name">{$t(`settings.prompts.task_${p.task}`)}</span>
			</button>
		{/each}
	</div>

	<!-- 右: 編集エリア -->
	<div class="prompt-editor">
		{#if $selectedPrompt && $currentDetail}
			<!-- メタ情報バー -->
			<div class="meta-bar">
				<div class="meta-items">
					<span class="meta-item">
						<span class="meta-label">{$t('settings.prompts.version')}</span>
						<span class="meta-value">v{$currentDetail.version}</span>
					</span>
					<span class="meta-item">
						<span class="meta-label">{$t('settings.prompts.source')}</span>
						<span class="meta-value source-badge source-{$currentDetail.source}">{sourceLabel($currentDetail.source)}</span>
					</span>
					<span class="meta-item">
						<span class="meta-label">{$t('settings.prompts.updated_at')}</span>
						<span class="meta-value">{formatDate($currentDetail.updated_at)}</span>
					</span>
					{#if $currentDetail.fitness_score !== undefined && $currentDetail.fitness_score > 0}
						<span class="meta-item">
							<span class="meta-label">{$t('settings.prompts.fitness_score')}</span>
							<span class="meta-value">{$currentDetail.fitness_score.toFixed(3)}</span>
						</span>
					{/if}
				</div>
			</div>

			<!-- 保護セクション注意書き -->
			{#if $editContent.includes('<!-- PROTECTED -->')}
				<div class="protected-notice">
					{$t('settings.prompts.protected_warning')}
				</div>
			{/if}

			<!-- テキストエリア -->
			<textarea
				class="prompt-textarea"
				value={$editContent}
				oninput={(e) => updateEditContent(e.currentTarget.value)}
				spellcheck="false"
			></textarea>

			<!-- アクションバー -->
			<div class="action-bar">
				<div class="action-left">
					<button
						type="button"
						class="btn-secondary"
						onclick={() => loadHistory()}
						disabled={$promptSaving}
					>
						{$t('settings.prompts.history')}
					</button>
					{#if $selectedPrompt.category === 'system'}
						<button
							type="button"
							class="btn-secondary"
							onclick={() => reloadFromDisk()}
							disabled={$promptSaving}
						>
							{$t('settings.prompts.reload')}
						</button>
					{/if}
				</div>
				<div class="action-right">
					<button
						type="button"
						class="btn-reset"
						onclick={() => resetEdit()}
						disabled={!$promptDirty || $promptSaving}
					>
						{$t('settings.reset')}
					</button>
					<button
						type="button"
						class="btn-save"
						onclick={() => savePrompt()}
						disabled={!$promptDirty || $promptSaving}
					>
						{#if $promptSaving}
							{$t('settings.prompts.saving')}
						{:else}
							{$t('settings.prompts.save')}
						{/if}
					</button>
				</div>
			</div>

			<!-- 履歴パネル -->
			{#if $historyOpen}
				<div class="history-panel">
					<div class="history-header">
						<h4>{$t('settings.group_prompt_history')}</h4>
						<button type="button" class="btn-close" onclick={() => historyOpen.set(false)}>
							&times;
						</button>
					</div>
					{#if $promptHistory.length === 0}
						<p class="history-empty">{$t('settings.prompts.no_history')}</p>
					{:else}
						<div class="history-list">
							{#each $promptHistory as h}
								<div class="history-item">
									<span class="history-version">v{h.version}</span>
									<span class="history-file">{h.file}</span>
									<button
										type="button"
										class="btn-rollback"
										onclick={() => handleRollback(h.version)}
										disabled={$promptSaving}
									>
										{$t('settings.prompts.rollback')}
									</button>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		{:else if $promptsLoading}
			<div class="prompt-placeholder">
				<p>{$t('settings.loading')}</p>
			</div>
		{:else}
			<div class="prompt-placeholder">
				<p>{$t('settings.prompts.select_prompt')}</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.prompt-settings {
		display: grid;
		grid-template-columns: 200px 1fr;
		height: 100%;
		overflow: hidden;
	}

	/* サイドバー */
	.prompt-sidebar {
		border-right: 0.5px solid var(--border);
		background: var(--bg-secondary);
		overflow-y: auto;
		padding: 12px 8px;
	}

.prompt-item {
		display: flex;
		align-items: center;
		gap: 6px;
		width: 100%;
		padding: 8px 12px;
		background: transparent;
		border: none;
		border-radius: 6px;
		color: var(--text-secondary);
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
		text-align: left;
		transition: background 0.15s;
	}

	.prompt-item:hover {
		background: color-mix(in srgb, var(--accent) 6%, transparent);
	}

	.prompt-item-active {
		color: var(--text-primary);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		font-weight: 500;
	}

	:global([data-theme='dark']) .prompt-item-active {
		background: color-mix(in srgb, var(--accent) 22%, transparent);
	}

	.prompt-name {
		flex: 1;
	}

	/* エディタ部 */
	.prompt-editor {
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.meta-bar {
		padding: 10px 16px;
		border-bottom: 0.5px solid var(--border);
		background: var(--bg-secondary);
	}

	.meta-items {
		display: flex;
		flex-wrap: wrap;
		gap: 16px;
	}

	.meta-item {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 12px;
	}

	.meta-label {
		color: var(--text-muted);
	}

	.meta-value {
		color: var(--text-primary);
		font-weight: 500;
	}

	.source-badge {
		padding: 1px 6px;
		border-radius: 3px;
		font-size: 11px;
	}

	.source-default {
		background: color-mix(in srgb, var(--text-muted) 15%, transparent);
	}

	.source-manual {
		background: color-mix(in srgb, var(--accent) 20%, transparent);
		color: var(--accent);
	}

	.source-evolution {
		background: color-mix(in srgb, var(--color-success) 20%, transparent);
		color: var(--color-success);
	}

	.protected-notice {
		padding: 6px 16px;
		font-size: 12px;
		color: var(--color-warning);
		background: color-mix(in srgb, var(--color-warning) 10%, transparent);
		border-bottom: 0.5px solid color-mix(in srgb, var(--color-warning) 30%, transparent);
	}

	.prompt-textarea {
		flex: 1;
		width: 100%;
		padding: 16px;
		font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
		font-size: 13px;
		line-height: 1.6;
		color: var(--text-primary);
		background: var(--bg-primary);
		border: none;
		outline: none;
		resize: none;
		tab-size: 2;
	}

	.prompt-textarea:focus {
		outline: none;
	}

	/* アクションバー */
	.action-bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 10px 16px;
		border-top: 0.5px solid var(--border);
		background: var(--bg-secondary);
	}

	.action-left,
	.action-right {
		display: flex;
		gap: 8px;
	}

	.btn-secondary {
		padding: 6px 14px;
		background: none;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
		cursor: pointer;
		transition: background 0.15s;
	}

	.btn-secondary:hover:not(:disabled) {
		background: var(--control-bg);
	}

	.btn-secondary:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn-reset {
		padding: 6px 14px;
		background: none;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
		cursor: pointer;
	}

	.btn-reset:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn-save {
		padding: 6px 18px;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
		font-weight: 500;
		cursor: pointer;
		transition: opacity 0.15s;
	}

	.btn-save:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	/* 履歴パネル */
	.history-panel {
		border-top: 0.5px solid var(--border);
		background: var(--bg-secondary);
		max-height: 200px;
		overflow-y: auto;
	}

	.history-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 8px 16px;
		border-bottom: 0.5px solid var(--border);
	}

	.history-header h4 {
		margin: 0;
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
	}

	.btn-close {
		background: none;
		border: none;
		color: var(--text-muted);
		font-size: 18px;
		cursor: pointer;
		padding: 0 4px;
		line-height: 1;
	}

	.history-empty {
		padding: 16px;
		text-align: center;
		color: var(--text-muted);
		font-size: 13px;
	}

	.history-list {
		padding: 4px 0;
	}

	.history-item {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 6px 16px;
	}

	.history-version {
		font-weight: 600;
		font-size: 12px;
		color: var(--accent);
		min-width: 36px;
	}

	.history-file {
		flex: 1;
		font-size: 12px;
		color: var(--text-muted);
		font-family: monospace;
	}

	.btn-rollback {
		padding: 3px 10px;
		background: none;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
		border-radius: 3px;
		font-size: 11px;
		font-family: inherit;
		cursor: pointer;
	}

	.btn-rollback:hover:not(:disabled) {
		background: var(--control-bg);
	}

	.btn-rollback:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	/* プレースホルダー */
	.prompt-placeholder {
		display: flex;
		justify-content: center;
		align-items: center;
		height: 100%;
		color: var(--text-muted);
		font-size: 14px;
	}

	/* レスポンシブ */
	@media (max-width: 767px) {
		.prompt-settings {
			grid-template-columns: 1fr;
			grid-template-rows: auto 1fr;
		}

		.prompt-sidebar {
			border-right: none;
			border-bottom: 0.5px solid var(--border);
			display: flex;
			overflow-x: auto;
			overflow-y: hidden;
			padding: 8px;
			gap: 4px;
		}

.prompt-item {
			white-space: nowrap;
			padding: 6px 12px;
		}
	}
</style>
