<script lang="ts">
	import { t } from '$lib/i18n';
	import { importTasks, listTasks, ApiError, type TaskInfo } from '$lib/free/api';
	import { addToast } from '$lib/free/stores/toast';
	import { projectId } from '$lib/free/stores/loop_control';

	let open = $state(true);
	let prdText = $state('');
	let importing = $state(false);
	let loadingTasks = $state(false);
	let tasks = $state<TaskInfo[]>([]);
	let nextTaskId = $state<string | null>(null);

	function errorDetail(e: unknown): string {
		if (e instanceof ApiError) return e.message;
		return e instanceof Error ? e.message : String(e);
	}

	async function handleFile(e: Event): Promise<void> {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		prdText = await file.text();
		input.value = '';
	}

	async function handleImport(): Promise<void> {
		const pid = $projectId.trim();
		if (!pid) {
			addToast({ type: 'warning', i18nKey: 'loop.tasks.need_project' });
			return;
		}
		if (!prdText.trim()) {
			addToast({ type: 'warning', i18nKey: 'loop.tasks.need_prd' });
			return;
		}
		importing = true;
		try {
			const res = await importTasks(pid, prdText);
			addToast({ type: 'success', i18nKey: 'loop.tasks.imported', params: { count: res.imported } });
			await refreshTasks();
		} catch (e) {
			addToast({ type: 'error', i18nKey: 'loop.control.action_failed', params: { detail: errorDetail(e) } });
		} finally {
			importing = false;
		}
	}

	async function refreshTasks(): Promise<void> {
		const pid = $projectId.trim();
		if (!pid) {
			addToast({ type: 'warning', i18nKey: 'loop.tasks.need_project' });
			return;
		}
		loadingTasks = true;
		try {
			const res = await listTasks(pid);
			tasks = res.tasks;
			nextTaskId = res.next_task_id;
		} catch (e) {
			addToast({ type: 'error', i18nKey: 'loop.tasks.load_failed', params: { detail: errorDetail(e) } });
		} finally {
			loadingTasks = false;
		}
	}
</script>

<section class="task-panel">
	<button type="button" class="panel-header" aria-expanded={open} onclick={() => (open = !open)}>
		<span class="caret" class:open>▶</span>
		<span class="panel-title">{$t('loop.tasks.title')}</span>
	</button>

	{#if open}
		<div class="panel-body">
			<label class="prd-label" for="prd-textarea">{$t('loop.tasks.prd_label')}</label>
			<textarea
				id="prd-textarea"
				class="prd-input"
				bind:value={prdText}
				placeholder={$t('loop.tasks.prd_placeholder')}
				rows="6"
			></textarea>

			<div class="task-actions">
				<label class="btn file-btn">
					{$t('loop.tasks.choose_file')}
					<input type="file" accept=".json,application/json" onchange={handleFile} hidden />
				</label>
				<button type="button" class="btn btn-primary" onclick={handleImport} disabled={importing}>
					{importing ? $t('loop.tasks.importing') : $t('loop.tasks.import')}
				</button>
				<button type="button" class="btn" onclick={refreshTasks} disabled={loadingTasks}>
					{$t('loop.tasks.refresh')}
				</button>
			</div>

			<div class="task-list">
				<div class="list-head">
					<span class="list-title">{$t('loop.tasks.list_title')}</span>
					<span class="list-total">{$t('loop.tasks.total', { total: tasks.length })}</span>
				</div>
				{#if tasks.length === 0}
					<div class="list-empty">{$t('loop.tasks.no_tasks')}</div>
				{:else}
					<ul class="list-items">
						{#each tasks as task (task.fact_id)}
							<li class="task-item">
								<span class="task-status" data-status={task.status}>{task.status}</span>
								<span class="task-title">{task.title}</span>
								{#if task.task_id === nextTaskId}
									<span class="task-next">{$t('loop.tasks.next')}</span>
								{/if}
							</li>
						{/each}
					</ul>
				{/if}
			</div>
		</div>
	{/if}
</section>

<style>
	.task-panel {
		border-bottom: 0.5px solid var(--input-border);
		background: var(--bg-primary);
		flex-shrink: 0;
	}
	.panel-header {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		padding: 8px 16px;
		background: var(--bg-secondary, var(--bg-primary));
		border: none;
		cursor: pointer;
		color: var(--text-primary);
		font-family: inherit;
		text-align: left;
	}
	.caret {
		display: inline-block;
		transition: transform 0.15s;
		font-size: 0.7rem;
		color: var(--text-secondary);
	}
	.caret.open {
		transform: rotate(90deg);
	}
	.panel-title {
		font-weight: 600;
		font-size: 0.9rem;
	}
	.panel-body {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 10px 16px 14px;
	}
	.prd-label {
		font-size: 0.8rem;
		color: var(--text-secondary);
	}
	.prd-input {
		background: var(--input-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		padding: 8px;
		font-family: var(--font-monospace, monospace);
		font-size: 0.8rem;
		resize: vertical;
		outline: none;
	}
	.prd-input:focus {
		border-color: var(--accent);
	}
	.task-actions {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.btn {
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		padding: 4px 12px;
		font-size: 0.85rem;
		cursor: pointer;
		font-family: inherit;
		transition: background 0.15s;
	}
	.btn:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
		border-color: var(--accent);
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.file-btn {
		display: inline-flex;
		align-items: center;
	}
	.btn-primary:not(:disabled) {
		border-color: var(--accent);
		color: var(--accent);
	}
	.task-list {
		margin-top: 4px;
	}
	.list-head {
		display: flex;
		align-items: baseline;
		gap: 8px;
		margin-bottom: 4px;
	}
	.list-title {
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--text-primary);
	}
	.list-total {
		font-size: 0.75rem;
		color: var(--text-secondary);
	}
	.list-empty {
		font-size: 0.8rem;
		color: var(--text-secondary);
		font-style: italic;
		padding: 4px 0;
	}
	.list-items {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
		max-height: 200px;
		overflow-y: auto;
	}
	.task-item {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.82rem;
	}
	.task-status {
		font-size: 0.7rem;
		padding: 1px 6px;
		border-radius: 3px;
		border: 0.5px solid var(--input-border);
		color: var(--text-secondary);
		min-width: 64px;
		text-align: center;
	}
	.task-status[data-status='done'] {
		color: var(--success, var(--accent));
		border-color: var(--success, var(--accent));
	}
	.task-status[data-status='failed'] {
		color: var(--error, var(--text-secondary));
		border-color: var(--error, var(--text-secondary));
	}
	.task-status[data-status='in_progress'] {
		color: var(--warning, var(--accent));
		border-color: var(--warning, var(--accent));
	}
	.task-title {
		color: var(--text-primary);
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.task-next {
		font-size: 0.7rem;
		padding: 1px 6px;
		border-radius: 3px;
		background: color-mix(in srgb, var(--accent) 25%, transparent);
		color: var(--text-primary);
	}
</style>
