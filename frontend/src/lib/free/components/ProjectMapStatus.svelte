<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import type { ProjectMapStatus } from '$lib/free/api';
	import { getProjectMapStatus, updateProjectMap } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatDate } from '$lib/free/utils/format';
	import { addToast } from '$lib/free/stores/toast';

	let status = $state<ProjectMapStatus | null>(null);
	let updating = $state(false);

	onMount(load);

	async function load() {
		const result = await handleApiCall(() => getProjectMapStatus(), {
			fallbackKey: 'cartridge.project_map.load_failed'
		});
		status = result ?? null;
	}

	async function handleUpdate() {
		if (updating) return;
		updating = true;
		try {
			const result = await handleApiCall(() => updateProjectMap(), {
				fallbackKey: 'cartridge.project_map.update_failed'
			});
			if (result) {
				addToast({
					type: 'success',
					i18nKey: 'cartridge.project_map.update_done',
					params: { count: result.updated_roots }
				});
				await load();
			}
		} finally {
			updating = false;
		}
	}
</script>

{#if status}
	<section class="project-map" data-testid="project-map-status">
		<div class="head">
			<h3>{$t('cartridge.project_map.title')}</h3>
			<button class="update-btn" onclick={handleUpdate} disabled={updating || !status.enabled}>
				{updating ? $t('cartridge.project_map.updating') : $t('cartridge.project_map.update')}
			</button>
		</div>
		{#if !status.enabled}
			<p class="muted">{$t('cartridge.project_map.disabled')}</p>
		{:else}
			<ul class="roots">
				{#each status.roots as r (r.root)}
					<li>
						<span class="root">{r.root}</span>
						{#if r.version}
							<span>
								{$t('cartridge.project_map.nodes')}: {r.nodes} /
								{$t('cartridge.project_map.edges')}: {r.edges}
							</span>
							<span>{$t('cartridge.project_map.updated_at')}: {formatDate(r.written_at)}</span>
							{#if Object.keys(r.languages).length > 0}
								<span>
									{$t('cartridge.project_map.languages')}:
									{Object.entries(r.languages)
										.map(([lang, n]) => `${lang} ${n}`)
										.join(', ')}
								</span>
							{/if}
						{:else}
							<span class="muted">{$t('cartridge.project_map.no_version')}</span>
						{/if}
					</li>
				{/each}
			</ul>
		{/if}
	</section>
{/if}

<style>
	.project-map {
		margin-bottom: 16px;
		padding: 12px;
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		background-color: var(--bg-secondary);
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	h3 {
		margin: 0;
		font-size: 1rem;
		color: var(--text-primary);
	}
	.update-btn {
		padding: 4px 12px;
		background-color: var(--accent);
		color: var(--text-on-accent);
		border: none;
		border-radius: var(--border-radius);
		cursor: pointer;
		font-size: 0.9rem;
	}
	.update-btn:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.roots {
		list-style: none;
		margin: 8px 0 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 0.9rem;
		color: var(--text-primary);
	}
	.roots li {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
	}
	.root {
		font-family: monospace;
		font-weight: 600;
	}
	.muted {
		color: var(--text-secondary);
		margin: 8px 0 0;
		font-size: 0.9rem;
	}
</style>
