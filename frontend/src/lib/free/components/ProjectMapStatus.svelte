<script lang="ts">
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import type { ProjectMapStatus } from '$lib/free/api';
	import { getProjectMapStatus, updateProjectMap } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { formatDate } from '$lib/free/utils/format';
	import { addToast } from '$lib/free/stores/toast';
	import Button from './ui/Button.svelte';
	import Disclosure from './ui/Disclosure.svelte';

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
	<!-- 開発者向けの状態。通常は sleep-time が自動で更新するので既定で閉じる -->
	<section class="project-map" data-testid="project-map-status">
		<Disclosure label={$t('cartridge.project_map.title')}>
			<p class="muted">{$t('cartridge.project_map.description')}</p>
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
			<div class="tools">
				<Button size="sm" onclick={handleUpdate} disabled={updating || !status.enabled}>
					{updating ? $t('cartridge.project_map.updating') : $t('cartridge.project_map.update')}
				</Button>
			</div>
		</Disclosure>
	</section>
{/if}

<style>
	.project-map {
		margin-top: var(--space-5);
	}
	.roots {
		list-style: none;
		margin: var(--space-2) 0 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		font-size: var(--font-size-sm);
		color: var(--text-primary);
	}
	.roots li {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1) var(--space-3);
	}
	.root {
		font-family: monospace;
		font-weight: 600;
	}
	.muted {
		color: var(--text-secondary);
		margin: var(--space-1) 0 0;
		font-size: var(--font-size-sm);
	}
	.tools {
		margin-top: var(--space-3);
	}
</style>
