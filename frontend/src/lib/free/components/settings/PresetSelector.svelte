<script lang="ts">
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n';
	import { fetchPresets, applyPreset } from '$lib/free/services/configService';
	import { loadConfig } from '$lib/free/stores/settings';
	import { startServer, type ServerName } from '$lib/free/api';
	import { refreshServerStatus } from '$lib/free/stores/server';
	import { refreshVramStatus } from '$lib/free/stores/vram';
	import { addToast } from '$lib/free/stores/toast';

	const PRESET_IDS = ['light', 'balanced', 'performance'] as const;
	type PresetId = (typeof PRESET_IDS)[number];

	let available = $state<string[]>([]);
	let current = $state<string | null>(null);
	let applying = $state<string | null>(null);
	let restartServers = $state<string[]>([]);
	let restarting = $state(false);
	let loadFailed = $state(false);

	onMount(async () => {
		try {
			const resp = await fetchPresets();
			available = resp.presets.map((p) => p.id);
			current = resp.current;
		} catch {
			loadFailed = true;
		}
	});

	async function handleApply(id: PresetId) {
		const name = $t(`settings.presets.${id}.label`);
		if (!confirm($t('settings.presets.confirm', { name }))) return;

		applying = id;
		try {
			const result = await applyPreset(id);
			await loadConfig(); // 全タブの値を再取得して反映
			current = result.applied;
			restartServers = result.restart_servers;
			addToast({ type: 'success', i18nKey: 'settings.presets.applied' });
		} catch {
			addToast({ type: 'error', i18nKey: 'settings.presets.apply_failed' });
		} finally {
			applying = null;
		}
	}

	async function handleRestart() {
		if (restartServers.length === 0) return;

		restarting = true;
		try {
			for (const name of restartServers) {
				await startServer(name as ServerName, true);
			}
			await Promise.all([refreshServerStatus(), refreshVramStatus()]);
			restartServers = [];
			addToast({ type: 'success', i18nKey: 'settings.presets.restart_done' });
		} catch {
			addToast({ type: 'error', i18nKey: 'settings.presets.restart_failed' });
		} finally {
			restarting = false;
		}
	}
</script>

{#if !loadFailed && available.length > 0}
	<div class="preset-card">
		<div class="preset-head">
			<h3 class="preset-title">{$t('settings.presets.title')}</h3>
			<p class="preset-desc">{$t('settings.presets.description')}</p>
		</div>

		<div class="preset-grid">
			{#each PRESET_IDS as id (id)}
				{#if available.includes(id)}
					<button
						type="button"
						class="preset-btn"
						class:active={current === id}
						disabled={applying !== null}
						onclick={() => handleApply(id)}
					>
						<span class="preset-label">
							{$t(`settings.presets.${id}.label`)}
							{#if current === id}
								<span class="preset-badge">{$t('settings.presets.current_badge')}</span>
							{/if}
						</span>
						<span class="preset-sub">{$t(`settings.presets.${id}.description`)}</span>
						{#if applying === id}
							<span class="preset-applying">{$t('settings.applying')}</span>
						{/if}
					</button>
				{/if}
			{/each}
		</div>

		{#if restartServers.length > 0}
			<div class="preset-restart">
				<span class="preset-restart-msg">{$t('settings.presets.restart_required')}</span>
				<button
					type="button"
					class="preset-restart-btn"
					disabled={restarting}
					onclick={handleRestart}
				>
					{restarting
						? $t('settings.presets.restarting')
						: $t('settings.presets.restart_button')}
				</button>
			</div>
		{/if}
	</div>
{/if}

<style>
	.preset-card {
		padding: 16px 24px;
		border-bottom: 0.5px solid var(--border);
		background: var(--bg-secondary);
	}

	.preset-head {
		margin-bottom: 12px;
	}

	.preset-title {
		margin: 0;
		font-size: 14px;
		font-weight: 600;
	}

	.preset-desc {
		margin: 4px 0 0;
		font-size: 12px;
		color: var(--text-secondary);
	}

	.preset-grid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 12px;
	}

	.preset-btn {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 12px 14px;
		text-align: left;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 6px;
		cursor: pointer;
		font-family: inherit;
		transition:
			border-color 0.15s,
			background 0.15s;
	}

	.preset-btn:hover:not(:disabled) {
		border-color: var(--accent);
	}

	.preset-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.preset-btn.active {
		border-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}

	.preset-label {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 13px;
		font-weight: 600;
	}

	.preset-badge {
		padding: 1px 6px;
		font-size: 10px;
		font-weight: 500;
		color: white;
		background: var(--accent);
		border-radius: 8px;
	}

	.preset-sub {
		font-size: 11px;
		line-height: 1.4;
		color: var(--text-secondary);
	}

	.preset-applying {
		margin-top: 2px;
		font-size: 11px;
		color: var(--accent);
	}

	.preset-restart {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		margin-top: 12px;
		padding: 10px 14px;
		background: color-mix(in srgb, var(--accent) 8%, transparent);
		border: 0.5px solid var(--accent);
		border-radius: 6px;
	}

	.preset-restart-msg {
		font-size: 12px;
	}

	.preset-restart-btn {
		flex: none;
		padding: 6px 16px;
		font-size: 13px;
		font-family: inherit;
		color: white;
		background: var(--accent);
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	.preset-restart-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	@media (max-width: 767px) {
		.preset-card {
			padding: 12px 16px;
		}
		.preset-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
