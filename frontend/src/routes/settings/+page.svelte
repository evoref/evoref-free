<script lang="ts">
	import type { Component } from 'svelte';
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import SettingsTabs from '$lib/free/components/settings/SettingsTabs.svelte';
	import GeneralSettings from '$lib/free/components/settings/GeneralSettings.svelte';
	import InferenceSettings from '$lib/free/components/settings/InferenceSettings.svelte';
	import ModelSettings from '$lib/free/components/settings/ModelSettings.svelte';
	import RagSettings from '$lib/free/components/settings/RagSettings.svelte';
	import MemorySettings from '$lib/free/components/settings/MemorySettings.svelte';
	import LearningSettings from '$lib/free/components/settings/LearningSettings.svelte';
	import StorageSettings from '$lib/free/components/settings/StorageSettings.svelte';
	import IntegrationSettings from '$lib/free/components/settings/IntegrationSettings.svelte';
	import EditorSettings from '$lib/free/components/settings/EditorSettings.svelte';
	import ModeGenerationSettings from '$lib/free/components/settings/ModeGenerationSettings.svelte';
	import PromptSettings from '$lib/free/components/settings/PromptSettings.svelte';
	import DeveloperSettings from '$lib/free/components/settings/DeveloperSettings.svelte';
	import { loadConfig, activeTab, configLoaded, configLoadError } from '$lib/free/stores/settings';

	/** タブID → コンポーネントのレジストリ
	 * `--develop=<level>` の SSOT に集約された (詳細 docs/f_06_cli.md §6)。
	 */
	const TAB_COMPONENTS: Record<string, Component> = {
		general: GeneralSettings,
		inference: InferenceSettings,
		model: ModelSettings,
		rag: RagSettings,
		memory: MemorySettings,
		learning: LearningSettings,
		storage: StorageSettings,
		integration: IntegrationSettings,
		generation: ModeGenerationSettings,
		editor: EditorSettings,
		prompts: PromptSettings,
		develop: DeveloperSettings
	};

	onMount(() => {
		loadConfig();
	});
</script>

<PageLayout title={$t('sidebar.settings')} fullHeight={true}>
	{#if $configLoaded}
		<div class="settings-container">
			<SettingsTabs />
			<div class="settings-content">
				{#if TAB_COMPONENTS[$activeTab]}
					<svelte:component this={TAB_COMPONENTS[$activeTab]} />
				{/if}
			</div>
		</div>
	{:else if $configLoadError}
		<div class="load-error">
			<p>{$t('settings.load_failed')}</p>
			<button class="retry-btn" onclick={() => loadConfig()}>{$t('settings.retry')}</button>
		</div>
	{:else}
		<div class="loading">
			<p>{$t('settings.loading')}</p>
		</div>
	{/if}
</PageLayout>

<style>
	.settings-container {
		display: grid;
		grid-template-columns: 220px 1fr;
		height: 100%;
		overflow: hidden;
	}

	.settings-content {
		overflow-y: auto;
		height: 100%;
	}

	.loading {
		display: flex;
		justify-content: center;
		align-items: center;
		height: 200px;
		color: var(--text-muted);
		font-size: 14px;
	}

	.load-error {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		gap: 12px;
		height: 200px;
	}

	.load-error p {
		color: var(--color-error);
		font-size: 15px;
		font-weight: 600;
		margin: 0;
	}

	.retry-btn {
		padding: 6px 16px;
		background: none;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
	}

	@media (max-width: 767px) {
		.settings-container {
			grid-template-columns: 1fr;
			grid-template-rows: auto 1fr;
		}
	}
</style>
