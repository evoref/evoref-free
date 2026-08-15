<script lang="ts">
	import { configData, updateField } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';

	let local = $derived(configSection($configData, 'local_paths'));
	let history = $derived(configSection($configData, 'history'));

	const localPathKeys = [
		'lora_adapter', 'lora_versions_dir', 'aux_experience_file', 'lora_archive_dir',
		'embed_lora_adapter', 'embed_lora_versions_dir',
		'vectors_dir', 'knowledge_dir', 'experience_file', 'eval_core_file',
		'model_state_file', 'memory_dir', 'prompts_dir', 'cartridges_dir',
		'history_dir', 'learned_patterns_file', 'themes_dir'
	];
</script>

<SettingsSection tabId="storage">
	<FieldGroup label="settings.group_history">
		<ToggleField label="settings.history.auto_save" value={Boolean(history.auto_save ?? true)} onchange={fieldUpdater('history', 'auto_save')} />
		<NumberField label="settings.history.checkpoint_interval" value={Number(history.checkpoint_interval ?? 10)} min={1} onchange={fieldUpdater('history', 'checkpoint_interval')} />
		<NumberField label="settings.history.retention_full_days" value={Number(history.retention_full_days ?? 90)} min={1} onchange={fieldUpdater('history', 'retention_full_days')} />
		<NumberField label="settings.history.retention_compressed_days" value={Number(history.retention_compressed_days ?? 365)} min={1} onchange={fieldUpdater('history', 'retention_compressed_days')} />
		<NumberField label="settings.history.max_storage_mb" value={Number(history.max_storage_mb ?? 200)} min={1} step={10} onchange={fieldUpdater('history', 'max_storage_mb')} />
		<NumberField label="settings.history.summary_batch_size" value={Number(history.summary_batch_size ?? 5)} min={1} onchange={fieldUpdater('history', 'summary_batch_size')} />
	</FieldGroup>

	<!-- Local Paths: full width (many items) -->
	<FieldGroup label="settings.group_local_paths" fullWidth>
		{#each localPathKeys as key}
			{#if local[key] !== undefined}
				<TextField label="settings.local_paths.{key}" value={String(local[key] ?? '')} onchange={(v) => updateField('local_paths', key, v)} />
			{/if}
		{/each}
	</FieldGroup>
</SettingsSection>
