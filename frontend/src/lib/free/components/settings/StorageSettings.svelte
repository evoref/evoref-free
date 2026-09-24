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

</script>

<SettingsSection tabId="storage">
	<FieldGroup label="settings.group_history">
		<ToggleField label="settings.history.auto_save" value={Boolean(history.auto_save ?? true)} onchange={fieldUpdater('history', 'auto_save')} />
		<NumberField label="settings.history.retention_full_days" value={Number(history.retention_full_days ?? 90)} min={1} onchange={fieldUpdater('history', 'retention_full_days')} />
		<NumberField label="settings.history.retention_compressed_days" value={Number(history.retention_compressed_days ?? 365)} min={1} onchange={fieldUpdater('history', 'retention_compressed_days')} />
		<NumberField label="settings.history.max_storage_mb" value={Number(history.max_storage_mb ?? 200)} min={1} step={10} onchange={fieldUpdater('history', 'max_storage_mb')} />
		<NumberField label="settings.history.summary_batch_size" value={Number(history.summary_batch_size ?? 5)} min={1} onchange={fieldUpdater('history', 'summary_batch_size')} />
	</FieldGroup>

	<!-- データはデータ根 (userdata/) の下に置く。利用者が変えられるのは生成物の既定の書込み先だけ -->
	<FieldGroup label="settings.group_local_paths" fullWidth>
		<TextField label="settings.local_paths.outputs_dir" value={String(local.outputs_dir ?? 'outputs/')} onchange={(v) => updateField('local_paths', 'outputs_dir', v)} />
	</FieldGroup>
</SettingsSection>
