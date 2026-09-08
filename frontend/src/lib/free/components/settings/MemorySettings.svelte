<script lang="ts">
	import { t } from '$lib/i18n';
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';

	let memory = $derived(configSection($configData, 'memory'));
</script>

<SettingsSection tabId="memory">
	<FieldGroup label="settings.group_working_memory">
		<NumberField label="settings.memory.working_max_turns" value={Number(memory.working_max_turns ?? 10)} min={1} onchange={fieldUpdater('memory', 'working_max_turns')} />
		<NumberField label="settings.memory.working_max_tokens" value={Number(memory.working_max_tokens ?? 2048)} min={256} onchange={fieldUpdater('memory', 'working_max_tokens')} />
	</FieldGroup>

	<FieldGroup label="settings.group_short_term">
		<NumberField label="settings.memory.short_term_max_notes" value={Number(memory.short_term_max_notes ?? 100)} min={1} onchange={fieldUpdater('memory', 'short_term_max_notes')} />
	</FieldGroup>

	<FieldGroup label="settings.group_conflict">
		<NumberField label="settings.memory.conflict_similarity_threshold" value={Number(memory.conflict_similarity_threshold ?? 0.85)} min={0} max={1} step={0.05} onchange={fieldUpdater('memory', 'conflict_similarity_threshold')} />
		<NumberField label="settings.memory.conflict_batch_size" value={Number(memory.conflict_batch_size ?? 5)} min={1} onchange={fieldUpdater('memory', 'conflict_batch_size')} />
	</FieldGroup>

	<FieldGroup label="settings.group_note_evolution">
		<ToggleField label="settings.memory.note_evolution_enabled" value={Boolean(memory.note_evolution_enabled ?? true)} onchange={fieldUpdater('memory', 'note_evolution_enabled')} />
		<NumberField label="settings.memory.note_evolution_batch" value={Number(memory.note_evolution_batch ?? 10)} min={1} onchange={fieldUpdater('memory', 'note_evolution_batch')} />
		<NumberField label="settings.memory.note_evolution_context_k" value={Number(memory.note_evolution_context_k ?? 3)} min={1} onchange={fieldUpdater('memory', 'note_evolution_context_k')} />
	</FieldGroup>

	<FieldGroup label="settings.group_llm_call">
		<NumberField label="settings.memory.llm_call_base_interval" value={Number(memory.llm_call_base_interval ?? 1.0)} min={0} step={0.1} onchange={fieldUpdater('memory', 'llm_call_base_interval')} />
	</FieldGroup>
</SettingsSection>
