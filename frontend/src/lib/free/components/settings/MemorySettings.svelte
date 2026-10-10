<script lang="ts">
	import { t } from '$lib/i18n';
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, nestedFieldUpdater, deepNestedFieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SelectField from './fields/SelectField.svelte';
	import PinnedFacts from './PinnedFacts.svelte';
	import ProSection from './ProSection.svelte';
	import { autoNumber } from '$lib/free/utils/autotune_view';

	let memory = $derived(configSection($configData, 'memory'));
	let injection = $derived(memory.injection ?? {});
	let retention = $derived(memory.evidence?.retention ?? {});
	let proKnowledge = $derived((($configData.pro ?? {}) as { knowledge?: { enabled?: boolean } }).knowledge ?? {});
</script>

<SettingsSection tabId="memory">
	<FieldGroup label="settings.group_working_memory">
		<NumberField label="settings.memory.working_max_turns" value={Number(memory.working_max_turns ?? 10)} min={1} onchange={fieldUpdater('memory', 'working_max_turns')} />
		<NumberField label="settings.memory.working_max_tokens" value={autoNumber(memory.working_max_tokens, true, 4096)} min={256} onchange={fieldUpdater('memory', 'working_max_tokens')} />
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

	<FieldGroup label="settings.group_memory_injection">
		<NumberField label="settings.memory.injection_chat_budget_tokens" value={Number(injection.chat_budget_tokens ?? 800)} min={0} onchange={nestedFieldUpdater('memory', 'injection', 'chat_budget_tokens')} />
		<NumberField label="settings.memory.injection_create_budget_tokens" value={Number(injection.create_budget_tokens ?? 2000)} min={0} onchange={nestedFieldUpdater('memory', 'injection', 'create_budget_tokens')} />
	</FieldGroup>

	<FieldGroup label="settings.group_memory_retention" description="settings.memory.retention_note">
		<NumberField label="settings.memory.retention_short_days" value={Number(retention.short_days ?? 14)} min={1} onchange={deepNestedFieldUpdater('memory', 'evidence', 'retention', 'short_days')} />
		<NumberField label="settings.memory.retention_long_max_records" value={Number(retention.long_max_records ?? 50000)} min={100} onchange={deepNestedFieldUpdater('memory', 'evidence', 'retention', 'long_max_records')} />
		<NumberField label="settings.memory.retention_idx_max_records" value={Number(retention.idx_max_records ?? 20000)} min={100} onchange={deepNestedFieldUpdater('memory', 'evidence', 'retention', 'idx_max_records')} />
		<NumberField label="settings.memory.retention_events_keep_months" value={Number(retention.events_keep_months ?? 3)} min={1} onchange={deepNestedFieldUpdater('memory', 'evidence', 'retention', 'events_keep_months')} />
		<NumberField label="settings.memory.retention_snapshots_keep" value={Number(retention.snapshots_keep ?? 3)} min={1} onchange={deepNestedFieldUpdater('memory', 'evidence', 'retention', 'snapshots_keep')} />
	</FieldGroup>

	<PinnedFacts />

	<ProSection columns={1}>
		<ToggleField label="settings.pro.knowledge_enabled" description="settings.pro.knowledge_enabled_desc" value={Boolean(proKnowledge.enabled ?? true)} onchange={nestedFieldUpdater('pro', 'knowledge', 'enabled')} />
	</ProSection>
</SettingsSection>
