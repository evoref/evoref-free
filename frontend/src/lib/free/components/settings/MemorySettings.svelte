<script lang="ts">
	import { t } from '$lib/i18n';
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, validateFadeWeights } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';

	let memory = $derived(configSection($configData, 'memory'));

	let fadeWeights = $derived(validateFadeWeights(
		memory.fade_alpha ?? 0.4,
		memory.fade_beta ?? 0.3,
		memory.fade_gamma ?? 0.3
	));
</script>

<SettingsSection tabId="memory">
	<FieldGroup label="settings.group_working_memory">
		<NumberField label="settings.memory.working_max_turns" value={Number(memory.working_max_turns ?? 10)} min={1} onchange={fieldUpdater('memory', 'working_max_turns')} />
		<NumberField label="settings.memory.working_max_tokens" value={Number(memory.working_max_tokens ?? 2048)} min={256} onchange={fieldUpdater('memory', 'working_max_tokens')} />
	</FieldGroup>

	<FieldGroup label="settings.group_short_term">
		<NumberField label="settings.memory.short_term_max_notes" value={Number(memory.short_term_max_notes ?? 100)} min={1} onchange={fieldUpdater('memory', 'short_term_max_notes')} />
		<NumberField label="settings.memory.lightmem_decay_days" value={Number(memory.lightmem_decay_days ?? 7)} min={1} onchange={fieldUpdater('memory', 'lightmem_decay_days')} />
	</FieldGroup>

	<FieldGroup label="settings.group_fade_mem">
		<NumberField label="settings.memory.fade_alpha" value={Number(memory.fade_alpha ?? 0.4)} min={0} max={1} step={0.05} onchange={fieldUpdater('memory', 'fade_alpha')} />
		<NumberField label="settings.memory.fade_beta" value={Number(memory.fade_beta ?? 0.3)} min={0} max={1} step={0.05} onchange={fieldUpdater('memory', 'fade_beta')} />
		<NumberField label="settings.memory.fade_gamma" value={Number(memory.fade_gamma ?? 0.3)} min={0} max={1} step={0.05} onchange={fieldUpdater('memory', 'fade_gamma')} />
		<p class="fade-sum" class:fade-sum-error={!fadeWeights.valid}>
			{$t('settings.memory.fade_sum')}: {fadeWeights.sum.toFixed(2)}
			{#if !fadeWeights.valid}
				({$t('settings.memory.fade_sum_must_be_1')})
			{/if}
		</p>
		<NumberField label="settings.memory.fade_threshold" value={Number(memory.fade_threshold ?? 0.15)} min={0} max={1} step={0.05} onchange={fieldUpdater('memory', 'fade_threshold')} />
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

<style>
	.fade-sum {
		font-size: 12px;
		color: var(--text-secondary);
		margin: 4px 0;
	}
	.fade-sum-error {
		color: var(--color-error);
		font-weight: 500;
	}
</style>
