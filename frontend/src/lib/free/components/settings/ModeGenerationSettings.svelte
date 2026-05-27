<script lang="ts">
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, nestedFieldUpdater } from '$lib/free/stores/settingsHelpers';
	import type { ChatModeConfig, CodingModeConfig, LongFormConfig } from '$lib/types/settings';
	import SettingsSection from './SettingsSection.svelte';
	import ProSection from './ProSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SliderField from './fields/SliderField.svelte';

	let modes = $derived(configSection($configData, 'modes'));
	let chat = $derived((modes.chat ?? {}) as ChatModeConfig);
	let coding = $derived((modes.coding ?? {}) as CodingModeConfig);
	let longForm = $derived(configSection($configData, 'long_form') as LongFormConfig);
</script>

<SettingsSection tabId="generation">
	<FieldGroup label="settings.group_mode_chat">
		<SliderField label="settings.modes.temperature" value={Number(chat.temperature ?? 0.7)} min={0} max={2} step={0.05} description="settings.modes.temperature_desc" onchange={nestedFieldUpdater('modes', 'chat', 'temperature')} />
		<SliderField label="settings.modes.top_p" value={Number(chat.top_p ?? 0.9)} min={0} max={1} step={0.01} description="settings.modes.top_p_desc" onchange={nestedFieldUpdater('modes', 'chat', 'top_p')} />
		<NumberField label="settings.modes.top_k" value={Number(chat.top_k ?? 40)} min={0} max={1000} description="settings.modes.top_k_desc" onchange={nestedFieldUpdater('modes', 'chat', 'top_k')} />
		<SliderField label="settings.modes.presence_penalty" value={Number(chat.presence_penalty ?? 0)} min={-2} max={2} step={0.1} description="settings.modes.presence_penalty_desc" onchange={nestedFieldUpdater('modes', 'chat', 'presence_penalty')} />
	</FieldGroup>

	<!-- Pro: コーディングモードは Pro ワークモード専用 -->
	<ProSection>
		<FieldGroup label="settings.group_mode_coding">
			<SliderField label="settings.modes.temperature" value={Number(coding.temperature ?? 0.3)} min={0} max={2} step={0.05} description="settings.modes.temperature_desc" onchange={nestedFieldUpdater('modes', 'coding', 'temperature')} />
			<SliderField label="settings.modes.top_p" value={Number(coding.top_p ?? 0.95)} min={0} max={1} step={0.01} description="settings.modes.top_p_desc" onchange={nestedFieldUpdater('modes', 'coding', 'top_p')} />
			<NumberField label="settings.modes.top_k" value={Number(coding.top_k ?? 20)} min={0} max={1000} description="settings.modes.top_k_desc" onchange={nestedFieldUpdater('modes', 'coding', 'top_k')} />
			<SliderField label="settings.modes.presence_penalty" value={Number(coding.presence_penalty ?? 0)} min={-2} max={2} step={0.1} description="settings.modes.presence_penalty_desc" onchange={nestedFieldUpdater('modes', 'coding', 'presence_penalty')} />
		</FieldGroup>
	</ProSection>

	<FieldGroup label="settings.group_long_form">
		<NumberField label="settings.long_form.max_units" value={Number(longForm.max_units ?? 20)} min={1} onchange={fieldUpdater('long_form', 'max_units')} />
		<NumberField label="settings.long_form.unit_max_tokens" value={Number(longForm.unit_max_tokens ?? 2000)} min={100} onchange={fieldUpdater('long_form', 'unit_max_tokens')} />
		<NumberField label="settings.long_form.rolling_short_term_chars" value={Number(longForm.rolling_short_term_chars ?? 1000)} min={100} onchange={fieldUpdater('long_form', 'rolling_short_term_chars')} />
		<ToggleField label="settings.long_form.review_enabled" value={Boolean(longForm.review_enabled ?? true)} onchange={fieldUpdater('long_form', 'review_enabled')} />
		<NumberField label="settings.long_form.max_revisions" value={Number(longForm.max_revisions ?? 1)} min={0} onchange={fieldUpdater('long_form', 'max_revisions')} />
		<ToggleField label="settings.long_form.rag_per_unit" value={Boolean(longForm.rag_per_unit ?? true)} onchange={fieldUpdater('long_form', 'rag_per_unit')} />
		<NumberField label="settings.long_form.rag_top_k_per_unit" value={Number(longForm.rag_top_k_per_unit ?? 3)} min={1} onchange={fieldUpdater('long_form', 'rag_top_k_per_unit')} />
	</FieldGroup>
</SettingsSection>
