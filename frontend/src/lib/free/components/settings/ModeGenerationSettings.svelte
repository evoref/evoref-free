<script lang="ts">
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, nestedFieldUpdater } from '$lib/free/stores/settingsHelpers';
	import type { ChatModeConfig, CreateModeConfig, LongFormConfig } from '$lib/types/settings';
	import SettingsSection from './SettingsSection.svelte';
	import ProSection from './ProSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SliderField from './fields/SliderField.svelte';

	// 応答の生成タブ: サンプリング (modes.*) と応答の長さ (llama.max_tokens)、長文の主要項目だけを出す。
	// 長文の内部調整 (rolling_short_term_chars / max_revisions / rag_per_unit / rag_top_k_per_unit) は config.yaml で書く
	let modes = $derived(configSection($configData, 'modes'));
	let llama = $derived(configSection($configData, 'llama'));
	let chat = $derived((modes.chat ?? {}) as ChatModeConfig);
	let create = $derived((modes.create ?? {}) as CreateModeConfig);
	let longForm = $derived(configSection($configData, 'long_form') as LongFormConfig);
</script>

<SettingsSection tabId="generation">
	<FieldGroup label="settings.group_response_length">
		<NumberField label="settings.llama.max_tokens" value={Number(llama.max_tokens ?? 2048)} min={0} description="settings.llama.max_tokens_desc" onchange={fieldUpdater('llama', 'max_tokens')} />
	</FieldGroup>

	<FieldGroup label="settings.group_mode_chat">
		<SliderField label="settings.modes.temperature" value={Number(chat.temperature ?? 0.7)} min={0} max={2} step={0.05} description="settings.modes.temperature_desc" onchange={nestedFieldUpdater('modes', 'chat', 'temperature')} />
		<SliderField label="settings.modes.top_p" value={Number(chat.top_p ?? 0.9)} min={0} max={1} step={0.01} description="settings.modes.top_p_desc" onchange={nestedFieldUpdater('modes', 'chat', 'top_p')} />
		<NumberField label="settings.modes.top_k" value={Number(chat.top_k ?? 40)} min={0} max={1000} description="settings.modes.top_k_desc" onchange={nestedFieldUpdater('modes', 'chat', 'top_k')} />
		<SliderField label="settings.modes.presence_penalty" value={Number(chat.presence_penalty ?? 0)} min={-2} max={2} step={0.1} description="settings.modes.presence_penalty_desc" onchange={nestedFieldUpdater('modes', 'chat', 'presence_penalty')} />
	</FieldGroup>

	<!-- Pro: クリエイトモードは Pro ワークモード専用 -->
	<ProSection>
		<FieldGroup label="settings.group_mode_create">
			<SliderField label="settings.modes.temperature" value={Number(create.temperature ?? 0.3)} min={0} max={2} step={0.05} description="settings.modes.temperature_desc" onchange={nestedFieldUpdater('modes', 'create', 'temperature')} />
			<SliderField label="settings.modes.top_p" value={Number(create.top_p ?? 0.95)} min={0} max={1} step={0.01} description="settings.modes.top_p_desc" onchange={nestedFieldUpdater('modes', 'create', 'top_p')} />
			<NumberField label="settings.modes.top_k" value={Number(create.top_k ?? 20)} min={0} max={1000} description="settings.modes.top_k_desc" onchange={nestedFieldUpdater('modes', 'create', 'top_k')} />
			<SliderField label="settings.modes.presence_penalty" value={Number(create.presence_penalty ?? 0)} min={-2} max={2} step={0.1} description="settings.modes.presence_penalty_desc" onchange={nestedFieldUpdater('modes', 'create', 'presence_penalty')} />
		</FieldGroup>
	</ProSection>

	<FieldGroup label="settings.group_long_form">
		<NumberField label="settings.long_form.max_units" value={Number(longForm.max_units ?? 20)} min={1} onchange={fieldUpdater('long_form', 'max_units')} />
		<NumberField label="settings.long_form.unit_max_tokens" value={Number(longForm.unit_max_tokens ?? 2000)} min={100} onchange={fieldUpdater('long_form', 'unit_max_tokens')} />
		<ToggleField label="settings.long_form.review_enabled" value={Boolean(longForm.review_enabled ?? true)} onchange={fieldUpdater('long_form', 'review_enabled')} />
	</FieldGroup>
</SettingsSection>
