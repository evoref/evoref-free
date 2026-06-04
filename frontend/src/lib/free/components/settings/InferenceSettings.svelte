<script lang="ts">
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SelectField from './fields/SelectField.svelte';
	import TagListField from './fields/TagListField.svelte';

	let llama = $derived(configSection($configData, 'llama'));

	const cacheTypeOptions = [
		{ value: 'f16', label: 'f16' },
		{ value: 'bf16', label: 'bf16' },
		{ value: 'q8_0', label: 'q8_0' },
		{ value: 'q5_1', label: 'q5_1' },
		{ value: 'q5_0', label: 'q5_0' },
		{ value: 'q4_1', label: 'q4_1' },
		{ value: 'q4_0', label: 'q4_0' }
	];
	// reasoning (enable_thinking) はモデルプロファイル (models/profiles/<arch>.yaml) が
	// SSOT・起動時のみ反映。管理画面からの実行時切替は持たない (docs/c_15)。
</script>

<SettingsSection tabId="inference">
	<FieldGroup label="settings.group_llama_server">
		<TextField label="settings.llama.host" value={String(llama.host ?? 'localhost')} onchange={fieldUpdater('llama', 'host')} />
		<NumberField label="settings.llama.port" value={Number(llama.port ?? 8080)} min={1024} max={65535} onchange={fieldUpdater('llama', 'port')} />
	</FieldGroup>

	<FieldGroup label="settings.group_llama_performance">
		<NumberField label="settings.llama.context_size" value={Number(llama.context_size ?? 4096)} min={512} onchange={fieldUpdater('llama', 'context_size')} />
		<NumberField label="settings.llama.gpu_layers" value={Number(llama.gpu_layers ?? 999)} min={-1} onchange={fieldUpdater('llama', 'gpu_layers')} />
		<NumberField label="settings.llama.threads" value={Number(llama.threads ?? 0)} min={0} description="settings.llama.threads_desc" onchange={fieldUpdater('llama', 'threads')} />
		<NumberField label="settings.llama.batch_size" value={Number(llama.batch_size ?? 512)} min={1} onchange={fieldUpdater('llama', 'batch_size')} />
		<ToggleField label="settings.llama.flash_attn" value={Boolean(llama.flash_attn ?? true)} onchange={fieldUpdater('llama', 'flash_attn')} />
		<ToggleField label="settings.llama.mlock" value={Boolean(llama.mlock ?? false)} onchange={fieldUpdater('llama', 'mlock')} />
		<NumberField label="settings.llama.slots" value={Number(llama.slots ?? 2)} min={1} max={16} onchange={fieldUpdater('llama', 'slots')} />
	</FieldGroup>

	<FieldGroup label="settings.group_llama_generation">
		<ToggleField label="settings.llama.cache_prompt" value={Boolean(llama.cache_prompt ?? true)} onchange={fieldUpdater('llama', 'cache_prompt')} />
		<SelectField label="settings.llama.cache_type_k" value={String(llama.cache_type_k ?? 'q8_0')} options={cacheTypeOptions} onchange={fieldUpdater('llama', 'cache_type_k')} />
		<SelectField label="settings.llama.cache_type_v" value={String(llama.cache_type_v ?? 'q8_0')} options={cacheTypeOptions} onchange={fieldUpdater('llama', 'cache_type_v')} />
		<NumberField label="settings.llama.max_tokens" value={Number(llama.max_tokens ?? 1024)} min={0} description="settings.llama.max_tokens_desc" onchange={fieldUpdater('llama', 'max_tokens')} />
		<TextField label="settings.llama.lora_target" value={String(llama.lora_target ?? 'auto')} onchange={fieldUpdater('llama', 'lora_target')} />
		<TagListField label="settings.llama.extra_args" value={(llama.extra_args as string[]) ?? []} placeholder="--arg value" onchange={fieldUpdater('llama', 'extra_args')} />
	</FieldGroup>
</SettingsSection>
