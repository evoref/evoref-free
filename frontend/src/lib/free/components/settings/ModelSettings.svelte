<script lang="ts">
	import { configData, updateField, loadConfig } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, nestedFieldUpdater } from '$lib/free/stores/settingsHelpers';
	import type { AssistModelLocalConfig } from '$lib/types/settings';
	import SettingsSection from './SettingsSection.svelte';
	import ProSection from './ProSection.svelte';
	import ComponentMigrateButton from './ComponentMigrateButton.svelte';
	import BaseModelMigrateButton from './BaseModelMigrateButton.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SelectField from './fields/SelectField.svelte';

	let models = $derived(configSection($configData, 'model_paths'));
	let assistModel = $derived(configSection($configData, 'assist_model'));
	let local = $derived((assistModel.local ?? {}) as AssistModelLocalConfig);
	let concurrency = $derived((assistModel.concurrency ?? {}) as Record<string, number>);
	let embedding = $derived(configSection($configData, 'embedding'));
	let reranker = $derived(configSection($configData, 'reranker'));
</script>

<SettingsSection tabId="model">
	<!-- ベースモデル (migrate 専用。config 直保存は不可) -->
	<FieldGroup label="settings.group_model_base">
		<BaseModelMigrateButton currentModel={String(models.base_model ?? '')} onMigrated={loadConfig} />
		<!-- Pro: コーディングモードは Pro ワークモード専用。model_state 非追跡のため通常編集可 -->
		<ProSection columns={1}>
			<TextField label="settings.model_paths.coding_model" value={String(models.coding_model ?? '')} description="settings.model_paths.coding_model_desc" onchange={(v) => updateField('model_paths', 'coding_model', v || null)} />
		</ProSection>
	</FieldGroup>

	<!-- アシストモデル -->
	<FieldGroup label="settings.group_model_assist">
		<ToggleField label="settings.assist_model.enabled" value={Boolean(assistModel.enabled ?? true)} onchange={fieldUpdater('assist_model', 'enabled')} />
		<ComponentMigrateButton component="assist" currentModel={String(models.assist_model ?? '')} onMigrated={loadConfig} />
		<TextField label="settings.assist_model.local.host" value={String(local.host ?? '127.0.0.1')} onchange={nestedFieldUpdater('assist_model', 'local', 'host')} />
		<NumberField label="settings.assist_model.local.port" value={Number(local.port ?? 8081)} min={1024} max={65535} onchange={nestedFieldUpdater('assist_model', 'local', 'port')} />
		<NumberField label="settings.assist_model.local.context_size" value={Number(local.context_size ?? 2048)} min={512} onchange={nestedFieldUpdater('assist_model', 'local', 'context_size')} />
		<NumberField label="settings.assist_model.timeout" value={Number(assistModel.timeout ?? 10)} min={0.1} step={0.5} onchange={fieldUpdater('assist_model', 'timeout')} />
		<NumberField label="settings.assist_model.concurrency_realtime" value={Number(concurrency.realtime ?? 1)} min={1} onchange={nestedFieldUpdater('assist_model', 'concurrency', 'realtime')} />
		<NumberField label="settings.assist_model.concurrency_background" value={Number(concurrency.background ?? 1)} min={1} onchange={nestedFieldUpdater('assist_model', 'concurrency', 'background')} />
		<NumberField label="settings.assist_model.concurrency_learning" value={Number(concurrency.learning ?? 1)} min={1} onchange={nestedFieldUpdater('assist_model', 'concurrency', 'learning')} />
		<ProSection>
			<SelectField
				label="settings.assist_model.backend"
				value={String(assistModel.backend ?? 'local')}
				options={[
					{ value: 'local', label: 'Local', i18nLabel: 'settings.option.local' },
					{ value: 'external', label: 'External', i18nLabel: 'settings.option.external' },
					{ value: 'hybrid', label: 'Hybrid', i18nLabel: 'settings.option.hybrid' }
				]}
				onchange={fieldUpdater('assist_model', 'backend')}
			/>
		</ProSection>
	</FieldGroup>

	<!-- 埋め込みモデル -->
	<FieldGroup label="settings.group_model_embedding">
		<ComponentMigrateButton component="embedding" currentModel={String(models.embed_model ?? '')} onMigrated={loadConfig} />
		<TextField label="settings.embedding.llama_host" value={String(embedding.llama_host ?? 'localhost')} onchange={fieldUpdater('embedding', 'llama_host')} />
		<NumberField label="settings.embedding.llama_port" value={Number(embedding.llama_port ?? 8082)} min={1024} max={65535} onchange={fieldUpdater('embedding', 'llama_port')} />
		<NumberField label="settings.embedding.dim" value={Number(embedding.dim ?? 1024)} min={1} onchange={fieldUpdater('embedding', 'dim')} />
		<NumberField label="settings.embedding.timeout" value={Number(embedding.timeout ?? 30)} min={0.1} step={0.5} onchange={fieldUpdater('embedding', 'timeout')} />
		<NumberField label="settings.embedding.max_length" value={Number(embedding.max_length ?? 8192)} min={1} onchange={fieldUpdater('embedding', 'max_length')} />
		<TextField label="settings.embedding.model_name" value={String(embedding.model_name ?? '')} onchange={fieldUpdater('embedding', 'model_name')} />
		<ToggleField label="settings.embedding.cache_enabled" value={Boolean(embedding.cache_enabled ?? true)} onchange={fieldUpdater('embedding', 'cache_enabled')} />
		<NumberField label="settings.embedding.cache_max_mb" value={Number(embedding.cache_max_mb ?? 100)} min={1} onchange={fieldUpdater('embedding', 'cache_max_mb')} />
		<TextField label="settings.embedding.cache_dir" value={String(embedding.cache_dir ?? '')} onchange={fieldUpdater('embedding', 'cache_dir')} />
	</FieldGroup>

	<!-- リランカー -->
	<FieldGroup label="settings.group_model_reranker">
		<ToggleField label="settings.reranker.enabled" value={Boolean(reranker.enabled ?? false)} onchange={fieldUpdater('reranker', 'enabled')} />
		<ComponentMigrateButton component="reranker" currentModel={String(models.reranker_model ?? '')} onMigrated={loadConfig} />
		<TextField label="settings.reranker.backend" value={String(reranker.backend ?? 'llama-cpp')} onchange={fieldUpdater('reranker', 'backend')} />
		<TextField label="settings.reranker.host" value={String(reranker.host ?? 'localhost')} onchange={fieldUpdater('reranker', 'host')} />
		<NumberField label="settings.reranker.port" value={Number(reranker.port ?? 8083)} min={1024} max={65535} onchange={fieldUpdater('reranker', 'port')} />
		<TextField label="settings.reranker.model_name" value={String(reranker.model_name ?? '')} onchange={fieldUpdater('reranker', 'model_name')} />
		<NumberField label="settings.reranker.timeout" value={Number(reranker.timeout ?? 30)} min={0.1} step={0.5} onchange={fieldUpdater('reranker', 'timeout')} />
		<NumberField label="settings.reranker.candidates_multiplier" value={Number(reranker.candidates_multiplier ?? 3)} min={1} max={10} onchange={fieldUpdater('reranker', 'candidates_multiplier')} />
	</FieldGroup>
</SettingsSection>
