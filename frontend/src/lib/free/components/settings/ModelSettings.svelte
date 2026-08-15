<script lang="ts">
	import { configData, loadConfig, saveModelPathsField } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, nestedFieldUpdater, deepNestedFieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import ProSection from './ProSection.svelte';
	import ComponentMigrateButton from './ComponentMigrateButton.svelte';
	import ModelServerControl from './ModelServerControl.svelte';
	import EmbeddingRebuildButton from './EmbeddingRebuildButton.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SelectField from './fields/SelectField.svelte';

	let models = $derived(configSection($configData, 'model_paths'));
	let embedding = $derived(configSection($configData, 'embedding'));
</script>

<SettingsSection tabId="model">
	<!-- ベースモデル (migrate 専用。config 直保存は不可。再起動は別途手動) -->
	<FieldGroup label="settings.group_model_base">
		<!-- base llama-server の起動/停止/再起動。base 切替は手動再起動が必要 (create も同 slot) -->
		<ModelServerControl server="base" />
		<ComponentMigrateButton component="base" currentModel={String(models.base_model ?? '')} onMigrated={loadConfig} />
		<!-- Pro: クリエイトモードは Pro ワークモード専用。model_state 非追跡のため切替で即 config 保存 -->
		<ProSection columns={1}>
			<ComponentMigrateButton component="create" currentModel={String(models.create_model ?? '')} onMigrated={loadConfig} onApply={(p) => saveModelPathsField('create_model', p || null)} />
		</ProSection>
	</FieldGroup>

	<!-- 埋め込みモデル -->
	<FieldGroup label="settings.group_model_embedding">
		<!-- embed llama-server の起動/停止/再起動 (ServerName は "embed") -->
		<ModelServerControl server="embed" />
		<ComponentMigrateButton component="embedding" currentModel={String(models.embed_model ?? '')} onMigrated={loadConfig}>
			{#snippet actionsTrailing()}
				<!-- 埋め込みモデル切替後の再構築。RAG ベクトルと、URL/コマンドリコール用の
				     SemMem fact 埋め込みは別ストアだが、このボタン1つで両方まとめて再構築する。 -->
				<EmbeddingRebuildButton onRebuilt={loadConfig} />
			{/snippet}
		</ComponentMigrateButton>
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
</SettingsSection>
