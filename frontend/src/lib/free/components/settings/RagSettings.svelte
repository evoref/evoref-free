<script lang="ts">
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';
	import SelectField from './fields/SelectField.svelte';

	let rag = $derived(configSection($configData, 'rag'));
</script>

<SettingsSection tabId="rag">
	<FieldGroup label="settings.group_rag" fullWidth columns={2}>
		<NumberField label="settings.rag.chunk_size" value={Number(rag.chunk_size ?? 512)} min={64} max={4096} onchange={fieldUpdater('rag', 'chunk_size')} />
		<NumberField label="settings.rag.chunk_overlap" value={Number(rag.chunk_overlap ?? 128)} min={0} onchange={fieldUpdater('rag', 'chunk_overlap')} />
		<SelectField label="settings.rag.chunking_strategy" value={String(rag.chunking_strategy ?? 'semantic')} options={[{value:'semantic',label:'Semantic',i18nLabel:'settings.option.semantic'},{value:'fixed',label:'Fixed',i18nLabel:'settings.option.fixed'}]} onchange={fieldUpdater('rag', 'chunking_strategy')} />
		<NumberField label="settings.rag.semantic_min_chunk" value={Number(rag.semantic_min_chunk ?? 64)} min={1} onchange={fieldUpdater('rag', 'semantic_min_chunk')} />
		<NumberField label="settings.rag.semantic_max_chunk" value={Number(rag.semantic_max_chunk ?? 512)} min={1} onchange={fieldUpdater('rag', 'semantic_max_chunk')} />
		<NumberField label="settings.rag.top_k" value={Number(rag.top_k ?? 5)} min={1} max={50} onchange={fieldUpdater('rag', 'top_k')} />
		<ToggleField label="settings.rag.hybrid_search" value={Boolean(rag.hybrid_search ?? true)} onchange={fieldUpdater('rag', 'hybrid_search')} />
		<NumberField label="settings.rag.bm25_weight" value={Number(rag.bm25_weight ?? 0.3)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'bm25_weight')} />
		<NumberField label="settings.rag.vector_weight" value={Number(rag.vector_weight ?? 0.7)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'vector_weight')} />
		<SelectField label="settings.rag.fusion_method" value={String(rag.fusion_method ?? 'rrf')} options={[{value:'rrf',label:'RRF',i18nLabel:'settings.option.rrf'},{value:'weighted',label:'Weighted',i18nLabel:'settings.option.weighted'}]} onchange={fieldUpdater('rag', 'fusion_method')} />
		<NumberField label="settings.rag.rrf_k" value={Number(rag.rrf_k ?? 60)} min={1} onchange={fieldUpdater('rag', 'rrf_k')} />
		<ToggleField label="settings.rag.contextual_retrieval" value={Boolean(rag.contextual_retrieval ?? true)} onchange={fieldUpdater('rag', 'contextual_retrieval')} />
		<NumberField label="settings.rag.contextual_prefix_max_tokens" value={Number(rag.contextual_prefix_max_tokens ?? 128)} min={1} onchange={fieldUpdater('rag', 'contextual_prefix_max_tokens')} />
		<NumberField label="settings.rag.contextual_max_doc_chars" value={Number(rag.contextual_max_doc_chars ?? 6000)} min={100} onchange={fieldUpdater('rag', 'contextual_max_doc_chars')} />
		<NumberField label="settings.rag.contextual_batch_size" value={Number(rag.contextual_batch_size ?? 10)} min={1} onchange={fieldUpdater('rag', 'contextual_batch_size')} />
		<SelectField label="settings.rag.quantization" value={String(rag.quantization ?? 'none')} options={[{value:'none',label:'None',i18nLabel:'settings.option.none'},{value:'int8',label:'INT8'}]} onchange={fieldUpdater('rag', 'quantization')} />
		<NumberField label="settings.rag.rescore_candidates" value={Number(rag.rescore_candidates ?? 50)} min={0} onchange={fieldUpdater('rag', 'rescore_candidates')} />
		<NumberField label="settings.rag.memmap_threshold" value={Number(rag.memmap_threshold ?? 10000)} min={100} onchange={fieldUpdater('rag', 'memmap_threshold')} />
		<NumberField label="settings.rag.relevance_threshold" value={Number(rag.relevance_threshold ?? 0.65)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'relevance_threshold')} />
		<NumberField label="settings.rag.support_threshold" value={Number(rag.support_threshold ?? 0.5)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'support_threshold')} />
		<NumberField label="settings.rag.confidence_threshold" value={Number(rag.confidence_threshold ?? 0.8)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'confidence_threshold')} />
		<NumberField label="settings.rag.hysteresis_band" value={Number(rag.hysteresis_band ?? 0.02)} min={0} max={0.5} step={0.01} onchange={fieldUpdater('rag', 'hysteresis_band')} />
	</FieldGroup>
</SettingsSection>
