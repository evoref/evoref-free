<script lang="ts">
	import { t } from '$lib/i18n';
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, nestedFieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import SelectField from './fields/SelectField.svelte';
	import ThresholdCalibrate from './ThresholdCalibrate.svelte';
	import { systemHealth } from '$lib/free/stores/server';

	let rag = $derived(configSection($configData, 'rag'));
	let rerankMode = $derived(normalizeRerankMode(rag.rerank?.mode));
	let rerankStatus = $derived($systemHealth?.rerank ?? null);

	/** YAML 1.1 では引用符無しの on / off が真偽値になる。未設定の既定は on */
	function normalizeRerankMode(mode: unknown): 'on' | 'off' {
		if (mode === false || mode === 'off') return 'off';
		return 'on';
	}
</script>

<SettingsSection tabId="rag">
	<FieldGroup label="settings.group_rag" fullWidth columns={2}>
		<NumberField label="settings.rag.chunk_size" value={Number(rag.chunk_size ?? 512)} min={64} max={4096} onchange={fieldUpdater('rag', 'chunk_size')} />
		<NumberField label="settings.rag.chunk_overlap" value={Number(rag.chunk_overlap ?? 128)} min={0} onchange={fieldUpdater('rag', 'chunk_overlap')} />
		<SelectField label="settings.rag.chunking_strategy" value={String(rag.chunking_strategy ?? 'semantic')} options={[{value:'semantic',label:'Semantic',i18nLabel:'settings.option.semantic'},{value:'fixed',label:'Fixed',i18nLabel:'settings.option.fixed'}]} onchange={fieldUpdater('rag', 'chunking_strategy')} />
		<NumberField label="settings.rag.semantic_min_chunk" value={Number(rag.semantic_min_chunk ?? 64)} min={1} onchange={fieldUpdater('rag', 'semantic_min_chunk')} />
		<NumberField label="settings.rag.semantic_max_chunk" value={Number(rag.semantic_max_chunk ?? 512)} min={1} onchange={fieldUpdater('rag', 'semantic_max_chunk')} />
		<NumberField label="settings.rag.top_k" value={Number(rag.top_k ?? 5)} min={1} max={50} onchange={fieldUpdater('rag', 'top_k')} />
		<SelectField label="settings.rag.quantization" value={String(rag.quantization ?? 'none')} options={[{value:'none',label:'None',i18nLabel:'settings.option.none'},{value:'int8',label:'INT8'}]} onchange={fieldUpdater('rag', 'quantization')} />
		<NumberField label="settings.rag.rescore_candidates" value={Number(rag.rescore_candidates ?? 50)} min={0} onchange={fieldUpdater('rag', 'rescore_candidates')} />
		<NumberField label="settings.rag.memmap_threshold" value={Number(rag.memmap_threshold ?? 10000)} min={100} onchange={fieldUpdater('rag', 'memmap_threshold')} />
		<NumberField label="settings.rag.relevance_threshold" value={Number(rag.relevance_threshold ?? 0.65)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'relevance_threshold')} />
		<NumberField label="settings.rag.support_threshold" value={Number(rag.support_threshold ?? 0.5)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'support_threshold')} />
		<NumberField label="settings.rag.confidence_threshold" value={Number(rag.confidence_threshold ?? 0.8)} min={0} max={1} step={0.05} onchange={fieldUpdater('rag', 'confidence_threshold')} />
		<NumberField label="settings.rag.hysteresis_band" value={Number(rag.hysteresis_band ?? 0.02)} min={0} max={0.5} step={0.01} onchange={fieldUpdater('rag', 'hysteresis_band')} />
	</FieldGroup>

	<FieldGroup label="settings.group_rerank" description="settings.rag.rerank_note">
		<SelectField
			label="settings.rag.rerank_mode"
			value={rerankMode}
			options={[
				{ value: 'on', label: 'On', i18nLabel: 'settings.option.on' },
				{ value: 'off', label: 'Off', i18nLabel: 'settings.option.off' }
			]}
			onchange={nestedFieldUpdater('rag', 'rerank', 'mode')}
		/>
		<p class="rerank-status" data-testid="rerank-status">
			{#if rerankStatus === null}
				{$t('settings.rag.rerank_status_unknown')}
			{:else if rerankStatus.enabled}
				{$t('settings.rag.rerank_status_enabled')}
			{:else}
				{$t('settings.rag.rerank_status_disabled', { reason: rerankStatus.reason || '-' })}
			{/if}
		</p>
	</FieldGroup>

	<!-- 埋め込みモデル切替後の閾値調整ツール (安全正規化 + インデックスからの推定) -->
	<ThresholdCalibrate />
</SettingsSection>

<style>
	.rerank-status {
		margin: 0;
		font-size: 12px;
		color: var(--text-secondary);
	}
</style>
