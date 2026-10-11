<script lang="ts">
	/**
	 * モデル画面 (設定の「モデルと起動」タブ)
	 *
	 * base / 埋め込み / リランカー (/ Pro の create) を同じ形のカードで並べる。各カードは
	 * ① モデルファイル (切替ボタン) ② 起動オプション (launch_args、c_16 §7.2.4) ③ サーバーの起動・停止。
	 * どれもその場で保存・実行する (タブの「適用」は無い)。create は base と同じ llama-server・同じ
	 * 起動オプションで動くので ② ③ を持たない。
	 */
	import { t } from '$lib/i18n';
	import { configData, loadConfig, saveModelPathsField } from '$lib/free/stores/settings';
	import { configSection } from '$lib/free/stores/settingsHelpers';
	import ProSection from './ProSection.svelte';
	import ComponentMigrateButton from './ComponentMigrateButton.svelte';
	import ModelServerControl from './ModelServerControl.svelte';
	import EmbeddingRebuildButton from './EmbeddingRebuildButton.svelte';
	import LaunchArgsField from './LaunchArgsField.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';

	let models = $derived(configSection($configData, 'model_paths'));
	let llama = $derived(configSection($configData, 'llama'));
	let embedding = $derived(configSection($configData, 'embedding'));
	let rerank = $derived((configSection($configData, 'rag').rerank ?? {}) as Record<string, unknown>);
</script>

<div class="model-cards">
	<!-- ベースモデル (切替は migrate 専用。反映は再起動) -->
	<FieldGroup label="settings.group_model_base">
		<div class="card-part">
			<span class="part-label">{$t('settings.model_card.model_file')}</span>
			<ComponentMigrateButton component="base" currentModel={String(models.base_model ?? '')} onMigrated={loadConfig} />
		</div>
		<div class="card-part">
			<LaunchArgsField target="llama" value={String(llama.launch_args ?? '')} />
		</div>
		<div class="card-part">
			<ModelServerControl server="base" />
		</div>
	</FieldGroup>

	<!-- 埋め込みモデル -->
	<FieldGroup label="settings.group_model_embedding">
		<div class="card-part">
			<span class="part-label">{$t('settings.model_card.model_file')}</span>
			<ComponentMigrateButton component="embedding" currentModel={String(models.embed_model ?? '')} onMigrated={loadConfig}>
				{#snippet actionsTrailing()}
					<!-- 埋め込みモデル切替後の再構築。RAG ベクトルと、URL/コマンドリコール用の
					     SemMem fact 埋め込みは別ストアだが、このボタン1つで両方まとめて再構築する。 -->
					<EmbeddingRebuildButton onRebuilt={loadConfig} />
				{/snippet}
			</ComponentMigrateButton>
		</div>
		<div class="card-part">
			<LaunchArgsField target="embedding" value={String(embedding.launch_args ?? '')} />
		</div>
		<div class="card-part">
			<!-- embed llama-server の起動/停止/再起動 (ServerName は "embed") -->
			<ModelServerControl server="embed" />
		</div>
	</FieldGroup>

	<!-- リランカー (model_state 非追跡。パスは config へ即保存) -->
	<FieldGroup label="settings.group_model_rerank">
		<div class="card-part">
			<span class="part-label">{$t('settings.model_card.model_file')}</span>
			<ComponentMigrateButton
				component="rerank"
				currentModel={String(models.rerank_model ?? '')}
				onMigrated={loadConfig}
				onApply={(p) => saveModelPathsField('rerank_model', p || null)}
			/>
		</div>
		<div class="card-part">
			<LaunchArgsField target="rerank" value={String(rerank.launch_args ?? '')} />
		</div>
		<div class="card-part">
			<ModelServerControl server="rerank" />
		</div>
	</FieldGroup>

	<!-- Pro: クリエイトモードは Pro ワークモード専用。model_state 非追跡のため切替で即 config 保存 -->
	<ProSection>
		<FieldGroup label="settings.group_model_create">
			<div class="card-part">
				<span class="part-label">{$t('settings.model_card.model_file')}</span>
				<ComponentMigrateButton component="create" currentModel={String(models.create_model ?? '')} onMigrated={loadConfig} onApply={(p) => saveModelPathsField('create_model', p || null)} />
			</div>
			<p class="card-note">{$t('settings.model_card.create_shared_note')}</p>
		</FieldGroup>
	</ProSection>
</div>

<style>
	.model-cards {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
		align-content: start;
		padding: 16px 24px;
		height: 100%;
		overflow-y: auto;
		box-sizing: border-box;
	}
	.card-part {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 10px 0;
		border-top: 0.5px solid var(--border);
	}
	.card-part:first-child {
		border-top: none;
		padding-top: 2px;
	}
	.part-label {
		font-size: 13px;
		color: var(--text-secondary);
		font-weight: 500;
	}
	.card-note {
		margin: 0;
		padding-top: 10px;
		border-top: 0.5px solid var(--border);
		font-size: 11px;
		color: var(--text-muted);
	}
	@media (max-width: 900px) {
		.model-cards {
			grid-template-columns: 1fr;
		}
	}
	@media (max-width: 767px) {
		.model-cards {
			padding: 12px 16px;
		}
	}
</style>
