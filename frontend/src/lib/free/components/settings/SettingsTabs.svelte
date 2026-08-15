<script lang="ts">
	import { t } from '$lib/i18n';
	import { isPro, isDevelop } from '$lib/edition';
	import { activeTab, dirtyTabs, TAB_IDS } from '$lib/free/stores/settings';
	import { promptDirty } from '$lib/free/stores/prompts';

	const TAB_LABELS: Record<string, string> = {
		general: 'settings.tab_general',
		inference: 'settings.tab_inference',
		model: 'settings.tab_model',
		rag: 'settings.tab_rag',
		memory: 'settings.tab_memory',
		learning: 'settings.tab_learning',
		storage: 'settings.tab_storage',
		integration: 'settings.tab_integration',
		generation: 'settings.tab_generation',
		editor: 'settings.tab_editor',
		prompts: 'settings.tab_prompts',
		develop: 'settings.tab_develop'
	};

	/** Pro 機能を含むタブ (バッジ表示対象)
	 *
	 * - `editor`: エディタ機能自体が Pro ワークモード専用 (config.yaml の "Pro ワークモード" 区画 / CodeMirrorEditor 等は lib/pro/)
	 * - `learning`: level1_idle_minutes_pro / 上位オプティマイザが Pro 限定
	 * - `integration`: external_api / widget_proxy が Pro 限定
	 * - `generation`: modes.create (クリエイトモードの生成パラメータ) が Pro 限定
	 * - `prompts`: create システムプロンプトが Pro 限定
	 *
	 * ProSection を含まないタブ (memory 等) は Pro バッジ対象外。
	 */
	const PRO_TABS = new Set(['model', 'learning', 'integration', 'editor', 'generation', 'prompts']);

	/** タブ全体が Pro/Develop 専用で、Free では機能ごと存在しないタブ。
	 * Free ビルドでは visibleTabs から完全に除外する。
	 */
	const PRO_ONLY_TABS = new Set(['editor']);
	const DEVELOP_ONLY_TABS = new Set(['develop']);

	let visibleTabs = $derived(
		[...TAB_IDS, 'prompts', 'develop'].filter((tabId) => {
			if (!isPro && PRO_ONLY_TABS.has(tabId)) return false;
			if (!isDevelop && DEVELOP_ONLY_TABS.has(tabId)) return false;
			return true;
		})
	);
</script>

<nav class="tabs-nav">
	{#each visibleTabs as tabId}
		<button
			type="button"
			class="tab-btn"
			class:tab-active={$activeTab === tabId}
			onclick={() => activeTab.set(tabId)}
		>
			<span class="tab-label">
				{$t(TAB_LABELS[tabId] ?? tabId)}
				{#if isPro && PRO_TABS.has(tabId)}
					<span class="pro-badge">Pro</span>
				{/if}
			</span>
			{#if tabId === 'prompts' ? $promptDirty : $dirtyTabs[tabId]}
				<span class="tab-dirty" aria-label="unsaved"></span>
			{/if}
		</button>
	{/each}
</nav>

<style>
	.tabs-nav {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 12px 8px;
		border-right: 0.5px solid var(--border);
		background: var(--bg-secondary);
		min-width: 180px;
		overflow-y: auto;
	}

	.tab-btn {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 8px 12px;
		background: transparent;
		border: none;
		border-radius: 6px;
		color: var(--text-secondary);
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
		text-align: left;
		transition: background 0.15s;
	}

	.tab-btn:hover {
		background: color-mix(in srgb, var(--accent) 6%, transparent);
	}

	.tab-active {
		color: var(--text-primary);
		background: color-mix(in srgb, var(--accent) 22%, transparent);
		font-weight: 500;
	}

	:global([data-theme='dark']) .tab-active {
		background: color-mix(in srgb, var(--accent) 32%, transparent);
	}

	.tab-label {
		flex: 1;
	}

	.pro-badge {
		font-size: 9px;
		color: white;
		background: var(--accent);
		font-weight: 600;
		margin-left: 6px;
		padding: 1px 5px;
		border-radius: 3px;
		letter-spacing: 0.03em;
		line-height: 1.4;
		vertical-align: middle;
	}

	.tab-dirty {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--accent);
		flex-shrink: 0;
	}

	/* モバイル: 横スクロールバー */
	@media (max-width: 767px) {
		.tabs-nav {
			flex-direction: row;
			border-right: none;
			border-bottom: 0.5px solid var(--border);
			min-width: unset;
			overflow-x: auto;
			overflow-y: hidden;
			padding: 8px;
			gap: 4px;
		}

		.tab-btn {
			white-space: nowrap;
			padding: 6px 12px;
			font-size: 12px;
		}
	}
</style>
