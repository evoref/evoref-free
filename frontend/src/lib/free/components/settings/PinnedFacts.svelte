<script lang="ts">
	/**
	 * ピン留めされた記憶 (SemanticFact) の一覧と、解除 / 追加。
	 *
	 * 設定の Apply とは独立で、操作は即時に `/api/memory/{pinned,pin,unpin}` へ送る。
	 * ロック中 (`pin_locked_until` が未来) のピンは解除に 409 が返り、確認のうえ force で再試行する。
	 */
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n';
	import {
		ApiError,
		listPinnedFacts,
		pinMemoryContent,
		unpinMemoryFact,
		type PinnedFactInfo
	} from '$lib/free/api';
	import { addToast } from '$lib/free/stores/toast';
	import FieldGroup from './fields/FieldGroup.svelte';

	let facts = $state<PinnedFactInfo[]>([]);
	let loading = $state(true);
	let loadFailed = $state(false);
	let busy = $state(false);
	let newContent = $state('');

	async function reload() {
		try {
			facts = (await listPinnedFacts()).facts;
			loadFailed = false;
		} catch {
			loadFailed = true;
		} finally {
			loading = false;
		}
	}

	onMount(reload);

	function isLocked(fact: PinnedFactInfo): boolean {
		return fact.pin_locked_until !== null && fact.pin_locked_until * 1000 > Date.now();
	}

	async function add() {
		const content = newContent.trim();
		if (!content || busy) return;
		busy = true;
		try {
			await pinMemoryContent(content);
			newContent = '';
			addToast({ type: 'success', i18nKey: 'settings.memory.pins.added' });
			await reload();
		} catch {
			addToast({ type: 'error', i18nKey: 'settings.memory.pins.add_failed' });
		} finally {
			busy = false;
		}
	}

	async function unpin(fact: PinnedFactInfo) {
		if (busy) return;
		busy = true;
		try {
			try {
				await unpinMemoryFact({ fact_id: fact.id, scope: fact.scope });
			} catch (e) {
				const locked = e instanceof ApiError && e.code === 'E0409';
				if (!locked || !confirm($t('settings.memory.pins.force_confirm'))) throw e;
				await unpinMemoryFact({ fact_id: fact.id, scope: fact.scope, force: true });
			}
			addToast({ type: 'success', i18nKey: 'settings.memory.pins.removed' });
			await reload();
		} catch {
			addToast({ type: 'error', i18nKey: 'settings.memory.pins.remove_failed' });
		} finally {
			busy = false;
		}
	}
</script>

<FieldGroup label="settings.group_memory_pins" description="settings.memory.pins.note" fullWidth>
	<div class="pins" data-testid="pinned-facts">
		{#if loading}
			<p class="muted">{$t('settings.memory.pins.loading')}</p>
		{:else if loadFailed}
			<p class="muted">{$t('settings.memory.pins.load_failed')}</p>
		{:else if facts.length === 0}
			<p class="muted">{$t('settings.memory.pins.empty')}</p>
		{:else}
			<ul>
				{#each facts as fact (fact.id)}
					<li data-testid="pinned-fact">
						<div class="fact-text">
							<span class="fact-object">{fact.object}</span>
							<span class="fact-meta">{fact.subject} / {fact.predicate}</span>
							{#if isLocked(fact)}
								<span class="lock-badge">{$t('settings.memory.pins.locked')}</span>
							{/if}
						</div>
						<button type="button" class="unpin-btn" disabled={busy} onclick={() => unpin(fact)}>
							{$t('settings.memory.pins.unpin')}
						</button>
					</li>
				{/each}
			</ul>
		{/if}

		<div class="add-row">
			<input
				type="text"
				class="add-input"
				placeholder={$t('settings.memory.pins.add_placeholder')}
				bind:value={newContent}
				disabled={busy}
				onkeydown={(e) => e.key === 'Enter' && add()}
			/>
			<button type="button" class="add-btn" disabled={busy || !newContent.trim()} onclick={add}>
				{$t('settings.memory.pins.add')}
			</button>
		</div>
	</div>
</FieldGroup>

<style>
	.pins {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	li {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 6px 10px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
	}
	.fact-text {
		display: flex;
		flex-direction: column;
		gap: 2px;
		min-width: 0;
	}
	.fact-object {
		font-size: 13px;
		color: var(--text-primary);
		overflow-wrap: anywhere;
	}
	.fact-meta,
	.muted {
		font-size: 11px;
		color: var(--text-secondary);
		margin: 0;
	}
	.lock-badge {
		font-size: 11px;
		color: var(--color-warning, var(--accent));
	}
	.add-row {
		display: flex;
		gap: 8px;
	}
	.add-input {
		flex: 1;
		padding: 6px 10px;
		background: var(--control-bg);
		color: var(--text-primary);
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		font-size: 13px;
		font-family: inherit;
	}
	.add-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	button {
		padding: 6px 12px;
		border-radius: 4px;
		border: 0.5px solid var(--input-border);
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 13px;
		cursor: pointer;
		white-space: nowrap;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
