<script lang="ts">
	/**
	 * システムタブ — 環境調整 (auto-tune) の実行・確認 (docs/c_16 §7.2.3)
	 *
	 * config を編集しないので SettingsSection (dirty / 適用) は使わない。
	 * 現在の PC・各項目の提案値と状態・実行ボタン (進捗は 1 秒間隔のポーリング)・
	 * 環境移行の確認 (AutoTuneBanner を共用) を出す。
	 */
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n';
	import type { AutoTuneItem } from '$lib/free/api';
	import AutoTuneBanner from '$lib/free/components/AutoTuneBanner.svelte';
	import { activeTab } from '$lib/free/stores/settings';
	import { refreshServerStatus } from '$lib/free/stores/server';
	import { classifyItem, formatItemValue } from '$lib/free/utils/autotune_view';
	import {
		autoTuneDetail,
		decideAndRefresh,
		refreshAutoTune,
		startAutoTune,
		watchAutoTune
	} from '$lib/free/stores/autotune';

	let starting = $state(false);

	let detail = $derived($autoTuneDetail);
	let running = $derived(starting || detail?.state === 'running');
	let progress = $derived(detail?.progress ?? null);

	onMount(async () => {
		await Promise.all([refreshAutoTune(), refreshServerStatus()]);
		if ($autoTuneDetail?.state === 'running') watchAutoTune();
	});

	async function run() {
		starting = true;
		try {
			// 確認待ち / 見送り中の移行は、実行ボタンを承認とみなして先に答える (force で確認を飛ばさない)
			const decision = $autoTuneDetail?.decision;
			if (decision === 'pending' || decision === 'declined') {
				if (!(await decideAndRefresh('accepted'))) return;
			}
			await startAutoTune();
		} finally {
			starting = false;
		}
	}

	/** 項目名: 未知のキーは生のキーのまま */
	function itemName(key: string): string {
		const k = `settings.system.item.${key}`;
		const s = $t(k);
		return s === k ? key : s;
	}

	function statusText(item: AutoTuneItem): string {
		const kind = classifyItem(item);
		const fv = formatItemValue(item.value);
		const key =
			kind === 'manual' && fv.scalar === null
				? 'settings.system.status_manual_rows'
				: `settings.system.status_${kind}`;
		return $t(key, { value: fv.scalar ?? '', reason: item.reason });
	}

	let appliedCount = $derived(detail?.items.filter((i) => i.applied).length ?? 0);
	let manualCount = $derived(detail?.items.filter((i) => !i.applied).length ?? 0);
</script>

<div class="system-settings">
	<h2 class="title">{$t('settings.system.title')}</h2>
	<p class="desc">{$t('settings.system.description')}</p>

	<AutoTuneBanner />

	{#if detail}
		<section class="block">
			<h3 class="block-title">{$t('settings.system.pc_title')}</h3>
			<dl class="pc">
				<dt>{$t('settings.system.pc_hostname')}</dt>
				<dd data-testid="pc-hostname">{detail.pc.hostname}</dd>
				<dt>{$t('settings.system.pc_cpu')}</dt>
				<dd>{detail.pc.cpu}</dd>
				<dt>{$t('settings.system.pc_cores')}</dt>
				<dd>{detail.pc.logical_cores}</dd>
				<dt>{$t('settings.system.pc_memory')}</dt>
				<dd>{detail.pc.memory_gb} GB</dd>
				<dt>{$t('settings.system.pc_gpu')}</dt>
				<dd>
					{#if detail.pc.gpus.length}
						{detail.pc.gpus.join(' / ')}
					{:else}
						{$t('settings.system.pc_gpu_none')}
					{/if}
				</dd>
			</dl>
		</section>

		<section class="block">
			<h3 class="block-title">{$t('settings.system.run_title')}</h3>
			<div class="run-row">
				<button type="button" class="run-btn" disabled={running} onclick={run}>
					{#if running}<span class="spinner" aria-hidden="true"></span>{/if}
					{running ? $t('settings.system.running') : $t('settings.system.run')}
				</button>
				{#if running && progress}
					<span class="progress" data-testid="autotune-progress">
						{progress.phase} ({progress.current}/{progress.total})
					</span>
				{/if}
			</div>
			<div class="decision" data-testid="autotune-decision">
				{$t('settings.system.decision_label')}: {$t(`autotune.decision.${detail.decision}`)}
				{#if detail.changed_axes.length}
					({detail.changed_axes.join(', ')})
				{/if}
			</div>

			{#if !running && detail.state === 'failed'}
				<div class="error-box" role="alert">
					{$t('settings.system.failed')}{detail.error ? `: ${detail.error}` : ''}
				</div>
			{/if}
			{#if !running && detail.state === 'done'}
				<div class="result-box" data-testid="autotune-done">
					{$t('settings.system.done', { applied: appliedCount, manual: manualCount })}
				</div>
			{/if}
			{#if detail.restart_required}
				<div class="restart-box" data-testid="autotune-restart">
					<span>{$t('settings.system.restart_required')}</span>
					<button type="button" class="link-btn" onclick={() => activeTab.set('model')}>
						{$t('settings.system.open_model_tab')}
					</button>
				</div>
			{/if}
		</section>

		<section class="block">
			<h3 class="block-title">{$t('settings.system.items_title')}</h3>
			{#if detail.items.length === 0}
				<p class="empty">{$t('settings.system.items_empty')}</p>
			{:else}
				<table class="items">
					<thead>
						<tr>
							<th>{$t('settings.system.col_item')}</th>
							<th>{$t('settings.system.col_value')}</th>
							<th>{$t('settings.system.col_status')}</th>
						</tr>
					</thead>
					<tbody>
						{#each detail.items as item (item.key)}
							{@const fv = formatItemValue(item.value)}
							{@const kind = classifyItem(item)}
							<tr data-testid="item-{item.key}">
								<td data-label={$t('settings.system.col_item')}>
									<div class="name">{itemName(item.key)}</div>
									<div class="keys">
										{item.key}{#if item.config_key} · {item.config_key}{/if}
									</div>
								</td>
								<td data-label={$t('settings.system.col_value')}>
									{#if fv.scalar !== null}
										<span class="num">{fv.scalar}</span>
									{:else}
										<ul class="rows">
											{#each fv.rows as row, i (i)}
												<li class:nested={row.depth === 1}>
													<span class="k">{row.label}</span>{#if row.text}<span class="v">{row.text}</span>{/if}
												</li>
											{/each}
										</ul>
									{/if}
									<span class="source">({$t(`settings.system.source_${item.source}`)})</span>
									{#if fv.manual.length}
										<div class="manual-badge" data-testid="manual-{item.key}">
											{$t('settings.system.manual_keys', { keys: fv.manual.join(', ') })}
										</div>
									{/if}
								</td>
								<td data-label={$t('settings.system.col_status')} class="status status-{kind}">
									{statusText(item)}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</section>
	{:else}
		<p class="empty">{$t('common.loading')}</p>
	{/if}
</div>

<style>
	.system-settings {
		padding: 24px;
		max-width: 860px;
	}
	.title {
		margin: 0 0 6px;
		font-size: 1.05rem;
		font-weight: 600;
		color: var(--text-primary);
	}
	.desc {
		margin: 0 0 16px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.6;
	}
	.block {
		margin-bottom: 20px;
	}
	.block-title {
		margin: 0 0 8px;
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
	}
	.pc {
		display: grid;
		grid-template-columns: 140px 1fr;
		gap: 4px 12px;
		margin: 0;
		font-size: 13px;
	}
	.pc dt {
		color: var(--text-muted);
	}
	.pc dd {
		margin: 0;
		color: var(--text-primary);
	}
	.run-row {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	.run-btn {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		background: var(--accent);
		color: var(--text-on-accent);
		border: none;
		border-radius: 6px;
		padding: 8px 16px;
		font-size: 13px;
		font-family: inherit;
		cursor: pointer;
	}
	.run-btn:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
	.progress,
	.decision {
		font-size: 12px;
		color: var(--text-muted);
	}
	.decision {
		margin-top: 8px;
	}
	.error-box {
		margin-top: 10px;
		padding: 8px 12px;
		border-radius: 4px;
		background: color-mix(in srgb, var(--color-error) 12%, transparent);
		color: var(--color-error);
		font-size: 13px;
	}
	.result-box,
	.restart-box {
		margin-top: 10px;
		padding: 8px 12px;
		border-radius: 4px;
		border: 0.5px solid var(--border);
		font-size: 13px;
		color: var(--text-secondary);
	}
	.restart-box {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	.link-btn {
		background: none;
		border: none;
		color: var(--accent);
		text-decoration: underline;
		cursor: pointer;
		font-family: inherit;
		font-size: 13px;
		padding: 0;
	}
	.items {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
		table-layout: fixed;
	}
	.items th:nth-child(1) {
		width: 24%;
	}
	.items th:nth-child(3) {
		width: 28%;
	}
	.items th,
	.items td {
		text-align: left;
		padding: 6px 8px;
		border-bottom: 0.5px solid var(--border);
		vertical-align: top;
		overflow-wrap: anywhere;
		word-break: break-word;
	}
	.items th {
		color: var(--text-muted);
		font-weight: 500;
	}
	.name {
		color: var(--text-primary);
		font-weight: 500;
	}
	.keys {
		margin-top: 2px;
		font-size: 10.5px;
		color: var(--text-muted);
		font-family: ui-monospace, monospace;
	}
	.num {
		font-family: ui-monospace, monospace;
		color: var(--text-primary);
	}
	.rows {
		list-style: none;
		margin: 0;
		padding: 0;
	}
	.rows li {
		display: flex;
		gap: 8px;
		line-height: 1.6;
	}
	.rows li.nested {
		padding-left: 14px;
	}
	.k {
		color: var(--text-muted);
		flex: 0 0 auto;
	}
	.k::after {
		content: ':';
	}
	.v {
		color: var(--text-primary);
		min-width: 0;
	}
	.manual-badge {
		display: inline-block;
		margin-top: 4px;
		padding: 1px 6px;
		border-radius: 4px;
		font-size: 11px;
		color: var(--text-secondary);
		border: 0.5px solid var(--border);
	}
	.source {
		color: var(--text-muted);
	}
	.status-failed {
		color: var(--color-error);
		font-weight: 600;
	}
	.status-scheduled,
	.status-requires_stop {
		color: var(--accent);
	}
	@media (max-width: 767px) {
		.system-settings {
			padding: 16px;
		}
		.items,
		.items tbody,
		.items tr,
		.items td {
			display: block;
			width: 100%;
		}
		.items thead {
			display: none;
		}
		.items tr {
			margin-bottom: 10px;
			border: 0.5px solid var(--border);
			border-radius: 6px;
			padding: 4px 0;
		}
		.items td {
			border-bottom: none;
			padding: 4px 10px;
		}
		.items td::before {
			content: attr(data-label);
			display: block;
			font-size: 10.5px;
			color: var(--text-muted);
		}
		.items td:first-child::before {
			display: none;
		}
	}
	.empty {
		font-size: 13px;
		color: var(--text-muted);
	}
	.spinner {
		width: 12px;
		height: 12px;
		border: 2px solid color-mix(in srgb, var(--text-on-accent) 40%, transparent);
		border-top-color: var(--text-on-accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
