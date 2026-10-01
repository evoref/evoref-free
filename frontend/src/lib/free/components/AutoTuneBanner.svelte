<script lang="ts">
	/**
	 * 環境移行の確認バナー (docs/c_16 §7.2.3)
	 *
	 * /api/status の auto_tune.state が pending のとき、実行 / 今はしない / 変更なし
	 * (ホスト名のみの変更時だけ) を出す。[実行する] は decision=accepted → run の順に
	 * 呼び、完了まで進捗を見せてから消える。ChatInput とシステムタブの両方で使う。
	 */
	import { t } from '$lib/i18n';
	import { autoTuneStatus } from '$lib/free/stores/server';
	import {
		autoTuneDetail,
		decideAndRefresh,
		isHostnameOnly,
		startAutoTune
	} from '$lib/free/stores/autotune';

	let acting = $state(false);

	let pending = $derived($autoTuneStatus?.state === 'pending');
	let axes = $derived($autoTuneStatus?.changed_axes ?? []);
	let progress = $derived($autoTuneDetail?.progress ?? null);

	function axisLabel(axis: string): string {
		const key = `autotune.axis.${axis}`;
		const label = $t(key);
		return label === key ? axis : label;
	}

	async function accept() {
		acting = true;
		try {
			if (await decideAndRefresh('accepted')) {
				await startAutoTune();
			}
		} finally {
			acting = false;
		}
	}

	async function decline() {
		acting = true;
		try {
			await decideAndRefresh('declined');
		} finally {
			acting = false;
		}
	}

	async function unchanged() {
		acting = true;
		try {
			await decideAndRefresh('unchanged');
		} finally {
			acting = false;
		}
	}
</script>

{#if pending || acting}
	<div class="autotune-banner" role="status" data-testid="autotune-banner">
		{#if acting && !pending}
			<span class="spinner" aria-hidden="true"></span>
			<span class="banner-text">
				{$t('autotune.banner.running')}
				{#if progress}
					<span class="banner-progress">
						{progress.phase} ({progress.current}/{progress.total})
					</span>
				{/if}
			</span>
		{:else}
			<span class="banner-text">
				{$t('autotune.banner.message')}
				{#if axes.length}
					<span class="banner-axes">
						{$t('autotune.banner.changed_axes', { axes: axes.map(axisLabel).join(', ') })}
					</span>
				{/if}
			</span>
			<span class="banner-actions">
				<button type="button" class="banner-btn" disabled={acting} onclick={accept}>
					{$t('autotune.banner.accept')}
				</button>
				<button type="button" class="banner-btn" disabled={acting} onclick={decline}>
					{$t('autotune.banner.decline')}
				</button>
				{#if isHostnameOnly(axes)}
					<button type="button" class="banner-btn" disabled={acting} onclick={unchanged}>
						{$t('autotune.banner.unchanged')}
					</button>
				{/if}
			</span>
		{/if}
	</div>
{/if}

<style>
	.autotune-banner {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
		padding-bottom: 8px;
		font-size: 12px;
		color: var(--text-secondary);
	}
	.banner-axes,
	.banner-progress {
		margin-left: 6px;
		color: var(--text-muted);
	}
	.banner-actions {
		display: inline-flex;
		gap: 8px;
	}
	.banner-btn {
		font-size: 12px;
		text-decoration: underline;
		background: none;
		border: none;
		color: var(--accent);
		cursor: pointer;
		font-family: inherit;
		padding: 0;
	}
	.banner-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.spinner {
		width: 12px;
		height: 12px;
		border: 2px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
