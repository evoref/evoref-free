<script lang="ts">
	/**
	 * 検索閾値の調整ツール (埋め込みモデル切替後)
	 *
	 * - 安全調整 (A): rag.score_normalization を minmax に切替。埋め込みモデル差による
	 *   絶対スコアのズレを層内正規化で緩和する (可逆・低リスク)。
	 * - 推定 (B): 再構築済みベクトルのスコア分布から rag.* 閾値の推奨値を算出して提示。
	 *   レビューして「適用」で書き込む。**正解ラベル不在のため助言値** (自動適用しない)。
	 *
	 * いずれも reindex 後に実行するのが前提 (新しいインデックスに対してのみ意味を持つ)。
	 */
	import { t } from '$lib/i18n';
	import { loadConfig } from '$lib/free/stores/settings';
	import {
		updateConfigSection,
		calibrateThresholds,
		ApiError,
		type CalibrateThresholdsResponse
	} from '$lib/free/api';

	type Props = {
		/** 現在の rag.score_normalization 値 (A の状態表示用) */
		scoreNormalization?: string;
	};

	let { scoreNormalization = 'none' }: Props = $props();

	let busy = $state(false);
	let calib = $state<CalibrateThresholdsResponse | null>(null);
	let message = $state<string | null>(null);
	let error = $state<string | null>(null);

	function toError(e: unknown): string {
		if (e instanceof ApiError) return e.message;
		if (e instanceof Error) return e.message;
		return String(e);
	}

	async function enableNormalization() {
		if (busy) return;
		busy = true;
		error = null;
		message = null;
		try {
			await updateConfigSection('rag', { score_normalization: 'minmax' });
			await loadConfig();
			message = $t('settings.threshold.normalization_enabled');
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function runCalibrate() {
		if (busy) return;
		busy = true;
		error = null;
		message = null;
		calib = null;
		try {
			calib = await calibrateThresholds();
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}

	async function applySuggestions() {
		if (busy || !calib?.suggestions) return;
		busy = true;
		error = null;
		try {
			await updateConfigSection('rag', { ...calib.suggestions });
			await loadConfig();
			message = $t('settings.threshold.applied');
			calib = null;
		} catch (e) {
			error = toError(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="threshold-calibrate">
	<div class="hint">{$t('settings.threshold.hint')}</div>

	<div class="row">
		<button type="button" disabled={busy} onclick={enableNormalization}>
			{$t('settings.threshold.enable_normalization')}
		</button>
		<span class="state">
			{$t('settings.threshold.normalization_state', { value: scoreNormalization })}
		</span>
	</div>

	<div class="row">
		<button type="button" class="calib-btn" disabled={busy} onclick={runCalibrate}>
			{busy ? $t('settings.threshold.calibrating') : $t('settings.threshold.calibrate')}
		</button>
		<span class="note">{$t('settings.threshold.calibrate_note')}</span>
	</div>

	{#if calib && !calib.ok}
		<div class="result error">
			{calib.reason === 'insufficient_vectors'
				? $t('settings.threshold.insufficient', { n: calib.n_vectors })
				: $t('settings.threshold.no_vectors')}
		</div>
	{/if}

	{#if calib?.ok && calib.suggestions}
		<div class="result info">
			<div class="advisory">{$t('settings.threshold.advisory')}</div>
			<table>
				<tbody>
					<tr>
						<td>relevance_threshold</td>
						<td><strong>{calib.suggestions.relevance_threshold}</strong></td>
					</tr>
					<tr>
						<td>support_threshold</td>
						<td><strong>{calib.suggestions.support_threshold}</strong></td>
					</tr>
					<tr>
						<td>confidence_threshold</td>
						<td><strong>{calib.suggestions.confidence_threshold}</strong></td>
					</tr>
				</tbody>
			</table>
			{#if calib.distribution}
				<div class="dist">
					{$t('settings.threshold.distribution', {
						n: calib.n_vectors,
						matchp50: calib.distribution.match_top1_p50,
						bgp95: calib.distribution.background_p95
					})}
				</div>
			{/if}
			<div class="actions">
				<button type="button" class="apply-btn" disabled={busy} onclick={applySuggestions}>
					{$t('settings.threshold.apply')}
				</button>
				<button type="button" disabled={busy} onclick={() => (calib = null)}>
					{$t('settings.threshold.dismiss')}
				</button>
			</div>
		</div>
	{/if}

	{#if message}
		<div class="result success">{message}</div>
	{/if}
	{#if error}
		<div class="result error">{error}</div>
	{/if}
</div>

<style>
	.threshold-calibrate {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-top: 0.75rem;
		padding-top: 0.75rem;
		border-top: 1px solid var(--border-color, #ddd);
	}
	.hint {
		font-size: 0.85rem;
		color: var(--text-secondary, #555);
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	.state,
	.note {
		font-size: 0.8rem;
		color: var(--text-secondary, #555);
	}
	.advisory {
		font-size: 0.8rem;
		margin-bottom: 0.4rem;
	}
	button {
		padding: 0.4rem 0.8rem;
		border-radius: 4px;
		border: 1px solid var(--border-color, #999);
		background: var(--bg-secondary, #f5f5f5);
		color: var(--text-primary, #222);
		cursor: pointer;
		font-size: 0.9rem;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	button.calib-btn,
	button.apply-btn {
		background: var(--accent-bg, #2563eb);
		color: var(--accent-text, #fff);
		border-color: var(--accent-bg, #2563eb);
	}
	table {
		border-collapse: collapse;
		font-size: 0.85rem;
		margin: 0.25rem 0;
	}
	td {
		padding: 0.15rem 0.75rem 0.15rem 0;
	}
	.dist {
		font-size: 0.78rem;
		color: var(--text-secondary, #555);
		margin-bottom: 0.4rem;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 0.4rem;
	}
	.result {
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.85rem;
	}
	.result.info {
		background: var(--bg-info-soft, #eff6ff);
		color: var(--text-info, #1e40af);
		border: 1px solid var(--border-info, #93c5fd);
	}
	.result.success {
		background: var(--bg-success-soft, #ecfdf5);
		color: var(--text-success, #065f46);
		border: 1px solid var(--border-success, #6ee7b7);
	}
	.result.error {
		background: var(--bg-error-soft, #fef2f2);
		color: var(--text-error, #991b1b);
		border: 1px solid var(--border-error, #fca5a5);
	}
</style>
