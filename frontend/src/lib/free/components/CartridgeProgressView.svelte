<script lang="ts" module>
	/**
	 * カートリッジ作成・インストール時の進捗表示コンポーネントの型定義
	 *
	 * `<script module>` ブロックで型エクスポートする (Svelte 5 仕様)。
	 */
	export interface PhaseSpec {
		id: string;
		labelKey: string;
	}

	export interface ProgressState {
		/** 進行中のフェーズ ID。null = 未開始 or 完了 */
		currentPhase: string | null;
		/** 完了済みフェーズの ID 集合 */
		completedPhases: Set<string>;
		/** 現在のループ進捗 */
		current: number;
		/** ループ総数 (0 = 件数なし、フェーズ単位) */
		total: number;
		/** 現在処理中の項目名 (ファイル名など) */
		detail: string;
	}
</script>

<script lang="ts">
	/**
	 * カートリッジ作成・インストール時の進捗表示コンポーネント
	 *
	 * SSE バックエンドから流れてくる step フレームを集約した状態を受け取り、
	 * 各フェーズの状態 (待機 / 実行中 / 完了) を行単位で表示する。
	 * ループ系フェーズ (current/total が指定されたフェーズ) ではプログレスバー
	 * と件数を併記する。
	 *
	 * Free install / Pro create の両方で再利用される。
	 */
	import { t } from '$lib/i18n';

	interface Props {
		phases: PhaseSpec[];
		state: ProgressState;
		cancelling: boolean;
		onCancel?: () => void;
	}

	let { phases, state, cancelling, onCancel }: Props = $props();

	function statusOf(phaseId: string): 'waiting' | 'running' | 'done' {
		if (state.completedPhases.has(phaseId)) return 'done';
		if (state.currentPhase === phaseId) return 'running';
		return 'waiting';
	}

	function progressPct(): number {
		if (state.total <= 0) return 0;
		return Math.min(100, Math.round((state.current / state.total) * 100));
	}
</script>

<div class="progress-view">
	<ul class="phase-list">
		{#each phases as phase}
			{@const status = statusOf(phase.id)}
			<li class="phase-row" class:running={status === 'running'} class:done={status === 'done'}>
				<span class="phase-icon" aria-hidden="true">
					{#if status === 'done'}✓{:else if status === 'running'}●{:else}○{/if}
				</span>
				<span class="phase-label">{$t(phase.labelKey)}</span>
				<span class="phase-status">
					{#if status === 'done'}
						{$t('cartridge.progress.done')}
					{:else if status === 'running'}
						{#if state.total > 0}
							<span class="counter">{state.current} / {state.total}</span>
							<span class="bar" aria-hidden="true">
								<span class="bar-fill" style:width="{progressPct()}%"></span>
							</span>
						{:else}
							{$t('cartridge.progress.running')}
						{/if}
					{:else}
						{$t('cartridge.progress.waiting')}
					{/if}
				</span>
				{#if status === 'running' && state.detail}
					<span class="phase-detail">→ {state.detail}</span>
				{/if}
			</li>
		{/each}
	</ul>

	{#if onCancel}
		<div class="actions">
			<button class="btn-cancel" onclick={onCancel} disabled={cancelling}>
				{cancelling ? $t('cartridge.progress.cancelling') : $t('cartridge.progress.cancel')}
			</button>
		</div>
	{/if}
</div>

<style>
	.progress-view {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.phase-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.phase-row {
		display: grid;
		grid-template-columns: 18px 1fr auto;
		gap: 8px;
		align-items: center;
		padding: 8px 10px;
		border-radius: var(--border-radius);
		background-color: var(--bg-secondary);
		font-size: 0.92rem;
		color: var(--text-secondary);
	}
	.phase-row.running {
		color: var(--text-primary);
		border: 1px solid var(--accent);
	}
	.phase-row.done {
		color: var(--text-primary);
		opacity: 0.7;
	}
	.phase-icon {
		font-size: 0.95rem;
		text-align: center;
		color: var(--accent);
	}
	.phase-row.running .phase-icon {
		animation: pulse 1.2s ease-in-out infinite;
	}
	.phase-row.done .phase-icon {
		color: var(--color-success, var(--accent));
	}
	.phase-label {
		font-weight: 500;
	}
	.phase-status {
		display: flex;
		align-items: center;
		gap: 8px;
		font-variant-numeric: tabular-nums;
		font-size: 0.85rem;
		min-width: 0;
	}
	.counter {
		white-space: nowrap;
	}
	.bar {
		display: inline-block;
		width: 90px;
		height: 6px;
		background-color: var(--border);
		border-radius: 3px;
		overflow: hidden;
	}
	.bar-fill {
		display: block;
		height: 100%;
		background-color: var(--accent);
		transition: width 0.18s ease;
	}
	.phase-detail {
		grid-column: 2 / -1;
		font-size: 0.8rem;
		color: var(--text-secondary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		opacity: 0.85;
	}
	.actions {
		display: flex;
		justify-content: flex-end;
	}
	.btn-cancel {
		padding: 6px 14px;
		background-color: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--border);
		border-radius: var(--border-radius);
		font-size: 0.9rem;
		cursor: pointer;
	}
	.btn-cancel:hover:not(:disabled) {
		opacity: 0.85;
	}
	.btn-cancel:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 0.5;
		}
		50% {
			opacity: 1;
		}
	}
</style>
