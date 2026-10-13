<script lang="ts">
	/**
	 * Level 2 (成長) の承認待ちの提案 (docs/f_04 §6.8): 巻き戻しの提案と、保留中の提案の
	 * 差分・採用・却下。`showMeasure` で比較の内訳 (勝敗・ガード) を併記する (Pro の詳細向け)。
	 */
	import { t } from '$lib/i18n';
	import {
		ApiError,
		adoptGrowthProposal,
		getGrowthProposal,
		rejectGrowthProposal,
		rollbackGrowth,
		type GrowthAssetKind,
		type GrowthGate,
		type GrowthProposalDetail,
		type GrowthResponse,
		type Level1ResultEntry
	} from '$lib/free/api';

	interface GrowthProposalsProps {
		growth: GrowthResponse;
		/** 比較の内訳 (勝敗・ガード) を出す */
		showMeasure?: boolean;
		/** 採用・却下・巻き戻しの後に呼ぶ (親が再取得する) */
		onchange?: () => void;
		/** 直近の Level 1 の結果 (モード別)。提案が無いときに「なぜ無いか」を 1 行で出す */
		lastResults?: Record<string, Level1ResultEntry>;
	}

	let { growth, showMeasure = false, onchange, lastResults = {} }: GrowthProposalsProps = $props();

	const KNOWN_ERROR_REASONS = [
		'stale_parent',
		'learning_running',
		'identical_after_normalization',
		'no_rollback_target',
		'proposal_not_found',
		'data_readonly'
	];

	let busy = $state<string | null>(null);
	let errorText = $state('');
	let openId = $state<string | null>(null);
	let detail = $state<GrowthProposalDetail | null>(null);

	let suggestions = $derived(growth.adoptions.filter((a) => a.rollback_suggested));

	const RESULT_MODES = ['chat', 'create'];
	const KNOWN_LAST_REASONS = [
		'no_pairwise_gain',
		'unreachable_net_wins',
		'no_selection_pressure',
		'no_eval_cases',
		'insufficient_cases',
		'no_regression_samples',
		'sample_regression',
		'within_noise_floor',
		'candidate_identical',
		'proposal_withheld'
	];
	const KNOWN_EXCLUDED = [
		'model_origin',
		'stalled_class',
		'depends_on_context',
		'tool_context_missing',
		'tool_context_truncated',
		'shortcut_tool',
		'corpus_grounded',
		'long_form',
		'unresolved_correction'
	];

	function excludedDetail(excluded: Record<string, number>): string {
		return Object.entries(excluded)
			.filter(([, count]) => count > 0)
			.sort((a, b) => b[1] - a[1])
			.map(([key, count]) =>
				`${KNOWN_EXCLUDED.includes(key) ? $t(`dashboard.growth_last_excluded_${key}`) : key} ${count}`
			)
			.join(' / ');
	}

	/** 提案が無いときの理由 (モードごとに 1 行)。経験不足で始めなかったモードは出さない */
	let lastRuns = $derived.by(() =>
		RESULT_MODES.flatMap((mode) => {
			const result = lastResults[mode];
			if (!result?.reason || result.reason === 'insufficient_experiences') return [];
			const excluded = result.excluded ?? {};
			const excludedCount = Object.values(excluded).reduce((sum, n) => sum + n, 0);
			const reason = KNOWN_LAST_REASONS.includes(result.reason)
				? $t(`dashboard.growth_last_reason_${result.reason}`)
				: result.reason;
			return [
				$t('dashboard.growth_last_run', {
					mode,
					measured: (result.wins ?? 0) + (result.losses ?? 0) + (result.ties ?? 0),
					excluded:
						excludedCount > 0
							? $t('dashboard.growth_last_run_excluded', {
									count: excludedCount,
									detail: excludedDetail(excluded)
								})
							: '',
					reason
				})
			];
		})
	);

	function gateText(gate: GrowthGate | null): string {
		if (!gate) return '-';
		let text = $t('dashboard.growth_gate', {
			wins: gate.wins ?? 0,
			losses: gate.losses ?? 0,
			ties: gate.ties ?? 0
		});
		if (gate.guards_passed !== null || gate.guards_failed !== null) {
			text += ` ${$t('dashboard.growth_guards', {
				passed: gate.guards_passed ?? 0,
				failed: gate.guards_failed ?? 0
			})}`;
		}
		return text;
	}

	function kindLabel(kind: string): string {
		return kind === 'prompt' || kind === 'generation_params'
			? $t(`dashboard.growth_kind_${kind}`)
			: kind;
	}

	function errorMessage(err: unknown): string {
		if (err instanceof ApiError) {
			const reason = String(err.context?.reason ?? '');
			if (KNOWN_ERROR_REASONS.includes(reason)) return $t(`dashboard.growth_error_${reason}`);
			return err.message;
		}
		return err instanceof Error ? err.message : String(err);
	}

	async function run(key: string, action: () => Promise<unknown>) {
		busy = key;
		errorText = '';
		try {
			await action();
			openId = null;
			detail = null;
			onchange?.();
		} catch (err) {
			errorText = errorMessage(err);
		} finally {
			busy = null;
		}
	}

	async function toggleDiff(id: string) {
		if (openId === id) {
			openId = null;
			detail = null;
			return;
		}
		errorText = '';
		try {
			detail = await getGrowthProposal(id);
			openId = id;
		} catch (err) {
			errorText = errorMessage(err);
		}
	}

	/** 行の集合差で足した行 / 消した行を出す (順序を保つ簡易な差分) */
	let diffLines = $derived.by(() => {
		const d = detail;
		if (!d) return [] as { sign: '+' | '-'; text: string }[];
		if (d.asset_kind !== 'prompt') {
			return Object.entries(d.candidate_deltas).map(([k, v]) => ({
				sign: '+' as const,
				text: `${k}: ${d.current_deltas[k] ?? 0} → ${v}`
			}));
		}
		const current = d.current_text.split('\n');
		const candidate = d.candidate_text.split('\n');
		const currentSet = new Set(current);
		const candidateSet = new Set(candidate);
		return [
			...current.filter((l) => l.trim() && !candidateSet.has(l)).map((text) => ({ sign: '-' as const, text })),
			...candidate.filter((l) => l.trim() && !currentSet.has(l)).map((text) => ({ sign: '+' as const, text }))
		];
	});
</script>

<div class="growth-proposals">
	{#each suggestions as adoption (`${adoption.asset_kind}:${adoption.mode}`)}
		<div class="suggestion" role="alert">
			<span>
				{$t('dashboard.growth_rollback_suggested', {
					kind: kindLabel(adoption.asset_kind),
					mode: adoption.mode,
					version: adoption.adopted_version
				})}
			</span>
			<button
				type="button"
				disabled={busy !== null}
				onclick={() =>
					run(`rollback:${adoption.asset_kind}:${adoption.mode}`, () =>
						rollbackGrowth(adoption.asset_kind as GrowthAssetKind, adoption.mode)
					)}
			>
				{$t('dashboard.growth_rollback')}
			</button>
		</div>
	{/each}

	{#if growth.proposals.length === 0}
		<p class="hint-text">{$t('dashboard.growth_no_proposals')}</p>
		{#each lastRuns as line (line)}
			<p class="hint-text" data-testid="growth-last-run">{line}</p>
		{/each}
	{:else}
		<ul class="proposals">
			{#each growth.proposals as proposal (proposal.proposal_id)}
				<li class="proposal" data-testid="growth-proposal">
					<div class="proposal-head">
						<span class="proposal-kind">{kindLabel(proposal.asset_kind)} · {proposal.mode}</span>
						{#if showMeasure}
							<span class="proposal-gate">{gateText(proposal.measure)}</span>
						{/if}
						{#if proposal.stale}
							<span class="stale-badge">{$t('dashboard.growth_stale')}</span>
						{/if}
					</div>
					<div class="proposal-actions">
						<button type="button" onclick={() => toggleDiff(proposal.proposal_id)}>
							{openId === proposal.proposal_id
								? $t('dashboard.growth_hide_diff')
								: $t('dashboard.growth_show_diff')}
						</button>
						<button
							type="button"
							class="primary"
							disabled={busy !== null || proposal.stale}
							onclick={() =>
								run(`adopt:${proposal.proposal_id}`, () => adoptGrowthProposal(proposal.proposal_id))}
						>
							{$t('dashboard.growth_adopt')}
						</button>
						<button
							type="button"
							disabled={busy !== null}
							onclick={() =>
								run(`reject:${proposal.proposal_id}`, () => rejectGrowthProposal(proposal.proposal_id))}
						>
							{$t('dashboard.growth_reject')}
						</button>
					</div>
					{#if openId === proposal.proposal_id && detail}
						<pre class="diff" data-testid="growth-diff">{#each diffLines as line}<span
									class:removed={line.sign === '-'}>{line.sign} {line.text}
</span>{:else}{$t('dashboard.growth_no_diff')}{/each}</pre>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}

	{#if errorText}
		<p class="error-text" role="alert">{errorText}</p>
	{/if}
</div>

<style>
	.growth-proposals {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.hint-text {
		margin: 0;
		font-size: 0.8125rem;
		color: var(--text-secondary);
	}
	.suggestion {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		padding: 6px 8px;
		border-radius: var(--border-radius);
		font-size: 0.8125rem;
		color: var(--text-primary);
		border: 1px solid var(--color-warning);
	}
	.proposals {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.proposal {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 0.8125rem;
	}
	.proposal-head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
	}
	.proposal-kind {
		color: var(--text-primary);
		font-weight: 500;
	}
	.proposal-gate {
		color: var(--text-secondary);
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
	}
	.stale-badge {
		font-size: 0.6875rem;
		color: var(--text-secondary);
		border: 1px solid var(--border);
		padding: 0 4px;
		border-radius: 3px;
	}
	.proposal-actions {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.proposal-actions button,
	.suggestion button {
		font-size: 0.75rem;
		padding: 2px 10px;
		border-radius: 4px;
		border: 1px solid var(--border);
		background: var(--control-bg);
		color: var(--text-primary);
		cursor: pointer;
	}
	.proposal-actions button.primary {
		border-color: var(--accent);
		font-weight: 600;
	}
	button:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.diff {
		margin: 0;
		padding: 6px 8px;
		max-height: 240px;
		overflow: auto;
		font-size: 0.6875rem;
		white-space: pre-wrap;
		background-color: var(--bg-secondary);
		border-radius: 4px;
	}
	.diff .removed {
		color: var(--text-secondary);
		text-decoration: line-through;
	}
	.error-text {
		margin: 0;
		font-size: 0.8125rem;
		color: var(--color-error);
	}
</style>
