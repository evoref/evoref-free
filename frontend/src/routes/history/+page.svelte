<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import { restoreSession, switchMode, nextMessageId } from '$lib/free/stores/chat';
	import type { ChatMessage } from '$lib/free/stores/chat';
	import { isPro } from '$lib/edition';
	import { proActive } from '$lib/free/stores/server';
	import { groupByDate } from '$lib/free/utils/history';
	import type { SessionSummary, SessionDetailData } from '$lib/free/types/history';
	import {
		listHistory,
		getHistoryDetail,
		deleteHistorySession,
		getHistoryStats,
		compactHistory,
		batchDeleteHistory,
		listHistorySessionIds,
	} from '$lib/free/api';
	import type { HistoryStats } from '$lib/free/api';
	import { addToast } from '$lib/free/stores/toast';
	import SessionList from '$lib/free/components/history/SessionList.svelte';
	import SessionDetail from '$lib/free/components/history/SessionDetail.svelte';
	import HistoryStatsBar from '$lib/free/components/history/HistoryStatsBar.svelte';
	import HistoryConfirmDialog from '$lib/free/components/history/HistoryConfirmDialog.svelte';

	/** 履歴一覧の 1 ページあたり件数 */
	const HISTORY_PAGE_SIZE = 20;

	// ── 一覧の状態 ──

	let sessions = $state<SessionSummary[]>([]);
	let total = $state(0);
	let loading = $state(true);
	let error = $state('');
	let query = $state('');
	let modeFilter = $state('');
	let dateFrom = $state('');
	let dateTo = $state('');
	let offset = $state(0);
	const limit = HISTORY_PAGE_SIZE;

	// ── 統計 / 圧縮 / 一括削除の状態 ──

	let stats = $state<HistoryStats | null>(null);
	/** 開いている確認ダイアログ (null = 閉じている) */
	let confirming = $state<'compact' | 'bulk_delete' | null>(null);
	let actionBusy = $state(false);

	// ── 詳細パネルの状態 ──

	let selectedId = $state<string | null>(null);
	let detail = $state<SessionDetailData | null>(null);
	let detailLoading = $state(false);
	let detailError = $state('');

	// ── AbortController でレースコンディション対策 ──

	let fetchAbort: AbortController | null = null;

	async function fetchSessions(reset = false) {
		fetchAbort?.abort();
		fetchAbort = new AbortController();
		const signal = fetchAbort.signal;

		if (reset) {
			offset = 0;
			sessions = [];
		}
		loading = true;
		error = '';
		try {
			const data = await listHistory(
				{
					limit,
					offset,
					mode: modeFilter || undefined,
					from: dateFrom || undefined,
					to: dateTo || undefined,
					q: query || undefined,
				},
				signal
			);
			total = data.total;
			if (reset) {
				sessions = data.sessions;
			} else {
				sessions = [...sessions, ...data.sessions];
			}
		} catch (e) {
			if (e instanceof DOMException && e.name === 'AbortError') return;
			error = 'load_failed';
		} finally {
			loading = false;
		}
	}

	async function fetchStats() {
		try {
			stats = await getHistoryStats();
		} catch {
			// 統計は補助表示。取れなくても一覧は使える
			stats = null;
		}
	}

	// ── 詳細 fetch ──

	async function selectSession(id: string) {
		if (selectedId === id) return;
		selectedId = id;
		detail = null;
		detailLoading = true;
		detailError = '';
		try {
			detail = await getHistoryDetail(id);
		} catch {
			detailError = 'detail_load_failed';
		} finally {
			detailLoading = false;
		}
	}

	// ── 続きから再開 ──

	async function resumeSession() {
		if (!detail) return;
		const restored: ChatMessage[] = detail.turns.map((turn) => ({
			id: nextMessageId(),
			role: turn.role as 'user' | 'assistant',
			content: turn.content,
			timestamp: turn.timestamp ? new Date(turn.timestamp).getTime() : Date.now(),
		}));
		// Free では create セッションを chat に丸める (Sidebar の露出方針に合わせる)。
		// switchMode は await 後にモード別バッファで messages/sessionId を上書き
		// するため、復元値が勝つよう先に await してから復元する (モード別
		// バッファも一緒に更新しないと次のモード往復で復元前の状態に戻る)。
		await switchMode(isPro ? detail.mode : 'chat');
		restoreSession(detail.session_id, restored);
		goto('/');
	}

	// ── セッション削除 ──

	async function deleteSession(e: MouseEvent, sid: string) {
		e.stopPropagation();
		if (!confirm($t('history_page.delete_confirm'))) return;
		try {
			await deleteHistorySession(sid);
		} catch {
			return;
		}
		sessions = sessions.filter(s => s.session_id !== sid);
		total = Math.max(0, total - 1);
		fetchStats();
		if (selectedId === sid) {
			selectedId = null;
			detail = null;
		}
	}

	// ── 圧縮 / 一括削除 (確認ダイアログの「実行」) ──

	async function runCompact() {
		actionBusy = true;
		try {
			const r = await compactHistory();
			addToast({
				type: 'success',
				i18nKey: 'history_page.compact_done',
				params: { compressed: r.compressed, summarized: r.summarized, deleted: r.deleted, freed: r.freed_mb },
			});
			await Promise.all([fetchSessions(true), fetchStats()]);
		} catch {
			addToast({ type: 'error', i18nKey: 'history_page.compact_failed' });
		} finally {
			actionBusy = false;
			confirming = null;
		}
	}

	async function runBulkDelete() {
		actionBusy = true;
		try {
			const ids = await listHistorySessionIds({
				mode: modeFilter || undefined,
				from: dateFrom || undefined,
				to: dateTo || undefined,
				q: query || undefined,
			});
			const deleted = ids.length > 0 ? await batchDeleteHistory(ids) : 0;
			addToast({ type: 'success', i18nKey: 'history_page.bulk_deleted', params: { count: deleted } });
			selectedId = null;
			detail = null;
			await Promise.all([fetchSessions(true), fetchStats()]);
		} catch {
			addToast({ type: 'error', i18nKey: 'history_page.bulk_delete_failed' });
		} finally {
			actionBusy = false;
			confirming = null;
		}
	}

	// ── イベントハンドラ ──

	function handleSearch() {
		fetchSessions(true);
	}

	function handleFilterChange() {
		fetchSessions(true);
	}

	function loadMore() {
		offset += limit;
		fetchSessions(false);
	}

	// ── 派生状態 ──

	let dateGroups = $derived(groupByDate(sessions));
	let hasMore = $derived(sessions.length < total);

	onMount(() => {
		fetchSessions(true);
		fetchStats();
	});
</script>

<PageLayout title={$t('history_page.title')} fullHeight>
	{#snippet actions()}
		<div class="history-controls">
			<input
				type="text"
				class="search-input"
				placeholder={$t('history_page.search_placeholder')}
				aria-label={$t('history_page.search_placeholder')}
				bind:value={query}
				onkeydown={(e) => { if (e.key === 'Enter') handleSearch(); }}
			/>
			<select class="mode-select" aria-label={$t('sidebar.mode')} bind:value={modeFilter} onchange={handleFilterChange}>
				<option value="">{$t('history_page.mode_all')}</option>
				<option value="chat">{$t('sidebar.mode_chat')}</option>
				{#if $proActive}
					<option value="create">{$t('sidebar.mode_create')}</option>
				{/if}
			</select>
			<input
				type="date"
				class="date-input"
				aria-label={$t('history_page.date_from')}
				bind:value={dateFrom}
				max={dateTo || undefined}
				onchange={handleFilterChange}
			/>
			<span class="date-sep">–</span>
			<input
				type="date"
				class="date-input"
				aria-label={$t('history_page.date_to')}
				bind:value={dateTo}
				min={dateFrom || undefined}
				onchange={handleFilterChange}
			/>
		</div>
	{/snippet}

	<div class="history-body">
		{#if stats}
			<HistoryStatsBar
				{stats}
				matchedTotal={total}
				oncompact={() => (confirming = 'compact')}
				onbulkdelete={() => (confirming = 'bulk_delete')}
			/>
		{/if}

		<div class="history-layout">
			<SessionList
				{dateGroups}
				{loading}
				{error}
				sessionsEmpty={sessions.length === 0}
				{hasMore}
				{selectedId}
				{query}
				onselect={selectSession}
				ondelete={deleteSession}
				onloadmore={loadMore}
			/>
			<SessionDetail
				{selectedId}
				{detail}
				{detailLoading}
				{detailError}
				onresume={resumeSession}
			/>
		</div>
	</div>
</PageLayout>

{#if confirming === 'compact'}
	<HistoryConfirmDialog
		title={$t('history_page.compact_confirm_title')}
		message={$t('history_page.compact_confirm_message')}
		confirmLabel={$t('history_page.compact')}
		busy={actionBusy}
		onConfirm={runCompact}
		onCancel={() => (confirming = null)}
	/>
{:else if confirming === 'bulk_delete'}
	<HistoryConfirmDialog
		title={$t('history_page.bulk_delete_confirm_title')}
		message={$t('history_page.bulk_delete_confirm_message', { count: total })}
		confirmLabel={$t('common.delete')}
		busy={actionBusy}
		onConfirm={runBulkDelete}
		onCancel={() => (confirming = null)}
	/>
{/if}

<style>
	.history-controls {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.search-input {
		padding: 6px 10px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 14px;
		font-family: inherit;
		width: min(200px, 100%);
	}
	.search-input::placeholder {
		color: var(--text-secondary);
	}
	.date-input {
		padding: 5px 8px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 14px;
		font-family: inherit;
	}
	.date-sep {
		color: var(--text-secondary);
	}
	.mode-select {
		padding: 6px 8px;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		background: var(--control-bg);
		color: var(--text-primary);
		font-size: 14px;
		font-family: inherit;
	}
	.history-body {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.history-layout {
		flex: 1;
		display: flex;
		height: 100%;
		min-height: 0;
		gap: 1px;
		background: var(--sidebar-border);
	}
	@media (max-width: 767px) {
		.history-layout {
			flex-direction: column;
			overflow-y: auto;
		}
	}
</style>
