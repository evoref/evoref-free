<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import { messages, sessionId, switchMode, nextMessageId } from '$lib/free/stores/chat';
	import type { ChatMessage } from '$lib/free/stores/chat';
	import { groupByDate } from '$lib/free/utils/history';
	import type { SessionSummary, SessionDetailData } from '$lib/free/types/history';
	import {
		listHistory,
		getHistoryDetail,
		deleteHistorySession,
	} from '$lib/free/api';
	import SessionList from '$lib/free/components/history/SessionList.svelte';
	import SessionDetail from '$lib/free/components/history/SessionDetail.svelte';

	// ── 一覧の状態 ──

	let sessions = $state<SessionSummary[]>([]);
	let total = $state(0);
	let loading = $state(true);
	let error = $state('');
	let query = $state('');
	let modeFilter = $state('');
	let offset = $state(0);
	const limit = 20;

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
				{ limit, offset, mode: modeFilter || undefined, q: query || undefined },
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

	function resumeSession() {
		if (!detail) return;
		const restored: ChatMessage[] = detail.turns.map((turn) => ({
			id: nextMessageId(),
			role: turn.role as 'user' | 'assistant',
			content: turn.content,
			timestamp: turn.timestamp ? new Date(turn.timestamp).getTime() : Date.now(),
		}));
		switchMode(detail.mode);
		messages.set(restored);
		sessionId.set(detail.session_id);
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
		if (selectedId === sid) {
			selectedId = null;
			detail = null;
		}
	}

	// ── イベントハンドラ ──

	function handleSearch() {
		fetchSessions(true);
	}

	function handleModeChange() {
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
	});
</script>

<PageLayout title={$t('history_page.title')} fullHeight>
	{#snippet actions()}
		<div class="history-controls">
			<input
				type="text"
				class="search-input"
				placeholder={$t('history_page.search_placeholder')}
				bind:value={query}
				onkeydown={(e) => { if (e.key === 'Enter') handleSearch(); }}
			/>
			<select class="mode-select" bind:value={modeFilter} onchange={handleModeChange}>
				<option value="">{$t('history_page.mode_all')}</option>
				<option value="chat">{$t('sidebar.mode_chat')}</option>
				<option value="coding">{$t('sidebar.mode_coding')}</option>
			</select>
		</div>
	{/snippet}

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
</PageLayout>

<style>
	.history-controls {
		display: flex;
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
		width: 200px;
	}
	.search-input::placeholder {
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
	.history-layout {
		display: flex;
		height: 100%;
		min-height: 0;
		gap: 1px;
		background: var(--sidebar-border);
	}
</style>
