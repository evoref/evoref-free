/**
 * staged クリエイトの run 再接続サービス (Phase 3b、f_05 §4.5)
 *
 * リロード / タブ閉じで SSE が切れても、事実は backend の run.json / events.jsonl
 * にある (f_10 §7) ので、SSE を再購読せず `GET /api/create/runs*` をポーリングして
 * 表示 (進捗ステップ / needs_input の問い / 成果物) を復元する。
 */
import { get } from 'svelte/store';
import { t } from '$lib/i18n';
import {
	getCreateRun,
	getCreateRunArtifacts,
	getCreateRunEvents,
	listCreateRuns
} from '$lib/free/api/createRuns';
import { eventToStep } from '$lib/free/api/sse_progress';
import { createPoller, type Poller } from '$lib/free/stores/_polling';
import { activeCreateRun, clearActiveCreateRun } from '$lib/free/stores/createRun';
import {
	addMessage,
	addStepToLastAssistant,
	isStreaming,
	messages,
	nextMessageId,
	pushGeneratedEditorCode,
	restoreSession,
	setEditorRouteToLastAssistant
} from '$lib/free/stores/chat';

const POLL_INTERVAL_MS = 3000;

let poller: Poller | null = null;

/** ページロード時 (create モード) に前回の進行中 run があれば再接続する。 */
export async function reattachCreateRun(): Promise<void> {
	const stored = get(activeCreateRun);
	if (!stored) return;

	let runs;
	try {
		runs = await listCreateRuns(stored.session_id);
	} catch {
		// 通信失敗: バックエンド未起動等の一過性。stored は残し、次回起動時に再試行する。
		return;
	}

	const run = runs.find((r) => r.run_id === stored.run_id);
	if (!run) {
		clearActiveCreateRun();
		return;
	}

	if (run.status === 'working') {
		await beginReconnect(stored.session_id, run.run_id);
		return;
	}

	if (run.status === 'needs_input') {
		showNeedsInput(run.question);
		return; // 次ターンで再開するので stored は残す
	}

	// done | failed | cancelled | timeout
	await deliverArtifacts(run.run_id);
	isStreaming.set(false);
	clearActiveCreateRun();
}

function showNeedsInput(question: string | undefined): void {
	addMessage({
		id: nextMessageId(),
		role: 'assistant',
		content: question ?? '',
		timestamp: Date.now()
	});
	isStreaming.set(false);
}

async function beginReconnect(sessionIdValue: string, runId: string): Promise<void> {
	// modeSessions / messages を stored のセッションへ合わせる (次ターンが同じ
	// セッションで再開できるように。`restoreSession` は `stores/chat.ts` の
	// 「モード別バッファと store を一緒に更新する」作法をそのまま使う)。
	restoreSession(sessionIdValue, get(messages));
	addMessage({
		id: nextMessageId(),
		role: 'assistant',
		content: get(t)('chat.create_reattached'),
		timestamp: Date.now()
	});
	isStreaming.set(true);

	let after = 0;

	const poll = async (): Promise<void> => {
		try {
			const res = await getCreateRunEvents(runId, after);
			after = res.last_seq;
			for (const ev of res.events) {
				const step = eventToStep(ev.kind, ev.payload);
				if (step) addStepToLastAssistant(step);
			}

			const current = await getCreateRun(runId);
			if (current.status === 'working') return;

			poller?.stop();
			poller = null;

			if (current.status === 'needs_input') {
				showNeedsInput(current.question);
				return;
			}

			// done | failed | cancelled | timeout
			await deliverArtifacts(runId);
			isStreaming.set(false);
			clearActiveCreateRun();
		} catch {
			// 通信失敗は次の interval で再試行 (best-effort)
		}
	};

	poller = createPoller(poll, POLL_INTERVAL_MS);
	poller.start();
}

async function deliverArtifacts(runId: string): Promise<void> {
	try {
		const res = await getCreateRunArtifacts(runId);
		for (const artifact of res.artifacts) {
			pushGeneratedEditorCode({
				filename: artifact.path,
				language: artifact.language,
				content: artifact.content
			});
		}
		if (res.artifacts.length > 0) {
			setEditorRouteToLastAssistant('editor');
		}
	} catch {
		// best-effort: 成果物取得に失敗してもチャット状態は復旧済みなので諦める
	}
}
