/**
 * 自律ループイベント観察 store
 *
 * バックエンド `/api/loop/stream` を購読してリアルタイムに
 * `LoopEvent` を受信し、最大 500 件のリングバッファに保持する。
 *
 * 設計原則:
 * - EventSource を 1 本のみ保持し、多重接続を防止
 * - 切断時は指数バックオフ (初回 1s → 最大 30s) で自動再接続
 * - pause 中はバッファ追記を停止 (受信そのものは破棄、再開後の連続性は保証しない)
 * - disconnect は意図的切断として再接続を抑止
 * - バックエンド `LoopEvent.as_dict()`
 *   ([backend/free/loop/events.py:65](../../../../../../backend/free/loop/events.py#L65))
 *   とフィールドを一致させる
 */
import { writable } from 'svelte/store';

export type LoopEventKind =
	| 'task_picked'
	| 'action_executed'
	| 'gate_result'
	| 'fact_written'
	| 'iteration_started'
	| 'iteration_ended'
	| 'loop_started'
	| 'loop_paused'
	| 'loop_resumed'
	| 'loop_stopped';

/** SSE で送信される `LoopEvent` の TS 表現 (バックエンドの `as_dict()` と対応)。 */
export interface LoopEvent {
	event: LoopEventKind;
	trace_id: string;
	timestamp: number;
	iteration: number;
	project_id: string | null;
	data: Record<string, unknown>;
	/** クライアント側で振る一意 ID (Svelte の {#each key} 用)。 */
	clientId: string;
}

/** 観察 UI 用の接続状態。 */
export type ConnectionStatus =
	| 'idle'
	| 'connecting'
	| 'open'
	| 'reconnecting'
	| 'closed'
	| 'paused';

export const LOOP_EVENT_KINDS: readonly LoopEventKind[] = [
	'task_picked',
	'action_executed',
	'gate_result',
	'fact_written',
	'iteration_started',
	'iteration_ended',
	'loop_started',
	'loop_paused',
	'loop_resumed',
	'loop_stopped'
] as const;

export const MAX_BUFFER_SIZE = 500;
export const INITIAL_BACKOFF_MS = 1_000;
export const MAX_BACKOFF_MS = 30_000;
export const STREAM_URL = '/api/loop/stream';

/** リングバッファ (最古→最新)。 */
export const events = writable<LoopEvent[]>([]);

/** 現在の接続状態 (UI 表示用)。 */
export const connectionStatus = writable<ConnectionStatus>('idle');

/** 連続再接続試行回数 (UI 表示・テスト用)。openOnce で 0 にリセット。 */
export const reconnectAttempt = writable<number>(0);

/** 構造化されていない / pause 中に受信したイベントの破棄カウント (テスト用)。 */
export const droppedCount = writable<number>(0);

let source: EventSource | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let backoffMs = INITIAL_BACKOFF_MS;
let paused = false;
let intentionallyClosed = false;
let currentUrl: string = STREAM_URL;

/**
 * EventSource ファクトリ。テストから差し替え可能。
 *
 * jsdom 環境では `EventSource` が未定義のため、テストでは差し替えて使う。
 */
let eventSourceFactory: (url: string) => EventSource = (url) => new EventSource(url);

export function _setEventSourceFactory(
	factory: (url: string) => EventSource
): void {
	eventSourceFactory = factory;
}

export function _resetEventSourceFactory(): void {
	eventSourceFactory = (url) => new EventSource(url);
}

function clearReconnectTimer(): void {
	if (reconnectTimer !== null) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
}

function closeSource(): void {
	if (source !== null) {
		try {
			source.close();
		} catch {
			// ignore
		}
		source = null;
	}
}

function parseEvent(raw: string): LoopEvent | null {
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch {
		return null;
	}
	if (typeof parsed !== 'object' || parsed === null) return null;
	const obj = parsed as Record<string, unknown>;
	const kind = obj.event;
	if (
		typeof kind !== 'string' ||
		!LOOP_EVENT_KINDS.includes(kind as LoopEventKind)
	) {
		return null;
	}
	const data = obj.data;
	return {
		event: kind as LoopEventKind,
		trace_id: typeof obj.trace_id === 'string' ? obj.trace_id : '',
		timestamp: typeof obj.timestamp === 'number' ? obj.timestamp : 0,
		iteration: typeof obj.iteration === 'number' ? obj.iteration : 0,
		project_id: typeof obj.project_id === 'string' ? obj.project_id : null,
		data:
			typeof data === 'object' && data !== null
				? (data as Record<string, unknown>)
				: {},
		clientId:
			typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
				? crypto.randomUUID()
				: `${Date.now()}-${Math.random().toString(36).slice(2)}`
	};
}

function appendEvent(evt: LoopEvent): void {
	events.update((list) => {
		const startIdx =
			list.length >= MAX_BUFFER_SIZE ? list.length - MAX_BUFFER_SIZE + 1 : 0;
		const next = list.slice(startIdx);
		next.push(evt);
		return next;
	});
}

function handleMessageEvent(e: MessageEvent): void {
	if (paused) {
		droppedCount.update((n) => n + 1);
		return;
	}
	if (typeof e.data !== 'string') return;
	const evt = parseEvent(e.data);
	if (evt === null) {
		droppedCount.update((n) => n + 1);
		return;
	}
	appendEvent(evt);
}

function attachListeners(es: EventSource): void {
	es.onopen = () => {
		backoffMs = INITIAL_BACKOFF_MS;
		reconnectAttempt.set(0);
		connectionStatus.set('open');
	};
	// バックエンドは `event: <kind>\ndata: ...` 形式で named event を送るため、
	// onmessage では拾えず addEventListener 必須。keepalive コメント (`: ...`) は
	// EventSource 側で破棄される。
	for (const kind of LOOP_EVENT_KINDS) {
		es.addEventListener(kind, (e) => handleMessageEvent(e as MessageEvent));
	}
	// 念のため unnamed event もハンドル (将来の互換性向け)。
	es.onmessage = (e) => handleMessageEvent(e);
	es.onerror = () => {
		closeSource();
		if (intentionallyClosed) return;
		scheduleReconnect();
	};
}

function openOnce(): void {
	closeSource();
	connectionStatus.set('connecting');
	let es: EventSource;
	try {
		es = eventSourceFactory(currentUrl);
	} catch {
		scheduleReconnect();
		return;
	}
	source = es;
	attachListeners(es);
}

function scheduleReconnect(): void {
	if (reconnectTimer !== null) return;
	if (intentionallyClosed) return;
	connectionStatus.set('reconnecting');
	const delay = backoffMs;
	reconnectAttempt.update((n) => n + 1);
	reconnectTimer = setTimeout(() => {
		reconnectTimer = null;
		if (intentionallyClosed || paused) {
			if (paused) connectionStatus.set('paused');
			return;
		}
		backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS);
		openOnce();
	}, delay);
}

/** SSE 接続を開始する。多重呼び出しは idempotent。 */
export function connect(url: string = STREAM_URL): void {
	currentUrl = url;
	intentionallyClosed = false;
	if (paused) {
		connectionStatus.set('paused');
		return;
	}
	if (source !== null || reconnectTimer !== null) return;
	openOnce();
}

/** 意図的切断。再接続は走らない。 */
export function disconnect(): void {
	intentionallyClosed = true;
	clearReconnectTimer();
	closeSource();
	connectionStatus.set('closed');
}

/** バッファをクリアする (接続状態は維持)。 */
export function clearEvents(): void {
	events.set([]);
}

/**
 * 受信を一時停止する。SSE 接続自体は維持し、到来するイベントは破棄する。
 *
 * バックエンド側の publish はキュー満杯時に最古を捨てる drop-oldest なので、
 * pause 中もバックエンドの周回はブロックされない。
 */
export function pause(): void {
	if (paused) return;
	paused = true;
	connectionStatus.set('paused');
}

/** 受信を再開する。切断中なら再接続を試行する。 */
export function resume(): void {
	if (!paused) return;
	paused = false;
	if (intentionallyClosed) {
		// disconnect 済みなら resume では繋ぎ直さない (明示的な connect を要求)
		return;
	}
	if (source === null && reconnectTimer === null) {
		openOnce();
	} else if (source !== null) {
		connectionStatus.set('open');
	}
}

/** 現在の接続を強制リセットして再接続をトリガする (UI の「再接続」ボタン用)。 */
export function reconnectNow(): void {
	clearReconnectTimer();
	closeSource();
	intentionallyClosed = false;
	backoffMs = INITIAL_BACKOFF_MS;
	reconnectAttempt.set(0);
	if (paused) {
		connectionStatus.set('paused');
		return;
	}
	openOnce();
}

/** テストフック: 全状態をリセットする。 */
export function _resetForTest(): void {
	clearReconnectTimer();
	closeSource();
	paused = false;
	intentionallyClosed = false;
	backoffMs = INITIAL_BACKOFF_MS;
	currentUrl = STREAM_URL;
	events.set([]);
	connectionStatus.set('idle');
	reconnectAttempt.set(0);
	droppedCount.set(0);
}

/** テストフック: 内部状態を読み取る。 */
export function _peekInternal(): {
	hasSource: boolean;
	hasReconnectTimer: boolean;
	backoffMs: number;
	paused: boolean;
	intentionallyClosed: boolean;
	currentUrl: string;
} {
	return {
		hasSource: source !== null,
		hasReconnectTimer: reconnectTimer !== null,
		backoffMs,
		paused,
		intentionallyClosed,
		currentUrl
	};
}
