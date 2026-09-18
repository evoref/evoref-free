/**
 * staged クリエイトの進行中 run を viewer 単位で覚えておくストア (再接続用、f_05 §4.5)。
 *
 * `localStorage['evoref.create.session']` へ永続する。読み書きは try/catch
 * (private window 等で失敗しても動作は変えない、`corpusMode` と同じ作法)。
 * `create_run` SSE フレーム受信時に set し、run が終端 (`done|failed|cancelled|timeout`)
 * に達したら clear する。`needs_input` は次ターンで再開するため clear しない。
 */
import { writable } from 'svelte/store';

export interface ActiveCreateRun {
	session_id: string;
	run_id: string;
	started_at: string;
}

const STORAGE_KEY = 'evoref.create.session';

function loadActiveCreateRun(): ActiveCreateRun | null {
	try {
		const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
		if (!raw) return null;
		const parsed = JSON.parse(raw);
		if (
			parsed &&
			typeof parsed.session_id === 'string' &&
			typeof parsed.run_id === 'string' &&
			typeof parsed.started_at === 'string'
		) {
			return parsed as ActiveCreateRun;
		}
		return null;
	} catch {
		return null;
	}
}

export const activeCreateRun = writable<ActiveCreateRun | null>(loadActiveCreateRun());

activeCreateRun.subscribe((v) => {
	try {
		if (typeof localStorage === 'undefined') return;
		if (v) {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(v));
		} else {
			localStorage.removeItem(STORAGE_KEY);
		}
	} catch {
		/* private window 等: 保持できなくても動作は変えない */
	}
});

/** `create_run` SSE フレーム受信時に呼ぶ */
export function setActiveCreateRun(run: ActiveCreateRun): void {
	activeCreateRun.set(run);
}

/** run が終端に達した (再接続の必要が無くなった) ときに呼ぶ */
export function clearActiveCreateRun(): void {
	activeCreateRun.set(null);
}
