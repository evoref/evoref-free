/**
 * ポーリングライフサイクル共通化ヘルパ。
 *
 * `server` / `vram` ストアで重複していた `setInterval` / `clearInterval`
 * 管理 (start: 既存停止 → 即時 refresh → 定期実行 / stop: クリア) を集約する。
 * 各ストアは公開関数 (`startPolling` / `stopPolling` 等) のシグネチャを保ったまま
 * 内部実装を委譲する。
 */

export interface Poller {
	/** 既存ポーリングを止めて即時 refresh し、`intervalMs` 間隔で再開する。 */
	start: (intervalMs?: number) => void;
	/** ポーリングを停止する (未開始でも安全)。 */
	stop: () => void;
}

/**
 * `refreshFn` を `defaultIntervalMs` 間隔でポーリングする {@link Poller} を生成する。
 */
export function createPoller(
	refreshFn: () => void | Promise<void>,
	defaultIntervalMs: number
): Poller {
	let intervalId: ReturnType<typeof setInterval> | null = null;

	const stop = (): void => {
		if (intervalId !== null) {
			clearInterval(intervalId);
			intervalId = null;
		}
	};

	const start = (intervalMs: number = defaultIntervalMs): void => {
		stop();
		refreshFn();
		intervalId = setInterval(refreshFn, intervalMs);
	};

	return { start, stop };
}
