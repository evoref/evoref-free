import { writable } from 'svelte/store';

export type ToastType = 'error' | 'success' | 'warning' | 'info';

export interface Toast {
	id: string;
	type: ToastType;
	i18nKey: string;
	params?: Record<string, string | number>;
	duration?: number;
}

const DEFAULT_DURATION = 5000;

/** トースト通知のリスト */
export const toasts = writable<Toast[]>([]);

/** トースト通知を追加（duration ms 後に自動削除、0 で永続表示） */
export function addToast(toast: Omit<Toast, 'id'>): void {
	const id = crypto.randomUUID();
	const duration = toast.duration ?? DEFAULT_DURATION;

	toasts.update((t) => [...t, { ...toast, id }]);

	if (duration > 0) {
		setTimeout(() => removeToast(id), duration);
	}
}

/** トースト通知を削除 */
export function removeToast(id: string): void {
	toasts.update((t) => t.filter((toast) => toast.id !== id));
}

/** 全トーストをクリア */
export function clearToasts(): void {
	toasts.set([]);
}
