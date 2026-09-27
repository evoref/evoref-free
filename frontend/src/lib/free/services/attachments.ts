/**
 * 添付ファイルをチャット要求の `file_contexts` にする (f_03 §11)
 *
 * 添付ごとにバックエンドで抽出し、取り込めたものだけを返す。失敗したファイルは
 * 名前付きのトーストで知らせ、添付したことにしない。全添付の合計がチャット要求の
 * 上限を超えないよう、後ろのファイルのチャンクから落とす。
 */

import { ApiError, extractFile, type FileContext, type FileExtractResponse } from '$lib/free/api';
import { addToast } from '$lib/free/stores/toast';
import { FILE_CONTEXT_MAX_TOTAL_CHARS, FILE_CONTEXT_MAX_TOTAL_CHUNKS } from '$lib/free/constants';

export interface PreparedAttachments {
	/** 取り込めた添付の名前 (発言バブルの 📎 表示用) */
	names: string[];
	fileContexts: FileContext[];
}

/** トーストの差し込み値に使える値だけを残す */
function toastParams(context: Record<string, unknown>): Record<string, string | number> {
	const params: Record<string, string | number> = {};
	for (const [key, value] of Object.entries(context)) {
		if (typeof value === 'string' || typeof value === 'number') params[key] = value;
	}
	return params;
}

function notifyFailure(name: string, error: unknown): void {
	if (error instanceof ApiError && error.i18nKey) {
		addToast({
			type: 'error',
			i18nKey: error.i18nKey,
			params: { ...toastParams(error.context), filename: name }
		});
	} else {
		addToast({ type: 'error', i18nKey: 'file.attach_failed', params: { filename: name } });
	}
	console.error('[Attachment Error]', name, error);
}

/** 添付を抽出し、`file_contexts` と 📎 表示用の名前にする */
export async function prepareAttachments(files: File[]): Promise<PreparedAttachments> {
	const results = await Promise.allSettled(files.map((file) => extractFile(file)));
	const names: string[] = [];
	const fileContexts: FileContext[] = [];
	let usedChunks = 0;
	let usedChars = 0;

	results.forEach((result, i) => {
		const name = files[i].name;
		if (result.status === 'rejected') {
			notifyFailure(name, result.reason);
			return;
		}
		const extracted: FileExtractResponse = result.value;
		const kept: string[] = [];
		for (const chunk of extracted.chunks) {
			if (
				usedChunks + 1 > FILE_CONTEXT_MAX_TOTAL_CHUNKS ||
				usedChars + chunk.length > FILE_CONTEXT_MAX_TOTAL_CHARS
			) {
				break;
			}
			kept.push(chunk);
			usedChunks += 1;
			usedChars += chunk.length;
		}
		if (kept.length === 0) {
			addToast({ type: 'error', i18nKey: 'file.attach_over_budget', params: { filename: name } });
			return;
		}
		if (extracted.truncated || kept.length < extracted.chunks.length) {
			addToast({ type: 'warning', i18nKey: 'file.attach_truncated', params: { filename: name } });
		}
		names.push(name);
		fileContexts.push({ filename: extracted.filename, chunks: kept });
	});

	return { names, fileContexts };
}
