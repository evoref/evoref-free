/** 添付ファイルの取り込み API (Free、f_03 §11)
 *
 * `POST /api/files/extract` は添付 1 件からテキストを抽出してチャンクを返す
 * (保存しない)。応答の `filename` / `chunks` はそのままチャット要求の
 * `file_contexts` の 1 件になる。
 */

import { requestFormData } from './_client';

/** チャット要求の `file_contexts` の 1 件 (backend `FileContext`) */
export interface FileContext {
	filename: string;
	chunks: string[];
}

/** `POST /api/files/extract` の応答 (backend `FileExtractResponse`) */
export interface FileExtractResponse {
	/** パス成分を落としたファイル名 */
	filename: string;
	chunks: string[];
	/** チャット要求の上限に合わせて後ろを落としたか */
	truncated: boolean;
	/** 抽出した本文全体の文字数 (切る前) */
	chars: number;
}

/** 添付 1 件をバックエンドで抽出してチャンクにする。失敗は `ApiError` */
export async function extractFile(file: File): Promise<FileExtractResponse> {
	const formData = new FormData();
	formData.append('file', file, file.name);
	return requestFormData<FileExtractResponse>('/files/extract', formData);
}
