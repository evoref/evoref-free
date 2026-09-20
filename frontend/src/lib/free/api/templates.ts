/** 文書テンプレート API (c_16 §4.5.2)
 *
 * `GET /api/templates` — インストール済みテンプレートの一覧。
 * `POST /api/templates/register` — 単一ファイルを体裁継承エントリ (`base` のみ)
 * 1 つの `.evocart` にしてその場で install する。`outline` / `fields` を持つ
 * エントリは manifest を書いたパッケージ (カートリッジ) 経由でしか配れない。
 */

import { loggedFetch as fetch } from '$lib/devlog';
import { ApiError, BASE_URL, request, type ApiErrorDetail } from './_client';

/** インストール済みテンプレート 1 件 (`GET /api/templates` の要素) */
export interface TemplateSummary {
	/** `<package_id>:<entry_id>` (`POST /api/chat` の `template` にそのまま渡す) */
	key: string;
	doc_type: string;
	aliases: string[];
	lang: string;
	description: string;
	package: string;
	version: string;
	/** 体裁の継承元を持つか */
	has_base: boolean;
	/** 構成 (見出し + key_points) を持つか */
	has_outline: boolean;
	/** 帳票 (穴埋め用フィールド) を持つか */
	has_fields: boolean;
}

/** `POST /api/templates/register` の結果 */
export interface RegisterTemplateResult {
	key: string;
	package: string;
	version: string;
	doc_type: string;
	aliases: string[];
}

/** インストール済みテンプレート一覧 */
export async function listTemplates(): Promise<TemplateSummary[]> {
	const data = await request<{ templates: TemplateSummary[] }>('GET', '/templates');
	return data.templates;
}

/**
 * 単一ファイルを体裁継承エントリとして登録する。
 *
 * バックエンドの検証エラーは構造化 `detail`
 * (`{code,message,i18n_key,context}`) と単純な文字列 `detail` の両方が混在
 * する (doc_type 未指定 / 読み込み不可 / サイズ超過は文字列、未対応拡張子は
 * 構造化)。共通の `parseApiError` (`_client.ts`) は文字列 `detail` を汎用
 * メッセージへ丸めてしまうため、ここでは両方の形を読んでメッセージを組み立てる。
 */
export async function registerTemplate(
	file: File,
	docType: string,
	aliases: string[],
	lang: string
): Promise<RegisterTemplateResult> {
	const fd = new FormData();
	fd.append('file', file);
	fd.append('doc_type', docType);
	for (const alias of aliases) fd.append('aliases', alias);
	fd.append('lang', lang);

	const res = await fetch(`${BASE_URL}/templates/register`, { method: 'POST', body: fd });
	if (!res.ok) {
		throw await parseRegisterTemplateError(res);
	}
	return res.json() as Promise<RegisterTemplateResult>;
}

async function parseRegisterTemplateError(res: Response): Promise<ApiError> {
	let message = res.statusText || `HTTP ${res.status}`;
	try {
		const body = await res.json();
		if (body?.detail && typeof body.detail === 'object' && body.detail.code) {
			return new ApiError(body.detail as ApiErrorDetail);
		}
		if (typeof body?.detail === 'string' && body.detail.trim()) {
			message = body.detail;
		}
	} catch {
		// JSON parse failed: keep fallback message
	}
	return new ApiError({ code: `E0${res.status}`, message, i18n_key: '', context: {} });
}
