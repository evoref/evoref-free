/**
 * Markdown 内のコードブロック抽出と言語正規化。
 *
 * MarkdownRenderer (チャット応答描画) と Pro 側エディタ連携 (生成コードを
 * エディタタブへ流し込む) の双方が同じ言語判定を使うため、ここに集約する。
 */

import { marked } from 'marked';
import type { EditorLanguage } from '$lib/free/types/editor_language';

export const SUPPORTED_LANGS: ReadonlySet<EditorLanguage> = new Set([
	'markdown',
	'python',
	'javascript',
	'typescript',
	'json',
	'html',
	'css',
	'xml',
	'yaml',
	'sql',
	'php'
]);

export const LANG_ALIASES: Record<string, EditorLanguage> = {
	js: 'javascript',
	jsx: 'javascript',
	ts: 'typescript',
	tsx: 'typescript',
	py: 'python',
	md: 'markdown',
	htm: 'html',
	yml: 'yaml',
	shell: 'markdown',
	bash: 'markdown',
	sh: 'markdown',
	text: 'markdown',
	plaintext: 'markdown'
};

/** コードフェンスの言語タグを EditorLanguage に正規化する（不明は markdown） */
export function normalizeLanguage(raw?: string): EditorLanguage {
	if (!raw) return 'markdown';
	const lower = raw.toLowerCase().trim().split(/\s+/)[0];
	if (SUPPORTED_LANGS.has(lower as EditorLanguage)) {
		return lower as EditorLanguage;
	}
	return LANG_ALIASES[lower] ?? 'markdown';
}

export interface ExtractedCodeBlock {
	language: EditorLanguage;
	content: string;
}

/** Markdown を字句解析し、コードフェンス (```...```) を出現順に抽出する */
export function extractCodeBlocks(markdown: string): ExtractedCodeBlock[] {
	const tokens = marked.lexer(markdown ?? '');
	const blocks: ExtractedCodeBlock[] = [];
	for (const token of tokens) {
		if (token.type === 'code') {
			const code = token as { lang?: string; text: string };
			blocks.push({ language: normalizeLanguage(code.lang), content: code.text });
		}
	}
	return blocks;
}
