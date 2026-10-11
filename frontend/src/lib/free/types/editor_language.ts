/**
 * コードブロックの言語型 — CodeViewer（読み取り専用 CodeMirror ラッパ）と
 * code_blocks.ts が共有する。
 */
export type EditorLanguage =
	| 'markdown' | 'python' | 'javascript' | 'typescript'
	| 'json' | 'html' | 'css' | 'yaml' | 'xml' | 'sql' | 'php';
