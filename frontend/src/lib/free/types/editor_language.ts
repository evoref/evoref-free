/**
 * エディタ用言語型 — Free / Pro 共通で使う最小限の型のみ。
 *
 * 編集機能を持つ CodeMirror 系（EditorPanel / CodeMirrorEditor / create store）は
 * Pro 専用に集約されているが、Free 用 CodeViewer（読み取り専用 CodeMirror
 * ラッパ）でも `EditorLanguage` 型は必要なため、ここに切り出して双方から参照する。
 */
export type EditorLanguage =
	| 'markdown' | 'python' | 'javascript' | 'typescript'
	| 'json' | 'html' | 'css' | 'yaml' | 'xml' | 'sql' | 'php';
