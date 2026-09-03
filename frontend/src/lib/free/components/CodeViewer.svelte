<script lang="ts">
	/**
	 * CodeViewer — Free 版チャット応答内コードブロック用の読み取り専用 CodeMirror ラッパ。
	 *
	 * Free / Pro でハイライト見た目を一貫させるため、Pro の CodeMirrorEditor と同じ
	 * 言語拡張 + `defaultHighlightStyle` を使う。ただし以下を排除し、編集 UI を一切
	 * 露出しない:
	 *   - history / historyKeymap / indentWithTab
	 *   - onchange / ondropfiles コールバック
	 *   - drop ハンドラ
	 *   - EditorView.updateListener
	 *
	 * 編集不可は二重に保証する:
	 *   - EditorState.readOnly.of(true)         … doc が dispatch でしか変えられない
	 *   - EditorView.editable.of(false)         … contenteditable=false / フォーカス不可
	 *
	 * Pro 版の編集 UI (EditorPanel / CodeMirrorEditor) とは独立。Free 配下の
	 * edition boundary test (edition_boundary.test.ts) が `$lib/pro/...` への
	 * import / `EditorView.editable.of(true)` / `onchange` props を検知して fail する。
	 */
	import { onMount, onDestroy, untrack } from 'svelte';
	import { EditorView, lineNumbers } from '@codemirror/view';
	import { EditorState, type Extension } from '@codemirror/state';
	import { syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language';
	import { markdown } from '@codemirror/lang-markdown';
	import { python } from '@codemirror/lang-python';
	import { javascript } from '@codemirror/lang-javascript';
	import { json } from '@codemirror/lang-json';
	import { html } from '@codemirror/lang-html';
	import { css } from '@codemirror/lang-css';
	import { xml } from '@codemirror/lang-xml';
	import { yaml } from '@codemirror/lang-yaml';
	import { sql } from '@codemirror/lang-sql';
	import { php } from '@codemirror/lang-php';
	import type { EditorLanguage } from '$lib/free/types/editor_language';

	const languageExtensions: Record<EditorLanguage, () => Extension> = {
		markdown,
		python,
		javascript,
		typescript: () => javascript({ typescript: true }),
		json,
		html,
		css,
		xml,
		yaml,
		sql,
		php
	};

	let {
		content = '',
		language = 'markdown' as EditorLanguage,
		showLineNumbers = false
	}: {
		content?: string;
		language?: EditorLanguage;
		showLineNumbers?: boolean;
	} = $props();

	let container: HTMLDivElement | undefined = $state();
	let view: EditorView | undefined = $state();

	function buildThemeExtension(): Extension {
		return EditorView.theme({
			'&': {
				backgroundColor: 'var(--code-bg)',
				color: 'var(--code-text)',
				fontSize: '14px',
				borderRadius: '4px'
			},
			'.cm-scroller': {
				overflow: 'auto',
				fontFamily: 'var(--editor-font-family, monospace)',
				lineHeight: '1.5'
			},
			'.cm-content': {
				padding: '10px 0',
				caretColor: 'transparent'
			},
			'.cm-gutters': {
				backgroundColor: 'transparent',
				color: 'var(--text-secondary)',
				border: 'none'
			},
			'.cm-line': {
				padding: '0 12px'
			}
		});
	}

	function buildExtensions(): Extension[] {
		const langFn = languageExtensions[language] ?? markdown;
		const exts: Extension[] = [
			langFn(),
			syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
			EditorState.readOnly.of(true),
			EditorView.editable.of(false),
			buildThemeExtension()
		];
		if (showLineNumbers) {
			exts.unshift(lineNumbers());
		}
		return exts;
	}

	onMount(() => {
		if (!container) return;
		view = new EditorView({
			state: EditorState.create({ doc: content, extensions: buildExtensions() }),
			parent: container
		});
	});

	onDestroy(() => {
		view?.destroy();
	});

	// content 変更時はストリーミング応答更新を想定して view へ dispatch。
	//
	// **全文置換ではなく差分だけ流す。** 以前は `from: 0, to: doc.length` で毎回
	// 文書全体を差し替えていたため、コードブロックが伸びている間、トークンが
	// 届くたびに CodeMirror が全文を再パース・再ハイライトしていた。実測
	// (2026-09-03 動作検証): 本文だけの応答はメインスレッド遅延 p95 11ms、
	// コードブロック付きの応答は p95 802ms / 最大 923ms。ストリーミングは
	// 末尾追記が本質なので、共通接頭辞を除いた差分 (通常は追記分だけ) を
	// dispatch すれば、CodeMirror は増分パースで済む。
	$effect(() => {
		const c = content;
		untrack(() => {
			if (!view) return;
			const cur = view.state.doc.toString();
			if (c === cur) return;
			let from = 0;
			const max = Math.min(cur.length, c.length);
			while (from < max && cur.charCodeAt(from) === c.charCodeAt(from)) from++;
			view.dispatch({ changes: { from, to: cur.length, insert: c.slice(from) } });
		});
	});

	// language 変更時はエディタを再構築 (言語拡張は state レベルで切替不可のため)
	let prevLanguage: EditorLanguage | '' = '';
	$effect(() => {
		const lang = language;
		untrack(() => {
			if (!view || !container) return;
			if (lang === prevLanguage) return;
			prevLanguage = lang;
			const currentDoc = view.state.doc.toString();
			view.destroy();
			view = new EditorView({
				state: EditorState.create({ doc: currentDoc, extensions: buildExtensions() }),
				parent: container
			});
		});
	});
</script>

<div class="code-viewer" bind:this={container}></div>

<style>
	.code-viewer {
		border-radius: 4px;
		overflow: hidden;
		margin: 8px 0;
		border: 1px solid var(--border);
	}
</style>
