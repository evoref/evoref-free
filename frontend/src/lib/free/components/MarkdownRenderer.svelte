<script lang="ts">
	/**
	 * MarkdownRenderer — チャット応答の Markdown を marked.lexer でトークン化し、
	 * code ブロックは CodeViewer (読み取り専用 CodeMirror) で、それ以外は
	 * marked.parser + DOMPurify でサニタイズした HTML として描画する。
	 *
	 * 従来 MessageBubble が `marked.parse()` 一発で HTML 化していた箇所を置き換える。
	 * これにより `<pre><code>` ベタ書きから CodeMirror の defaultHighlightStyle を
	 * 通したシンタックスハイライト表示へ移行できる。
	 *
	 * Free 専用。Pro の CodeMirrorEditor とは依存独立 (Pro/Free 共通 import なし)。
	 */
	import { marked, type Token } from 'marked';
	import DOMPurify from 'dompurify';
	import CodeViewer from './CodeViewer.svelte';
	import MermaidDiagram from './MermaidDiagram.svelte';
	import { normalizeLanguage } from '$lib/free/utils/code_blocks';
	import { t } from '$lib/i18n';

	function isMermaid(lang: string | undefined): boolean {
		return (lang ?? '').toLowerCase().trim() === 'mermaid';
	}

	// suppressCode: コーディングモードでコードをエディタへ流す場合、チャット側の
	// コードブロックを「エディタに出力」プレースホルダに置換する。
	let { content, suppressCode = false }: { content: string; suppressCode?: boolean } = $props();

	let tokens = $derived(marked.lexer(content ?? ''));

	function renderNonCode(token: Token): string {
		// 1 トークンずつ HTML 化。リンク参照定義 ([id]: url) は通常チャット応答では稀。
		const html = marked.parser([token]) as string;
		return DOMPurify.sanitize(html);
	}
</script>

<div class="markdown-renderer">
	{#each tokens as token, i (i)}
		{#if token.type === 'code'}
			{#if isMermaid(token.lang)}
				<!-- 設計フローチャート等: suppressCode でも図は描画する -->
				<MermaidDiagram code={token.text} />
			{:else if suppressCode}
				<div class="code-in-editor">{$t('chat.code_in_editor', { lang: normalizeLanguage(token.lang) })}</div>
			{:else}
				<CodeViewer content={token.text} language={normalizeLanguage(token.lang)} />
			{/if}
		{:else}
			{@html renderNonCode(token)}
		{/if}
	{/each}
</div>

<style>
	.markdown-renderer {
		line-height: 1.5;
		word-break: break-word;
	}
	.code-in-editor {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		margin: 6px 0;
		padding: 4px 10px;
		font-size: 0.85em;
		color: var(--text-secondary);
		background: var(--code-bg);
		border: 1px dashed var(--border);
		border-radius: var(--border-radius);
	}
	.markdown-renderer :global(code) {
		background-color: var(--code-bg);
		color: var(--code-text);
		padding: 2px 4px;
		border-radius: 3px;
		font-size: 0.95em;
	}
	.markdown-renderer :global(p) {
		margin: 0 0 8px;
	}
	.markdown-renderer :global(p:last-child) {
		margin-bottom: 0;
	}
	.markdown-renderer :global(strong) {
		color: var(--md-bold);
	}
	.markdown-renderer :global(h1),
	.markdown-renderer :global(h2),
	.markdown-renderer :global(h3),
	.markdown-renderer :global(h4),
	.markdown-renderer :global(h5),
	.markdown-renderer :global(h6) {
		color: var(--md-heading);
		margin: 16px 0 8px;
		line-height: 1.3;
	}
	.markdown-renderer :global(h1) {
		font-size: 1.4em;
		border-bottom: 1px solid var(--md-heading-border);
		padding-bottom: 6px;
	}
	.markdown-renderer :global(h2) {
		font-size: 1.25em;
		border-bottom: 1px solid var(--md-heading-border);
		padding-bottom: 4px;
	}
	.markdown-renderer :global(h3) {
		font-size: 1.1em;
	}
	.markdown-renderer :global(h4),
	.markdown-renderer :global(h5),
	.markdown-renderer :global(h6) {
		font-size: 1em;
	}
	.markdown-renderer :global(blockquote) {
		border-left: 3px solid var(--md-heading);
		padding-left: 12px;
		margin: 8px 0;
		color: var(--text-secondary);
	}
	.markdown-renderer :global(hr) {
		border: none;
		border-top: 1px solid var(--md-heading-border);
		margin: 12px 0;
	}
	.markdown-renderer :global(ul),
	.markdown-renderer :global(ol) {
		padding-left: 24px;
		margin: 4px 0 8px;
	}
	.markdown-renderer :global(li) {
		margin-bottom: 2px;
	}
	.markdown-renderer :global(table) {
		border-collapse: collapse;
		margin: 8px 0;
		width: 100%;
	}
	.markdown-renderer :global(th) {
		background-color: var(--code-bg);
		color: var(--md-heading);
		font-weight: 600;
		text-align: left;
		padding: 6px 10px;
		border: 1px solid var(--border);
	}
	.markdown-renderer :global(td) {
		padding: 6px 10px;
		border: 1px solid var(--border);
	}
</style>
