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
	import { untrack } from 'svelte';
	import { marked, type Token } from 'marked';
	import DOMPurify from 'dompurify';
	import CodeViewer from './CodeViewer.svelte';
	import MermaidDiagram from './MermaidDiagram.svelte';
	import { normalizeLanguage } from '$lib/free/utils/code_blocks';
	import { installKatexExtension, KATEX_SANITIZE_OPTIONS } from '$lib/free/utils/katex_marked';
	import { t } from '$lib/i18n';
	import 'katex/dist/katex.min.css';

	// marked シングルトンへの登録はモジュール初期化時に一度だけ行う。
	installKatexExtension();

	function isMermaid(lang: string | undefined): boolean {
		return (lang ?? '').toLowerCase().trim() === 'mermaid';
	}

	/**
	 * ブロック単位の描画結果キャッシュ (token.raw -> sanitize 済み HTML)。
	 *
	 * ストリーミング中は SSE 1 トークンごとに content が伸びて再描画が走るが、
	 * 末尾ブロック以外は内容が変わらない。キャッシュが無いと毎フレーム全ブロックを
	 * marked.parser + DOMPurify.sanitize し直すため O(n^2) になる。
	 *
	 * 実測 (2026-07-25, 7,702 字 / 54 ブロックの応答):
	 *   lexer のみ                      0.50 ms/回
	 *   lexer + parser + sanitize       100.17 ms/回  ← 99.5% がここ
	 * 4,073 トークンのストリームでは累計で 1 コアの約 44% を再描画のみに消費し、
	 * ページが応答不能になっていた (CDP evaluate が 45 秒でタイムアウト)。
	 *
	 * token.raw が同一なら marked.parser も DOMPurify も決定論的に同じ出力を返すため、
	 * キーとして安全。コンポーネントインスタンス単位で保持し、メッセージが破棄されれば
	 * 一緒に GC される (メッセージ間で共有すると長文が他メッセージの分を evict する)。
	 *
	 * 実測の改善 (400 トークンのストリーム再現): 504.0 秒 -> 36.2 秒 = 92.8% 削減。
	 */
	const renderCache = new Map<string, string>();
	const RENDER_CACHE_MAX = 512;

	function cachePut(key: string, html: string): void {
		// FIFO で上限を保つ (Map は挿入順を保持する)。
		if (renderCache.size >= RENDER_CACHE_MAX) {
			const oldest = renderCache.keys().next().value;
			if (oldest !== undefined) renderCache.delete(oldest);
		}
		renderCache.set(key, html);
	}

	// suppressCode: クリエイトモードでコードをエディタへ流す場合、チャット側の
	// コードブロックを「エディタに出力」プレースホルダに置換する。
	let { content, suppressCode = false }: { content: string; suppressCode?: boolean } = $props();

	/**
	 * 描画をフレームへ合流させる。
	 *
	 * ブロック単位キャッシュ (上記) は **確定したブロック** の再描画を消すが、
	 * 伸び続ける末尾ブロックは毎回 raw が変わるのでキャッシュに乗らず、
	 * `marked.lexer` も毎回全文を舐める。SSE トークンが来るたびに描画すると
	 * この 2 つが 1 トークンごとに走る。
	 *
	 * 実測 (2026-09-03 ライブ監査、ストリーミング中): 200ms タイマーの遅延が
	 * 700-840ms、完了直後は 0-1ms。メインスレッドの約 8 割がここで潰れており、
	 * 12 秒の sleep を挟む CDP evaluate が 45 秒の上限を超えた。投機デコードは
	 * トークンをまとめて吐く (draft acceptance 2.5 前後) ので、山は実際に立つ。
	 *
	 * 1 フレームに 1 回へ丸めれば、描画回数はトークン数ではなく経過時間に比例する。
	 * 表示は人間の知覚上変わらない (60fps を超える更新は見えない)。
	 */
	// 初回は **同期** に描く。遅らせてよいのは「伸びている途中の再描画」だけで、
	// 最初の 1 枚まで 1 フレーム遅らせると、確定済みの過去メッセージまで一瞬
	// 空になる (テストのように同期で内容を読む経路も壊れる)。
	// `untrack` は「初期値だけを取る」意図の明示 (以降の更新は下の $effect が拾う)。
	let renderSource = $state(untrack(() => content) ?? '');
	let pendingFrame = 0;

	const schedule =
		typeof requestAnimationFrame === 'function'
			? requestAnimationFrame
			: (cb: FrameRequestCallback) => setTimeout(() => cb(0), 16) as unknown as number;
	const unschedule =
		typeof cancelAnimationFrame === 'function'
			? cancelAnimationFrame
			: (id: number) => clearTimeout(id);

	$effect(() => {
		const next = content ?? '';
		if (next === renderSource) return;
		// 既にフレームを予約済みなら、そのフレームが **最新の** content を拾う。
		// ここで cancel/再予約すると、連続更新中に永久に発火しない危険がある。
		if (pendingFrame !== 0) return;
		pendingFrame = schedule(() => {
			pendingFrame = 0;
			renderSource = content ?? '';
		});
	});

	$effect(() => () => {
		if (pendingFrame !== 0) {
			unschedule(pendingFrame);
			pendingFrame = 0;
		}
	});

	let tokens = $derived(marked.lexer(renderSource));

	function renderNonCode(token: Token): string {
		// 1 トークンずつ HTML 化。リンク参照定義 ([id]: url) は通常チャット応答では稀。
		const key = token.raw ?? '';
		const hit = key ? renderCache.get(key) : undefined;
		if (hit !== undefined) return hit;
		const html = marked.parser([token]) as string;
		const safe = DOMPurify.sanitize(html, KATEX_SANITIZE_OPTIONS);
		if (key) cachePut(key, safe);
		return safe;
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
	/* 長い数式は折り返さず横スクロールさせる (本文側は折り返したまま) */
	.markdown-renderer :global(.math-block) {
		overflow-x: auto;
		overflow-y: hidden;
		margin: 8px 0;
		padding-bottom: 2px;
	}
	.markdown-renderer :global(.katex) {
		word-break: normal;
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
	/*
	 * Tailwind の preflight が ul/ol を list-style: none にリセットするため、
	 * markdown 本文のリストだけマーカーを復元する。padding-left しか戻して
	 * いなかったので、箇条書きの記号と番号付きリストの番号が両方消えていた
	 * (実測 2026-07-25: 「手順を短く番号付きでまとめて」に対し ol は正しく
	 *  生成されているのに 1. 2. 3. が表示されず、手順の順序が読み取れなかった)。
	 */
	.markdown-renderer :global(ul),
	.markdown-renderer :global(ol) {
		padding-left: 24px;
		margin: 4px 0 8px;
	}
	.markdown-renderer :global(ul) {
		list-style: disc outside;
	}
	.markdown-renderer :global(ol) {
		list-style: decimal outside;
	}
	.markdown-renderer :global(ul ul) {
		list-style-type: circle;
	}
	.markdown-renderer :global(ul ul ul) {
		list-style-type: square;
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
