<script lang="ts">
	/**
	 * MermaidDiagram — ```mermaid コードブロックを図として描画する。
	 *
	 * mermaid は重い依存のため動的 import し、初期バンドルを汚さない。生成された
	 * SVG は DOMPurify (svg プロファイル) で sanitize してから {@html} で挿入する
	 * (frontend ルール: raw HTML は必ず sanitize)。htmlLabels:false で foreignObject
	 * を避け、SVG プロファイルでラベルが落ちないようにする。描画失敗時は元の
	 * mermaid テキストをコードブロックでフォールバック表示する。
	 *
	 * Free 専用。設計フローチャート (staged コーディング) 等の表示に使う。
	 */
	import DOMPurify from 'dompurify';

	let { code }: { code: string } = $props();

	let svg = $state('');
	let failed = $state(false);

	$effect(() => {
		let cancelled = false;
		const src = code ?? '';
		void (async () => {
			try {
				const mermaid = (await import('mermaid')).default;
				mermaid.initialize({
					startOnLoad: false,
					securityLevel: 'strict',
					flowchart: { htmlLabels: false }
				});
				const id =
					'mmd-' +
					(globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2));
				const { svg: out } = await mermaid.render(id, src);
				if (cancelled) return;
				svg = DOMPurify.sanitize(out, { USE_PROFILES: { svg: true, svgFilters: true } });
				failed = false;
			} catch {
				if (cancelled) return;
				failed = true;
			}
		})();
		return () => {
			cancelled = true;
		};
	});
</script>

{#if svg && !failed}
	<div class="mermaid-diagram">{@html svg}</div>
{:else}
	<pre class="mermaid-fallback"><code>{code}</code></pre>
{/if}

<style>
	.mermaid-diagram {
		margin: 8px 0;
		overflow-x: auto;
		text-align: center;
	}
	.mermaid-diagram :global(svg) {
		max-width: 100%;
		height: auto;
	}
	.mermaid-fallback {
		background: var(--code-bg);
		color: var(--code-text);
		padding: 8px 10px;
		border-radius: var(--border-radius);
		overflow-x: auto;
		font-size: 0.9em;
	}
</style>
