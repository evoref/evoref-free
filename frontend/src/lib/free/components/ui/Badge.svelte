<script lang="ts">
	/**
	 * 状態ラベル。区別は文言と枠の濃淡で行い、色相だけに意味を持たせない。
	 *
	 * - strong: 濃い枠と太字 (適用中・読込み済など「今効いている」状態)
	 * - default: 通常の枠
	 * - muted: 点線の枠と薄い文字 (未使用・無効など)
	 */
	import type { Snippet } from 'svelte';

	let {
		tone = 'default',
		title,
		children
	}: {
		tone?: 'strong' | 'default' | 'muted';
		title?: string;
		children: Snippet;
	} = $props();
</script>

<span class="badge badge-{tone}" {title}>{@render children()}</span>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		padding: 0 var(--space-2);
		border-radius: 999px;
		font-size: var(--font-size-xs);
		line-height: 1.6;
		white-space: nowrap;
	}
	.badge-strong {
		border: 1.5px solid var(--text-primary);
		color: var(--text-primary);
		font-weight: 600;
	}
	.badge-default {
		border: 1px solid var(--input-border);
		color: var(--text-secondary);
	}
	.badge-muted {
		border: 1px dashed var(--input-border);
		color: var(--text-muted);
	}
</style>
