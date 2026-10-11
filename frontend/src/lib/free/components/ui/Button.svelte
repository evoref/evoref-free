<script lang="ts">
	/**
	 * 共通ボタン。色相に頼らず、塗り・枠・文字の濃淡で階層を表す (テーマなしのモノトーンで読める)。
	 *
	 * - primary: 塗り。画面に 1 つの主操作
	 * - secondary: 枠のみ。通常の操作
	 * - danger: 枠と文字を強調色 (--color-error) で太く。取り消せない操作
	 * - ghost: 枠なし。行内の控えめな操作
	 */
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';

	type Props = HTMLButtonAttributes & {
		variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
		size?: 'sm' | 'md';
		children: Snippet;
	};

	let {
		variant = 'secondary',
		size = 'md',
		type = 'button',
		class: className = '',
		children,
		...rest
	}: Props = $props();
</script>

<button {type} class="btn btn-{variant} btn-{size} {className}" {...rest}>
	{@render children()}
</button>

<style>
	.btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-1);
		border-radius: var(--radius-lg);
		font-family: inherit;
		line-height: 1.3;
		cursor: pointer;
		white-space: nowrap;
		transition:
			background-color 0.15s var(--ease),
			border-color 0.15s var(--ease);
	}
	.btn-md {
		padding: 7px 16px;
		font-size: var(--font-size-base);
	}
	.btn-sm {
		padding: 3px 10px;
		font-size: var(--font-size-sm);
	}
	.btn-primary {
		background-color: var(--accent);
		color: var(--text-on-accent);
		border: 1px solid var(--accent);
		font-weight: 600;
	}
	.btn-secondary {
		background-color: transparent;
		color: var(--text-primary);
		border: 1px solid color-mix(in srgb, var(--input-border) 70%, transparent);
	}
	.btn-danger {
		background-color: transparent;
		color: var(--color-error);
		border: 1.5px solid var(--color-error);
		font-weight: 600;
	}
	.btn-ghost {
		background-color: transparent;
		color: var(--text-secondary);
		border: 1px solid transparent;
	}
	.btn:not(:disabled):hover {
		background-color: color-mix(in srgb, var(--text-primary) 8%, transparent);
	}
	.btn-primary:not(:disabled):hover {
		background-color: color-mix(in srgb, var(--accent) 85%, var(--text-primary));
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
</style>
