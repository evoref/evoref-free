<script lang="ts">
	import { t } from '$lib/i18n';
	import FieldShell from './FieldShell.svelte';

	interface Props {
		label: string;
		value: boolean;
		description?: string;
		disabled?: boolean;
		error?: string;
		onchange: (value: boolean) => void;
	}

	let { label, value, description = '', disabled = false, error = '', onchange }: Props = $props();
</script>

<FieldShell {label} {description} {error}>
	<button
		type="button"
		class="toggle"
		class:toggle-on={value}
		{disabled}
		onclick={() => onchange(!value)}
		role="switch"
		aria-checked={value}
		aria-label={$t(label)}
	>
		<span class="toggle-knob"></span>
		<span class="toggle-label">{value ? 'ON' : 'OFF'}</span>
	</button>
</FieldShell>

<style>
	.toggle {
		display: flex;
		align-items: center;
		gap: 8px;
		width: fit-content;
		position: relative;
		min-width: 40px;
		height: 22px;
		border-radius: 11px;
		border: none;
		background: var(--border);
		cursor: pointer;
		transition: background 0.2s;
		padding: 0;
		padding-right: 36px;
	}
	.toggle:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.toggle-on {
		background: var(--accent);
	}
	.toggle-knob {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: white;
		transition: transform 0.2s;
	}
	.toggle-on .toggle-knob {
		transform: translateX(18px);
	}
	.toggle-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-on-accent);
		margin-left: 44px;
		user-select: none;
	}
</style>
