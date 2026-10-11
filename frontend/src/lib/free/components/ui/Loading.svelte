<script lang="ts">
	import { t } from '$lib/i18n';

	let { message }: { message?: string } = $props();
</script>

<p class="loading" role="status">{message ?? $t('common.loading')}</p>

<style>
	.loading {
		margin: 0;
		padding: var(--space-5) 0;
		text-align: center;
		color: var(--text-secondary);
	}
</style>
