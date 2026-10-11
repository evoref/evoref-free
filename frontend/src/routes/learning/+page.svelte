<script lang="ts">
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import FewShotCurator from '$lib/free/components/FewShotCurator.svelte';
</script>

<PageLayout title={$t('fewshot.title')}>
	{#snippet description()}
		{$t('fewshot.page_description')}
		<a href="/dashboard">{$t('fewshot.dashboard_link')}</a>
	{/snippet}
	<FewShotCurator />
</PageLayout>
