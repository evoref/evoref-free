<script lang="ts">
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import FewShotCurator from '$lib/free/components/FewShotCurator.svelte';
</script>

<PageLayout title={$t('fewshot.title')}>
	<FewShotCurator />
</PageLayout>
