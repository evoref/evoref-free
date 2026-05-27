<script lang="ts">
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import ThemeManager from '$lib/free/components/ThemeManager.svelte';
</script>

<PageLayout title={$t('theme_manager.title')}>
	<ThemeManager />
</PageLayout>
