<script lang="ts">
	import { t } from '$lib/i18n';
	import PageLayout from '$lib/free/components/PageLayout.svelte';
	import ThemeManager from '$lib/free/components/ThemeManager.svelte';
	import ThemeInstallButton from '$lib/free/components/ThemeInstallButton.svelte';
</script>

<PageLayout title={$t('theme_manager.title')}>
	{#snippet description()}{$t('theme_manager.page_description')}{/snippet}
	{#snippet actions()}
		<ThemeInstallButton />
	{/snippet}
	<ThemeManager />
</PageLayout>
