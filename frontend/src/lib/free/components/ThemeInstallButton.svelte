<script lang="ts">
	/** テーマ画面の主操作「インストール」(.zip)。導入直後のテーマは常に未信頼 (docs/c_11 §1)。 */
	import { t } from '$lib/i18n';
	import { availableThemes } from '$lib/free/stores/theme';
	import { getThemes, installThemeApi } from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import Button from './ui/Button.svelte';

	let fileInput: HTMLInputElement | undefined = $state();
	let installing = $state(false);

	async function handleInstall(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		installing = true;
		const result = await handleApiCall(() => installThemeApi(file), {
			fallbackKey: 'theme_manager.install_failed'
		});
		if (result) {
			const data = await handleApiCall(() => getThemes(), {
				silent: true,
				fallbackKey: 'error.themes_failed'
			});
			if (data) availableThemes.set(data.themes ?? []);
		}
		installing = false;
		input.value = '';
	}
</script>

<Button variant="primary" onclick={() => fileInput?.click()} disabled={installing}>
	{installing ? $t('theme_manager.installing') : $t('theme_manager.install')}
</Button>
<input
	bind:this={fileInput}
	type="file"
	accept=".zip"
	class="hidden"
	aria-label={$t('theme_manager.install')}
	onchange={handleInstall}
/>

<style>
	.hidden {
		display: none;
	}
</style>
