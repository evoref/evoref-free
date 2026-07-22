<script lang="ts">
	import { configData, updateField } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import { t, promptLocale as promptLocaleCache, promptLocaleSwitching } from '$lib/i18n';
	import { performPromptLocaleSwitch } from '$lib/free/services/configService';
	import { addToast } from '$lib/free/stores/toast';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import SelectField from './fields/SelectField.svelte';

	let instance = $derived(configSection($configData, 'instance'));
	let server = $derived(configSection($configData, 'server'));
	let theme = $derived(configSection($configData, 'theme'));
	let i18n = $derived(configSection($configData, 'i18n'));

	async function handlePromptLocaleChange(newLocale: string | boolean | null) {
		if (typeof newLocale !== 'string') return;
		const currentLocale = String(i18n.prompt_locale ?? 'ja');
		if (newLocale === currentLocale) return;
		if (!confirm($t('settings.i18n.prompt_locale_confirm'))) return;

		// サイドバーの switchLocale() と同じ promptLocaleSwitching ストアを
		// 共有する。どちらの画面から開始した切替でも両方の select を
		// disable し、同一の排他操作 (prompt_locale 切替) への二重発火を
		// 防ぐ (2026-07-22 監査で判明)。
		promptLocaleSwitching.set(true);
		try {
			const { relearning } = await performPromptLocaleSwitch(newLocale);
			updateField('i18n', 'prompt_locale', newLocale);
			promptLocaleCache.set(newLocale);
			const key = relearning
				? 'settings.i18n.prompt_locale_switched_with_relearn'
				: 'settings.i18n.prompt_locale_switched';
			addToast({ type: 'success', i18nKey: key });
		} catch {
			addToast({ type: 'error', i18nKey: 'settings.i18n.prompt_locale_failed' });
		} finally {
			promptLocaleSwitching.set(false);
		}
	}
</script>

<SettingsSection tabId="general">
	<FieldGroup label="settings.group_instance">
		<TextField
			label="settings.instance.name"
			value={String(instance.name ?? '')}
			onchange={fieldUpdater('instance', 'name')}
		/>
	</FieldGroup>

	<FieldGroup label="settings.group_server">
		<TextField
			label="settings.server.host"
			value={String(server.host ?? '')}
			onchange={fieldUpdater('server', 'host')}
		/>
		<NumberField
			label="settings.server.port"
			value={Number(server.port ?? 8000)}
			min={1024}
			max={65535}
			onchange={fieldUpdater('server', 'port')}
		/>
		<NumberField
			label="settings.server.frontend_port"
			value={Number(server.frontend_port ?? 5173)}
			min={1024}
			max={65535}
			onchange={fieldUpdater('server', 'frontend_port')}
		/>
		<NumberField
			label="settings.server.timeout"
			value={Number(server.timeout ?? 30)}
			min={1}
			onchange={fieldUpdater('server', 'timeout')}
		/>
	</FieldGroup>

	<FieldGroup label="settings.group_theme">
		<SelectField
			label="settings.theme.color_mode"
			value={String(theme.color_mode ?? 'dark')}
			options={[
				{ value: 'dark', label: 'Dark', i18nLabel: 'settings.option.dark' },
				{ value: 'light', label: 'Light', i18nLabel: 'settings.option.light' }
			]}
			onchange={fieldUpdater('theme', 'color_mode')}
		/>
		<SelectField
			label="settings.theme.cli_layout_mode"
			value={String(theme.cli_layout_mode ?? 'auto')}
			options={[
				{ value: 'auto', label: 'Auto', i18nLabel: 'settings.option.auto' },
				{ value: 'split', label: 'Split', i18nLabel: 'settings.option.split' },
				{ value: 'sequential', label: 'Sequential', i18nLabel: 'settings.option.sequential' }
			]}
			onchange={fieldUpdater('theme', 'cli_layout_mode')}
		/>
	</FieldGroup>

	<FieldGroup label="settings.group_i18n">
		<SelectField
			label="settings.i18n.locale"
			value={String(i18n.locale ?? 'ja')}
			options={[
				{ value: 'ja', label: '日本語' },
				{ value: 'en', label: 'English' }
			]}
			onchange={fieldUpdater('i18n', 'locale')}
		/>
		<SelectField
			label="settings.i18n.fallback"
			value={String(i18n.fallback ?? 'ja')}
			options={[
				{ value: 'ja', label: '日本語' },
				{ value: 'en', label: 'English' }
			]}
			onchange={fieldUpdater('i18n', 'fallback')}
		/>
		<SelectField
			label="settings.i18n.prompt_locale"
			value={String(i18n.prompt_locale ?? 'ja')}
			options={[
				{ value: 'ja', label: '日本語' },
				{ value: 'en', label: 'English' }
			]}
			disabled={$promptLocaleSwitching}
			onchange={handlePromptLocaleChange}
		/>
	</FieldGroup>
</SettingsSection>
