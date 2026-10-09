<script lang="ts">
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import ProSection from './ProSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';

	let tools = $derived(configSection($configData, 'tools'));
	let widgetProxy = $derived(configSection($configData, 'widget_proxy'));
</script>

<SettingsSection tabId="integration">
	<!-- Free: Tools -->
	<FieldGroup label="settings.group_tools">
		<ToggleField label="settings.tools.fetch_url_enabled" value={Boolean(tools.fetch_url_enabled ?? true)} onchange={fieldUpdater('tools', 'fetch_url_enabled')} />
		<NumberField label="settings.tools.fetch_url_timeout" value={Number(tools.fetch_url_timeout ?? 10)} min={1} onchange={fieldUpdater('tools', 'fetch_url_timeout')} />
	</FieldGroup>

	<!-- Pro: Widget Proxy -->
	<ProSection>
		<FieldGroup label="settings.group_widget_proxy">
			<ToggleField label="settings.widget_proxy.enabled" value={Boolean(widgetProxy.enabled ?? false)} onchange={fieldUpdater('widget_proxy', 'enabled')} />
			<TextField label="settings.widget_proxy.global_rate_limit" value={String(widgetProxy.global_rate_limit ?? '60/min')} onchange={fieldUpdater('widget_proxy', 'global_rate_limit')} />
			<NumberField label="settings.widget_proxy.request_timeout_sec" value={Number(widgetProxy.request_timeout_sec ?? 10)} min={1} onchange={fieldUpdater('widget_proxy', 'request_timeout_sec')} />
			<NumberField label="settings.widget_proxy.max_response_size_kb" value={Number(widgetProxy.max_response_size_kb ?? 512)} min={1} onchange={fieldUpdater('widget_proxy', 'max_response_size_kb')} />
			<NumberField label="settings.widget_proxy.cache_ttl_sec" value={Number(widgetProxy.cache_ttl_sec ?? 60)} min={0} onchange={fieldUpdater('widget_proxy', 'cache_ttl_sec')} />
		</FieldGroup>
	</ProSection>
</SettingsSection>
