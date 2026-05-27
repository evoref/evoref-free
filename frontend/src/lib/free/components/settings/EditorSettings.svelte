<script lang="ts">
	import { t } from '$lib/i18n';
	import { configData, updateField } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater, toggleArrayItem } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import TextField from './fields/TextField.svelte';
	import NumberField from './fields/NumberField.svelte';
	import SelectField from './fields/SelectField.svelte';
	import ToggleField from './fields/ToggleField.svelte';

	/** CodeMirror パッケージとしてインストール済みの言語一覧 */
	const AVAILABLE_LANGUAGES = [
		'markdown', 'python', 'javascript', 'typescript',
		'json', 'html', 'css', 'yaml', 'xml', 'sql', 'php'
	] as const;

	let editor = $derived(configSection($configData, 'editor'));
	let enabledLanguages = $derived(editor.highlight_languages ?? []);

	function toggleLanguage(lang: string, checked: boolean) {
		updateField('editor', 'highlight_languages', toggleArrayItem(enabledLanguages, lang, checked));
	}
</script>

<SettingsSection tabId="editor">
	<FieldGroup label="settings.group_editor_font">
		<TextField
			label="settings.editor.font_family"
			description="settings.editor.font_family_desc"
			value={String(editor.font_family ?? 'monospace')}
			onchange={fieldUpdater('editor', 'font_family')}
		/>
		<NumberField
			label="settings.editor.font_size"
			value={Number(editor.font_size ?? 14)}
			min={8}
			max={48}
			onchange={fieldUpdater('editor', 'font_size')}
		/>
		<NumberField
			label="settings.editor.line_height"
			value={Number(editor.line_height ?? 1.5)}
			min={1.0}
			max={3.0}
			onchange={fieldUpdater('editor', 'line_height')}
		/>
	</FieldGroup>

	<FieldGroup label="settings.group_editor_display">
		<ToggleField
			label="settings.editor.word_wrap"
			description="settings.editor.word_wrap_desc"
			value={Boolean(editor.word_wrap ?? false)}
			onchange={fieldUpdater('editor', 'word_wrap')}
		/>
		<NumberField
			label="settings.editor.tab_size"
			value={Number(editor.tab_size ?? 4)}
			min={1}
			max={8}
			onchange={fieldUpdater('editor', 'tab_size')}
		/>
		<ToggleField
			label="settings.editor.show_line_numbers"
			value={Boolean(editor.show_line_numbers ?? true)}
			onchange={fieldUpdater('editor', 'show_line_numbers')}
		/>
		<ToggleField
			label="settings.editor.show_active_line"
			value={Boolean(editor.show_active_line ?? true)}
			onchange={fieldUpdater('editor', 'show_active_line')}
		/>
		<ToggleField
			label="settings.editor.show_toolbar"
			value={Boolean(editor.show_toolbar ?? true)}
			onchange={fieldUpdater('editor', 'show_toolbar')}
		/>
	</FieldGroup>

	<FieldGroup label="settings.group_editor_file">
		<SelectField
			label="settings.editor.default_encoding"
			value={String(editor.default_encoding ?? 'utf-8')}
			options={[
				{ value: 'utf-8', label: 'UTF-8' },
				{ value: 'shift_jis', label: 'Shift_JIS' },
				{ value: 'euc-jp', label: 'EUC-JP' },
				{ value: 'iso-2022-jp', label: 'ISO-2022-JP' }
			]}
			onchange={fieldUpdater('editor', 'default_encoding')}
		/>
		<SelectField
			label="settings.editor.default_line_ending"
			value={String(editor.default_line_ending ?? 'lf')}
			options={[
				{ value: 'lf', label: 'LF (Unix/macOS)' },
				{ value: 'crlf', label: 'CRLF (Windows)' },
				{ value: 'cr', label: 'CR (Classic Mac)' }
			]}
			onchange={fieldUpdater('editor', 'default_line_ending')}
		/>
	</FieldGroup>

	<FieldGroup label="settings.group_editor_highlight">
		<div class="field">
			<span class="field-label">{$t('settings.editor.highlight_languages')}</span>
			<div class="lang-grid">
				{#each AVAILABLE_LANGUAGES as lang}
					<label class="lang-check">
						<input
							type="checkbox"
							checked={enabledLanguages.includes(lang)}
							onchange={(e) => toggleLanguage(lang, e.currentTarget.checked)}
						/>
						<span>{lang}</span>
					</label>
				{/each}
			</div>
		</div>
	</FieldGroup>
</SettingsSection>

<style>
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 6px 0;
	}

	.field-label {
		font-size: 13px;
		color: var(--text-secondary);
		font-weight: 500;
	}

	.lang-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
		gap: 6px 12px;
	}

	.lang-check {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 13px;
		color: var(--text-primary);
		cursor: pointer;
	}

	.lang-check input[type="checkbox"] {
		accent-color: var(--accent);
		width: 15px;
		height: 15px;
		cursor: pointer;
	}
</style>
