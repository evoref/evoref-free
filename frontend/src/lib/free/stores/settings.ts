/** 設定画面の状態管理 */

import { writable, get } from 'svelte/store';
import { fetchConfig, fetchStatus, saveSection } from '$lib/free/services/configService';
import { refreshServerStatus } from './server';
import { refreshVramStatus } from './vram';
import { addToast } from './toast';
import { instanceName } from './app';
import { colorMode } from './theme';
import { setLocale } from '$lib/i18n';
import { layout } from './theme';

// ── TAB_SECTIONS（イミュータブル） ──

/** タブ → 設定セクションのベースマッピング（変更不可）
 *
 * CLI フラグ `--develop=<level>` (debug | investigate | evolve) の SSOT に
 * 集約された (詳細 docs/f_06_cli.md §18)。`config.yaml` の `debug:` セクションは
 * 廃止 (起動時 ValidationError)、サーバ側の `EvorefConfig` も `debug` フィールドを
 * 持たないため `/api/config/debug` は 404 を返す。
 */
const BASE_TAB_SECTIONS: Readonly<Record<string, readonly string[]>> = {
	general: ['instance', 'server', 'theme', 'i18n'],
	inference: ['llama'],
	model: ['model_paths', 'assist_model', 'embedding', 'reranker'],
	rag: ['rag'],
	memory: ['memory'],
	learning: ['learning', 'agent'],
	storage: ['local_paths', 'history'],
	integration: ['tools'],
	generation: ['modes', 'long_form'],
	editor: ['editor']
} as const;

/** エディションに応じたタブセクションマッピングを生成 */
export function buildTabSections(edition: string): Record<string, string[]> {
	const sections: Record<string, string[]> = {};
	for (const [key, value] of Object.entries(BASE_TAB_SECTIONS)) {
		sections[key] = [...value];
	}
	if (edition === 'pro') {
		sections.integration = ['external_api', 'widget_proxy', 'tools'];
	}
	return sections;
}

/** 現在のタブセクションマッピング（エディション反映済み） */
export const tabSections = writable<Record<string, string[]>>(buildTabSections('free'));

/** タブ一覧（キーはエディションに依存しない） */
export const TAB_IDS = Object.keys(BASE_TAB_SECTIONS);

// ── ストア ──

/** 全設定データ（セクション → フィールド → 値） */
export const configData = writable<Record<string, Record<string, unknown>>>({});

/** 設定読み込み済みフラグ */
export const configLoaded = writable(false);

/** アクティブタブ */
export const activeTab = writable('general');

/** タブごとの dirty フラグ */
export const dirtyTabs = writable<Record<string, boolean>>({});

/** タブごとの保存中フラグ */
export const savingTabs = writable<Record<string, boolean>>({});

/** タブごとのバリデーションエラー */
export const tabErrors = writable<Record<string, string[]>>({});

/** バックエンドから取得したエディション */
export const configEdition = writable<string>('free');

/** 設定読み込みエラー */
export const configLoadError = writable(false);

// ── スナップショット（dirty 検出用） ──

let originalSnapshot: Record<string, string> = {};

function snapshotSection(section: string, data: Record<string, unknown>): void {
	originalSnapshot[section] = JSON.stringify(data);
}

function snapshotAll(config: Record<string, Record<string, unknown>>): void {
	originalSnapshot = {};
	for (const [section, data] of Object.entries(config)) {
		if (data && typeof data === 'object') {
			snapshotSection(section, data);
		}
	}
}

// ── syncAppStores レジストリ ──

type SyncHandler = (config: Record<string, Record<string, unknown>>) => void | Promise<void>;

/** セクション → 同期ハンドラのレジストリ */
const syncHandlers = new Map<string, SyncHandler>([
	[
		'instance',
		async () => {
			try {
				const status = await fetchStatus();
				instanceName.set(status.instance_name ?? 'evoref');
			} catch {
				const config = get(configData);
				const name = config.instance?.name;
				if (typeof name === 'string') instanceName.set(name);
			}
		}
	],
	[
		'theme',
		(config) => {
			const mode = config.theme?.color_mode;
			if (mode === 'light' || mode === 'dark') {
				colorMode.set(mode);
			}
		}
	],
	[
		'i18n',
		(config) => {
			const locale = config.i18n?.locale;
			if (typeof locale === 'string') {
				setLocale(locale);
			}
		}
	],
	[
		'editor',
		(config) => {
			const ed = config.editor;
			if (ed && typeof ed === 'object') {
				layout.update((l) => ({
					...l,
					coding: {
						...l.coding,
						editor: {
							show_line_numbers: ed.show_line_numbers as boolean ?? l.coding.editor.show_line_numbers,
							word_wrap: ed.word_wrap as boolean ?? l.coding.editor.word_wrap,
							tab_size: ed.tab_size as number ?? l.coding.editor.tab_size,
							font_family: ed.font_family as string ?? l.coding.editor.font_family,
							font_size: ed.font_size as number ?? l.coding.editor.font_size,
							line_height: ed.line_height as number ?? l.coding.editor.line_height,
							show_toolbar: ed.show_toolbar as boolean ?? l.coding.editor.show_toolbar,
							show_active_line: ed.show_active_line as boolean ?? l.coding.editor.show_active_line,
						}
					}
				}));
			}
		}
	]
]);

// ── 関数 ──

/** 設定を読み込む */
export async function loadConfig(): Promise<void> {
	configLoadError.set(false);
	try {
		const resp = await fetchConfig();
		configData.set(resp.config as Record<string, Record<string, unknown>>);
		configEdition.set(resp.edition);
		tabSections.set(buildTabSections(resp.edition));
		snapshotAll(resp.config as Record<string, Record<string, unknown>>);

		configLoaded.set(true);
		dirtyTabs.set({});
		tabErrors.set({});
	} catch {
		configLoadError.set(true);
		addToast({ type: 'error', i18nKey: 'settings.load_failed' });
	}
}

/** フィールド値を更新 */
export function updateField(section: string, key: string, value: unknown): void {
	configData.update((config) => {
		const sectionData = { ...(config[section] || {}) };
		sectionData[key] = value;
		config = { ...config, [section]: sectionData };
		return config;
	});

	checkDirty(section);
}

/** ネストされたフィールド値を更新 */
export function updateNestedField(
	section: string,
	parentKey: string,
	childKey: string,
	value: unknown
): void {
	configData.update((config) => {
		const sectionData = { ...(config[section] || {}) };
		const parent = { ...(sectionData[parentKey] as Record<string, unknown> || {}) };
		parent[childKey] = value;
		sectionData[parentKey] = parent;
		config = { ...config, [section]: sectionData };
		return config;
	});

	checkDirty(section);
}

/** セクションの dirty 状態をチェック */
function checkDirty(section: string): void {
	const config = get(configData);
	const current = JSON.stringify(config[section] || {});
	const original = originalSnapshot[section] || '{}';
	const isDirty = current !== original;

	const tabId = findTabForSection(section);
	if (tabId) {
		dirtyTabs.update((d) => ({ ...d, [tabId]: isDirty || isTabDirtyExcept(tabId, section) }));
	}
}

/** タブの指定セクション以外が dirty かチェック */
function isTabDirtyExcept(tabId: string, exceptSection: string): boolean {
	const sections = get(tabSections)[tabId] || [];
	const config = get(configData);
	return sections.some((s) => {
		if (s === exceptSection) return false;
		const current = JSON.stringify(config[s] || {});
		const original = originalSnapshot[s] || '{}';
		return current !== original;
	});
}

/** セクション名からタブIDを逆引き */
function findTabForSection(section: string): string | null {
	for (const [tabId, sections] of Object.entries(get(tabSections))) {
		if (sections.includes(section)) return tabId;
	}
	return null;
}

// ── applySection 分割 ──

/** セクション群を順次保存し、エラーがあればメッセージ配列で返す */
async function saveSections(
	sections: string[],
	config: Record<string, Record<string, unknown>>
): Promise<string[]> {
	const errors: string[] = [];
	for (const section of sections) {
		const data = config[section];
		if (!data || typeof data !== 'object') continue;

		try {
			await saveSection(section, data);
			snapshotSection(section, data);
		} catch (e: unknown) {
			const msg = e instanceof Error ? e.message : String(e);
			errors.push(`${section}: ${msg}`);
		}
	}
	return errors;
}

/** バックエンドから最新設定を再取得してストアに反映 */
async function refreshConfig(): Promise<void> {
	try {
		const resp = await fetchConfig();
		configData.set(resp.config as Record<string, Record<string, unknown>>);
		snapshotAll(resp.config as Record<string, Record<string, unknown>>);
	} catch {
		// リロード失敗は無視（保存自体は成功済み）
	}
}

/**
 * 設定変更後にアプリ全体のストアを同期する
 *
 * config.yaml の変更が他のストア（instanceName, colorMode, locale 等）に
 * 反映されるよう、変更されたセクションに応じて各ストアを更新する。
 */
async function syncAppStores(changedSections: string[]): Promise<void> {
	const config = get(configData);
	for (const section of changedSections) {
		const handler = syncHandlers.get(section);
		if (handler) await handler(config);
	}
}

/** サーバープロセスに影響する設定セクション (保存後にサイドバーを即時更新する) */
const SERVER_AFFECTING_SECTIONS = new Set([
	'model_paths',
	'assist_model',
	'embedding',
	'reranker',
	'llama'
]);

/**
 * サーバー関連セクションを保存した場合、サイドバーの状態 (緑ランプ / VRAM) を
 * 即時更新する。
 *
 * assist_model.enabled の ON/OFF はバックエンドで llama-server プロセスの
 * 自動起動 / 停止を伴うため、次のポーリングを待たずに緑ランプとメモリ容量を
 * 揃えて反映させる。
 */
async function refreshSidebarIfServerSections(changedSections: string[]): Promise<void> {
	if (!changedSections.some((s) => SERVER_AFFECTING_SECTIONS.has(s))) return;
	await Promise.all([refreshServerStatus(), refreshVramStatus()]);
}

/** タブの設定を適用（保存） */
export async function applySection(tabId: string): Promise<boolean> {
	const sections = get(tabSections)[tabId];
	if (!sections) return false;

	savingTabs.update((s) => ({ ...s, [tabId]: true }));
	tabErrors.update((e) => ({ ...e, [tabId]: [] }));

	try {
		const config = get(configData);
		const errors = await saveSections(sections, config);

		if (errors.length > 0) {
			tabErrors.update((e) => ({ ...e, [tabId]: errors }));
			addToast({ type: 'error', i18nKey: 'settings.apply_failed' });
			return false;
		}

		await refreshConfig();
		await syncAppStores(sections);
		await refreshSidebarIfServerSections(sections);

		dirtyTabs.update((d) => ({ ...d, [tabId]: false }));
		addToast({ type: 'success', i18nKey: 'settings.applied' });
		return true;
	} finally {
		savingTabs.update((s) => ({ ...s, [tabId]: false }));
	}
}

/** タブの変更をリセット */
export function resetSection(tabId: string): void {
	const sections = get(tabSections)[tabId];
	if (!sections) return;

	configData.update((config) => {
		const updated = { ...config };
		for (const section of sections) {
			const original = originalSnapshot[section];
			if (original) {
				updated[section] = JSON.parse(original);
			}
		}
		return updated;
	});

	dirtyTabs.update((d) => ({ ...d, [tabId]: false }));
	tabErrors.update((e) => ({ ...e, [tabId]: [] }));
}
