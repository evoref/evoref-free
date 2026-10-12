/** 設定画面の状態管理 */

import { writable, get } from 'svelte/store';
import { fetchConfig, fetchStatus, saveSection } from '$lib/free/services/configService';
import { refreshServerStatus } from './server';
import { refreshVramStatus } from './vram';
import { addToast } from './toast';
import { instanceName } from './app';
import { colorMode, themeId } from './theme';
import { setLocale, promptLocale, t } from '$lib/i18n';
import { ApiError } from '$lib/free/api';

// ── TAB_SECTIONS（イミュータブル） ──

/** タブ → 設定セクションのベースマッピング（変更不可）
 *
 * CLI フラグ `--develop=<level>` (debug | investigate | evolve) の SSOT に
 * 集約された (詳細 docs/f_06_cli.md §6)。`config.yaml` の `debug:` セクションは
 * 廃止 (起動時 ValidationError)、サーバ側の `EvorefConfig` も `debug` フィールドを
 * 持たないため `/api/config/debug` は 404 を返す。
 */
const BASE_TAB_SECTIONS: Readonly<Record<string, readonly string[]>> = {
	general: ['instance', 'server', 'theme', 'i18n'],
	// モデル画面はカードごとに即保存する (パスは切替ボタン、起動オプションは saveLaunchArgs)。タブの「適用」は無い
	model: [],
	rag: ['rag'],
	memory: ['memory'],
	learning: ['learning', 'agent', 'schedule'],
	storage: ['local_paths', 'history'],
	integration: ['tools'],
	// llama は最大応答トークン数 (max_tokens) だけを画面に出す
	generation: ['modes', 'llama', 'long_form'],
	// config を編集しないタブ (環境調整の実行・確認だけ)。セクション無し
	system: []
} as const;

/** エディションに応じたタブセクションマッピングを生成 */
export function buildTabSections(edition: string): Record<string, string[]> {
	const sections: Record<string, string[]> = {};
	for (const [key, value] of Object.entries(BASE_TAB_SECTIONS)) {
		sections[key] = [...value];
	}
	if (edition === 'pro') {
		sections.integration = ['widget_proxy', 'tools'];
		sections.memory = ['memory', 'pro'];
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

/** セクションの保存済みの値 (スナップショット)。無ければ null */
function snapshotOf(section: string): Record<string, unknown> | null {
	const raw = originalSnapshot[section];
	return raw ? (JSON.parse(raw) as Record<string, unknown>) : null;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
	return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/**
 * ``original`` から ``current`` で変わったキーだけを入れ子のまま返す (dict は再帰、配列などは値ごと)。
 *
 * PUT /api/config/{section} はサーバ側で deep merge されるので、変えたキーだけを送れば足りる。
 * GET は schema 既定で補完した全キーを返すため、丸ごと送ると触っていないキーまで config.yaml で
 * 明示扱いになる (例: modes.* を固定するとモデルプロファイルの推奨サンプリングが効かなくなる)。
 */
export function sectionDiff(
	original: Record<string, unknown>,
	current: Record<string, unknown>
): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(current)) {
		const before = original[key];
		if (isPlainObject(value) && isPlainObject(before)) {
			const sub = sectionDiff(before, value);
			if (Object.keys(sub).length > 0) out[key] = sub;
		} else if (JSON.stringify(value) !== JSON.stringify(before)) {
			out[key] = value;
		}
	}
	return out;
}

/** ``base`` に ``diff`` を重ねたコピー (サーバの _deep_merge と同じ規則) */
function mergeDiff(
	base: Record<string, unknown>,
	diff: Record<string, unknown>
): Record<string, unknown> {
	const out: Record<string, unknown> = { ...base };
	for (const [key, value] of Object.entries(diff)) {
		const prev = out[key];
		out[key] = isPlainObject(value) && isPlainObject(prev) ? mergeDiff(prev, value) : value;
	}
	return out;
}

/** セクションの未保存の編集 (スナップショットからの差分)。スナップショットが無ければ空 */
function pendingDiff(
	section: string,
	config: Record<string, Record<string, unknown>>
): Record<string, unknown> {
	const original = snapshotOf(section);
	const current = config[section];
	if (!original || !isPlainObject(current)) return {};
	return sectionDiff(original, current);
}

/**
 * 未保存の編集 ``diff`` から、基準 ``before`` 以降にサーバ側 (``after``) で値が変わったキーを除く。
 * 除いたキーは ``conflicts`` に ``prefix.key`` で積む。
 */
function dropServerChanged(
	diff: Record<string, unknown>,
	before: Record<string, unknown>,
	after: Record<string, unknown>,
	prefix: string,
	conflicts: string[]
): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(diff)) {
		const b = before[key];
		const a = after[key];
		if (isPlainObject(value) && isPlainObject(b) && isPlainObject(a)) {
			const sub = dropServerChanged(value, b, a, `${prefix}.${key}`, conflicts);
			if (Object.keys(sub).length > 0) out[key] = sub;
		} else if (JSON.stringify(b) !== JSON.stringify(a)) {
			if (JSON.stringify(value) !== JSON.stringify(a)) conflicts.push(`${prefix}.${key}`);
		} else {
			out[key] = value;
		}
	}
	return out;
}

/**
 * サーバの最新値を基準 (スナップショット) にし、その上に未保存の編集を重ね直す。
 *
 * 再読込 (モデル切替後・保存後・ページ再表示) で未保存の編集を黙って消さないため。
 * ただし基準以降にサーバ側で変わったキー (モデル切替・閾値の推定・CLI 等の明示操作の結果) は
 * サーバ値を採り、捨てた編集を通知する。編集を残すと次の適用でサーバ側の変更を黙って巻き戻す。
 */
function rebaseOnServer(server: Record<string, Record<string, unknown>>): void {
	const current = get(configData);
	const next: Record<string, Record<string, unknown>> = {};
	const conflicts: string[] = [];
	for (const [section, data] of Object.entries(server)) {
		if (!isPlainObject(data)) {
			next[section] = data;
			continue;
		}
		const before = snapshotOf(section) ?? {};
		const diff = dropServerChanged(pendingDiff(section, current), before, data, section, conflicts);
		next[section] = Object.keys(diff).length > 0 ? mergeDiff(data, diff) : data;
	}
	configData.set(next);
	snapshotAll(server);
	recomputeDirtyTabs();
	if (conflicts.length > 0) {
		addToast({
			type: 'warning',
			i18nKey: 'settings.server_changed_discarded',
			params: { keys: conflicts.join(', ') },
			duration: 10000
		});
	}
}

function isSectionDirty(section: string, config: Record<string, Record<string, unknown>>): boolean {
	const current = JSON.stringify(config[section] || {});
	const original = originalSnapshot[section] || '{}';
	return current !== original;
}

/** 全タブの dirty を現在値とスナップショットから引き直す */
function recomputeDirtyTabs(): void {
	const config = get(configData);
	const dirty: Record<string, boolean> = {};
	for (const [tabId, sections] of Object.entries(get(tabSections))) {
		dirty[tabId] = sections.some((s) => isSectionDirty(s, config));
	}
	dirtyTabs.set(dirty);
}

// ── syncAppStores レジストリ ──

type SyncHandler = (
	config: Record<string, Record<string, unknown>>,
	diff: Record<string, unknown>
) => void | Promise<void>;

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
		(config, diff) => {
			// 変えていないキーで他ストアを巻き戻さない (サイドバーで切り替えた値を保つ)
			if (!('color_mode' in diff)) return;
			const mode = config.theme?.color_mode;
			if (mode === 'light' || mode === 'dark') {
				colorMode.set(mode);
			}
		}
	],
	[
		'i18n',
		(config, diff) => {
			if (!('locale' in diff)) return;
			const locale = config.i18n?.locale;
			if (typeof locale === 'string') {
				setLocale(locale);
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
		configEdition.set(resp.edition);
		tabSections.set(buildTabSections(resp.edition));
		// 未保存の編集は温存する (呼び出し元: ページ表示・モデル切替後・再構築後・閾値の適用後)
		rebaseOnServer(resp.config as Record<string, Record<string, unknown>>);

		configLoaded.set(true);
		const dirty = get(dirtyTabs);
		tabErrors.update((e) => Object.fromEntries(Object.entries(e).filter(([tab]) => dirty[tab])));
	} catch {
		configLoadError.set(true);
		addToast({ type: 'error', i18nKey: 'settings.load_failed' });
	}
}

/**
 * promptLocale ストア (サイドバー/GeneralSettings 双方の switchLocale 系
 * 関数が成功後に更新する) を購読し、configData.i18n.prompt_locale を
 * 追随させる。config 未ロード時 (configData.i18n が無い) は loadConfig()
 * が後で正しい値を取得するため何もしない。
 */
promptLocale.subscribe((newLocale) => {
	const config = get(configData);
	if (config.i18n && config.i18n.prompt_locale !== newLocale) {
		syncPersistedField('i18n', 'prompt_locale', newLocale);
	}
});

/**
 * サイドバーのカラーモード切替を configData.theme.color_mode に追随させる。
 *
 * テーマが有効なときの切替は POST /api/themes/activate が config.yaml へ保存するので、
 * 基準 (snapshot) も同じ値にして偽の dirty を出さない。テーマ無しの切替は
 * config.yaml に書かれないので触らない (Apply は変えたキーだけを送るので巻き戻さない)。
 */
colorMode.subscribe((mode) => {
	const config = get(configData);
	if (config.theme && get(themeId) && config.theme.color_mode !== mode) {
		syncPersistedField('theme', 'color_mode', mode);
	}
});

/**
 * model_paths の単一フィールドを即時保存する (migrate 非対象キー用。create_model 等)
 *
 * base/embed のような model_state 追跡キーは migrate API 経由でしか変更できないが、
 * create_model は非追跡なので config 直保存でよい。追跡キーは現在値のまま同梱して送ると
 * バックエンドの immutable ガードを通過する (値が変化したキーのみ 403)。
 * 保存後の dirty クリア / 再スナップショットは呼び出し側の loadConfig 再取得に委ねる。
 */
export async function saveModelPathsField(key: string, value: string | null): Promise<void> {
	const current = { ...(get(configData).model_paths ?? {}), [key]: value };
	await saveSection('model_paths', current);
}

/** 起動オプション (launch_args) の置き場。rerank は rag.rerank の下 (c_16 §7.2.4) */
export type LaunchArgsTarget = 'llama' | 'embedding' | 'rerank';

/**
 * 起動オプションを即保存する (モデル画面のカードの「保存」)。
 *
 * 送るのは launch_args だけ (節の他のキーは config.yaml に書き戻さない)。成功したら configData と
 * dirty 判定の snapshot の launch_args だけを差し替える — 同じ節を持つ別タブの未保存の編集を消さない。
 * 反映には該当サーバーの再起動が要る。
 */
export async function saveLaunchArgs(target: LaunchArgsTarget, value: string): Promise<void> {
	const [section, path]: [string, string[]] =
		target === 'rerank' ? ['rag', ['rerank', 'launch_args']] : [target, ['launch_args']];
	await saveSection(section, withPath({}, path, value));

	configData.update((config) => ({ ...config, [section]: withPath(config[section] ?? {}, path, value) }));
	const existing = originalSnapshot[section];
	const snapshotObj: Record<string, unknown> = existing ? JSON.parse(existing) : {};
	originalSnapshot[section] = JSON.stringify(withPath(snapshotObj, path, value));
	checkDirty(section);
}

/** ``obj`` の ``path`` に ``value`` を置いたコピー (途中の dict もコピーする) */
function withPath(obj: Record<string, unknown>, path: string[], value: unknown): Record<string, unknown> {
	const [head, ...rest] = path;
	if (rest.length === 0) return { ...obj, [head]: value };
	const child = (obj[head] as Record<string, unknown> | undefined) ?? {};
	return { ...obj, [head]: withPath(child, rest, value) };
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

/**
 * Settings 画面を経由せず別経路 (サイドバーの言語切替等) で既にバックエンドへ
 * 直接永続化済みの値を configData に反映する。dirty 判定の基準となる
 * snapshot 側もキー単位で同時更新するため、Apply ボタンを押さなくても
 * 「未保存の変更あり」表示が誤って出ない (かつ同一セクション内の他の
 * 未保存編集を誤って "保存済み" 扱いにもしない)。
 *
 * 2026-07-22 監査で判明: サイドバーの switchLocale() は promptLocale
 * ストアのみ更新し configData には触れないため、Settings 画面を開いたまま
 * サイドバーで言語を切替えると、その後の Apply (無関係なフィールドへの
 * 変更でも) が configData に残った古い prompt_locale を送信し、
 * 直前に切替えたばかりの値を config.yaml 上で静かに巻き戻していた。
 */
function syncPersistedField(section: string, key: string, value: unknown): void {
	configData.update((config) => {
		const sectionData = { ...(config[section] || {}), [key]: value };
		return { ...config, [section]: sectionData };
	});

	const existing = originalSnapshot[section];
	const snapshotObj: Record<string, unknown> = existing ? JSON.parse(existing) : {};
	snapshotObj[key] = value;
	originalSnapshot[section] = JSON.stringify(snapshotObj);

	checkDirty(section);
}

/**
 * 設定画面の「適用」以外で保存した値 (閾値の推定の適用等) を configData と基準の両方へ反映する。
 * 書いたキーの未保存の編集は捨てて保存した値を採る (同じセクションの他の編集は残す)。
 */
export function syncPersistedFields(section: string, data: Record<string, unknown>): void {
	for (const [key, value] of Object.entries(data)) {
		syncPersistedField(section, key, value);
	}
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

/** 2 段ネスト (section.parent.child.key) のフィールド値を更新 */
export function updateDeepNestedField(
	section: string,
	parentKey: string,
	childKey: string,
	leafKey: string,
	value: unknown
): void {
	configData.update((config) => {
		const sectionData = { ...(config[section] || {}) };
		const parent = { ...(sectionData[parentKey] as Record<string, unknown> || {}) };
		const child = { ...(parent[childKey] as Record<string, unknown> || {}) };
		child[leafKey] = value;
		parent[childKey] = child;
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

/** Pydantic の検証メッセージの定型の前置き */
const PYDANTIC_PREFIX = /^(Value error|Assertion failed), /;

/**
 * 保存失敗を画面に出す短い文言にする。
 *
 * 訳のある i18n_key → 検証エラー (422 の errors) → エラーコード別の定型文の順。
 * バックエンドの英語のメッセージ (pydantic の全文等) はそのまま出さない。
 */
export function describeSaveError(e: unknown): string {
	const translate = get(t);
	if (e instanceof ApiError) {
		if (e.i18nKey) {
			const params = Object.fromEntries(
				Object.entries(e.context ?? {}).map(([k, v]) => [k, String(v)])
			);
			const text = translate(e.i18nKey, Object.keys(params).length > 0 ? params : undefined);
			if (!text.startsWith(e.i18nKey)) return text;
		}
		if (e.errors.length > 0) {
			return e.errors.map((m) => m.replace(PYDANTIC_PREFIX, '')).join(' / ');
		}
		switch (e.code) {
			case 'E0423':
				return translate('settings.error.readonly');
			case 'E0403':
				return translate('settings.error.forbidden');
			case 'E0404':
				return translate('settings.error.not_found');
			case 'E0422':
				return translate('settings.error.invalid');
			default:
				return translate('settings.error.server', { code: e.code });
		}
	}
	return translate('settings.error.network');
}

interface SaveOutcome {
	errors: string[];
	restartRequired: boolean;
	/** 保存できたセクション → 送った差分 */
	saved: Record<string, Record<string, unknown>>;
}

/** セクション群の未保存の編集 (差分) を順次保存する。変更の無いセクションは送らない */
async function saveSections(sections: string[]): Promise<SaveOutcome> {
	const errors: string[] = [];
	const saved: Record<string, Record<string, unknown>> = {};
	let restartRequired = false;
	for (const section of sections) {
		// 各回で読み直す: 前のセクションの保存中にサイドバー等で同期された値を古い値で巻き戻さない
		const diff = pendingDiff(section, get(configData));
		if (Object.keys(diff).length === 0) continue;

		try {
			const res = await saveSection(section, diff);
			if (res?.restart_required === true) restartRequired = true;
			// 基準は「送った時点の値」。保存中に入力された値は差分として残る
			snapshotSection(section, mergeDiff(snapshotOf(section) ?? {}, diff));
			saved[section] = diff;
		} catch (e: unknown) {
			errors.push(`${section}: ${describeSaveError(e)}`);
		}
	}
	return { errors, restartRequired, saved };
}

/** バックエンドから最新設定を再取得し、未保存の編集を重ね直す */
async function refreshConfig(): Promise<void> {
	try {
		const resp = await fetchConfig();
		rebaseOnServer(resp.config as Record<string, Record<string, unknown>>);
	} catch {
		// リロード失敗は無視（保存自体は成功済み）。dirty は保存後の基準で引き直す
		recomputeDirtyTabs();
	}
}

/**
 * 設定変更後にアプリ全体のストアを同期する
 *
 * config.yaml の変更が他のストア（instanceName, colorMode, locale 等）に
 * 反映されるよう、保存したキーに応じて各ストアを更新する。
 */
async function syncAppStores(saved: Record<string, Record<string, unknown>>): Promise<void> {
	const config = get(configData);
	for (const [section, diff] of Object.entries(saved)) {
		const handler = syncHandlers.get(section);
		if (handler) await handler(config, diff);
	}
}

/** サーバープロセスに影響する設定セクション (保存後にサイドバーを即時更新する) */
const SERVER_AFFECTING_SECTIONS = new Set([
	'model_paths',
	'embedding',
	'llama'
]);

/**
 * サーバー関連セクションを保存した場合、サイドバーの状態 (緑ランプ / VRAM) を
 * 即時更新する。
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
		const { errors, restartRequired, saved } = await saveSections(sections);
		const savedSections = Object.keys(saved);

		if (savedSections.length > 0) {
			await refreshConfig();
			await syncAppStores(saved);
			await refreshSidebarIfServerSections(savedSections);
		} else {
			recomputeDirtyTabs();
		}

		if (errors.length > 0) {
			tabErrors.update((e) => ({ ...e, [tabId]: errors }));
			addToast({ type: 'error', i18nKey: 'settings.apply_failed' });
			// 先に保存できたセクションの再起動要否を捨てない
			if (restartRequired) {
				addToast({ type: 'warning', i18nKey: 'settings.applied_partial_restart_required' });
			}
			return false;
		}

		addToast({
			type: 'success',
			i18nKey: restartRequired ? 'settings.applied_restart_required' : 'settings.applied'
		});
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

/** 全タブの未保存の編集を捨てる (未保存のまま設定画面を離れるとき) */
export function discardAllChanges(): void {
	configData.update((config) => {
		const updated = { ...config };
		for (const section of Object.keys(originalSnapshot)) {
			const original = snapshotOf(section);
			if (original) updated[section] = original;
		}
		return updated;
	});
	recomputeDirtyTabs();
	tabErrors.set({});
}
