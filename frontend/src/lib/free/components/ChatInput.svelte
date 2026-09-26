<script lang="ts">
	import { t } from '$lib/i18n';
	import {
		isStreaming,
		addMessage,
		appendToLastAssistant,
		addStepToLastAssistant,
		addStepResultToLastAssistant,
		setRagDebugToLastAssistant,
		setSourcesToLastAssistant,
		setEditorRouteToLastAssistant,
		setLongFormProgressToLastAssistant,
		pushGeneratedEditorCode,
		clearGeneratedEditorCode,
		setStreamingEditorCode,
		clearStreamingEditorCode,
		markLastUserTruncated,
		markLastAssistantTruncated,
		nextMessageId,
		registerStreamAbort,
		tokenInfo,
		sessionId,
		currentMode,
		modeRestartStatus,
		attachedFiles,
		clearFiles, corpusMode, selectedTemplate, templatesRevision, templateHint
	} from '$lib/free/stores/chat';
	import {
		chatStream,
		cancelChat,
		confirmReembed,
		listTemplates,
		type TemplateSummary
	} from '$lib/free/api';
	import { handleApiCall } from '$lib/free/utils/error';
	import { get } from 'svelte/store';
	import { themeSlots } from '$lib/free/stores/theme';
	import { setActiveCreateRun, clearActiveCreateRun } from '$lib/free/stores/createRun';
	import { reattachCreateRun } from '$lib/free/services/createReattach';
	import { addToast } from '$lib/free/stores/toast';
	import { refreshServerStatus, serverState } from '$lib/free/stores/server';
	import FileUpload from './FileUpload.svelte';
	import FilePreview from './FilePreview.svelte';

	/** インストール済みの文書テンプレート一覧 (0 件なら選択 UI 自体を出さない) */
	let templates = $state<TemplateSummary[]>([]);
	let selectedTemplateInfo = $derived(
		templates.find((tpl) => tpl.key === $selectedTemplate) ?? null
	);

	// 起動時と、選べる様式の集合が変わったとき (カートリッジの install / 削除 /
	// 様式の登録 = templatesRevision) に取り直す。消えた様式を選んだままにしない。
	$effect(() => {
		void $templatesRevision;
		let cancelled = false;
		handleApiCall(() => listTemplates(), { silent: true, fallback: [] }).then((result) => {
			if (cancelled) return;
			templates = result ?? [];
			const current = get(selectedTemplate);
			if (current !== null && !templates.some((tpl) => tpl.key === current)) {
				selectedTemplate.set(null);
			}
		});
		return () => {
			cancelled = true;
		};
	});

	/** 選択中テンプレートが持つ部品 (体裁 / 構成 / 帳票) の表示ラベル */
	function templatePartsLabel(tpl: { has_base: boolean; has_outline: boolean; has_fields: boolean }): string {
		const parts: string[] = [];
		if (tpl.has_base) parts.push($t('chat.template_part_base'));
		if (tpl.has_outline) parts.push($t('chat.template_part_outline'));
		if (tpl.has_fields) parts.push($t('chat.template_part_fields'));
		return parts.join(' / ');
	}

	/**
	 * 「使える様式がある」通知 (`template_hint`) のうち、インストール済み一覧
	 * にまだ存在する候補だけ (削除済みの様式を選ばせない)。0 件なら通知自体を
	 * 出さない。
	 */
	let visibleTemplateHints = $derived(
		($templateHint?.templates ?? []).filter((entry) => templates.some((tpl) => tpl.key === entry.key))
	);

	/** 通知のボタンから様式を選ぶ (自動送信はしない — 依頼文は改めて送る) */
	function useTemplateHint(key: string): void {
		selectedTemplate.set(key);
		templateHint.set(null);
	}

	function dismissTemplateHint(): void {
		templateHint.set(null);
	}

	/** 埋め込みモデルの変更を確認し、確認待ちのストアの埋め直しを始める */
	async function handleConfirmReembed(): Promise<void> {
		const result = await handleApiCall(() => confirmReembed(), {
			fallbackKey: 'chat.reembed_confirm_failed'
		});
		if (result) {
			addToast({ type: 'info', i18nKey: 'chat.reembed_started' });
			await refreshServerStatus();
		}
	}

	let inputText = $state('');
	let textarea: HTMLTextAreaElement | undefined = $state();
	let abortController: AbortController | null = null;
	let cancelled = false;
	/** 進行中ターンの request_id (`agent_layer` フレームで届く)。キャンセルの宛先 */
	let activeRequestId: string | undefined;
	/** このターンで partial editor_code を受信したか (final を streaming タブへ確定するため) */
	let sawPartialEditor = false;
	/** このターンでエディタへコードを出力したか (editor_code 受信)。完了通知の発火条件 */
	let sawEditorCode = false;
	/** このターンで needs_input を受信したか (再接続用 run 記録を残すかの判定) */
	let sawNeedsInput = false;
	/** 制作中で受け付けられなかったとき (E0409) の走っている run。後始末の後で再接続する */
	let busyRun: { session_id: string; run_id: string } | null = null;

	/** long_form ユニットステップの detail ("[3/9] GameGrid: 803 tokens") を進捗に分解する */
	function parseLongFormProgress(detail: string, done: boolean) {
		const m = detail.match(/\[(\d+)\/(\d+)\]\s*(.+)$/);
		if (!m) return null;
		const label = m[3].split(/[:(]/)[0].trim();
		return { current: Number(m[1]), total: Number(m[2]), label, done };
	}

	async function handleSend() {
		const text = inputText.trim();
		if (!text) return;
		// 生成中は送信できないが、**入力は消さずに残す**。以前は textarea 自体を
		// disabled にしていたため、生成中に打った文字がどこにも入らず Enter も
		// 無反応で、ユーザーから見るとメッセージが黙って消えた (プレースホルダも
		// 変わらないので手掛かりが一切無い)。実インシデント 2026-08-14 ライブ監査:
		// ターン24 と 25 が 2 回連続で消失した。
		if ($isStreaming) {
			addToast({ type: 'error', i18nKey: 'chat.send_blocked_while_streaming' });
			return;
		}
		// モード切替 (llama-server のモデル入替を伴い数十秒かかる) の最中に
		// 送ると、切替完了時の messages 差し替えでターンごと破棄される。
		if ($modeRestartStatus === 'restarting') return;

		const files = get(attachedFiles).map((f) => f.name);
		// 選択中のテンプレートはこのターンにだけ効かせる (先に読み、送信後は
		// 「なし」へ戻す。c_16 §4.5.2 — 貼り付いたまま別の依頼に効くのを防ぐ)。
		const templateKey = get(selectedTemplate);

		addMessage({
			id: nextMessageId(),
			role: 'user',
			content: text,
			timestamp: Date.now(),
			files: files.length > 0 ? files : undefined
		});

		inputText = '';
		clearFiles();
		selectedTemplate.set(null);
		templateHint.set(null);
		clearGeneratedEditorCode();
		clearStreamingEditorCode();
		sawPartialEditor = false;
		sawEditorCode = false;
		sawNeedsInput = false;
		busyRun = null;
		isStreaming.set(true);
		cancelled = false;
		abortController = new AbortController();
		// 送信時のモード / セッションでターン全体を扱う (完了通知の判定を
		// 完了時点のモードで行うと、生成中に切り替えた場合に食い違う)。
		const mode = get(currentMode);
		const turnSessionId = get(sessionId);
		activeRequestId = undefined;
		// 「新しいチャット」等の外部からの中断を受け付ける。
		registerStreamAbort(handleCancel);

		addMessage({
			id: nextMessageId(),
			role: 'assistant',
			content: '',
			timestamp: Date.now()
		});

		try {
			for await (const event of chatStream(text, mode, turnSessionId, files, abortController.signal, get(corpusMode), templateKey)) {
				if (event.type === 'token' && event.token) {
					appendToLastAssistant(event.token);
				} else if (event.type === 'agent_layer') {
					activeRequestId = event.request_id;
				} else if (event.type === 'token_info' && event.token_info) {
					tokenInfo.set(event.token_info);
				} else if (event.type === 'create_run' && event.run_id && event.session_id) {
					// staged クリエイトの run 識別子。localStorage へ永続し、リロード後の
					// 再接続 (createReattach.ts) の手掛かりにする (f_05 §4.5)。
					setActiveCreateRun({
						session_id: event.session_id,
						run_id: event.run_id,
						started_at: new Date().toISOString()
					});
				} else if (event.type === 'step' && event.step) {
					if (import.meta.env.DEV) {
						console.debug('[Chat Step]', event.step.type, event.step.status, event.step.detail?.slice(0, 120));
					}
					if (event.step.type === 'needs_input') {
						// blocked のまま次ターンで再開するので、run 記録はここでは消さない。
						sawNeedsInput = true;
					}
					if (event.step.type === 'task_result') {
						// Meta-Cognitive 最終タスク結果 → 結果ボックスのみ
						addStepResultToLastAssistant(event.step.detail, event.step.status ?? 'done');
					} else {
						// long_form ユニットステップ → 進捗をチャットに常時表示
						if (event.step.type === 'long_form_unit_start' || event.step.type === 'long_form_unit_done') {
							const p = parseLongFormProgress(
								event.step.detail,
								event.step.type === 'long_form_unit_done'
							);
							if (p) setLongFormProgressToLastAssistant(p);
						}
						// 通常のステップ → AgenticSteps 欄 (詳細ログ、折りたたみ)
						addStepToLastAssistant({
							type: event.step.type,
							detail: event.step.detail,
							status: event.step.status,
							elapsed_ms: event.step.elapsed_ms
						});
					}
				} else if (event.type === 'input_truncated' && event.input_truncated) {
					// 発言バブルに内訳を残し、見落とし防止にトーストも出す
					markLastUserTruncated(event.input_truncated);
					addToast({ type: 'error', i18nKey: 'chat.input_truncated_toast' });
				} else if (event.type === 'output_truncated' && event.output_truncated) {
					// 応答バブルに注記を付ける。本文へ連結すると履歴に保存され、
					// 次ターンでモデルが逐語復唱する (2026-08-25 実測)。
					markLastAssistantTruncated(event.output_truncated);
				} else if (event.type === 'rag_debug' && event.rag_debug) {
					setRagDebugToLastAssistant(event.rag_debug);
				} else if (event.type === 'sources' && event.sources) {
					setSourcesToLastAssistant(event.sources);
				} else if (event.type === 'template_hint' && event.template_hint) {
					templateHint.set(event.template_hint);
				} else if (event.type === 'editor_route' && event.editor_route) {
					setEditorRouteToLastAssistant(event.editor_route.target);
				} else if (event.type === 'editor_code' && event.editor_code) {
					sawEditorCode = true;
					// partial=true: long_form 生成途中の逐次更新 → streaming タブへ上書き。
					// final: partial を受けていれば streaming タブを確定、そうでなければ
					// (meta_cognitive 等の単発出力) 従来通り新規タブへ流す。
					if (event.editor_code.partial) {
						sawPartialEditor = true;
						setStreamingEditorCode(event.editor_code);
					} else if (sawPartialEditor) {
						setStreamingEditorCode(event.editor_code);
					} else {
						pushGeneratedEditorCode(event.editor_code);
					}
				} else if (event.type === 'error') {
					// 別の制作が走っていて受け付けられなかった (f_10 §3)。run が分かれば
					// 後始末の後で再接続し、進捗の表示と中止をそこから行う (f_05 §4.5)。
					const ctx = event.error_context ?? {};
					if (
						event.error_code === 'E0409' &&
						mode === 'create' &&
						typeof ctx.session_id === 'string' &&
						typeof ctx.run_id === 'string' &&
						ctx.session_id &&
						ctx.run_id
					) {
						busyRun = { session_id: ctx.session_id, run_id: ctx.run_id };
					}
					// 内部コードをそのまま画面に出さない。`stream_timeout` は
					// 実際にユーザーが目にした唯一のケースで、生文字列
					// `Error: stream_timeout` が本文の代わりに表示されていた
					// (2026-09-03 ライブ監査)。訳のあるコードは訳を出す。
					const detail =
						event.error === 'stream_timeout'
							? $t('chat.stream_timeout')
							: event.error;
					appendToLastAssistant(`\n\n**${$t('common.error')}:** ${detail}`);
				}
			}
			// クリエイトモードでエディタへコードを出力したターンの完了通知
			if (!cancelled && mode === 'create' && sawEditorCode) {
				addStepResultToLastAssistant($t('chat.code_output_done'), 'done');
			}
		} catch (e) {
			if (!cancelled) {
				appendToLastAssistant(`\n\n**${$t('common.error')}:** ${$t('error.connection.failed')}`);
				addToast({ type: 'error', i18nKey: 'error.connection.failed' });
				console.error('[Chat Error]', e);
			}
		} finally {
			if (cancelled) {
				appendToLastAssistant(`\n\n*${$t('chat.cancelled')}*`);
			}
			registerStreamAbort(null);
			isStreaming.set(false);
			abortController = null;
			activeRequestId = undefined;
			// ターンが終端 (完了 / キャンセル) に達したら再接続用の run 記録を消す。
			// needs_input で止まった run は次ターンで再開するため残す。
			if (mode === 'create' && !sawNeedsInput) {
				clearActiveCreateRun();
			}
			if (busyRun) {
				setActiveCreateRun({ ...busyRun, started_at: new Date().toISOString() });
				busyRun = null;
				void reattachCreateRun();
			}
		}
	}

	function handleCancel() {
		if (!$isStreaming || cancelled) return;
		cancelled = true;
		// 中断はこのターンの session_id + request_id へ送る (「新しいチャット」が
		// ID を回した後に呼ばれても、バックエンドで走っている生成は旧 ID に紐づく。
		// request_id があれば同一セッションの別リクエストを巻き込まない)。
		const sid = get(sessionId);
		abortController?.abort();
		cancelChat(sid, activeRequestId);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	}

	function handleWindowKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape' && $isStreaming) {
			e.preventDefault();
			handleCancel();
		}
	}

	function autoResize() {
		if (textarea) {
			textarea.style.height = 'auto';
			textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
		}
	}

	/** エディタタブのドロップを受け付け */
	function handleInputDragOver(e: DragEvent) {
		if (e.dataTransfer?.types.includes('text/x-editor-tab')) {
			e.preventDefault();
			e.dataTransfer.dropEffect = 'copy';
		}
	}

	function handleInputDrop(e: DragEvent) {
		const filename = e.dataTransfer?.getData('text/x-editor-tab-filename');
		if (!filename) return;
		e.preventDefault();
		const pos = textarea?.selectionStart ?? inputText.length;
		inputText = inputText.slice(0, pos) + filename + inputText.slice(pos);
	}
</script>

<svelte:window onkeydown={handleWindowKeydown} />

<div class="chat-input-area">
	<FilePreview />
	{#if templates.length > 0}
		<div class="template-row">
			<label for="template-select" class="template-label">{$t('chat.template_label')}</label>
			<select
				id="template-select"
				class="template-select"
				value={$selectedTemplate ?? ''}
				onchange={(e) => {
					const v = e.currentTarget.value;
					selectedTemplate.set(v === '' ? null : v);
				}}
			>
				<option value="">{$t('chat.template_none')}</option>
				{#each templates as tpl (tpl.key)}
					<option value={tpl.key}>{tpl.doc_type}</option>
				{/each}
			</select>
			{#if selectedTemplateInfo}
				<span class="template-parts">{templatePartsLabel(selectedTemplateInfo)}</span>
			{/if}
		</div>
	{/if}
	{#if visibleTemplateHints.length > 0}
		<div class="template-hint" role="status">
			{#if visibleTemplateHints.length === 1}
				<span class="template-hint-text">
					{$t('chat.template_hint_single', { doc_type: visibleTemplateHints[0].doc_type })}
				</span>
				<button
					type="button"
					class="template-hint-use"
					onclick={() => useTemplateHint(visibleTemplateHints[0].key)}
				>
					{$t('chat.template_hint_use')}
				</button>
			{:else}
				<span class="template-hint-text">{$t('chat.template_hint_multiple')}</span>
				{#each visibleTemplateHints as entry (entry.key)}
					<button type="button" class="template-hint-use" onclick={() => useTemplateHint(entry.key)}>
						{entry.doc_type}
					</button>
				{/each}
			{/if}
			<button
				type="button"
				class="template-hint-dismiss"
				onclick={dismissTemplateHint}
				aria-label={$t('chat.template_hint_dismiss')}
			>
				×
			</button>
		</div>
	{/if}
	{#if $serverState.dataHealth?.readonly}
		<div class="readonly-note" role="status">{$t('chat.data_readonly_banner')}</div>
	{/if}
	{#if $serverState.dataHealth?.served_model_mismatch}
		<div class="readonly-note" role="status">
			{$t('chat.served_model_mismatch_banner', {
				served: $serverState.dataHealth.served_model,
				expected: $serverState.dataHealth.expected_model
			})}
		</div>
	{/if}
	{#if Object.keys($serverState.dataHealth?.formats ?? {}).length}
		<div class="readonly-note" role="status">
			{$t('chat.format_health_banner', {
				count: Object.keys($serverState.dataHealth?.formats ?? {}).length
			})}
		</div>
	{/if}
	{#if $serverState.dataHealth?.reembed_pending?.length}
		<div class="readonly-note" role="status">
			{$t('chat.reembed_pending_banner', {
				count: $serverState.dataHealth.reembed_pending.length
			})}
			<button type="button" class="reembed-confirm" onclick={handleConfirmReembed}>
				{$t('chat.reembed_confirm')}
			</button>
		</div>
	{/if}
	<div class="input-row">
		<FileUpload />
		<textarea
			bind:this={textarea}
			bind:value={inputText}
			placeholder={$modeRestartStatus === 'restarting'
				? $t('chat.mode_restarting')
				: $t('chat.input_placeholder')}
			aria-label={$t('chat.input_placeholder')}
			rows="1"
			onkeydown={handleKeydown}
			oninput={autoResize}
			ondragover={handleInputDragOver}
			ondrop={handleInputDrop}
			disabled={$modeRestartStatus === 'restarting'}
		></textarea>
		{#if $isStreaming}
			<button class="cancel-btn" onclick={handleCancel} aria-label={$t('chat.cancel')}>
				{$t('chat.cancel')}
			</button>
		{:else}
			<button
				class="send-btn"
				onclick={handleSend}
				disabled={!inputText.trim() || $modeRestartStatus === 'restarting'}
				aria-label={$t('chat.send')}
			>
				{$t('chat.send')}
			</button>
		{/if}
		<!-- input_suffix スロット -->
		{#if $themeSlots.input_suffix}
			{@const InputSuffix = $themeSlots.input_suffix}
			<InputSuffix />
		{/if}
	</div>
</div>

<style>
	.chat-input-area {
		border-top: 0.5px solid var(--border);
		padding: 0 24px 16px;
	}
	.input-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding-top: 12px;
	}
	textarea {
		flex: 1;
		resize: none;
		border: 0.5px solid var(--input-border);
		border-radius: 6px;
		padding: 8px 12px;
		font-size: 14px;
		background: var(--input-bg);
		color: var(--text-primary);
		font-family: inherit;
		line-height: 1.5;
		max-height: 200px;
		outline: none;
	}
	textarea:focus {
		border-color: var(--accent);
	}
	textarea:disabled {
		opacity: 0.6;
	}
	.send-btn {
		padding: 6px 16px;
		background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 85%, #000) 100%);
		color: var(--text-on-accent);
		border: none;
		border-radius: 6px;
		cursor: pointer;
		font-size: 13px;
		white-space: nowrap;
	}
	.send-btn:hover:not(:disabled) {
		opacity: 0.9;
	}
	.send-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.cancel-btn {
		padding: 6px 16px;
		background: var(--text-primary);
		color: var(--bg-primary);
		border: 1px solid var(--border);
		border-radius: 6px;
		cursor: pointer;
		font-size: 13px;
		white-space: nowrap;
	}
	.cancel-btn:hover {
		opacity: 0.85;
	}
	.template-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding-top: 8px;
		font-size: 12px;
		color: var(--text-secondary);
	}
	.template-select {
		font-size: 12px;
		padding: 2px 6px;
		border: 0.5px solid var(--input-border);
		border-radius: 4px;
		background: var(--input-bg);
		color: var(--text-primary);
	}
	.template-parts {
		opacity: 0.8;
	}
	.readonly-note {
		padding-bottom: 8px;
		font-size: 12px;
		color: var(--error, #c0392b);
	}

	.reembed-confirm {
		margin-left: 8px;
		font-size: 12px;
		text-decoration: underline;
	}

	.template-hint {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
		padding-top: 8px;
		font-size: 12px;
		color: var(--text-secondary);
	}
	.template-hint-text {
		white-space: nowrap;
	}
	.template-hint-use {
		padding: 2px 10px;
		font-size: 12px;
		background: var(--input-bg);
		color: var(--accent);
		border: 0.5px solid var(--accent);
		border-radius: 4px;
		cursor: pointer;
	}
	.template-hint-use:hover {
		opacity: 0.85;
	}
	.template-hint-dismiss {
		margin-left: auto;
		padding: 0 4px;
		font-size: 14px;
		line-height: 1;
		background: transparent;
		color: var(--text-secondary);
		border: none;
		cursor: pointer;
	}
	.template-hint-dismiss:hover {
		color: var(--text-primary);
	}
</style>
