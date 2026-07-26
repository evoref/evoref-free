<script lang="ts">
	import { t } from '$lib/i18n';
	import {
		isStreaming,
		addMessage,
		appendToLastAssistant,
		addStepToLastAssistant,
		addStepResultToLastAssistant,
		setRagDebugToLastAssistant,
		setEditorRouteToLastAssistant,
		setLongFormProgressToLastAssistant,
		pushGeneratedEditorCode,
		clearGeneratedEditorCode,
		setStreamingEditorCode,
		clearStreamingEditorCode,
		markLastUserTruncated,
		nextMessageId,
		tokenInfo,
		sessionId,
		currentMode,
		attachedFiles,
		clearFiles
	} from '$lib/free/stores/chat';
	import { chatStream, cancelChat } from '$lib/free/api';
	import { get } from 'svelte/store';
	import { themeSlots } from '$lib/free/stores/theme';
	import { addToast } from '$lib/free/stores/toast';
	import FileUpload from './FileUpload.svelte';
	import FilePreview from './FilePreview.svelte';

	let inputText = $state('');
	let textarea: HTMLTextAreaElement | undefined = $state();
	let abortController: AbortController | null = null;
	let cancelled = false;
	/** このターンで partial editor_code を受信したか (final を streaming タブへ確定するため) */
	let sawPartialEditor = false;
	/** このターンでエディタへコードを出力したか (editor_code 受信)。完了通知の発火条件 */
	let sawEditorCode = false;

	/** long_form ユニットステップの detail ("[3/9] GameGrid: 803 tokens") を進捗に分解する */
	function parseLongFormProgress(detail: string, done: boolean) {
		const m = detail.match(/\[(\d+)\/(\d+)\]\s*(.+)$/);
		if (!m) return null;
		const label = m[3].split(/[:(]/)[0].trim();
		return { current: Number(m[1]), total: Number(m[2]), label, done };
	}

	async function handleSend() {
		const text = inputText.trim();
		if (!text || $isStreaming) return;

		const files = get(attachedFiles).map((f) => f.name);

		addMessage({
			id: nextMessageId(),
			role: 'user',
			content: text,
			timestamp: Date.now(),
			files: files.length > 0 ? files : undefined
		});

		inputText = '';
		clearFiles();
		clearGeneratedEditorCode();
		clearStreamingEditorCode();
		sawPartialEditor = false;
		sawEditorCode = false;
		isStreaming.set(true);
		cancelled = false;
		abortController = new AbortController();

		addMessage({
			id: nextMessageId(),
			role: 'assistant',
			content: '',
			timestamp: Date.now()
		});

		try {
			for await (const event of chatStream(text, get(currentMode), get(sessionId), files, abortController.signal)) {
				if (event.type === 'token' && event.token) {
					appendToLastAssistant(event.token);
				} else if (event.type === 'token_info' && event.token_info) {
					tokenInfo.set(event.token_info);
				} else if (event.type === 'step' && event.step) {
					if (import.meta.env.DEV) {
						console.debug('[Chat Step]', event.step.type, event.step.status, event.step.detail?.slice(0, 120));
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
				} else if (event.type === 'rag_debug' && event.rag_debug) {
					setRagDebugToLastAssistant(event.rag_debug);
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
					appendToLastAssistant(`\n\n**Error:** ${event.error}`);
				}
			}
			// コーディングモードでエディタへコードを出力したターンの完了通知
			if (!cancelled && get(currentMode) === 'coding' && sawEditorCode) {
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
			isStreaming.set(false);
			abortController = null;
		}
	}

	function handleCancel() {
		if (!$isStreaming || cancelled) return;
		cancelled = true;
		abortController?.abort();
		cancelChat(get(sessionId));
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
	<div class="input-row">
		<FileUpload />
		<textarea
			bind:this={textarea}
			bind:value={inputText}
			placeholder={$t('chat.input_placeholder')}
			aria-label={$t('chat.input_placeholder')}
			rows="1"
			onkeydown={handleKeydown}
			oninput={autoResize}
			ondragover={handleInputDragOver}
			ondrop={handleInputDrop}
			disabled={$isStreaming}
		></textarea>
		{#if $isStreaming}
			<button class="cancel-btn" onclick={handleCancel} aria-label={$t('chat.cancel')}>
				{$t('chat.cancel')}
			</button>
		{:else}
			<button class="send-btn" onclick={handleSend} disabled={!inputText.trim()} aria-label={$t('chat.send')}>
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
</style>
