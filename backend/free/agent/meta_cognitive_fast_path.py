"""FastPathMixin — meta_cognitive_fast_path"""

from __future__ import annotations

from pathlib import Path
from backend.free.core.session_mode import is_create_mode
from backend.free.agent.meta_cognitive_tasks import TaskItem
from backend.free.agent.output_format import wants_fetched_table
from backend.free.agent.meta_cognitive_utils import (
    call_callback,
    extract_literal_write_content,
    generated_content_rejection,
    is_tool_error,
    previous_answer_write_content,
    rescue_quoted_write_literal,
    strip_generator_scaffold_block,
    strip_markdown_wrapper,
    strip_output_lead_in,
    strip_prompt_scaffold_lines,
    strip_task_log_scaffold,
    summarize_tool_args,
    text_looks_like_code,
    tool_result_succeeded,
)

from backend.free.agent.meta_cognitive_defs import (
    _DATA_BEARING_TOOLS,
    read_existing_file,
)

from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive")


class _FastPathMixin:
    """ツールループを介さない直接実行 (ファストパス)。

    ツール名と引数が決定論で確定しているタスクを、LLM のツールループを
    回さずに実行する層。書込みでは本文の解決・検証・救出までを担う。
    """

    async def _execute_tool_fast(
        self,
        tool_name: str,
        tool_args: dict,
        task: TaskItem,  # noqa: ARG002
        tools_registry,
        on_step=None,
        prefix: str = "",
    ) -> tuple[str, list[dict]]:
        """read_file / run_command / search_code のファストパス実行"""
        logger.info("Tool fast path: %s(%s)", tool_name, tool_args)

        # search_code は create 専用 (modes=["create"]) だが、この判定は
        # ToolCallJudge の rule/aux 判定結果をそのまま実行するため
        # ToolDefinition.modes を経由しない。_execute_tool と同じ理由で
        # search_code のみ mode ゲートを追加する (write_file は対象外)。
        if tool_name == "search_code" and not is_create_mode(self._mode):
            result_text = f"Error: search_code is not available in mode '{self._mode}'"
            logger.warning(
                "Tool fast path not allowed in mode %s: %s", self._mode, tool_name,
            )
            return result_text, [{
                "tool": tool_name,
                "args": tool_args,
                "success": False,
            }]

        if on_step:
            args_summary = summarize_tool_args(tool_name, tool_args)
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} {tool_name}({args_summary})",
                "status": "running",
            })

        try:
            result = await tools_registry.execute(tool_name, **tool_args)
            result_text = str(result)
            is_success = tool_result_succeeded(tool_name, result_text)
        except Exception as e:
            result_text = f"Error: {e}"
            is_success = False
            logger.error("Tool fast path failed: %s - %s", tool_name, e)

        # データ取得結果をタスク横断アキュムレータへ (後続 write タスクの素材に再利用)。
        # ファストパス経由の fetch_url 等もここで蓄積する (ツールループ経路と対称)。
        if tool_name in _DATA_BEARING_TOOLS and is_success:
            self._fetched_tool_outputs.append(result_text)

        tool_entry = {
            "tool": tool_name,
            "args": tool_args,
            "success": is_success,
        }

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} {tool_name}: {result_text[:100]}",
                "status": "done" if is_success else "failed",
            })

        logger.info("Tool fast path completed: %s → %s", tool_name, result_text[:80])
        return result_text, [tool_entry]

    async def _execute_write_fast(
        self,
        task: TaskItem,
        original_query: str,
        file_path: str,
        llm_client,
        tools_registry,
        on_step=None,
        prefix: str = "",
    ) -> tuple[str, list[dict]]:
        """書き込みタスクのファストパス実行"""
        # 出力先を確定 (ディレクトリ→output ファイル / bare 名→クエリ指定ディレクトリ配下)。
        # planner/judge が発明した CWD 相対の bare 名をユーザー指定の場所へ寄せる。
        file_path = self._resolve_write_path(file_path, original_query)
        logger.info("Write fast path: %s → %s", task.description[:60], file_path)

        async def _notify_generating() -> None:
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成中 → {file_path}",
                    "status": "running",
                })

        content, rejection = await self._resolve_write_content(
            file_path=file_path,
            original_query=original_query,
            task_description=task.description,
            llm_client=llm_client,
            notify_generating=_notify_generating,
        )

        if content.startswith("(Content generation failed:"):
            logger.warning("Write fast path: content generation failed for %s", file_path)
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗",
                    "status": "failed",
                })
            return f"Error: {content}", []

        if rejection:
            logger.warning(
                "Write fast path: generated content still rejected (%s) "
                "after retry, aborting: %r",
                rejection, content[:120],
            )
            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: コンテンツ生成失敗（{rejection}）",
                    "status": "failed",
                })
            return (
                f"Error: Content generation produced invalid output ({rejection}), "
                "not actual content",
                [],
            )

        return await self._write_file(
            file_path, content, tools_registry, on_step, prefix,
        )

    async def _resolve_write_content(
        self,
        *,
        file_path: str,
        original_query: str,
        task_description: str,
        llm_client,
        notify_generating=None,
    ) -> tuple[str, str | None]:
        """書込み本文を「決定論 → 生成 → 救済」の順で解決する単一の合流点。

        fast path と tool-loop が本文解決を各々持っており、決定論経路
        (ユーザー literal / 直前応答) が fast path にしか無かったため、
        同じ依頼でも経路が変わると実況文が書き込まれた。両経路をここへ
        集約し、解決順序を 1 箇所で決める。

        Returns:
            ``(content, rejection)``。``rejection`` が None なら書込み可。
        """
        # 1. 取得済みの実テーブル (転記させるとハルシネーション/行脱落が出る)
        if wants_fetched_table(file_path):
            fetched_table = self._extract_fetched_table_markdown()
            if fetched_table:
                logger.info(
                    "Write content from fetched table (deterministic): "
                    "%d chars -> %s", len(fetched_table), file_path,
                )
                return fetched_table, None

        # 2. ユーザーが引用符で本文そのものを指定している (高精度マッチ)
        literal = extract_literal_write_content(original_query, file_path)
        if literal:
            logger.info(
                "Write content from user literal (deterministic): "
                "%d chars -> %s", len(literal), file_path,
            )
            return literal, None

        # 3. 「この案内文を保存して」型: 書くべき本文は直前の応答そのもの
        previous = previous_answer_write_content(
            original_query, getattr(self, "_conversation", None),
        )
        if previous:
            logger.info(
                "Write content from previous answer (deterministic): "
                "%d chars -> %s", len(previous), file_path,
            )
            return previous, None

        if notify_generating:
            await notify_generating()

        # 4. 生成 (棄却されたら 1 度だけ再生成)
        content = await self._generate_content(
            original_query, task_description, llm_client, file_path=file_path,
        )
        content, rejection = self._validate_generated_content(
            content, file_path, original_query,
        )
        if rejection and not content.startswith("(Content generation failed:"):
            logger.warning(
                "Write: generated content rejected (%s), retrying content "
                "generation: %r", rejection, content[:120],
            )
            content = await self._generate_content(
                original_query, task_description, llm_client,
                file_path=file_path,
            )
            content, rejection = self._validate_generated_content(
                content, file_path, original_query,
            )

        # 5. 救済: 生成が失敗と確定した後に限り、緩い引用抽出で本文を拾う。
        #    誤爆リスクは「既に生成が棄却されている」状態に閉じ込めてある。
        if rejection:
            rescued = rescue_quoted_write_literal(original_query, file_path)
            if rescued:
                logger.info(
                    "Write content rescued from user quote after %s: "
                    "%d chars -> %s", rejection, len(rescued), file_path,
                )
                return rescued, None

        return content, rejection

    @staticmethod
    def _validate_generated_content(
        content: str, file_path: str, instruction: str = "",
    ) -> tuple[str, str | None]:
        """生成コンテンツを scaffold 除去してから書込み適性を検証する。

        「タスクログ + 本文」の連結は本文だけに救済し、本文が残らない
        エコー (task_log_echo / prompt_echo / instruction_echo / path_only /
        csv_without_rows) は棄却理由を返して呼出側で再生成・中断させる。

        ``instruction`` (元のユーザー依頼文) を渡すと、依頼文の逐語コピーを
        本文として書き込む退化も棄却できる。

        上書き対象が既に存在する場合は既存内容も渡し、編集依頼なのに 1 文字も
        変わっていない生成 (edit_without_change) を棄却する。書込み自体は
        成功してしまうため、これを見ないと「完了しました」と誤報告される。
        """
        if content.startswith("(Content generation failed:"):
            return content, "generation_failed"
        descaffolded = strip_prompt_scaffold_lines(content)
        if descaffolded != content:
            logger.info(
                "Write: stripped prompt scaffold labels from generated content "
                "(%d -> %d chars)", len(content), len(descaffolded),
            )
            content = descaffolded
        stripped = strip_task_log_scaffold(content)
        if stripped and stripped != content:
            logger.info(
                "Write fast path: stripped task-log scaffold from generated "
                "content (%d -> %d chars)", len(content), len(stripped),
            )
            content = stripped
        descaffolded_block = strip_generator_scaffold_block(content, file_path)
        if descaffolded_block != content:
            logger.info(
                "Write: stripped generator-directed scaffold block "
                "(%d -> %d chars)", len(content), len(descaffolded_block),
            )
            content = descaffolded_block
        without_lead_in = strip_output_lead_in(content, file_path)
        if without_lead_in != content:
            logger.info(
                "Write: stripped answer lead-in naming the output path "
                "(%d -> %d chars)", len(content), len(without_lead_in),
            )
            content = without_lead_in
        return content, generated_content_rejection(
            content, file_path, instruction,
            existing_content=read_existing_file(file_path),
        )

    async def _write_file(
        self,
        file_path: str,
        content: str,
        tools_registry,
        on_step,
        prefix: str,
    ) -> tuple[str, list[dict]]:
        """write_file を実行して結果を返す"""
        tool_args = {"file_path": file_path, "content": content}
        try:
            result = await tools_registry.execute("write_file", **tool_args)
            result_text = str(result)
            is_success = not is_tool_error(result_text)
            if is_success:
                verify_error = self._verify_written_file(file_path, content)
                if verify_error:
                    result_text = f"Error: {verify_error}"
                    is_success = False
                    logger.error(
                        "write_file post-verification failed: %s (%s)",
                        file_path, verify_error,
                    )
        except Exception as e:
            result_text = f"Error: {e}"
            is_success = False
            logger.error("write_file failed: %s", e)

        tool_entry = {
            "tool": "write_file",
            "args": tool_args,
            "success": is_success,
        }

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file: {result_text[:100]}",
                "status": "done" if is_success else "failed",
            })

        logger.info("write_file completed: %s → %s", file_path, result_text[:80])
        return result_text, [tool_entry]

    @staticmethod
    def _verify_written_file(file_path: str, content: str) -> str | None:
        """書込み後にディスク上の実ファイルを読み戻して内容を突合する。

        「Written N bytes」という成功申告と実ファイルの乖離 (書込み経路の
        取り違え・変換事故) を success にしないための最終ガード。リッチ文書
        (xlsx/docx 等) は export 変換で内容が変わるため対象外。改行は
        ``write_text`` のプラットフォーム変換 (LF→CRLF) を正規化して比較する。
        検証自体の失敗 (読み戻し不可等) は書込み失敗と区別できないため
        エラーにせず None (成功維持) を返す。
        """
        from backend.free.agent.tools.builtin import _EXPORT_DOC_EXTS

        try:
            p = Path(file_path)
            if p.suffix.lower() in _EXPORT_DOC_EXTS:
                return None
            if len(content) > 2_000_000:
                return None
            on_disk = p.read_text(encoding="utf-8")
        except Exception:
            return None
        if on_disk.replace("\r\n", "\n") != content.replace("\r\n", "\n"):
            return (
                f"post-write verification failed: on-disk content of "
                f"'{file_path}' does not match the generated content "
                f"({len(on_disk)} vs {len(content)} chars)"
            )
        return None

    async def _recover_write_from_text(
        self,
        text: str,
        task: TaskItem,
        original_query: str,
        llm_client,
        tools_registry,
        on_step,
        prefix: str,
    ) -> dict | None:
        """LLM がツールコール JSON を出力しなかった場合の自動リカバリー"""
        from backend.free.agent.tool_call_judge import _extract_file_path

        # 「同じファイルに追記して」型はタスク文にパスが無く、直前ターンに
        # しか無い。プラン生成側 (write fast path) は既に
        # ``_referential_write_path`` で会話から解決しているのに、リカバリー
        # 側はタスク文しか見ておらず非対称だった。その結果、書込みが 2 回とも
        # 失敗して「実行されませんでした」で終わる (実インシデント
        # 2026-08-08 ライブ監査 ターン6: ルータは local_write_intent へ
        # 正しく振ったが、ここで毎回 no file path になっていた)。
        # 裸のファイル名も同じく未確定として会話から解決する (2026-08-09)。
        file_path = _extract_file_path(task.description)
        file_path = self._referential_write_path(file_path or None) or file_path
        if not file_path:
            logger.warning(
                "Auto-recovery skipped: no file path in task or conversation: %s",
                task.description[:80],
            )
            return None

        # 出力先を確定 (ディレクトリ→output / bare→クエリ dir)。
        file_path = self._resolve_write_path(file_path, original_query)

        logger.info(
            "Auto-recovery: LLM returned plain text for write task, "
            "extracting content for %s",
            file_path,
        )

        # この経路だけの特殊性: モデルが既に成果物 (コード) を平文で吐いている
        # なら、それが書くべき本文そのもの。合流点より前に採用する。
        content = ""
        if not wants_fetched_table(file_path):
            candidate = strip_markdown_wrapper(text)
            if text_looks_like_code(candidate):
                validated, rejection = self._validate_generated_content(
                    candidate, file_path, original_query,
                )
                if not rejection:
                    content = validated

        # それ以外は共通の合流点へ (取得テーブル / ユーザー literal / 直前応答 /
        # 生成 + 再生成 + 救済)。この経路だけ決定論解決を持たず、実況文が本文
        # として書き込まれる穴になっていた。
        if not content:
            async def _notify_generating() -> None:
                if on_step:
                    await call_callback(on_step, {
                        "type": "tool_call",
                        "detail": f"{prefix} コンテンツ生成中（自動リカバリー） → {file_path}",
                        "status": "running",
                    })

            content, rejection = await self._resolve_write_content(
                file_path=file_path,
                original_query=original_query,
                task_description=task.description,
                llm_client=llm_client,
                notify_generating=_notify_generating,
            )
            if content.startswith("(Content generation failed:"):
                logger.warning("Auto-recovery content generation failed: %s", file_path)
                return None
            if rejection:
                logger.warning(
                    "Auto-recovery: generated content rejected (%s), aborting: %r",
                    rejection, content[:120],
                )
                return None

        if on_step:
            await call_callback(on_step, {
                "type": "tool_call",
                "detail": f"{prefix} write_file({file_path}, {len(content)}文字) [自動リカバリー]",
                "status": "running",
            })

        tool_args = {"file_path": file_path, "content": content}
        try:
            result = await tools_registry.execute("write_file", **tool_args)
            result_text = str(result)
            is_success = not is_tool_error(result_text)
            logger.info("Auto-recovery write_file: %s → %s", file_path, result_text[:100])

            if on_step:
                await call_callback(on_step, {
                    "type": "tool_call",
                    "detail": f"{prefix} write_file: {result_text[:100]}",
                    "status": "done" if is_success else "failed",
                })

            return {
                "tool": "write_file",
                "args": tool_args,
                "success": is_success,
                "result": result_text,
            }
        except Exception as e:
            logger.error("Auto-recovery write_file failed: %s", e)
            return None
