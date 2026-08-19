"""ContentGenerationMixin — meta_cognitive_content"""

from __future__ import annotations

import asyncio
import time

from pathlib import Path
from backend.config import resolve_context_size_for_mode
from backend.free.agent.context_budget import SEND_GUARD_RESERVE_TOKENS
from backend.free.core.prompt_blocks import current_datetime_block
from backend.free.agent.output_format import (
    is_rich_table_output,
    is_table_output,
)
from backend.free.agent.meta_cognitive_utils import (
    EXISTING_CONTENT_BLOCK_HEADING,
    FETCHED_DATA_BLOCK_HEADING,
    FETCHED_DATA_BLOCK_NOTE,
    content_language_directive,
    fewshot_contains_task_log,
    fewshot_seems_relevant,
    looks_like_task_log_residue,
    strip_markdown_wrapper,
    truncate_repetition,
    unwrap_sole_code_fence,
)
from backend.utils import estimate_tokens as _estimate_tokens

from backend.free.agent.meta_cognitive_defs import (
    CONTENT_GENERATION_PROMPT,
    CSV_CONTENT_INSTRUCTION,
    MARKDOWN_CONTENT_INSTRUCTION,
    RICH_DOC_CONTENT_INSTRUCTION,
    TABLE_CONTENT_INSTRUCTION,
    _PRIOR_CONTENT_REFERENCE_RE,
)

from backend.log_config import get_logger

logger = get_logger("agent.meta_cognitive")


class _ContentGenerationMixin:
    """write_file へ渡す本文の生成。

    プロンプト組み立て (既存内容 / 取得済みデータ / 日時 / few-shot の注入)、
    トークン予算の算出、ストリーミング生成と後処理をまとめる。
    """

    async def _generate_content(
        self,
        original_query: str,
        task_description: str,
        llm_client,
        file_path: str = "",
    ) -> str:
        """write_file 用のコンテンツを LLM に生成させる"""
        ctx_size = resolve_context_size_for_mode(self.config, self._mode)

        existing_content = self._read_existing_file(file_path)
        user_prompt = f"{original_query}\n\nタスク: {task_description}"
        # 「この案内文を保存して」のように直前の成果物を指す依頼は、クエリと
        # タスク文だけでは書くべき本文が決まらない。素材が無いままだと小型
        # モデルは system の few-shot 書式を真似た架空の Q&A を本文として
        # 書き出す (実インシデント 2026-07-27 → looks_like_fewshot_echo)。
        # ただし既存ファイルがある場合は _inject_existing_content が同じ役割の
        # 素材を渡すので重ねない。ラベル付きブロックを増やすほど小型モデルは
        # ラベルごと本文へ書き写す (実測 2026-07-27: 上書き依頼で
        # 「[現在の日付 (UTC基準)] …」「[直近の会話] …」がファイルに出た)。
        #
        # 例外は「先ほどの〜の内容で上書きして」型。書くべき本文は会話にあり、
        # 既存内容はむしろ捨てたいものなので、既存内容だけを素材にすると
        # 別物を再生成する (実インシデント 2026-08-10 ライブ監査: 断り書きが
        # 書き込まれたファイルを「先ほどの JSON Schema の内容で上書きして」と
        # 直させたら、draft-07 の別スキーマを作って書いた。同じ会話の次の
        # ターンではコード生成が正しい方のスキーマを使えており、素材が
        # 渡っていないことが原因だと確定した)。
        if not existing_content or _PRIOR_CONTENT_REFERENCE_RE.search(user_prompt):
            user_prompt = self._inject_recent_conversation(
                user_prompt, getattr(self, "_conversation", None), ctx_size,
            )
        # 「今月」「今日」等の相対表現を取り違えないよう現在日付を前置する
        # (カレンダー/予定表など日付依存の成果物で年月・曜日がズレるのを防ぐ)。
        user_prompt = self._inject_current_date(user_prompt)
        # 前ステップで取得した実データ (fetch_url 等) を「使うべき素材」として注入し、
        # データに無い内容の創作 (ハルシネーション) を抑止する。
        user_prompt = self._inject_fetched_data(user_prompt, ctx_size)
        user_prompt = self._inject_existing_content(
            user_prompt, existing_content, file_path, ctx_size,
        )

        # Level 1 で進化した few-shot を参考例として system に注入する
        system_content = CONTENT_GENERATION_PROMPT
        # 出力先がスプレッドシート系なら GFM 表生成を強制する (xlsx 実セル化のため)。
        # .csv は raw 書込みのため CSV 行そのものを出力させる。
        # リッチ文書系 (docx/pptx) は表を強制せず、コード文字列の出力のみ禁じる。
        if file_path.lower().endswith(".csv"):
            system_content = f"{system_content}\n{CSV_CONTENT_INSTRUCTION}"
        elif is_table_output(file_path):
            system_content = f"{system_content}\n{TABLE_CONTENT_INSTRUCTION}"
        elif is_rich_table_output(file_path):
            system_content = f"{system_content}\n{RICH_DOC_CONTENT_INSTRUCTION}"
        elif file_path.lower().endswith((".md", ".markdown")):
            system_content = f"{system_content}\n{MARKDOWN_CONTENT_INSTRUCTION}"
        # 出力言語指示 (locale 追従)。ここは write 全経路 (ツールループ /
        # ファストパス / auto-recovery / editor タスク) の合流点なので、
        # この 1 箇所で全ファイル出力に効く。
        system_content = f"{system_content}\n{content_language_directive()}"
        # Level 1 few-shot は query 類似度だけでなく fitness も加味して選ばれる
        # ため、現在のタスクと無関係でも再利用されうる。無関係な例文をその
        # まま注入すると、モデルがその例文自体を繰り返す退化を誘発しうる
        # (#incident) ため、タスク文との粗い関連度チェックを通してから注入する。
        # さらに応答例がタスク進捗ノート形式 (- [done] ... Written N bytes) の
        # 場合は「報告だけ出せば正解」バイアスを与えるため注入しない
        # (#incident 2026-07-15: 本文なし極小ファイル 10 件)。
        if (
            self._fewshot_block
            and fewshot_seems_relevant(
                f"{original_query}\n{task_description}", self._fewshot_block,
            )
            and not fewshot_contains_task_log(self._fewshot_block)
        ):
            system_content = f"{system_content}\n\n[参考例]\n{self._fewshot_block}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        gen_max_tokens = self._calc_gen_max_tokens(
            system_content + user_prompt, ctx_size,
        )

        return await self._stream_and_clean(
            llm_client, messages, gen_max_tokens,
            preserve_markdown=file_path.lower().endswith((".md", ".markdown")),
        )

    #: 生成プロンプトに添える直近会話の上限。
    #:
    #: 以前は **メッセージ数 4 固定** だった ("保存対象は直前に作った成果物で
    #: あることが大半なので深い履歴は不要" という前提)。その前提は
    #: 「ここまでの試算を3行で」のように **会話をさかのぼって集計する依頼** で
    #: 崩れる。実インシデント (2026-08-09 2 回目のライブ監査): 必要な値
    #: (480人 / 3200円 / 1536000円) がすべて 4 メッセージ窓の外にあり、視界に
    #: 残っていた直前のツール結果 ``3,025`` を 3 項目すべてに貼り、さらに
    #: 3025×3025=9,150,625 まで計算してファイルに書いた。しかもユーザーには
    #: 「書き込みました」と成功報告される。
    #:
    #: ``ctx_size=8192`` に対し実際の入力は ~443 トークンで予算は大量に
    #: 余っていた。数で切らず予算で切る。
    _CONTENT_CONTEXT_CHARS = 1500          # 1 メッセージあたりの上限
    _CONTENT_CONTEXT_MAX_MESSAGES = 30     # 暴走防止 (WorkingMemory の窓と同じ)
    _CONTENT_CONTEXT_CTX_FRACTION = 3      # ctx_size の 1/3 を会話素材へ
    _CONTENT_CONTEXT_FALLBACK_CHARS = 6000  # ctx_size 不明時

    @classmethod
    def _content_context_budget_chars(cls, ctx_size: int) -> int:
        """会話素材ブロックに使える文字数 (純粋関数)。

        トークン予算を char へ概算換算する (日本語混在で ~2 char/token)。
        """
        if ctx_size <= 0:
            return cls._CONTENT_CONTEXT_FALLBACK_CHARS
        return max(
            cls._CONTENT_CONTEXT_CHARS,
            (ctx_size // cls._CONTENT_CONTEXT_CTX_FRACTION) * 2,
        )

    @staticmethod
    def _strip_task_log_lines(text: str) -> str:
        """会話素材から進捗ノート行を落とす (純粋関数)。

        素材に「- [done] …」「Written N bytes to …」が混ざっていると、小型
        モデルはその**書式ごと**真似て本文の代わりに進捗ノートを書く。生成物は
        ``looks_like_task_log_echo`` が弾くので壊れたファイルは残らないが、
        再生成しても同じ素材を見るので**書込みが 2 回とも失敗する**
        (実インシデント 2026-08-10 ライブ監査: 「そのテストコードを
        E:\\tmp\\test_overlap.py に保存して」が 2 回とも
        ``- [done] …\\n    Written 1385 bytes to …`` を生成して中断)。
        """
        if not text or not looks_like_task_log_residue(text):
            return text
        return "\n".join(
            ln for ln in text.split("\n") if not looks_like_task_log_residue(ln)
        )

    @classmethod
    def _inject_recent_conversation(
        cls, user_prompt: str, conversation: list[dict] | None,
        ctx_size: int = 0,
    ) -> str:
        """直近会話を「素材」として user_prompt に前置する (純粋関数)。

        新しい発言から予算いっぱいまで遡って採る。予算に関係なく最低 1 件は
        載せる (極小 ctx_size でも従来同様に直前の成果物は渡す)。
        """
        if not conversation:
            return user_prompt
        budget = cls._content_context_budget_chars(ctx_size)
        picked: list[str] = []
        used = 0
        for msg in reversed(conversation[-cls._CONTENT_CONTEXT_MAX_MESSAGES:]):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            text = cls._strip_task_log_lines(content).strip()
            if not text:
                continue
            speaker = "ユーザー" if role == "user" else "あなた"
            line = f"{speaker}: {text[:cls._CONTENT_CONTEXT_CHARS]}"
            if picked and used + len(line) > budget:
                break
            picked.append(line)
            used += len(line)
        if not picked:
            return user_prompt
        block = "\n".join(reversed(picked))
        return (
            "[直近の会話] 依頼が「この文章」「さっきの案」等を指す場合、"
            "書くべき本文はこの会話の中にある。会話に無い内容を新たに創作しないこと。\n"
            f"{block}\n\n{user_prompt}"
        )

    @staticmethod
    def _inject_current_date(user_prompt: str) -> str:
        """現在日付 (UTC 基準) を user_prompt 先頭に前置する。

        ``_generate_content`` は履歴を持たないため、モデルは現在日付を知らない。
        「今月のカレンダー」のような相対日付依存の生成で年月・曜日を取り違え
        ないよう、現在日付と曜日を明示する。内部時刻不変則 (naive 禁止) に従い
        ``utc_now_dt()`` を使う。
        """
        date_ctx = current_datetime_block(
            "「今月」「今日」等の相対表現はこの日付を基準に解釈すること。",
        )
        return f"{date_ctx}\n\n{user_prompt}"

    @staticmethod
    def _read_existing_file(file_path: str) -> str:
        """既存ファイルの内容を読み込む（存在しなければ空文字列）"""
        if not file_path:
            return ""
        p = Path(file_path)
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
        return ""

    @staticmethod
    def _inject_existing_content(
        user_prompt: str,
        existing_content: str,
        file_path: str,
        ctx_size: int,
    ) -> str:
        """既存ファイル内容をプロンプトに注入する"""
        if not existing_content:
            return user_prompt
        base_tokens = _estimate_tokens(
            CONTENT_GENERATION_PROMPT + user_prompt
        )
        existing_tokens = _estimate_tokens(existing_content)
        if base_tokens + existing_tokens < ctx_size // 2:
            # 差し替え規則は既存内容ブロックの「後」に置く。前に置くと 9B base が
            # 従わず、古い項目の括弧書きだけが残った (2026-07-28 実測)。
            user_prompt += (
                f"\n\n{EXISTING_CONTENT_BLOCK_HEADING} ({file_path})\n"
                f"以下の既存内容を含め、タスクの内容を統合した完全なファイルを出力してください。\n"
                f"```\n{existing_content}\n```\n"
                f"差し替え規則: ある項目を別の項目に置き換えるときは、行を丸ごと"
                f"書き直すこと。古い項目に付いていた補足 (括弧書き・単位・説明) を"
                f"新しい項目に流用してはいけない。新しい項目に合う補足を書くか、"
                f"補足なしにする。\n"
                f"例: 「水筒（500ml〜1L）」を「ヘッドランプ」に差し替える → "
                f"「ヘッドランプ（予備電池も）」または「ヘッドランプ」。"
                f"「ヘッドランプ（500ml〜1L）」は誤り。"
            )
        else:
            logger.info(
                "Skipping existing content injection: "
                "base=%d + existing=%d tokens > ctx_size/2=%d",
                base_tokens, existing_tokens, ctx_size // 2,
            )
        return user_prompt

    def _inject_fetched_data(self, user_prompt: str, ctx_size: int) -> str:
        """タスク横断で取得したツール結果を「使うべき実データ」として注入する。

        コンテキスト予算 (おおよそ ctx_size/2) に収まる範囲で取得データを付与し、
        モデルにデータ由来の出力を促す。予算不足なら付与しない (安全縮退)。
        """
        outputs = getattr(self, "_fetched_tool_outputs", [])
        if not outputs:
            return user_prompt
        combined = "\n\n".join(outputs)
        base_tokens = _estimate_tokens(CONTENT_GENERATION_PROMPT + user_prompt)
        budget_tokens = ctx_size // 2 - base_tokens
        if budget_tokens < 100:
            return user_prompt
        # token 予算をおおまかに char 予算へ換算 (日本語混在で ~2 char/token 見込み)
        snippet = combined[: budget_tokens * 2]
        return user_prompt + (
            f"\n\n{FETCHED_DATA_BLOCK_HEADING}\n"
            f"{FETCHED_DATA_BLOCK_NOTE}\n"
            f"{snippet}"
        )

    def _calc_gen_max_tokens(
        self, prompt_text: str, ctx_size: int,
    ) -> int:
        """コンテンツ生成用の max_tokens を計算する

        ``SEND_GUARD_RESERVE_TOKENS`` を残すのは、送信直前ガード
        (``LocalClient._enforce_context_budget``) が
        ``budget = n_ctx - マージン - max_tokens`` でプロンプト上限を決めるため。
        残さないと budget が必ずプロンプト長を下回り、**このパスのプロンプトが
        毎回中略される** (2026-08-19 ライブ監査で実測)。
        """
        input_tokens = _estimate_tokens(prompt_text)
        available = max(
            ctx_size - input_tokens - 128 - SEND_GUARD_RESERVE_TOKENS, 1024,
        )
        gen_max_tokens = min(self._execute_max_tokens, available)
        logger.debug(
            "Content generation: input_tokens≈%d, ctx_size=%d, max_tokens=%d",
            input_tokens, ctx_size, gen_max_tokens,
        )
        return gen_max_tokens

    async def _stream_and_clean(
        self,
        llm_client,
        messages: list[dict],
        gen_max_tokens: int,
        *,
        preserve_markdown: bool = False,
    ) -> str:
        """LLM ストリーミング生成 + 後処理（フェンス除去・繰り返し切除）

        ``preserve_markdown`` は出力先が Markdown 文書 (.md/.markdown) の場合に
        立てる。``strip_markdown_wrapper`` は「最長のフェンス内」だけを取り出す
        ため、見出し・散文・```python フェンスが混在する正しい Markdown を渡すと
        **地の文を全部落としてコードだけ** を残す (実インシデント 2026-08-14
        ライブ監査 ターン13-14)。Markdown 文書ではフェンスは成果物の一部なので、
        「全体が 1 つのフェンス」のときだけ剥がす ``unwrap_sole_code_fence`` に
        切り替える。

        最初の1トークンまでは別枠の長めタイムアウト
        (``content_gen_first_token_timeout``) で待つ。これは llama-server が他の
        生成で busy な間にキュー待ちしているリクエストを「停止」と誤判定しないため。
        2トークン目以降はトークン間アイドル (無出力) タイムアウト
        (``content_gen_idle_timeout``) で「停止したストリーム」を素早く諦めつつ、
        低速だが進行中の生成は総上限 (``content_gen_timeout``) まで継続させる。
        総ウォールクロックで一律に打ち切らない。
        """
        stream = await llm_client.generate(
            messages, stream=True,
            max_tokens=gen_max_tokens,
            id_slot=getattr(llm_client, 'background_slot', -1),
        )
        agen = stream.__aiter__()
        chunks: list[str] = []
        start = time.monotonic()
        first_token = True
        try:
            while True:
                wait_timeout = (
                    self._content_gen_first_token_timeout
                    if first_token
                    else self._content_gen_idle_timeout
                )
                try:
                    token = await asyncio.wait_for(
                        agen.__anext__(),
                        timeout=wait_timeout,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    if first_token:
                        logger.warning(
                            "Content generation produced no first token within "
                            "%ds (llama-server likely busy with another generation)",
                            self._content_gen_first_token_timeout,
                        )
                        return (
                            f"(Content generation failed: no output within "
                            f"{self._content_gen_first_token_timeout}s)"
                        )
                    logger.warning(
                        "Content generation stalled (no token for %ds)",
                        self._content_gen_idle_timeout,
                    )
                    return (
                        f"(Content generation failed: stalled after "
                        f"{self._content_gen_idle_timeout}s without output)"
                    )
                first_token = False
                chunks.append(token)
                if time.monotonic() - start > self._content_gen_timeout:
                    logger.warning(
                        "Content generation exceeded total cap %ds",
                        self._content_gen_timeout,
                    )
                    return f"(Content generation failed: timeout after {self._content_gen_timeout}s)"
            raw = "".join(chunks).strip()
            content = (
                unwrap_sole_code_fence(raw)
                if preserve_markdown
                else strip_markdown_wrapper(raw)
            )
            content = truncate_repetition(content)
            if not content:
                logger.warning("Content generation returned empty content")
                return "(Content generation failed: empty output)"
            logger.debug("Content generated: %d chars", len(content))
            return content
        except Exception as e:
            logger.error("Content generation failed: %s", e)
            return f"(Content generation failed: {e})"
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
