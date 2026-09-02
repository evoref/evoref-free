"""Contextual Retrieval プレフィックス生成

Anthropic 方式の Contextual Retrieval を実装する。
チャンク登録時に補助タスクでドキュメント全体の文脈を要約した
50-100 トークンのプレフィックスを付加し、検索精度を向上させる。

参考: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.llm.aux_client import AuxClient

logger = get_logger("rag.contextual_prefix")

# 連続失敗でログレベルを引き上げる閾値
_CONSECUTIVE_FAIL_WARN_THRESHOLD = 3

# プレフィックス生成プロンプトテンプレート (default = 日本語)
# ``rag.contextual_prefix.prompt_template`` を config で空以外に設定すると
# そちらが優先される。プレースホルダは ``{document}`` ``{chunk}`` の 2 つ。
_DEFAULT_PREFIX_PROMPT = """\
<document>
{document}
</document>

上記ドキュメントの一部であるチャンクについて、\
検索精度を向上させるための短い文脈説明（50-100トークン）を生成してください。\
チャンクがドキュメント内でどの位置にあり、何について述べているかを簡潔に説明してください。

<chunk>
{chunk}
</chunk>"""


class ContextualPrefixGenerator:
    """補助タスクを使ったコンテキストプレフィックス生成"""

    def __init__(
        self,
        aux_client: AuxClient,
        config: dict,
    ):
        self.aux_client = aux_client
        rag_cfg = config.get("rag", {})
        # 設定は rag.contextual_prefix 配下のネスト構造
        cp_cfg = rag_cfg.get("contextual_prefix", {}) or {}
        self.max_tokens: int = int(cp_cfg.get("max_tokens", 128))
        self.max_doc_chars: int = int(cp_cfg.get("max_doc_chars", 6000))
        # プロンプトテンプレート。空 (既定) なら _DEFAULT_PREFIX_PROMPT を使用。
        # 多言語対応 / 英語専用補助タスク切替時はここを config で上書き。
        prompt_template = str(cp_cfg.get("prompt_template", "") or "").strip()
        self._prompt_template: str = prompt_template or _DEFAULT_PREFIX_PROMPT
        self._consecutive_failures: int = 0

    async def generate_prefix(
        self, document_text: str, chunk_text: str,
    ) -> str:
        """1 チャンク分のコンテキストプレフィックスを生成

        Args:
            document_text: ソースドキュメントの全文（上限超過時はトランケート済み）
            chunk_text: プレフィックスを付加するチャンクのテキスト

        Returns:
            50-100 トークンの文脈説明テキスト。生成失敗時は空文字列。
        """
        doc_truncated = document_text[:self.max_doc_chars]

        prompt = self._prompt_template.format(
            document=doc_truncated,
            chunk=chunk_text,
        )

        try:
            # cache_prompt=True: 同一ソースの連続チャンクでドキュメント部分の
            # prefill を llama-server 側の KV キャッシュで再利用するため有効化
            # する。prompt 構造は {document(固定)} ... {chunk(可変)} の順なので、
            # 2 チャンク目以降はドキュメント部分の prefill がほぼゼロになる。
            result = await self.aux_client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.3,
                purpose="contextual_prefix",
                cache_prompt=True,
            )
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            prefix = content.strip()
            self._consecutive_failures = 0
            logger.debug(
                "Generated prefix: chunk=%s, prefix_len=%d",
                chunk_text[:50], len(prefix),
            )
            return prefix
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _CONSECUTIVE_FAIL_WARN_THRESHOLD:
                logger.warning(
                    "Failed to generate prefix (transient, %d consecutive): %s",
                    self._consecutive_failures, e,
                )
            else:
                logger.info(
                    "Failed to generate prefix (transient): %s", e,
                )
            return ""
        except (KeyError, ValueError) as e:
            self._consecutive_failures += 1
            logger.error(
                "Failed to generate prefix (config/response format): %s", e,
            )
            return ""
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                "Failed to generate prefix (unexpected, %d consecutive): %s",
                self._consecutive_failures, e,
            )
            return ""
