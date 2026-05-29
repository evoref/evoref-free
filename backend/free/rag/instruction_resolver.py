"""instruction-aware モデル共通の instruction 解決・テンプレート整形ヘルパ。

Qwen3-Embedding (``embedding_llamacpp``) と Qwen3-Reranker
(``reranker_llamacpp``) で AST 一致していた ``_resolve_instruction`` /
テンプレート整形ロジックを集約する (EvorefGen / RAG pillar 内)。

ログのバックエンド識別子は呼び出し側が ``backend_label`` で渡す
(``"embed_query"`` / ``"rerank"`` 等。元実装のログ prefix を保つ)。
"""

from __future__ import annotations

from backend.log_config import get_logger

logger = get_logger("rag.instruction_resolver")

# モード未指定時のデフォルト ("chat")。
DEFAULT_MODE = "chat"

# instructions 設定が空 / 不正な場合のフォールバック (起動失敗回避)。
FALLBACK_INSTRUCTION = (
    "Given a user question, retrieve relevant passages that answer the query"
)


def resolve_instruction(
    mode: str,
    instructions: dict[str, str],
    *,
    backend_label: str,
) -> str:
    """``mode`` から instruction 文字列を解決する。

    順序: ``instructions[mode]`` → ``instructions["chat"]`` →
    :data:`FALLBACK_INSTRUCTION`。
    """
    if mode in instructions:
        return instructions[mode]
    if DEFAULT_MODE in instructions:
        logger.warning(
            "%s: unknown mode=%r, falling back to %r",
            backend_label, mode, DEFAULT_MODE,
        )
        return instructions[DEFAULT_MODE]
    logger.warning(
        "%s: no instructions configured (mode=%r), using fallback",
        backend_label, mode,
    )
    return FALLBACK_INSTRUCTION


def format_with_instruction(
    text: str,
    template: str,
    instructions: dict[str, str],
    mode: str,
    *,
    backend_label: str,
) -> str:
    """テンプレート駆動でテキストを整形する。

    ``template`` が空文字列なら素のテキストを返す (BGE 系等の
    非 instruction-aware モデル向け fast-path)。それ以外は
    ``template.format(task=<instruction>, query=text)``。
    """
    if not template:
        return text
    instruction = resolve_instruction(mode, instructions, backend_label=backend_label)
    return template.format(task=instruction, query=text)
