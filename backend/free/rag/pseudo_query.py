"""疑似クエリ生成 (f_01 §6.4)。

corpus チャンクごとに「このチャンクだけで答えられる問い」を補助タスクに
作らせる。応答パスからは呼ばない — sleep-time Step 5.9 が
:class:`~backend.free.rag.corpus.pseudo_queries.PseudoQueryIndex` へ書く。
"""

from __future__ import annotations

from collections.abc import Sequence

from typing import TYPE_CHECKING

from backend.free.llm.json_schemas import PseudoQueryQuestions
from backend.log_config import get_logger


class PseudoQueryPreempted(Exception):
    """チャット要求に横取りされた (予算不足ではない)。呼出側はこのサイクルを畳む。"""

if TYPE_CHECKING:
    from backend.free.llm.aux_client import AuxClient

logger = get_logger("rag.pseudo_query")

#: 問いの生成プロンプト。断片の語をそのまま写させない (言い換えの問いで
#: 引けるようにするのが目的なので、逐語の問いは本体の埋め込みと重なるだけ)。
PROMPT_TEMPLATE = (
    "次の技術文書の断片を読み、この断片だけで答えられる質問を日本語で "
    "{count} つ作ってください。開発者が実際に聞きそうな自然な言い回しで、"
    "断片の語をそのまま写さず言い換えてください。\n\n---\n{text}\n---"
)

#: 見出し経路 (文書名 › 節見出し) の前置き。断片が節の途中で主語を欠くとき、
#: 問いに主語 (「疑似クエリ索引の書き手は…」) を持たせる (f_01 §6.4)。
CONTEXT_TEMPLATE = (
    "この断片は節「{context}」の一部です。質問には節の主題が分かる語 (見出しの名詞) を"
    "含め、文書名や節番号は書かないでください。\n\n"
)

#: 取りこぼした問い (f_01 §6.4 の misses) を添えるときの追記。断片が答えられる
#: ときだけその問いを言い換えて足す — 答えられない断片に問いを貼らせない。
HINT_TEMPLATE = (
    "\n\nユーザーが実際にした問い: {hints}\n"
    "この断片がその問いに答えられる場合だけ、その問いを自然に言い換えた 1 つを"
    "上の {count} つに加えてください。答えられない場合は加えないでください。"
)

#: 1 問あたりの出力トークン見積もり (JSON の器を含む)。
_TOKENS_PER_QUESTION = 60


class PseudoQueryGenerator:
    """1 チャンク → 問いのリスト。

    Args:
        aux_client: 補助タスククライアント。
        questions_per_chunk: 1 チャンクあたりの問いの本数。
        max_chunk_chars: プロンプトへ渡す本文の上限文字数。
    """

    def __init__(
        self,
        aux_client: "AuxClient",
        *,
        questions_per_chunk: int = 2,
        max_chunk_chars: int = 1200,
    ) -> None:
        self.aux_client = aux_client
        self.questions_per_chunk = max(1, int(questions_per_chunk))
        self.max_chunk_chars = max(100, int(max_chunk_chars))

    async def generate(
        self, chunk_text: str, hint_questions: Sequence[str] = (),
        context: str = "",
    ) -> list[str]:
        """問いを返す。失敗 / 空 / 壊れた JSON は ``[]`` (呼出側が次サイクルで再試行)。

        ``hint_questions`` は取りこぼした問い (f_01 §6.4 の misses)。断片が
        答えられるときだけ言い換えを 1 つ足すよう生成側に委ねる。
        ``context`` は見出し経路 (文書名 › 節見出し)。プロンプトにだけ渡す。
        """
        text = (chunk_text or "").strip()[: self.max_chunk_chars]
        if not text:
            return []
        prompt = PROMPT_TEMPLATE.format(count=self.questions_per_chunk, text=text)
        ctx = (context or "").strip()
        if ctx:
            prompt = CONTEXT_TEMPLATE.format(context=ctx[:160]) + prompt
        hints = [h.strip() for h in hint_questions if h and h.strip()][:3]
        count = self.questions_per_chunk + (1 if hints else 0)
        if hints:
            prompt += HINT_TEMPLATE.format(
                hints=" / ".join(f"「{h[:120]}」" for h in hints),
                count=self.questions_per_chunk,
            )
        try:
            result = await self.aux_client.generate_json(
                prompt,
                max_tokens=_TOKENS_PER_QUESTION * count + 20,
                temperature=0.3,
                purpose="pseudo_query",
                list_key="questions",
                response_schema=PseudoQueryQuestions,
            )
        except TimeoutError as exc:
            if getattr(exc, "contended", False):
                # チャットに横取りされた。予算不足ではないので WARNING にせず、
                # 呼出側 (Step 5.9) にサイクルを畳ませる。
                raise PseudoQueryPreempted(str(exc)) from exc
            logger.warning("pseudo_query generation timed out")
            return []
        questions = result.get("questions") if isinstance(result, dict) else None
        if not isinstance(questions, list):
            return []
        out: list[str] = []
        for item in questions:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        # ヒント付きは言い換え 1 問ぶん多く受ける (f_01 §6.4 の misses)。
        return out[:count]

    def hinted_from(self, hint_questions: Sequence[str]) -> int | None:
        """:meth:`generate` の戻りのうちヒント由来の言い換えが始まる位置 (無ければ ``None``)。"""
        if any(h and h.strip() for h in hint_questions):
            return self.questions_per_chunk
        return None


__all__ = [
    "CONTEXT_TEMPLATE", "HINT_TEMPLATE", "PROMPT_TEMPLATE", "PseudoQueryGenerator",
    "PseudoQueryPreempted",
]
