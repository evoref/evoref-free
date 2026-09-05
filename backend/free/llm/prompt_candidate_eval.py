"""base prompt 候補の実測評価 (PromptCandidateEval)

Level 1 phase1 の採用ゲート (f_04 §4.5) の実体。候補 system prompt で失敗した
実ターンの query を **ベースモデルの専有スロット** (``background_slot``) で再生成
し、その応答を ``AuxClient`` (purpose=``prompt_candidate_judge``、文法制約 JSON)
に採点させる。現行と候補を同じケース・同じ judge で採点し、呼出側は差だけを
使う。

本クラスは ``backend.free.optimizer.prompt_eval.PromptEvalProtocol`` を**明示
継承しない** (構造的部分型で満たす)。Gen pillar が Learn pillar の Protocol を
import すると依存方向を逆転させるため、duck typing で満たし wire 時に注入する
(``EmbedInstructionEval`` と同じ)。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.free.llm.aux_client import AuxClient
from backend.free.llm.json_extract import extract_json_object
from backend.free.llm.json_schemas import PromptCandidateJudgement
from backend.free.llm.utils import extract_content
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("llm.prompt_candidate_eval")

#: 再生成の応答上限。チャット応答の典型長で足り、judge への入力も抑える。
_RESPONSE_MAX_TOKENS = 512
_RESPONSE_EXCERPT_CHARS = 1500
_QUERY_EXCERPT_CHARS = 600
_HINT_EXCERPT_CHARS = 400

_JUDGE_SYSTEM_PROMPT = (
    "あなたはアシスタント応答の評価者です。"
    "提示されたユーザー発話に対する応答が、期待された振る舞いにどれだけ沿うかを "
    "0.0〜1.0 で採点してください。"
    "評価軸: (1) 発話に正面から答えているか (2) 手元に無い情報を推測で断定せず、"
    "不明なら不明と述べているか (3) 発話や参考情報の復唱・前置き・内部ラベルが無いか "
    "(4) 期待された振る舞いのヒントがある場合、それと矛盾していないか。"
    "「訂正後に受け入れられた回答」がある場合は、それと同じ結論・同じ事実を"
    "述べているかを最も重く見てください。"
    "ヒントは「そのとき何が期待されていたか」の手掛かりであり、"
    "応答がヒントの固有値を知り得ない場合は、知らないと正直に述べる応答を"
    "捏造する応答より高く評価してください。"
    "reason は 30 字以内で簡潔に。"
)

#: judge の出力上限。score が先頭に出るので reason が切れても採点は取れる。
#: 実測 (2026-09-04) で 256 だと reason の decode に 35〜50 秒かかっていた。
_JUDGE_MAX_TOKENS = 96

_KIND_HINT_TEMPLATES: dict[str, str] = {
    "correction": "この応答の後、ユーザーは次のように訂正しました: {hint}",
    "rephrase": "ユーザーはこの後、同じ質問を言い直しました (最初の応答が噛み合っていなかった)。",
    "failed": "元の応答は途中で壊れる / 未完のまま終わりました。",
}

_BARE_SCORE_RE = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


class PromptCandidateEval:
    """候補 system prompt で失敗ケースを再生成し、judge の採点を返す。"""

    def __init__(
        self,
        llm_client: Any,
        *,
        config: dict | None = None,
        debug_logger: "DebugLogger | None" = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        """
        Args:
            llm_client: ``generate(messages, stream=False, ..., id_slot=)`` と
                ``background_slot`` を持つクライアント (``LLMClient`` ファサード)。
                judge 用の ``AuxClient`` は ``.local`` (無ければ本体) から組む。
            should_abort: ケース間で呼ぶ中断判定 (協調 yield)。True なら残りを
                打ち切り、それまでの採点だけ返す。
        """
        self._llm_client = llm_client
        self._config = config or {}
        self._debug_logger = debug_logger
        self._should_abort = should_abort
        self._judge: AuxClient | None = None

    def _resolve_judge(self) -> AuxClient | None:
        if self._judge is not None:
            return self._judge
        local = getattr(self._llm_client, "local", self._llm_client)
        if local is None or not hasattr(local, "generate_constrained"):
            return None
        self._judge = AuxClient(
            local, config=self._config, debug_logger=self._debug_logger,
        )
        return self._judge

    async def score_prompt(
        self, prompt_text: str, cases: list[Any],
    ) -> dict[str, float]:
        """``prompt_text`` を system prompt として各ケースを再生成・採点する。

        生成 / 採点に失敗したケースは結果に含めない。judge が組めない構成
        (ベース未接続 / 文法制約非対応) は空 dict。
        """
        judge = self._resolve_judge()
        if judge is None or not cases or not prompt_text.strip():
            return {}
        # 再生成を全ケース → 採点を全ケース、の 2 段にする。再生成 (候補 system
        # prompt) と judge (別 system prompt) は同じ背景スロットを使うので、
        # 交互に呼ぶと毎回 KV プレフィクスが捨てられ ~3.5k tok を再プリフィル
        # する (実測 2026-09-04: 38 字の応答に 48 秒、6 ケース × 2 で 19.5 分)。
        # 同じ system prompt を続けて投げれば 2 回目以降はプレフィクスが乗る。
        responses: list[tuple[Any, str]] = []
        for case in cases:
            if self._aborted(len(responses), len(cases), "regenerate"):
                break
            response = await self._regenerate(prompt_text, case.query)
            if response is not None:
                responses.append((case, response))
        scores: dict[str, float] = {}
        for case, response in responses:
            if self._aborted(len(scores), len(responses), "judge"):
                break
            score = await self._judge_one(aux_client=judge, case=case, response=response)
            if score is not None:
                scores[case.case_id] = score
        return scores

    def _aborted(self, done: int, total: int, stage: str) -> bool:
        if self._should_abort is None or not self._should_abort():
            return False
        logger.info(
            "prompt candidate eval aborted (yield requested) at %s after %d/%d",
            stage, done, total,
        )
        return True

    async def _regenerate(self, prompt_text: str, query: str) -> str | None:
        """候補 system prompt で query を再生成する (greedy、背景スロット)。"""
        try:
            result = await self._llm_client.generate(
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": query},
                ],
                stream=False,
                temperature=0.0,
                max_tokens=_RESPONSE_MAX_TOKENS,
                id_slot=getattr(self._llm_client, "background_slot", -1),
            )
        except Exception as exc:  # noqa: BLE001 - 1 ケースの失敗は欠測にする
            logger.warning("prompt candidate regenerate failed: %r", exc)
            return None
        if not isinstance(result, dict):
            return None
        content = (extract_content(result) or "").strip()
        return content or None

    async def _judge_one(
        self, *, aux_client: AuxClient, case: Any, response: str,
    ) -> float | None:
        kind = getattr(case, "kind", "")
        hint = (getattr(case, "hint", "") or "")[:_HINT_EXCERPT_CHARS]
        tmpl = _KIND_HINT_TEMPLATES.get(kind, "")
        hint_line = tmpl.format(hint=hint) if tmpl else ""
        reference = (getattr(case, "reference", "") or "")[:_HINT_EXCERPT_CHARS]
        user = (
            f"ユーザー発話: {case.query[:_QUERY_EXCERPT_CHARS]}\n"
            f"応答: {response[:_RESPONSE_EXCERPT_CHARS]}\n"
            + (f"ヒント: {hint_line}\n" if hint_line else "")
            # 訂正後にユーザーが受け入れた回答。訂正文 (「違います」だけの
            # こともある) より具体的な「期待された振る舞い」の証拠になる。
            + (f"訂正後に受け入れられた回答: {reference}\n" if reference else "")
        )
        try:
            result = await aux_client.generate(
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=_JUDGE_MAX_TOKENS,
                temperature=0.1,
                purpose="prompt_candidate_judge",
                response_schema=PromptCandidateJudgement,
            )
        except Exception as exc:  # noqa: BLE001 - 採点失敗は欠測にする
            logger.warning("prompt candidate judge failed: %r", exc)
            return None
        content = extract_content(result) if isinstance(result, dict) else ""
        parsed = extract_json_object(content)
        raw: Any = parsed.get("score") if isinstance(parsed, dict) else None
        if raw is None:
            m = _BARE_SCORE_RE.search(content or "")
            raw = m.group(0) if m else None
        try:
            score = float(raw)
        except (TypeError, ValueError):
            logger.debug("prompt candidate judge: unparseable: %s", (content or "")[:120])
            return None
        if not 0.0 <= score <= 1.0:
            return None
        return score


__all__ = ["PromptCandidateEval"]
