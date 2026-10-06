"""問いの条件 (限定語・期間・単位・対象) を逐語で抜き出す — 判定点 ``answer_conditions``。

資料を見せたターンの読み違いの多くは、資料ではなく **問いの限定語の取りこぼし**
から来る (PipesHub の失敗分析: 見せた資料の読み違い 14 件のうち 9 件 — 「solo」を
無視、「retired from」を「最後の所属」に置換、時点の取り違え)。問いが答えに課す
条件を発話から逐語で抜き、生クエリの直後へ「満たすべき条件」として注記する
(``core.inference._answer_conditions_note``、docs/f_03 §7.1.1)。

- 抜き出しは補助タスク ``answer_conditions`` (文法制約 JSON、``CHAT_PATH_PURPOSES``
  なので分類器スロット)。検索と並走させ、**検索の回収が終わった時点で未完なら捨てる**
  (組み立てを待たせない。ツール判定と分類器スロットを取り合うので棄権が増えうる)。
- **逐語の門**: span が発話に逐語で在るものだけ残す (言い換えを通さない)。照合の
  正規化は訂正の門と同じ 1 本 (``correction_verdict.norm_span``、#14 (a))。
- 0 件・JSON の失敗・未完・``aux_client=None`` はすべて **棄権 = 注記なし**
  (従来の動作)。
- 注記は条件を付け足すだけで、資料の証拠を置き換えない (#15)。

判定点の記録 (``decision.jsonl``) には抜いた件数・門で落ちた件数・所要 ms を
``detail`` で残す (発話の本文は残さない)。
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from backend.free.core.correction_verdict import norm_span
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.log_config import get_logger

logger = get_logger("core.answer_conditions")

PREDICATE_NAME = "answer_conditions"
#: 補助タスクの purpose (``aux_client.PURPOSE_TIMEOUT_DEFAULTS`` / ``PURPOSE_SCHEMAS``)。
PURPOSE = "answer_conditions"
#: 注記を付けた帯の値。
NOTE_LABEL = "note"
MAX_CONDITIONS = 5
MAX_SPAN_CHARS = 40
#: span の最小長 (空白を除いた正規化後)。「人」「in」のような 1 文字は条件にならない。
MIN_SPAN_CHARS = 2
#: 条件 5 件 × (span 40 字 + kind) の JSON に足りる量。
_MAX_TOKENS = 320

ConditionKind = Literal["qualifier", "period", "unit", "target", "ref_date"]
KINDS: frozenset[str] = frozenset({"qualifier", "period", "unit", "target", "ref_date"})

Status = Literal[
    "ok", "timeout", "not_ready", "failed", "empty", "aux_unavailable", "no_evidence",
]

_SYSTEM = (
    "You list the conditions a question places on its answer. Do not answer the "
    "question. Return JSON only.\n"
    "- conditions: at most 5 items. Each span must be copied character for character "
    "from the question (no paraphrase, no translation, at most 40 characters).\n"
    "- kind: \"qualifier\" (a word that narrows which things count, e.g. solo, "
    "surviving, first, only, excluding), \"period\" (a time range the answer must "
    "hold for, e.g. in 2023, during the war), \"unit\" (a unit or form the value "
    "must be given in, e.g. in km, in yen, as a percentage), \"target\" (the specific "
    "entity asked about when several may appear), \"ref_date\" (the point in time "
    "the answer is evaluated at, e.g. as of August 1, 2024, at the time of "
    "retirement).\n"
    "- Do not list the topic itself or the question word. Return an empty list when "
    "the question has no such condition."
)


def answer_conditions_enabled(cfg: Mapping[str, Any] | None) -> bool:
    """``prompt.answer_conditions`` が ``on`` か (既定 ``off``)。"""
    prompt = (cfg or {}).get("prompt") or {}
    return isinstance(prompt, Mapping) and prompt.get("answer_conditions") == "on"


@dataclass(frozen=True, slots=True)
class Condition:
    span: str
    kind: str


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """抜き出し 1 回の結果。``conditions`` は逐語の門を通った分だけ。"""

    status: Status
    conditions: tuple[Condition, ...] = ()
    #: モデルが返した件数 (門の前)。
    extracted: int = 0
    #: 門で落ちた件数。
    dropped: int = 0
    elapsed_ms: int = 0

    @property
    def spans(self) -> tuple[str, ...]:
        return tuple(c.span for c in self.conditions)


def _ascii_word_in(span: str, text: str) -> bool:
    """ASCII の span が語の途中でなく現れるか (前後が英数字でない。空白の差は無視)。"""
    words = unicodedata.normalize("NFKC", span).lower().split()
    if not words:
        return False
    pattern = (
        r"(?<![a-z0-9])" + r"\s*".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    )
    return re.search(pattern, unicodedata.normalize("NFKC", text).lower()) is not None


def gate_conditions(text: str, items: Any) -> tuple[tuple[Condition, ...], int]:
    """逐語の門 (純粋関数)。``(残った条件, 落とした件数)`` を返す。

    残すのは: 形が正しく (``span`` / ``kind`` が文字列で ``kind`` が既知)、span が
    2〜40 字で、発話に逐語で在り (空白と全角半角の違いだけを無視。ASCII の span は
    語の途中に当たらないこと)、発話全体ではなく、既に残した span と重ならないもの。
    最大 :data:`MAX_CONDITIONS` 件。
    """
    if not isinstance(items, list):
        return (), 0
    haystack = norm_span(text)
    whole = haystack
    kept: list[Condition] = []
    seen: set[str] = set()
    dropped = 0
    for item in items:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        span = item.get("span")
        kind = item.get("kind")
        if not isinstance(span, str) or not isinstance(kind, str) or kind not in KINDS:
            dropped += 1
            continue
        span = span.strip()
        key = norm_span(span)
        if (
            len(key) < MIN_SPAN_CHARS
            or len(span) > MAX_SPAN_CHARS
            or key not in haystack
            or (key.isascii() and not _ascii_word_in(span, text))
            or key == whole
            or key in seen
            or len(kept) >= MAX_CONDITIONS
        ):
            dropped += 1
            continue
        seen.add(key)
        kept.append(Condition(span=span, kind=kind))
    return tuple(kept), dropped


async def extract_answer_conditions(aux_client: Any, text: str) -> ExtractionOutcome:
    """補助タスクで条件を抜き、逐語の門に掛ける (待つかどうかは呼出側)。例外を投げない。"""
    started = time.monotonic()

    def _ms() -> int:
        return int((time.monotonic() - started) * 1000)

    if aux_client is None or not (text or "").strip():
        return ExtractionOutcome(status="aux_unavailable")
    try:
        raw = await aux_client.generate_json(
            f"[question] {text.strip()}",
            system=_SYSTEM,
            purpose=PURPOSE,
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return ExtractionOutcome(status="timeout", elapsed_ms=_ms())
    except Exception as exc:  # noqa: BLE001 - 注記を諦めて従来の動作へ倒す
        logger.info("Answer-condition extraction failed: %s", exc)
        return ExtractionOutcome(status="failed", elapsed_ms=_ms())
    items = raw.get("conditions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return ExtractionOutcome(status="empty", elapsed_ms=_ms())
    kept, dropped = gate_conditions(text, items)
    return ExtractionOutcome(
        status="ok", conditions=kept, extracted=len(items),
        dropped=dropped, elapsed_ms=_ms(),
    )


@dataclass(slots=True)
class PendingExtraction:
    """検索と並走中の抜き出し。:func:`take_answer_conditions` か :meth:`discard` で閉じる。"""

    task: asyncio.Task
    started: float = field(default_factory=time.monotonic)

    def discard(self) -> None:
        if not self.task.done():
            self.task.cancel()

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)


def start_answer_conditions(
    aux_client: Any, text: str, cfg: Mapping[str, Any] | None,
) -> PendingExtraction | None:
    """設定が ``on`` のときだけ抜き出しを起動する (``off`` は補助タスクを呼ばない)。

    ``aux_client=None`` (degraded) でも起動はする — 即座に ``aux_unavailable`` で
    終わり、棄権として記録される。
    """
    if not answer_conditions_enabled(cfg):
        return None
    return PendingExtraction(
        task=asyncio.create_task(extract_answer_conditions(aux_client, text)),
    )


def take_answer_conditions(pending: PendingExtraction) -> ExtractionOutcome:
    """**待たずに** 結果を取る。終わっていなければ捨てて ``not_ready`` (棄権)。

    呼ぶのは検索の回収 (と記憶の注入) が終わった後。組み立てを抜き出しのために
    止めない — 並走できた範囲で終わった結果だけを使う。
    """
    task = pending.task
    if not task.done():
        pending.discard()
        return ExtractionOutcome(status="not_ready", elapsed_ms=pending.elapsed_ms())
    if task.cancelled():
        return ExtractionOutcome(status="not_ready", elapsed_ms=pending.elapsed_ms())
    exc = task.exception()
    if exc is not None:  # 抜き出し側は例外を投げない契約だが念のため
        logger.info("Answer-condition extraction task failed: %s", exc)
        return ExtractionOutcome(status="failed", elapsed_ms=pending.elapsed_ms())
    return task.result()


class _OutcomeStage:
    """抜き出しの結果を帯にする段 (補助タスクの段。往復は呼出側が並走で済ませる)。"""

    name = PREDICATE_NAME

    def evaluate(
        self, text: str, ctx: Mapping[str, Any] | None = None,  # noqa: ARG002 - Predicate の面
    ) -> Verdict:
        outcome = (ctx or {}).get("outcome")
        if not isinstance(outcome, ExtractionOutcome):
            return Verdict(
                value=None, score=0.0, band="abstain", evidence="no_outcome",
                predicate=self.name, stage="aux",
            )
        detail = {
            "extracted": outcome.extracted,
            "kept": len(outcome.conditions),
            "dropped": outcome.dropped,
            "elapsed_ms": outcome.elapsed_ms,
        }
        if outcome.status == "no_evidence":
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip", evidence="no_evidence",
                predicate=self.name, stage="aux", detail=detail,
            )
        if outcome.status != "ok":
            return Verdict(
                value=None, score=0.0, band="abstain", evidence=outcome.status,
                predicate=self.name, stage="aux", detail=detail,
            )
        if not outcome.conditions:
            return Verdict(
                value=None, score=0.0, band="abstain",
                evidence="gate_dropped_all" if outcome.extracted else "none_extracted",
                predicate=self.name, stage="aux", detail=detail,
            )
        return Verdict(
            value=NOTE_LABEL,
            score=len(outcome.conditions) / max(outcome.extracted, 1),
            band="fire", evidence="verbatim_conditions",
            predicate=self.name, stage="aux", detail=detail,
        )


#: プロセス共通の判定点。``chat._build_messages_with_search`` がターンに 1 回引く。
#: 段は 1 つ (補助タスクの結果を帯にするだけ) なので方針は既定の ``complement``。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_OutcomeStage(),
        candidates=[NOTE_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def answer_conditions_verdict(text: str, outcome: ExtractionOutcome) -> Verdict:
    """判定点として評価して記録する (ターンに 1 回)。"""
    return predicate.evaluate(text or "", {"outcome": outcome})


def note_spans(verdict: Verdict, outcome: ExtractionOutcome) -> Sequence[str]:
    """注記に載せる span (発火したときだけ)。"""
    return outcome.spans if verdict.fired else ()
