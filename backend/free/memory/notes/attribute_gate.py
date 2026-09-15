"""属性スロットの補完ゲート — trigger 語が 1 つも当たらない発話を事例で拾う。

``fact_attributes.yaml`` の照合は **部分一致 + 記載順** で、trigger 語を 1 つも
含まない言い回しは 1 件も当たらない。当たらなかった発話のファクトは
``mem.<kind>.user`` (汎用スロット) に落ちる。汎用スロットは
:func:`~backend.free.memory.semantic.subject_key.is_generic_subject` が言うとおり
「分類できなかった発話の置き場」で、

- ``asked_attrs`` の免除に掛からない (尋ねられてもコサインの棒を超えないと注入
  されない)
- 訂正の宛先にならない
- 固有スロットが supersede されたら影として一緒に畳まれる

ため、**値としては残るが属性としては失われる**。監査で繰り返し出ている実害
(「勤めています」で employer が立たない / て形「〜をしていて」が occupation の
trigger に無い / 括弧が trigger 語を分断する) はすべてこの形。

語形を足す対処は漏れが必ず残るので、**当たらなかったときだけ** 事例の近傍で
スロットを提案する。設計上の約束は 3 つ:

1. **字句の結果を上書きしない** (``CascadePolicy`` の ``complement``)。
   trigger が当たった発話にはこのゲートは触れない。
2. **棄権したら従来どおり** 汎用スロットへ落ちる。誤って閉じない。
3. **sleep-time でしか動かない**。チャット応答パスからは呼ばれないので
   TTFT に影響せず、不変則 #1 の制約も受けない。間違えても Full の再実行で
   立て直せる。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    Verdict,
    load_exemplars,
)
from backend.log_config import get_logger

logger = get_logger("memory.attribute_gate")

#: 同梱事例 (tracked)。user override は今は持たない — トリガ辞書と違い、
#: 事例は「監査で実害が出た語形」を積むもので、リポジトリ利用者が同じ挙動を
#: 再現できる必要がある (config.md の「新トリガは default 側に必ず置く」と同じ理由)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent.parent / "_defaults" / "attribute_exemplars.jsonl"
)

#: 判定点の名前。``decision.jsonl`` の ``decision_point`` に出る。
PREDICATE_NAME = "fact_attribute_slot"

#: 事例が対象にする fact_type。``personal_fact`` だけに絞ってある —
#: preference / emotion / opinion は汎用スロットへ落ちても実害が小さく、
#: 事例も集まっていない。広げるときは陰性例を先に増やすこと。
TARGET_FACT_TYPE = "personal_fact"

#: 近傍投票数と発火に必要な得票率。**実測で決める** (bench_predicate_gate.py)。
#:
#: **多値ゲートでは ``k`` がクラスあたりの事例数に縛られる。** 発火には
#: ``ceil(k × fire_ratio)`` 票が要るので、事例がそれ未満のスロットは何を
#: 入力しても勝てない (近傍は必ず k 件選ばれるので、票が割れて棄権になるだけ
#: でエラーにもならない)。
#:
#: 実例 (2026-09-15): k=7 / ratio=0.6 (4 票必要) にしたところ 18 スロット中
#: 15 が事例 3 件以下で発火不能になり、実発話 928 件に対して **1 度も発火
#: しなかった**。LOO の正解率 0.944 はほぼ陰性クラス (28 件) だけを測っていた。
#: ``ExemplarPredicate.unreachable_labels`` が warmup 時に警告する。
#:
#: 2026-09-15 / bge-m3-q8_0 / 事例 140 件 (全 19 ラベルが 4 件以上) の LOO::
#:
#:     k=5 ratio=0.7  acc(判定)=1.000  棄権 107/140  被覆 24%   ← 現行
#:
#: 掃引 (事例 125 件時点。傾向は同じ)::
#:
#:     k=3 ratio=0.6  acc(判定)=0.772  棄権  46/125
#:     k=3 ratio=0.8  acc(判定)=0.960  棄権 100/125
#:     k=5 ratio=0.6  acc(判定)=0.833  棄権  65/125
#:     k=5 ratio=0.7  acc(判定)=1.000  棄権 102/125   ← 採用 (4 票)
#:
#: このゲートは **棄権が安い** (従来どおり汎用スロットへ落ちるだけ = 現状維持)
#: 一方、誤ったスロットは有害 (別の subject にファクトが入り、無関係な値と
#: supersede し合う)。したがって正解率を優先し、低い被覆 (18%) は許容する。
#: 4 票必要なので **全ラベルを 4 件以上** に揃えてある (下回ると発火不能)。
DEFAULT_K = 5
DEFAULT_FIRE_RATIO = 0.7


def load_slot_exemplars(
    path: Path | None = None,
    *,
    valid_slots: Iterable[str] | None = None,
) -> list[Exemplar]:
    """事例を読み、``fact_attributes.yaml`` に無いスロット名を落とす。

    事例ファイルの label は YAML のスラグと **一致していなければ意味がない**
    (存在しないスロットを提案すると subject が作られて誰も読まない)。
    綴り違いは黙って落とし、件数を WARNING に出す。
    """
    exemplars = load_exemplars(path or DEFAULT_EXEMPLARS_FILE)
    if valid_slots is None:
        return exemplars
    allowed = set(valid_slots) | {NEGATIVE_LABEL}
    kept = [e for e in exemplars if e.label in allowed]
    dropped = len(exemplars) - len(kept)
    if dropped:
        unknown = sorted({e.label for e in exemplars if e.label not in allowed})
        logger.warning(
            "attribute exemplars: dropped %d records with unknown slots: %s",
            dropped, unknown,
        )
    return kept


def known_slots(triggers_dir: str | Path | None = None) -> set[str]:
    """``fact_attributes.yaml`` の ``personal_fact`` スラグ集合。"""
    from backend.free.memory.notes.note_builder import (
        get_fact_attributes,
        resolve_fact_attributes_path,
    )

    attrs = get_fact_attributes(resolve_fact_attributes_path(triggers_dir))
    per_type = (attrs.get("chat") or {}).get(TARGET_FACT_TYPE) or ()
    return {spec.slug for spec in per_type}


class AttributeSlotGate:
    """属性スロットの事例ゲート。

    ``propose`` は sleep-time の非同期ステップから **バッチで** 呼ぶ。抽出器
    (:class:`~backend.free.memory.extractors.chat.ChatExtractor`) は同期なので、
    結果を ``ExtractionContext.attribute_hints`` に載せて渡す。``value_anchored``
    / ``inherited`` を先に解決してから渡しているのと同じ形。
    """

    def __init__(
        self,
        embedder: Any,
        *,
        exemplars: Sequence[Exemplar] | None = None,
        triggers_dir: str | Path | None = None,
        k: int = DEFAULT_K,
        fire_ratio: float = DEFAULT_FIRE_RATIO,
        debug_logger: Any = None,
    ) -> None:
        slots = known_slots(triggers_dir)
        records = (
            list(exemplars)
            if exemplars is not None
            else load_slot_exemplars(valid_slots=slots)
        )
        self._slots = slots
        self._exemplar = ExemplarPredicate(
            PREDICATE_NAME,
            embedder,
            exemplars=records,
            k=k,
            mode="chat",
            fire_ratio=fire_ratio,
        )
        self._cascade = CascadePredicate(
            PREDICATE_NAME,
            exemplar=self._exemplar,
            policy="complement",
            debug_logger=debug_logger,
            candidates=sorted(slots) + [NEGATIVE_LABEL],
            scope="sleep",
        )

    @property
    def predicate(self) -> CascadePredicate:
        return self._cascade

    def is_ready(self) -> bool:
        return self._exemplar.is_ready()

    def bind_debug_logger(self, debug_logger: Any) -> None:
        self._cascade.bind_debug_logger(debug_logger)

    def reset(self, embedder: Any = None) -> None:
        """埋め込みモデル差し替え時に事例ベクトルと較正を捨てる。"""
        self._exemplar.reset(embedder)

    async def warmup(self) -> bool:
        return await self._exemplar.warmup()

    def calibration(self) -> dict[str, float | bool | int]:
        return self._exemplar.calibration

    def self_check(self) -> dict[str, object]:
        """warmup 時の LOO 自己診断 (正解率 / 被覆 / 閾値を満たしたか)。"""
        return self._exemplar.self_check

    def leave_one_out(self, *, with_errors: bool = False) -> dict[str, object]:
        """事例集合の LOO 精度 (事例を足したときに効果をその場で測る口)。"""
        return self._exemplar.leave_one_out(with_errors=with_errors)

    async def propose(self, texts: Sequence[str]) -> dict[str, str]:
        """``{発話: スロット}`` を返す。決められなかった発話は入れない。

        ``NEGATIVE_LABEL`` (本人の属性を述べていない) と棄権は **どちらも
        結果に入れない** — 呼出側から見れば「提案が無い」で同じ扱いになり、
        従来どおり汎用スロットへ落ちる。
        """
        if not self.is_ready():
            return {}
        out: dict[str, str] = {}
        for text in texts:
            if not text or text in out:
                continue
            verdict: Verdict = await self._cascade.aevaluate(text)
            if verdict.band != "fire":
                continue
            slug = str(verdict.value)
            if slug in self._slots:
                out[text] = slug
        if out:
            logger.info(
                "Attribute slot gate proposed %d/%d slots: %s",
                len(out), len(texts), sorted(set(out.values())),
            )
        return out
