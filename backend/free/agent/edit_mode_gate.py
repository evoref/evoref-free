"""既存ファイルの編集の種類 (追記 / 書き直し / 部分修正) の判定点 ``edit_mode``。

追記だけの依頼は既存内容をモデルに通さず ``既存内容 + 区切り + 追加分`` を決定論で
連結する (docs/f_11 §5)。それ以外は既存内容の全体を素材にモデルが本文を作り直す。
以前は ``REVISE_REQUEST_RE`` に当たる語が 1 つでもあれば書き直しにしており、
「これを報告書に追記して同じファイルを更新してください。」の **更新** (ファイルへ
書き戻せという動詞) で全文の書き直しへ回った (2026-10-02 ライブ監査 D07#3:
9B が既存 858 文字を写して ``edit_without_change`` で 2 回棄却、136 秒)。

**字句段** (``intent_vocab.edit_mode_rule``) は語の有無ではなく「依頼が既存の本文の
どこかを変えよと言っているか」を構造で読む — 置換の値の形 / 本文を変える操作の依頼の
形 / 書き戻しの動詞の目的語が本文の一部か。

誤りのコストは非対称:

- 追記を書き直しと誤る → 全文の再生成で遅くなる (D07#3 の 136 秒)。既存内容は
  モデルを通るが、写しの棄却 (``edit_without_change``) と書込み後の照合があり、
  依頼そのものは満たされる
- 書き直しを追記と誤る → 既存内容は壊れず末尾に足されるが、**依頼の変更が黙って
  落ちたまま「完了しました」と報告される** (利用者は気づけない)

重いのは後者 (append の誤発火) なので **append は確かめられたときだけ** 出す
(fail-closed):

- 字句段が ``append`` を出すのは、追記の節と書き戻しだけの節で依頼が尽きている
  (``append_residual`` が空) ときだけ。構造で読めない残りがあれば **棄権**
- 棄権したときだけ事例段が残りの節を比べ、``append`` / ``rewrite`` を決める。
  事例が棄権・未 warmup・埋め込み失敗・ゲート無しなら棄権のまま = 書き直し
- 字句段が ``rewrite`` / ``patch`` と読んだ依頼は事例へ聞かない (append へ上げない)

機構は ``CascadePredicate`` の ``complement`` (字句の棄権を事例が埋める) だが、
字句段が append を「確かめられたときだけ」しか出さないので、判定点としては
「append の発火に確認を要する」``confirm`` の意図を fail-closed で実装している
(``confirm`` は事例が棄権・未 warmup のとき字句の append をそのまま通すので、
埋め込みの無い構成で後者の誤りを防げない。2026-10-02 レビュー中 1)。

**評価はターンに 1 回** (meta 経路は ``MetaCognitiveAgent._process_impl``、長文経路は
``long_form_write_file``)。編集の動詞が 1 つも無い依頼は評価も記録もしない。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.core.intent_vocab import (
    EDIT_MODE_APPEND,
    EDIT_MODE_PATCH,
    EDIT_MODE_REWRITE,
    append_residual,
    edit_mode_rule,
)
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    Verdict,
    load_exemplars,
)
from backend.log_config import get_logger

logger = get_logger("agent.edit_mode_gate")

#: 同梱事例 (tracked)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "edit_mode_exemplars.jsonl"
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "edit_mode"

#: 近傍投票数と発火に必要な得票率。**実測で決める** (bench_predicate_gate.py)。
#:
#: **LOO ではなく検証セット (事例に無い 44 件) で決める** — 初版は事例 43 件の LOO が
#: 1.000 だったが、事例の大半が同じ型で、事例に無い言い回し 16 件で 8 件誤った
#: (2026-10-02 レビュー中 2)。事例 94 件 (append 48 / rewrite 46、日英) に拡充し、
#: 検証セット 48 件 (24 / 24) で bench_edit_mode_heldout.py が判定点全体を測った
#: (bge-m3-q8_0)::
#:
#:     k=3 ratio=0.6  rewrite->append 5/24  append->rewrite 1/24  (LOO 0.856)
#:     k=3 ratio=0.8  rewrite->append 0/24  append->rewrite 3/24  (LOO 0.981)
#:     k=5 ratio=0.6  rewrite->append 5/24  append->rewrite 1/24  (LOO 0.878)
#:     k=5 ratio=0.8  rewrite->append 0/24  append->rewrite 2/24  (LOO 0.939)
#:     k=7 ratio=0.6  rewrite->append 0/24  append->rewrite 2/24  (LOO 0.961)  ← 採用
#:     k=7 ratio=0.8  rewrite->append 0/24  append->rewrite 3/24  (LOO 1.000)
#:     k=9 ratio=0.6  rewrite->append 0/24  append->rewrite 1/24  (LOO 0.949)
#:
#: rewrite->append (変更が黙って落ちる) を 0 にする設定のうち、append->rewrite が最少で
#: warmup の自己診断 (LOO ≥ 0.95) も満たす k=7 / 0.6 を採る (docs/c_17 §3.16)。
#: 棄権は書き直し側へ倒れる (fail-closed)。
DEFAULT_K = 7
DEFAULT_FIRE_RATIO = 0.6


class _EditModeRule:
    """字句段 (``Predicate`` プロトコル)。根拠を構造ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Any = None) -> Verdict:  # noqa: ARG002
        label, evidence = edit_mode_rule(text)
        if label == EDIT_MODE_APPEND and append_residual(text):
            # 構造で読めない依頼の残りがある。事例で確かめるまで追記と決めない。
            return Verdict(
                value=None, score=0.0, band="abstain",
                evidence="unexplained_residual", predicate=self.name, stage="lexical",
            )
        if not label:
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=label, score=1.0, band="fire",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _EditModeRule()


class EditModeGate:
    """字句の ``append`` に事例の確認を掛け、編集の種類を返す。"""

    def __init__(
        self,
        embedder: Any,
        *,
        exemplars: Sequence[Exemplar] | None = None,
        k: int = DEFAULT_K,
        fire_ratio: float = DEFAULT_FIRE_RATIO,
        debug_logger: Any = None,
    ) -> None:
        records = (
            list(exemplars)
            if exemplars is not None
            else load_exemplars(DEFAULT_EXEMPLARS_FILE)
        )
        # 事例も問いと同じく「追記の節を除いた残り」で比べる (append_residual)。
        records = [
            Exemplar(residual, e.label, e.evidence_id)
            for e in records
            if (residual := append_residual(e.text))
        ]
        self._embedder = embedder
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
            lexical=_RULE,
            exemplar=self._exemplar,
            # 字句段は append を確かめられたときだけ出し、残りがあれば棄権する。
            # その棄権だけを事例が埋める (モジュール docstring)。
            policy="complement",
            debug_logger=debug_logger,
            candidates=[EDIT_MODE_APPEND, EDIT_MODE_REWRITE, EDIT_MODE_PATCH, NEGATIVE_LABEL],
            scope="request",
        )

    def is_ready(self) -> bool:
        return self._exemplar.is_ready()

    def bind_debug_logger(self, debug_logger: Any) -> None:
        self._cascade.bind_debug_logger(debug_logger)

    def reset(self, embedder: Any = None) -> None:
        self._exemplar.reset(embedder)
        if embedder is not None:
            self._embedder = embedder

    async def warmup(self) -> bool:
        return await self._exemplar.warmup()

    def calibration(self) -> dict[str, float | bool | int]:
        return self._exemplar.calibration

    def self_check(self) -> dict[str, object]:
        """warmup 時の LOO 自己診断 (正解率 / 被覆 / 閾値を満たしたか)。"""
        return self._exemplar.self_check

    def leave_one_out(self, *, with_errors: bool = False) -> dict[str, object]:
        return self._exemplar.leave_one_out(with_errors=with_errors)

    async def decide(self, query: str) -> Verdict:
        """判定して記録する。

        事例段へ進むのは字句段が棄権した依頼 (追記の語があり、構造で読めない残り
        ``append_residual`` がある) だけ。事例段は文全体ではなく残りの節の埋め込みで
        投票する (``query_vec`` で渡す)。未 warmup・埋め込み失敗は字句段の棄権のまま
        (= 書き直し側、fail-closed)。
        """
        if _RULE.evaluate(query).band != "abstain" or not self.is_ready():
            return self._cascade.evaluate(query)
        try:
            vec = await self._embedder.embed_query(append_residual(query), mode="chat")
        except Exception as e:
            logger.info("Edit mode residual embed failed: %s", e)
            return self._cascade.evaluate(query)
        return await self._cascade.aevaluate(query, query_vec=np.asarray(vec, dtype=np.float32))


async def resolve_edit_mode(gate: Any, query: str) -> Verdict | None:
    """ターンの編集の種類を 1 回だけ評価する。編集の動詞が無ければ ``None`` (記録しない)。

    ゲートが無い (埋め込み無しの構成・テスト) ときは字句段だけで決める (記録はしない。
    構造で読めない残りがあれば棄権 = 書き直し側)。ゲートの失敗も字句段へ縮退する。
    """
    if not edit_mode_rule(query)[0]:
        return None
    if isinstance(gate, EditModeGate):
        try:
            return await gate.decide(query)
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.warning("Edit mode gate failed, falling back to the rule: %s", e)
    return _RULE.evaluate(query)


def is_append_verdict(verdict: Verdict | None) -> bool:
    """判定が追記だけか (棄権・反対・書き直し・部分修正は False)。"""
    return verdict is not None and verdict.band == "fire" and verdict.value == EDIT_MODE_APPEND
