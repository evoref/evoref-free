"""問いが **今の会話の窓** に結び付いているか — 照応・継続・省略の判定 (純粋関数 + 判定点)。

2 つの読み手が同じ規則を読む (不変則 #14 (a)):

- 学習側 (``learning.corrected_pairs`` / ``learning.fewshot_pool`` /
  ``learning.context_bound_gate``): 単独では意味を成さない問いを手本・評価の
  候補から外す (:func:`refers_to_previous_turn`)。
- 記憶の注入 (``memory.pipeline.injector``): 省略・指し直しの追い質問
  (「両方を1日で回るプランを。」「前者に行くのにおすすめの時間帯は？」) で、
  **別の会話** の記憶をコサインだけで載せない (判定点 ``query_context_bound``)。
  判定点が見るのは閉じた指し直しの類と短い継続指示だけで、学習側の照応語は
  使わない (:func:`context_bound_reason`)。

以前は :func:`refers_to_previous_turn` が ``learning.corrected_pairs`` にだけ在り、
記憶側は照応を知らなかった。実インシデント (2026-10-05 trace 8232204694b3):
東京タワー / スカイツリーの会話の「両方を1日で回るプランを。」に、別の会話の
「1日目は嵐山、2日目は東山にしたいです。」が cosine 0.64 で注入され、モデルは
京都のプランを答えた。

継続指示の「依頼の文末」は ``core.text_quality`` の 1 本を読む
(:func:`~backend.free.core.text_quality.ends_with_elided_predicate_request`)。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.core.text_quality import ends_with_elided_predicate_request

#: 直前ターンへの照応・継続 (「修正版に」「2 案を採用」「その」「続けて」)。
#:
#: ``その他`` / ``それぞれ`` は **照応ではない** — 前者は「その他大勢」の
#: 一語、後者は複数対象への配分を表す副詞で、どちらも直前ターンを指さない。
#: 素の ``その`` / ``それ`` で拾うと、文脈非依存の問いまで文脈依存に倒れる。
#: 実測 (2026-09-06): 正しく組めた訂正ペア 4 件のうち 2 件がこの 2 語だけで
#: 棄却され、eval_core / few-shot への追加が 0 件のままだった (F-01 の宛先を
#: 直しても受け皿に届かない)。
#:
#: やり直しの動詞 (「計算し直して」「書き直して」) と、述べた値の訂正
#: (「18 kg ではなく 22 kg でした」) も直前ターンを指す — やり直す対象、
#: 訂正で入れ替わらなかった残りの前提 (積載量 4.5 t / 6 便) は前のターンに
#: しか無い。実インシデント (2026-09-09 ライブ監査 (d) D-08): 「すみません、
#: 荷物は 18 kg ではなく 22 kg でした。1 台あたりの個数と 1 日の総個数を
#: 計算し直してください。」が few-shot に採用され、手本の応答が問いに無い
#: 4.5 トン・6 便を前提に答える形 (= 無い前提を補う型) を教えていた。
#: 「見直す」(検討する) は対象を含む新規の問いに現れるので含めない。
_ANAPHORA_RE = re.compile(
    r"それ(?!ぞれ)|その(?!他)|これ|この|あれ|あの|さっき|先(?:ほど|程)|直前"
    r"|(?<![名以事手])前の"
    r"|(?<!の)上の|上記|同じ|同様|続き|続けて|もう一度|再度|最初の|ここまで|今の"
    r"|修正版|最終版|改訂版|案を採用|を採用"
    r"|[しりきぎみびいえけせてねめれ]直(?:し|す|せ|さ)"
    r"|(?:ではなく|じゃなく)[^。！？!?\n]{0,24}?でした",
)
#: **既に述べられたことを思い出させる形** (「何でしたか」「いつでしたっけ」
#: 「覚えていますか」)。照応語が無くても直前の文脈が無いと答えられないので
#: 手本にならない。2026-09-10 ライブ監査 (h) H-09: 「娘の学年と習い事は何
#: でしたか」「要約は何文字でしたか」が決定論ゲートを素通りし、LLM の品質
#: floor に落ちるまでプールに滞留した。
_RECALL_FORM_RE = re.compile(
    r"覚え|記憶|言いました|言ったか|でしたか|でしたっけ|だったか|だっけ",
)
#: 短い継続指示 (「表にしてください。」「箇条書きで。」「続けて。」): 対象を
#: 言わない依頼形の文末で、この長さ以下。「べき等性とは何ですか。」のような
#: 短い問いは対象を含むので拾わない。
_MAX_CONTINUATION_CHARS = 14
#: 継続指示の文末。述語を省いて目的語で終わる依頼 (「〜を。」「〜を3つ。」) は
#: ここに書かず ``text_quality`` の 1 本 (:func:`ends_with_elided_predicate_request`)
#: を読む — 記憶側の「依頼の文末」と同じ定義にするため (不変則 #14 (a))。
_CONTINUATION_TAIL_RE = re.compile(
    r"(?:にして|で|に|も)(?:ください|下さい|お願いします|くれ|ね)?[。.!！]?\s*$"
    r"|を(?:ください|下さい|お願いします|くれ|ね)[。.!！]?\s*$"
    r"|(?:続けて|続きを|もう一度|再度)[。.!！]?\s*$",
)
#: **複数の対象を指し直す** 閉じた類 (両方 / 前者 / 後者 / それぞれ / どちら)。
#: 指す対象は今の会話の窓にしか無い。few-shot 側の照応 (``_ANAPHORA_RE``) が
#: ``それぞれ`` を外しているのは「手本として自立するか」の判定だからで、
#: 「窓に結び付いているか」では配分の副詞も対象を窓に求める。
#: ``どちら`` は場所の丁寧語 (「どちらにお住まいですか」「どちらから」) を除く。
_DEICTIC_SET_RE = re.compile(
    r"両方|前者|後者|それぞれ|どちら(?!に|へ|から|さま|様|まで)",
)


def _is_short_continuation(q: str) -> bool:
    return len(q) <= _MAX_CONTINUATION_CHARS and bool(
        _CONTINUATION_TAIL_RE.search(q) or ends_with_elided_predicate_request(q)
    )


def refers_to_previous_turn(query: str) -> bool:
    """問いが直前ターンへの照応・継続か (純粋関数)。

    照応語 (「その」「修正版に」「2 案を採用」) と、短すぎる継続指示
    (「表にしてください。」「両方を1日で回るプランを。」) を拾う。few-shot の
    入口 (``find_content_rejection``) がこれで単独では意味を成さない問いを
    手本から外す。
    """
    q = (query or "").strip()
    if _is_short_continuation(q):
        return True
    return bool(_ANAPHORA_RE.search(q) or _RECALL_FORM_RE.search(q))


# ── 判定点 ``query_context_bound`` ─────────────────────────────────────

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "query_context_bound"

#: 今の会話の窓に結び付いている。
CONTEXT_BOUND_LABEL = "context_bound"


def context_bound_reason(query: str) -> tuple[str | None, str]:
    """``(label, evidence)`` を返す (純粋関数)。

    手掛かりは **閉じた 2 つだけ** — 複数対象の指し直し (:data:`_DEICTIC_SET_RE`) と
    短い継続指示。学習側の照応 (:data:`_ANAPHORA_RE`) は使わない: ``この`` /
    ``これ`` / ``今の`` / ``同じ`` は初回の自立した問い (「この辺でおすすめの
    レストランは？」「今の時期におすすめの旅行先は？」) にも、前の会話を指す
    時の副詞 (「この前話した旅行の計画は？」) にも現れ、注入側で発火すると
    本人の記憶まで落とす (2026-10-05 独立レビュー H2)。

    - 指し直し / 短い継続指示があり、想起形が無い →
      ``(CONTEXT_BOUND_LABEL, <根拠>)``
    - 想起形 (「何でしたか」「覚えていますか」) を伴う → ``(None, "recall_form")``
      — 想起は前の会話を指しうるので、窓に結び付くとは言えない (棄権)。
    - どれも無い → ``(NEGATIVE_LABEL, "no_match")``
    """
    q = (query or "").strip()
    if not q:
        return NEGATIVE_LABEL, "empty"
    if _DEICTIC_SET_RE.search(q):
        evidence = "deictic_set"
    elif _is_short_continuation(q):
        evidence = "continuation"
    else:
        evidence = ""
    if _RECALL_FORM_RE.search(q):
        return (None, "recall_form") if evidence else (NEGATIVE_LABEL, "recall_form")
    if evidence:
        return CONTEXT_BOUND_LABEL, evidence
    return NEGATIVE_LABEL, "no_match"


class _ContextBoundRule:
    """字句段 (``Predicate`` プロトコル)。根拠を形ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002 - Predicate プロトコルの引数
        label, evidence = context_bound_reason(text or "")
        if label is None:
            return Verdict(
                value=None, score=0.5, band="abstain",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        if label == CONTEXT_BOUND_LABEL:
            return Verdict(
                value=label, score=1.0, band="fire",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NEGATIVE_LABEL, score=0.0, band="skip",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


_RULE = _ContextBoundRule()

#: プロセス共通の判定点。チャット応答パスの記憶注入がターンに 1 回引く。
#: 誤発火のほうが重い (別の会話の正しい記憶を落とす) ので ``confirm``。
#: 事例段は未装着 — 付けるときは ``exemplar=`` を渡すだけで足りる。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="confirm",
        candidates=[CONTEXT_BOUND_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def query_context_bound_verdict(query: str) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(query or "")


__all__ = [
    "CONTEXT_BOUND_LABEL",
    "PREDICATE_NAME",
    "bind_debug_logger",
    "context_bound_reason",
    "predicate",
    "query_context_bound_verdict",
    "refers_to_previous_turn",
]
