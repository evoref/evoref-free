"""計画のタスクが **書込みのタスク** かの判定点 ``task_write_intent`` (c_17 §3.18)。

meta 経路の書込みに関わる判定 (書込みを起こす側と、書込みを期待して帳簿を付ける
側) はすべてこの 1 実装を読む (不変則 #14 (a)、docs/f_03 §4.3)。以前は書込み動詞の
部分一致 (``task_expects_write``) と、それに層の振り分けの宛先の証拠を足した
``task_names_write`` の 2 系統が読み手ごとに分かれ、``address`` の ``add`` を
書込みと数え、「Read E:\\…\\x.docx」を宛先の名指しと数えていた。

| band | 条件 | evidence |
|---|---|---|
| ``fire`` | 書込み動詞 (``WRITE_VERB_WORD_RE``、英語は語境界で区切る) | ``write_verb`` |
| ``abstain`` | 書込み動詞はあるが、計画モデルの種別が ``process`` / ``retrieve`` で、タスク文に宛先のパス (宛先の標識 / ドライブ付きのパス) が無い | ``write_verb_vs_plan_kind`` |
| ``abstain`` | 動詞は無いが、宛先の標識が導くファイルパスがある (``to`` / ``into`` / ``onto`` の後ろ、``に保存`` 等の手前。``tool_judge_args.extract_marked_write_target``) | ``marked_destination`` |
| ``skip`` | どちらも無い | ``none`` |

計画モデルの種別は字句の ``fire`` を **確認** する票 (confirm 方針、不変則 #14): 反対
(``process`` / ``retrieve``) なら棄権へ倒す。2026-10-05 ライブ監査 T5 で、計画モデルが
``process`` と付けた「Identify the bugs and determine the fixes」の ``fixes`` を書込みと
数え、テキストからの書込みの救出が fixed.py を書いた (次のタスクも同じ内容を書いて
「内容が変わらなかった」と失敗した)。宛先のパスがタスク文にあれば種別より強い証拠
なので ``fire`` のまま。種別が無い・``write`` / ``other`` / ``retrieve_then_process`` は
従来どおり。

読み手は ``fire`` だけを書込みとする (書込みを起こす側も、状態を ``failed`` にしたり
答えを隠したりする帳簿の側も)。棄権は **観察のみ** — 宛先の標識はタスク文のどこに
あっても立つので (「Go to E:\\…\\README.md and summarize it」「Compare a.txt to
b.txt」「Refer to E:\\…\\spec.md」)、これを書込みに数えると読むタスクが失敗にされ、
正しい答えが隠れる。読みの動詞の一覧で絞ることはしない (不変則 #14)。棄権の帯は
planner がタスクに付けるラベルで埋める予定 (PR3、棄権のときだけ読む)。例外は
「書込みを起こさない・近道を止めるだけ」の 3 つの読み手で、``fire`` と ``abstain`` を
書込みのタスクと読む (``_answers_from_retrieved`` / ``_continues_after_fetch`` は
ツールループへ回すだけ、``_write_left_to_later_task`` は書込みを止めるだけ。
「Save the summary to X」を取得済みデータの答えにして書かずに終えないため、
docs/f_03 §4.3)。

利用者の発話の書込み動詞 (``_ensure_build_task``) も同じ字句段
(:func:`has_write_verb`) を通す — 発話と計画のタスクで別の正規表現を使うと、片方だけ
当たって計画が生成タスクに置き換わる。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.free.core.intent_vocab import EXPLICIT_WINDOWS_PATH_RE, WRITE_VERB_WORD_RE
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

PREDICATE_NAME = "task_write_intent"

WRITE_LABEL = "write"

#: 字句の書込み ``fire`` に反対票を投じる計画モデルの種別 (``json_schemas.PLAN_TASK_KINDS``)。
NON_WRITE_PLAN_KINDS = frozenset({"process", "retrieve"})
#: 計画モデルの種別が字句の書込みに反対した棄権の ``evidence``。
PLAN_KIND_ABSTAIN_EVIDENCE = "write_verb_vs_plan_kind"


def _names_destination_path(description: str) -> bool:
    """タスク文に書込みの宛先になりうるパスがあるか (宛先の標識 / ドライブ付きのパス)。"""
    from backend.free.agent.tool_judge_args import extract_marked_write_target

    return bool(
        extract_marked_write_target(description)
        or EXPLICIT_WINDOWS_PATH_RE.search(description or "")
    )


def has_write_verb(text: str) -> bool:
    """書込み動詞を含むか (字句段。パス中の語は数えない。純粋関数)。"""
    # パス中の語 (``E:\tmp\create_01\DESIGN.md`` の ``create``) を動詞と数えない
    # (2026-09-19 ライブ監査)
    return bool(WRITE_VERB_WORD_RE.search(EXPLICIT_WINDOWS_PATH_RE.sub(" ", text or "")))


class _TaskWriteRule:
    """字句段 (動詞) と構造段 (宛先の標識) を 1 つの ``Verdict`` にする (``Predicate``)。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        description = text or ""
        if has_write_verb(description):
            plan_kind = (ctx or {}).get("plan_kind")
            if plan_kind in NON_WRITE_PLAN_KINDS and not _names_destination_path(description):
                return Verdict(
                    value=None, score=0.5, band="abstain",
                    evidence=PLAN_KIND_ABSTAIN_EVIDENCE, predicate=self.name,
                    stage="lexical",
                )
            return Verdict(
                value=WRITE_LABEL, score=1.0, band="fire",
                evidence="write_verb", predicate=self.name, stage="lexical",
            )
        from backend.free.agent.tool_judge_args import extract_marked_write_target

        if extract_marked_write_target(description):
            return Verdict(
                value=None, score=0.5, band="abstain",
                evidence="marked_destination", predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=NEGATIVE_LABEL, score=0.0, band="skip",
            evidence="none", predicate=self.name, stage="lexical",
        )


_RULE = _TaskWriteRule()

predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[WRITE_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def destination_clause_requests_write(description: str, file_path: str) -> bool:
    """宛先のパスを含む節が、書込みの命令形で始まるか (純粋関数)。

    棄権 (``marked_destination``) のタスクで書込みを **起こす** 読み手
    (``_write_after_search``) が、宛先の標識だけで書かないための構造の確認。宛先の
    手前 (最後に現れた位置) を文 (``split_sentences``) と節の境界 (``CLAUSE_BOUNDARY_RE``)
    で切った最後の節の頭を見る (パスの綴りは除く)。「Search for TODO in D. Save the
    matching lines to R」は 2 文目の頭の ``Save`` を見る。パスが本文に見つからなければ
    偽 (書かない側に倒す)。
    """
    from backend.free.agent.tool_judge_args import without_drive_paths
    from backend.free.core.intent_vocab import (
        CLAUSE_BOUNDARY_RE,
        clause_head_is_write_verb,
        split_sentences,
    )

    index = (description or "").rfind(file_path) if file_path else -1
    if index < 0:
        return False
    sentences = split_sentences(without_drive_paths(description[:index], " "))
    if not sentences:
        return False
    clause = CLAUSE_BOUNDARY_RE.split(sentences[-1])[-1]
    return clause_head_is_write_verb(clause)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def task_write_verdict(description: str, plan_kind: str | None = None) -> Verdict:
    """タスク文を判定する (記録しない純粋関数)。``plan_kind`` は計画モデルの種別。"""
    verdict = _RULE.evaluate(description, {"plan_kind": plan_kind})
    return Verdict(
        value=verdict.value, score=verdict.score, band=verdict.band,
        evidence=verdict.evidence, predicate=PREDICATE_NAME, stage=verdict.stage,
    )


def record_task_write_verdict(description: str, plan_kind: str | None = None) -> Verdict:
    """判定点として評価する (``decision.jsonl`` と死活監視へ記録する。タスクに 1 回)。"""
    return predicate.evaluate(description or "", {"plan_kind": plan_kind})


__all__ = [
    "NON_WRITE_PLAN_KINDS",
    "PLAN_KIND_ABSTAIN_EVIDENCE",
    "PREDICATE_NAME",
    "WRITE_LABEL",
    "bind_debug_logger",
    "destination_clause_requests_write",
    "has_write_verb",
    "predicate",
    "record_task_write_verdict",
    "task_write_verdict",
]
