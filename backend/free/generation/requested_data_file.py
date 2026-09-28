"""判定点 ``requested_data_file`` — 依頼文が名指したデータファイルが「用意の対象」か。

設計書 f_10 §11.1-1、c_17 §3.10 準拠。

staged v2 は、依頼文が「用意 / 作成 / 付けて / 同梱」の対象として名指したデータ拡張子のファイルが
骨組みにも成果物にも配信先にも無いとき、本文で「作成していない」と伝える (自動生成はしない)。
依頼文に名前があるだけでは足りない — 同じ ``access.log`` でも「サンプルログ access.log も用意して」
(用意の対象)・「access.log を読み込み」(入力)・「結果は report.log に出力する」(実行時の出力) がある。
境界は語彙ではなく **その名前を支配する述語** で決まる意味の分類なので、c_17 の契約に載せる。

- ファイル名の抽出は字句の鍵 (:data:`~backend.free.core.intent_vocab.FILE_NAME_IN_TEXT_RE`、判定点の外)。
  直後に ``.<英数字>`` が続く名前 (``access.log.1`` の中の ``access.log``) は別の名前なので採らない。
  データ拡張子の表は ``validators.DATA_OR_DOCUMENT_SUFFIXES`` の 1 本。
- 名前の後の同じ文で、用意の述語 (依頼の形) が出力・入力の述語より先に来たら ``fire``。
  出力 (``に`` / ``へ`` が名前に続く、``保存`` / ``出力`` …)・入力 (``から``、``読み込`` / ``解析`` …) が先なら
  ``skip``。用意の述語が依頼の形でない (「access.log を作成するスクリプト」= 実行時に作る) か、
  どの述語も読めない (「access.log 形式の」) なら ``abstain``。**棄権は通知しない**。
- 英語は名前の前を見る (``include`` / ``provide`` … が同じ文にあれば ``fire``、直前が ``to`` / ``into`` /
  ``from`` / ``reads`` … なら ``skip``)。

事例段は持たない (``text_unit_fabrication`` / ``create_target_named`` と同じ形)。述語は文の中の位置関係で
決まり、ライブ監査で誤発火 / 取りこぼしが出たらその実データから ``confirm`` の相手を作る。
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from collections.abc import Callable
from typing import Any

from backend.free.core.intent_vocab import FILE_NAME_IN_TEXT_RE
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.core.script_ranges import HIRAGANA
from backend.free.generation.validators import DATA_OR_DOCUMENT_SUFFIXES

__all__ = [
    "PREDICATE_NAME",
    "REQUESTED_LABEL",
    "bind_debug_logger",
    "named_data_files",
    "predicate",
    "requested_names",
    "requested_verdict",
]

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "requested_data_file"

#: 発火ラベル (名前が用意の対象)。
REQUESTED_LABEL = "requested"

#: 名前の直後に続く別の拡張子 (``access.log.1`` / ``data.csv.gz``)。
_LONGER_NAME_RE = re.compile(r"\.[A-Za-z0-9]")

#: 文の区切り (英語のピリオドは後ろが空白 / 末尾のときだけ。``access.log`` の点では切らない)。
_SENTENCE_END_RE = re.compile(r"[。！？!?\n]|\.(?=\s|$)")

#: 名前の直後の飾り (閉じ引用符・括弧・空白、名前に添えた括弧書き「（100 行程度）」)。
_AFTER_NAME_NOISE_RE = re.compile(r"(?:[\s」』\"'`’”)）\]】]|[（(][^（()）]{0,40}[)）])*")

#: 名前の直後の格助詞で決まるもの (宛先 = 実行時の出力 / 起点 = 入力)。
_DESTINATION_PARTICLE_RE = re.compile(r"^(?:に|へ)(?!ついて|関して|よる|より)")
_SOURCE_PARTICLE_RE = re.compile(r"^から")

#: 用意の述語の語幹 (依頼の形かは :data:`_PROVIDE_REQUEST_RE` で見る)。
_PROVIDE_STEM_RE = re.compile(r"用意|作成|生成|同梱|添付|作っ|作る|作り|付け|つけ|添え|含め|入れ|置い|置き|置く")
#: テ形の後が依頼で終わる (「用意してください」「付けておいて」「作って。」)。
#: 「作って保存する」(テ形 + 別の動作 = 実行時の処理の説明) は依頼ではない。
_REQUEST_END = r"(?=\s*(?:ください|下さい|ほしい|欲しい|もら|くれ|いただ|頂|お(?:いて|く|き)|[。、,!！]|$))"
#: 用意の述語の依頼の形 (語幹の位置から当てる)。「作成するスクリプト」「作った」は依頼ではない。
#: 体言止め (「access.log も用意。」「test.dat も用意」) も依頼 — 名前の後ろの文は文末で切ってある。
_PROVIDE_REQUEST_RE = re.compile(
    r"(?:用意|作成|生成|同梱|添付)(?:して" + _REQUEST_END + r"|し(?=、)|する(?:こと|ように)|を(?:お願い|して)"
    r"|(?=\s*$))"
    r"|(?:作っ|付け|つけ|添え|含め|入れ|置い)て" + _REQUEST_END
)
#: 名前と用意の述語の間に置ける助詞 (これと名詞句だけなら、用意の述語は名前を支配する)。
_GAP_PARTICLE_RE = re.compile(r"という|といった|として|など|および|及び|または|[もをとやのは、,・]")
#: 名前と用意の述語の間に別の成果物 (「〜する スクリプト を作成」) が挟まる印。
_GAP_DELIVERABLE_RE = re.compile(
    r"(?i)スクリプト|ツール|プログラム|アプリ|コマンド|モジュール|パッケージ|関数|クラス"
    r"|\b(?:cli|script|tool|program|app|command|module|package)\b"
)
#: 連体形の動詞・形容詞の送り仮名 (助詞を除いた後に残るひらがな)。
_HIRAGANA_RE = re.compile(f"[{HIRAGANA}]")
#: 入力の述語 (名前を読む・使う側)。
_CONSUME_STEM_RE = re.compile(r"読み込|読込|読ん|読み|読む|解析|集計|処理|開い|開く|開き|入力|受け取|参照|使っ|使う|使い|使用")
#: 実行時の出力の述語 (名前へ書く側)。
_OUTPUT_STEM_RE = re.compile(r"出力|書き出|書出|書き込|書込|保存|記録|追記")

#: 英語: 名前の前の同じ文の用意の動詞 / 名前の直前の前置詞・動詞。
_EN_PROVIDE_RE = re.compile(r"(?i)\b(?:include|provide|prepare|create|add|bundle|ship|generate|supply)\b")
_EN_NOT_PROVIDED_RE = re.compile(
    r"(?i)\b(?:to|into|in|from|reads?|reading|parses?|parsing|loads?|loading|opens?|processes|"
    r"writes?|writing|saves?|saving|logs?|outputs?)\s+(?:(?:the|a|an|its|each|sample|given)\s+)*$"
)
#: 英語: 用意の動詞と名前の間に挟まると、動詞が名前を支配しない語 (関係節・目的の to・前置詞・成果物)。
_EN_GAP_BLOCK_RE = re.compile(
    r"(?i)\b(?:that|which|who|to|for|from|into|in|by|using|with|of|"
    r"cli|script|tool|program|app|command|module|package|function)\b"
)
#: 英語: 用意の動詞と名前の間に置ける語数の上限 (「include a small sample」)。
_EN_GAP_MAX_WORDS = 4


def named_data_files(text: str) -> list[tuple[str, int, int]]:
    """依頼文が名指すデータ拡張子のファイル名 ``(名前, 開始, 終了)`` (出現順、同じ名前も出現ごと)。

    字句の鍵 (判定点の外)。``access.log.1`` の中の ``access.log`` と、表に無い拡張子 (``ASP.NET``) は採らない。
    同じ名前が 2 回出る依頼 (「access.log を解析するスクリプトを作って。サンプルの access.log も用意して」) は
    出現ごとに判定する (:func:`requested_names`)。
    """
    out: list[tuple[str, int, int]] = []
    for m in FILE_NAME_IN_TEXT_RE.finditer(text or ""):
        if _LONGER_NAME_RE.match(text, m.end()):
            continue
        name = m.group().lstrip(".")
        if PurePosixPath(name).suffix.lower() not in DATA_OR_DOCUMENT_SUFFIXES:
            continue
        out.append((name, m.start(), m.end()))
    return out


def _has_code_file(text: str) -> bool:
    return any(
        PurePosixPath(m.group()).suffix.lower() not in DATA_OR_DOCUMENT_SUFFIXES
        for m in FILE_NAME_IN_TEXT_RE.finditer(text)
    )


def _governs(gap: str) -> bool:
    """名前と用意の述語の間 ``gap`` が助詞と名詞句だけか (= 用意の述語が名前を支配する)。

    「sales.csv を要約する スクリプトを 作成して」のように、連体形の動詞 (助詞を除いて残るひらがな)・
    コードのファイル名・別の成果物 (スクリプト / ツール …) が挟まれば、述語は別の成果物にかかる。
    """
    if _GAP_DELIVERABLE_RE.search(gap) or _has_code_file(gap):
        return False
    return not _HIRAGANA_RE.search(_GAP_PARTICLE_RE.sub("", gap))


def _english(head: str) -> tuple[str, str]:
    """名前の前 ``head`` (同じ文) を見る英語の判定。"""
    if _EN_NOT_PROVIDED_RE.search(head):
        return "skip", "en_not_provided"
    verbs = list(_EN_PROVIDE_RE.finditer(head))
    if not verbs:
        return "abstain", "no_governing_verb"
    between = head[verbs[-1].end():]
    if (
        _EN_GAP_BLOCK_RE.search(between) or _has_code_file(between)
        or len(between.split()) > _EN_GAP_MAX_WORDS
    ):
        return "abstain", "en_provide_governs_another"
    return "fire", "en_provide"


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    begin = 0
    for m in _SENTENCE_END_RE.finditer(text, 0, start):
        begin = m.end()
    m = _SENTENCE_END_RE.search(text, end)
    return begin, (m.start() if m else len(text))


def _first(patterns: dict[str, re.Pattern[str]], text: str) -> tuple[str, int] | None:
    hits = [(m.start(), kind) for kind, pat in patterns.items() if (m := pat.search(text))]
    if not hits:
        return None
    pos, kind = min(hits)
    return kind, pos


def _judge(text: str, start: int, end: int) -> tuple[str, str]:
    """``(band, evidence)`` を返す (純粋関数)。"""
    begin, stop = _sentence_bounds(text, start, end)
    head = text[begin:start]
    tail = text[end:stop]
    tail = tail[_AFTER_NAME_NOISE_RE.match(tail).end():]
    if _DESTINATION_PARTICLE_RE.match(tail):
        return "skip", "output_destination"
    if _SOURCE_PARTICLE_RE.match(tail):
        return "skip", "input_source"
    first = _first(
        {"provide": _PROVIDE_STEM_RE, "consume": _CONSUME_STEM_RE, "output": _OUTPUT_STEM_RE}, tail,
    )
    if first is not None:
        kind, pos = first
        if kind == "consume":
            return "skip", "input_verb"
        if kind == "output":
            return "skip", "output_verb"
        if not _governs(tail[:pos]):
            # 用意の述語は間に挟まった別の成果物にかかる (「sales.csv を要約するスクリプトを作成」= 入力)
            return "abstain", "provide_governs_another"
        if _PROVIDE_REQUEST_RE.match(tail, pos):
            return "fire", "provide_request"
        return "abstain", "provide_not_requested"
    return _english(head)


class _RequestedDataFileRule:
    """字句段: 依頼文 ``text`` と名前の位置 (``ctx["start"]`` / ``ctx["end"]``) → :class:`Verdict`。

    ``LexicalPredicate`` は根拠を 1 語に固定するので、どの述語で決めたかを残すために
    :class:`~backend.free.core.predicate.Predicate` を直接実装する。
    """

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: dict[str, Any] | None = None) -> Verdict:
        ctx = ctx or {}
        start, end = int(ctx.get("start", -1)), int(ctx.get("end", -1))
        if not text or not 0 <= start < end <= len(text):
            return Verdict(
                value=None, score=0.0, band="abstain",
                evidence="no_position", predicate=self.name, stage="lexical",
            )
        band, evidence = _judge(text, start, end)
        if band == "fire":
            return Verdict(
                value=REQUESTED_LABEL, score=1.0, band="fire",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        if band == "skip":
            return Verdict(
                value=NEGATIVE_LABEL, score=0.0, band="skip",
                evidence=evidence, predicate=self.name, stage="lexical",
            )
        return Verdict(
            value=None, score=0.0, band="abstain",
            evidence=evidence, predicate=self.name, stage="lexical",
        )


#: プロセス共通の判定点。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RequestedDataFileRule(),
        policy="complement",
        candidates=[REQUESTED_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def requested_verdict(text: str, start: int, end: int) -> Verdict:
    """``text[start:end]`` の名前が用意の対象か (判定は必ず ``log_decision`` へ出る)。"""
    return predicate.evaluate(text or "", {"start": start, "end": end})


def requested_names(text: str, *, skip: Callable[[str], bool] | None = None) -> list[str]:
    """依頼文が用意の対象として名指したデータファイル名 (出現順、重複なし)。

    同じ名前の出現はそれぞれ判定して記録し、どれかが ``fire`` ならその名前を採る。``skip(name)`` が真の
    名前 (骨組みにある・配信先に実在する) は判定しない (記録を増やさない)。
    """
    fired: dict[str, str] = {}
    skipped: set[str] = set()
    for name, start, end in named_data_files(text):
        key = name.lower()
        if key in skipped:
            continue
        if skip is not None and skip(name):
            skipped.add(key)
            continue
        if requested_verdict(text, start, end).fired:
            fired.setdefault(key, name)
    return list(fired.values())
