"""直近ファイルへの参照が何を求めているかの判定点 ``recent_file_reference`` (c_17 §3.11)。

「保存したファイルに…」「作成していただいたファイルを…」「そのファイルは
どこ？」のように **直近に扱ったファイル** を指す発話が、そのファイルへ書き込め
と言っているのか (``write``)、場所を尋ねているのか (``locate``)、どちらとも
言えないのか (棄権) を返す。経路は docs/f_03 §1.6。

2026-09-27 ライブ監査で、説明節 (「保存した」) の扱いが実装ごとに逆だった:

- F5 C05#5「保存したファイルに「git stash の使い方」を追記してください。」—
  ルータが説明節を宛先の証拠ごと消し、deliberative で「追記するツールが
  利用できない」と答えた。
- F7 C07#5「保存したファイルの場所を教えてください。」— 説明節の「保存」が
  書込み動詞とみなされ、格下げ後に ``list_directory('.')`` がリポジトリ根を
  列挙した。

「説明節 + に」を素朴に宛先へ足すと、「保存したファイルにある表を CSV に
書き出して」で保存済みファイルを上書きし、「保存したファイルに追記すると
どうなりますか？」で追記を実行する。判別しているのは語ではなく **格と支配する
動詞とモダリティ** なので、字句段で格を読み、決めきれない形は棄権する
(棄権 = 実行せず確認する、が安い側)。

字句段だけのカスケード (不変則 #14 / c_17)。``create_target_gate`` と同じ形で、
事例段が要る局面になれば ``exemplar=`` を足す。そのときは deliberative が
:func:`recent_file_reference_rule` を直接引いている箇所を ``Verdict`` の受け渡しに
変えること。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from backend.free.agent.file_ledger import last_written_path, restore_from_conversation
from backend.free.core.intent_vocab import (
    DESCRIBED_OBJECT_PATTERN,
    EXPLICIT_WINDOWS_PATH_RE,
    FILE_NAME_IN_TEXT_RE,
    FORWARD_REFERENCE_JA,
    QUESTION_END_RE,
    REFERENTIAL_WRITE_TARGET_RE,
    WRITE_VERB_RE,
    ascii_boundary_alternation,
    find_file_reference_clauses,
    is_request_sentence,
    split_sentences,
    strip_file_reference_clauses,
    write_prohibited,
)
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.free.core.script_ranges import KANJI, KATAKANA

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "recent_file_reference"

#: 直近ファイルへ書き込め (meta_cognitive の write 経路)。
WRITE_LABEL = "write"
#: 直近ファイルの場所を尋ねている (deliberative の所在の事実注記)。
LOCATE_LABEL = "locate"

#: 所在を尋ねる語 (「パスワード」の「パス」は除く)。
_LOCATION_ASK_RE = re.compile(
    r"場所|どこ|在り?処|ありか|保存先|置き場|パス(?!ワード)|フォルダ|ディレクトリ"
    r"|" + ascii_boundary_alternation("where", "path", "location", "folder", "directory"),
    re.IGNORECASE,
)
#: ファイルを動かす・変える操作。所在の語と並んでも所在の問いではない
#: (「別のフォルダに移動して」「場所を変えて」「フォルダごと zip にして」)。
_RELOCATE_VERB_RE = re.compile(
    r"移動|移し|移す|変え|変更|コピー|複製|圧縮|削除|消し|名前を"
    r"|" + ascii_boundary_alternation("zip", "move", "copy", "rename", "delete"),
    re.IGNORECASE,
)

#: 宛先の格 (``に`` / ``へ``) の後ろで直近ファイルへ書き込む動詞。「加えて」は
#: 「保存したファイルに加えて、…も作って」(「に加えて」= 〜のほかに) と区別
#: できないので語に含めない (書き加 / 付け足 / 足して は含める)。
_WRITE_INTO_VERB_RE = re.compile(
    r"追記|追加|書き足|付け足|書き加|足して|上書き|書き直|書き換|更新|反映"
    r"|保存|書き込|書込"
    r"|" + ascii_boundary_alternation("append", "add", "update", "overwrite", "save", "write"),
    re.IGNORECASE,
)
#: 書込みの述語の直後に続く **依頼のテ形 / 「お願い」**。この形のときだけ動詞が
#: 宛先句を支配しているとみなす。「追記**する**と」「追記**すべき**内容」
#: 「追記**した**内容」「更新**したい**」「反映**されて**いるか」「書き込**んだ**」は
#: 連体修飾・名詞化・願望・受身で、書込みの述語ではない (2026-09-27 レビュー)。
#: テ形の後の「ある / いる / いた / おり」(「追記してある内容」「追記しており」) も除く。
_GOVERNING_TAIL_RE = re.compile(
    r"^(?:し|ん|え)?(?:て|で)(?!あ[るっ]|い[るたな]|お[りら]|た)"
    r"|^(?:し|ん|え)?(?:と|ど)(?=い)"
    r"|^\s*(?:を)?\s*(?:お願い|頼)",
)
#: 宛先句と書込みの述語の間に別の述語がある形 (「誤字がないか確認して追記して」)。
_INTERVENING_PREDICATE_RE = re.compile(r"して|って|んで|か確認|たら|なら|ので|けど|、")
#: 引用の中身 (「git stash の使い方」) は述語の解析から外す。
_QUOTED_SPAN_RE = re.compile(r"「[^」]*」|『[^』]*』|\"[^\"]*\"")
#: 「保存したファイルに加えて」— 「に加えて」は宛先ではなく「〜のほかに」。
_IN_ADDITION_TO_RE = re.compile(r"^\s*に\s*加え")
#: ``を`` 格で直近ファイルそのものを書き換える動詞 (「作成していただいたファイルを
#: 更新して」)。「保存したファイルを読んで」「…を削除して」は含めない。
_REWRITE_OBJECT_VERB_RE = re.compile(
    r"更新|上書き|書き直|書き換|追記|書き足"
    r"|" + ascii_boundary_alternation("update", "overwrite", "append"),
    re.IGNORECASE,
)
#: どこかへ保存・書き出す動詞 (直近ファイルを出典として使いながら書こうとして
#: いるかの手掛かり。宛先が直近ファイルかどうかは決まらない)。
_SAVE_ELSEWHERE_VERB_RE = re.compile(
    r"保存|書き出|書出|書き込|書込|出力|エクスポート|追記|上書き|書き足"
    r"|" + ascii_boundary_alternation("save", "export", "write", "append", "overwrite"),
    re.IGNORECASE,
)

#: 説明節の直後の格が **出典・話題** (宛先ではない) の形。
_SOURCE_CASE_RE = re.compile(
    r"^\s*(?:に|へ)\s*(?:ある|あった|書いてあ|書かれ|載って|含まれ|入って|記載"
    r"|基づ|もとづ|ついて|関して|関する|対して|対する|の)",
)
#: 説明節の直後の宛先の格 (閉じ括弧を挟む形も)。
_DESTINATION_CASE_RE = re.compile(r"^[)）」』\s]*(?:に|へ)")
#: 「保存したファイルの末尾に」型の宛先。
_POSITION_DESTINATION_RE = re.compile(
    r"^\s*の\s*(?:末尾|最後|最終行|先頭|冒頭|最初|後ろ|続き|下|終わり)\s*(?:に|へ)",
)
#: 目的語の格。
_OBJECT_CASE_RE = re.compile(r"^\s*を")
#: 宛先句と書込みの動詞の間の目的語 (「次の一文**を**追加して」) の中の連体形の
#: 述語 (「今日決め**た**内容」「追加す**る**項目」「新し**い**節」)。語ではなく
#: 活用語尾で見る (2026-09-28): う段 / い は直後に名詞 (漢字・カタカナ・英数) が
#: 続く形、た / だ は後ろの字種を問わない (「今日決め**た**ことを」— 2026-09-28
#: レビュー M2 で「こと / もの / とおり」が抜けていた)。
_ADNOMINAL_PREDICATE_RE = re.compile(
    rf"[うくぐすつぬぶむるい](?=[{KANJI}{KATAKANA}A-Za-z0-9０-９])|[ただ](?=.)",
)
#: 同じ目的語の中の別の宛先の格 (「CSV **に**した表を」)。
_OTHER_CASE_IN_OBJECT_RE = re.compile(r"[にへ]")
#: 目的語の出典が別のもの (「メモ帳**の内容**を」「別**のファイル**を」「あ**のファイル**を」)。
#: 名詞の集合は説明節の対象の名詞 (``DESCRIBED_OBJECT_PATTERN``) と同じ。近称・中称の
#: 指示詞 (「この / その内容を」) は直前の会話の内容を指すので出典の別扱いにしない
#: (「同じ内容を」「このメッセージを」と揃える、2026-09-28 再レビュー 4)。ファイルは
#: 指示詞が付いても別のファイル。
_OBJECT_FROM_ELSEWHERE_RE = re.compile(
    rf"(?<![こそ])の\s*{DESCRIBED_OBJECT_PATTERN}|の\s*ファイル",
)
#: 依頼文と中身 (本文) の区切り (「次の一文を追加してください：以上です。」)。
#: ASCII の ``:`` はパス (``E:\``) や URL にも現れるので、後ろに空白が続く形だけ。
_PAYLOAD_COLON_RE = re.compile(r"：|:(?=\s)")
#: 後ろに続く中身を指す前方参照 (「次の一文を」「以下を」)。
_FORWARD_REFERENCE_RE = re.compile(FORWARD_REFERENCE_JA)
#: 中身の無い括弧書きだけの後半 (「（中身は後で送ります）」)。
_PARENTHETICAL_ONLY_RE = re.compile(r"^[（(][^）)]*[）)][。．.\s]*$")
#: 後半が前半を引用して続ける形 (「：と言ったら」「：って言われたら」)。
_QUOTING_TAIL_RE = re.compile(r"^(?:と|って)")

#: 問いの文末 (``intent_vocab.QUESTION_END_RE`` が SSOT)。
_QUESTION_END_RE = QUESTION_END_RE


def _verdict(value: str | None, evidence: str) -> Verdict:
    if value is None:
        return Verdict(
            value=None, score=0.0, band="abstain",
            evidence=evidence, predicate=f"{PREDICATE_NAME}_rule", stage="lexical",
        )
    fired = value != NEGATIVE_LABEL
    return Verdict(
        value=value, score=1.0 if fired else 0.0,
        band="fire" if fired else "skip",
        evidence=evidence, predicate=f"{PREDICATE_NAME}_rule", stage="lexical",
    )


def _names_other_destination(text: str) -> bool:
    """説明節とは別の宛先 (パス / 宛先のファイル名 / 参照表現) が立っているか。"""
    # router が本モジュールを import する (層の振り分け) ので、逆向きは遅延
    # import にして循環を作らない。
    from backend.free.agent.router import indicates_write_destination

    return indicates_write_destination(text)


def _asks_location(text: str) -> bool:
    """直近ファイルの場所を尋ねているか (書込みの動詞を伴わない)。

    参照は「ファイル」に掛かる形に限る (説明節 + ファイル / 指示詞 + ファイル =
    ``REFERENTIAL_WRITE_TARGET_RE``)。``file_ledger.references_recent_file`` は
    「パスが抽出できなかった」後に使う緩い判定で、指示詞と対象語が離れていても
    当たる (「ファイルパスがないタスクは**その**まま」) ので、ここでは使わない。
    """
    if not _LOCATION_ASK_RE.search(text):
        return False
    if not REFERENTIAL_WRITE_TARGET_RE.search(text):
        return False
    stripped = strip_file_reference_clauses(text)
    return not (
        WRITE_VERB_RE.search(stripped)
        or _SAVE_ELSEWHERE_VERB_RE.search(stripped)
        or _RELOCATE_VERB_RE.search(stripped)
    )


def _governing_write_verb(rest: str, verb_re: re.Pattern[str]) -> bool:
    """宛先句の後の **最初の述語** が依頼のテ形の書込み動詞か。

    ``rest`` は格助詞の直後からの本文 (引用の中身は伏せてある)。最初に現れる
    書込みの動詞が依頼のテ形 / 「お願い」で終わり、それより前に別の述語が
    挟まっていないときだけ真。
    """
    m = verb_re.search(rest)
    if m is None:
        return False
    if _INTERVENING_PREDICATE_RE.search(rest[:m.start()]):
        return False
    return bool(_GOVERNING_TAIL_RE.match(rest[m.end():]))


def _plain_object_gap(rest: str, verb_re: re.Pattern[str]) -> bool:
    """宛先句と書込みの動詞の間に目的語 (「<名詞句>を」) があれば、それが素の名詞句か。

    名詞句に連体形の述語 (「今日決めた内容を」)・別の宛先の格 (「CSV にした表を」)・
    別のファイル名 / パスが入ると、何をどこへ書くかが句の中で別の出来事に
    掛かっている。目的語が無い形 (「1行追記して」) はこの検査の対象外。
    """
    m = verb_re.search(rest)
    gap = rest[:m.start()] if m else ""
    cut = gap.rfind("を")
    if cut < 0:
        return True
    phrase = gap[:cut]
    return not (
        _ADNOMINAL_PREDICATE_RE.search(phrase)
        or _OTHER_CASE_IN_OBJECT_RE.search(phrase)
        or _OBJECT_FROM_ELSEWHERE_RE.search(phrase)
        or FILE_NAME_IN_TEXT_RE.search(phrase)
        or EXPLICIT_WINDOWS_PATH_RE.search(phrase)
    )


def _second_write_verb(rest: str, verb_re: re.Pattern[str]) -> bool:
    """最初の書込みの述語の後ろに、別の書込みの述語 (依頼のテ形) が続くか。

    「保存したファイルに全部を追記して上書きして」— 追記と上書きのどちらを
    するのかが決まらない (2026-09-28 レビュー L6)。
    """
    first = verb_re.search(rest)
    if first is None:
        return False
    for m in verb_re.finditer(rest, first.end()):
        if m.group(0) != first.group(0) and _GOVERNING_TAIL_RE.match(rest[m.end():]):
            return True
    return False


def _split_payload(sentence: str) -> tuple[str, str | None]:
    """「：」で依頼 (前半) と中身 (後半) に割る。区切りが無ければ ``(文, None)``。

    「作ったファイルに次の一文を追加してください：以上です。」の「以上です。」は
    追記する中身で、文末として依頼か否かを決めない (2026-09-28)。依頼か報告か問いかは
    **前半** で読む (「追記しておきました：箇条書きで」は報告 — レビュー L4)。
    """
    m = _PAYLOAD_COLON_RE.search(sentence)
    if m is None:
        return sentence, None
    return sentence[:m.start()], sentence[m.end():]


def _payload_like(head: str, tail: str) -> bool:
    """後半が前半の依頼の **中身** らしいか (2026-09-28 レビュー H2)。

    前半が中身を前方参照していて (「次の一文を」「以下を」)、後半が問い・引用の続き
    (「：と言ったら」)・別の依頼 (「：いや、やっぱりやめて」)・中身の無い括弧書き
    (「：（中身は後で送ります）」) でないときだけ真。改行で後ろの行へ続く形
    (後半が空) は中身が次の行にある。
    """
    if not _FORWARD_REFERENCE_RE.search(head):
        return False
    body = tail.strip()
    if not body:
        return True
    return not (
        _QUESTION_END_RE.search(body)
        or _QUOTING_TAIL_RE.match(body)
        or is_request_sentence(body)
        or _PARENTHETICAL_ONLY_RE.match(body)
    )


def _clause_role(clause: Any, rest: str) -> tuple[str, str]:
    """説明節の直後の格から ``(役割, 格の後ろの本文)`` を返す。

    役割は ``destination`` (に / へ / の末尾に) / ``object`` (を) / ``source`` (それ以外)。
    """
    if not clause.names_file or _IN_ADDITION_TO_RE.match(rest) or _SOURCE_CASE_RE.match(rest):
        return "source", rest
    m = _POSITION_DESTINATION_RE.match(rest) or _DESTINATION_CASE_RE.match(rest)
    if m:
        return "destination", rest[m.end():]
    m = _OBJECT_CASE_RE.match(rest)
    if m:
        return "object", rest[m.end():]
    return "source", rest


def _clause_verdict(
    text: str, sentence: str, ctx: Mapping[str, Any] | None,
) -> Verdict | None:
    """1 文の中の説明節を読む。説明節が無ければ ``None``。

    write は **狭い型だけ** (c_17 §3.11): 宛先 / 目的語の格の後の最初の述語が
    依頼のテ形の書込み動詞、文が依頼の形、主体が user でない、直近のファイルが
    ある。書込みの動詞が絡んで型に当たらない依頼は棄権 (実行せず確認する)。
    """
    sentence, payload = _split_payload(sentence)
    clauses = find_file_reference_clauses(sentence)
    if not clauses:
        return None
    masked = _QUOTED_SPAN_RE.sub("「」", sentence)
    is_request = is_request_sentence(sentence)
    best: Verdict | None = None
    for clause in find_file_reference_clauses(masked) or clauses:
        rest = masked[clause.end:]
        wants_write = bool(
            _WRITE_INTO_VERB_RE.search(rest) or _SAVE_ELSEWHERE_VERB_RE.search(rest),
        )
        if not wants_write:
            verdict = _verdict(NEGATIVE_LABEL, "no_write_verb")
        elif not is_request:
            # 問い (「追記するとどうなりますか」)・報告 (「追記しました」)・願望
            # (「更新したいです」) は依頼ではない。実行も確認もしない。
            evidence = (
                "question_form" if _QUESTION_END_RE.search(sentence.strip())
                else "not_a_request"
            )
            verdict = _verdict(NEGATIVE_LABEL, evidence)
        elif clause.subject == "user":
            # 台帳はアシスタントの書込みしか持たない。ユーザーのファイルを
            # 台帳の最新へ解決すると別のファイルを書き換える。
            verdict = _verdict(None, "subject_user")
        else:
            verdict = _write_or_confirm(text, clause, rest, ctx)
            if verdict.fired and payload is not None and not _payload_like(sentence, payload):
                # 「：」の後ろが中身らしくない (問い・取り消し・中身の無い括弧書き、
                # または前半が中身を前方参照していない)。前半だけで書込みを確定
                # しない (2026-09-28 レビュー H2)。
                verdict = _verdict(None, "colon_not_payload")
        if best is None or _strength(verdict) > _strength(best):
            best = verdict
    return best


def _write_or_confirm(
    text: str, clause: Any, rest: str, ctx: Mapping[str, Any] | None,
) -> Verdict:
    """依頼の文で書込みの動詞がある説明節を、write か棄権か不発に振り分ける。"""
    role, after = _clause_role(clause, rest)
    if role == "destination" and _governing_write_verb(after, _WRITE_INTO_VERB_RE):
        if not _plain_object_gap(after, _WRITE_INTO_VERB_RE):
            # 「保存したファイルに今日決めた内容を追記して」— 何を書くかが句の中の
            # 別の出来事に掛かる。write に広げず確認する (2026-09-28)。
            return _verdict(None, "object_not_plain")
        if _second_write_verb(after, _WRITE_INTO_VERB_RE):
            return _verdict(None, "verbs_conflict")
        evidence = "clause_destination"
    elif role == "object" and _governing_write_verb(after, _REWRITE_OBJECT_VERB_RE):
        evidence = "clause_update_object"
    else:
        # 直近ファイルを出典・話題として使う、または書込みの動詞が節を支配して
        # いない。別の宛先が書かれていれば従来の規則 (local_write_intent) に任せる。
        if _names_other_destination(text):
            return _verdict(NEGATIVE_LABEL, "other_destination")
        return _verdict(None, "clause_as_source" if role == "source" else "not_governing")
    if ctx is not None and ctx.get("has_recent_file") is False:
        # 書込み先が決まらない (新しいセッションの「前回保存したファイル」)。
        return _verdict(None, "no_recent_file")
    return _verdict(WRITE_LABEL, evidence)


def _strength(verdict: Verdict) -> int:
    """発火 > 棄権 > 不発。"""
    return {"fire": 2, "abstain": 1}.get(verdict.band, 0)


def _rule(text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
    """``ctx["has_recent_file"]`` は直近のファイルがあるか (呼出側が台帳の復元後に
    渡す)。渡されなければ字句だけで判定する。"""
    if not text or write_prohibited(text):
        return _verdict(NEGATIVE_LABEL, "no_text" if not text else "prohibited")
    if _asks_location(text):
        return _verdict(LOCATE_LABEL, "location_query")
    best: Verdict | None = None
    for sentence in split_sentences(text) or [text]:
        verdict = _clause_verdict(text, sentence, ctx)
        if verdict is not None and (best is None or _strength(verdict) > _strength(best)):
            best = verdict
    return best if best is not None else _verdict(NEGATIVE_LABEL, "no_reference")


class _RecentFileReferenceRule:
    """字句段 (``Predicate`` プロトコル)。根拠を段ごとに書き分けるため
    ``LexicalPredicate`` ではなく ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        return _rule(text or "", ctx)


_RULE = _RecentFileReferenceRule()

#: プロセス共通の判定点。chat モードのターンごとに ``chat.py`` が 1 回だけ引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[WRITE_LABEL, LOCATE_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def recent_file_reference_verdict(
    text: str, ctx: Mapping[str, Any] | None = None,
) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。ターンに 1 回)。"""
    return predicate.evaluate(text or "", ctx)


def recent_file_reference_rule(
    text: str, ctx: Mapping[str, Any] | None = None,
) -> Verdict:
    """字句段だけを **記録せずに** 評価する (純粋関数)。

    同じターンの 2 度目以降の読み手 (ルータの縮退・deliberative の短絡) が使う。
    記録はターンに 1 回 (:func:`recent_file_reference_verdict`)。
    """
    return _RULE.evaluate(text or "", ctx)


def recent_file_context(session_id: str, conversation: list[dict] | None) -> dict[str, bool]:
    """判定点の文脈: 台帳を会話履歴から復元した上で、直近に **書いた** ファイルがあるか。

    台帳はプロセス内だけなので、再起動後・別セッションでは空になる。空のまま
    write にすると書込み先の無いタスクが meta へ入る (2026-09-27 レビュー M2)。
    読んだだけのファイルは「保存したファイル」の宛先にしない (2026-09-28 レビュー
    H1、宛先の ``chat._recent_file_write_target`` と同じ ``last_written_path``)。
    """
    restore_from_conversation(session_id, conversation)
    return {"has_recent_file": bool(session_id and last_written_path(session_id))}


def is_sole_write_request(text: str) -> bool:
    """直近ファイルへの書込みの述語が、発話の **唯一の依頼** か (計画を飛ばしてよいか)。

    判定点が ``write`` を返したターンのうち、書込みの述語だけで依頼が尽きている
    形だけを真にする (docs/f_03 §4.3)。前段 (「東京の天気を調べて、保存したファイルに
    追記して」)・後段 (「追記して、report.md にも同じ内容を書いて」「追記してから要約
    して」)・別の依頼の文 (「…追記してください。あと、Python の使い方も教えて」) が
    あれば偽 — 1 タスクに潰すと残りの段が消える (2026-09-28 レビュー H2)。判定の部品は
    ``write`` の判定と同じもの (説明節・格・支配する動詞・間に挟まる述語) を使う。
    コロンの後ろの中身は依頼に数えない。
    """
    requests = [
        s for s in split_sentences(text or "") or [text or ""]
        if is_request_sentence(_split_payload(s)[0])
    ]
    if len(requests) != 1:
        return False
    head, _payload = _split_payload(requests[0])
    masked = _QUOTED_SPAN_RE.sub("「」", head)
    clauses = find_file_reference_clauses(masked)
    if len(clauses) != 1:
        return False
    clause = clauses[0]
    if _INTERVENING_PREDICATE_RE.search(masked[:clause.start]):
        return False
    role, after = _clause_role(clause, masked[clause.end:])
    verb_re = {"destination": _WRITE_INTO_VERB_RE, "object": _REWRITE_OBJECT_VERB_RE}.get(role)
    verb = verb_re.search(after) if verb_re is not None else None
    if verb is None:
        return False
    tail = _GOVERNING_TAIL_RE.match(after[verb.end():])
    if tail is None:
        return False
    return not _INTERVENING_PREDICATE_RE.search(after[verb.end() + tail.end():])


__all__ = [
    "LOCATE_LABEL",
    "PREDICATE_NAME",
    "WRITE_LABEL",
    "bind_debug_logger",
    "is_sole_write_request",
    "predicate",
    "recent_file_context",
    "recent_file_reference_rule",
    "recent_file_reference_verdict",
]
