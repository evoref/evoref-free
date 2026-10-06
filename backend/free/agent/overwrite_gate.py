"""既存ファイルが今の依頼の書込みの対象かの判定点 ``overwrite_target`` (c_17 §3.21)。

書込みゲート (``write_gate.check_write_target``、docs/f_03 §4.y) が、``write_file`` の
書込み先が **既に在るファイル** のときだけ引く (制作ステージの配信は引かない)。入力は
**このターンの依頼文** (ユーザーの発話) で、計画のタスク文・LLM の引数は見ない (不変則 #15:
計画モデルが「このファイル」をパスに展開したタスク文は近道であって依頼ではない)。

実インシデント (2026-10-05 ライブ監査 T4): 「このファイルに対する pytest のテストを3つ
書いてください。」で、計画のタスク「Generate three pytest tests for E:\\…\\util_fixed.py」が
``write_file`` で util_fixed.py をテストの本文で上書きした (テストが import するモジュール
そのものが消えた)。派生物 (テスト・要約・翻訳・報告) の **題材** として挙がっただけの
ファイルが派生物で上書きされる型の再発で、経路ごとに塞いできたものを書込みの最後の
合流点で止める。

| band / value | 条件 | evidence |
|---|---|---|
| ``fire`` / ``target`` | 名前 / パスが宛先の格 / 目的語の格 + 書込み・編集・作成の動詞 | ``named_destination`` / ``named_object`` |
| ``fire`` / ``target`` | 名前 / パスと書き戻しの語 (上書き / 保存し直 / overwrite) が同じ文 (別の宛先無し) | ``named_overwrite`` |
| ``fire`` / ``target`` | 参照表現が宛先 / 目的語 + 動詞 / 書き戻しの語そのもの (台帳に在るファイル) | ``referential_destination`` / ``referential_object`` / ``referential_token`` |
| ``fire`` / ``target`` | 宛先に名指したファイルの長文 SPLIT の出力 (``{stem}_NN_*`` / ``{stem}_INDEX.md``) | ``named_split_output`` |
| ``fire`` / ``edit`` | 名前 / 台帳に在る参照表現が話題・根拠の格以外で現れ、別の宛先が無い (動詞は問わない) | ``named_edit`` / ``referential_edit`` |
| ``fire`` / ``edit`` | 宛先の格でも後ろが追記の語 (追記は既存の行を残すはず) | ``named_append`` / ``referential_append`` |
| ``fire`` / ``edit`` | 拡張子無しの名前 (stem がフォルダで一意) — 普通の語と同じ綴りなので格によらず | ``stem_edit`` |
| ``fire`` / ``edit`` | 話題・根拠の格でも、依頼に保存の動詞がある | ``topic_with_save`` |
| ``fire`` / ``edit`` | 依頼に現れないが、セッションで最後に書いたファイル | ``last_written_edit`` |
| ``fire`` / ``edit`` | 名前も参照表現も無く、祖先のフォルダが宛先の役 | ``folder_destination`` |
| ``skip`` / ``none`` | 上のどれでもない | ``named_as_source`` / ``referential_as_source`` / ``other_destination`` / ``not_named`` / ``prohibited`` |

``edit`` は語では決まらない形で、呼出側が **書く中身** で確かめる (:func:`retains_existing`):
編集は既存の本文の大部分を引き継ぎ、派生物は引き継がない。動詞の一覧で ``edit`` と
``skip`` を分けない — 「変えて」「リファクタして」が一覧に無く断られ、「refactor util.py」と
非対称だった (2026-10-05 独立レビュー H2)。``skip`` に残すのは格と宛先で言える形だけ。
棄権の帯は持たない — 断る側が害の残らない側 (生成した本文は応答に残る)。

格の語 (宛先 / 目的語 / 話題・根拠) は ``recent_file_reference`` と同じ 1 本
(``file_reference_gate.case_role`` / ``is_topic_case``)、フォルダの役は
``file_ledger.folder_role``、保存の動詞はルータの ``has_save_verb`` を読む (不変則 #14 (a))。
語形は足していない。
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.free.core.intent_vocab import (
    APPEND_HINT_RE,
    EN_ADD_VERB_RE,
    EN_DESTINATION_BEFORE_RE,
    FILE_NAME_IN_TEXT_RE,
    REFERENTIAL_WRITE_TARGET_RE,
    split_sentences,
    write_prohibited,
)
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)

PREDICATE_NAME = "overwrite_target"

#: 依頼がこのファイルを書込みの対象に挙げている (そのまま書く)。
TARGET_LABEL = "target"
#: 題材として挙げている (書く中身が既存の本文を引き継ぐときだけ書く)。
EDIT_LABEL = "edit"

#: ``edit`` のとき、既存の本文の中身のある行のうち新しい本文に残るべき割合。
#: 編集 (バグの修正・語の差し替え・追記) は大部分を残し、派生物 (テスト・要約・翻訳) は
#: ほとんど残さない。
_RETAINED_LINE_RATIO = 0.5
#: 引継ぎを確かめるために読む既存ファイルの上限 (``write_file`` の読みと同じ)。
_MAX_COMPARE_BYTES = 2_000_000
#: 拡張子無しの名前 (stem) で名指しとみなす最短の長さ (「a」「io」を語と取り違えない)。
_MIN_STEM_CHARS = 3

#: 名指しの直後の閉じ括弧・引用符 (「「util.py」を修正して」)。
_CLOSING_RE = re.compile(r"^[)）」』】\]\"'`]+")
#: 長文 SPLIT の出力の名前の後ろ半分 (``chat_stream_output._resolve_split_unit_path`` /
#: ``split_write_index`` が ``{stem}`` の後ろに付ける形)。
_SPLIT_SUFFIX_RE = re.compile(r"_(?:\d{2}_.+|INDEX\.md)$", re.IGNORECASE)
#: フォルダを宛先にした SPLIT の ``{stem}`` (``_resolve_split_unit_path`` / ``split_write_index``)。
_SPLIT_FOLDER_STEM = "output"


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _name_re(name: str, *, stem: bool = False) -> re.Pattern[str]:
    """ファイル名の出現 (前後が ASCII の名前の文字でないこと。和文は区切りに数える)。

    ``stem`` (拡張子無しの名前) は後ろに拡張子が続く形 (「util.txt」) を除く。
    """
    tail = r"(?![A-Za-z0-9_\-]|\.[A-Za-z0-9])" if stem else r"(?![A-Za-z0-9_\-])"
    return re.compile(rf"(?<![A-Za-z0-9_.\-]){re.escape(name)}{tail}", re.IGNORECASE)


def _unique_stem(path: str) -> str:
    """フォルダの中で拡張子無しの名前が 1 つだけなら、その名前 (でなければ空文字)。"""
    p = Path(path)
    if not p.suffix or len(p.stem) < _MIN_STEM_CHARS:
        return ""
    try:
        siblings = [e.name for e in os.scandir(p.parent) if e.is_file()]
    except OSError:
        return ""
    same = [n for n in siblings if Path(n).stem.lower() == p.stem.lower()]
    return p.stem if len(same) == 1 else ""


def _drive_spans(sentence: str) -> list[tuple[int, int]]:
    from backend.free.agent.tool_judge_args import drive_path_spans

    return drive_path_spans(sentence)


def _role_at(sentence: str, start: int, end: int) -> tuple[str, str]:
    """``sentence[start:end]`` の名指しの役割 ``(destination | object | source, 格の後ろ)``。"""
    from backend.free.agent.file_reference_gate import case_role, verb_directly_before

    before = sentence[:start]
    rest = _CLOSING_RE.sub("", sentence[end:])
    if EN_DESTINATION_BEFORE_RE.search(before):
        # 英語は動詞が名指しの前に来る (「append tests to util.py」) — 追記の語は前で読む
        return "destination", before
    if verb_directly_before(before):
        return "object_governed", rest
    return case_role(rest)


def _is_target_role(role: str, after: str) -> str:
    """役割が書込みの対象なら ``destination`` / ``object``、でなければ空文字。"""
    from backend.free.agent.file_reference_gate import object_governed_by_write

    if role == "destination":
        return "destination"
    if role == "object_governed" or (role == "object" and object_governed_by_write(after)):
        return "object"
    return ""


def _mentions(
    sentence: str, path: str,
) -> tuple[list[tuple[int, int, str]], list[str]]:
    """文の中のこのファイルの名指し ``(開始, 終わり, named | stem)`` と、宛先に挙がっている
    別のファイル / フォルダ。"""
    key = _path_key(path)
    name = os.path.basename(path)
    spans = _drive_spans(sentence)
    own: list[tuple[int, int, str]] = []
    destinations: list[str] = []
    for start, end in spans:
        literal = sentence[start:end].strip("\"'「」『』")
        literal_key = _path_key(literal)
        if literal_key == key:
            own.append((start, end, "named"))
            continue
        if key.startswith(literal_key.rstrip("\\/") + os.sep):
            continue  # このファイルの祖先のフォルダ (置き場の指定で、別の宛先ではない)
        if _is_target_role(*_role_at(sentence, start, end)):
            destinations.append(literal)
    patterns = [(_name_re(name), "named")] if name else []
    stem = _unique_stem(path)
    if stem:
        patterns.append((_name_re(stem, stem=True), "stem"))
    for pattern, kind in patterns:
        for m in pattern.finditer(sentence):
            if any(s <= m.start() < e for s, e in spans) or any(s == m.start() for s, _, _ in own):
                continue  # パスの一部 (上で見た) / 同じ名指し
            own.append((m.start(), m.end(), kind))
    for m in FILE_NAME_IN_TEXT_RE.finditer(sentence):
        if any(s <= m.start() < e for s, e in spans) or m.group(0).lower().endswith(name.lower()):
            continue
        if _is_target_role(*_role_at(sentence, m.start(), m.end())):
            destinations.append(m.group(0))
    return own, destinations


def _referential_mentions(sentence: str) -> list[tuple[int, int, bool]]:
    """参照表現の区間と、それが書き戻しの語そのもの (上書き / 保存し直 …) か。"""
    from backend.free.agent.file_reference_gate import names_write_verb

    found = []
    for m in REFERENTIAL_WRITE_TARGET_RE.finditer(sentence):
        text = m.group(0).rstrip()
        token = names_write_verb(text) and not text.lower().endswith(("ファイル", "file"))
        found.append((m.start(), m.end(), token))
    return found


def _split_output_of(path: str, destinations: list[str], query: str) -> bool:
    """``path`` が長文 SPLIT の出力の作り直しか。

    宛先に名指したファイルの ``{stem}_NN_*`` / ``{stem}_INDEX.md`` (同じフォルダ) か、親フォルダ
    そのものが宛先の役のときの ``output_NN_*`` / ``output_INDEX.md``。
    """
    from backend.free.agent.file_ledger import folder_role

    p = Path(path)
    for dest in destinations:
        d = Path(dest)
        if not d.suffix:
            continue  # 別のフォルダ (このファイルの親は ``destinations`` に入らない)
        parent = d.parent if d.is_absolute() or len(d.parts) > 1 else p.parent
        if _path_key(str(parent)) != _path_key(str(p.parent)):
            continue
        rest = p.name[len(d.stem):] if p.name.lower().startswith(d.stem.lower()) else ""
        if rest and _SPLIT_SUFFIX_RE.fullmatch(rest):
            return True
    stem = _SPLIT_FOLDER_STEM
    if p.name.lower().startswith(stem) and _SPLIT_SUFFIX_RE.fullmatch(p.name[len(stem):]):
        return folder_role(query, str(p.parent)) == "destination"
    return False


def _verdict(value: str, evidence: str) -> Verdict:
    fired = value != NEGATIVE_LABEL
    return Verdict(
        value=value, score=1.0 if fired else 0.0, band="fire" if fired else "skip",
        evidence=evidence, predicate=f"{PREDICATE_NAME}_rule", stage="lexical",
    )


def _ancestors(path: str) -> list[str]:
    p = Path(os.path.abspath(path)).parent
    out = []
    while True:
        out.append(str(p))
        if p.parent == p:
            return out
        p = p.parent


def _rule(query: str, ctx: Mapping[str, Any] | None) -> Verdict:
    """``ctx["path"]`` は書込み先 (既存のファイル)、``ctx["in_session"]`` はこのセッションの
    ファイル台帳にそのファイルが在るか (参照表現が指しうるか)、``ctx["last_written"]`` は
    セッションで最後に書いたファイルか。"""
    from backend.free.agent.file_ledger import folder_role
    from backend.free.agent.file_reference_gate import is_topic_case
    from backend.free.agent.router import has_save_verb

    ctx = ctx or {}
    path = str(ctx.get("path") or "")
    if not query or not path:
        return _verdict(NEGATIVE_LABEL, "not_named")
    if write_prohibited(query):
        return _verdict(NEGATIVE_LABEL, "prohibited")
    in_session = bool(ctx.get("in_session"))
    referenced = False
    edit_evidence = ""
    topic_evidence = ""
    destinations: list[str] = []
    for sentence in split_sentences(query) or [query]:
        own, others = _mentions(sentence, path)
        destinations += others
        refs = _referential_mentions(sentence)
        overwrite_word = any(token for _, _, token in refs)
        mentions = list(own)
        if in_session:
            mentions += [
                (start, end, "referential_token" if token else "referential")
                for start, end, token in refs
            ]
        for start, end, kind in mentions:
            referenced = True
            role, after = _role_at(sentence, start, end)
            if role == "source" and is_topic_case(after):
                topic_evidence = topic_evidence or f"{kind}_as_source"
                continue
            if kind == "stem":
                # 拡張子無しの名前は普通の語と同じ綴り (notes / data / report / test)。
                # 格によらず中身で確かめる側に置く (2 周目レビュー HIGH-2)
                edit_evidence = edit_evidence or "stem_edit"
                continue
            if kind == "referential_token" or (kind == "named" and overwrite_word and not others):
                return _verdict(
                    TARGET_LABEL,
                    "referential_token" if kind == "referential_token" else "named_overwrite",
                )
            target = _is_target_role(role, after)
            if target == "destination" and (
                APPEND_HINT_RE.search(after) or EN_ADD_VERB_RE.search(after)
            ):
                # 追記は既存の行を残すはず — 中身で確かめる (2 周目 LOW-3 / 3 周目 MED-B)
                edit_evidence = edit_evidence or f"{kind}_append"
                continue
            if target:
                return _verdict(TARGET_LABEL, f"{kind}_{target}")
            edit_evidence = edit_evidence or f"{kind}_edit"
    if _split_output_of(path, destinations, query):
        return _verdict(TARGET_LABEL, "named_split_output")
    if destinations:
        # 題材と別の宛先がある (「util.py を要約して summary.md に保存して」)。現れない
        # ファイル (最後に書いたファイル / フォルダの宛先) も同じく別の宛先の書込みではない
        return _verdict(NEGATIVE_LABEL, "other_destination")
    if referenced:
        if edit_evidence:
            return _verdict(EDIT_LABEL, edit_evidence)
        if has_save_verb(query):
            return _verdict(EDIT_LABEL, "topic_with_save")
        return _verdict(NEGATIVE_LABEL, topic_evidence or "named_as_source")
    if ctx.get("last_written"):
        return _verdict(EDIT_LABEL, "last_written_edit")
    # フォルダを宛先に挙げただけでは、その中のどのファイルを書き換えるかを言っていない
    # (「E:\work にテストを書いて保存して」で同じフォルダのモジュールを上書きしない)。
    # 作り直し (同じ依頼の 2 度目) と区別できないので中身で確かめる側 (edit) に置く。
    if any(folder_role(query, folder) == "destination" for folder in _ancestors(path)):
        return _verdict(EDIT_LABEL, "folder_destination")
    return _verdict(NEGATIVE_LABEL, "not_named")


class _OverwriteTargetRule:
    """字句段 + 構造段 (``Predicate`` プロトコル)。根拠を書き分けるため ``Verdict`` を直接返す。"""

    name = f"{PREDICATE_NAME}_rule"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:
        return _rule(text or "", ctx)


_RULE = _OverwriteTargetRule()

predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_RULE,
        policy="complement",
        candidates=[TARGET_LABEL, EDIT_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def _ctx(path: str, in_session: bool, last_written: bool) -> dict[str, Any]:
    return {"path": path, "in_session": in_session, "last_written": last_written}


def overwrite_target_rule(
    query: str, path: str, *, in_session: bool = False, last_written: bool = False,
) -> Verdict:
    """字句段だけを **記録せずに** 評価する (純粋関数)。"""
    return _RULE.evaluate(query or "", _ctx(path, in_session, last_written))


def overwrite_target_verdict(
    query: str, path: str, *, in_session: bool = False, last_written: bool = False,
) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する)。"""
    return predicate.evaluate(query or "", _ctx(path, in_session, last_written))


def retains_existing(path: str, content: str | None) -> bool | None:
    """新しい本文が既存のファイルの本文を引き継いでいるか (確かめられなければ ``None``)。

    既存の中身のある行 (前後の空白を除く) のうち、新しい本文の行に残る割合が
    :data:`_RETAINED_LINE_RATIO` 以上なら真。既存が空なら失うものが無いので真。
    テキストとして読めない (Office 文書・符号化の判別不能)・大きすぎる・本文が無いときは
    ``None`` (呼出側は確かめられないとして従来どおり書く)。
    """
    from backend.io.text_file import TextFile, read_text_for_edit

    if content is None:
        return None
    try:
        existing = read_text_for_edit(Path(path), max_bytes=_MAX_COMPARE_BYTES)
    except OSError:
        return None
    if not isinstance(existing, TextFile):
        return None
    old_lines = {line.strip() for line in existing.text.splitlines() if line.strip()}
    if not old_lines:
        return True
    new_lines = {line.strip() for line in content.splitlines() if line.strip()}
    return len(old_lines & new_lines) / len(old_lines) >= _RETAINED_LINE_RATIO


__all__ = [
    "EDIT_LABEL",
    "PREDICATE_NAME",
    "TARGET_LABEL",
    "bind_debug_logger",
    "overwrite_target_rule",
    "overwrite_target_verdict",
    "predicate",
    "retains_existing",
]
