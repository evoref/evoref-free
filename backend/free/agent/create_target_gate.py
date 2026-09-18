"""staged create の「作成対象が発話に名指しされているか」の判定点 (Phase 3b)。

``chat.py::_ProductionStageSelector`` はタスクグラフ合成が 0 件 (モジュール
未検出) を返したとき、従来は無条件で longform へフォールバックしていた
(「何か」を黙って作る)。要求にパス / 言語 / 成果物の種類のいずれも含まれて
いないときは、代わりに問い返し (``needs_input``、f_03 §4.4) へ倒す —
その判定を担う。

字句段だけのカスケード (不変則 #14 / c_17)。``social_formula_gate`` /
``text_fabrication`` と同じ形: 事例段は未装着 (誤発火の余地が薄い全文検査
ではなく語彙の有無検査なので、``confirm`` の相手が要る局面になれば
``exemplar=`` を足す)。
"""

from __future__ import annotations

import re
from typing import Any

from backend.free.core.intent_vocab import ascii_boundary_alternation
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    LexicalPredicate,
    register_predicate,
)

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "create_target_named"

#: 発火ラベル (作成対象が名指しされている)。
TARGET_NAMED_LABEL = "target_named"

#: パス / ファイル名トークン (``tool_judge_signals._CODE_PATH_TOKEN_RE`` 相当)。
#: フルパス (区切り付き) と裸のファイル名 (拡張子付き、例: ``idgen.py``) の
#: 両方を拾う。
_TARGET_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?[\w.\-]+(?:[/\\][\w.\-]+)*\.[A-Za-z0-9]{1,6}\b",
)

#: 言語名 (≤ 20 語)。ASCII 語は境界必須 (``go`` が ``algorithm`` の一部に
#: 当たる等の誤爆を避ける)。
_TARGET_LANGUAGE_RE = re.compile(
    r"パイソン|[Cc]#|[Cc]\+\+|"
    + ascii_boundary_alternation(
        "python", "typescript", "javascript", "rust", "go", "java",
        "html", "css", "sql", "bash", "powershell",
    ),
    re.IGNORECASE,
)

#: 成果物名詞 (≤ 20 語)。
_TARGET_NOUN_RE = re.compile(
    r"関数|クラス|モジュール|パッケージ|スクリプト|ライブラリ|テスト|設定ファイル|"
    + ascii_boundary_alternation(
        "function", "class", "module", "package", "script", "library",
        "API", "CLI",
    ),
    re.IGNORECASE,
)


#: 「中身」に数えない語 (閉じた小集合)。創作動詞・丁寧表現・プレースホルダ・総称名詞。
#: ここを増やして直す運用にはしない — 迷う語は足さずに通す (通す = 従来挙動、f_03 §4.4)。
_VACUOUS_WORDS_RE = re.compile(
    r"何か(?:しら)?|なにか|何らかの|なんらかの|適当(?:に|な)|とりあえず|いい感じ(?:に|の)|"
    r"プログラム|コード|ソース|アプリ(?:ケーション)?|ソフト(?:ウェア)?|ツール|ファイル|もの|物|"
    r"新しい|新規(?:に|の)?|簡単な|シンプルな|小さな|いい|良い|面白い|便利な|お?任せ(?:します|る)?|"
    r"作成|生成|実装|作っ|作る|作り|書い|書く|書き|出力|保存|"
    r"して(?:ください|くれますか|くれる|ほしい|欲しい|もらえますか)?|ください|お願い(?:します)?|"
    r"ちょうだい|てみて|みて|"
    + ascii_boundary_alternation(
        "something", "anything", "some", "an", "the", "new", "simple",
        "program", "code", "app", "application", "tool", "file", "thing",
        "create", "make", "write", "generate", "build", "implement",
        "please", "me", "for", "can", "you", "could",
    ),
    re.IGNORECASE,
)
#: 助詞・記号・空白 (内容語の残りを数える前に落とす)。
_FILLER_CHARS_RE = re.compile(r"[\s\W_をがはにでとのもへやてだです。、！!？?・「」『』()（）]+")


def _is_vacuous(text: str) -> bool:
    """創作動詞・プレースホルダ・総称名詞を除くと内容語が残らない依頼か。"""
    rest = _VACUOUS_WORDS_RE.sub(" ", text)
    rest = _FILLER_CHARS_RE.sub("", rest)
    return len(rest) < 2


def _rule(text: str) -> str:
    """対象 (パス / 言語 / 成果物名詞) があるか、内容語が残れば発火ラベル。

    陰性 (= 問い返す) は「中身が空の依頼」だけ (f_03 §4.4 改訂 2)。
    """
    if not text:
        return NEGATIVE_LABEL
    if (
        _TARGET_PATH_RE.search(text)
        or _TARGET_LANGUAGE_RE.search(text)
        or _TARGET_NOUN_RE.search(text)
    ):
        return TARGET_NAMED_LABEL
    if not _is_vacuous(text):
        return TARGET_NAMED_LABEL
    return NEGATIVE_LABEL


#: プロセス共通の判定点。``chat.py`` から同期 ``evaluate`` で引く。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=LexicalPredicate(
            f"{PREDICATE_NAME}_rule", _rule, evidence="target_vocabulary",
        ),
        policy="complement",
        candidates=[TARGET_NAMED_LABEL, NEGATIVE_LABEL],
        scope="request",
    ),
)


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    predicate.bind_debug_logger(debug_logger)


def names_creation_target(text: str) -> bool:
    """発話が作成対象を持つか (パス / 言語 / 成果物名詞、または内容語が残る)。

    偽になるのは「中身が空の依頼」(何かプログラムを作って / ファイルを作って) だけ。
    """
    return predicate.evaluate(text or "").fired


__all__ = [
    "PREDICATE_NAME",
    "TARGET_NAMED_LABEL",
    "bind_debug_logger",
    "names_creation_target",
    "predicate",
]
