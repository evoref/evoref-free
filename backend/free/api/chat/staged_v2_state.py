"""staged v2 の検証済みの版と、作り直しの悪化の判定 (f_10 §11.1-3 / §11.1-4)。

作り直しのたびに「配線後の版」と「その版の検査結果」を :class:`VerifiedState` に写し、
:func:`compare` で前の版と比べる。比べ方は smoke の作り直し・契約テストの作り直しで 1 本。

1. **検査の欠け** を先に見る: 片方だけが「前は検査できていたのに検査できなくなった」項目を持つなら、
   その側が悪い (未検査を改善に数えない)。import スモークは、走らなかった (時間切れ・起動できない) か、
   未検査のモジュールの集合が増えたときに欠けとみなす — 件数が 0 になったことからは決めない。
2. 欠けで決まらなければ、次の順に **項目ごと** に比べ、最初に差が出た項目で決める (少ないほど良い):
   生成できなかったモジュール → 構文・静的検査のエラー → import スモークのエラー →
   引数の数のエラー → 使い方の実行の不合格 → 契約テストの失敗件数。どちらかが未検査の項目は飛ばす。
   エラーの総数では比べない — 構文エラーを直すと import エラーが見えてくるので件数は単調でない。
- 契約テストは、契約テストのファイル (``contract_key``) が同じときだけ比べる (欠けの判定にも入れない)。
- 同点は 0 (呼出側は先の版を残す)。``compare(a, b)`` と ``compare(b, a)`` は符号が逆になる。

作り直しの停滞 (同じ本文・往復・同じエラーの集合) の判定のロジックは :class:`RepairHistory` の 1 本
(履歴の状態は smoke と契約テストの経路ごとに持つ)。
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from backend.free.core.check_outcome import CheckOutcome

#: :func:`score` の項目の並び (比べる順)
SCORE_FIELDS: tuple[str, ...] = ("missing", "static", "smoke", "arity", "usage", "contract")


@dataclass(frozen=True)
class VerifiedState:
    """検査を通した 1 つの版 (配線後の code_map) と、その検査結果。

    ``smoke_errors`` は検査できたモジュールの import エラーの件数。``smoke_ran=False`` は import スモークが
    最後まで走らなかった (時間切れ・起動できない)、``smoke_unchecked`` は隔離が書込みを止めて検査できなかった
    もの (モジュール名・書込み先)。``usage`` は ``None`` (実行しなかった) か未検査の結果なら未検査、
    ``contract_failed`` の ``None`` は未検査。
    """

    code_map: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)
    missing: tuple[str, ...] = ()
    static_errors: int = 0
    smoke_errors: int = 0
    smoke_ran: bool = True
    smoke_unchecked: frozenset[str] = frozenset()
    arity_errors: int = 0
    usage: CheckOutcome | None = None
    contract_failed: int | None = None
    contract_key: str | None = None
    round: int = 0


def _usage_score(usage: CheckOutcome | None) -> int | None:
    return None if usage is None or usage.is_unchecked else int(usage.is_failure)


def score(state: VerifiedState) -> tuple[int | None, ...]:
    """:data:`SCORE_FIELDS` の順の点 (少ないほど良い、``None`` は未検査)。"""
    return (
        len(state.missing), state.static_errors, state.smoke_errors if state.smoke_ran else None,
        state.arity_errors, _usage_score(state.usage), state.contract_failed,
    )


def lost_checks(new: VerifiedState, old: VerifiedState) -> list[str]:
    """``old`` では検査できていたのに ``new`` では検査できなくなった項目 (:data:`SCORE_FIELDS` の順)。"""
    lost: list[str] = []
    if old.smoke_ran and (not new.smoke_ran or not new.smoke_unchecked <= old.smoke_unchecked):
        lost.append("smoke")
    if _usage_score(old.usage) is not None and _usage_score(new.usage) is None:
        lost.append("usage")
    if (
        new.contract_key == old.contract_key
        and old.contract_failed is not None and new.contract_failed is None
    ):
        lost.append("contract")
    return lost


def compare(new: VerifiedState, old: VerifiedState) -> tuple[int, str | None]:
    """``new`` を ``old`` と比べる。(+1 = 悪化 / -1 = 改善 / 0 = 同点・比べられない, 決めた項目)。"""
    new_lost, old_lost = lost_checks(new, old), lost_checks(old, new)
    if new_lost and not old_lost:
        return 1, new_lost[0]
    if old_lost and not new_lost:
        return -1, old_lost[0]
    contract_comparable = new.contract_key == old.contract_key
    for name, n, o in zip(SCORE_FIELDS, score(new), score(old)):
        if (name == "contract" and not contract_comparable) or n is None or o is None:
            continue
        if n != o:
            return (1 if n > o else -1), name
    return 0, None


def is_worse(new: VerifiedState, old: VerifiedState) -> bool:
    """``new`` が ``old`` より厳密に悪いか (同点・比べられないときは False)。"""
    return compare(new, old)[0] > 0


def is_better(new: VerifiedState, old: VerifiedState) -> bool:
    """``new`` が ``old`` より厳密に良いか (同点は False — 先の版を残す)。"""
    return compare(new, old)[0] < 0


#: エラー文の一時フォルダ・絶対パス (Windows のドライブ始まり / POSIX の 2 階層以上)。ファイル名だけを残す
_ABS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"(),]*|/(?:[^\s'\"(),/]+/)+[^\s'\"(),/]*")
#: エラー文の行番号 (``line 12`` / ``foo.py:12:3``)
_LINE_NO_RE = re.compile(r"\bline \d+|(?<=\.py):\d+(?::\d+)?")
#: 構文のエラー (行番号を残す — 次の構文エラーが見えたのは進捗で、停滞ではない)
_SYNTAX_ERROR_RE = re.compile(
    r"SyntaxError|IndentationError|TabError|invalid syntax|unexpected indent|unindent does not match"
    r"|expected an indented block|was never closed|unterminated"
)


def normalize_body(text: str) -> str:
    """本文の比較用の正規化 (行末の空白と、先頭・末尾の空行の差だけを無視する。内部の空行は保つ)。"""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def normalize_errors(errors: list[str]) -> frozenset[str]:
    """エラーの集合の比較用の正規化 (一時フォルダのパスと、構文以外のエラーの行番号を落とす)。"""

    def _one(error: str) -> str:
        error = _ABS_PATH_RE.sub(lambda m: re.split(r"[\\/]", m.group(0))[-1], error)
        if _SYNTAX_ERROR_RE.search(error):
            return error.strip()
        return _LINE_NO_RE.sub(lambda m: "line N" if m.group(0).startswith("line") else "", error).strip()

    return frozenset(_one(e) for e in errors if e.strip())


def _digest(text: str) -> str:
    return hashlib.sha256(normalize_body(text).encode("utf-8")).hexdigest()


@dataclass
class RepairHistory:
    """作り直しの停滞の判定 (f_10 §11.1-3)。判定のロジックはこの 1 本、履歴の状態は経路 (smoke / 契約) ごと。

    モジュールごとに、いまの版の本文 (``current``)・試した本文 (``tried``)・直前の回のエラーの集合を覚える。
    作り直しが同じ本文 (``"same"``)・前に試した本文 (``"repeat"``、A → B → A)・直前の回と同じエラーの集合
    (``"errors"``) を返したら、そのモジュールを外す (:meth:`drop`)。外したモジュールのエラーの集合が
    兄弟の作り直しで変わったら戻す (:meth:`observe_errors`)。停滞は悪化ではない — 悪化の取り消し
    (:func:`compare`) は呼出側でそのまま掛かる。
    """

    current: dict[str, set[str]] = field(default_factory=dict)
    tried: dict[str, set[str]] = field(default_factory=dict)
    last_errors: dict[str, frozenset[str]] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    #: 打ち切りが起きた回数 (外したモジュールを戻して再び外せば 2 回)
    stops: int = 0

    def remember(self, path: str, body: str | None) -> None:
        """``body`` を ``path`` のいまの版 (と試した本文) に加える (配線前と配線後の両方を渡せる)。"""
        if body is None:
            return
        d = _digest(body)
        self.current.setdefault(path, set()).add(d)
        self.tried.setdefault(path, set()).add(d)

    def body_verdict(self, path: str, body: str, *, previous: str | None = None) -> str | None:
        """作り直しの本文の停滞 (``"same"`` / ``"repeat"`` / ``None``)。停滞でなければいまの版にする。"""
        self.remember(path, previous)
        d = _digest(body)
        if d in self.current.get(path, set()):
            return "same"
        if d in self.tried.get(path, set()):
            return "repeat"
        self.current[path] = {d}
        self.tried.setdefault(path, set()).add(d)
        return None

    def errors_verdict(self, path: str, errors: list[str]) -> str | None:
        """作り直した版のエラーの集合が直前の回と同じなら ``"errors"``。空の集合 (直った) は停滞にしない。"""
        key = normalize_errors(errors)
        previous = self.last_errors.get(path)
        self.last_errors[path] = key
        return "errors" if key and key == previous else None

    def observe_errors(self, path: str, errors: list[str]) -> bool:
        """作り直していないモジュールのエラーの集合を覚える。外したモジュールの集合が変わったら戻す (True)。"""
        key = normalize_errors(errors)
        changed = path in self.last_errors and key != self.last_errors[path]
        self.last_errors[path] = key
        if changed and path in self.dropped:
            del self.dropped[path]
            return True
        return False

    def drop(self, path: str, reason: str) -> None:
        """``path`` をそれ以降の作り直しから外す (外している間の 2 回目は数えない)。"""
        if path not in self.dropped:
            self.dropped[path] = reason
            self.stops += 1


def compile_error_count(code_map: Mapping[str, str]) -> int:
    """構文として読めない Python ファイルの数。"""
    count = 0
    for path, content in code_map.items():
        if not path.endswith(".py"):
            continue
        try:
            ast.parse(content, filename=path)
        except (SyntaxError, ValueError):
            count += 1
    return count


__all__ = [
    "SCORE_FIELDS", "RepairHistory", "VerifiedState", "compare", "compile_error_count", "is_better", "is_worse",
    "lost_checks", "normalize_body", "normalize_errors", "score",
]
