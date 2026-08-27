"""直前に生成した長文成果物をセッション単位で保持する。

長文生成の出力は ``record_long_form_response`` が ``WorkingMemory`` へ積むが、
**次のターンには残らない**。実測 (2026-08-27 ライブ監査 T10):

    6696 文字の計画書を出力した次のターンで
    ``_trim_history: 5/5 turns kept, 812 estimated tokens (max=1612)``

履歴予算は 1612 トークンで、計画書は 4000 トークン超。入らないので落ちる。
その結果:

- 「いまの計画書は全部で何章になりましたか。」
  → 「提示された会話履歴には ... 情報が含まれていないため、現在の章数を
     特定することはできません。**計画書のテキストを共有いただければ**」
  (ユーザーが 1 ターン前に受け取ったばかりの本文を貼り直せ、と言っている)
- 「第2章だけを要約してください。」/「全体を300文字で要約してください。」も同文で拒否
- それでいて「``plan.md`` に保存してください。」には応じ、**元の本文ではなく
  別の 3867 文字の文書を新規生成** して書き込んだ (再生成した事実は非開示)

モデルの「履歴に含まれていない」という申告自体は正直で、欠陥は
**成果物が次ターンから参照できる形でどこにも残らない** ことにある。

本モジュールは ``_continuation`` と同じ形のセッションレジストリで成果物を
保持する。予算に収まらない長文は節単位で配分した抜粋へ縮退させる
(:func:`render_artifact_block`)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from backend.utils import utc_now_dt

if TYPE_CHECKING:
    from backend.app_state import AppState

__all__ = [
    "ARTIFACT_TTL",
    "LastArtifact",
    "forget_artifact",
    "outline_of",
    "peek_artifact",
    "references_artifact",
    "remember_artifact",
    "render_artifact_block",
    "requested_section",
    "split_sections",
]

#: 成果物の有効期限。これを過ぎた参照は「もう手元に無い」として扱う。
ARTIFACT_TTL = timedelta(hours=2)

#: セッション辞書の上限 (``_continuation`` と同じ単純な間引き)。
_MAX_ARTIFACT_SESSIONS = 32

#: 保持する本文の上限。これを超える成果物も **切り詰めて保持する** —
#: 全文が無くても構造は取れるので、「何章あるか」には答えられる。
MAX_ARTIFACT_CHARS = 60_000

#: Markdown 見出し (``#`` 〜 ``######``)。長文生成は ``## 1. 計画の概要と目的``
#: の形で章を切る。
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class LastArtifact:
    """直前ターンで生成した長文成果物。"""

    #: 本文 (``MAX_ARTIFACT_CHARS`` まで)。
    text: str
    #: 生成を指示したユーザー発話。「その計画書」が何を指すかの手掛かり。
    query: str
    #: 生成時のモード ("chat" / "create")。
    mode: str
    #: 生成時刻 (UTC)。TTL 判定に使う。
    at: datetime
    #: 本文が ``MAX_ARTIFACT_CHARS`` で切られたか。
    clipped: bool = False


def _registry(state: "AppState") -> dict[str, LastArtifact]:
    return state.last_artifacts


def remember_artifact(
    state: "AppState", session_id: str, *,
    text: str, query: str, mode: str,
) -> None:
    """このセッションの「直前の成果物」を差し替える。

    空文字 (キャンセル / 生成失敗) では武装しない — 参照する材料が無い。
    """
    body = (text or "").strip()
    if not body:
        return
    registry = _registry(state)
    if len(registry) >= _MAX_ARTIFACT_SESSIONS:
        oldest = min(registry, key=lambda k: registry[k].at)
        registry.pop(oldest, None)
    clipped = len(body) > MAX_ARTIFACT_CHARS
    registry[session_id] = LastArtifact(
        text=body[:MAX_ARTIFACT_CHARS],
        query=(query or "").strip(),
        mode=mode,
        at=utc_now_dt(),
        clipped=clipped,
    )


def peek_artifact(
    state: "AppState", session_id: str, *, now: datetime | None = None,
) -> LastArtifact | None:
    """有効期限内の成果物を返す (**消費しない**)。

    ``take_continuation`` と違って pop しないのは、同じ成果物に対して
    「何章?」「第2章を要約」「保存して」と **複数ターン続けて聞く** のが
    普通だから。1 回で消すと 2 問目から再び答えられなくなる。
    """
    artifact = _registry(state).get(session_id)
    if artifact is None:
        return None
    if (now or utc_now_dt()) - artifact.at > ARTIFACT_TTL:
        _registry(state).pop(session_id, None)
        return None
    return artifact


def forget_artifact(state: "AppState", session_id: str) -> None:
    """成果物を破棄する (セッション切替時など)。"""
    _registry(state).pop(session_id, None)


def outline_of(text: str) -> list[str]:
    """本文から見出しの一覧を返す (純粋関数)。"""
    return [
        f"{'#' * len(marks)} {title}"
        for marks, title in _HEADING_RE.findall(text or "")
    ]


def split_sections(text: str) -> list[tuple[str, str]]:
    """本文を ``(見出し, その節の本文)`` の列へ切る (純粋関数)。

    見出しが 1 つも無ければ ``[("", 全文)]``。空文字なら空リスト。
    """
    body = text or ""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("", stripped)] if stripped else []
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((m.group(0).strip(), body[start:end].strip()))
    return sections


#: 直前の成果物を指す **指示** の形。
#:
#: 語彙で成果物の種類 (計画書 / レポート / 文書 …) を数えない —
#: それは 2026-07 以降 4 回破れた属性語の列挙と同じ失敗の形になる。
#: ここで見るのは **指示詞** という閉じた文法クラスだけで、「何を指すか」は
#: 「直前のターンが長文成果物だった」という **観測事実** が決める。
_DEMONSTRATIVE_RE = re.compile(
    r"(?:その|それ|この|これ|these|this|that|いまの|今の|さっきの|先ほどの"
    r"|上記の|前の)",
)

#: 成果物そのものを対象にする操作。指示詞が無くても、直前が成果物なら
#: これらは成果物に掛かっているとみなす (「全体を要約して」「何章ある?」)。
_ARTIFACT_OPERATION_RE = re.compile(
    r"(?:全体|全部|ぜんぶ|何章|何節|何ページ|何文字|何行|章立て|目次|見出し)",
)

#: 「第2章」「2章」「セクション3」のような **節番号の指定**。
#: 章題の語彙 (概要 / ルート / 装備 …) は数えない — 構造の指定だけを見る。
_SECTION_REF_RE = re.compile(
    r"(?:第\s*)?([0-9０-９一二三四五六七八九十]+)\s*(?:章|節|section)",
    re.IGNORECASE,
)

#: 全角数字と漢数字 (1〜10) の読み替え。節番号の指定にしか使わない。
_FULLWIDTH_DIGITS = {c: str(i) for i, c in enumerate("０１２３４５６７８９")}
_KANJI_NUMERALS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def references_artifact(query: str) -> bool:
    """``query`` が直前の成果物を指しているか (純粋関数)。

    **呼出側は「直前ターンが長文成果物だった」ことを既に確認している** 前提。
    その上で、この発話が新しい話題ではなく手元の成果物に掛かっているかを見る。

    指示詞か、成果物そのものを対象にする操作語、または節番号の指定。
    成果物の **種類** を表す語は数えない — 語彙の列挙は必ず漏れる。
    """
    if not query:
        return False
    return bool(
        _DEMONSTRATIVE_RE.search(query)
        or _ARTIFACT_OPERATION_RE.search(query)
        or _SECTION_REF_RE.search(query),
    )


def requested_section(query: str) -> int | None:
    """``query`` が指している節番号 (1 始まり)。無ければ ``None``。"""
    m = _SECTION_REF_RE.search(query or "")
    if not m:
        return None
    raw = "".join(_FULLWIDTH_DIGITS.get(c, c) for c in m.group(1))
    if raw.isdigit():
        return int(raw) or None
    if len(raw) == 1 and raw in _KANJI_NUMERALS:
        return _KANJI_NUMERALS[raw]
    return None


def render_artifact_block(
    artifact: LastArtifact, *, budget_chars: int, query: str = "",
) -> str:
    """成果物を参照ブロックへ整形する。

    予算に収まれば **全文**。収まらなければ次の順で落とす:

    1. 質問が節を名指ししている (「第2章だけを要約」) → **その節の本文** を
       優先し、残りで全体の構成を添える
    2. それ以外 → 見出し + **各節の冒頭を均等配分** した抜粋

    見出しの一覧だけに落とすのは不足だった。実機検証 (2026-08-27) で
    「何章になりましたか」→「7章です」は答えられたが、「第2章だけを要約」
    「全体を300文字で要約」は **本文が無いので** 依然として
    「履歴に含まれていないため要約できません」を返した (6165 文字の本文に
    対し、渡っていたのは見出しだけの 149 文字)。章の数は構造から出せても、
    要約には本文が要る。

    **何も渡さない選択はしない** — 「本文を共有してください」はユーザーが
    1 ターン前に受け取った本文を貼り直せという要求で、縮退として最悪の形。
    """
    header = "[直前に作成した成果物]"
    body = artifact.text
    if len(body) <= budget_chars:
        return f"{header}\n{body}"

    sections = split_sections(body)
    if not sections:
        excerpt = body[: max(0, budget_chars - len(header) - 40)]
        return (
            f"{header} (全文は長いため冒頭のみ。本文 {len(body)} 文字)\n"
            f"{excerpt}"
        )

    titles = [title for title, _ in sections if title]
    lead = (
        f"{header} (全 {len(sections)} 節 / 本文 {len(body)} 文字。長いため抜粋)"
    )
    remaining = max(0, budget_chars - len(lead) - 8)

    wanted = requested_section(query)
    if wanted is not None and 1 <= wanted <= len(sections):
        title, content = sections[wanted - 1]
        # 名指しされた節は本文をできるだけ渡す (残予算の 8 割まで)。
        take = max(0, int(remaining * 0.8))
        picked = f"{title}\n{content[:take]}".strip()
        rest = max(0, remaining - len(picked))
        if titles and rest > 0:
            outline = "\n".join(titles)
            return f"{lead}\n{picked}\n\n(全体の構成)\n{outline[:rest]}"
        return f"{lead}\n{picked}"

    # 節ごとに均等配分。全体の要約・章立ての把握はこれで足りる。
    per_section = max(0, remaining // max(1, len(sections)))
    parts: list[str] = []
    for title, content in sections:
        chunk = content[:per_section].strip()
        part = f"{title}\n{chunk}".strip() if title else chunk
        if part:
            parts.append(part)
    return f"{lead}\n" + "\n\n".join(parts)
