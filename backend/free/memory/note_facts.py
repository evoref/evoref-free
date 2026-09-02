"""STM ノート由来の ``SemanticFact`` を作る唯一の正規経路。

## なぜ要るか

SemMem の書込経路のうち **キュレーター系 (Step 8.4 / 8.5 / 8.6) は
``make_fact()`` を素で呼んでいた**。その結果:

- ``provenances=[]`` / ``session_ids=set()`` で **追跡情報がゼロ**。どのノート・
  どのセッション由来かが永久に分からない。
- ノートの ``private`` が **落ちる**。private ターンで踏んだ URL・実行した
  コマンド・そのときの質問文が ``private=False`` のファクトとして永続化され、
  ``MemoryInjector._classify_fact`` の private 除外にも ``ToolCallJudge`` の
  リコールにも掛からない (2026-09-01 監査 F2 で再現)。

この 2 つは同じ欠陥の別の症状で、根本原因は **ノート → ファクトの境界が
ノートの素性を捨てていること**。各キュレーターが ``if note.private`` を手書き
する形 (``_curator_common.public_notes``) は「忘れてはいけない」規約でしかなく、
実際 Step 8.4-8.6 の 3 つが揃って忘れた。

そこで **privacy をデータに載せる**。:func:`fact_from_note` を通せば、

1. provenance がノートから自動で埋まり (note_id / session_id / trace_id /
   mode / project_id / source)、
2. ``fact.private`` が **ノートから継承される**。

ガードを書き忘れても、ファクトは ``private=True`` で生まれ、読み出し側
(注入 / リコール) が既に落とす。「忘れてはいけない」が「忘れられない」に変わる。

## 使い分け

- **キュレーター / 単発のノート由来ファクト** → :func:`fact_from_note`
- **抽出器 (Step 8)** → ``extractors/base.py`` の ``build_fact()``。subject の
  正規化 / 発話時刻の継承 / セッション別キャップ等の抽出器固有の処理を伴うため
  独自実装だが、provenance と private の扱いは本モジュールと同じ意味論に
  揃えてある (:func:`privacy_of` が SSOT)。

新しく ``MemoryNote`` からファクトを作る経路を足すときはここを通すこと。
``backend/free/memory/tests/test_note_fact_provenance.py`` が AST 走査で
``make_fact(`` の直呼びを検出する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.free.memory.types import (
    FactType,
    MemoryMode,
    Provenance,
    SemanticFact,
    make_fact,
)

if TYPE_CHECKING:
    from backend.free.memory.stores.short_term import MemoryNote

__all__ = ["fact_from_note", "privacy_of", "provenance_of"]


def privacy_of(note: "MemoryNote | None") -> bool:
    """ノートの private 属性を返す (ファクトが継承すべき値)。

    ``None`` (ノート不明) は ``False`` — 由来が辿れないものを private 扱い
    すると、既存の公開ファクトが読み出しから消える方へ倒れる。private の
    伝播は「ノートがあると分かっている経路」で確実に効けばよい。
    """
    return bool(getattr(note, "private", False))


def provenance_of(
    note: "MemoryNote | None",
    *,
    captured_at: float,
    default_mode: MemoryMode = "chat",
    project_id: str | None = None,
) -> Provenance:
    """ノートから :class:`Provenance` を組み立てる。

    ``extractors/base.py::build_fact`` と同じ形。片方だけ増やすと、経路に
    よって追跡情報の粒度が変わる。
    """
    return Provenance(
        note_id=getattr(note, "id", None),
        session_id=getattr(note, "session_id", None) or None,
        trace_id=getattr(note, "trace_id", None),
        mode=getattr(note, "mode", None) or default_mode,
        project_id=getattr(note, "project_id", None) or project_id,
        source=getattr(note, "source", None),
        captured_at=captured_at,
    )


def fact_from_note(
    note: "MemoryNote | None",
    *,
    subject: str,
    predicate: str,
    object_: str,
    type: FactType,
    scope: str,
    now: float,
    mode_origin: MemoryMode | None = None,
    confidence: float = 0.5,
    project_id: str | None = None,
    **overrides: Any,
) -> SemanticFact:
    """ノート由来の ``SemanticFact`` を作る (provenance と private を継承)。

    Args:
        note: 由来の STM ノート。``None`` でも作れるが、その場合 provenance は
            空に近くなり private も継承されない (由来不明の経路は
            :func:`privacy_of` の説明どおり ``False``)。
        now: ``created_at`` / ``accessed_at`` / ``captured_at`` に使う epoch 秒。
        mode_origin: 省略時はノートの ``mode``、それも無ければ ``"chat"``。
        overrides: ``SemanticFact`` の任意フィールドを上書きする
            (``_extra`` / ``subject_aliases`` 等)。**``private`` を明示指定
            した場合はそれを尊重する** — 呼出側が「これは公開してよい」と
            判断できる場面 (例: ユーザーが明示 pin した) のための逃げ道。

    Returns:
        provenance 1 件と ``private`` を継承した ``SemanticFact``。
    """
    resolved_mode: MemoryMode = (
        mode_origin or getattr(note, "mode", None) or "chat"
    )
    fact = make_fact(
        subject=subject,
        predicate=predicate,
        object_=object_,
        type=type,
        scope=scope,
        mode_origin=resolved_mode,
        confidence=confidence,
        now=now,
    )
    fact.provenances = [
        provenance_of(
            note,
            captured_at=now,
            default_mode=resolved_mode,
            project_id=project_id,
        ),
    ]
    session_id = getattr(note, "session_id", None)
    if session_id:
        fact.session_ids = {session_id}
    trace_id = getattr(note, "trace_id", None)
    if trace_id:
        fact.trace_id = trace_id
    # **private はノートから継承する** — これが本モジュールの存在理由。
    fact.private = privacy_of(note)
    for key, value in overrides.items():
        setattr(fact, key, value)
    return fact
