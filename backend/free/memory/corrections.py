"""訂正ノートと被訂正ノートの紐付け (EvorefMem 内の SSOT)。

``is_correction`` が立ったノートは「直前に述べた値の言い直し」だが、**どの
言明を言い直したのか**はノート自身には書かれていない。訂正は属性名詞を落として
言うのが普通だからである (「訂正します、締切は10月15日に変更になりました。」に
「あさひプロジェクト」は出てこない)。

この紐付けは 2 箇所が必要とする:

- ``pipeline.search_pipeline.attach_superseding_corrections`` — 訂正前のノートが
  検索で採用されたとき、訂正も随伴させる。
- ``sleep.assertion_curator`` — 訂正に **対象と同じ subject slug** を継がせる。
  別 slug になると SemMem の競合検出が対にできず supersede できない。

同じ関係を 2 実装に分けると片方だけ直る (本リポジトリで繰り返し起きている
「食い違った複製」)。ここを唯一の出所とする。

**埋め込みは使えない**。訂正と対象の類似度は実測で真 0.575〜0.714 /
偽 0.429〜0.529 と重なり、閾値を置ける分離が無い
(``extractors.chat.resolve_inherited_attributes`` の説明を参照)。一方 keyword は
訂正側にも話題語が残る。実測 (2026-08-19、STM 83 件): ``is_correction`` は
2 件のみ (2.4%) で、keyword を持つ 1 件は同一セッションの先行 6 ノート中
**1 件だけ**と重なり (「締切」)、残り 5 件は重なりゼロだった。
"""

from __future__ import annotations

from typing import Any

#: 訂正ノートと被訂正ノートを結ぶ最小 keyword 重なり数。
CORRECTION_LINK_MIN_OVERLAP = 1


def _created_at(note: Any) -> float:
    return float(getattr(note, "created_at", 0.0) or 0.0)


def _keywords(note: Any) -> set[str]:
    return set(getattr(note, "keywords", None) or ())


def correction_target(correction: Any, notes: list) -> Any | None:
    """``correction`` が言い直している **元の言明ノート** を返す (純粋関数)。

    条件は「同一セッション」「訂正より前」「keyword が
    :data:`CORRECTION_LINK_MIN_OVERLAP` 語以上重なる」。候補が複数あれば
    **最も新しいもの** を採る (直前の言明を言い直すのが訂正の常態)。
    """
    corr_kw = _keywords(correction)
    if not corr_kw:
        return None
    corr_at = _created_at(correction)
    corr_sess = getattr(correction, "session_id", None)
    best = None
    for note in notes:
        if note is correction or getattr(note, "is_correction", False):
            continue
        if getattr(note, "session_id", None) != corr_sess:
            continue
        note_at = _created_at(note)
        if note_at >= corr_at:
            continue
        if len(corr_kw & _keywords(note)) < CORRECTION_LINK_MIN_OVERLAP:
            continue
        if best is None or note_at > _created_at(best):
            best = note
    return best


def corrections_by_target(notes: list) -> dict[str, Any]:
    """``被訂正 note_id -> その現在値を持つ訂正ノート`` を返す (純粋関数)。

    同じ対象を複数回訂正している場合は **最後の訂正** が現在値。
    """
    out: dict[str, Any] = {}
    for note in notes:
        if not getattr(note, "is_correction", False):
            continue
        target = correction_target(note, notes)
        if target is None:
            continue
        target_id = getattr(target, "id", None)
        if not target_id:
            continue
        prev = out.get(target_id)
        if prev is None or _created_at(note) > _created_at(prev):
            out[target_id] = note
    return out
