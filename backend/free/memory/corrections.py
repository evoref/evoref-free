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

**訂正ノート自身も後続の訂正の対象になる** (2026-09-02 監査 S-A3)。
A ← B ← C の連鎖で B (訂正) を候補から外していると、B は永久に「現在値」
として提示され、C は A にしか付かない。連鎖は :func:`corrections_by_target`
が終端 (それ以上訂正されていない訂正) まで辿って解く。

**候補は user 発話に限り、順位は重なり語数が先** (2026-08-30 ライブ監査)。
チャットでは値の申告の直後にアシスタントが必ず復唱し、その復唱ノートは
(a) 対象と同じ keyword を持ち (b) 対象より新しく (c) ``assertion_slug`` を
持たない。「直前」だけで選ぶと訂正は毎回この復唱に結び付き、
``assertion_curator`` は継ぐ slug が無いまま別 slug を書き、
``attach_superseding_corrections`` は復唱の方に supersede 印を付ける。
実測: 「デプロイ先は AWS…」→ 訂正「AWS ではなく GCP…」で
``mem.world.assertion.deployment_region`` (AWS) と
``…deployment_target`` (GCP) が **両方 live のまま並び**、以後の全ターンへ
AWS が注入され続けた。復唱を除いても「直前」だけでは足りない —
間に挟まる別属性の申告 (「インスタンスタイプは t3.medium」) の方が新しい。
重なり語数を第一キー、新しさを同点時のキーにすると、訂正は
**話題語を最も多く共有する言明** に付く。
"""

from __future__ import annotations

from typing import Any

#: 訂正ノートと被訂正ノートを結ぶ最小 keyword 重なり数。
CORRECTION_LINK_MIN_OVERLAP = 1


def _created_at(note: Any) -> float:
    return float(getattr(note, "created_at", 0.0) or 0.0)


def _keywords(note: Any) -> set[str]:
    return set(getattr(note, "keywords", None) or ())


def _is_user_note(note: Any) -> bool:
    """ユーザー自身の発話ノートか。

    ``source`` を持たない古いノート / テスト用ダブルは user とみなす
    (訂正の対象になり得ないのはアシスタント発話だけなので、既定は通す)。
    """
    return (getattr(note, "source", None) or "user") != "assistant"


def has_correction_shape(note: Any) -> bool:
    """ノートが訂正の **候補** の形を持つか (純粋関数、検証の有無は問わない)。

    ``is_correction`` (応答パスの ``restates_a_value`` / 取り込みの
    ``restates_attribute_value`` が立てる) か、本文が訂正の形
    (``extractors.chat.has_correction_form``: 「ではなく」「正しくは」「変わりました」…)
    を持つか。検証 (Step 8.0) の対象選び・Step 8 の据え置き・Step 8.4 の持ち越し・
    注入の未検証の注記は **すべてこの 1 実装** を読む (不変則 #14(a))。以前は注入側
    だけが ``is_correction`` しか見ておらず、検証の対象になる形だけのノートに
    注記が付かなかった (2026-10-02 監査)。
    """
    if getattr(note, "is_correction", False):
        return True
    from backend.free.memory.extractors.chat import has_correction_form

    return has_correction_form(str(getattr(note, "content", "") or ""))


def is_correction_candidate(note: Any) -> bool:
    """未検証の訂正候補か (Step 8.0 の検証対象・Step 8 の据え置き・注入の注記の SSOT)。

    **ユーザー発話** で、本文が空でなく、まだ検証されておらず
    (``correction_verified_at`` が無い)、:func:`has_correction_shape` を満たすもの。
    据え置きだけ形を見て候補選びが ``is_correction`` だけを見ていたため、
    ``is_correction=False`` で形を持つノートが永久に検証も抽出もされなかった
    (2026-10-01、不変則 #14(a))。
    """
    if str(getattr(note, "source", "user") or "user") != "user":
        return False
    if getattr(note, "correction_verified_at", None) is not None:
        return False
    if not str(getattr(note, "content", "") or "").strip():
        return False
    return has_correction_shape(note)


def correction_target(correction: Any, notes: list) -> Any | None:
    """``correction`` が言い直している **元の言明ノート** を返す (純粋関数)。

    条件は「同一セッション」「ユーザー発話」「訂正より前」「keyword が
    :data:`CORRECTION_LINK_MIN_OVERLAP` 語以上重なる」。候補が複数あれば
    **重なり語数が最も多いもの**、同数なら **最も新しいもの** を採る。
    先行する訂正ノートも候補に含める (連鎖 A ← B ← C で B が対象になる)。

    アシスタントの復唱を候補から外す理由と、重なり語数を新しさより優先する
    理由はモジュール docstring を参照。
    """
    corr_kw = _keywords(correction)
    if not corr_kw:
        return None
    corr_at = _created_at(correction)
    corr_sess = getattr(correction, "session_id", None)
    best = None
    best_rank = (0, 0.0)
    for note in notes:
        if note is correction:
            continue
        if not _is_user_note(note):
            continue
        if getattr(note, "session_id", None) != corr_sess:
            continue
        note_at = _created_at(note)
        if note_at >= corr_at:
            continue
        overlap = len(corr_kw & _keywords(note))
        if overlap < CORRECTION_LINK_MIN_OVERLAP:
            continue
        rank = (overlap, note_at)
        if best is None or rank > best_rank:
            best, best_rank = note, rank
    return best


def corrections_by_target(notes: list) -> dict[str, Any]:
    """``被訂正 note_id -> その現在値を持つ訂正ノート`` を返す (純粋関数)。

    同じ対象を複数回訂正している場合は **最後の訂正** が現在値。訂正が
    さらに訂正されている連鎖 (A ← B ← C) では、A も B も **終端の C** に
    解決する — B は「訂正済み」の印が付く側であって現在値ではない。
    """
    direct: dict[str, Any] = {}
    for note in notes:
        if not getattr(note, "is_correction", False):
            continue
        target = correction_target(note, notes)
        if target is None:
            continue
        target_id = getattr(target, "id", None)
        if not target_id:
            continue
        prev = direct.get(target_id)
        if prev is None or _created_at(note) > _created_at(prev):
            direct[target_id] = note

    def _terminal(corr: Any) -> Any:
        seen: set[str] = set()
        while True:
            cid = getattr(corr, "id", None)
            nxt = direct.get(cid) if cid else None
            if nxt is None or cid in seen:
                return corr
            seen.add(cid)
            corr = nxt

    return {target_id: _terminal(corr) for target_id, corr in direct.items()}
