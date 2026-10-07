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


def span_correction_target(correction: Any, notes: list) -> Any | None:
    """**検証済みの訂正** の旧値 span を逐語で含む前のユーザーノート (純粋関数)。

    訂正は話題語を落として言うので keyword が重ならないことがある (2026-10-07
    audit_replay R26: 訂正「2泊3日ではなく1泊2日に変更になりました」の keyword
    [変更, 行程] は旧ノート「…京都へ2泊3日の旅行…」の [京都, 旅行, …] と 0 語)。
    検証器の ``wrong_claim`` は門 (``core.correction_verdict``) を通った逐語 span
    なので、それを含む同じセッションの前のユーザーノートが **ちょうど 1 件** の
    ときだけ宛先にする。複数なら ``None`` — 「18日」は試験日にも給料日にも在る
    (2026-09-27 H2)。字句の候補だけの訂正 (未検証) は対象外 (不変則 #12)。
    """
    from backend.free.core.correction_verdict import norm_span, strip_copula
    from backend.free.memory.extractors.base import note_is_verified_correction

    if not note_is_verified_correction(correction):
        return None
    needle = norm_span(
        strip_copula(str(getattr(correction, "correction_wrong_claim", "") or "")),
    )
    if not needle:
        return None
    corr_at = _created_at(correction)
    corr_sess = getattr(correction, "session_id", None)
    found = [
        note for note in notes
        if note is not correction
        and _is_user_note(note)
        and getattr(note, "session_id", None) == corr_sess
        and _created_at(note) < corr_at
        and needle in norm_span(str(getattr(note, "content", "") or ""))
    ]
    return found[0] if len(found) == 1 else None


def _contains_value(content: str, needle: str) -> bool:
    """``needle`` (``norm_span`` 済み) が ``content`` に数字の境界を守って在るか。

    値が数字で始まる / 終わるなら、出現の前 / 後ろが数字の箇所は数えない
    (「3歳」が「13歳」、「8日」が「18日」に当たらない)。
    """
    from backend.free.core.correction_verdict import norm_span

    hay = norm_span(content)
    if not needle:
        return False
    start = hay.find(needle)
    while start >= 0:
        end = start + len(needle)
        head_ok = not (needle[0].isdigit() and start > 0 and hay[start - 1].isdigit())
        tail_ok = not (needle[-1].isdigit() and end < len(hay) and hay[end].isdigit())
        if head_ok and tail_ok:
            return True
        start = hay.find(needle, start + 1)
    return False


def _structural_old_values(correction: Any) -> list[tuple[str, tuple[str, ...]]]:
    """訂正発話の平叙文の「X ではなく Y」から ``(旧値 X, X より前の話題語)`` を取る。

    分解は :func:`~backend.free.core.correction_target.contrast_pairs`、話題語は
    :func:`~backend.free.memory.pipeline.injector.correction_form_topics` の 1 実装
    (不変則 #14a)。引用の内側は落とし、問い・依頼の文 (「…にしたら費用は？」) は見ない。
    """
    from backend.free.core.correction_target import contrast_pairs, split_sentences
    from backend.free.core.correction_verdict import (
        mask_quoted_speech,
        norm_span,
        strip_copula,
    )
    from backend.free.core.intent_vocab import is_plain_statement
    from backend.free.memory.pipeline.injector import correction_form_topics

    out: list[tuple[str, tuple[str, ...]]] = []
    masked = mask_quoted_speech(str(getattr(correction, "content", "") or ""))
    for sentence in split_sentences(masked):
        if not is_plain_statement(sentence):
            continue
        form = correction_form_topics(sentence)
        topics = form[1] if form is not None else ()
        for old, _new in contrast_pairs(sentence):
            old = strip_copula(old)
            at = sentence.find(old)
            before = tuple(t for t in topics if 0 <= sentence.find(t) < at)
            if norm_span(old):
                out.append((old, before))
    return out


def structural_correction_target(correction: Any, notes: list) -> Any | None:
    """**前提の変更** と判定された訂正候補の宛先を、発話の構造だけで探す (純粋関数・読むだけ)。

    検証器が ``premise_change`` と答えると ``wrong_claim`` が空で
    :func:`span_correction_target` が解けず、話題語を落とした訂正
    (「2泊3日ではなく1泊2日に変更になりました」) は keyword でも結べない
    (2026-10-07 audit_replay R26 #2)。書き込み側 (J-03、
    ``extractors.base.value_update_spans``) は同じ発話を本人の値の更新として
    既に扱っているので、**注入の随伴** (:func:`correction_links_by_target`) に
    限って同じ旧値で宛先を探す。SemMem の supersede や ``assertion_curator`` の
    slug 継承には使わない (不変則 #12 / #13)。

    次のすべてを満たすときだけ宛先を返す (それ以外は ``None`` = 棄権):

    - 検証済み (``correction_verified_at`` が在る — 検証前の候補は注入側の未検証の
      注記が受け持つ) で、本人の値更新として消費してよい (``note_may_update_own_value``) 平叙文で、
      仮定・時間の対比・伝聞の標識 (``marks_not_own_restatement``) が無い
    - 旧値 X を (数字の境界を守って) 含む、同じセッション・ユーザー発話・
      訂正より前のノートが **ちょうど 1 件**
    - X より前の話題語 (「給料日」「次男」「去年」「友人」) が宛先にすべて在る
    - 訂正より後のユーザーノートが X を含まない (A→B→A の戻し)。ただしその
      ノート自身がこの訂正を構造で言い直している連鎖 (A ← B ← C) は除く
    """
    from backend.free.core.correction_target import old_value_core
    from backend.free.core.correction_verdict import marks_not_own_restatement, norm_span
    from backend.free.memory.extractors.base import note_may_update_own_value

    # 検証前の字句の候補は対象外 — 注入側の未検証の注記が受け持つ (不変則 #12)。
    if getattr(correction, "correction_verified_at", None) is None:
        return None
    if not _is_user_note(correction) or not note_may_update_own_value(correction):
        return None
    if marks_not_own_restatement(str(getattr(correction, "content", "") or "")):
        return None
    corr_at = _created_at(correction)
    corr_sess = getattr(correction, "session_id", None)
    same_session = [
        note for note in notes
        if note is not correction
        and _is_user_note(note)
        and getattr(note, "session_id", None) == corr_sess
    ]
    earlier = [n for n in same_session if _created_at(n) < corr_at]
    later = [n for n in same_session if _created_at(n) > corr_at]
    for old, topics in _structural_old_values(correction):
        for value in dict.fromkeys((old, old_value_core(old))):
            needle = norm_span(value)
            found = [
                n for n in earlier
                if _contains_value(str(getattr(n, "content", "") or ""), needle)
            ]
            if not found:
                continue
            found = [
                n for n in found
                if all(
                    norm_span(t) in norm_span(str(getattr(n, "content", "") or ""))
                    for t in topics
                )
            ]
            if len(found) != 1:
                return None
            for n in later:
                if not _contains_value(str(getattr(n, "content", "") or ""), needle):
                    continue
                if structural_correction_target(n, notes) is not correction:
                    return None
            return found[0]
    return None


#: 随伴の種別。``correction`` は検証済みの訂正 / 従来の経路、``update`` は
#: :func:`structural_correction_target` だけで結んだ未検証の値の更新 (注記の文言を分ける)。
LINK_CORRECTION = "correction"
LINK_UPDATE = "update"


def correction_links_by_target(notes: list) -> dict[str, tuple[Any, str]]:
    """``被訂正 note_id -> (現在値を持つ訂正ノート, 随伴の種別)`` を返す (純粋関数)。

    同じ対象を複数回訂正している場合は **最後の訂正** が現在値。訂正が
    さらに訂正されている連鎖 (A ← B ← C) では、A も B も **終端の C** に
    解決する — B は「訂正済み」の印が付く側であって現在値ではない。

    宛先は検証済みの訂正なら旧値の span (:func:`span_correction_target`) を先に、
    解けなければ keyword の重なり (:func:`correction_target`)、それでも解けなければ
    発話の構造 (:func:`structural_correction_target`) で探す。種別は対象を **直接**
    言い直した訂正の結び方で決める (構造だけで結び、検証済みの訂正でなければ
    :data:`LINK_UPDATE`)。読み手は注入の随伴
    (``search_pipeline.attach_superseding_corrections``) だけ。
    """
    from backend.free.memory.extractors.base import note_is_verified_correction

    direct: dict[str, tuple[Any, str]] = {}
    for note in notes:
        if not getattr(note, "is_correction", False):
            continue
        kind = LINK_CORRECTION
        target = span_correction_target(note, notes) or correction_target(note, notes)
        if target is None:
            target = structural_correction_target(note, notes)
            if target is not None and not note_is_verified_correction(note):
                kind = LINK_UPDATE
        if target is None:
            continue
        target_id = getattr(target, "id", None)
        if not target_id:
            continue
        prev = direct.get(target_id)
        if prev is None or _created_at(note) > _created_at(prev[0]):
            direct[target_id] = (note, kind)

    def _terminal(corr: Any) -> Any:
        seen: set[str] = set()
        while True:
            cid = getattr(corr, "id", None)
            nxt = direct.get(cid) if cid else None
            if nxt is None or cid in seen:
                return corr
            seen.add(cid)
            corr = nxt[0]

    return {
        target_id: (_terminal(corr), kind)
        for target_id, (corr, kind) in direct.items()
    }


def corrections_by_target(notes: list) -> dict[str, Any]:
    """``被訂正 note_id -> その現在値を持つ訂正ノート`` (:func:`correction_links_by_target` の値だけ)。"""
    return {
        target_id: corr
        for target_id, (corr, _kind) in correction_links_by_target(notes).items()
    }
