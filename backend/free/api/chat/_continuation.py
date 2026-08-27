"""切断応答の継続 (「続けて」) — セッション状態と意図判定。

``max_tokens`` 到達で切れた応答について、次ターンの「続けて」を **本当に
続きの生成** へ繋ぐ。従来は注記が「続けて と伝えてください」と案内する
だけで、それを受ける実装がどこにも無かった (2026-08-25 実インシデント:
「続けて」を 2 回入力して 2 回とも直前応答の末尾ブロックが逐語再掲された)。

設計の要点:

- **発火条件はモデルの解釈ではなく観測事実**。``finish_reason="length"``
  を観測したセッションだけが継続待ちになる。切断が無ければ「続けて」は
  従来どおり通常のチャットターンとして流れる (誤発火の上限がここで閉じる)。
- 語彙判定は **クエリ全体一致** に限る。「続けて」「続きをお願い」級の
  短文だけを拾い、「続けて説明すると〜」のような文中出現は拾わない。
- 継続待ちは 1 ターンで消費する。続きが再び切れたら、その時点で新しい
  末尾で再武装する (連続 3 回でも成立する)。
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from difflib import SequenceMatcher
from datetime import datetime, timedelta

from backend.app_state import AppState
from backend.free.core.intent_vocab import continuation_request
from backend.log_config import get_logger
from backend.utils import utc_now_dt

logger = get_logger("api.chat.continuation")

__all__ = [
    "TruncatedResponse",
    "arm_continuation",
    "disarm_continuation",
    "take_continuation",
    "is_continuation_request",
    "build_continuation_query",
    "strip_repeated_prefix",
    "resume_from_last_response",
]

#: 継続指示に再掲する直前応答の末尾文字数。履歴にも同じ本文が入っている
#: ため冗長だが、``_trim_history`` が長い応答を落とした場合の保険になる。
CONTINUATION_TAIL_CHARS = 400

#: 継続待ちの有効期限。これを過ぎた「続けて」は通常ターンとして扱う。
CONTINUATION_TTL = timedelta(hours=1)

#: セッション辞書の上限 (LRU ではなく古い順に間引く単純な上限)。
_MAX_PENDING_SESSIONS = 64


@dataclass(frozen=True)
class TruncatedResponse:
    """継続生成の材料 (直前応答の末尾)。"""

    #: 直前応答の末尾 (``CONTINUATION_TAIL_CHARS`` まで)。
    tail: str
    #: 直前ターンのモード ("chat" / "create")。継続も同じモードで走らせる。
    mode: str
    #: 武装時刻 (UTC)。TTL 判定に使う。
    at: datetime
    #: 直前応答が ``finish_reason="length"`` で **切れていた** か。
    #: False は「完結した応答に対して続きを求められた」ケースで、指示文が変わる
    #: (:func:`build_continuation_query`)。切れてもいないのに「上限で切れて
    #: います」と伝えると、モデルは存在しない切断箇所を繕おうとする。
    truncated: bool = True


def _pending(state: AppState) -> dict[str, TruncatedResponse]:
    """AppState 上の継続待ちレジストリ。"""
    return state.truncated_responses


def arm_continuation(
    state: AppState, session_id: str, *, response: str, mode: str,
) -> None:
    """このセッションを「続きを求められたら継続生成する」状態にする。

    ``response`` が空 (キャンセル等) なら武装しない — 続ける材料が無い。
    """
    tail = (response or "").rstrip()
    if not tail:
        return
    registry = _pending(state)
    if len(registry) >= _MAX_PENDING_SESSIONS:
        oldest = min(registry, key=lambda k: registry[k].at)
        registry.pop(oldest, None)
    registry[session_id] = TruncatedResponse(
        tail=tail[-CONTINUATION_TAIL_CHARS:], mode=mode, at=utc_now_dt(),
    )


def disarm_continuation(state: AppState, session_id: str) -> None:
    """継続待ちを解除する (切断せずに完結したターンの後始末)。"""
    _pending(state).pop(session_id, None)


def take_continuation(
    state: AppState, session_id: str, query: str, mode: str | None = None,
) -> TruncatedResponse | None:
    """``query`` が継続要求で、かつ継続待ちなら状態を取り出して解除する。

    取り出しは **消費** (pop)。継続生成そのものが再び切れた場合は、その層が
    新しい末尾で :func:`arm_continuation` し直す。

    ``mode`` を渡すと、切断ターンと同じモードのときだけ継続する。chat で
    切れた本文を create モードで書き継ぐのは別の作業なので、モードが変わって
    いたら通常ターンとして扱う (待ちは残さず捨てる — 続きを書く文脈自体が
    もう無い)。
    """
    registry = _pending(state)
    pending = registry.get(session_id)
    if pending is None:
        return None
    if utc_now_dt() - pending.at > CONTINUATION_TTL:
        registry.pop(session_id, None)
        return None
    if mode is not None and mode != pending.mode:
        registry.pop(session_id, None)
        return None
    if not is_continuation_request(query):
        return None
    registry.pop(session_id, None)
    return pending


def is_continuation_request(query: str) -> bool:
    """クエリ全体が「続きを書いて」だけを意味しているか。

    部分一致は取らない。「続けて説明してください」のような **別の依頼** を
    継続生成に化けさせないため (継続経路は元の依頼文を見ないので、誤発火
    すると質問そのものが失われる)。
    """
    return continuation_request(query)


#: 復唱とみなす最短の重なり。これ未満は偶然の一致がありうる
#: (「また、」「その後」のような接続句)。
MIN_REPEATED_OVERLAP_CHARS = 12

#: 1 文がどれだけ ``tail`` と重なったら復唱とみなすか (文の長さに対する割合)。
#: 実インシデントの復唱は 77% だった。0.6 は「言い換えを含む再掲」を拾い、
#: 「同じ語をいくつか使った続き」は拾わない位置。
REPEATED_SENTENCE_RATIO = 0.6

#: 先頭から見る文の数の上限。それ以上に長い再掲は「続き」ではなく別の問題。
MAX_REPEATED_SENTENCES = 3

_SENTENCE_END_RE = re.compile(r"(?<=[。．.!！?？])")


def _normalize_for_overlap(text: str) -> str:
    """重なり判定用に空白と改行を落とす (純粋関数)。"""
    return "".join((text or "").split())


def _longest_common_run(a: str, b: str) -> int:
    """``a`` と ``b`` の最長共通部分文字列の長さ (純粋関数)。"""
    if not a or not b:
        return 0
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b),
    )
    return match.size


def strip_repeated_prefix(continuation: str, tail: str) -> str:
    """継続出力の先頭にある **直前応答の末尾の再掲** を落とす (純粋関数)。

    接続点を示すために ``[直前の応答の末尾]`` を見せると、モデルはそこから
    書き始める — つまり再掲する。プロンプトには「既に書いた部分は繰り返さない」
    と明記してあるが、実測では効かない。

    実インシデント (2026-08-27 ライブ監査 T10-3)::

        2/8 の末尾: 「…継続的な交流や次なる企画立案の基盤として活用していく
                     ことで、旅の意義を長期的に持続させることができる。」
        3/8 の冒頭: 「長期的な交流や企画立案の基盤として活用していくことで、
                     旅の意義を長期的に持続させることができる。」

    **逐語ではなく前半が言い換えられている** ので、接尾辞と接頭辞の完全一致
    では捕まらない。文単位で見て、その文の :data:`REPEATED_SENTENCE_RATIO`
    以上が ``tail`` 内の連続部分として現れるなら再掲とみなして落とす。
    上の実データの重なりは 77% だった。
    """
    body = (continuation or "").lstrip()
    if not body or not tail:
        return continuation or ""
    flat_tail = _normalize_for_overlap(tail)
    if not flat_tail:
        return continuation or ""

    sentences = [x for x in _SENTENCE_END_RE.split(body) if x]
    dropped = 0
    for sentence in sentences[:MAX_REPEATED_SENTENCES]:
        flat = _normalize_for_overlap(sentence)
        if len(flat) < MIN_REPEATED_OVERLAP_CHARS:
            break
        run = _longest_common_run(flat, flat_tail)
        if run < MIN_REPEATED_OVERLAP_CHARS:
            break
        if run / len(flat) < REPEATED_SENTENCE_RATIO:
            break
        dropped += len(sentence)
    if dropped <= 0:
        return continuation or ""
    logger.info(
        "Continuation: dropped %d chars restated from the previous response "
        "tail", dropped,
    )
    return body[dropped:].lstrip()


def build_continuation_query(pending: TruncatedResponse) -> str:
    """継続生成ターンの user 本文を組み立てる。

    ベースモデルは「続けて」だけでは直前応答を丸ごと再掲する (実測)。
    **何をしないか** を明示し、接続点となる末尾を再掲して繋ぎ先を固定する。
    否定形だけの指示は退行するため、肯定形の指示 (「本文の続きだけを書く」)
    を先に置き、禁止事項を後置する。

    ``pending.truncated`` が False (完結した応答への「続けて」) のときは
    切断を前提にした文言を出さない。切れていないのに「上限に達して途中で
    終わっています」と伝えると、モデルは存在しない切断箇所を繕おうとする。
    """
    if pending.truncated:
        head = (
            "直前の応答は出力トークン上限に達して文の途中で終わっています。"
            "その続きだけを、下の末尾に直接つながる形で書いてください。\n"
            "- 既に書いた部分は繰り返さない (見出しの再掲・要約の再掲も含む)\n"
            "- 前置き・お詫び・「続きです」等のメタ発言を書かない\n"
            "- 末尾の文が途中で切れている場合は、その文の途中から続ける\n"
        )
    else:
        head = (
            "直前の応答の続きを書いてください。同じ話題を、下の末尾から先へ"
            "進める形で掘り下げます。\n"
            "- 既に書いた部分は繰り返さない (見出しの再掲・要約の再掲も含む)\n"
            "- 前置き・お詫び・「続きです」等のメタ発言を書かない\n"
            "- 書くことが尽きている場合だけ、その旨を一文で述べる\n"
        )
    return (
        head
        + f"\n[直前の応答の末尾]\n{pending.tail}\n\n"
        "[ここから続きを書く]"
    )


def resume_from_last_response(
    history: list[dict], query: str, mode: str,
) -> "TruncatedResponse | None":
    """切断待ちが無いときの「続けて」を、直前応答の続きへ繋ぐ材料を返す。

    継続待ちが武装するのは ``finish_reason="length"`` を観測したときだけで、
    **正常終了した応答の直後の「続けて」** はここへ降りてくる。従来はそのまま
    通常ターンとして流れ、``short_query`` → reactive_light でモデルが直前の
    user 発話を逐語復唱した (2026-08-25 ライブ監査 T6-5)。層分類を
    deliberative へ上げただけでは足りない: 「続けて」は検索意図を持たない
    3 文字なので、SemMem ブロックが噛み合わず「あなたについて、現在確認できる
    情報はありません。」という無関係な応答になった (同日の修正検証で実測)。

    直前の assistant 応答の末尾を材料として渡し、切断ケースと同じ経路
    (SemMem / RAG / ツール判定なし) で続きだけを書かせる。

    Returns:
        継続材料。継続要求でない / 直前の assistant 応答が無い場合は ``None``。
    """
    if not continuation_request(query):
        return None
    for message in reversed(history or ()):
        if message.get("role") != "assistant":
            continue
        tail = (message.get("content") or "").rstrip()
        if not tail:
            return None
        return TruncatedResponse(
            tail=tail[-CONTINUATION_TAIL_CHARS:],
            mode=mode,
            at=utc_now_dt(),
            truncated=False,
        )
    return None
