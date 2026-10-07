"""会話全体の整合性検査 — ターンをまたぐ矛盾の決定論検出 (純関数)。

単一の応答だけを見る検査 (``core.response_arithmetic`` / ``core.text_quality``) では
原理的に取れない欠陥がある (2026-09-23 の集計: 監査の欠陥 10 件中、自動検出は 1〜3 件。
残りは訂正の後の旧い値の再使用・前のターンの値を使った再計算・他セッションの値の混入)。
本モジュールはターンの列を受け取り、次の 3 種だけを検出する。

- ``stale_old_value``: 「X ではなく Y」と値を置き換えた (訂正 / 計画の変更) **後の** 応答が、
  同じ属性について旧値 X をまた述べている。同一セッション (``scope=same_session``) と、
  後のセッション (``scope=cross_session``、記憶経由で旧値が戻る形) を分けて記録する。
- ``stale_derived_value``: 数値の前提を置き換えたターンの応答が、置き換える前の前提で
  計算した量 (同じラベル・同じ単位・同じ値) をそのまま述べている (再計算されていない)。
- ``quantity_drift``: 同じラベル・同じ単位の量を、訂正も新しい数の提示も無いまま、
  後のターンで違う値で述べている (例: 空き容量 138GB → 553GB)。

方針 (不変則 #12 / #14 / #15):

- 訂正の分解は ``core.correction_target.contrast_pairs`` (対比の SSOT)、引用の除外・仮定 /
  時間の対比 / 伝聞の標識・同値・否定は ``core.correction_verdict`` の門、数の読み取りは
  ``core.response_arithmetic.iter_ja_numbers`` をそのまま使う。新しい語彙リストは持たない。
- 訂正は **検証済みの形だけ** を使う: 平叙の文の「X ではなく Y」で、X が同じセッションの
  前のターンに逐語で在り、X と Y が同値でなく、訂正への応答が Y を退けていないもの
  (``correction_verdict`` の逐語・同値・否定の門の考え方。LLM の検証器は呼ばない)。
- 迷ったら検出しない (適合率優先)。新値 Y も同じ応答に在る・旧値が引用 / 仮定 / 否定の文に
  在る・問い自身が旧値を口にした・ラベルが 1 応答の中で複数の値を持つ、はすべて見送る。
- 例外を外へ出さない。壊れた入力は空として扱い、1 ターンの失敗は警告ログだけで続ける。
- 計算量は総文字数に線形 (1 本文の上限 :data:`MAX_TEXT_CHARS`、追跡する訂正の上限
  :data:`MAX_ACTIVE_CORRECTIONS`、文の長さの上限 :data:`MAX_SENTENCE_CHARS`)。

配線 (睡眠時の段への組み込み) はまだしていない。測定は
``scripts/bench/consistency_eval/eval_on_replay.py``。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from backend.free.core.correction_target import (
    _CONTRAST_PARTICLES,
    DEFAULT_LOOKBACK,
    KANJI_RUN_RE,
    KATAKANA_RUN_RE,
    contrast_pairs,
    old_value_core,
    split_sentences,
)
from backend.free.core.correction_verdict import (
    answer_disputes_value,
    claims_equivalent,
    marks_not_own_restatement,
    mask_quoted_speech,
    norm_span,
    strip_copula,
)
from backend.free.core.intent_vocab import is_plain_statement
from backend.free.core.response_arithmetic import (
    _APPROX_RE,
    _UNIT_RE,
    iter_ja_numbers,
)
from backend.free.core.script_ranges import KANJI, KANJI_MARKS, KATAKANA_WORD
from backend.log_config import get_logger

logger = get_logger("learning.conversation_consistency")

FindingKind = Literal["stale_old_value", "stale_derived_value", "quantity_drift"]

#: 1 本文 (問い / 応答) で読む上限。これより後ろは見ない (線形の上限を固定する)。
MAX_TEXT_CHARS = 20000
#: 訂正の検出に使う文の長さの上限。訂正の文は短い。対比の正規表現は区切りの無い
#: 長い文で探索が伸びるので、長い文は最初から渡さない。
MAX_SENTENCE_CHARS = 300
#: 同時に追跡する訂正の上限 (古いものから捨てる)。
MAX_ACTIVE_CORRECTIONS = 32
#: 証拠として残す本文の抜粋の長さ。
EVIDENCE_CHARS = 160
#: ラベル (量の名前) の最大長と、ラベルに要る漢字・カタカナの最少文字数。
_LABEL_MAX = 16
_LABEL_MIN_CONTENT = 2
#: ラベルと数のあいだに許す区切り。助詞は主題・主格の 2 つだけ (「睡眠時間を 2 時過ぎに」の
#: を・「100 万円で」の で は量の名前と値の関係ではない)。
_LABEL_SEPARATORS = frozenset(":：=＝")
_LABEL_PARTICLES = frozenset("はが")
#: 量の揺れ (quantity_drift) で同じ値とみなす相対の幅 (丸め・概数の言い換え)。
_DRIFT_REL_TOLERANCE = 0.01


def _in_ranges(ch: str, ranges: str) -> bool:
    """``ch`` が ``script_ranges`` 形式の文字範囲 (``a-b`` の並び) に入るか。"""
    i = 0
    while i < len(ranges):
        if i + 2 < len(ranges) and ranges[i + 1] == "-":
            if ranges[i] <= ch <= ranges[i + 2]:
                return True
            i += 3
        else:
            if ch == ranges[i]:
                return True
            i += 1
    return False


def _is_content_char(ch: str) -> bool:
    """漢字・カタカナ (長音を含む) か。"""
    return _in_ranges(ch, KANJI + KANJI_MARKS) or _in_ranges(ch, KATAKANA_WORD)


def _is_label_char(ch: str) -> bool:
    """ラベルを構成してよい文字 (漢字・カタカナ・英数字・連体の「の」)。"""
    return _is_content_char(ch) or (ch.isascii() and ch.isalnum()) or ch == "の"


@dataclass(frozen=True)
class Finding:
    """ターンをまたぐ矛盾 1 件。``turn_index`` は入力の列の位置 (0 起点)。"""

    kind: FindingKind
    turn_index: int
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Turn:
    index: int
    turn_id: str
    session: str
    query: str
    response: str


@dataclass
class _Correction:
    turn_index: int
    session: str
    old: str
    new: str
    attribute: str
    context_terms: frozenset[str]
    sentence: str
    old_numbers: tuple[float, ...]
    new_numbers: tuple[float, ...]
    active: bool = True


@dataclass(frozen=True)
class _Quantity:
    label: str
    unit: str
    value: float
    scale: float
    approx: bool
    start: int
    sentence: str


# ---------------------------------------------------------------------------
# 入力の正規化
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return value[:MAX_TEXT_CHARS] if isinstance(value, str) else ""


def _normalize_turns(turns: Any) -> list[_Turn]:
    if not isinstance(turns, Iterable) or isinstance(turns, (str, bytes, Mapping)):
        return []
    out: list[_Turn] = []
    for i, raw in enumerate(turns):
        if not isinstance(raw, Mapping):
            raw = {}
        turn_id = raw.get("turn_id")
        session = raw.get("session_id")
        out.append(_Turn(
            index=i,
            turn_id=str(turn_id) if turn_id is not None else str(i),
            session=str(session) if session is not None else "",
            query=_text(raw.get("query")),
            response=_text(raw.get("response")),
        ))
    return out


def _snippet(text: str) -> str:
    return " ".join(text.split())[:EVIDENCE_CHARS]


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


# ---------------------------------------------------------------------------
# 訂正の抽出 (検証済みの形だけ)
# ---------------------------------------------------------------------------


def _attribute_before(sentence: str, old: str) -> str:
    """対比の旧値の直前の主題 (「コタロウは3歳ではなく」の コタロウ)。無ければ空。"""
    pos = sentence.find(old)
    if pos <= 0:
        return ""
    head = sentence[:pos].rstrip()
    if head and head[-1] in _CONTRAST_PARTICLES:
        head = head[:-1]
    j = len(head)
    while j > 0 and _is_content_char(head[j - 1]):
        j -= 1
    return head[j:]


def _content_terms(text: str) -> set[str]:
    return {m.group(0) for m in KANJI_RUN_RE.finditer(text)} | {
        m.group(0) for m in KATAKANA_RUN_RE.finditer(text)
    }


def _numbers(text: str) -> tuple[float, ...]:
    return tuple(n.value for n in iter_ja_numbers(_nfkc(text)))


def _first_sentence_with(texts: Iterable[str], needle: str) -> str:
    """``needle`` (正規化済み) を最初に含む文 (引用の内側は除く)。"""
    for text in texts:
        for sentence in split_sentences(mask_quoted_speech(text)):
            if needle in norm_span(sentence):
                return sentence
    return ""


def _resolve_old(old: str, prior_texts: list[str]) -> str:
    """旧値が前のターンに逐語で在れば、その形を返す (区間 → 先頭のかなを外した形)。"""
    joined = norm_span("\n".join(prior_texts))
    for candidate in (old, old_value_core(old)):
        candidate = strip_copula(candidate)
        if candidate and norm_span(candidate) in joined:
            return candidate
    return ""


def _extract_corrections(turn: _Turn, prior_texts: list[str]) -> list[_Correction]:
    """このターンの問いに在る検証済みの「X ではなく Y」をすべて返す。"""
    if not turn.query or marks_not_own_restatement(turn.query):
        return []
    found: list[_Correction] = []
    for sentence in split_sentences(mask_quoted_speech(turn.query)):
        if len(sentence) > MAX_SENTENCE_CHARS or not is_plain_statement(sentence):
            continue
        for raw_old, raw_new in contrast_pairs(sentence):
            new = strip_copula(raw_new)
            old = _resolve_old(raw_old, prior_texts)
            if not old or not new or claims_equivalent(old, new):
                continue
            if answer_disputes_value(turn.response, new):
                continue
            attribute = _attribute_before(sentence, old)
            origin = _first_sentence_with(prior_texts, norm_span(old))
            value_text = norm_span(old) + norm_span(new)
            terms = {
                t for t in _content_terms(origin) | _content_terms(attribute)
                if norm_span(t) not in value_text
            }
            found.append(_Correction(
                turn_index=turn.index, session=turn.session, old=old, new=new,
                attribute=attribute, context_terms=frozenset(terms), sentence=sentence,
                old_numbers=_numbers(old), new_numbers=_numbers(new),
            ))
    return found


# ---------------------------------------------------------------------------
# stale_old_value
# ---------------------------------------------------------------------------


def _bounded_find(haystack: str, needle: str) -> list[int]:
    """``needle`` の出現位置。数の途中 (「3歳」に対する「13歳」) は除く。"""
    hits: list[int] = []
    start = 0
    while needle and (pos := haystack.find(needle, start)) >= 0:
        start = pos + 1
        before = haystack[pos - 1] if pos > 0 else ""
        after_pos = pos + len(needle)
        after = haystack[after_pos] if after_pos < len(haystack) else ""
        if needle[0].isdigit() and before and (before.isdigit() or before in ".,"):
            continue
        if needle[-1].isdigit() and after and (after.isdigit() or after in ".,"):
            continue
        hits.append(pos)
    return hits


def _attribute_in(attribute: str, scope: str) -> bool:
    """属性名が範囲に在るか。2 文字以上の漢字・カタカナ名は 2 文字の部分でもよい。"""
    if not attribute:
        return False
    if attribute in scope:
        return True
    if len(attribute) < 3:
        return False
    return any(attribute[i:i + 2] in scope for i in range(len(attribute) - 1))


def _label_named_in(label: str, text: str) -> bool:
    """ラベルの漢字・カタカナの並び (2 文字以上) のどれかが ``text`` に在るか。"""
    runs = _content_terms(label) | {
        label[i:i + 2] for i in range(len(label) - 1)
        if _is_content_char(label[i]) and _is_content_char(label[i + 1])
    }
    return any(r in text for r in runs)


def _same_attribute(correction: _Correction, scope: str) -> bool:
    if _attribute_in(correction.attribute, scope):
        return True
    return any(term in scope for term in correction.context_terms)


def _stale_old_value_hits(correction: _Correction, turn: _Turn) -> list[dict[str, Any]]:
    response = turn.response
    if not response:
        return []
    old_n = norm_span(correction.old)
    if old_n in norm_span(turn.query):
        return []  # 問い自身が旧値を口にした
    if norm_span(correction.new) in norm_span(response):
        return []  # 新値も述べている (経緯・対比の説明)
    hits: list[dict[str, Any]] = []
    for sentence in split_sentences(mask_quoted_speech(response)):
        compact = "".join(_nfkc(sentence).split())
        positions = _bounded_find(compact, "".join(_nfkc(correction.old).split()))
        if not positions:
            continue
        if marks_not_own_restatement(sentence) or answer_disputes_value(sentence, correction.old):
            continue
        if not _same_attribute(correction, sentence + "\n" + turn.query):
            continue
        hits.append({"span": _snippet(sentence)})
        break
    return hits


# ---------------------------------------------------------------------------
# 量 (ラベル + 数 + 単位) の抽出
# ---------------------------------------------------------------------------


def _label_before(text: str, start: int) -> tuple[str, bool]:
    """数の直前のラベルと「約」の有無。ラベルが取れなければ空。"""
    i = start
    while i > 0 and text[i - 1].isspace():
        i -= 1
    head = text[max(0, i - 8):i]
    approx = False
    for m in _APPROX_RE.finditer(head):
        if head[m.end():].strip() == "":
            approx = True
            i -= len(head) - m.start()
            break
    while i > 0 and text[i - 1].isspace():
        i -= 1
    if i > 0 and (text[i - 1] in _LABEL_PARTICLES or text[i - 1] in _LABEL_SEPARATORS):
        i -= 1
    while i > 0 and text[i - 1].isspace():
        i -= 1
    j = i
    while j > 0 and i - j < _LABEL_MAX and _is_label_char(text[j - 1]):
        j -= 1
    while j < i and text[j] == "の":
        j += 1
    label = text[j:i]
    k = j
    while k > 0 and text[k - 1].isspace():
        k -= 1
    # 前の数の単位の続き (「100 万円を 10 年」の 万円、「2027年1月19日」の 年1月) は名前でない
    if k > 0 and text[k - 1].isdigit():
        return "", approx
    tail = label
    for idx in range(len(label) - 1, -1, -1):
        if label[idx].isdigit():
            tail = label[idx + 1:]
            break
    if len(tail) < len(label) and len(tail) <= 1:
        return "", approx  # 数 + 1 文字で終わる (日付・単位つきの数の並び)
    if sum(1 for ch in tail if _is_content_char(ch)) < _LABEL_MIN_CONTENT:
        return "", approx
    return label, approx


def _quantities(response: str) -> list[_Quantity]:
    out: list[_Quantity] = []
    for sentence in split_sentences(mask_quoted_speech(response)):
        text = _nfkc(sentence)
        for num in iter_ja_numbers(text):
            unit_m = _UNIT_RE.match(text, num.end)
            if unit_m is None:
                continue
            label, approx = _label_before(text, num.start)
            if not label:
                continue
            out.append(_Quantity(
                label=label, unit=_unit_of(unit_m.group("unit")),
                value=num.value * (-1.0 if num.negative else 1.0), scale=num.scale,
                approx=approx, start=num.start, sentence=sentence,
            ))
    return out


def _unit_of(unit: str) -> str:
    """単位の正規形。英字は英字の並び (GB / kg)、漢字・カタカナはその並び (円 / 時間 /
    時過)、それ以外は先頭の 1 文字 (%)。「時間」と「時過ぎ」を同じ単位にしない。"""
    if unit[:1].isascii() and unit[:1].isalpha():
        run = ""
        for ch in unit:
            if not (ch.isascii() and ch.isalpha()):
                break
            run += ch
        return run
    if _is_content_char(unit[:1]):
        run = ""
        for ch in unit:
            if not _is_content_char(ch):
                break
            run += ch
        return run
    return unit[:1]


def _same_value_strict(a: _Quantity, b: _Quantity) -> bool:
    """表記の桁の丸めの範囲で同じ値か (「約9万円」と「91,855円」は同じ)。"""
    return abs(a.value - b.value) <= 0.5 * max(a.scale, b.scale) + 1e-9 * max(abs(a.value), 1.0)


def _same_value_loose(a: _Quantity, b: _Quantity) -> bool:
    """丸め・概数の言い換えまで含めて同じ値か (量の揺れの判定は広めに同値とみなす)。"""
    if _same_value_strict(a, b):
        return True
    big = max(abs(a.value), abs(b.value))
    rel = 0.05 if (a.approx or b.approx) else _DRIFT_REL_TOLERANCE
    return abs(a.value - b.value) <= rel * big


def _quantities_by_key(quantities: list[_Quantity]) -> tuple[dict[tuple[str, str], _Quantity], set[tuple[str, str]]]:
    """1 応答の量をキーごとに 1 つへ。1 応答の中で値が割れたキーは曖昧として返す。"""
    picked: dict[tuple[str, str], _Quantity] = {}
    ambiguous: set[tuple[str, str]] = set()
    for q in quantities:
        key = (q.label, q.unit)
        prev = picked.get(key)
        if prev is None:
            picked[key] = q
        elif not _same_value_loose(prev, q):
            ambiguous.add(key)
    for key in ambiguous:
        picked.pop(key, None)
    return picked, ambiguous


def _sentence_mentions_any(sentence: str, values: Iterable[str]) -> bool:
    compact = norm_span(sentence)
    return any(v and norm_span(v) in compact for v in values)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


def find_inconsistencies(turns: Any) -> list[Finding]:
    """ターンの列 (古い順) からターンをまたぐ矛盾を列挙する (純関数・例外を出さない)。

    Args:
        turns: ``[{"turn_id", "session_id", "query", "response", ...}, ...]``。
            ``session_id`` が無いターンは同じ 1 セッション (空文字) とみなす。
            ``query`` / ``response`` が文字列でなければ空として扱う。

    Returns:
        :class:`Finding` の列 (ターン順)。``evidence`` に旧値・新値・訂正のターン・
        引いた文の抜粋などの根拠を持つ。
    """
    try:
        return _find(_normalize_turns(turns))
    except Exception:  # noqa: BLE001 - 検査は失敗しても呼出側を止めない
        logger.warning("conversation consistency check failed", exc_info=True)
        return []


def _find(turns: list[_Turn]) -> list[Finding]:
    findings: list[Finding] = []
    corrections: list[_Correction] = []
    session_texts: dict[str, list[str]] = {}
    # セッションごとの量の記録: key -> (ターン, 量, そのターンの問い)
    last_quantity: dict[str, dict[tuple[str, str], tuple[_Turn, _Quantity]]] = {}
    ambiguous_keys: dict[str, set[tuple[str, str]]] = {}
    # 数の前提を持つ量 (stale_derived_value 用): key -> [(ターン, 量)]
    history: dict[str, list[tuple[_Turn, _Quantity]]] = {}
    # セッションごとにユーザーが問いで与えた数 (入力であって計算結果でない)
    user_numbers: dict[str, list[float]] = {}

    for turn in turns:
        try:
            prior = session_texts.setdefault(turn.session, [])
            new_corrections = _extract_corrections(turn, prior)
            new_olds = {norm_span(c.old) for c in new_corrections}
            query_n = norm_span(turn.query)
            # 後のユーザー発話が旧値をまた述べたら、その訂正は以後使わない
            for c in corrections:
                if c.active and norm_span(c.old) in query_n and norm_span(c.old) not in new_olds:
                    c.active = False

            # --- stale_old_value ---
            for c in corrections:
                if not c.active or c.turn_index >= turn.index:
                    continue
                for hit in _stale_old_value_hits(c, turn):
                    findings.append(Finding("stale_old_value", turn.index, {
                        "turn_id": turn.turn_id,
                        "session_id": turn.session,
                        "scope": "same_session" if c.session == turn.session else "cross_session",
                        "old": c.old,
                        "new": c.new,
                        "attribute": c.attribute,
                        "correction_turn_index": c.turn_index,
                        "correction_sentence": _snippet(c.sentence),
                        **hit,
                    }))
                    break

            quantities = _quantities(turn.response)
            picked, ambiguous = _quantities_by_key(quantities)
            amb = ambiguous_keys.setdefault(turn.session, set())
            amb |= ambiguous
            seen = last_quantity.setdefault(turn.session, {})
            hist = history.setdefault(turn.session, [])

            query_numbers = _numbers(turn.query)
            user_nums = user_numbers.setdefault(turn.session, [])
            user_nums.extend(query_numbers[:64])
            if len(user_nums) > MAX_ACTIVE_CORRECTIONS * 8:
                del user_nums[: len(user_nums) - MAX_ACTIVE_CORRECTIONS * 8]

            # --- stale_derived_value ---
            for c in new_corrections:
                if not c.old_numbers or not c.new_numbers:
                    continue
                findings.extend(_stale_derived(c, turn, picked, hist, user_nums))

            # --- quantity_drift ---
            if new_corrections or query_numbers:
                seen.clear()  # 前提が変わりうる: 以前の量とは比べない
            else:
                for key, q in picked.items():
                    if key in amb or key not in seen:
                        continue
                    prev_turn, prev_q = seen[key]
                    if _same_value_loose(prev_q, q):
                        continue
                    if marks_not_own_restatement(q.sentence):
                        continue
                    if any(_same_value_loose(prev_q, other) for other in quantities):
                        continue  # 前の値にも触れている (比較・言い直し)
                    findings.append(Finding("quantity_drift", turn.index, {
                        "turn_id": turn.turn_id,
                        "session_id": turn.session,
                        "label": key[0],
                        "unit": key[1],
                        "previous_turn_index": prev_turn.index,
                        "previous_value": prev_q.value,
                        "previous_span": _snippet(prev_q.sentence),
                        "value": q.value,
                        "span": _snippet(q.sentence),
                    }))
            for key in amb:
                seen.pop(key, None)
            for key, q in picked.items():
                if key not in amb:
                    seen[key] = (turn, q)
                    hist.append((turn, q))

            corrections.extend(new_corrections)
            if len(corrections) > MAX_ACTIVE_CORRECTIONS:
                del corrections[: len(corrections) - MAX_ACTIVE_CORRECTIONS]
            if len(hist) > MAX_ACTIVE_CORRECTIONS * 8:
                del hist[: len(hist) - MAX_ACTIVE_CORRECTIONS * 8]
            prior.append(turn.query)
            prior.append(turn.response)
            # 旧値を探す範囲は訂正の宛先の同定と同じ遡りの上限 (問いと応答で 2 本ずつ)
            if len(prior) > DEFAULT_LOOKBACK * 2:
                del prior[: len(prior) - DEFAULT_LOOKBACK * 2]
        except Exception:  # noqa: BLE001 - 1 ターンの失敗で全体を止めない
            logger.warning("conversation consistency: turn skipped", turn_index=turn.index, exc_info=True)
    return findings


def _stale_derived(
    correction: _Correction,
    turn: _Turn,
    picked: dict[tuple[str, str], _Quantity],
    hist: list[tuple[_Turn, _Quantity]],
    user_numbers: Iterable[float],
) -> list[Finding]:
    """数の前提を置き換えたターンで、前の前提の計算結果がそのまま残っているか。

    見るのは、訂正の問いが名指しした量 (「毎月の返済額を計算し直して」の 返済額) だけ。
    """
    user_numbers = tuple(user_numbers)
    old_n = norm_span(correction.old)
    out: list[Finding] = []
    for key, q in picked.items():
        if any(abs(q.value - v) <= 1e-9 for v in user_numbers):
            continue  # ユーザーが与えた入力 (置き換えた前提・据え置きの条件) は計算結果でない
        if _attribute_in(correction.attribute, q.label):
            continue
        if not _label_named_in(q.label, turn.query):
            continue  # やり直しを頼まれた量でない (前提に依らず据え置きでよい量がある)
        if _sentence_mentions_any(q.sentence, (correction.old,)):
            continue  # 旧い前提と並べて述べている (比較)
        for prev_turn, prev_q in reversed(hist):
            if (prev_q.label, prev_q.unit) != key:
                continue
            # その量を計算したターンの問い (かそれ以前) に旧い前提が在ったこと
            if old_n not in norm_span(prev_turn.query) and old_n not in norm_span(prev_turn.response):
                break
            if _same_value_strict(prev_q, q):
                out.append(Finding("stale_derived_value", turn.index, {
                    "turn_id": turn.turn_id,
                    "session_id": turn.session,
                    "label": key[0],
                    "unit": key[1],
                    "old": correction.old,
                    "new": correction.new,
                    "previous_turn_index": prev_turn.index,
                    "previous_span": _snippet(prev_q.sentence),
                    "value": q.value,
                    "span": _snippet(q.sentence),
                }))
            break
    return out


__all__ = ["Finding", "FindingKind", "find_inconsistencies"]
