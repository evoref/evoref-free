"""押し出したターンの事実スレート (f_02 §1.2、2026-09-03 設計変更)。

WM の押し出しで窓から消えたターンは、次の想起まで検索頼みになる。訂正後の値
(奥多摩→秩父) が窓から出た瞬間に、モデルは検索が当たるまでその値を知らない。
スレートはその穴を埋める **決定論の要点表**:

- 押し出された **user 発話**から、抽出側と同じ属性辞書
  (``note_builder.resolve_fact_attribute``、``fact_attributes.yaml``) で属性を
  同定し、``fact_type.attribute`` をキーに **その発話の文をそのまま** 値として持つ。
  散文要約は使わない (要約は主題しか残さず固有名詞・数値を落とす、
  arXiv 2503.19114)。LLM は関与しない。
- 同じキーは新しい発話が勝つ (訂正の supersede と同じ規則)。
- 属性に載らない発話 (問い・依頼) は **やり取りの行** (``問: … → 答: …``) として
  持つ。答えは押し出された assistant 応答の先頭の文を機械的に抜いたもの
  (補助タスクは使わない、不変則 #1)。以前は user 発話だけを見ていたため、
  「初心者向けの野菜は？」が ``- food: 初心者向けの野菜は？`` のように問いに
  ラベルを付けただけの行になり、後で「ここまでをまとめて」と頼まれても落とした
  分の答えを復元できなかった (2026-10-09 実機 301 ターン)。問いの文 (？/? で
  終わる文) は属性の値にしない — 利用者自身の値の言明ではない。
- 更新は押し出しに同期する。押し出しの無いターンではスレートは 1 文字も
  変わらない。描く位置は system の外 (履歴の先頭、``core.inference.build_messages``
  の ``history_preamble``) — system に足すと押し出しのたびに system が変わり、
  途中から巻き戻せない hybrid (recurrent) モデルでは system ごと全再計算になる。
- セッション終了で消える。永続化は従来どおり STM → SemMem。
"""

from __future__ import annotations

import re
from collections import OrderedDict

from backend.log_config import get_logger
from backend.utils import estimate_tokens

logger = get_logger("memory.fact_slate")

#: 値として持つ 1 文の上限 (文字)。固有名詞・数値を落とさない長さで、かつ
#: スレート全体の予算 (``prompt.fact_slate_max_tokens``) に複数件が収まる長さ。
_VALUE_MAX_CHARS = 80

#: 属性同定の対象 fact_type (injector の ``_USER_ATTRIBUTE_FACT_TYPES`` と同じ集合)。
_FACT_TYPES: tuple[str, ...] = ("personal_fact", "preference", "emotion", "opinion")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。．.!?！？\n])")

#: やり取りの行の問い / 答えの上限 (文字)。1 行がおよそ 100 トークンに収まる長さ。
_QUESTION_MAX_CHARS = 60
_ANSWER_MAX_CHARS = 80
#: 答えの先頭の文がこれより短ければ (「はい。」「いい質問ですね。」) 次の文も足す。
_ANSWER_MIN_CHARS = 20
#: 答えの行頭の Markdown 記号 (箇条書き・引用・番号)。見出し行は丸ごと落とす。
_LINE_MARKER_RE = re.compile(r"^\s*(?:[-*+・•>]\s*|\d+[.)）](?!\d)\s*)+")
_EXCHANGE_PREFIX = "exchange."

#: 数量の型付き抽出 (構造の抽出であって意味の分類ではない、不変則 #14)。属性辞書に載らない
#: 計画の数量 (旅行の日数・人数・金額) を、押し出し後も「最初に言った〜」で引けるようにする。
#: 2026-10-08 監査: 「京都に2泊3日で…」が窓から出た後の「最初に言った日数は」に答えられなかった。
_QUANTITY_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("日数", re.compile(r"\d+泊\d+日|\d+日間")),
    ("人数", re.compile(r"\d+(?:人|名)(?!分)")),
    ("金額", re.compile(r"\d[\d,]*(?:\.\d+)?(?:万|億)?円")),
)

#: プロンプトに描くときの見出し (locale 別)。
_HEADINGS: dict[str, str] = {
    "ja": "[会話の要点] (窓から外れた発話の要点。今回の会話で言い直されていればそちらが優先)",
    "en": "[Conversation facts] (points from turns no longer in the window; later statements in this conversation take precedence)",
}
#: やり取りの行のラベル (問い, 答え) (locale 別)。
_EXCHANGE_LABELS: dict[str, tuple[str, str]] = {
    "ja": ("問", "答"),
    "en": ("Q", "A"),
}


class SessionFactSlate:
    """セッション単位の要点表。``WorkingMemory`` が 1 つ持つ。"""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, str] = OrderedDict()
        self.version: int = 0
        #: やり取りの行の通し番号 (キーの一意化)。
        self._exchange_seq: int = 0
        #: 答えを待っているやり取りの行のキー。押し出しの連鎖が 1 件を残して止まると
        #: 問いと答えが別々の押し出しで届くので、次に届いた assistant をここへ付ける。
        self._pending_exchange: str | None = None
        #: やり取りの行の答え (キー → 要旨)。問いは ``_entries`` が持つ。
        self._answers: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def absorb(self, turns: list[dict]) -> int:
        """押し出されたターン列から要点を取り込む。取り込んだ件数を返す。

        user 発話は属性 / 数量の行に、属性に載らなければやり取りの行にする。
        assistant 応答は直前のやり取りの行の答えになる (問いの無い応答は捨てる)。
        """
        try:
            from backend.free.memory.notes.note_builder import resolve_fact_attribute
        except Exception:  # pragma: no cover - 辞書が無い構成
            return 0
        added = 0
        for turn in turns:
            role = turn.get("role") or ""
            content = str(turn.get("content") or "")
            if not content.strip():
                continue
            if role == "assistant":
                added += self._attach_answer(content)
                continue
            if role != "user":
                continue
            # 前の問いの答えが届かないまま次の問いが来たら、前の行は問いだけで閉じる。
            self._pending_exchange = None
            matched_attribute = False
            for sentence in _SENTENCE_SPLIT_RE.split(content):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if not _is_question(sentence):
                    for fact_type in _FACT_TYPES:
                        attr = resolve_fact_attribute(sentence, fact_type, mode="chat")
                        if not attr:
                            continue
                        matched_attribute = True
                        key = f"{fact_type}.{attr}"
                        value = sentence[:_VALUE_MAX_CHARS]
                        if self._entries.get(key) == value:
                            continue
                        # 新しい発話が勝つ。順序も更新順に寄せる。
                        self._entries.pop(key, None)
                        self._entries[key] = value
                        added += 1
                # 数量は値ごとに別の行 (同じ種類でも言い直しの前後を両方残し、時系列で読める)。
                for label, pattern in _QUANTITY_KINDS:
                    for token in pattern.findall(sentence):
                        key = f"quantity.{label}:{token}"
                        if key in self._entries:
                            continue
                        self._entries[key] = sentence[:_VALUE_MAX_CHARS]
                        added += 1
            if not matched_attribute:
                # 属性の行は発話そのものを持つので、やり取りの行は属性に載らない
                # 発話 (問い・依頼) だけに作る (同じ文を二重に載せない)。
                self._exchange_seq += 1
                key = f"{_EXCHANGE_PREFIX}{self._exchange_seq}"
                self._entries[key] = _clip(_one_line(content), _QUESTION_MAX_CHARS)
                self._pending_exchange = key
                added += 1
        if added:
            self.version += 1
            logger.debug("fact slate absorbed %d entr(y/ies), total=%d", added, len(self._entries))
        return added

    def _attach_answer(self, content: str) -> int:
        """答えを待っている行へ、応答の要旨 (先頭の文) を付ける。付けた件数を返す。"""
        key = self._pending_exchange
        if key is None or key not in self._entries:
            return 0
        self._pending_exchange = None
        gist = _answer_gist(content)
        if not gist:
            return 0
        self._answers[key] = gist
        return 1

    def render(self, max_tokens: int, locale: str = "ja") -> str:
        """予算内に収まる分だけ、新しいものから描く ("" = 無し / 無効)。"""
        if max_tokens <= 0 or not self._entries:
            return ""
        heading = _HEADINGS.get(locale, _HEADINGS["ja"])
        q_label, a_label = _EXCHANGE_LABELS.get(locale, _EXCHANGE_LABELS["ja"])
        lines: list[str] = []
        total = estimate_tokens(heading)
        for key, value in reversed(self._entries.items()):
            if key.startswith(_EXCHANGE_PREFIX):
                line = f"- {q_label}: {value}"
                if answer := self._answers.get(key):
                    line += f" → {a_label}: {answer}"
            else:
                line = f"- {key.split('.', 1)[-1]}: {value}"
            cost = estimate_tokens(line)
            if total + cost > max_tokens:
                break
            lines.append(line)
            total += cost
        if not lines:
            return ""
        return heading + "\n" + "\n".join(reversed(lines))

    def clear(self) -> None:
        self._entries.clear()
        self._answers.clear()
        self._pending_exchange = None
        self.version += 1


def _is_question(sentence: str) -> bool:
    """文末が疑問符の文か (句読点の構造で見る。語彙の判定はしない)。"""
    return sentence.rstrip().endswith(("?", "？"))


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _answer_gist(content: str) -> str:
    """応答の先頭の文 (短ければ次の文まで) を機械的に抜く。

    コードブロックの中身・見出し行・行頭の Markdown 記号は落とす。要約はしない — 抜いた
    文は応答に逐語で在る。
    """
    lines: list[str] = []
    in_fence = False
    for raw in content.splitlines():
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or raw.lstrip().startswith("#"):
            # 見出しは題であって答えではない
            continue
        line =_LINE_MARKER_RE.sub("", raw.replace("**", "")).strip()
        if line:
            lines.append(line)
    gist = ""
    for sentence in _SENTENCE_SPLIT_RE.split(" ".join(lines)):
        sentence = sentence.strip()
        if not sentence:
            continue
        gist = f"{gist} {sentence}" if gist else sentence
        if len(gist) >= _ANSWER_MIN_CHARS:
            break
    return _clip(gist, _ANSWER_MAX_CHARS)
