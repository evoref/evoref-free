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
- 更新は押し出しに同期する。押し出しの無いターンではスレートは 1 文字も
  変わらないので、接頭辞 KV を余分に壊さない。
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

#: プロンプトに描くときの見出し (locale 別)。
_HEADINGS: dict[str, str] = {
    "ja": "[会話の要点] (窓から外れた発話の要点。今回の会話で言い直されていればそちらが優先)",
    "en": "[Conversation facts] (points from turns no longer in the window; later statements in this conversation take precedence)",
}


class SessionFactSlate:
    """セッション単位の要点表。``WorkingMemory`` が 1 つ持つ。"""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, str] = OrderedDict()
        self.version: int = 0

    def __len__(self) -> int:
        return len(self._entries)

    def absorb(self, turns: list[dict]) -> int:
        """押し出されたターン列から要点を取り込む。取り込んだ件数を返す。"""
        try:
            from backend.free.memory.notes.note_builder import resolve_fact_attribute
        except Exception:  # pragma: no cover - 辞書が無い構成
            return 0
        added = 0
        for turn in turns:
            if (turn.get("role") or "") != "user":
                continue
            content = str(turn.get("content") or "")
            if not content.strip():
                continue
            for sentence in _SENTENCE_SPLIT_RE.split(content):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for fact_type in _FACT_TYPES:
                    attr = resolve_fact_attribute(sentence, fact_type, mode="chat")
                    if not attr:
                        continue
                    key = f"{fact_type}.{attr}"
                    value = sentence[:_VALUE_MAX_CHARS]
                    if self._entries.get(key) == value:
                        continue
                    # 新しい発話が勝つ。順序も更新順に寄せる。
                    self._entries.pop(key, None)
                    self._entries[key] = value
                    added += 1
        if added:
            self.version += 1
            logger.debug("fact slate absorbed %d entr(y/ies), total=%d", added, len(self._entries))
        return added

    def render(self, max_tokens: int, locale: str = "ja") -> str:
        """予算内に収まる分だけ、新しいものから描く ("" = 無し / 無効)。"""
        if max_tokens <= 0 or not self._entries:
            return ""
        heading = _HEADINGS.get(locale, _HEADINGS["ja"])
        lines: list[str] = []
        total = estimate_tokens(heading)
        for key, value in reversed(self._entries.items()):
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
        self.version += 1
