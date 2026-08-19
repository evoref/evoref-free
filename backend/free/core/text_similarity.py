"""文字 bi-gram ベースのテキスト類似度 (埋め込み不要の決定論指標)。

埋め込みを持たない層 (訂正・言い直しの検出、few-shot の多様性判定) が
「2 つの発話がどれくらい同じか」を測るための共通実装。

**文字集合の Jaccard を使わないこと。** 順序も出現回数も無視するため、日本語の
定型文では助詞・語尾・句読点の共通だけで閾値を超える。実測 (2026-08-18 の経験
136 件):

    「あなたの名前を教えて」→「あなたの得意なことを教えて」   集合 Jaccard 0.571

は別の質問だが、旧実装 (閾値 0.5) は言い直しとして記録していた。bi-gram の
コサインなら 0.577 で、閾値を分布の実測に合わせれば分離できる。
"""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt


def char_bigrams(text: str) -> Counter:
    """テキストから文字 bi-gram の出現頻度を返す。

    1 文字以下は自身を 1 件として扱う (空文字は空 Counter)。
    """
    t = (text or "").lower().strip()
    if len(t) < 2:
        return Counter({t: 1}) if t else Counter()
    return Counter(t[i:i + 2] for i in range(len(t) - 1))


def counter_cosine(a: Counter, b: Counter) -> float:
    """2 つの Counter 間のコサイン類似度。どちらかが空なら 0.0。"""
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(key, 0) for key, count in a.items())
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def bigram_cosine(a: str, b: str) -> float:
    """2 つのテキストの文字 bi-gram コサイン類似度。"""
    return counter_cosine(char_bigrams(a), char_bigrams(b))


def bigram_coverage(base: str, other: str) -> float:
    """``base`` の bi-gram のうち ``other`` にも現れる割合 [0, 1]。

    「前の発話がほぼそのまま残っているか」を測る非対称な指標。コサインと違い
    **追加された語の量に影響されない**ので、「同じ問いに制約を足しただけ」
    (深掘り) と「言い直し」を分けるのに使う。

    実測 (2026-08-18、旧実装が言い直しと判定した 17 組):

        深掘り (「Xを3行で」→「Xを、Yに絞って3行で」) … 0.929〜0.955
        言い直し (「もう一度説明して」)                 … 0.857
        別の質問 (名前 → 得意なこと)                   … 0.667
        話題転換 (GIL → Go の並行処理)                 … 0.368
    """
    base_grams = set(char_bigrams(base))
    if not base_grams:
        return 0.0
    return len(base_grams & set(char_bigrams(other))) / len(base_grams)


#: 日本語の **機能語を担う文字** と記号。内容語 (漢字 / カタカナ / ラテン / 数字)
#: だけを残すために落とす。
#:
#: 生の bi-gram コサインは日本語の定型文で機能語が支配的になり、**内容が違う
#: 質問どうしが、内容が同じ言い直しより高く出る**。実測 (2026-08-18):
#:
#:   言い直し 「Pythonのリスト操作を教えて」→「Pythonでリストの操作方法は？」
#:            生 0.516 / 内容語 0.870
#:   別の質問 「あなたの名前を教えて」→「あなたの得意なことを教えて」
#:            生 0.577 / 内容語 0.000
#:   別の質問 「欠損値の扱いを3行で教えて」→「外れ値の検出を3行で教えて」
#:            生 0.706 / 内容語 0.333
#:
#: 生スコアでは真偽が逆転している (0.516 < 0.577) が、内容語だけで測ると
#: 完全に分離する (真 0.866〜1.000 / 偽 0.000〜0.333)。
#: ``ー`` (長音) は落とさない — カタカナ語の一部で、内容語側の情報を持つ。
_JA_FUNCTION_CHARS_RE = re.compile(
    r"[ぁ-ん\s、。，．！？!?…「」『』（）()・:：;；,.\-–—]+",
)


def content_form(text: str) -> str:
    """内容語だけを残した形を返す (日本語の機能語文字と記号を落とす)。"""
    return _JA_FUNCTION_CHARS_RE.sub("", text or "")


def content_bigram_cosine(a: str, b: str) -> float:
    """内容語だけで測った bi-gram コサイン類似度。

    どちらかの内容語が空になる場合 (ひらがなだけの発話など) は、判別材料が
    無いので生のテキストで測った値へ縮退する。
    """
    ca, cb = content_form(a), content_form(b)
    if not ca or not cb:
        return bigram_cosine(a, b)
    return bigram_cosine(ca, cb)
