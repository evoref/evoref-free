"""生成テキストの決定論的な品質チェック (pillar 非依存の共有基盤)

モデル差し替えで静かに劣化する「表記の崩れ」を、LLM 採点に頼らず決定論で検出する。
判定器をここに集約するのは、**同じ崩れを 2 箇所で別々に定義すると片方だけ直る**
ためで、実際に以下 2 系統が同一の判定を必要とする:

- :mod:`backend.free.learning.fewshot_pool` — 崩れた応答を手本に採用しない (入口ゲート)
- :mod:`backend.free.llm.quality_probe` — モデル切替時に崩れを検出する (事前ゲート)

アシスト採点は使わない。小型モデルは自分と同種の崩れを問題と認識できず、実測で
空白混入例の quality 平均 0.80 に対し正常例 0.89 と 0.09 しか差が付かなかった
(混入例に 0.95 が 4 件)。決定論でのみ分離できる。
"""

from __future__ import annotations

import re

#: 日本語の語間に混じった空白。
#:
#: 正常な日本語では和文文字が空白で分かたれることはない (実測: Qwen3.5-9B 時代の
#: 応答 96 件中 0 件)。一方 gemma-4-12b では 76〜83% に混入し、``temperature=0.0``
#: の貪欲法でも再現した — サンプリングではなく出力分布そのものの性質。
_JA_INTERWORD_SPACE_RE = re.compile(r"[ぁ-んァ-ヶ一-龥][ 　]+[ぁ-んァ-ヶ一-龥]")

#: コードブロック。整形済みコードやログ引用では半角空白が正常に現れる。
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

#: 和文文字。応答が日本語かどうかの判定に使う。
_JA_CHAR_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")

#: 日本語判定の下限。これ未満の和文文字しか無い応答は英語応答等とみなし、
#: 語間空白チェックの母数から外す (英文の空白を誤検出しないため)。
_JA_MIN_CHARS = 20


def has_broken_ja_spacing(text: str) -> bool:
    """日本語部分に不自然な語間空白が混じっているかを判定する (純粋関数)。

    コードブロック内は対象外。整形済みコードやログの引用で半角空白が現れるのは
    正常なため、fence で囲まれた領域を除いてから判定する。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    return bool(_JA_INTERWORD_SPACE_RE.search(outside))


def is_japanese_text(text: str) -> bool:
    """語間空白チェックの母数に含めてよい日本語応答かを判定する。

    和文文字が :data:`_JA_MIN_CHARS` 未満の応答 (英語応答 / 空応答 / コードのみ)
    は False。日本語で答えていないモデルを「空白混入 0%」と誤って合格させない
    ため、母数側でも呼び出し元が本関数で足切りする。
    """
    outside = _CODE_FENCE_RE.sub("\n", text)
    return len(_JA_CHAR_RE.findall(outside)) >= _JA_MIN_CHARS


__all__ = ["has_broken_ja_spacing", "is_japanese_text"]
