"""語彙索引・検索共通のトークナイザ（文字 n-gram + ASCII 分割 + ストップワード）

MeCab を要求しないまま日本語を索引できるようにするための、辞書なしトークナイザ。
索引側 (:mod:`backend.free.rag.evidence.lexical_index`) と、決定論的な語の重なり
判定 (履歴検索 / チャンク内容ゲート / 記憶注入) が **同じ切り方** を使う必要が
あるため、実装をここ 1 箇所に置く。切り方がずれると同じ文字列が語として一致
しなくなる。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


# 接頭辞・接尾辞になりやすい高頻度のつなぎ語のみを絞って列挙し、
# 内容語バイグラムを誤って除去しないよう保守的に維持する。
DEFAULT_STOPWORD_BIGRAMS: frozenset[str] = frozenset(
    [
        "のは", "のが", "のに", "のを", "のと", "のも", "のか", "ので", "のよ",
        "はの", "はが", "はに", "はを", "はと", "はま",
        "がの", "がは", "がに", "がを", "がと", "がで",
        "にの", "には", "にが", "にを", "にと", "にし", "にな", "にお",
        "をの", "をは", "をが", "をに", "をと", "をし", "をお",
        "とが", "とは", "とに", "とを", "との", "とし",
        "です", "ます", "でし", "まし", "だっ", "った",
        "する", "した", "して", "しま", "され", "せる",
        "ある", "あり", "いる", "いま", "なる", "なっ", "なり",
        "この", "その", "あの", "どの", "これ", "それ", "あれ", "どれ",
    ]
)


#: 日本語 n-gram を切る「連続領域」。単語構成文字 (かな / 漢字 / 長音 等) の
#: 連続で、ASCII 英数字・underscore・空白・句読点・記号で途切れる。
_JA_RUN_RE = re.compile(r"[^\W\d_a-z]+")


def _split_ascii_token(raw: str) -> list[str]:
    """camelCase / snake_case / 連続数字の ASCII トークンをサブトークンに分割する。

    元トークン自体は呼び出し元で別途追加される。本関数はサブトークン
    （小文字化済み）のみを返す。元と同一になるサブトークンは除外。
    """
    subs: list[str] = []
    # まず underscore で区切る
    for part in re.split(r"_+", raw):
        if not part:
            continue
        # camelCase/PascalCase/連続数字を分解
        for m in re.finditer(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", part):
            sub = m.group(0).lower()
            if sub:
                subs.append(sub)
    return subs


#: トークナイザの版。切り方を変えたら上げる。索引の ``LexicalParams`` に焼き付き、
#: ロード時の drift 警告で「再構築まで旧い切り方の索引を引いている」ことが分かる。
#: 2: 長さ 1 の run を unigram として出す / ストップワード除去で空なら除去前へ戻す
#:    (c_16 §6.2「短い問いの取りこぼし」)。
TOKENIZER_VERSION = 2


def tokenize_ja(
    text: str,
    *,
    use_trigrams: bool = False,
    split_ascii: bool = True,
    stopwords: Iterable[str] | None = None,
) -> list[str]:
    """MeCab 不要の日本語トークナイザ。

    - NFKC 正規化
    - ASCII トークンは大小文字保持した上で小文字化したトークンを出力
    - ``split_ascii`` が真なら camelCase / snake_case を追加トークン化（元トークンも保持）
    - 日本語部分は文字 bi-gram を生成、``use_trigrams`` が真なら tri-gram も併用
    - ``stopwords`` に含まれる n-gram は除外
    """
    nfkc = unicodedata.normalize("NFKC", text)
    # ASCII span を case-preserving で抽出
    ascii_spans = re.findall(r"[A-Za-z0-9_]+", nfkc)
    ascii_tokens: list[str] = []
    for span in ascii_spans:
        low = span.lower()
        ascii_tokens.append(low)
        if split_ascii:
            for sub in _split_ascii_token(span):
                if sub != low:
                    ascii_tokens.append(sub)

    # 日本語 n-gram: ASCII / 空白 / 記号で区切った **連続領域ごと** に生成する。
    # 以前は除去後の文字列を 1 本に繋いでから bi-gram を切っていたため、
    # 「〜を使うで Python を〜」→「うで」、「〜です。次に〜」→「。次」のような
    # 境界をまたぐ偽トークンが生まれていた。偽トークンは df が極端に小さく、
    # 希少語判定で「珍しい語」として拾われて floor 免除の根拠になる。
    lower = nfkc.lower()
    ja_runs = _JA_RUN_RE.findall(lower)

    stop_set = frozenset(stopwords) if stopwords is not None else frozenset()

    bigrams = [
        run[i : i + 2] for run in ja_runs for i in range(len(run) - 1)
    ]
    # 長さ 1 の run (「猫」「猫、犬」「Python と Rust」の「と」) は bi-gram を
    # 1 つも出さず、問いが語彙外と同じ空になっていた。1 文字をそのまま出す。
    # 助詞は df が高く ``max_df_ratio`` の剪定で自然に落ちる。
    unigrams = [run for run in ja_runs if len(run) == 1]
    if stop_set:
        kept = [b for b in bigrams if b not in stop_set]
        # ストップワードだけの問い (「これは？」) は除去で空になる。索引側にも
        # 同じ語は無いが、空のまま返すと「使える語が無い」ことすら分からない。
        if kept or ascii_tokens or unigrams:
            bigrams = kept

    tokens = ascii_tokens + unigrams + bigrams

    if use_trigrams:
        trigrams = [
            run[i : i + 3] for run in ja_runs for i in range(len(run) - 2)
        ]
        if stop_set:
            trigrams = [t for t in trigrams if t not in stop_set]
        tokens += trigrams

    return tokens


# 旧名を後方互換用に残す（内部利用のみ、テスト/外部参照は tokenize_ja を推奨）。
_tokenize_ja = tokenize_ja
