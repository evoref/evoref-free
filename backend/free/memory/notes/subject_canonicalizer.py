"""Subject 正規化

EvorefMem における **意味記憶の subject 正規化** を担う。

## 役割

SemanticFact の `subject` フィールドは検索・索引・コンフリクト判定のキー
となるため、表記ゆれを抑える必要がある。本モジュールは「最小エントリの
辞書による *exact match* 正規化」を提供する。日本語/英語の一人称や所有
表現を `mem.personal.user` / `mem.personal.company` 等の canonical な subject にまとめる。

## バイパスルール (重要)

`learn.policy.create.search.top_k` のように pillar 層が機械的に命名
した subject (`^(loop|learn|mem)\\.` プレフィックス) は **辞書ルックアップを
完全にスキップ** する。これは:

- Loop / Learn / Mem の各 pillar が `loop.*` / `learn.*` / `mem.*`
  名前空間を占有しており、ここに辞書正規化が掛かると機械的命名が壊れる
- `facts_by_pillar.idx` が 3 prefix を前提に索引化されているため、正規化
  されると索引が無効化される

`extraction_skip_subject_canonicalize_regex` (config: ``memory.facts``) で
バイパス正規表現を上書きできるが、デフォルトは ``^(loop|learn|mem)\\.``。

## SubjectKey ベース整形

:class:`backend.free.memory.semantic.subject_key.SubjectKey` を使い、
バイパス経路 / 辞書ヒット経路の双方で「構造を伴う canonical 形」に落とし込む。
具体的には:

- バイパスヒット: :meth:`SubjectKey.try_parse` で structured 表現を得て、
  :meth:`SubjectKey.canonical` で再生成する (well-formed subject では入力と
  同一結果を返す idempotent な変換)
- 辞書ヒット: 辞書から得た canonical 文字列 (例: ``"mem.personal.company"``) に
  対して同じ SubjectKey round-trip を掛ける

いずれも :meth:`SubjectKey.try_parse` が ``None`` を返す不正形式 (空セグメント
など) の場合は passthrough にフォールバックする。

## auto_expand について

「最小から、自動拡張なし」と決まっている。本モジュールはエントリの
自動学習・自動追加を **行わない**。エントリ追加はユーザーによる
``subject_dictionary.json`` の手編集、または将来の専用管理 API 経由で
行う想定。

## ファイル形式

``local/memory/semantic/subject_dictionary.json``::

    {
      "version": 1,
      "entries": {
        "私": "user",
        "僕": "user",
        ...
      }
    }

破損や欠損時は ``ensure_default_subject_dictionary`` がデフォルトを書き
出して回復する (後方互換は提供しない)。

## API

- :class:`SubjectDictionary` — 不変の辞書ラッパ (大文字小文字を吸収する
  内部索引を持つ)
- :class:`SubjectCanonicalizer` — `^(loop|learn|mem)\\.` バイパス + 辞書ルックアップ
- :func:`load_subject_dictionary` — JSON ファイルから読み込む
- :func:`write_subject_dictionary` — JSON ファイルへ書き出す
- :func:`ensure_default_subject_dictionary` — 無ければデフォルトを書き出す
- :func:`default_subject_dictionary` — メモリ上のデフォルト辞書を返す
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Pattern

from backend.free.memory.semantic.subject_key import SubjectKey
from backend.log_config import get_logger

logger = get_logger("memory.subject_canonicalizer")


SUBJECT_DICTIONARY_FILE_VERSION = 1
"""``subject_dictionary.json`` のフォーマットバージョン。

互換性のない変更を入れる場合のみインクリメントし、古い値を読んだ場合は
警告ログを出して既定値で再生成する (後方互換は提供しない方針)。
"""

DEFAULT_BYPASS_REGEX = r"^(loop|learn|mem)\."
"""3 pillar namespace (``loop`` / ``learn`` / ``mem``) の正規化バイパス既定パターン。

``facts_by_pillar.idx`` の索引化と整合する。
"""

DEFAULT_SUBJECT_ENTRIES: dict[str, str] = {
    # 一人称代名詞 (日本語) → "mem.personal.user"
    "私": "mem.personal.user",
    "僕": "mem.personal.user",
    "俺": "mem.personal.user",
    "自分": "mem.personal.user",
    "我々": "mem.personal.user",
    "うち": "mem.personal.user",
    # 一人称代名詞 (英語) → "mem.personal.user"
    # 大文字/小文字は SubjectDictionary 内部で吸収するため、ここでは
    # 自然な書き方 (大文字 I) を保持する。
    "I": "mem.personal.user",
    "me": "mem.personal.user",
    "my": "mem.personal.user",
    # 所属組織 → "mem.personal.company"
    "うちの会社": "mem.personal.company",
    "我が社": "mem.personal.company",
    "my company": "mem.personal.company",
}
"""最小初期エントリ。

合計 12 件 (10 件以上)。自動拡張なし。エントリ追加はユーザーによる
``subject_dictionary.json`` の手編集を想定している。

値は現行の subject 規約 ``mem.<kind>.<attr>`` (3 セグメント) に揃える。
旧値 ``mem.user`` (2 セグメント) は :func:`~backend.free.memory.notes
.note_builder.is_single_valued_subject` 等の 3 セグメント前提と噛み合わない
(2026-09-02 監査)。``^(loop|learn|mem)\\.`` バイパスがあるため実経路では
到達しないが、辞書の形は規約と一致させておく。
"""

#: :data:`DEFAULT_SUBJECT_ENTRIES` の値が従うべき形 (テストが全件検証する)。
DEFAULT_SUBJECT_VALUE_RE = re.compile(r"^mem\.[a-z_]+\.[a-z_]+$")


# ──────────────────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CanonicalResult:
    """:meth:`SubjectCanonicalizer.canonicalize` の返り値。

    Attributes:
        canonical: 正規化後の subject (マッチしなかった場合は ``original``
            と同一)
        original: 入力された subject 文字列 (前後空白だけ trim 済み)
        bypassed: ``^(loop|learn|mem)\\.`` バイパスにヒットして辞書
            ルックアップをスキップしたか
        matched: 辞書エントリにマッチしたか (バイパス時は常に False)
    """

    canonical: str
    original: str
    bypassed: bool
    matched: bool


@dataclass(frozen=True)
class SubjectDictionary:
    """Subject 正規化辞書。

    `entries` は不変の dict。内部に大文字小文字を吸収する補助索引
    (``_lower_index``) を持つため、``__post_init__`` で構築する。

    Notes:
        Python の ``dict`` は厳密には不変ではないが、本クラスは
        ``frozen=True`` の dataclass として「外から書き換える API を
        露出しない」ことで実質的な不変性を確保している。
    """

    entries: Mapping[str, str]
    _lower_index: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        # frozen=True なので setattr 経由で初期化
        lower: dict[str, str] = {}
        for key, value in self.entries.items():
            lower.setdefault(key.lower(), value)
        object.__setattr__(self, "_lower_index", lower)

    def lookup(self, subject: str) -> str | None:
        """辞書から canonical 形を探す。見つからなければ None。

        ルックアップ順:
        1. 完全一致 (大文字小文字の差を保ったまま)
        2. 小文字化での一致 (英語一人称の "I" / "i" / "Me" 等を吸収)
        """
        if subject in self.entries:
            return self.entries[subject]
        return self._lower_index.get(subject.lower())

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, subject: object) -> bool:
        if not isinstance(subject, str):
            return False
        return self.lookup(subject) is not None


# ──────────────────────────────────────────────────────────────────────────
# 既定辞書 / 読み書き
# ──────────────────────────────────────────────────────────────────────────


def default_subject_dictionary() -> SubjectDictionary:
    """既定エントリのみを含む :class:`SubjectDictionary` を返す。"""
    return SubjectDictionary(entries=dict(DEFAULT_SUBJECT_ENTRIES))


def write_subject_dictionary(
    path: Path, entries: Mapping[str, str], *, version: int = SUBJECT_DICTIONARY_FILE_VERSION,
) -> Path:
    """``subject_dictionary.json`` を書き出す。

    親ディレクトリは必要に応じて作成する。``ensure_ascii=False`` で日本語
    エントリをそのまま保存する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": version, "entries": dict(entries)}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote subject dictionary: %s (entries=%d)", path, len(entries))
    return path


def load_subject_dictionary(path: Path) -> SubjectDictionary:
    """``subject_dictionary.json`` を読み込む。

    Raises:
        FileNotFoundError: ファイルが存在しない
        ValueError: フォーマットが不正
    """
    if not path.exists():
        raise FileNotFoundError(f"subject_dictionary.json not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"subject_dictionary.json is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"subject_dictionary.json must be a JSON object: {path}")

    version = payload.get("version")
    if version != SUBJECT_DICTIONARY_FILE_VERSION:
        logger.warning(
            "subject_dictionary.json version mismatch: expected=%s actual=%s (path=%s)",
            SUBJECT_DICTIONARY_FILE_VERSION, version, path,
        )

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        raise ValueError(
            f"subject_dictionary.json must contain an 'entries' object: {path}",
        )
    entries: dict[str, str] = {}
    for key, value in raw_entries.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"subject_dictionary.json entries must be string->string: {path}",
            )
        entries[key] = value
    return SubjectDictionary(entries=entries)


def ensure_default_subject_dictionary(path: Path) -> SubjectDictionary:
    """ファイルが無ければデフォルトを書き出してから読み込む。

    あれば既存をそのまま読み込む。読み込みに失敗した場合は
    ``ValueError`` を再送出する (回復は呼び出し側の判断に委ねる)。
    """
    if not path.exists():
        write_subject_dictionary(path, DEFAULT_SUBJECT_ENTRIES)
    return load_subject_dictionary(path)


# ──────────────────────────────────────────────────────────────────────────
# SubjectKey 連携
# ──────────────────────────────────────────────────────────────────────────


def _structured_canonical_or_passthrough(subject: str) -> str:
    """:class:`SubjectKey` で parse + canonical を試み、失敗時は原文を返す。

    pillar prefix を持ち SubjectKey に分解可能な文字列 (例:
    ``"mem.personal.company"``) は round-trip 経由で整形され、分解不能な
    文字列 (空セグメント / 自然文 subject) はそのまま返される。いずれも
    例外は投げない (silent passthrough)。
    """
    key = SubjectKey.try_parse(subject)
    if key is None:
        return subject
    return key.canonical()


# ──────────────────────────────────────────────────────────────────────────
# Canonicalizer 本体
# ──────────────────────────────────────────────────────────────────────────


class SubjectCanonicalizer:
    """Subject 正規化器。

    - ``^(loop|learn|mem)\\.`` (または ``bypass_regex`` で指定したパターン) に
      マッチする subject は辞書ルックアップを **完全にスキップ** する。
    - それ以外は :class:`SubjectDictionary` で exact match (大文字小文字
      は吸収) を試み、ヒットすれば canonical 形へ書き換える。
    - ヒットしない場合は **入力をそのまま返す** (silent passthrough)。
      これは「自動拡張なし」原則および「subject の人手命名を許容する」
      ためで、ヒットしないことはエラーではない。

    Args:
        dictionary: 既定辞書または手編集された辞書
        bypass_regex: バイパス用正規表現 (文字列。コンパイル済みパターン
            を直接渡したい場合は ``bypass_pattern`` を使う)
        bypass_pattern: 既にコンパイル済みの ``re.Pattern``。
            ``bypass_regex`` より優先される。
    """

    def __init__(
        self,
        dictionary: SubjectDictionary,
        *,
        bypass_regex: str = DEFAULT_BYPASS_REGEX,
        bypass_pattern: Pattern[str] | None = None,
    ) -> None:
        self._dictionary = dictionary
        if bypass_pattern is not None:
            self._bypass_pattern = bypass_pattern
        else:
            try:
                self._bypass_pattern = re.compile(bypass_regex)
            except re.error as exc:
                raise ValueError(
                    f"invalid bypass_regex={bypass_regex!r}: {exc}",
                ) from exc

    @property
    def dictionary(self) -> SubjectDictionary:
        return self._dictionary

    @property
    def bypass_pattern(self) -> Pattern[str]:
        return self._bypass_pattern

    def is_bypassed(self, subject: str) -> bool:
        """``subject`` がバイパス正規表現にマッチするか。"""
        return bool(self._bypass_pattern.search(subject))

    def canonicalize(self, subject: str) -> CanonicalResult:
        """``subject`` を正規化する。

        前後空白は trim する。空文字は ``CanonicalResult(canonical="",
        original="", bypassed=False, matched=False)`` を返す
        (呼び出し側で扱うかは任意)。

        バイパスヒット / 辞書ヒットで得られた canonical 文字列はさらに
        :class:`SubjectKey` で parse + canonical round-trip に掛ける。
        well-formed な pillar subject では入力と同一結果を返す idempotent
        な変換であり、:meth:`SubjectKey.try_parse` が ``None`` を返す不正
        形式 (空セグメント等) の場合は passthrough にフォールバックする。
        """
        original = subject.strip() if subject else ""
        if not original:
            return CanonicalResult(canonical="", original="", bypassed=False, matched=False)

        if self.is_bypassed(original):
            canonical = _structured_canonical_or_passthrough(original)
            return CanonicalResult(
                canonical=canonical, original=original, bypassed=True, matched=False,
            )

        hit = self._dictionary.lookup(original)
        if hit is not None:
            canonical = _structured_canonical_or_passthrough(hit)
            return CanonicalResult(
                canonical=canonical, original=original, bypassed=False, matched=True,
            )
        return CanonicalResult(
            canonical=original, original=original, bypassed=False, matched=False,
        )

    def __call__(self, subject: str) -> str:
        """canonical 形だけ欲しい呼び出し向けの糖衣 (`canonicalize().canonical`)。"""
        return self.canonicalize(subject).canonical


def build_default_canonicalizer(
    *,
    dictionary_path: Path | None = None,
    bypass_regex: str = DEFAULT_BYPASS_REGEX,
) -> SubjectCanonicalizer:
    """既定の :class:`SubjectCanonicalizer` を構築する補助関数。

    ``dictionary_path`` が与えられればファイル (なければ書き出してから
    読み込み) を、None なら in-memory のデフォルト辞書を使う。
    """
    if dictionary_path is not None:
        dictionary = ensure_default_subject_dictionary(dictionary_path)
    else:
        dictionary = default_subject_dictionary()
    return SubjectCanonicalizer(dictionary, bypass_regex=bypass_regex)
