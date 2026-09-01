"""

``SemanticFact.subject`` 文字列を structured な 3 要素 (pillar / category /
segments) として扱うための dataclass。on-disk での subject フィールドは引き
続き文字列のまま保持し、in-memory 操作時のみ本 dataclass を噛ませて
parse / canonical / migrate の責務を集中化する。

## 位置付け

導入された ``backend.free.memory.notes.subject_ns`` は
「subject 生成 API (``make_loop_subject`` 等)」と「pillar prefix 検証
(``validate_subject_namespace``)」を提供する。本モジュールはその上に
構造化されたキー表現を追加し、将来の category rename / segment 拡張を
:meth:`SubjectKey.with_category` で機械的に扱えるようにする。

## 役割分担

| モジュール | 責務 |
|---|---|
| ``subject_ns`` | subject 生成 / pillar prefix 検証 / kind allowlist |
| ``subject_key`` | parse / canonical / with_category (構造化) |
| ``subject_canonicalizer`` | 日本語一人称の正規化 (辞書ベース) |

``subject_canonicalizer`` は pillar prefix subject の bypass 時に
:meth:`SubjectKey.try_parse` → :meth:`SubjectKey.canonical` を通すことで、
前後空白 / 先頭ドット / 連続ドット等の軽微な表記揺れを除去する。

## 設計原則

- dataclass は **frozen** (immutable)。`with_category` などの操作は
  新インスタンスを返す
- parse は pillar prefix を厳格に検査するが、category / segments の
  文字種は subject_ns の allowlist に縛られない
  (自然文 extractor 出力や自動命名の多様性を許容)
- ``SemanticFact.subject`` フィールドは文字列のまま。本 dataclass は
  on-disk 層には露出せず、in-memory 操作層に閉じる
- pillar 境界 上は EvorefMem 内部モジュール。他 pillar
  は Fact View 経由でアクセスする (本モジュールの直接 import 禁止)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

SubjectPillar = Literal["loop", "learn", "mem"]
"""3 pillar namespace。``subject_ns.SubjectPillar`` と同値 (型互換)。

``subject_ns`` から再 export せずここでも定義するのは、`subject_key` が
`subject_ns` に依存しないシンプルな dataclass として完結させるため。
"""

ALL_PILLARS: frozenset[SubjectPillar] = frozenset(get_args(SubjectPillar))


class SubjectKeyError(ValueError):
    """:class:`SubjectKey` の parse / canonical / with_category の入力不正。"""


@dataclass(frozen=True)
class SubjectKey:
    """``{pillar}.{category}.{segments...}`` の構造化表現。

    Attributes:
        pillar: ``"loop"`` / ``"learn"`` / ``"mem"`` のいずれか
        category: 2 番目のセグメント (``create_task`` / ``policy`` /
            ``failure`` / ``user`` 等)。``subject_ns.{LOOP,LEARN,MEM}_KINDS``
            と概ね対応するが、parse 時は allowlist チェックを行わない
        segments: 3 番目以降の可変長セグメント。空 tuple (segments なし) は
            ``mem.user`` のような 2 セグメント subject を表す

    Examples:
        >>> SubjectKey.parse("mem.create_task.proj42.abc123")
        SubjectKey(pillar='mem', category='create_task', segments=('proj42', 'abc123'))

        >>> SubjectKey.parse("mem.user").canonical()
        'mem.user'

        >>> SubjectKey("learn", "policy", ("create", "search", "top_k")).canonical()
        'learn.policy.create.search.top_k'

        >>> SubjectKey.parse("mem.create_task.p1.x").with_category("create_history").canonical()
        'mem.create_history.p1.x'
    """

    pillar: SubjectPillar
    category: str
    segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.pillar not in ALL_PILLARS:
            raise SubjectKeyError(
                f"unknown pillar prefix: {self.pillar!r} "
                f"(allowed: {sorted(ALL_PILLARS)})",
            )
        if not isinstance(self.category, str) or not self.category:
            raise SubjectKeyError("category must be non-empty str")
        if "." in self.category:
            raise SubjectKeyError(
                f"category must not contain '.': {self.category!r}",
            )
        if not isinstance(self.segments, tuple):
            raise SubjectKeyError(
                f"segments must be tuple, got {type(self.segments).__name__}",
            )
        for i, seg in enumerate(self.segments):
            if not isinstance(seg, str) or not seg:
                raise SubjectKeyError(
                    f"segments[{i}] must be non-empty str, got {seg!r}",
                )
            if "." in seg:
                raise SubjectKeyError(
                    f"segments[{i}] must not contain '.': {seg!r}",
                )

    def canonical(self) -> str:
        """``pillar.category[.segments...]`` の canonical 文字列を返す。"""
        parts: list[str] = [self.pillar, self.category, *self.segments]
        return ".".join(parts)

    @classmethod
    def parse(cls, subject: str) -> SubjectKey:
        """文字列 subject を :class:`SubjectKey` に分解する。

        Raises:
            SubjectKeyError: 以下のいずれかの場合:

                - 引数が str でない / 空文字 / 前後 trim 後に空
                - ピリオド区切りが 1 セグメント以下 (``pillar.category`` 不成立)
                - pillar prefix が ``loop`` / ``learn`` / ``mem`` のいずれでもない
                - category または segment のいずれかが空 (``a..b`` / ``a.`` 等)
                - category に ``.`` が含まれる (parse ロジック上あり得ないが防衛)
        """
        if not isinstance(subject, str):
            raise SubjectKeyError(
                f"subject must be str, got {type(subject).__name__}",
            )
        trimmed = subject.strip()
        if not trimmed:
            raise SubjectKeyError("subject must be non-empty after strip")
        tokens = trimmed.split(".")
        if len(tokens) < 2:
            raise SubjectKeyError(
                f"subject requires at least pillar.category: {subject!r}",
            )
        pillar = tokens[0]
        if pillar not in ALL_PILLARS:
            raise SubjectKeyError(
                f"unknown pillar prefix: {pillar!r} "
                f"(allowed: {sorted(ALL_PILLARS)}, input={subject!r})",
            )
        category = tokens[1]
        if not category:
            raise SubjectKeyError(
                f"empty category in subject: {subject!r}",
            )
        segments_raw = tokens[2:]
        for i, seg in enumerate(segments_raw):
            if not seg:
                raise SubjectKeyError(
                    f"empty segment at index {i} in subject: {subject!r}",
                )
        # mypy narrowing: pillar は ALL_PILLARS メンバーなので cast 不要
        return cls(
            pillar=pillar,  # type: ignore[arg-type]
            category=category,
            segments=tuple(segments_raw),
        )

    @classmethod
    def try_parse(cls, subject: str) -> SubjectKey | None:
        """parse 失敗時に ``None`` を返す silent variant。

        pillar prefix を持たない自然文 subject / 不正形式を拒否せずに
        通す必要がある呼出し側 (例: :class:`SubjectCanonicalizer` の
        bypass 経路) で使用する。
        """
        try:
            return cls.parse(subject)
        except SubjectKeyError:
            return None

    def with_category(self, new_category: str) -> SubjectKey:
        """category を差し替えた新 :class:`SubjectKey` を返す (segments は保持)。

        ``SubjectCategoryRenameMigration`` など category rename 補助で用いる。
        ``with_segments`` 相当の機能が必要になったら別途追加する想定。
        """
        if not isinstance(new_category, str) or not new_category:
            raise SubjectKeyError("new_category must be non-empty str")
        if "." in new_category:
            raise SubjectKeyError(
                f"new_category must not contain '.': {new_category!r}",
            )
        # frozen=True のため replace でも良いが、dataclasses.replace は
        # validator を呼ぶので明示的に新 instance を作る。
        return SubjectKey(
            pillar=self.pillar,
            category=new_category,
            segments=self.segments,
        )


#: 属性を解決できなかった発話が落ちる汎用フォールバックの葉。
#: ``extractors.chat`` が ``resolve_fact_attribute_matches`` で 0 件のときに使う。
GENERIC_ATTRIBUTE = "user"


def is_generic_subject(subject: str) -> bool:
    """``mem.<kind>.user`` — 属性としての身元を持たない汎用スロットか。

    汎用スロットは「分類できなかった発話」の置き場で、固有の属性を主張しない。
    そのため訂正の宛先にもならず、``asked_attrs`` の免除にも掛からない。実質
    その発話の **影** であり、影の元 (同じノートから起きた固有スロットの
    ファクト) が supersede されたら一緒に畳む必要がある
    (:meth:`SemanticFactStore._supersede_generic_shadows`)。
    """
    parts = (subject or "").split(".")
    return (
        len(parts) == 3
        and parts[0] == "mem"
        and parts[2] == GENERIC_ATTRIBUTE
    )


__all__ = [
    "ALL_PILLARS",
    "GENERIC_ATTRIBUTE",
    "SubjectKey",
    "SubjectKeyError",
    "SubjectPillar",
    "is_generic_subject",
]
