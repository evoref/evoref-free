"""

4 pillar アーキテクチャ における :class:`SemanticFact.subject` の
命名規則 (``loop.*`` / ``learn.*`` / ``mem.*``) を生成・検証するヘルパ群。

## 設計方針 (CLAUDE.md §8 / docs/f_02_memory_system.md §7.1 / docs/c_05_data_schemas.md §21.3)

- 全 subject は ``{pillar}.{kind}.{...parts}`` の形式
- pillar prefix は owner pillar に一致させる (例: ``policy`` は EvorefLearn
  所有 → ``learn.policy.*``)
- ``<kind>`` は owner pillar 配下の論理種別 (``task`` / ``progress`` /
  ``policy`` / ``fewshot`` / ``personal`` / ``decision`` 等)
- ``<parts>`` は任意個数の階層パート (例: ``<mode>.<domain>.<param>`` など)

## 適用履歴

本モジュールは subject 構築 helper として導入され、既存コードへ
全面適用された (旧 ``harness.*`` prefix は撤去済)。
新規コードは本 helper 経由で subject を構築すること。

## 検証と正規化

- 英数字・ハイフン・アンダースコア・ドット以外の文字は除外 (subject として
  索引キーになるため)
- 空パートや先頭ドットは :class:`SubjectNamespaceError` を raise
- :func:`validate_subject_namespace` は subject prefix (``loop`` / ``learn`` /
  ``mem``) を返し、それ以外 (自然文 subject) は ``None`` を返す
  (自然文 subject は Fact View 層の :meth:`_assert_subject_owner` で
  passthrough 扱い)
"""

from __future__ import annotations

import re
from typing import Literal, get_args

# ──────────────────────────────────────────────────────────────────────────
# 型
# ──────────────────────────────────────────────────────────────────────────

SubjectPillar = Literal["loop", "learn", "mem"]
"""subject prefix として許容される 3 pillar (``harness`` は含まない)。

``harness`` は read-only reader として ownership には登場するが、subject prefix
としては 全廃される予定のため本 Literal には含めない
"""

ALL_SUBJECT_PILLARS: frozenset[SubjectPillar] = frozenset(get_args(SubjectPillar))
"""許容 subject prefix の集合。"""


# ``<kind>`` および ``<parts>`` に許容される文字 (英数字 / ``_`` / ``-``)
_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SUBJECT_PREFIX_RE = re.compile(r"^(loop|learn|mem)\.")


# ──────────────────────────────────────────────────────────────────────────
# 例外
# ──────────────────────────────────────────────────────────────────────────


class SubjectNamespaceError(ValueError):
    """subject namespace helper の入力不正 / 形式違反で送出される。"""


# ──────────────────────────────────────────────────────────────────────────
# 内部ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def _validate_part(part: str, *, field: str) -> None:
    """単一パートの形式を検証する。"""
    if not isinstance(part, str):
        raise SubjectNamespaceError(
            f"{field} must be str, got {type(part).__name__}",
        )
    if not part:
        raise SubjectNamespaceError(f"{field} must not be empty")
    if not _SAFE_PART_RE.match(part):
        raise SubjectNamespaceError(
            f"{field} contains invalid characters: {part!r} "
            f"(allowed: alnum / '_' / '-', must start with alnum)",
        )


def _join_subject(pillar: SubjectPillar, kind: str, parts: tuple[str, ...]) -> str:
    _validate_part(kind, field="kind")
    for i, p in enumerate(parts):
        _validate_part(p, field=f"parts[{i}]")
    segments = [pillar, kind, *parts]
    return ".".join(segments)


# ──────────────────────────────────────────────────────────────────────────
# 既知の kind (pillar ごと)
# ──────────────────────────────────────────────────────────────────────────

# 各 pillar が生成しうる kind 集合。実装ミス (例: loop に ``personal`` を
# 生成する) を早期検出するための allowlist。
LOOP_KINDS: frozenset[str] = frozenset({
    "task",
    "progress",
    "failure",
    "artifact",
})
"""EvorefLoop owned subject の ``<kind>`` 集合 (docs/c_05 §21.3)。"""

LEARN_KINDS: frozenset[str] = frozenset({
    "policy",
    "fewshot",
    "metric",
    # PolicyAdjuster 由来の集約失敗パターン。FactType
    # ``learned_failure_pattern`` に対応するが subject kind は短く
    # ``failure_pattern`` を採用する (Loop owned の ``loop.failure.*`` とは
    # prefix が異なるため衝突しない)。
    "failure_pattern",
})
"""EvorefLearn owned subject の ``<kind>`` 集合 (docs/c_05 §21.3)。"""

MEM_KINDS: frozenset[str] = frozenset({
    "personal",
    "world",
    "preference",
    "emotion",
    "opinion",
    "belief",
    "decision",
    "commitment",
    "project",
    "coding",
    "coding_task",
    "model",
})
"""EvorefMem owned subject の ``<kind>`` 集合。

task FactType に対応する subject kind。owner が ``mem`` であるため
``mem.coding_task.*`` 名前空間に配置する (設計書 §7.2 に準拠)。
"""


def _assert_kind(pillar: SubjectPillar, kind: str, allowlist: frozenset[str]) -> None:
    if kind not in allowlist:
        raise SubjectNamespaceError(
            f"{pillar!r} subject does not allow kind={kind!r}; "
            f"expected one of {sorted(allowlist)}",
        )


# ──────────────────────────────────────────────────────────────────────────
# subject 生成 API
# ──────────────────────────────────────────────────────────────────────────


def make_loop_subject(kind: str, *parts: str) -> str:
    """EvorefLoop 所有 subject (``loop.<kind>.<parts...>``) を生成する。

    許容 ``<kind>``: :data:`LOOP_KINDS` (``task`` / ``progress`` / ``failure`` /
    ``artifact``)。

    例:
        >>> make_loop_subject("task", "refactor-auth-42")
        'loop.task.refactor-auth-42'
        >>> make_loop_subject("failure", "a1b2c3d4e5f6")
        'loop.failure.a1b2c3d4e5f6'

    Raises:
        SubjectNamespaceError: kind が allowlist 外、または parts に不正文字。
    """
    _assert_kind("loop", kind, LOOP_KINDS)
    return _join_subject("loop", kind, parts)


def make_learn_subject(kind: str, *parts: str) -> str:
    """EvorefLearn 所有 subject (``learn.<kind>.<parts...>``) を生成する。

    許容 ``<kind>``: :data:`LEARN_KINDS` (``policy`` / ``fewshot`` / ``metric``)。

    例:
        >>> make_learn_subject("policy", "coding", "search", "top_k")
        'learn.policy.coding.search.top_k'
        >>> make_learn_subject("fewshot", "coding", "abc123")
        'learn.fewshot.coding.abc123'

    Raises:
        SubjectNamespaceError: kind が allowlist 外、または parts に不正文字。
    """
    _assert_kind("learn", kind, LEARN_KINDS)
    return _join_subject("learn", kind, parts)


def make_mem_subject(kind: str, *parts: str) -> str:
    """EvorefMem 所有 subject (``mem.<kind>.<parts...>``) を生成する。

    許容 ``<kind>``: :data:`MEM_KINDS` (``personal`` / ``world`` / ``preference`` /
    ``emotion`` / ``opinion`` / ``belief`` / ``decision`` / ``commitment`` /
    ``project`` / ``coding`` / ``coding_task`` / ``model``)。

    例:
        >>> make_mem_subject("personal", "favorite-language")
        'mem.personal.favorite-language'
        >>> make_mem_subject("coding_task", "my-project", "abc123def456")
        'mem.coding_task.my-project.abc123def456'

    Raises:
        SubjectNamespaceError: kind が allowlist 外、または parts に不正文字。
    """
    _assert_kind("mem", kind, MEM_KINDS)
    return _join_subject("mem", kind, parts)


# ──────────────────────────────────────────────────────────────────────────
# subject 検証
# ──────────────────────────────────────────────────────────────────────────


def validate_subject_namespace(subject: str) -> SubjectPillar | None:
    """``subject`` の prefix が 3 pillar namespace のいずれに属するかを判定する。

    Returns:
        ``"loop"`` / ``"learn"`` / ``"mem"``: pillar prefix にマッチした場合
        ``None``: pillar prefix を持たない (自然文 subject)

    3 pillar namespace は全面適用済。本関数は検出用途
    (index 分類 / 整合性モニタ) と自然文 subject (pillar prefix を持たない)
    の検出に使う。

    Raises:
        SubjectNamespaceError: ``subject`` が str でない / 空。
    """
    if not isinstance(subject, str):
        raise SubjectNamespaceError(
            f"subject must be str, got {type(subject).__name__}",
        )
    if not subject:
        raise SubjectNamespaceError("subject must not be empty")
    match = _SUBJECT_PREFIX_RE.match(subject)
    if match is None:
        return None
    pillar = match.group(1)
    # mypy/runtime narrowing: match pattern guarantees membership
    return pillar  # type: ignore[return-value]


def is_pillar_subject(subject: str) -> bool:
    """``subject`` が 3 pillar namespace のいずれかに属するか (真偽)。"""
    try:
        return validate_subject_namespace(subject) is not None
    except SubjectNamespaceError:
        return False


__all__ = [
    "ALL_SUBJECT_PILLARS",
    "LEARN_KINDS",
    "LOOP_KINDS",
    "MEM_KINDS",
    "SubjectNamespaceError",
    "SubjectPillar",
    "is_pillar_subject",
    "make_learn_subject",
    "make_loop_subject",
    "make_mem_subject",
    "validate_subject_namespace",
]
