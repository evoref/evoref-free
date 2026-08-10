"""Step 9: history summary → SemMem decision / commitment 昇格

``sleep_update.SleepTimeWorker._step9_promote_summaries_to_semmem``
として実装された history 要約昇格ロジックを、Decision D7 に従い独立 module
に切り出したもの。

処理対象は ``HistoryManager.index.sessions`` のうち ``summary`` が設定済かつ
``promoted_to_semmem=False`` のエントリ。要約テキストを簡易ルールベースで
``decision`` / ``commitment`` の 2 型に分類し、
:class:`~backend.free.memory.semantic.store.SemanticFactStore` に書き込む。
スコープは create モードかつ project_id 既知であれば ``project:<id>``、
それ以外は ``global``。Subject namespace は 定義された
``mem.<type>.history.session.<id12>`` を用いる。

本 module は EvorefMem pillar 内部扱いのため SemanticFactStore を直接参照する。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from backend.free.core.session_mode import (
    is_create_mode,
    is_valid_session_mode,
    normalize_session_mode,
)
from backend.free.memory.notes.subject_ns import make_mem_subject
from backend.free.memory.types import (
    Provenance,
    SemanticFact,
    make_fact,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.history.history_manager import HistoryManager
    from backend.free.memory.semantic.store import SemanticFactStore

logger = get_logger("memory.sleep.promotion")


# ──────────────────────────────────────────────────────────────────────────
# commitment キーワード辞書のロード (YAML 外部化)
# ──────────────────────────────────────────────────────────────────────────
#
# shipped default: ``backend/free/memory/_defaults/triggers/classify_triggers.yaml``
# user override  : ``<triggers_dir>/classify_triggers.yaml``
#     (``triggers_dir`` は通常 ``local/triggers/``; ``.gitignore`` で除外)


_CLASSIFY_LOCK = threading.Lock()
_CLASSIFY_CACHE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}


def _load_classify_triggers_raw(
    path: str | Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``classify_triggers.yaml`` をロードして ``(ja_kw, en_kw)`` を返す。

    - ja_kw: 大文字小文字を区別して原文に対しマッチ
    - en_kw: ``summary.lower()`` に対しマッチ (従って YAML に小文字で書く)

    ファイル欠落 / パース失敗時は空 tuple を返し、呼び出し側では必ず
    ``decision`` に倒れる仕様 (キーワード無し = 非 commitment 扱い)。
    """
    p = Path(path)
    empty: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    if not p.exists():
        logger.warning(
            "classify_triggers file not found: %s — all summaries classified as decision", p,
        )
        return empty
    try:
        raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("classify_triggers load failed (%s): %s", p, exc)
        return empty
    if not isinstance(raw, dict):
        logger.warning("classify_triggers root is not mapping: %s", p)
        return empty

    commitment = raw.get("commitment") or {}
    if not isinstance(commitment, dict):
        return empty

    def _coerce(items: Any) -> tuple[str, ...]:
        if not isinstance(items, list):
            return ()
        return tuple(it for it in items if isinstance(it, str) and it)

    ja = _coerce(commitment.get("ja"))
    en = _coerce(commitment.get("en"))
    logger.info(
        "classify_triggers loaded: commitment.ja=%d, commitment.en=%d (from %s)",
        len(ja), len(en), p,
    )
    return ja, en


def _get_classify_triggers(
    path: str | Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """パスをキーとしたプロセス内キャッシュ取得。"""
    key = str(Path(path).resolve())
    with _CLASSIFY_LOCK:
        cached = _CLASSIFY_CACHE.get(key)
        if cached is not None:
            return cached
        loaded = _load_classify_triggers_raw(path)
        _CLASSIFY_CACHE[key] = loaded
        return loaded


def reset_classify_triggers_cache() -> None:
    """テスト用: キャッシュ全消去。"""
    with _CLASSIFY_LOCK:
        _CLASSIFY_CACHE.clear()


def resolve_classify_triggers_path(triggers_dir: str | Path | None = None) -> Path:
    """``classify_triggers.yaml`` の解決パスを返す (user override → default)。"""
    from backend.free.memory._defaults import resolve_trigger_file

    return resolve_trigger_file("classify_triggers.yaml", triggers_dir=triggers_dir)


def _default_classify_triggers() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _get_classify_triggers(resolve_classify_triggers_path(None))


def _commitment_keywords() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """モジュールレベルの ``classify_summary_type`` が参照する辞書。

    現状は常に package default を使う。user override が必要な場合は
    将来的に ``SleepTimeWorker`` / ``promote_history_to_semmem`` 経由で
    ``triggers_dir`` を注入する拡張余地を残してある。
    """
    return _default_classify_triggers()


def classify_summary_type(summary: str) -> str:
    """要約テキストを ``decision`` / ``commitment`` のどちらに分類するか判定する。

    commitment 的なキーワード (予定 / will / TODO 等) が含まれていれば
    ``commitment``、それ以外は ``decision`` とする。完全一致ではなく
    部分一致 (大文字小文字は英語側のみ非区別)。

    キーワード辞書は :func:`_commitment_keywords` が
    ``classify_triggers.yaml`` からロードする。

    Args:
        summary: セッション要約テキスト (空 / None 相当は ``decision`` 扱い)。

    Returns:
        ``"decision"`` または ``"commitment"``。
    """
    if not summary:
        return "decision"
    ja, en = _commitment_keywords()
    text = summary
    text_lower = summary.lower()
    for kw in ja:
        if kw in text:
            return "commitment"
    for kw in en:
        if kw in text_lower:
            return "commitment"
    return "decision"


def _build_subject(fact_type: str, session_id: str) -> str:
    """Step 9 promotion の subject を mem.* namespace で構築する

    Subject は ``mem.<decision|commitment>.history.session.<id12>`` 形式。
    """
    return make_mem_subject(
        fact_type, "history", "session", session_id[:12],
    )


def _resolve_scope(
    entry_mode: str,
    entry_project_id: str | None,
    fallback_project_id: str | None,
) -> tuple[str, str | None]:
    """セッション情報から ``(scope, project_id)`` を解決する。

    - ``is_create_mode(mode)`` かつ project_id 既知 → ``project:<id>``
    - それ以外 → ``global``

    Returns:
        ``(scope_str, project_id)`` ペア。project 未確定時は ``project_id=None``。
    """
    project_id = entry_project_id or (
        fallback_project_id if is_create_mode(entry_mode) else None
    )
    if is_create_mode(entry_mode) and project_id:
        return SemanticFact.make_project_scope(project_id), project_id
    return "global", project_id


def promote_history_to_semmem(
    history_manager: "HistoryManager",
    store_provider: Callable[[str], "SemanticFactStore | None"],
    *,
    current_project_id: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> int:
    """古い history 要約を SemMem に decision / commitment として昇格する。

    対象は ``index.sessions`` のうち ``summary`` 設定済かつ
    ``promoted_to_semmem=False`` のセッション。本関数はセッションごとに
    ``promoted_to_semmem=True`` のフラグを
    :meth:`HistoryManager.mark_promoted_to_semmem` 経由で永続化する。

    Args:
        history_manager: HistoryManager インスタンス。
        store_provider: ``scope`` 文字列を受けて
            :class:`SemanticFactStore` (または ``None``) を返すコールバック。
        current_project_id: scope 未確定時のフォールバック project_id。
        is_cancelled: キャンセル判定コールバック (``True`` で途中中断)。

    Returns:
        実際に昇格したセッション数。

    Note:
        本関数は store_provider の例外を ``warning`` でログに残すのみで、
        sleep-time 全体は止めない。HistoryManager は呼び出し側の責任で
        初期化済のものを渡す (初期化失敗は呼び出し側で吸収する)。
    """
    index = history_manager._load_index()
    promoted = 0
    for entry in index.sessions:
        if is_cancelled is not None and is_cancelled():
            break
        if entry.promoted_to_semmem:
            continue
        if not entry.summary:
            continue
        session = history_manager.get_session(entry.session_id)
        if session is None:
            continue

        fact_type = classify_summary_type(entry.summary)
        scope_str, project_id = _resolve_scope(
            entry.mode, session.project_id, current_project_id,
        )

        try:
            store = store_provider(scope_str)
        except Exception as exc:
            logger.warning(
                "Step 9 promotion: failed to obtain store %s: %s",
                scope_str, exc,
            )
            continue
        if store is None:
            continue

        subject = _build_subject(fact_type, entry.session_id)
        fact = make_fact(
            subject=subject,
            predicate="summary_of_session",
            object_=entry.summary,
            type=fact_type,
            scope=scope_str,
            mode_origin=normalize_session_mode(entry.mode),
            confidence=0.5,
        )
        fact.provenances = [
            Provenance(
                session_id=entry.session_id,
                mode=entry.mode if is_valid_session_mode(entry.mode) else None,
                project_id=project_id,
                source="assistant",
                captured_at=time.time(),
            ),
        ]
        try:
            store.add_fact(fact)
        except Exception as exc:
            logger.warning(
                "Step 9 promotion: failed to add fact for session %s: %s",
                entry.session_id, exc,
            )
            continue
        history_manager.mark_promoted_to_semmem(entry.session_id)
        promoted += 1

    if promoted:
        logger.info(
            "Step 9 promotion: promoted %d session summaries to SemMem",
            promoted,
        )
    return promoted


__all__ = [
    "classify_summary_type",
    "promote_history_to_semmem",
    "reset_classify_triggers_cache",
    "resolve_classify_triggers_path",
]
