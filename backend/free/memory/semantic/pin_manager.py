"""

EvorefMem 統合仕様 における Pin 機能の実装

設計原則:
- Pin は永続: ``unpin`` されるまで消えない (FadeMem 半減期の対象外)
- Pin 数は無制限 (config 由来上限を持たない)
- ``pin_locked_until`` で「期限まで unpin 不可」のロックを表現できる
  (デフォルトは ``None`` で常時 unpin 可)
- Tier +1.0 ボーナスは MemoryInjector 側 で計上する
- 永続 store への書き込みは `SemanticFactStore.update_fact` 経由で
  (索引・JSONL の整合を維持)

本モジュールは純粋関数の集合であり、I/O は ``SemanticFactStore`` に
委譲する。API / CLI / sleep-time から共通に呼ばれる。
"""

from __future__ import annotations

import time

from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.semantic.pin_manager")


class PinLockedError(RuntimeError):
    """``pin_locked_until`` が未来のため unpin できない場合に送出"""

    def __init__(self, fact_id: str, locked_until: float) -> None:
        super().__init__(
            f"fact {fact_id} is pin-locked until {locked_until:.0f}",
        )
        self.fact_id = fact_id
        self.locked_until = locked_until


def pin_fact(
    store: SemanticFactStore,
    fact_id: str,
    *,
    lock_duration_s: float | None = None,
    now: float | None = None,
) -> SemanticFact:
    """既存ファクトを pin する。

    Args:
        store: 対象スコープのストア
        fact_id: ファクト ID
        lock_duration_s: ロック秒数。``None`` ならロック無し (即時 unpin 可)。
            正の値を指定すると ``pin_locked_until = now + lock_duration_s``
        now: 現在時刻 (テスト用)。省略時は ``time.time()``

    Returns:
        更新後の ``SemanticFact``
    """
    fact = store.get_fact(fact_id)
    if fact is None:
        raise KeyError(f"fact not found: {fact_id}")
    ts = time.time() if now is None else now
    locked_until: float | None
    if lock_duration_s is not None and lock_duration_s > 0:
        locked_until = ts + float(lock_duration_s)
    else:
        locked_until = None
    updated = store.update_fact(
        fact_id,
        pinned=True,
        pin_locked_until=locked_until,
    )
    logger.info(
        "pin_fact: id=%s scope=%s locked_until=%s",
        fact_id, fact.scope, locked_until,
    )
    return updated


def unpin_fact(
    store: SemanticFactStore,
    fact_id: str,
    *,
    force: bool = False,
    now: float | None = None,
) -> SemanticFact:
    """ファクトの pin を解除する。

    ``pin_locked_until`` が未来の場合 ``PinLockedError`` を送出する。
    ``force=True`` なら無視して解除する。
    """
    fact = store.get_fact(fact_id)
    if fact is None:
        raise KeyError(f"fact not found: {fact_id}")
    ts = time.time() if now is None else now
    if not force and fact.pin_locked_until and fact.pin_locked_until > ts:
        raise PinLockedError(fact_id, fact.pin_locked_until)
    updated = store.update_fact(
        fact_id,
        pinned=False,
        pin_locked_until=None,
    )
    logger.info("unpin_fact: id=%s scope=%s force=%s", fact_id, fact.scope, force)
    return updated


def list_pinned(store: SemanticFactStore) -> list[SemanticFact]:
    """ストア内の pinned ファクトを accessed_at 降順で返す"""
    facts = store.pinned_facts()
    facts.sort(key=lambda f: f.accessed_at, reverse=True)
    return facts
