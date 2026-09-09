"""一過性の補助タスク失敗を永続状態にしないための共通バックオフ (Theme 2 / G-03)。

``personal_fact_curator`` / ``assertion_curator`` / ``url_curator`` はいずれも
「補助タスクを呼ぶ → 結果を検証して書く → 冪等マーカー (``*_curated_at``) を
立てる」という同じ形をしている。従来はこの 3 者とも例外を握り潰して
「LLM が答えた」場合と同じ扱いでマーカーを立てていたため、一過性の aux
timeout (チャットとの GPU 併走由来、``AuxClient.resolve_effective_timeout`` の
校正が縮みすぎるケースを含む) が **その場で永久に再試行不能** になっていた
(実インシデント: 2026-09-08 ライブ監査 G-03)。

本モジュールは :class:`~backend.free.memory.pipeline.conflict_resolver.
ConflictResolver` の ``conflict_fail_count`` / ``conflict_cooldown_until``
(quarantine backoff) と同じ形を、curator 3 者が共用できるよう汎化したもの。
ノート側は 1 フィールド ``curation_failures: dict[str, dict]`` だけを持ち、
key はどの curator の失敗かを表す (aux purpose 名をそのまま使う:
``"personal_fact_split"`` / ``"assertion_naming"`` / ``"url_relevance_score"``)。

呼出側の約束:

- 補助タスクが **答えた** (中身が空でも / 検証で全部落ちても) 場合だけ
  ``*_curated_at`` を立てて :func:`clear_failure` を呼ぶ。
- 補助タスクが **例外を投げた** 場合は ``*_curated_at`` を立てず、
  :func:`record_transient_failure` を呼ぶ。``AuxTimeoutError.contended``
  (チャットとの GPU 併走で一時的に遅かっただけ) が立っていれば
  ``counts=False`` を渡し、カウントを消費しない
  (``AuxClient._record_success`` が競合サンプルを較正から弾くのと同じ理由)。
- 候補選定では :func:`in_cooldown` で cooldown 中のノートを外す。
"""

from __future__ import annotations

from typing import Any

#: 一過性失敗がこの回数に達したら quarantine cooldown へ入る。
#: ``ConflictResolver._FAIL_QUARANTINE_THRESHOLD`` と同じ値。
FAIL_QUARANTINE_THRESHOLD = 3

#: cooldown の長さ。恒久 ban ではなく一定時間後に再試行させる
#: (``ConflictResolver._COOLDOWN_SECONDS`` と同じ値)。
COOLDOWN_SECONDS = 6 * 3600.0


def in_cooldown(note: Any, key: str, now: float) -> bool:
    """``note.curation_failures[key]`` が cooldown 中か (純粋関数)。"""
    failures = getattr(note, "curation_failures", None) or {}
    entry = failures.get(key)
    if not entry:
        return False
    until = entry.get("cooldown_until")
    return until is not None and until > now


def record_transient_failure(
    note: Any, key: str, now: float, *, counts: bool = True,
) -> None:
    """一過性失敗 (補助タスクの例外) を記録する。``*_curated_at`` は立てない。

    ``counts=False`` はチャット併走由来の contended timeout。この purpose に
    必要な予算とは無関係な遅さなので、quarantine の閾値カウントには含めない
    — 何もせず戻る (次サイクルでそのまま再試行させる)。
    """
    if not counts:
        return
    failures = dict(getattr(note, "curation_failures", None) or {})
    entry = dict(failures.get(key) or {"count": 0, "cooldown_until": None})
    entry["count"] = int(entry.get("count") or 0) + 1
    if entry["count"] >= FAIL_QUARANTINE_THRESHOLD:
        entry["cooldown_until"] = now + COOLDOWN_SECONDS
        entry["count"] = 0
    failures[key] = entry
    note.curation_failures = failures


def clear_failure(note: Any, key: str) -> None:
    """成功 (補助タスクが答えた) 時に、当該 key の失敗記録を消す。"""
    failures = getattr(note, "curation_failures", None)
    if not failures or key not in failures:
        return
    failures = dict(failures)
    failures.pop(key, None)
    note.curation_failures = failures


__all__ = [
    "COOLDOWN_SECONDS",
    "FAIL_QUARANTINE_THRESHOLD",
    "clear_failure",
    "in_cooldown",
    "record_transient_failure",
]
