"""品質ゲートの結果データ型

クリエイトの staged パイプライン (``loop/staged/``) の test 工程と、失敗の記録
(``failure_note``) が共有する結果の型だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    """単一品質ゲートの実行結果。

    ``ok=True`` かつ ``skipped=False`` で「合格」。``skipped=True`` の場合は
    ゲート自体が無効化されているか、依存ツールが見つからずに実行を回避した
    ことを示し、合格にも失敗にもカウントしない。
    """

    name: str
    ok: bool
    skipped: bool
    returncode: int | None
    duration_ms: int
    stdout_tail: str
    stderr_tail: str
    error: str | None = None
    skip_reason: str | None = None

    def is_failure(self) -> bool:
        """ゲート実行が失敗とみなされる (= 後段のリトライ判定対象) か"""
        return not self.ok and not self.skipped


@dataclass(frozen=True)
class QualityGateOutcome:
    """品質ゲート群の集約結果"""

    ok: bool
    results: tuple[GateResult, ...]
    failed: tuple[str, ...]
    skipped: tuple[str, ...]


__all__ = [
    "GateResult",
    "QualityGateOutcome",
]
