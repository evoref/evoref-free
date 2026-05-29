"""LearnedPattern の JSON 永続化

`backend.free.agent.learned_patterns.LearnedPatternStore` からドメインロジックを
分離するための infra 層。`LearnedPatternRepository` は `LearnedPattern` の
シリアライズ / デシリアライズと JSON ファイル I/O のみを担い、
ドメインルール (重み付け、スコアリング、ストップワード判定、抽出判定) は
持たない。

レイヤー責務:
- `LearnedPatternStore`       — ドメイン (重み更新、マッチング、抽出、上限制御)
- `LearnedPatternRepository`  — インフラ (JSON 永続化、ファイル I/O)

このため `LearnedPatternRepository` は import 時に `LearnedPatternStore` を
参照せず、`LearnedPattern` dataclass のみに依存する (循環依存防止 +
単体テスト可能性確保)。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from backend.free.agent.learned_patterns_types import LearnedPattern
from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("agent.learned_pattern_store")


class LearnedPatternRepository:
    """LearnedPattern の純粋な永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    @staticmethod
    def serialize(patterns: dict[str, LearnedPattern]) -> list[dict]:
        """`LearnedPattern` 辞書を JSON-serializable な list[dict] に変換する。"""
        return [_pattern_to_dict(p) for p in patterns.values()]

    @staticmethod
    def deserialize(data: list[dict]) -> dict[str, LearnedPattern]:
        """list[dict] から `LearnedPattern` 辞書を再構築する。

        欠損フィールドはデフォルト値で埋め、後方互換 (古い JSON フォーマット)
        を維持する。キーはパターンの keyword を lower-case 化したもの
        (空 keyword のエントリは無視)。ドメインルール (ストップワード除外等)
        は呼び出し側で適用する。
        """
        patterns: dict[str, LearnedPattern] = {}
        for d in data:
            pattern = _pattern_from_dict(d)
            key = pattern.keyword.lower()
            if not key:
                continue
            patterns[key] = pattern
        return patterns

    @staticmethod
    def save(patterns: dict[str, LearnedPattern], path: str | Path) -> None:
        """`patterns` を JSON ファイルに書き出す。親ディレクトリは自動作成。"""
        path = Path(path)
        data = LearnedPatternRepository.serialize(patterns)
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved %d learned patterns to %s", len(data), path)

    @staticmethod
    def load(path: str | Path) -> dict[str, LearnedPattern] | None:
        """JSON ファイルから `LearnedPattern` 辞書を読み込む。

        ファイルが存在しない場合、または JSON のパースに失敗した場合は
        `None` を返す (空辞書とは区別する)。呼び出し側は `None` を
        「ファイル未存在 / 破損 = 既存状態を保持」と解釈できる。
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load learned patterns from %s: %s", path, exc)
            return None
        if not isinstance(data, list):
            logger.warning("Invalid learned patterns format (expected list) in %s", path)
            return None
        patterns = LearnedPatternRepository.deserialize(data)
        logger.info("Loaded %d learned patterns from %s", len(patterns), path)
        return patterns


# ──────────────────────────────────────────────────────────────────────────
# private serialize / deserialize helpers (純粋関数)
# ──────────────────────────────────────────────────────────────────────────


def _pattern_to_dict(pattern: LearnedPattern) -> dict:
    return asdict(pattern)


def _pattern_from_dict(d: dict) -> LearnedPattern:
    return LearnedPattern(
        keyword=d.get("keyword", ""),
        category=d.get("category", "correction"),
        weight=float(d.get("weight", 0.5)),
        hit_count=int(d.get("hit_count", 0)),
        source_count=int(d.get("source_count", 1)),
        first_seen=float(d.get("first_seen", 0.0)),
        last_seen=float(d.get("last_seen", 0.0)),
        last_hit=float(d.get("last_hit", 0.0)),
    )
