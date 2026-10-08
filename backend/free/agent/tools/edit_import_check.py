"""``apply_diff`` 成功後の import 警告 (docs/f_10 §8.1-3)。

編集で新しく生じた import エラーだけを、ツール結果の末尾へ 1 行足す。取り消し・作り直し・
ブロックはしない。決定論 (LLM 不使用)。編集前の内容と編集後の内容を同じ import スモークに
掛けて差を取るので、元から壊れているプロジェクトには偽の警告を出さない。
時間切れ・検査できない・生成物外の依存の欠落・副作用の恐れは、警告にせずログだけ残して見送る。
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("agent.tools.edit_import_check")

#: 検査に載せる同じディレクトリの ``.py`` の上限 (編集したファイルを含む)
MAX_FILES = 30
_MAX_MESSAGE_CHARS = 300
#: 実行ごとに変わる一時フォルダの場所 (``ImportError`` の末尾の括弧)。前後の比較と警告文から外す
_SMOKE_PATH_RE = re.compile(r" \([^()]*evoref_smoke_[^()]*\)")


def _read_siblings(target: Path) -> dict[str, str]:
    """編集したファイルを除く同ディレクトリの ``.py`` を名前順に (上限まで) 読む。"""
    out: dict[str, str] = {}
    for sibling in sorted(target.parent.glob("*.py")):
        if len(out) >= MAX_FILES - 1:
            break
        if sibling.name == target.name or not sibling.is_file():
            continue
        try:
            out[sibling.name] = sibling.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def _errors_of(result, prefix: str) -> list[str]:
    """スモーク結果のうち編集したモジュール自身に帰属するエラー (``<module>: `` を除いた本文)。"""
    return [_SMOKE_PATH_RE.sub("", e[len(prefix):]) for e in result.errors if e.startswith(prefix)]


def new_import_error(target: Path, before: str, after: str, *, timeout_sec: float) -> str:
    """編集後にだけ現れた import エラー (編集したモジュール帰属の最初の 1 件)。無い・見送りなら空。"""
    from backend.free.generation import package_layout as pl
    from backend.free.generation.smoke_validator import run_import_smoke

    siblings = _read_siblings(target)
    before_map = {**siblings, target.name: before}
    after_map = {**siblings, target.name: after}

    effect = pl.usage_side_effect(before_map) or pl.usage_side_effect(after_map)
    if effect:
        logger.info("edit import check skipped: side effect (%s) in %s", effect, target.name)
        return ""

    prefix = f"{target.stem}: "
    after_result = run_import_smoke(after_map, timeout_sec=timeout_sec)
    if not after_result.completed or after_result.unchecked:
        logger.info("edit import check skipped: smoke did not complete for %s", target.name)
        return ""
    after_errors = _errors_of(after_result, prefix)
    if not after_errors:
        return ""

    before_result = run_import_smoke(before_map, timeout_sec=timeout_sec)
    if not before_result.completed or before_result.unchecked:
        logger.info("edit import check skipped: baseline smoke did not complete for %s", target.name)
        return ""
    before_errors = set(_errors_of(before_result, prefix))
    for error in after_errors:
        if error not in before_errors:
            return " ".join(error.split())[:_MAX_MESSAGE_CHARS]
    return ""


def edit_import_warning(file_path: str, before: str, after: str, *, timeout_sec: float) -> str:
    """警告の 1 行 (``\\nWarning: ...``)。警告が無ければ空文字。"""
    error = new_import_error(Path(file_path), before, after, timeout_sec=timeout_sec)
    if not error:
        return ""
    logger.info("edit import check found a new error in %s: %s", file_path, error)
    return f"\nWarning: import check found a new error after this edit: {error}"
