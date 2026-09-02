"""structlog ベースの DebugLogger 共通基盤

``DebugLogger`` の 6 種 JSONL 出力 (``requests`` / ``rag`` / ``memory`` /
``learning`` / ``long_form`` / ``agent_trace``) を ``structlog`` の
processor チェーンに集約する。

公開 API (``DebugLogger.log_*``) は維持したまま、内部書き込みを本モジュー
ルの ``DebugLogSink.emit(category, payload)`` に委譲することで:

* ``trace_id`` 自動付与を processor 化し、``DebugLogger._write`` 内の
  ``get_trace_id()`` 直接呼び出しを廃止
* API キー / Bearer / メールアドレスの redaction を processor として共通適用
* 日付分割ファイル名 / 世代ローテ / retention は ``DebugLogSink`` 内部の
  ``_DebugFileSink`` に集約し、``DebugLogger`` から関心事を分離
* tenacity リトライ の ``before_sleep`` から structlog logger を
  直接呼び出す将来拡張に備える

JSONL 出力フォーマットは互換維持: structlog の ``event`` キーは出力しない。
``DebugLogger.log_*`` が組み立てた dict を、``trace_id`` を先頭に置きつつ
``json.dumps(..., ensure_ascii=False)`` でシリアライズした 1 行を書き出す。
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from structlog.types import EventDict, WrappedLogger

from backend.log_config import get_logger
from backend.trace_context import get_trace_id

_logger = get_logger("structlog_config")


# ---------------------------------------------------------------------------
# Processor: trace_id 自動付与
# ---------------------------------------------------------------------------


def _trace_id_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict  # noqa: ARG001
) -> EventDict:
    """``contextvars.ContextVar`` ベースの trace_id を event_dict 先頭に挿入する。

    旧 ``DebugLogger._write`` の ``data = {"trace_id": tid, **data}`` 相当。
    ``structlog.contextvars.merge_contextvars`` ではなく ``trace_context``
    モジュールの単一 ContextVar を直接参照する形にすることで、既存の
    ``set_trace_id`` / ``run_in_executor_with_context`` を
    変更せずに structlog 経由でも ``trace_id`` を取り出せるようにしている。
    """
    tid = get_trace_id()
    if tid:
        # event_dict の先頭に trace_id を置くため新しい dict を作って差し替える
        merged: EventDict = {"trace_id": tid}
        for key, value in event_dict.items():
            if key == "trace_id":
                continue
            merged[key] = value
        return merged
    return event_dict


# ---------------------------------------------------------------------------
# Processor: redaction (API キー / Bearer / メールアドレス)
# ---------------------------------------------------------------------------


# キー名ベースの完全マスク対象。case-insensitive 比較で扱うため小文字で持つ。
# キー名を追加。MCP 経由で `gh` CLI / GitHub API / クラウド連携が混入する
# 経路で、ペイロード dict に明示的なキーとしてトークンが乗るケースを完全
# マスクする。値ベースの regex (`_API_KEY_PREFIX_RE`) でカバーしきれない
# 任意フォーマットのトークンも、キー名で確実に redact できる。
_REDACT_KEYS_LOWER: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "x-api-key",
        "x_api_key",
        "bearer",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "client_secret",
        "anthropic-api-key",
        "openai-api-key",
        "github_token",
        "github_pat",
        "aws_access_key_id",
        "aws_secret_access_key",
        "slack_token",
        "gcp_api_key",
    }
)

# 値ベースの部分マスク (Bearer / API キー / メール)。
# Fine-grained PAT (``github_pat_*``)・AWS Access Key ID (``AKIA*``) を追加。
# 自由形式の文字列 (スタックトレース・``response_preview`` 等) に埋め込まれた
# トークンも検出できるようにする。各パターンの長さは GitHub / AWS の公開
# フォーマット仕様に基づく (Classic PAT は 36 文字以上、Fine-grained PAT は
# 82 文字以上、AWS Access Key ID は 16 文字)。``\b`` で境界を明示し、長い
# 識別子の中間に偶然マッチするケースを抑制する。
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]+")
_API_KEY_PREFIX_RE = re.compile(
    r"\b("
    r"(?:sk|pk|sk-ant)-[A-Za-z0-9_\-]{16,}"
    r"|ghp_[A-Za-z0-9]{36,}"
    r"|github_pat_[A-Za-z0-9_]{82,}"
    r"|AKIA[0-9A-Z]{16}"
    r")\b"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_REDACTED = "[REDACTED]"


def _redact_string(value: str) -> str:
    """文字列内の Bearer トークン / API キー / メールアドレスを置換する。"""
    redacted = _BEARER_RE.sub(f"Bearer {_REDACTED}", value)
    redacted = _API_KEY_PREFIX_RE.sub(_REDACTED, redacted)
    redacted = _EMAIL_RE.sub(_REDACTED, redacted)
    return redacted


#: 公開別名。ログ処理以外 (SemMem 索引ファクトの秘匿化) からも同じ規則で伏せる。
redact_string = _redact_string


def _redact_value(key: str, value: Any) -> Any:
    """key / value の組に対して redaction を適用する。

    ``key`` が ``_REDACT_KEYS_LOWER`` にマッチする場合は値全体を ``[REDACTED]``
    に置換、それ以外は文字列内の Bearer / API キー / メールパターンのみ
    マスクする。dict / list はキー単位で再帰的に走査する。
    """
    if key.lower() in _REDACT_KEYS_LOWER and value is not None:
        return _REDACTED
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        return {k: _redact_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    return value


def _redaction_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict  # noqa: ARG001
) -> EventDict:
    """event_dict 全体に redaction を適用する。"""
    return {k: _redact_value(k, v) for k, v in event_dict.items()}


# ---------------------------------------------------------------------------
# Processor: JSONL レンダラ (event キーは出力しない)
# ---------------------------------------------------------------------------


def _jsonl_renderer(
    logger: WrappedLogger, method_name: str, event_dict: EventDict  # noqa: ARG001
) -> str:
    """structlog の event_dict を JSONL 1 行 (``str``) に変換する。

    本実装は ``structlog.wrap_logger`` ではなく ``_render`` で processor
    を直接実行する設計のため、structlog 標準のメタキー (``level`` /
    ``logger`` / ``_record`` 等) は付加されない。``event`` は
    ``DebugLogger.log_agent_trace_event`` 等が業務上の値 (``"begin"`` /
    ``"step"`` / ``"end"``) として使うキーなので、特別扱いせずそのまま
    シリアライズする
    """
    return json.dumps(dict(event_dict), ensure_ascii=False)


# ---------------------------------------------------------------------------
# File sink: 日付分割 + 世代ローテ + retention
# ---------------------------------------------------------------------------


class _DebugFileSink:
    """カテゴリ別 JSONL ファイルへの追記とローテーションを担う sink。

    ``DebugLogger._write`` / ``_rotate`` / ``_cleanup_old_logs`` から
    機能を移管したもの。``structlog`` の最終 processor (`_jsonl_renderer`)
    が返した文字列を受け取り、日付付きファイルへ追記する。
    """

    def __init__(
        self,
        log_dir: Path,
        max_log_mb: int,
        max_log_generations: int,
        log_retention_days: int,
    ) -> None:
        self.log_dir = log_dir
        self.max_log_mb = max_log_mb
        self.max_log_generations = max_log_generations
        self.log_retention_days = log_retention_days
        # ローテーション競合シリアライズ用ロック (旧 DebugLogger._rotate_lock 相当)。
        # 「サイズ判定 → rename 群」を 1 つのクリティカルセクションに束ねる。
        self._rotate_lock = threading.Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_logs()

    def write_line(self, category: str, line: str) -> None:
        """``<category>_YYYY-MM-DD.jsonl`` に 1 行追記する。"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dated_filename = f"{category}_{today}.jsonl"
        path = self.log_dir / dated_filename

        with self._rotate_lock:
            try:
                if (
                    path.exists()
                    and path.stat().st_size > self.max_log_mb * 1024 * 1024
                ):
                    self._rotate(path)
            except OSError as exc:
                _logger.warning(
                    "Debug log rotation failed for %s: %s", path.name, exc,
                )

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            _logger.warning("Failed to write debug log: %s", exc)

    def _rotate(self, path: Path) -> None:
        """世代ローテーション: ``.1`` ... ``.N`` まで保持し、最古を削除する。

        各 rename / unlink は個別に try/except し、Windows で他プロセスが
        オープン中のファイルに対する PermissionError でも残りの世代シフト
        は継続する (旧 ``DebugLogger._rotate`` と同等)。
        """
        oldest = path.with_suffix(f".{self.max_log_generations}")
        if oldest.exists():
            try:
                oldest.unlink()
            except OSError as exc:
                _logger.warning(
                    "Failed to remove oldest generation %s: %s",
                    oldest.name, exc,
                )

        for gen in range(self.max_log_generations - 1, 0, -1):
            src = path.with_suffix(f".{gen}")
            dst = path.with_suffix(f".{gen + 1}")
            if not src.exists():
                continue
            try:
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
            except OSError as exc:
                _logger.warning(
                    "Failed to shift generation %s -> %s: %s",
                    src.name, dst.name, exc,
                )

        dst1 = path.with_suffix(".1")
        try:
            if dst1.exists():
                dst1.unlink()
            path.rename(dst1)
        except OSError as exc:
            _logger.warning(
                "Failed to promote current log to .1 for %s: %s",
                path.name, exc,
            )

    def _cleanup_old_logs(self) -> None:
        """retention_days を超えた古いログファイルを削除する (旧 ``_cleanup_old_logs``)."""
        if self.log_retention_days <= 0:
            return
        try:
            now = datetime.now(timezone.utc)
            for log_file in self.log_dir.glob("*.jsonl*"):
                try:
                    mtime = datetime.fromtimestamp(
                        log_file.stat().st_mtime, tz=timezone.utc,
                    )
                    age_days = (now - mtime).days
                    if age_days > self.log_retention_days:
                        log_file.unlink()
                        _logger.debug(
                            "Deleted old debug log: %s (age=%dd)",
                            log_file.name, age_days,
                        )
                except OSError:
                    pass
        except OSError as exc:
            _logger.warning("Failed to cleanup old logs: %s", exc)


# ---------------------------------------------------------------------------
# Public API: DebugLogSink
# ---------------------------------------------------------------------------


# レベル限定で書き出される自己進化用因果ログ)。``DebugLogger`` 側で
# enabled フラグを gate するため、``DebugLogSink`` は category 名の
# 妥当性のみ検証する。
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "requests", "rag", "memory", "learning", "long_form", "agent_trace",
        "decision", "outcome",
    }
)


# `evolve` 限定にするか議論したが、`debug` / `investigate` でログを併用する
# loop driver / 解析スクリプトが ``schema_version`` 欠損を分岐する複雑さを
# 避けるため全レベルで付与する設計を採る (容量 ~20 byte / entry の追加コスト)。
_CURRENT_SCHEMA_VERSION = 1


def _schema_version_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict  # noqa: ARG001
) -> EventDict:
    """全 JSONL エントリに ``schema_version`` を付与する

    ``_trace_id_processor`` の直後・``_redaction_processor`` の前に挿入され、
    キー順は ``trace_id`` (任意) → ``schema_version`` → 残り元エントリ。
    既に ``schema_version`` が指定されている場合は呼出側の値を尊重する
    (将来の schema 移行で個別 entry の version を変える余地を残す)。
    """
    if "schema_version" in event_dict:
        return event_dict
    # trace_id が処理直前で先頭に挿入されているため、それを保持しつつ
    # schema_version を 2 番目に挿入した辞書を返す。
    merged: EventDict = {}
    inserted = False
    for key, value in event_dict.items():
        merged[key] = value
        if key == "trace_id" and not inserted:
            merged["schema_version"] = _CURRENT_SCHEMA_VERSION
            inserted = True
    if not inserted:
        # trace_id が無い場合は冒頭に schema_version を挿入する
        return {"schema_version": _CURRENT_SCHEMA_VERSION, **event_dict}
    return merged


_DEFAULT_PROCESSORS: tuple = (
    _trace_id_processor,
    _schema_version_processor,
    _redaction_processor,
    _jsonl_renderer,
)


def _render(payload: Mapping[str, Any], processors: tuple = _DEFAULT_PROCESSORS) -> str:
    """processor チェーンを順に実行し、最終的に JSONL 1 行 (``str``) を返す。

    structlog の processor シグネチャ ``(logger, method_name, event_dict)``
    に従う関数を直列に実行する。最終 processor が ``str`` を返した時点で
    ループを抜けて結果を返す。``_jsonl_renderer`` が常に ``str`` を返すた
    め、デフォルト構成では必ずループ末尾の例外には到達しない。
    """
    event_dict: EventDict = dict(payload)
    for proc in processors:
        result = proc(None, "info", event_dict)  # type: ignore[arg-type]
        if isinstance(result, str):
            return result
        event_dict = result
    raise RuntimeError(
        "structlog processor chain did not produce a string output; "
        "the last processor must return str (got dict)"
    )


class DebugLogSink:
    """``DebugLogger`` から呼ばれる構造化ログ sink

    ``emit(category, payload)`` 1 メソッドで:

    1. payload を structlog 互換 processor チェーン
       (``_trace_id_processor`` → ``_redaction_processor`` → ``_jsonl_renderer``)
       に通す
    2. レンダリング結果 (JSONL 1 行) を ``_DebugFileSink`` 経由で
       ``<category>_YYYY-MM-DD.jsonl`` に追記する

    ``DebugLogger`` 側は本クラスを 1 個だけ生成し、各 ``log_*`` メソッド
    から ``emit("requests", entry)`` のように呼び出す。

    構成の単純さを優先して ``structlog.wrap_logger`` には依存せず、processor
    関数を ``_render`` で直列実行する形を採る。``processors`` キーワードで
    任意の processor チェーンを差し替えられるため、テスト時に redaction
    のみ単独検証することも可能。
    """

    def __init__(
        self,
        log_dir: Path,
        max_log_mb: int,
        max_log_generations: int,
        log_retention_days: int,
        processors: tuple = _DEFAULT_PROCESSORS,
    ) -> None:
        self._file_sink = _DebugFileSink(
            log_dir=log_dir,
            max_log_mb=max_log_mb,
            max_log_generations=max_log_generations,
            log_retention_days=log_retention_days,
        )
        self._processors = processors

    @property
    def log_dir(self) -> Path:
        return self._file_sink.log_dir

    def emit(self, category: str, payload: Mapping[str, Any]) -> None:
        """``category`` 別 JSONL に payload を追記する。

        Args:
            category: ``requests`` / ``rag`` / ``memory`` / ``learning`` /
                ``long_form`` / ``agent_trace`` のいずれか。
            payload: JSONL に書き出す dict。``trace_id`` / redaction は
                processor チェーンが付与・適用するため、呼出側は元の
                ペイロードのまま渡せばよい。
        """
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"Unknown debug log category: {category!r}. "
                f"Expected one of {sorted(_VALID_CATEGORIES)}"
            )
        rendered = _render(payload, self._processors)
        self._file_sink.write_line(category, rendered)
