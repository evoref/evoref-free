"""ログ設定

develop モードを 3 段階 (``debug`` / ``investigate`` / ``evolve``)
に再設計。``setup_logging(develop_level, project_root)`` は
``develop_level`` から backend.log のレベルとコンソール出力 ON/OFF を導出
する。``debug.enabled`` 設定キーは廃止。

| develop_level | backend.log | console (stderr) |
|---------------|-------------|------------------|
| off           | INFO+       | OFF              |
| debug         | DEBUG+      | ON               |
| investigate   | DEBUG+      | ON               |
| evolve        | DEBUG+      | OFF (loop 実行を妨げない) |
"""

from __future__ import annotations

import io
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

# develop モードレベル (SSOT)。``app_state.DevelopLevel`` と
# 同じ Literal だが、``log_config`` は循環 import を避けるため独立で型を
# 持つ (``app_state.py`` は ``fastapi.Request`` を import するため
# ``log_config`` 側へ落とさない)。``app_state.DevelopLevel`` は本モジュール
# を re-export する形で同一型を共有する。
DevelopLevel = Literal["off", "debug", "investigate", "evolve"]


def _make_utf8_stream() -> io.TextIOWrapper:
    """stderr を UTF-8 でラップしたストリームを返す（Windows cp932 文字化け対策）"""
    try:
        return open(
            sys.stderr.fileno(), "w",
            encoding="utf-8", errors="replace", closefd=False,
        )
    except (AttributeError, OSError):
        return sys.stderr  # type: ignore[return-value]


def _make_rotating_handler(
    path: Path,
    formatter: logging.Formatter,
    max_bytes: int,
    *,
    level: int | None = None,
) -> RotatingFileHandler | None:
    """ローテーションハンドラを作る。開けなければ ``None`` を返す。

    ``RotatingFileHandler`` は ``delay`` 未指定だとコンストラクタで即座に
    ファイルを開くため、開けないと ``setup_logging()` が例外を上げてアプリが
    **起動不能** になる。Windows では削除保留 (delete pending) のファイルが
    同名の再作成をブロックするため、``scripts/reset_local_data.py``
    (UI の初期化ボタン) が ``local/`` を wipe した直後にこれを踏みうる
    (2026-08-14 に実機で再現: ``PermissionError`` → ``Application startup failed``)。

    ログが 1 本落ちることとアプリが起動しないことは重大さが桁違いなので、
    開けない場合は stderr に警告して縮退する。
    """
    try:
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=3, encoding="utf-8",
        )
    except OSError as e:
        print(
            f"[log_config] WARNING: cannot open log file {path} ({e}); "
            f"continuing without this handler",
            file=sys.stderr,
        )
        return None
    handler.setFormatter(formatter)
    # private セッションの発話は **出力の直前** で伏せる。発話を書く logger
    # 呼び出しは router / tool_call_judge / search_pipeline / self_rag_judge /
    # bm25_retriever / injector / deliberative など十数箇所に散っており、
    # 個別修正は必ず漏れる。子ロガーから伝播した record も必ず通る handler
    # 側に付ける (logger の filter は伝播 record を通らない)。
    handler.addFilter(PrivateContentFilter())
    if level is not None:
        handler.setLevel(level)
    return handler


def _has_rotating_file_handler(lg: logging.Logger, target_path: Path) -> bool:
    """指定パスを baseFilename とする RotatingFileHandler が既に登録済みかを判定する"""
    return any(
        isinstance(h, RotatingFileHandler)
        and Path(getattr(h, "baseFilename", "")).resolve() == target_path
        for h in lg.handlers
    )


def _has_plain_stream_handler(lg: logging.Logger) -> bool:
    """RotatingFileHandler 以外の StreamHandler（コンソール等）が登録済みかを判定する"""
    return any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, RotatingFileHandler)
        for h in lg.handlers
    )


#: 通常起動 (INFO+) の backend.log / learning.log ローテ閾値。
_LOG_MAX_BYTES_DEFAULT = 5 * 1024 * 1024

#: develop モード (DEBUG+) のローテ閾値。
#:
#: DEBUG 出力は INFO の 10 倍以上の行数になるため、5MB × 3 世代では実測で
#: **30 分ぶんしか残らない** (2026-08-07 ライブ監査: 1 時間のセッションを
#: 調べようとした時点で開始 40 分ぶんが既にローテで消えており、観測した
#: ``stream_timeout`` を backend.log と突き合わせられなかった)。
#: JSONL 側 (:mod:`backend.debug_logger` の ``_LEVEL_CONFIGS``) が develop
#: レベルに応じて 100〜500MB を確保しているのに対し、標準ログだけが据え置き
#: だったのが不整合。世代数は据え置き (3) で 1 ファイルの上限だけ上げる。
_LOG_MAX_BYTES_DEVELOP = 50 * 1024 * 1024


def _log_max_bytes(develop_level: DevelopLevel) -> int:
    """develop レベルに応じたローテ閾値を返す (純粋関数)。"""
    if develop_level == "off":
        return _LOG_MAX_BYTES_DEFAULT
    return _LOG_MAX_BYTES_DEVELOP


#: private セッションの発話をログ行から伏せるときの印。
PRIVATE_MASK = "[PRIVATE]"

#: 伏せる対象とみなす最短の断片長。
#:
#: ログ側は 3 通りの形で発話を書く: そのまま / ``query[:80]`` のように切り詰め /
#: **発話から抽出した値だけ** (``Quantity grounding injected: 口座番号 =
#: 5551234509876``)。接頭辞照合では 3 つ目が漏れるため、発話との **共通部分
#: 文字列** で照合する。短すぎる断片まで消すと無関係な行を壊すので下限を置く。
#: private ターンの間だけ効くので、多少過剰に伏せる方が安全側。
_PRIVATE_MIN_FRAGMENT = 8


class PrivateContentFilter(logging.Filter):
    """private セッションのユーザー発話を、出力の直前で ``[PRIVATE]`` へ伏せる。

    ``ChatRequest.private`` の契約は「LTM/SemMem/履歴ディスク永続化に書き込ま
    ない」だが、**ログはその経路に入っていない**ため素通りしていた。実測
    (2026-09-03 ライブ監査 T16): private セッションで話した口座番号が
    ``local/logs/backend.log`` に平文で残り、うち 2 行は ``[INFO]``
    (develop モード無しの通常運用でも出る)。

    発話を書く logger 呼び出しは router / tool_call_judge / search_pipeline /
    self_rag_judge / bm25_retriever / injector / deliberative など十数箇所に
    散っており、個別に直すと必ず漏れる。**出力の合流点** (handler の filter)
    で、そのターンの発話文字列を含む行を一括して伏せる。

    ログ側は切り詰めて書くので、完全一致ではなく **長い方から接頭辞** を
    探して置換する (``query[:80]`` / ``query[:60]`` / ``query[:40]`` の
    いずれにも当たる)。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        from backend.trace_context import get_private_texts, is_private

        if not is_private():
            return True
        secrets = [t for t in get_private_texts() if len(t) >= _PRIVATE_MIN_FRAGMENT]
        if not secrets:
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - 整形不能な record は触らない
            return True
        masked = message
        for secret in secrets:
            masked = _mask_private(masked, secret)
        if masked != message:
            record.msg = masked
            record.args = ()
        return True


def _mask_private(message: str, secret: str) -> str:
    """``message`` のうち ``secret`` と共通する部分文字列を伏せる。

    接頭辞ではなく共通部分文字列で見るのは、ログが発話から **抽出した値だけ**
    を書くことがあるため (``Quantity grounding injected: 口座番号 = 5551234509876``
    は発話の接頭辞ではない)。左から貪欲に最長一致を取り、``[PRIVATE]`` へ置く。
    """
    out: list[str] = []
    i = 0
    n = len(message)
    while i < n:
        best = 0
        # i から始まる最長の共通部分文字列を探す (下限に満たなければ採らない)
        limit = min(n - i, len(secret))
        for length in range(limit, _PRIVATE_MIN_FRAGMENT - 1, -1):
            if message[i:i + length] in secret:
                best = length
                break
        if best:
            out.append(PRIVATE_MASK)
            i += best
        else:
            out.append(message[i])
            i += 1
    return "".join(out)


def setup_logging(
    develop_level: DevelopLevel = "off",
    project_root: Path | None = None,
) -> None:
    """``develop_level`` に基づきログハンドラを構築する

    Args:
        develop_level: ``"off"`` は通常起動 (backend.log INFO+ 固定、コン
            ソール OFF)、``"debug"`` / ``"investigate"`` は backend.log
            DEBUG+ + コンソール ON、``"evolve"`` は backend.log DEBUG+ だが
            コンソール OFF (loop 実行を妨げないため)。
        project_root: ログディレクトリ (`local/logs/`) のベース。
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    # develop_level に応じた level / コンソール出力 ON/OFF
    # ``debug.enabled`` 設定キーは廃止 (config.yaml レベルで完全削除)。
    if develop_level == "off":
        level = logging.INFO          # 通常時: 起動完了等の動作確認 INFO ログを出す
        console_enabled = False
    else:
        level = logging.DEBUG
        # evolve は loop 自己学習向けで console を出すと loop 実行を妨げる
        console_enabled = develop_level in ("debug", "investigate")

    # ログディレクトリ作成
    log_dir = project_root / "local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = _log_max_bytes(develop_level)

    # ログフォーマット
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # ルートロガー設定
    root = logging.getLogger("backend")
    root.setLevel(level)

    # backend.log ハンドラ（複数回呼び出しでも重複登録しない）
    backend_log_path = (log_dir / "backend.log").resolve()
    if not _has_rotating_file_handler(root, backend_log_path):
        backend_handler = _make_rotating_handler(
            log_dir / "backend.log", formatter, max_bytes,
        )
        if backend_handler is not None:
            root.addHandler(backend_handler)

    # コンソール出力（debug / investigate のみ）— UTF-8 強制
    # 非 RotatingFileHandler の StreamHandler が既に登録済みなら追加しない
    if console_enabled and not _has_plain_stream_handler(root):
        console = logging.StreamHandler(stream=_make_utf8_stream())
        console.setFormatter(formatter)
        console.addFilter(PrivateContentFilter())
        console.setLevel(level)
        root.addHandler(console)

    # learning.log ハンドラ（学習サイクル Level 0.5〜2 専用）
    # 設計書 c_07_error_handling_and_observability.md §6.2 に従い独立ファイルへ出力する
    _setup_learning_logging(log_dir, formatter, level, max_bytes)


# 学習サイクル系ロガー（学習サイクル Level 0.5〜2 を learning.log に集約する対象）
# - backend.learning.*       : Free 版の学習サイクル
# - backend.pro.learning.*   : Pro 版の Level 2 トレーナー等
# - backend.memory.sleep_update : Sleep-time update（Level 0.5）
# - backend.optimizer.*      : Level 1 進化アルゴリズム (prompt/embed_instruction/spsa)
_LEARNING_LOGGER_NAMES: tuple[str, ...] = (
    "backend.learning",
    "backend.pro.learning",
    "backend.memory.sleep_update",
    "backend.optimizer",
)


def _setup_learning_logging(
    log_dir: Path,
    formatter: logging.Formatter,
    level: int,
    max_bytes: int = _LOG_MAX_BYTES_DEFAULT,
) -> None:
    """learning.log ハンドラを学習サイクル系ロガーに登録する

    backend.log への重複出力を避けるため対象ロガーは ``propagate=False`` とする。
    複数回呼び出しても同一ファイルへのハンドラが重複しないようガードする。
    """
    target_path = (log_dir / "learning.log").resolve()

    target_loggers = [logging.getLogger(n) for n in _LEARNING_LOGGER_NAMES]
    if all(_has_rotating_file_handler(lg, target_path) for lg in target_loggers):
        return  # 既に全対象に登録済み（ハンドラ重複回避）

    learning_handler = _make_rotating_handler(
        log_dir / "learning.log", formatter, max_bytes, level=level,
    )
    if learning_handler is None:
        return

    for target_logger in target_loggers:
        if _has_rotating_file_handler(target_logger, target_path):
            continue
        target_logger.setLevel(level)
        target_logger.addHandler(learning_handler)
        target_logger.propagate = False


def setup_cli_logging(
    project_root: Path | None = None,
    debug: bool = False,
) -> None:
    """CLI 専用ロガーを初期化し cli.log に出力する

    backend.cli 名前空間のログを local/logs/cli.log へ書き込む。
    propagate=False でバックエンド側 (backend.log) への混在を防ぐ。
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    log_dir = project_root / "local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # backend.free.cli ロガー設定（重複呼び出しガード）
    cli_logger = logging.getLogger("backend.free.cli")
    cli_logger.setLevel(logging.DEBUG)
    cli_logger.propagate = False

    # cli.log ハンドラ (5MB × 3世代) — 同一パスが既に登録済みなら追加しない
    cli_log_path = (log_dir / "cli.log").resolve()
    if not _has_rotating_file_handler(cli_logger, cli_log_path):
        cli_handler = _make_rotating_handler(
            log_dir / "cli.log", formatter, 5 * 1024 * 1024,
            level=logging.DEBUG,
        )
        if cli_handler is not None:
            cli_logger.addHandler(cli_handler)

    # デバッグモード時は stderr にも出力 — UTF-8 強制
    if debug and not _has_plain_stream_handler(cli_logger):
        console = logging.StreamHandler(stream=_make_utf8_stream())
        console.setFormatter(formatter)
        console.addFilter(PrivateContentFilter())
        console.setLevel(logging.DEBUG)
        cli_logger.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    """名前付きロガー取得"""
    return logging.getLogger(f"backend.{name}")
