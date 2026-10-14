"""CLI サブコマンドの共通実行様式 (docs/f_06 §2.9.3)

- 接続先: :func:`default_backend_url` / :func:`resolve_backend_url` — ``--backend-url``
  の既定は ``config.yaml`` の ``server.port`` (読めなければ 8000)。
- HTTP エラーの解釈: :func:`request` / :class:`BackendError` — 届かなければ ``None``
  (= 停止中)、2xx 以外は c_06 §1.4 の ``{"error": {...}}`` と FastAPI の ``detail``
  の両方を読む。ヘッダは必ず :func:`~backend.free.cli.backend_headers.backend_headers`。
- ``--json``: :func:`emit` が stdout に JSON だけを出す。エラー・案内は
  :func:`diag_console` (``--json`` のときは stderr) へ。
- dual (API → 繋がらなければ停止中の直接): :func:`run_dual`。409 / 423 などの
  応答は :class:`BackendError` のまま上げ、直接へは倒さない。停止中の書き込みは
  :func:`writer_locked` (単一書き手ロック) の下で行う。

終了コードの規約:

- ``0`` (:data:`EXIT_OK`) 成功
- ``1`` (:data:`EXIT_FAIL`) 失敗・拒否・検査で問題あり
- ``2`` (:data:`EXIT_UNAVAILABLE`) 実行不能 (使い方の誤り・backend が要るのに居ない・
  ロック中・readonly)
- ``130`` (:data:`EXIT_INTERRUPTED`) 中断

例外: 既存の Pro コマンド (``hub`` / ``policy`` / ``airgap`` / ``cartridge`` /
``modelfit``) の「検証の失敗・稼働中での拒否 = 2」は互換のため維持する
(登録表の ``registry.EXIT_LEGACY_VALIDATION_2``)。
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import yaml
from rich.console import Console

from backend.data_root import store_root
from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.config_loader import _find_project_root, _load_yaml_section
from backend.free.cli.renderer import _is_no_color
from backend.i18n_helper import msg
from backend.io.writer_lock import WriterLock, acquire_writer_lock

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_UNAVAILABLE = 2
EXIT_INTERRUPTED = 130

DEFAULT_PORT = 8000


# ── 接続先 ──


def default_backend_url(root: Path | None = None) -> str:
    """``config.yaml`` の ``server.port`` を見た既定の backend URL (読めなければ 8000)。"""
    try:
        port = _load_yaml_section(root or _find_project_root(), "server").get("port")
    except (AttributeError, yaml.YAMLError):  # 壊れた YAML / server: null / トップレベルが dict でない
        port = None
    if isinstance(port, int) and not isinstance(port, bool):
        return f"http://localhost:{port}"
    return f"http://localhost:{DEFAULT_PORT}"


def resolve_backend_url(url: str | None, root: Path | None = None) -> str:
    """``--backend-url`` の値 (未指定なら :func:`default_backend_url`)。末尾の ``/`` は落とす。"""
    return (url or default_backend_url(root)).rstrip("/")


# ── HTTP エラーの解釈 ──


class BackendError(Exception):
    """backend が 2xx 以外を返した (表示用の理由・HTTP ステータス・検証エラーを持つ)。"""

    def __init__(self, status: int, detail: str, errors: list[str] | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.errors = errors or []


def error_from_response(resp: httpx.Response) -> BackendError:
    """2xx 以外の応答を :class:`BackendError` にする。

    本文は c_06 §1.4 の ``{"error": {...}}`` と FastAPI の ``{"detail": ...}`` の
    両方を読む。``i18n_key`` があれば CLI の言語で ``context`` の値を埋めて引き直し、
    ``context.reason`` を括弧で添える。読めない本文は ``HTTP <status>``。
    """
    status = resp.status_code
    fallback = f"HTTP {status}"
    try:
        body = resp.json()
    except ValueError:
        return BackendError(status, fallback)
    if not isinstance(body, dict):
        return BackendError(status, fallback)
    err = body.get("error")
    if not isinstance(err, dict):
        err = body.get("detail")
    if isinstance(err, str):
        return BackendError(status, err.strip() or fallback)
    if not isinstance(err, dict):
        return BackendError(status, fallback)
    errors = [str(e) for e in err.get("errors") or []]
    context = err.get("context")
    context = context if isinstance(context, dict) else {}
    reason = context.get("reason")
    key = err.get("i18n_key") or ""
    if key and msg(key) == key:  # CLI の言語ファイルに無いキーは英語の message で出す
        key = ""
    text = msg(key, **context) if key else str(err.get("message") or "").strip()
    # 文言が {reason} を埋め込むキー (api.policy_denied など) では括弧で重ねない
    if key and "{reason}" in msg(key):
        reason = None
    detail = f"{text} ({reason})" if reason else (text or fallback)
    return BackendError(status, detail, errors)


def request(client: httpx.Client, method: str, url: str, **kwargs: Any) -> Any:
    """backend を呼ぶ。届かなければ ``None`` (= 停止中)、2xx 以外は :class:`BackendError`。

    本文の無い 2xx (204 など) は ``{}``。
    """
    try:
        resp = client.request(method, url, **kwargs)
    except httpx.ConnectError:
        return None
    if not 200 <= resp.status_code < 300:
        raise error_from_response(resp)
    if not resp.content:
        return {}
    return resp.json()


def backend_client(timeout: float) -> httpx.Client:
    """:func:`backend_headers` を付けた同期クライアント。"""
    return httpx.Client(headers=backend_headers(), timeout=timeout)


# ── dual (API → 繋がらなければ停止中の直接) ──


def run_dual[T](
    remote: Callable[[httpx.Client], T | None],
    offline: Callable[[], T | None],
    *,
    timeout: float,
) -> T | None:
    """稼働中なら ``remote``、繋がらないとき (``remote`` が ``None``) だけ ``offline``。

    ``remote`` は :func:`request` で backend を呼ぶ。2xx 以外 (409 の実行中・423 の
    readonly など) は :class:`BackendError` のまま上がり、停止中の直接へは倒さない
    — backend が拒否した操作をロックの外から書き直さないため。
    """
    with backend_client(timeout) as client:
        out = remote(client)
    if out is None:
        return offline()
    return out


@contextlib.contextmanager
def writer_locked(data_root: Path) -> Iterator[WriterLock]:
    """停止中の書き込み用に単一書き手ロックを取る。serve の稼働中は ``WriterLockHeld``。"""
    lock = acquire_writer_lock(store_root(data_root))
    try:
        yield lock
    finally:
        lock.release()


# ── 出力 ──


def diag_console(as_json: bool, console: Console) -> Console:
    """エラー・案内の出力先。``--json`` のときは stderr (stdout は JSON だけにする)。"""
    if not as_json:
        return console
    return Console(stderr=True, width=None, no_color=_is_no_color(), highlight=False)


def emit(out: Any, *, as_json: bool, render: Callable[[], None]) -> int:
    """結果を出す。``--json`` なら stdout に JSON だけ、そうでなければ ``render()``。"""
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        render()
    return EXIT_OK


# ── 引数 ──


def add_common_args(
    parser: argparse.ArgumentParser, *, backend: bool = True, json: bool = True, data_root: bool = False,
) -> None:
    """``--backend-url`` / ``--json`` / ``--data-root`` を足す (既定はどれも未指定)。"""
    if backend:
        parser.add_argument(
            "--backend-url", default=None,
            help=f"Backend URL (default: http://localhost:<server.port in config.yaml>, else {DEFAULT_PORT})",
        )
    if json:
        parser.add_argument("--json", action="store_true", help="Print the result as JSON (stdout is JSON only)")
    if data_root:
        parser.add_argument(
            "--data-root", default=None, metavar="PATH",
            help="Data root used while the backend is stopped (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
        )


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be {minimum} or more: {value}")
        return value

    return parse


def add_page_args(parser: argparse.ArgumentParser) -> None:
    """一覧の任意ページング ``--limit N`` / ``--offset N`` を足す (API の ``limit`` / ``offset``、
    c_06 §1.6)。上限は backend が検証する (超えれば 422)。"""
    parser.add_argument("--limit", type=_int_at_least(1), default=None, metavar="N", help="At most N rows")
    parser.add_argument("--offset", type=_int_at_least(0), default=None, metavar="N", help="Skip the first N rows")


def page_params(args: argparse.Namespace) -> dict[str, int]:
    """:func:`add_page_args` の値のうち指定されたものだけをクエリにする (未指定なら空)。"""
    return {k: v for k in ("limit", "offset") if (v := getattr(args, k, None)) is not None}
