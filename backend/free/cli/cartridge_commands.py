"""/cartridge 対話コマンド: カートリッジの管理（インストール・一覧・ロード・アンロード・削除・再構築）

作成 (``/cartridge create``) は Pro の API を使うので Pro 側が
``command_parser.register_slash_command`` で差し込む (Free は「Pro 専用」の案内だけ)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.renderer import (
    render_error,
    render_info,
    render_table,
)
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.cartridge")


@dataclass
class CartridgeTimeouts:
    """カートリッジ CLI コマンドのタイムアウト設定"""

    default: float = 30.0
    install: float = 300.0
    rebuild: float = 300.0
    create: float = 600.0


_timeouts = CartridgeTimeouts()


def init_cartridge_timeouts(cli_config: dict) -> None:
    """config.yaml の cli セクションからタイムアウト値を初期化する

    Args:
        cli_config: config.yaml の cli セクション辞書
    """
    global _timeouts
    t = cli_config.get("timeouts", {})
    _timeouts = CartridgeTimeouts(
        default=t.get("default", 30.0),
        install=t.get("install", 300.0),
        rebuild=t.get("rebuild", 300.0),
        create=t.get("create", 600.0),
    )


def _extract_detail_message(detail) -> str:
    """API エラーレスポンスの detail から表示用メッセージを抽出する

    detail が構造化 dict の場合は message フィールドを、
    文字列の場合はそのまま返す。
    """
    if isinstance(detail, dict):
        return detail.get("message", "") or str(detail)
    return str(detail)


def _fmt_size(size_mb: float) -> str:
    """MB 値を KB/MB の適切な単位で表示"""
    if size_mb < 0.1:
        return f"{size_mb * 1024:.1f} KB"
    return f"{size_mb:.1f} MB"


def _fmt_time_jst(iso_str: str | None) -> str:
    """UTC ISO文字列をローカル時刻 (YYYY-MM-DD HH:MM:SS) に変換"""
    if not iso_str:
        return "never"
    from datetime import datetime
    try:
        utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        local = utc.astimezone()
        return local.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return iso_str


# ── 内部実装関数 ──


def _cartridge_list(backend_url: str, console) -> int:
    """インストール済みカートリッジ一覧を表示"""
    try:
        resp = httpx.get(
            f"{backend_url}/api/cartridges",
            headers=backend_headers(),
            timeout=_timeouts.default,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    data = resp.json()
    carts = data.get("cartridges", [])

    if not carts:
        render_info(console, msg("cli.cartridge_empty"))
        return 0

    render_info(console, msg("cli.cartridge_total", count=len(carts)))

    rows = []
    for c in carts:
        rows.append({
            "ID": c["id"],
            msg("cli.cartridge_col_name"): c.get("name", ""),
            msg("cli.cartridge_col_version"): c.get("version", ""),
            msg("cli.cartridge_col_status"): c.get("status", ""),
            msg("cli.cartridge_col_chunks"): str(c.get("chunks", 0)),
            msg("cli.cartridge_col_size"): _fmt_size(c.get("size_mb", 0)),
        })

    headers = [
        "ID",
        msg("cli.cartridge_col_name"),
        msg("cli.cartridge_col_version"),
        msg("cli.cartridge_col_status"),
        msg("cli.cartridge_col_chunks"),
        msg("cli.cartridge_col_size"),
    ]
    render_table(console, rows, headers)
    return 0


def _cartridge_install(backend_url: str, console, file_path: str) -> int:
    """カートリッジ ZIP をインストール"""
    zip_path = Path(file_path).resolve()
    if not zip_path.exists():
        render_error(console, msg("cli.cartridge_file_not_found", path=str(zip_path)))
        return 1
    if not zip_path.is_file():
        render_error(console, msg("cli.cartridge_file_not_found", path=str(zip_path)))
        return 1

    logger.debug("cartridge install: file=%s", zip_path)

    try:
        with console.status(msg("cli.cartridge_installing"), spinner="dots"):
            with open(zip_path, "rb") as f:
                files = {"file": (zip_path.name, f, "application/zip")}
                resp = httpx.post(
                    f"{backend_url}/api/cartridges/install",
                    headers=backend_headers(),
                    files=files,
                    timeout=_timeouts.install,
                )

        if resp.status_code == 400:
            detail = resp.json().get("detail", "Bad request")
            render_error(console, _extract_detail_message(detail))
            return 1
        resp.raise_for_status()

    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1
    except OSError as e:
        render_error(console, msg("cli.cartridge_file_not_found", path=str(e)))
        return 1

    result = resp.json()
    render_info(console, msg(
        "cli.cartridge_installed",
        name=result.get("name", ""),
        id=result.get("id", ""),
        chunks=result.get("chunks", 0),
    ))
    install_time = result.get("install_time_sec", 0)
    if install_time > 0:
        render_info(console, msg("cli.cartridge_install_time", seconds=f"{install_time:.1f}"))

    logger.debug(
        "Cartridge installed: id=%s, name=%s, chunks=%d, time=%.3fs",
        result.get("id"), result.get("name"), result.get("chunks", 0), install_time,
    )
    return 0


def _cartridge_show(backend_url: str, console, cartridge_id: str) -> int:
    """カートリッジ詳細情報を表示"""
    try:
        resp = httpx.get(
            f"{backend_url}/api/cartridges/{cartridge_id}",
            headers=backend_headers(),
            timeout=_timeouts.default,
        )
        if resp.status_code == 404:
            render_error(console, msg("cli.cartridge_not_found", id=cartridge_id))
            return 1
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    c = resp.json()

    render_info(console, msg("cli.cartridge_detail_header", id=c['id']))
    render_info(console, f"  {msg('cli.cartridge_col_name')}: {c.get('name', '')}")
    render_info(console, f"  {msg('cli.cartridge_col_version')}: {c.get('version', '')}")
    render_info(console, f"  {msg('cli.cartridge_col_status')}: {c.get('status', '')}")
    if c.get("author"):
        render_info(console, f"  {msg('cli.cartridge_detail_author')}: {c['author']}")
    if c.get("description"):
        render_info(console, f"  {msg('cli.cartridge_detail_description')}: {c['description']}")
    if c.get("tags"):
        render_info(console, f"  {msg('cli.cartridge_detail_tags')}: {', '.join(c['tags'])}")
    if c.get("language"):
        render_info(console, f"  {msg('cli.cartridge_detail_language')}: {c['language']}")
    render_info(console, f"  {msg('cli.cartridge_col_chunks')}: {c.get('chunks', 0)}")
    render_info(console, f"  {msg('cli.cartridge_detail_docs')}: {c.get('doc_count', 0)}")
    render_info(console, f"  {msg('cli.cartridge_col_size')}: {_fmt_size(c.get('size_mb', 0))}")
    if c.get("license"):
        render_info(console, f"  {msg('cli.cartridge_detail_license')}: {c['license']}")
    if c.get("active_version"):
        render_info(
            console,
            f"  {msg('cli.cartridge_detail_active_version')}: {c['active_version']}",
        )
    if c.get("embedding_model_id"):
        render_info(
            console,
            f"  {msg('cli.cartridge_detail_embedding_model')}: {c['embedding_model_id']}",
        )
    if c.get("installed_at"):
        render_info(console, f"  {msg('cli.cartridge_detail_installed_at')}: {_fmt_time_jst(c['installed_at'])}")
    if c.get("compatibility"):
        render_info(console, f"  {msg('cli.cartridge_detail_compatibility')}: {c['compatibility']}")

    return 0


def _cartridge_load(backend_url: str, console, cartridge_id: str) -> int:
    """カートリッジをメモリにロード"""
    logger.debug("cartridge load: id=%s", cartridge_id)

    try:
        resp = httpx.post(
            f"{backend_url}/api/cartridges/{cartridge_id}/load",
            headers=backend_headers(),
            timeout=_timeouts.default,
        )
        if resp.status_code == 404:
            render_error(console, msg("cli.cartridge_not_found", id=cartridge_id))
            return 1
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    result = resp.json()
    load_time = result.get("load_time_ms", 0)
    render_info(console, msg("cli.cartridge_loaded", id=cartridge_id, time_ms=f"{load_time:.0f}"))
    logger.debug("Cartridge loaded: id=%s, time=%.1fms", cartridge_id, load_time)
    return 0


def _cartridge_unload(backend_url: str, console, cartridge_id: str) -> int:
    """カートリッジをメモリからアンロード"""
    logger.debug("cartridge unload: id=%s", cartridge_id)

    try:
        resp = httpx.post(
            f"{backend_url}/api/cartridges/{cartridge_id}/unload",
            headers=backend_headers(),
            timeout=_timeouts.default,
        )
        if resp.status_code == 404:
            render_error(console, msg("cli.cartridge_not_found", id=cartridge_id))
            return 1
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    render_info(console, msg("cli.cartridge_unloaded", id=cartridge_id))
    logger.debug("Cartridge unloaded: id=%s", cartridge_id)
    return 0


def _cartridge_uninstall(backend_url: str, console, cartridge_id: str) -> int:
    """カートリッジを完全削除"""
    logger.debug("cartridge uninstall: id=%s", cartridge_id)

    try:
        resp = httpx.delete(
            f"{backend_url}/api/cartridges/{cartridge_id}",
            headers=backend_headers(),
            timeout=_timeouts.default,
        )
        if resp.status_code == 404:
            render_error(console, msg("cli.cartridge_not_found", id=cartridge_id))
            return 1
        # 204 No Content は正常
        if resp.status_code == 204:
            render_info(console, msg("cli.cartridge_uninstalled", id=cartridge_id))
            logger.debug("Cartridge uninstalled: id=%s", cartridge_id)
            return 0
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    render_info(console, msg("cli.cartridge_uninstalled", id=cartridge_id))
    return 0



def _cartridge_rebuild(backend_url: str, console, cartridge_id: str) -> int:
    """カートリッジのベクトルインデックスを再構築"""
    logger.debug("cartridge rebuild: id=%s", cartridge_id)

    try:
        with console.status(msg("cli.cartridge_rebuilding", id=cartridge_id), spinner="dots"):
            resp = httpx.post(
                f"{backend_url}/api/cartridges/{cartridge_id}/rebuild",
                headers=backend_headers(),
                timeout=_timeouts.rebuild,
            )
        if resp.status_code == 404:
            render_error(console, msg("cli.cartridge_not_found", id=cartridge_id))
            return 1
        if resp.status_code == 400:
            detail = resp.json().get("detail", "Bad request")
            render_error(console, _extract_detail_message(detail))
            return 1
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    result = resp.json()
    render_info(console, msg(
        "cli.cartridge_rebuilt",
        id=result.get("id", ""),
        chunks=result.get("chunks", 0),
        size=_fmt_size(result.get("size_mb", 0)),
    ))

    embedder_used = result.get("embedder_used", "")
    render_info(console, msg(
        "cli.cartridge_rebuild_embedder",
        embedder=embedder_used or "unknown",
    ))

    rebuild_time = result.get("rebuild_time_sec", 0)
    if rebuild_time > 0:
        render_info(console, msg("cli.cartridge_rebuild_time", seconds=f"{rebuild_time:.1f}"))

    logger.debug(
        "Cartridge rebuilt: id=%s, chunks=%d, embedder=%s, time=%.3fs",
        result.get("id"), result.get("chunks", 0),
        embedder_used or "unknown", rebuild_time,
    )
    return 0
