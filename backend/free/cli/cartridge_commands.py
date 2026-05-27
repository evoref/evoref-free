"""/cartridge 対話コマンド: カートリッジの管理（インストール・一覧・ロード・アンロード・削除・作成）"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

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
            timeout=_timeouts.default,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
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
        render_error(console, f"API error: {e.response.status_code}")
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
        render_error(console, f"API error: {e.response.status_code}")
        return 1

    c = resp.json()

    render_info(console, f"Cartridge: {c['id']}")
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
    if c.get("priority") is not None:
        render_info(console, f"  {msg('cli.cartridge_detail_priority')}: {c['priority']}")
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
        render_error(console, f"API error: {e.response.status_code}")
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
        render_error(console, f"API error: {e.response.status_code}")
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
        render_error(console, f"API error: {e.response.status_code}")
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
        render_error(console, f"API error: {e.response.status_code}")
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



# ── カートリッジ作成（Pro 専用） ──


def _collect_source_files(source_dir: str) -> list[Path] | None:
    """ソースディレクトリまたは glob パターンからファイルを収集する

    Args:
        source_dir: ディレクトリパス、ファイルパス、または glob パターン

    Returns:
        収集したファイルリスト。該当なしの場合は None。
    """
    import glob as glob_mod

    source_path = Path(source_dir)

    if source_path.is_dir():
        source_files = [p for p in source_path.rglob("*") if p.is_file()]
    elif "*" in source_dir or "?" in source_dir:
        source_files = [Path(p) for p in glob_mod.glob(source_dir, recursive=True) if Path(p).is_file()]
    elif source_path.is_file():
        source_files = [source_path]
    else:
        return None

    return source_files if source_files else None


def _build_create_multipart(
    source_files: list[Path],
    console,
    *,
    cart_id: str,
    name: str,
    version: str,
    author: str,
    description: str,
    tags: str,
    language: str,
    generate_eval: bool,
    eval_count: int,
) -> tuple[list, dict] | None:
    """マルチパートフォームデータを構築する

    Args:
        source_files: ソースファイルリスト
        console: CLI コンソール

    Returns:
        (multipart_files, form_data) のタプル。構築失敗時は None。
    """
    multipart_files = []
    for sf in source_files:
        try:
            multipart_files.append(
                ("files", (sf.name, open(sf, "rb"), "application/octet-stream"))
            )
        except OSError as e:
            render_error(console, f"Cannot read: {sf} ({e})")

    if not multipart_files:
        return None

    form_data = {
        "id": cart_id,
        "name": name,
        "version": version,
        "author": author,
        "description": description,
        "tags": tags,
        "language": language,
        "generate_eval": str(generate_eval).lower(),
        "eval_qa_count": str(eval_count),
    }

    return multipart_files, form_data


def _execute_create_request(
    backend_url: str,
    console,
    multipart_files: list,
    form_data: dict,
    cart_id: str,
    source_count: int,
) -> dict | None:
    """Pro API にカートリッジ作成リクエストを送信する

    Returns:
        成功時は API レスポンス dict、失敗時は None。
    """
    try:
        with console.status(
            msg("cli.cartridge_create_start", id=cart_id, count=source_count),
            spinner="dots",
        ):
            resp = httpx.post(
                f"{backend_url}/api/pro/cartridges/create",
                files=multipart_files,
                data=form_data,
                timeout=_timeouts.create,
            )

        if resp.status_code == 400:
            detail = resp.json().get("detail", "Bad request")
            render_error(console, _extract_detail_message(detail))
            return None
        if resp.status_code == 404:
            render_error(console, msg("cli.cartridge_create_pro_only"))
            return None
        resp.raise_for_status()

    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return None
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return None
    finally:
        for _, file_tuple in multipart_files:
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()

    return resp.json()


def _display_and_download_result(
    backend_url: str,
    console,
    result: dict,
    output: str | None,
    cart_id: str,
) -> int:
    """作成結果を表示し、ZIP をダウンロードする

    Returns:
        0: 成功, 1: エラー
    """
    render_info(console, msg(
        "cli.cartridge_create_done",
        id=result.get("id", ""),
        name=result.get("name", ""),
        doc_count=result.get("doc_count", 0),
        eval_qa_count=result.get("eval_qa_count", 0),
    ))

    if result.get("errors"):
        for err in result["errors"]:
            render_info(console, f"  Warning: {err}")

    duration = result.get("duration_sec", 0)
    if duration > 0:
        render_info(console, msg("cli.cartridge_create_duration", duration=f"{duration:.1f}"))

    download_url = result.get("download_url")
    if download_url:
        output_dir = Path(output) if output else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"{cart_id}.zip"

        try:
            dl_resp = httpx.get(
                f"{backend_url}{download_url}",
                timeout=_timeouts.default,
            )
            dl_resp.raise_for_status()
            zip_path.write_bytes(dl_resp.content)
            render_info(console, msg(
                "cli.cartridge_create_saved",
                path=str(zip_path),
            ))
        except Exception as e:
            render_error(console, f"Failed to download ZIP: {e}")
            return 1

    return 0


def _cartridge_create(
    backend_url: str,
    console,
    *,
    source_dir: str,
    cart_id: str,
    name: str,
    version: str,
    author: str,
    description: str,
    tags: str,
    language: str,
    generate_eval: bool,
    eval_count: int,
    output: str | None,
) -> int:
    """カートリッジ作成（Pro 専用）

    ソースディレクトリまたは glob パターンからファイルを収集し、
    Pro API を呼び出してカートリッジ ZIP を生成する。
    """
    source_files = _collect_source_files(source_dir)
    if source_files is None:
        render_error(console, msg("cli.cartridge_file_not_found", path=source_dir))
        return 1

    logger.debug("cartridge create: id=%s, files=%d", cart_id, len(source_files))

    multipart = _build_create_multipart(
        source_files, console,
        cart_id=cart_id, name=name, version=version, author=author,
        description=description, tags=tags, language=language,
        generate_eval=generate_eval, eval_count=eval_count,
    )
    if multipart is None:
        render_error(console, msg("cli.cartridge_create_no_files"))
        return 1

    multipart_files, form_data = multipart

    result = _execute_create_request(
        backend_url, console, multipart_files, form_data,
        cart_id, len(source_files),
    )
    if result is None:
        return 1

    return _display_and_download_result(backend_url, console, result, output, cart_id)
