"""``fetch_url`` ツール — URL 取得と取得前の検証

スキーム / リダイレクト / 私有 IP の検証を通してから取得し、本文抽出は
``html_text`` へ渡す。
"""

from __future__ import annotations

import asyncio
import socket

from ipaddress import ip_address
from urllib.parse import urljoin, urlparse
from backend.log_config import get_logger

from backend.free.agent.tools.html_text import (
    _contains_markdown_table,
    _html_to_text,
)

logger = get_logger("agent.tools.builtin")



_FETCH_URL_USER_AGENT = "evoref-fetch/1.0"
_FETCH_URL_MAX_BYTES = 5_000_000  # 5 MB (raw body) — DoS 抑止
_FETCH_URL_MAX_REDIRECTS = 5
_FETCH_URL_ALLOWED_SCHEMES = ("http", "https")
_FETCH_URL_TEXT_CONTENT_TYPE_PREFIXES = (
    "text/",
    "application/xhtml",
    "application/xml",
    "application/json",
)
# fetch_url 結果のプロンプト合流時の最大文字数。
# 20_000 ではベース LLM のプリフィルが 30〜50 秒に達して
# フロント側 SSE chunk timeout を引き起こしていたため 8_000 に抑制。
_FETCH_URL_MAX_TEXT_CHARS = 8_000
# 表を含むページは行データの取りこぼしを避けるため truncate 上限を引き上げる。
# 表はトークンが短く、メタ認知ループのツール結果として消費されるため、ベース LLM の
# プリフィル懸念 (8000 制限の理由) は当てはまりにくい。
_FETCH_URL_MAX_TEXT_CHARS_TABLE = 40_000
#: リダイレクト扱いにするステータス。``follow_redirects=True`` に任せると
#: 30x の Location 先 (127.0.0.1 / 169.254.169.254 等) が検証を通らずに
#: 取得されるため、追跡は自前で行い各ホップを ``_validate_fetch_url`` に通す。
_FETCH_URL_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


async def _validate_fetch_url(url: str, *, allow_private_ip: bool) -> str | None:
    """fetch_url の URL を検証する。エラー時はユーザー向け文字列、安全なら None。

    theme_installer.install_from_url の検証パターンをミラー。名前解決は
    イベントループの executor で行い、応答パスをブロックしない。
    """
    parsed = urlparse(url)
    if parsed.scheme not in _FETCH_URL_ALLOWED_SCHEMES:
        return f"Error: Unsupported URL scheme: {parsed.scheme!r}"
    if not parsed.hostname:
        return "Error: URL has no hostname"
    if allow_private_ip:
        return None
    try:
        loop = asyncio.get_running_loop()
        resolved = await loop.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        return f"Error: Failed to resolve hostname {parsed.hostname!r}: {e}"
    for _, _, _, _, sockaddr in resolved:
        addr = ip_address(sockaddr[0])
        if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
            return (
                "Error: Access to private/reserved addresses is not allowed "
                "(set tools.fetch_url_allow_private_ip: true to override)"
            )
    return None


def _redact_url_for_log(url: str) -> str:
    """ログ用に URL のクエリ文字列・fragment を除去 (PII / token 漏洩対策)"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


async def fetch_url(
    url: str,
    timeout: int = 10,
    *,
    allow_private_ip: bool = False,
) -> str:
    """URL を取得してテキスト化する"""
    if not url:
        return "Error: URL is required"

    err = await _validate_fetch_url(url, allow_private_ip=allow_private_ip)
    if err:
        return err

    import httpx as _httpx

    body_bytes = bytearray()
    truncated = False
    encoding = "utf-8"
    current_url = url
    try:
        async with _httpx.AsyncClient(
            follow_redirects=False,
            timeout=_httpx.Timeout(timeout),
            headers={"User-Agent": _FETCH_URL_USER_AGENT},
        ) as client:
            for _hop in range(_FETCH_URL_MAX_REDIRECTS + 1):
                async with client.stream("GET", current_url) as r:
                    location = r.headers.get("location")
                    if r.status_code in _FETCH_URL_REDIRECT_STATUSES and location:
                        next_url = urljoin(current_url, location)
                        err = await _validate_fetch_url(
                            next_url, allow_private_ip=allow_private_ip,
                        )
                        if err:
                            logger.warning(
                                "fetch_url redirect blocked: from=%s to=%s",
                                _redact_url_for_log(current_url),
                                _redact_url_for_log(next_url),
                            )
                            return err
                        current_url = next_url
                        continue
                    r.raise_for_status()
                    ctype = r.headers.get("content-type", "").lower()
                    if not any(
                        ctype.startswith(p)
                        for p in _FETCH_URL_TEXT_CONTENT_TYPE_PREFIXES
                    ):
                        return f"Error: Unsupported content-type: {ctype or '(none)'}"
                    encoding = r.encoding or "utf-8"
                    async for chunk in r.aiter_bytes():
                        body_bytes.extend(chunk)
                        if len(body_bytes) >= _FETCH_URL_MAX_BYTES:
                            truncated = True
                            break
                    break
            else:
                return (
                    f"Error: Too many redirects (limit {_FETCH_URL_MAX_REDIRECTS})"
                )
    except Exception as e:
        logger.warning("fetch_url failed: url=%s err=%r", _redact_url_for_log(url), e)
        return f"Error fetching URL ({type(e).__name__}): {e}"

    html_text = bytes(body_bytes).decode(encoding, errors="replace")

    # 本文抽出: ボイラープレート (nav/menu/footer/リンク一覧) を除去して
    # 本文を分離する。bs4 不在・過剰除去時は naive 抽出へ安全に退避。
    text = _html_to_text(html_text)

    # 表を含むページは行の取りこぼしを防ぐため truncate 上限を引き上げる。
    cap = (
        _FETCH_URL_MAX_TEXT_CHARS_TABLE
        if _contains_markdown_table(text)
        else _FETCH_URL_MAX_TEXT_CHARS
    )
    if len(text) > cap:
        text = text[:cap] + "\n... (truncated)"
    if truncated:
        text += f"\n... (response body truncated at {_FETCH_URL_MAX_BYTES} bytes)"
    return text


def _make_fetch_url(cfg: dict):
    """fetch_url ツールハンドラを生成（config をクロージャでバインド）

    戻り値は ``_FETCH_URL_MAX_TEXT_CHARS`` (表を含む場合は
    ``_FETCH_URL_MAX_TEXT_CHARS_TABLE``) で切り詰めた本文。
    """
    tools_cfg = cfg.get("tools", {})
    default_timeout = int(tools_cfg.get("fetch_url_timeout", 10))
    allow_private_ip = bool(tools_cfg.get("fetch_url_allow_private_ip", False))

    async def _fetch_url(url: str, timeout: int = default_timeout) -> str:
        return await fetch_url(
            url, timeout=timeout, allow_private_ip=allow_private_ip,
        )

    return _fetch_url
