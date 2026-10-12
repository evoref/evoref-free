"""外向き HTTP の宛先検査 (SSRF 対策) — Free / Pro 共通。

バックエンドのプロセスから利用者の指定した URL を叩く経路 (テーマの URL
導入 / Pro の ``know.*`` 取得器) は、検査しないと **社内ネットワークとクラウドの
メタデータ endpoint への踏み台** になる (``http://169.254.169.254/`` /
``http://127.0.0.1:8000/`` 等)。

検査は 2 段で掛ける:

- :func:`validate_public_url` — スキーム / ホスト / 名前解決の結果がすべて
  公開アドレスか。登録時と、リダイレクトの **ホップごと** に呼ぶ。
- :func:`public_only_transport` — 接続の瞬間に名前を引き直して検査し、
  **検査した IP へそのまま繋ぐ**。前段だけだと検査と接続の間で DNS の答えが
  変わったとき (DNS rebinding) に内部へ繋がる。Host ヘッダと SNI は元の
  ホスト名のまま (httpcore が URL の origin から組む)。

名前解決できないホストは **拒否する** (引けない名前は検査できない)。
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any, override
from urllib.parse import urlsplit

import httpcore
import httpx

from backend.log_config import get_logger

logger = get_logger("core.url_guard")

#: 許すスキーム。``file`` / ``ftp`` / ``gopher`` 等は対象外。
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: 追うリダイレクトの上限 (ホップ数)。
MAX_REDIRECTS = 5

#: NAT64 の well-known prefix。下位 32 bit が IPv4 で、そちらで判定する。
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class UnsafeUrlError(ValueError):
    """内部アドレス / 未対応スキーム / 解決できないホストなど、叩いてはいけない URL。"""


class UnresolvedHostError(UnsafeUrlError):
    """ホスト名が解決できない (非公開アドレスとは別の理由で拒否したことを呼出側が区別する)。"""


def is_public_ip(ip: IPAddress) -> bool:
    """公開ユニキャストのアドレスか。

    ``is_global`` で判定する (ループバック / プライベート / リンクローカル /
    CGNAT ``100.64.0.0/10`` / ULA / ``0.0.0.0`` を落とす)。IPv4 を包んだ
    IPv6 (``::ffff:127.0.0.1`` / NAT64 ``64:ff9b::/96``) は中の IPv4 で
    判定し直す — NAT64 の prefix 自体は global なので素通りする。
    """
    if isinstance(ip, ipaddress.IPv6Address):
        inner = ip.ipv4_mapped
        if inner is None and ip in _NAT64:
            inner = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if inner is not None:
            return is_public_ip(inner)
    return ip.is_global and not ip.is_multicast


def resolve_host(host: str) -> list[str]:
    """ホスト名を解決してアドレス文字列のリストを返す (同期、重複は除く)。

    Raises:
        UnresolvedHostError: 解決できない / 結果が空。
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError, ValueError) as e:
        raise UnresolvedHostError(f"cannot resolve host {host!r}: {e}") from e
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise UnresolvedHostError(f"host {host!r} resolved to no address")
    return addresses


def public_addresses(host: str) -> list[str]:
    """``host`` の接続先アドレス。1 つでも非公開アドレスを含めば拒否する。

    Raises:
        UnsafeUrlError: 解決できない / 非公開アドレスを含む。
    """
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if not is_public_ip(literal):
            raise UnsafeUrlError(f"address {host} is a private/reserved address")
        return [str(literal)]
    addresses = resolve_host(host)
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as e:
            raise UnsafeUrlError(f"host {host} resolved to {address!r}") from e
        if not is_public_ip(ip):
            raise UnsafeUrlError(
                f"host {host} resolves to a private/reserved address ({address})",
            )
    return addresses


def validate_public_url(url: str) -> str:
    """叩いてよい URL か検査して、そのまま返す (同期。名前解決を含む)。

    イベントループ上では :func:`avalidate_public_url` を使う。

    Raises:
        UnsafeUrlError: スキームが http/https でない / ホストが無い /
            ホストが解決できない / 解決結果に非公開アドレスを含む。
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"unsupported scheme: {url!r}")
    host = parts.hostname
    if not host:
        raise UnsafeUrlError(f"url has no host: {url!r}")
    public_addresses(host)
    return url


async def avalidate_public_url(url: str) -> str:
    """:func:`validate_public_url` をワーカースレッドで (名前解決でループを止めない)。"""
    return await asyncio.to_thread(validate_public_url, url)


class _PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """接続時に名前を引き直して検査し、検査済みの IP へ繋ぐネットワーク層。"""

    def __init__(self) -> None:
        self._inner = httpcore.AnyIOBackend()

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await asyncio.to_thread(public_addresses, host)
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self._inner.connect_tcp(
                    address, port, timeout=timeout,
                    local_address=local_address, socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                last_error = e
        assert last_error is not None
        raise last_error

    @override
    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        raise UnsafeUrlError("unix sockets are not allowed")

    @override
    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def public_only_transport(**kwargs: Any) -> httpx.AsyncHTTPTransport:
    """公開アドレスにしか繋がない ``httpx`` のトランスポート。

    ``kwargs`` は :class:`httpx.AsyncHTTPTransport` へそのまま渡す (``proxy`` は
    不可 — プロキシ越しだと接続先がプロキシになり検査が効かない)。
    リダイレクトは呼出側が ``follow_redirects=False`` で 1 ホップずつ追い、
    各ホップを :func:`avalidate_public_url` に通す。
    """
    if kwargs.get("proxy") is not None:
        raise ValueError("public_only_transport does not support proxies")
    transport = httpx.AsyncHTTPTransport(**kwargs)
    pool = transport._pool  # noqa: SLF001 — 公開 API に network_backend の口が無い
    if not isinstance(pool, httpcore.AsyncConnectionPool):
        raise RuntimeError(f"unexpected httpcore pool: {type(pool).__name__}")
    pool._network_backend = _PublicOnlyBackend()  # noqa: SLF001
    return transport


__all__ = [
    "ALLOWED_SCHEMES",
    "MAX_REDIRECTS",
    "UnresolvedHostError",
    "UnsafeUrlError",
    "avalidate_public_url",
    "is_public_ip",
    "public_addresses",
    "public_only_transport",
    "resolve_host",
    "validate_public_url",
]
