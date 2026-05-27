"""テーマインストール・ZIP展開ロジック"""

from __future__ import annotations

import json
import socket
import tempfile
import zipfile
from ipaddress import ip_address
from pathlib import Path
from typing import NotRequired, TypedDict
from urllib.parse import urlparse

import httpx

from backend.log_config import get_logger

logger = get_logger("themes.installer")

# ZIP アップロードサイズ上限 (10MB)
MAX_ZIP_SIZE = 10 * 1024 * 1024

# 組み込みテーマディレクトリ名（現在は制約なし — 全テーマ削除可能）
BUILTIN_THEMES: set[str] = set()


class WidgetManifest(TypedDict):
    """widget-manifest.json の required_apis 部分"""

    required_apis: list[str]


class ThemeInstallResult(TypedDict):
    """install_theme() / install_from_url() の戻り値型"""

    theme_id: str
    name: str
    version: str
    author: str
    trusted: bool
    widget_manifest: NotRequired[WidgetManifest]


async def install_from_url(url: str, themes_dir: Path) -> ThemeInstallResult:
    """URL からテーマ ZIP をダウンロードしてインストール"""
    # URL スキーム検証
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    # SSRF 対策: プライベート IP へのアクセスをブロック
    try:
        resolved = socket.getaddrinfo(parsed.hostname, None)
        for _, _, _, _, sockaddr in resolved:
            addr = ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_reserved:
                raise ValueError("Access to private/reserved addresses is not allowed")
    except socket.gaierror:
        raise ValueError(f"Failed to resolve hostname: {parsed.hostname}")

    # ダウンロード
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(30.0),
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Download failed: HTTP {e.response.status_code}")
    except httpx.HTTPError as e:
        raise ValueError(f"Download failed: {e}")

    content = resp.content
    if len(content) > MAX_ZIP_SIZE:
        raise ValueError(
            f"Downloaded file too large: {len(content)} bytes (max {MAX_ZIP_SIZE})"
        )

    # 一時ファイルに保存して install_theme() に委譲
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    logger.debug(
        "Downloaded theme ZIP from URL: %s (%d bytes)", url, len(content)
    )

    try:
        return install_theme(Path(tmp_path), themes_dir)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def install_theme(zip_path: Path, themes_dir: Path) -> ThemeInstallResult:
    """ZIP パッケージからテーマをインストール"""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if not zipfile.is_zipfile(str(zip_path)):
        raise ValueError("Not a valid ZIP file")

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()

        # theme.json を探す
        meta_file = _find_file_in_zip(names, "theme.json")
        if meta_file is None:
            raise ValueError("theme.json not found in ZIP")

        meta = json.loads(zf.read(meta_file))

        # バリデーション
        if not meta.get("name"):
            raise ValueError("theme.json missing 'name' field")
        if not meta.get("colors") or not isinstance(meta["colors"], dict):
            raise ValueError("theme.json missing 'colors' field")

        # theme_id の決定（ZIP 内のルートディレクトリ名、またはメタデータの name を slug 化）
        theme_id = _detect_theme_id(names, meta)

        # 組み込みテーマと同名は禁止
        if theme_id in BUILTIN_THEMES:
            raise ValueError(f"Cannot overwrite builtin theme '{theme_id}'")

        # 重複チェック
        target_dir = themes_dir / theme_id
        if target_dir.exists():
            raise FileExistsError(f"Theme '{theme_id}' already exists")

        # CSS ファイルの存在確認
        colors = meta.get("colors", {})
        light_css = colors.get("light", "colors-light.css")
        dark_css = colors.get("dark", "colors-dark.css")
        if not _find_file_in_zip(names, light_css):
            raise ValueError(f"{light_css} not found in ZIP")
        if not _find_file_in_zip(names, dark_css):
            raise ValueError(f"{dark_css} not found in ZIP")

        # 展開
        target_dir.mkdir(parents=True, exist_ok=True)
        prefix = _detect_zip_prefix(names)

        for name in names:
            if name.endswith("/"):
                continue
            # ZIP 内のプレフィックスを除去して展開
            rel_path = name[len(prefix):] if prefix and name.startswith(prefix) else name
            if not rel_path:
                continue
            out_path = target_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(name))

    # widget-manifest.json のパース
    widget_manifest = None
    manifest_path = target_dir / "widget-manifest.json"
    if manifest_path.exists():
        try:
            widget_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    result = ThemeInstallResult(
        theme_id=theme_id,
        name=meta.get("name", theme_id),
        version=meta.get("version", "1.0.0"),
        author=meta.get("author", ""),
        trusted=False,
    )
    if widget_manifest and "required_apis" in widget_manifest:
        result["widget_manifest"] = WidgetManifest(
            required_apis=widget_manifest["required_apis"],
        )

    logger.info("Theme installed: id=%s, name=%s", theme_id, meta.get("name"))
    return result


def _detect_theme_id(names: list[str], meta: dict) -> str:
    """ZIP からテーマ ID を推定"""
    # ZIP 内のルートディレクトリ名を使用
    for name in names:
        parts = name.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0]:
            return parts[0]

    # フォールバック: メタデータの name を slug 化
    name = meta.get("name", "unknown")
    return name.lower().replace(" ", "-").replace("_", "-")


def _detect_zip_prefix(names: list[str]) -> str:
    """ZIP 内の共通プレフィックスを検出"""
    dirs = set()
    for name in names:
        parts = name.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0]:
            dirs.add(parts[0])

    if len(dirs) == 1:
        return list(dirs)[0] + "/"
    return ""


def _find_file_in_zip(names: list[str], filename: str) -> str | None:
    """ZIP 内でファイル名を検索"""
    for name in names:
        basename = name.replace("\\", "/").split("/")[-1]
        if basename == filename:
            return name
    return None
