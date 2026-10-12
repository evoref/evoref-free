"""テーマインストール・ZIP展開ロジック"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import TypedDict
from urllib.parse import urljoin, urlparse

import httpx

from backend.free.core.url_guard import (
    MAX_REDIRECTS,
    avalidate_public_url,
    public_only_transport,
)
from backend.io.safe_extract import (
    ExtractBudget,
    ExtractLimits,
    UnsafeArchiveError,
    check_zip_members,
    normalize_member_name,
    resolve_under,
)
from backend.log_config import get_logger

logger = get_logger("themes.installer")

# ZIP アップロードサイズ上限 (10MB)
MAX_ZIP_SIZE = 10 * 1024 * 1024
#: 展開の上限。テーマは CSS / JSON / 画像の小さな束なので狭くする。
THEME_LIMITS = ExtractLimits(
    max_entries=2_000, max_member_bytes=16 << 20, max_total_bytes=64 << 20, max_ratio=200.0,
)
#: テーマ ID はディレクトリ名になるので、先頭がドットや区切りにならない安全な文字だけを許す。
THEME_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def is_valid_theme_id(theme_id: object) -> bool:
    """テーマ ID が themes_dir 直下の 1 段のディレクトリ名として安全か。

    ``THEME_ID_RE`` に加え、Windows で別名になる末尾のドット (``x.`` は ``x``) と
    予約デバイス名 (``con`` / ``nul.txt`` 等) を拒否する。
    """
    if not isinstance(theme_id, str) or not THEME_ID_RE.fullmatch(theme_id):
        return False
    try:
        return normalize_member_name(theme_id) is not None
    except UnsafeArchiveError:
        return False


class ThemeInstallResult(TypedDict):
    """install_theme() / install_from_url() の戻り値型"""

    theme_id: str
    name: str
    version: str
    author: str


def _http_client() -> httpx.AsyncClient:
    """ダウンロード用のクライアント。公開アドレスにしか繋がず、リダイレクトは自分で追う。"""
    return httpx.AsyncClient(
        transport=public_only_transport(),
        follow_redirects=False,
        timeout=httpx.Timeout(30.0),
    )


def _too_large(size: int) -> ValueError:
    return ValueError(f"Downloaded file too large: {size} bytes (max {MAX_ZIP_SIZE})")


async def _download(url: str) -> bytes:
    """``url`` を取る。リダイレクトはホップごとに宛先を検査し、本文は受信しながら上限で切る。

    ``follow_redirects=True`` に任せると、検査した URL が 302 で内部アドレスへ
    飛ばされたときに素通りする (SSRF)。
    """
    current = url
    async with _http_client() as client:
        for _ in range(MAX_REDIRECTS + 1):
            await avalidate_public_url(current)
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    if not location:
                        raise ValueError("Download failed: redirect without location")
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > MAX_ZIP_SIZE:
                    raise _too_large(int(declared))
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_ZIP_SIZE:
                        raise _too_large(total)
                    chunks.append(chunk)
                return b"".join(chunks)
    raise ValueError(f"Download failed: too many redirects (> {MAX_REDIRECTS})")


async def install_from_url(url: str, themes_dir: Path) -> ThemeInstallResult:
    """URL からテーマ ZIP をダウンロードしてインストール"""
    # URL スキーム検証
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    # ダウンロード (SSRF 検査は backend.free.core.url_guard。UnsafeUrlError は ValueError)
    try:
        content = await _download(url)
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Download failed: HTTP {e.response.status_code}")
    except httpx.HTTPError as e:
        raise ValueError(f"Download failed: {e}")

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

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            theme_id, meta, target_dir = _extract_theme(zf, themes_dir)
    except (zipfile.BadZipFile, zlib.error, EOFError) as e:
        raise ValueError(f"Corrupted ZIP file: {e}") from e

    result = ThemeInstallResult(
        theme_id=theme_id,
        name=meta.get("name", theme_id),
        version=meta.get("version", "1.0.0"),
        author=meta.get("author", ""),
    )

    logger.info("Theme installed: id=%s, name=%s", theme_id, meta.get("name"))
    return result


def _extract_theme(zf: zipfile.ZipFile, themes_dir: Path) -> tuple[str, dict, Path]:
    """検査してから一時ディレクトリへ展開し、``themes_dir/<theme_id>`` へ rename する。"""
    # 全エントリを先に検査する (パスの逃げ・予約名・シンボリックリンク・展開後サイズ)。
    # エントリ名をそのまま展開先に連結しない (docs/c_06 §1.5)。
    members = [
        (info, member) for info, member in check_zip_members(zf, THEME_LIMITS)
        if not _is_os_metadata(member)
    ]
    names = [str(member) for _, member in members]
    prefix = _detect_zip_prefix(names)

    # theme.json はテーマの根 (プレフィックス直下) のものだけを読む
    meta_file = prefix + "theme.json"
    if meta_file not in names:
        raise ValueError("theme.json not found in ZIP")

    meta = json.loads(zf.read(meta_file))

    # バリデーション
    if not isinstance(meta, dict):
        raise ValueError("theme.json must be a JSON object")
    if not meta.get("name") or not isinstance(meta["name"], str):
        raise ValueError("theme.json missing 'name' field")
    colors = meta.get("colors")
    if not colors or not isinstance(colors, dict):
        raise ValueError("theme.json missing 'colors' field")
    if not all(isinstance(v, str) for v in colors.values()):
        raise ValueError("theme.json 'colors' values must be file names")

    # theme_id の決定（ZIP 内のルートディレクトリ名、またはメタデータの name を slug 化）
    theme_id = _detect_theme_id(prefix, meta)

    if not is_valid_theme_id(theme_id):
        raise ValueError(f"Invalid theme id: {theme_id!r}")

    # 重複チェック
    target_dir = themes_dir / theme_id
    if target_dir.exists():
        raise FileExistsError(f"Theme '{theme_id}' already exists")

    # CSS ファイルの存在確認 (activate はテーマの根からの相対で読む)
    for css in (colors.get("light", "colors-light.css"), colors.get("dark", "colors-dark.css")):
        rel = normalize_member_name(css)
        if rel is None or prefix + str(rel) not in names:
            raise ValueError(f"{css} not found in ZIP")

    # 展開は一時ディレクトリへ行い、全ファイル書き終えてから rename する。
    # 直接 target_dir へ書くと途中で落ちたとき半端なテーマが残り、しかも
    # 上の重複チェック (target_dir.exists()) が以後の再インストールを
    # 永久に拒否する (2026-09-05 監査)。作業ディレクトリは要求ごとに一意にする
    # (固定名だと同じ ID の同時インストールが互いの展開を消す)。
    themes_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{theme_id}.", suffix=".installing", dir=themes_dir))

    budget = ExtractBudget(THEME_LIMITS)
    try:
        for info, member in members:
            # ZIP 内のプレフィックスを除去して展開
            parts = member.parts
            if prefix:
                parts = parts[1:]
            if not parts:
                continue
            out_path = resolve_under(staging_dir, PurePosixPath(*parts))
            with zf.open(info) as src:
                try:
                    budget.copy(src, out_path, compressed_size=info.compress_size)
                except FileExistsError as e:
                    # ZIP 内でファイルとディレクトリが同名。既存テーマとの衝突 (409) ではない
                    raise UnsafeArchiveError(f"conflicting entries in archive: {member}") from e
        staging_dir.rename(target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return theme_id, meta, target_dir


def _is_os_metadata(member: PurePosixPath) -> bool:
    """macOS の Finder が足す ``__MACOSX/`` と ``._*`` (テーマの中身ではない)。"""
    return member.parts[0] == "__MACOSX" or member.name.startswith("._")


def _detect_theme_id(prefix: str, meta: dict) -> str:
    """ZIP からテーマ ID を推定"""
    # ZIP 内のルートディレクトリ名を使用
    if prefix:
        return prefix.rstrip("/")

    # フォールバック: メタデータの name を slug 化
    name = meta.get("name", "unknown")
    return name.lower().replace(" ", "-").replace("_", "-")


def _detect_zip_prefix(names: list[str]) -> str:
    """全エントリが 1 つのディレクトリの下にあれば、そのディレクトリ (``"<dir>/"``)。無ければ ``""``"""
    tops = {name.split("/")[0] for name in names}
    if len(tops) == 1 and all("/" in name for name in names):
        return tops.pop() + "/"
    return ""
