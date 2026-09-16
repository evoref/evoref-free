"""画像参照の解決 — 全 Writer 共通

設計書: [docs/f_11_file_export.md](../../docs/f_11_file_export.md) §4.1。

``![alt](src)`` の ``src`` をローカルの実ファイルへ解決する。Writer から
ネットワークへは出ない (URL は取得せず落とす)。
"""

from __future__ import annotations

from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("export.media")

#: 埋め込み画像の既定の最大幅 (cm)。元画像のアスペクト比は保つ。
DEFAULT_MAX_WIDTH_CM = 12.0

_URL_PREFIXES = ("http://", "https://", "ftp://", "data:")

#: 相対画像パスの基準ディレクトリを Writer へ渡す ``ExportContent.metadata`` キー。
#: ``BytesWriterBase.write`` が出力先から設定する。``write_to_bytes`` (API の
#: ダウンロード経路) には出力先が無いので未設定になり、相対パスは解決できない。
BASE_DIR_METADATA_KEY = "_export_base_dir"


def export_base_dir(content) -> Path | None:
    """``ExportContent`` から相対画像パスの基準ディレクトリを取り出す。"""
    value = (content.metadata or {}).get(BASE_DIR_METADATA_KEY)
    return Path(value) if value else None


def resolve_image_path(src: str, base_dir: Path | None = None) -> Path | None:
    """画像の所在を実ファイルへ解決する。解決できなければ ``None``。

    - 絶対パスはそのまま、相対パスは ``base_dir`` (出力先ファイルのある
      ディレクトリ) から解決する。**プロセスの CWD は使わない**。
    - ``..`` を含むパスは拒否する (write_file の脱出検査と同じ方針)。
    - URL は取得しない。
    - 読めないものは落として WARNING。壊れた文書を書くより欠落を選ぶ。
    """
    text = (src or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered.startswith(_URL_PREFIXES):
        logger.warning("image source is a URL; writers do not fetch it: %s", text)
        return None

    if ".." in text.replace("\\", "/").split("/"):
        logger.warning("image source contains a '..' segment, rejected: %s", text)
        return None

    path = Path(text)
    if not path.is_absolute():
        if base_dir is None:
            logger.warning(
                "relative image source needs an output directory, skipped: %s", text,
            )
            return None
        path = base_dir / path

    try:
        if not path.is_file():
            logger.warning("image source not found, skipped: %s", path)
            return None
    except OSError as e:
        logger.warning("image source is not readable (%r), skipped: %s", e, path)
        return None
    return path


def scaled_width_cm(path: Path, max_width_cm: float = DEFAULT_MAX_WIDTH_CM) -> float:
    """埋め込み時の幅 (cm)。実寸が分からなければ上限をそのまま返す。"""
    try:
        from PIL import Image

        with Image.open(path) as img:
            width_px = img.width
    except Exception:
        return max_width_cm
    if width_px <= 0:
        return max_width_cm
    # 96dpi 換算で実寸へ落とし、上限で頭打ちにする。
    natural_cm = width_px / 96.0 * 2.54
    return min(natural_cm, max_width_cm)
