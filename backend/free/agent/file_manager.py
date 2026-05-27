"""セッションファイル管理: アップロード・一覧・取得・更新・削除"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from backend.extraction import get_registry
from backend.extraction.base import ExtractionError
from backend.log_config import get_logger

logger = get_logger("agent.file_manager")


def get_allowed_extensions() -> set[str]:
    """レジストリから対応拡張子を取得（遅延評価）"""
    registry = get_registry()
    exts = set(registry.supported_extensions())
    if not exts:
        # フォールバック: レジストリ未初期化時
        return {
            ".txt", ".md", ".json", ".yaml", ".yml", ".csv",
            ".py", ".js", ".ts", ".rb", ".pl", ".cgi", ".sh",
            ".html", ".htm", ".css", ".scss", ".sass", ".less",
            ".php", ".asp", ".aspx", ".jsp", ".twig", ".ejs", ".erb", ".cfm",
            ".xml", ".ini", ".conf", ".toml", ".env", ".htaccess", ".svg",
            ".pdf", ".docx", ".xlsx", ".pptx",
        }
    return exts


@dataclass
class SessionFile:
    """セッション内のファイル情報"""
    file_id: str
    filename: str
    size_bytes: int
    mime_type: str
    uploaded_at: str
    session_id: str


class SessionFileManager:
    """セッションスコープのファイル管理

    ファイルは local/tmp/{session_id}/ に保存され、
    セッション終了時に自動削除される。
    """

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self._files: dict[str, SessionFile] = {}

    def _session_dir(self, session_id: str) -> Path:
        d = self.tmp_dir / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def upload(
        self, data: bytes, filename: str, session_id: str,
    ) -> SessionFile:
        """ファイルをアップロード"""
        ext = Path(filename).suffix.lower()
        if ext not in get_allowed_extensions():
            raise ValueError(f"Unsupported file extension: {ext}")

        file_id = uuid.uuid4().hex[:12]
        dest = self._session_dir(session_id) / f"{file_id}_{filename}"
        dest.write_bytes(data)

        mime = _guess_mime(filename)
        info = SessionFile(
            file_id=file_id,
            filename=filename,
            size_bytes=len(data),
            mime_type=mime,
            uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            session_id=session_id,
        )
        self._files[file_id] = info
        logger.info("Uploaded file: %s (%d bytes)", filename, len(data))
        return info

    def list_files(self, session_id: str) -> list[SessionFile]:
        """セッション内のファイル一覧"""
        return [f for f in self._files.values() if f.session_id == session_id]

    def get_file(self, file_id: str) -> SessionFile | None:
        """ファイル情報を取得"""
        return self._files.get(file_id)

    def get_content(self, file_id: str) -> str | None:
        """ファイルの内容を取得（extraction モジュール経由）"""
        info = self._files.get(file_id)
        if info is None:
            return None
        path = self._file_path(info)
        if path is None or not path.exists():
            return None

        registry = get_registry()
        ext = Path(info.filename).suffix.lower()

        if registry.is_supported(ext):
            try:
                result = registry.extract(path)
                return result.text
            except ExtractionError:
                logger.warning("Extraction failed for %s, falling back to UTF-8", info.filename)
                return path.read_text(encoding="utf-8", errors="replace")
        else:
            return path.read_text(encoding="utf-8", errors="replace")

    def update_file(self, file_id: str, content: str) -> bool:
        """ファイルの内容を更新"""
        info = self._files.get(file_id)
        if info is None:
            return False
        path = self._file_path(info)
        if path is None:
            return False
        path.write_text(content, encoding="utf-8")
        info.size_bytes = len(content.encode("utf-8"))
        logger.info("Updated file: %s", info.filename)
        return True

    def delete_file(self, file_id: str) -> bool:
        """ファイルを削除"""
        info = self._files.pop(file_id, None)
        if info is None:
            return False
        path = self._file_path(info)
        if path and path.exists():
            path.unlink()
        logger.info("Deleted file: %s", info.filename)
        return True

    def _file_path(self, info: SessionFile) -> Path | None:
        """ファイルのディスク上のパスを返す"""
        session_dir = self.tmp_dir / info.session_id
        if not session_dir.exists():
            return None
        # file_id_filename 形式で保存されている
        for p in session_dir.iterdir():
            if p.name.startswith(info.file_id):
                return p
        return None


def _guess_mime(filename: str) -> str:
    """拡張子から MIME タイプを推定"""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".txt": "text/plain", ".md": "text/markdown",
        ".json": "application/json", ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml", ".csv": "text/csv",
        ".py": "text/x-python", ".js": "application/javascript",
        ".ts": "application/typescript", ".html": "text/html",
        ".htm": "text/html", ".css": "text/css",
        ".xml": "application/xml", ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    return mime_map.get(ext, "application/octet-stream")
