"""EML Extractor

.eml ファイルからテキスト部分を抽出する。
stdlib email モジュールを使用（外部依存なし）。
"""

from __future__ import annotations

import email
import email.policy
from pathlib import Path

from backend.extraction.base import ExtractionError, ExtractionResult
from backend.log_config import get_logger

logger = get_logger("extraction.email_eml")


class EmailEmlExtractor:
    """EML ファイルからテキストを抽出"""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".eml"})

    @property
    def requires(self) -> list[str]:
        return []

    def extract(self, path: Path) -> ExtractionResult:
        """EML ファイルからテキストを抽出"""
        raw = path.read_bytes()
        return self._parse_eml(raw, path.name)

    def extract_from_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """EML バイトデータからテキストを抽出"""
        return self._parse_eml(data, filename)

    def is_available(self) -> bool:
        return True

    def _parse_eml(self, data: bytes, filename: str) -> ExtractionResult:
        """EML データを解析してテキストを抽出"""
        msg = email.message_from_bytes(data, policy=email.policy.default)

        parts: list[str] = []

        # ヘッダー情報
        headers = []
        for key in ("From", "To", "Subject", "Date"):
            value = msg.get(key)
            if value:
                headers.append(f"{key}: {value}")
        if headers:
            parts.append("\n".join(headers))

        # 本文テキスト
        body = self._get_text_body(msg)
        if body:
            parts.append(body)

        text = "\n\n".join(parts)

        if not text.strip():
            raise ExtractionError("empty_content", f"No text in {filename}")

        logger.debug("EmailEmlExtractor: %s -> %d chars", filename, len(text))
        return ExtractionResult(
            text=text,
            metadata={
                "subject": msg.get("Subject", ""),
                "from": msg.get("From", ""),
            },
        )

    @staticmethod
    def _get_text_body(msg) -> str:
        """メールからテキスト本文を抽出"""
        if msg.is_multipart():
            texts: list[str] = []
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_content()
                    if isinstance(payload, str) and payload.strip():
                        texts.append(payload)
            return "\n\n".join(texts)
        else:
            if msg.get_content_type() == "text/plain":
                payload = msg.get_content()
                if isinstance(payload, str):
                    return payload
            return ""
