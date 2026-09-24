"""CLI → backend のリクエストに付けるヘッダ (docs/c_06 §1.5)

backend のアクセス制御は、状態を変えるリクエストに ``X-Evoref-Client`` を要求し、
``server.allow_remote: true`` の構成では全リクエストに ``X-Evoref-Token`` を要求する。
CLI の HTTP 呼び出しは全てここのヘッダを付ける。
"""

from __future__ import annotations

import os
from pathlib import Path

CLIENT_HEADER = "X-Evoref-Client"
TOKEN_HEADER = "X-Evoref-Token"
TOKEN_ENV = "EVOREF_API_TOKEN"


def _token_file() -> Path:
    # backend 側 (backend/factory/_access_guard.py) と同じ置き場 (データ根の run/)。
    from backend.config import resolve_data_path

    return resolve_data_path("run_dir") / "api_token"


def backend_headers() -> dict[str, str]:
    """backend へのリクエストに付けるヘッダ。トークンがあれば付ける。"""
    headers = {CLIENT_HEADER: "cli"}
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        try:
            token = _token_file().read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    if token:
        headers[TOKEN_HEADER] = token
    return headers
