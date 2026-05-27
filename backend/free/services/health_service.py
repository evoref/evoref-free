"""ヘルスチェックサービス

各種モデルサーバーへのヘルスチェックを提供する。
sync / async 両対応。
"""

from __future__ import annotations

import httpx

from backend.log_config import get_logger

logger = get_logger("services.health_service")

# デフォルトのヘルスチェックタイムアウト（秒）
HEALTH_CHECK_TIMEOUT = 2.0


def check_health_sync(host: str, port: int) -> bool:
    """同期ヘルスチェック"""
    try:
        resp = httpx.get(
            f"http://{host}:{port}/health",
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        return resp.status_code == 200
    except Exception:
        return False


async def check_health_async(host: str, port: int) -> bool:
    """非同期ヘルスチェック"""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as c:
            r = await c.get(f"http://{host}:{port}/health")
            return r.status_code == 200
    except Exception:
        return False
