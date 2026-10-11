"""チーム知識ハブの夜間書き出しの差込口 (docs/p_10 §5.2)。

Free は **Protocol と hook だけ** を持つ (e_01 §0.1 原則 4)。公開リスト・署名付き `.evocart` の
書き出し・取り込みは Pro (``backend/pro/team_hub/``) が持ち、``setup_pro`` 側で
``register_pro_handler(TEAM_HUB_EXPORTER_HANDLER, exporter)`` するだけ。

- 呼出元は sleep-time の Full (Step 5.885)。アイドル窓で回すだけで、書き出しは
  Evidence Store を **読むだけ** (配布物はデータ根の外へ書く、CLAUDE.md §6 #2)。
- 登録が無ければ (Free 単独・Pro でも自動書き出しが無効) :func:`run_team_hub_export` は 0 を
  返して何もしない。書き出しの失敗も 0 に畳んで WARNING を出す (サイクルを落とさない)。
"""

from __future__ import annotations

from typing import Protocol

from backend.log_config import get_logger

logger = get_logger("core.team_hub")

#: Plugin Registry の handler 名 (``backend.edition.register_pro_handler``)。
TEAM_HUB_EXPORTER_HANDLER = "team_hub_exporter"


class TeamHubExporter(Protocol):
    """夜間の書き出し器 (Pro が実装して登録する)。"""

    async def run_scheduled(self) -> int:
        """間隔と設定を見て必要なら 1 回書き出す。書き出した `.evocart` の件数 (0 / 1) を返す。"""
        ...


async def run_team_hub_export() -> int:
    """登録された書き出し器を回す。登録が無い・失敗したら 0。"""
    from backend.edition import get_pro_handler

    exporter: TeamHubExporter | None = get_pro_handler(TEAM_HUB_EXPORTER_HANDLER)
    if exporter is None:
        return 0
    try:
        return int(await exporter.run_scheduled())
    except Exception as exc:  # noqa: BLE001 - 書き出しの失敗で sleep-time を落とさない
        logger.warning("Team hub export failed: %r", exc)
        return 0


__all__ = ["TEAM_HUB_EXPORTER_HANDLER", "TeamHubExporter", "run_team_hub_export"]
