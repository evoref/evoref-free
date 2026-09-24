"""llama-server が実際に載せているモデルと config の照合 (c_05 §0.5.7)。

モデル移行は ``model_state.json`` と学習パーティションを切り替えるが、稼働中の
llama-server は差し替えない (再起動が要る)。2026-08-12 の調査では、この乖離に
気付かないまま **62.8 時間**別モデルが serve され、学習パーティションが別モデルの
出力で汚染されていた。起動時とモデル切替時に ``/props`` の ``model_path`` を
照合し、違えば ``AppState.served_model_mismatch`` に残す (``data_health`` に出る)。

照合は接続時に取得済みのメタデータで行い、``/props`` を改めて叩かない。
応答の生成中 (``is_serving_user``) は照合しない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.log_config import get_logger

logger = get_logger("factory.served_model")


def served_model_path(state: Any) -> str:
    """接続中の llama-server の ``/props`` の ``model_path`` (無ければ ``model_id``)。"""
    client = getattr(state, "local_client", None) if state is not None else None
    metadata = getattr(client, "metadata", None)
    return str(
        getattr(metadata, "model_path", "") or getattr(metadata, "model_id", "") or "",
    )


def _same_model(served: str, expected: Path) -> bool:
    """同じモデルか。両方のファイルが読めれば model_key、読めなければファイル名で比べる。"""
    served_path = Path(served)
    if served_path.is_file() and expected.is_file():
        from backend.model_key import ModelKeyError, model_identity

        try:
            return model_identity(served_path).model_key == model_identity(expected).model_key
        except ModelKeyError:
            pass
    return served_path.name == expected.name


def check_served_model(state: Any, expected_model: str | Path, project_root: Path) -> None:
    """serve 中のモデルが ``expected_model`` と同じか照合し、結果を state に残す。

    起動をブロックはしない (再起動すれば解消する運用上の乖離で、degraded では
    メタデータが取れない)。一致すれば記録を消し、食い違えば WARNING と
    ``state.served_model_mismatch`` (``served_filename`` / ``expected_filename`` /
    ``recommendation``)。メタデータが無い・応答の生成中なら何もしない。

    Args:
        expected_model: 載っているべきモデル (相対ならインストール根基準)。
    """
    if state is None or not expected_model:
        return
    llm = getattr(state, "llm_client", None)
    if llm is not None and getattr(llm, "is_serving_user", False):
        return
    served = served_model_path(state)
    if not served:
        return
    expected = Path(expected_model)
    if not expected.is_absolute():
        expected = Path(project_root) / expected
    if _same_model(served, expected):
        state.served_model_mismatch = None
        return

    served_name = Path(served).name
    recommendation = (
        "Restart llama-server (scripts/evoref-ctl.bat stop && start) so the "
        "served model matches the configured one. Until then, chat responses "
        "and any learning written to the active partition come from "
        f"'{served_name}', not '{expected.name}'."
    )
    logger.warning(
        "Served model mismatch: llama-server /props=%s, expected=%s. %s",
        served, expected, recommendation,
    )
    state.served_model_mismatch = {
        "served_filename": served_name,
        "expected_filename": expected.name,
        "recommendation": recommendation,
    }


__all__ = ["check_served_model", "served_model_path"]
