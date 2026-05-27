"""コマンド出力 API: run_command の全文出力取得"""

from fastapi import APIRouter

from backend.log_config import get_logger

logger = get_logger("api.commands")

router = APIRouter(prefix="/api/commands", tags=["commands"])


@router.get("/last-output")
async def get_last_output():
    """最後の run_command の全文出力を取得（/page コマンド用）

    切り詰めが行われた場合のみ全文が保持される。
    切り詰めが無かった場合は空の output を返す。
    """
    from backend.free.agent.tools.builtin import get_last_full_output

    output, total_lines = get_last_full_output()
    return {
        "output": output,
        "total_lines": total_lines,
        "has_output": bool(output),
    }


@router.delete("/last-output")
async def clear_last_output():
    """全文出力バッファをクリア"""
    from backend.free.agent.tools.builtin import clear_last_full_output

    clear_last_full_output()
    return {"cleared": True}
