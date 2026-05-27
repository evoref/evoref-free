"""evoref CLI

CLI モジュールの公開 API:
- main() — エントリーポイント
- SessionState — セッション状態管理
- CommandResult — コマンド実行結果
"""

from backend.free.cli.command_parser import CommandResult, SessionState
from backend.free.cli.main import main

__all__ = ["CommandResult", "SessionState", "main"]
