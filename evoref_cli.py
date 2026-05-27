"""evoref CLI エントリーポイント — カレントディレクトリの backend を優先参照する"""

import os
import sys


def main():
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from backend.free.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
