"""evoref backups サブコマンド — 上書き前の退避 (``bk/overwrite/``) の一覧と、指定した場所への復元。

    evoref backups list [--limit N] [--name 部分一致]
    evoref backups restore BACKUP --to DEST

退避は ``<日付>/<stamp>_<hex6>_<元の名前>`` で、元のパスは記録していない (docs/f_11 §5.6)。
そのため復元は戻し先 ``--to`` を必ず指定する。BACKUP は退避の置き場の中のファイルだけ
(絶対パス、または置き場の中の一意なファイル名)。DEST に違う中身が既にあれば、書き手が
上書きの前に退避する (復元も元に戻せる)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: 退避ファイル名 ``<stamp>_<hex6>_<元の名前>`` から元の名前を取る (stamp は ``utc_compact_stamp``)。
_NAME_PARTS = 3


def _build_parser() -> argparse.ArgumentParser:
    from backend.i18n_helper import msg

    parser = argparse.ArgumentParser(prog="evoref backups", description=msg("cli.help_backups"))
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    ls = sub.add_parser("list", help="List overwrite backups, newest first")
    ls.add_argument("--limit", type=int, default=20, help="Number of entries (default 20)")
    ls.add_argument("--name", default=None, help="Only entries whose original name contains this text")
    rs = sub.add_parser("restore", help="Copy a backup to the place you name")
    rs.add_argument("backup", help="Backup file (absolute path, or a unique file name under the backup folder)")
    rs.add_argument("--to", required=True, metavar="DEST", help="Where to put the restored file")
    return parser


def _original_name(path: Path) -> str:
    parts = path.name.split("_", _NAME_PARTS - 1)
    return parts[-1] if len(parts) == _NAME_PARTS else path.name


def _entries(root: Path) -> list[Path]:
    """退避ファイルを新しい順に返す (stamp は辞書順 = 時刻順の固定長なので名前で並べてよい)。"""
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    return sorted(files, key=lambda p: p.name, reverse=True)


def _find_backup(root: Path, text: str) -> Path:
    """BACKUP 引数を退避の置き場の中のファイルに解決する。外や曖昧なものは ValueError。"""
    candidate = Path(text)
    if candidate.is_absolute() or candidate.exists():
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            raise ValueError(f"not under the backup folder: {candidate}") from None
        if not resolved.is_file():
            raise ValueError(f"not a file: {candidate}")
        return resolved
    matches = [p for p in _entries(root) if p.name == text]
    if len(matches) != 1:
        raise ValueError(f"{'no' if not matches else 'multiple'} backup named {text!r} under {root}")
    return matches[0]


def run_backups(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref backups``)。終了コード: 0 = 成功 / 1 = 失敗。"""
    from backend.config import PathResolver
    from backend.data_root import DataRootError, resolve_data_root
    from backend.free.cli.config_loader import _find_project_root
    from backend.i18n_helper import init_i18n, msg
    from backend.io.user_file import UserFileWriteError, write_user_bytes

    init_i18n()
    args = _build_parser().parse_args(argv)
    try:
        data_root = resolve_data_root(args.data_root, root=_find_project_root())
    except DataRootError as e:
        print(msg("cli.data_root_invalid", detail=str(e)), file=sys.stderr)
        return 1
    root = PathResolver.layout_path(data_root, "backup_overwrite_dir")

    if args.action == "list":
        entries = _entries(root)
        if args.name:
            entries = [p for p in entries if args.name in _original_name(p)]
        if not entries:
            print(msg("cli.backups_empty"))
            return 0
        for path in entries[: max(1, args.limit)]:
            print(f"{path}  {path.stat().st_size} bytes  {_original_name(path)}")
        return 0

    try:
        source = _find_backup(root, args.backup)
        result = write_user_bytes(Path(args.to), source.read_bytes())
    except (ValueError, OSError, UserFileWriteError) as e:
        print(msg("cli.backups_failed", detail=str(e)), file=sys.stderr)
        return 1
    print(msg("cli.backups_restored", dest=str(result.path), size=result.bytes_written))
    if result.backup is not None:
        print(msg("cli.backups_previous_saved", backup=str(result.backup)))
    return 0
