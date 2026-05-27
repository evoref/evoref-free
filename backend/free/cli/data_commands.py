"""evoref export/import サブコマンド: ローカルデータのエクスポート・インポート

Free: 5カテゴリ一括エクスポート / merge インポート
Pro:  カテゴリ選択式エクスポート / merge+replace+dry-run インポート
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from backend.free.cli.renderer import (
    create_console,
    render_error,
    render_info,
    render_table,
)
from backend.i18n_helper import init_i18n, msg
from backend.log_config import get_logger

logger = get_logger("cli.data")

_DEFAULT_BACKEND = "http://localhost:8000"
_TIMEOUT = 60.0
_DOWNLOAD_TIMEOUT = 300.0

ALL_CATEGORIES = ["memory", "experience", "rag", "prompts", "lora", "cartridges", "history"]
DEFAULT_CATEGORIES = ["memory", "experience", "rag", "prompts", "lora", "cartridges"]


# ────────────────────────────────────────────
# エディション判定ヘルパー
# ────────────────────────────────────────────


def _check_pro_edition(backend_url: str) -> bool:
    """バックエンドの /api/status からエディション判定"""
    try:
        resp = httpx.get(f"{backend_url}/api/status", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("edition", "FREE") != "FREE"
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        pass
    return False


# ────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────


def build_export_parser() -> argparse.ArgumentParser:
    """export サブコマンド用パーサー"""
    parser = argparse.ArgumentParser(
        prog="evoref export",
        description="Export local data as a ZIP package",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path (directory or file path)",
    )
    parser.add_argument(
        "--categories",
        help="Categories to export (comma-separated) [Pro]",
    )
    parser.add_argument(
        "--exclude",
        help="Categories to exclude (comma-separated) [Pro]",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Backend host (default: localhost)",
    )
    parser.add_argument(
        "--port", default=8000, type=int,
        help="Backend port (default: 8000)",
    )
    return parser


def build_import_parser() -> argparse.ArgumentParser:
    """import サブコマンド用パーサー"""
    parser = argparse.ArgumentParser(
        prog="evoref import",
        description="Import data from an export package",
    )
    parser.add_argument(
        "file",
        help="Path to .evoref-export.zip file",
    )
    parser.add_argument(
        "--mode", default="merge",
        choices=["merge", "replace"],
        help="Import mode (default: merge) [replace: Pro]",
    )
    parser.add_argument(
        "--categories",
        help="Categories to import (comma-separated) [Pro]",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview import without applying changes [Pro]",
    )
    parser.add_argument(
        "--skip-reembed", action="store_true",
        help="Skip re-embedding after import",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Backend host (default: localhost)",
    )
    parser.add_argument(
        "--port", default=8000, type=int,
        help="Backend port (default: 8000)",
    )
    return parser


# ────────────────────────────────────────────
# export エントリーポイント
# ────────────────────────────────────────────


def run_export(argv: list[str]) -> int:
    """export サブコマンドのエントリーポイント"""
    parser = build_export_parser()
    args = parser.parse_args(argv)

    init_i18n()
    console = create_console(no_color=args.no_color)
    backend_url = f"http://{args.host}:{args.port}"

    is_pro = _check_pro_edition(backend_url)

    # カテゴリ決定
    categories = _resolve_export_categories(args, console, is_pro)
    if categories is None:
        return 1

    logger.debug("run_export: categories=%s, output=%s, pro=%s", categories, args.output, is_pro)

    # Pro: プレビュー取得・表示
    if is_pro:
        preview = _export_preview(backend_url, console, categories)
        if preview is None:
            return 1
        _display_export_preview(console, preview, categories)
    else:
        render_info(console, msg("cli.export_starting"))

    # エクスポート実行（ZIP ダウンロード）
    return _export_download(backend_url, console, categories, args.output)


def _resolve_export_categories(
    args: argparse.Namespace, console, is_pro: bool,
) -> list[str] | None:
    """--categories / --exclude からエクスポート対象カテゴリを決定

    Free: 常に None（API 側で FREE_CATEGORIES 固定）
    Pro:  カテゴリ選択・除外対応
    """
    if not is_pro:
        # Free 版: categories/exclude は無視
        return []  # 空リスト = API にカテゴリ指定なし（Free 版デフォルト）

    if args.categories and args.exclude:
        render_error(console, msg("cli.export_categories_conflict"))
        return None

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        invalid = set(cats) - set(ALL_CATEGORIES)
        if invalid:
            render_error(
                console,
                msg("cli.export_invalid_category", categories=", ".join(sorted(invalid))),
            )
            return None
        return cats

    if args.exclude:
        excluded = {c.strip() for c in args.exclude.split(",")}
        invalid = excluded - set(ALL_CATEGORIES)
        if invalid:
            render_error(
                console,
                msg("cli.export_invalid_category", categories=", ".join(sorted(invalid))),
            )
            return None
        return [c for c in DEFAULT_CATEGORIES if c not in excluded]

    return DEFAULT_CATEGORIES


def _export_preview(
    backend_url: str, console, categories: list[str],
) -> dict | None:
    """エクスポートプレビューを API から取得（Pro 専用）"""
    try:
        resp = httpx.post(
            f"{backend_url}/api/data/export/preview",
            json={"categories": categories},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None  # preview 非対応（Free）
        render_error(console, f"API error: {e.response.status_code}")
        return None


def _display_export_preview(
    console, preview: dict, categories: list[str],
) -> None:
    """エクスポートプレビューを表示"""
    render_info(console, msg("cli.export_starting"))
    render_info(console, "")

    for cat_info in preview.get("categories", []):
        if cat_info["name"] in categories and cat_info["included"]:
            render_info(
                console,
                f"  {cat_info['name']:<14} {cat_info['details']}",
            )

    render_info(console, "")


def _export_download(
    backend_url: str, console, categories: list[str],
    output_arg: str | None,
) -> int:
    """エクスポート ZIP をダウンロードして保存"""
    try:
        body = {"categories": categories} if categories else None
        with httpx.stream(
            "POST",
            f"{backend_url}/api/data/export",
            json=body,
            timeout=_DOWNLOAD_TIMEOUT,
        ) as resp:
            resp.raise_for_status()

            # Content-Disposition からファイル名を取得
            cd = resp.headers.get("content-disposition", "")
            default_filename = "evoref-export.evoref-export.zip"
            if "filename=" in cd:
                default_filename = cd.split("filename=")[-1].strip('" ')

            # 出力先決定
            output_path = _resolve_output_path(output_arg, default_filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # ストリーミング書き込み
            total_bytes = 0
            with open(output_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    total_bytes += len(chunk)

    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return 1
    except OSError as e:
        render_error(console, msg("cli.export_write_error", detail=str(e)))
        return 1

    size_mb = total_bytes / (1024 * 1024)
    render_info(
        console,
        msg("cli.export_completed", path=str(output_path), size=f"{size_mb:.1f}"),
    )
    logger.debug("Export completed: %s (%.1f MB)", output_path, size_mb)
    return 0


def _resolve_output_path(output_arg: str | None, default_filename: str) -> Path:
    """出力先パスを解決"""
    if output_arg is None:
        return Path.cwd() / default_filename

    output = Path(output_arg)
    if output.is_dir():
        return output / default_filename
    # ファイルパス指定（拡張子チェックなし — ユーザー指定を尊重）
    return output


# ────────────────────────────────────────────
# import エントリーポイント
# ────────────────────────────────────────────


def run_import(argv: list[str]) -> int:
    """import サブコマンドのエントリーポイント"""
    parser = build_import_parser()
    args = parser.parse_args(argv)

    init_i18n()
    console = create_console(no_color=args.no_color)
    backend_url = f"http://{args.host}:{args.port}"

    # ファイル存在チェック
    zip_path = Path(args.file)
    if not zip_path.exists():
        render_error(console, msg("cli.import_file_not_found", path=str(zip_path)))
        return 1
    if not zip_path.is_file():
        render_error(console, msg("cli.import_not_a_file", path=str(zip_path)))
        return 1

    is_pro = _check_pro_edition(backend_url)

    # Free 版制約の適用
    if not is_pro:
        args.mode = "merge"
        args.dry_run = False
        args.categories = None

    # カテゴリ解決
    cat_list = None
    if args.categories:
        cat_list = [c.strip() for c in args.categories.split(",")]
        invalid = set(cat_list) - set(ALL_CATEGORIES)
        if invalid:
            render_error(
                console,
                msg("cli.export_invalid_category", categories=", ".join(sorted(invalid))),
            )
            return 1

    logger.debug(
        "run_import: file=%s, mode=%s, categories=%s, dry_run=%s, skip_reembed=%s, pro=%s",
        zip_path, args.mode, cat_list, args.dry_run, args.skip_reembed, is_pro,
    )

    # 1. ドライランで互換性確認・プレビュー
    preview_result = _import_execute(
        backend_url, console, zip_path,
        mode=args.mode,
        categories=cat_list,
        dry_run=True,
        skip_reembed=args.skip_reembed,
    )
    if preview_result is None:
        return 1

    # 2. プレビュー表示
    _display_import_preview(console, preview_result)

    # ドライランの場合はここで終了
    if args.dry_run:
        render_info(console, msg("cli.import_dry_run_only"))
        return 0

    # 3. 埋め込みモデル不一致チェック・確認プロンプト
    compat = preview_result.get("compatibility", {})
    if not compat.get("embedding_match", True) and compat.get("reembed_required", False):
        if not args.skip_reembed:
            render_info(console, "")
            render_info(console, msg("cli.import_reembed_warning"))
            try:
                answer = input(msg("cli.import_confirm_prompt")).strip().lower()
            except (KeyboardInterrupt, EOFError):
                render_info(console, "")
                render_info(console, msg("cli.import_cancelled"))
                return 0
            if answer not in ("y", "yes"):
                render_info(console, msg("cli.import_cancelled"))
                return 0

    # 4. 実際のインポート実行
    render_info(console, "")
    render_info(
        console,
        msg("cli.import_starting", mode=args.mode),
    )

    result = _import_execute(
        backend_url, console, zip_path,
        mode=args.mode,
        categories=cat_list,
        dry_run=False,
        skip_reembed=args.skip_reembed,
    )
    if result is None:
        return 1

    # 5. 結果表示
    _display_import_results(console, result)
    return 0


def _import_execute(
    backend_url: str, console,
    zip_path: Path,
    mode: str,
    categories: list[str] | None,
    dry_run: bool,
    skip_reembed: bool,
) -> dict | None:
    """インポート API を呼び出し"""
    try:
        with open(zip_path, "rb") as f:
            files = {"file": (zip_path.name, f, "application/zip")}
            data: dict = {
                "mode": mode,
                "dry_run": str(dry_run).lower(),
                "skip_reembed": str(skip_reembed).lower(),
            }
            if categories:
                data["categories"] = ",".join(categories)

            resp = httpx.post(
                f"{backend_url}/api/data/import",
                files=files,
                data=data,
                timeout=_DOWNLOAD_TIMEOUT,
            )

        if resp.status_code == 400:
            detail = resp.json().get("detail", "Bad request")
            render_error(console, msg("cli.import_validation_error", detail=detail))
            return None
        if resp.status_code == 409:
            render_error(console, msg("cli.import_conflict"))
            return None
        resp.raise_for_status()
        return resp.json()

    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return None
    except httpx.HTTPStatusError as e:
        render_error(console, f"API error: {e.response.status_code}")
        return None
    except OSError as e:
        render_error(console, msg("cli.import_read_error", detail=str(e)))
        return None


def _display_import_preview(console, result: dict) -> None:
    """インポートプレビュー（互換性情報 + カテゴリ概要）を表示"""
    render_info(console, msg("cli.import_validating"))
    render_info(console, "")

    # 互換性情報
    compat = result.get("compatibility", {})
    emb_mark = "✓" if compat.get("embedding_match", False) else "✗"
    base_mark = "✓" if compat.get("base_model_match", False) else "✗"

    render_info(
        console,
        f"  {msg('cli.import_embedding_check')}: {emb_mark}",
    )
    render_info(
        console,
        f"  {msg('cli.import_base_model_check')}: {base_mark}",
    )

    # カテゴリ別結果
    results = result.get("results", [])
    if results:
        render_info(console, "")
        rows = []
        for r in results:
            rows.append({
                msg("cli.import_col_category"): r["category"],
                msg("cli.import_col_action"): r["action"],
                msg("cli.import_col_added"): str(r.get("added", 0)),
                msg("cli.import_col_updated"): str(r.get("updated", 0)),
                msg("cli.import_col_skipped"): str(r.get("skipped", 0)),
            })

        headers = [
            msg("cli.import_col_category"),
            msg("cli.import_col_action"),
            msg("cli.import_col_added"),
            msg("cli.import_col_updated"),
            msg("cli.import_col_skipped"),
        ]
        render_table(console, rows, headers)

    # 警告表示
    for r in results:
        for w in r.get("warnings", []):
            render_info(console, f"  ⚠ {w}")


def _display_import_results(console, result: dict) -> None:
    """インポート結果を表示"""
    results = result.get("results", [])
    if not results:
        render_info(console, msg("cli.import_no_results"))
        return

    render_info(console, "")
    for r in results:
        added = r.get("added", 0)
        updated = r.get("updated", 0)
        skipped = r.get("skipped", 0)
        parts = []
        if added:
            parts.append(f"+{added} {msg('cli.import_added')}")
        if updated:
            parts.append(f"{updated} {msg('cli.import_updated')}")
        if skipped:
            parts.append(f"{skipped} {msg('cli.import_skipped_label')}")
        detail = " / ".join(parts) if parts else msg("cli.import_col_action")

        render_info(console, f"  {r['category']:<14} {detail}")

        for w in r.get("warnings", []):
            render_info(console, f"    ⚠ {w}")

    render_info(console, "")
    render_info(console, msg("cli.import_completed"))
