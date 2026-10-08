"""evoref doctor サブコマンド — データ根の全件の照合 (読むだけ、G1 設計 §10.4)。

検査の本体は :mod:`backend.free.doctor`。ここは引数・表示・終了コードだけ。

終了コード: 0 = error なし / 1 = error あり / 2 = 走らせられない (データ根が無い・不正、
束の置き場が ``store/`` の中、想定外の例外)。serve の稼働中も走る (書き手ロックは取らず、
結果が変わりうることを 1 行出す)。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

#: 走らせられなかったときの終了コード。
EXIT_CANNOT_RUN = 2


def _build_parser() -> argparse.ArgumentParser:
    from backend.i18n_helper import msg

    parser = argparse.ArgumentParser(prog="evoref doctor", description=msg("cli.help_doctor"))
    parser.add_argument("--json", action="store_true", help="Print the machine-readable report")
    parser.add_argument(
        "--bundle", default=None, metavar="PATH",
        help="Write a bug-report zip (report, format versions, redacted log tails; no store content)",
    )
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root to check (default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    return parser


def _finding_line(finding: dict[str, Any]) -> str:
    from backend.i18n_helper import msg

    text = msg(
        f"cli.doctor_finding.{finding['code']}",
        format_id=finding.get("format_id") or "-",
        count=finding["count"],
        detail=finding.get("detail") or "",
    )
    paths = finding.get("paths") or []
    where = f" ({', '.join(paths)}{', …' if finding['count'] > len(paths) and paths else ''})" if paths else ""
    return f"[{msg('cli.doctor_severity.' + finding['severity'])}] {text}{where}"


def render_human(report: dict[str, Any], data_root: Path) -> list[str]:
    """人向けの要約 (i18n)。"""
    from backend.i18n_helper import msg

    summary = report["summary"]
    lines = [msg("cli.doctor_header", data_root=str(data_root))]
    runtime = report.get("runtime") or {}
    if runtime:
        lines.append(msg("cli.doctor_runtime", python=runtime.get("python") or "?", deps=" / ".join(
            f"{name}: {msg('cli.doctor_dependency_installed' if ok else 'cli.doctor_dependency_missing')}"
            for name, ok in (runtime.get("optional_dependencies") or {}).items()
        )))
    if report.get("serve_running"):
        lines.append(msg("cli.doctor_serve_running"))
    lines.append(msg("cli.doctor_scanned", formats=len(report["formats"]), files=summary["files"]))
    for name, store in (report.get("evidence") or {}).items():
        lines.append(msg(
            "cli.doctor_evidence", store=name, total=store.get("total", 0), live=store.get("live", 0),
            retracted=store.get("retracted", 0), superseded=store.get("superseded", 0),
            ignored=store.get("ignored", 0),
        ))
    history = report.get("history") or {}
    experience = report.get("experience") or {}
    lines.append(msg(
        "cli.doctor_history", sessions=history.get("sessions", 0), active=history.get("active", 0),
        experience=experience.get("records", 0),
    ))
    health = report.get("learning_health") or {}
    if health:
        lines.append(msg(
            "cli.doctor_learning_health", verdict=health["verdict"], turns=health["turns"],
            failures=health["detected_failures"], corrections=health["verified_corrections"],
            span=health["span_days"], active=health["active_days"],
        ))
        lines.extend(f"  - {reason}" for reason in health["reasons"])
        lines.extend(
            "  - " + msg("cli.doctor_learning_unlearned", model_key=key, experiences=part.get("experiences", 0))
            for key, part in (health.get("unlearned_partitions") or {}).items()
        )
    lines.extend(_finding_line(f) for f in report["findings"])
    lines.append(msg(
        "cli.doctor_summary",
        errors=summary["errors"], warnings=summary["warnings"], info=summary["info"],
    ))
    return lines


def _active_model_key(project_root: Path, data_root: Path) -> str | None:
    """config.yaml の ``model_paths.base_model`` の ``model_key`` (起動時に束ねる active)。

    serve 中にモデルを切り替えていれば、そちらは反映しない。base_model が無ければ ``None``。
    """
    from backend.config import PathResolver
    from backend.free.cli.config_loader import _load_yaml_section

    model_paths = _load_yaml_section(project_root, "model_paths")
    if not isinstance(model_paths, dict) or not model_paths.get("base_model"):
        return None
    return PathResolver({"model_paths": model_paths}, project_root, data_root=data_root).active_model_key


def run_doctor_command(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref doctor``)。"""
    from backend.data_root import DataRootError, resolve_data_root
    from backend.free.cli.config_loader import _find_project_root
    from backend.free.doctor import DoctorError, exit_code, run_doctor, write_bundle
    from backend.i18n_helper import init_i18n, msg

    init_i18n()
    args = _build_parser().parse_args(argv)
    project_root = _find_project_root()
    try:
        data_root = resolve_data_root(args.data_root, root=project_root)
    except DataRootError as e:
        print(msg("cli.data_root_invalid", detail=str(e)), file=sys.stderr)
        return EXIT_CANNOT_RUN
    # 読み手の WARNING (退避・readonly) は報告の指摘と重複するので出さない。
    logging.disable(logging.ERROR)
    try:
        report = run_doctor(data_root, active_model_key=_active_model_key(project_root, data_root))
        bundle = write_bundle(report, data_root, Path(args.bundle)) if args.bundle else None
    except DoctorError as e:
        print(msg("cli.doctor_failed", detail=str(e)), file=sys.stderr)
        return EXIT_CANNOT_RUN
    except Exception as e:  # noqa: BLE001 — 検査器自身の失敗は 2 で返す
        print(msg("cli.doctor_failed", detail=f"{type(e).__name__}: {e}"), file=sys.stderr)
        return EXIT_CANNOT_RUN
    finally:
        logging.disable(logging.NOTSET)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for line in render_human(report, data_root):
            print(line)
    if bundle is not None:
        print(msg("cli.doctor_bundle_written", path=str(bundle)), file=sys.stderr if args.json else sys.stdout)
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(run_doctor_command(sys.argv[1:]))
