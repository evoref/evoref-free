"""evoref config サブコマンド (c_05 §7.6)。

``normalize``: 版 (``config_version``) の古い ``config.yaml`` (と ``config.pro.yaml``)
を現行の形へ一度だけ直す (実体は :mod:`backend.config_normalize`)。単一書き手ロック
(``<data_root>/g1/store/.writer.lock``) の下で動かし、serve の稼働中 (ロックが
取られている) は書き換えずに拒否する。

``evoref serve`` / ``evoref-ctl start`` / setup は起動前に
``config normalize --if-needed`` を呼ぶ (現行の版の config には何もしない)。

``show [section]`` / ``get <section.key>`` / ``set <section.key> <value>``: GUI の
設定画面と同じ操作。backend が動いていれば ``/api/config`` 系 (能力キーの保護・
再起動要否の判定は backend 側) を使い、動いていなければ ``config.yaml`` を直接
読み書きする。直接書くときは書き手ロックを取る (serve 稼働中は拒否) — 能力キーを
変えられるのは evoref を止めた PC の持ち主だけ、という経路を保つため (docs/c_06 §1.5)。
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import httpx
import yaml
from pydantic import ValidationError
from rich.console import Console

from backend.config_normalize import (
    CONFIG_VERSION,
    NewerConfigVersionError,
    NormalizeReport,
    needs_normalize,
    normalize_config,
)
from backend.data_root import DataRootError, resolve_data_root, store_root
from backend.free.cli.backend_headers import backend_headers
from backend.free.cli.cli_common import resolve_backend_url
from backend.free.cli.config_loader import _find_project_root, _setup_encoding
from backend.free.cli.renderer import create_console, render_error, render_info
from backend.i18n_helper import init_i18n, msg
from backend.io.writer_lock import WriterLockHeld, acquire_config_write_lock, acquire_writer_lock
from backend.log_config import get_logger

logger = get_logger("cli.config")

_TIMEOUT = 30.0
_MISSING = object()

#: Free では書けないセクション (``backend/free/api/config/config_api.py`` の
#: ``PRO_ONLY_SECTIONS`` と同じ)。
_PRO_ONLY_SECTIONS = frozenset({"widget_proxy", "mode_models", "pro"})


def config_needs_normalize(config_path: Path) -> bool:
    """``config_path`` (と隣の ``config.pro.yaml``) が現行より古い版か。無ければ ``False``。"""
    return needs_normalize(config_path)


def normalize_under_lock(config_path: Path, data_root: Path) -> NormalizeReport:
    """書き手ロックを取って正規化する。serve が稼働中なら :class:`WriterLockHeld`。"""
    lock = acquire_writer_lock(store_root(data_root))
    try:
        # config.yaml はインストール根に 1 つ。別のデータ根で動く serve も排他にする
        config_lock = acquire_config_write_lock(config_path.parent)
        try:
            return normalize_config(config_path)
        finally:
            config_lock.release()
    finally:
        lock.release()


def render_normalize_report(console: Console, report: NormalizeReport) -> None:
    """正規化の結果 (退避先・落としたキー・値が不正なキー) を表示する。"""
    if not report.changed:
        render_info(console, msg("cli.config_normalize_unchanged"))
        return
    if report.backup is not None:
        render_info(console, msg("cli.config_normalize_done", backup=str(report.backup)))
    if report.pro_backup is not None:
        render_info(console, msg("cli.config_normalize_pro_done", backup=str(report.pro_backup)))
    if report.renamed:
        render_info(
            console,
            msg(
                "cli.config_normalize_renamed",
                count=len(report.renamed), keys=", ".join(report.renamed),
            ),
        )
    if report.dropped:
        render_info(
            console,
            msg(
                "cli.config_normalize_dropped",
                count=len(report.dropped), keys=", ".join(report.dropped),
            ),
        )
    if report.invalid:
        render_error(
            console,
            msg(
                "cli.config_normalize_invalid",
                count=len(report.invalid), keys="; ".join(report.invalid),
            ),
            level="warning",
        )


def normalize_before_start(project_root: Path, console: Console) -> int | None:
    """起動前の正規化 (現行の版の config には何もしない)。

    Returns:
        続行できるなら ``None``。serve の稼働中・書き換えの失敗は終了コード ``1``。
    """
    config_path = project_root / "config.yaml"
    try:
        if not config_needs_normalize(config_path):
            return None
        data_root = resolve_data_root(root=project_root)
        report = normalize_under_lock(config_path, data_root)
    except WriterLockHeld as e:
        render_error(console, msg("cli.config_normalize_locked", detail=str(e)))
        return 1
    except (OSError, ValueError, yaml.YAMLError, DataRootError) as e:
        logger.error("config normalize failed: %s", e)
        render_error(console, msg("cli.config_normalize_failed", detail=str(e)))
        return 1
    render_normalize_report(console, report)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoref config",
        description=msg("cli.help_config"),
    )
    parser.add_argument("action", choices=["normalize", "show", "get", "set"])
    parser.add_argument(
        "key", nargs="?", default=None, help="show: section / get, set: section.key",
    )
    parser.add_argument(
        "value", nargs="?", default=None, help="set: the value (parsed as YAML)",
    )
    parser.add_argument(
        "--if-needed", action="store_true",
        help="Do nothing (and take no lock) when config.yaml already has the current config_version",
    )
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help="Data root whose writer lock guards the rewrite "
        "(default: EVOREF_DATA_ROOT or <install_root>/userdata)",
    )
    parser.add_argument(
        "--backend-url", default=None,
        help="Backend URL (default: http://localhost:<server.port in config.yaml, else 8000>)",
    )
    parser.add_argument("--json", action="store_true", help="Print values as JSON")
    return parser


def run_config(argv: list[str]) -> int:
    """同期エントリーポイント (``evoref config normalize|show|get|set``)。"""
    if not _setup_encoding():
        return 1
    init_i18n()
    args = _build_parser().parse_args(argv)
    console = create_console()
    root = _find_project_root()
    if args.action != "normalize":
        return _run_value_action(args, root, console)
    config_path = root / "config.yaml"
    if not config_path.is_file():
        if args.if_needed:
            return 0
        render_error(console, msg("cli.config_not_found", path=str(config_path)))
        return 1
    try:
        if args.if_needed and not config_needs_normalize(config_path):
            return 0
        data_root = resolve_data_root(args.data_root, root=root)
        report = normalize_under_lock(config_path, data_root)
    except WriterLockHeld as e:
        render_error(console, msg("cli.config_normalize_locked", detail=str(e)))
        return 1
    except DataRootError as e:
        render_error(console, msg("cli.data_root_invalid", detail=str(e)))
        return 1
    except (OSError, ValueError, yaml.YAMLError) as e:
        logger.error("config normalize failed: %s", e)
        render_error(console, msg("cli.config_normalize_failed", detail=str(e)))
        return 1
    render_normalize_report(console, report)
    return 0


# ── show / get / set ──


class _BackendError(Exception):
    """backend が非 200 を返した (表示用の理由と検証エラーを持つ)。"""

    def __init__(self, status: int, detail: str, errors: list[str]) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.errors = errors


def _error_detail(resp: httpx.Response) -> tuple[str, list[str]]:
    """非 200 応答から (理由, 検証エラーの一覧) を取り出す。"""
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return f"HTTP {resp.status_code}", []
    if isinstance(detail, dict):
        errors = [str(e) for e in detail.get("errors") or []]
        message = str(detail.get("message") or "").strip()
        return message or f"HTTP {resp.status_code}", errors
    if isinstance(detail, str) and detail.strip():
        return detail.strip(), []
    return f"HTTP {resp.status_code}", []


def _request(client: httpx.Client, method: str, url: str, **kwargs) -> dict | None:
    """backend を呼ぶ。届かなければ ``None`` (= 停止中)、非 200 は :class:`_BackendError`。"""
    try:
        resp = client.request(method, url, **kwargs)
    except httpx.ConnectError:
        return None
    if resp.status_code != 200:
        detail, errors = _error_detail(resp)
        raise _BackendError(resp.status_code, detail, errors)
    return resp.json()


def parse_value(raw: str) -> object:
    """``set`` の値を YAML として読む (``true`` / ``0.3`` / ``[a, b]``)。読めなければ文字列のまま。

    空文字は空文字のまま (YAML では null)。先頭が 0 の整数 (``010``) は 10 進で読む
    (YAML 1.1 では 8 進の 8)。
    """
    if raw == "":
        return raw
    if re.fullmatch(r"[-+]?0[0-9]+", raw):
        return int(raw, 10)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    # 日付 (2026-01-01) と 60 進数 (1:20 → 80) は YAML 1.1 の暗黙変換で、文字列のつもりの値を変えてしまう
    if isinstance(value, (datetime.date, datetime.datetime)):
        return raw
    if isinstance(value, int) and not isinstance(value, bool) and re.fullmatch(r"[\d_]+(:[0-5]?\d)+", raw):
        return raw
    return value


_NULL_LITERALS = frozenset({"null", "Null", "NULL", "~"})


def _leaf_annotation(section: str, path: list[str]) -> object | None:
    """スキーマで ``section.path`` のフィールドの型注釈 (辿れなければ ``None``)。"""
    from pydantic import BaseModel

    from backend.schemas import EvorefConfig

    model: type[BaseModel] = EvorefConfig
    keys = [section, *path]
    for i, key in enumerate(keys):
        field = model.model_fields.get(key)
        if field is None:
            return None
        if i == len(keys) - 1:
            return field.annotation
        nested = [
            c for c in (field.annotation, *get_args(field.annotation))
            if isinstance(c, type) and issubclass(c, BaseModel)
        ]
        if not nested:
            return None
        model = nested[0]
    return None


def _text_field(annotation: object) -> tuple[bool, bool]:
    """(文字列だけを取るフィールドか, ``None`` も取るか)。``Literal`` の文字列も文字列とみなす。"""
    args = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    nullable = type(None) in args
    rest = [a for a in args if a is not type(None)]

    def is_text(a: object) -> bool:
        return a is str or (get_origin(a) is Literal and all(isinstance(x, str) for x in get_args(a)))

    return bool(rest) and all(is_text(a) for a in rest), nullable


def parse_value_for(section: str, path: list[str], raw: str) -> object:
    """``set`` の値をスキーマのフィールド型に合わせて読む。

    文字列だけを取るフィールドでは YAML として読まない (``instance.name 123`` は ``"123"``、
    ``off`` は ``False`` にしない)。``str | None`` の ``null`` / ``~`` は ``None``。
    それ以外は :func:`parse_value`。
    """
    annotation = _leaf_annotation(section, path) if path else None
    if annotation is not None:
        text, nullable = _text_field(annotation)
        if text:
            return None if nullable and raw in _NULL_LITERALS else raw
    return parse_value(raw)


def split_key(dotted: str) -> tuple[str, list[str]]:
    """``section.a.b`` を ``("section", ["a", "b"])`` に分ける。空の成分は ValueError。"""
    parts = dotted.split(".")
    if any(not p for p in parts):
        raise ValueError(dotted)
    return parts[0], parts[1:]


def dig(data: object, path: list[str]) -> object:
    """入れ子の dict を辿る (リストは数字の添字で辿る)。無ければ ``_MISSING``。"""
    node = data
    for key in path:
        if isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
            continue
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _nest(path: list[str], value: object) -> object:
    for key in reversed(path):
        value = {key: value}
    return value


def _format_value(value: object, as_json: bool) -> str:
    if as_json:
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip()
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _local_config(root: Path) -> dict:
    """停止中の読み手: serve と同じ規則で読み込み・検証した設定 (既定値の補完込み)。"""
    from backend.config import read_config_file

    return read_config_file(root / "config.yaml", root)


def _run_value_action(args: argparse.Namespace, root: Path, console: Console) -> int:
    base = resolve_backend_url(args.backend_url, root)
    try:
        with httpx.Client(headers=backend_headers(), timeout=_TIMEOUT) as client:
            if args.action == "set":
                return _run_set(args, root, console, client, base)
            return _run_read(args, root, console, client, base)
    except _BackendError as e:
        render_error(console, msg("cli.config_failed", detail=e.detail))
        return 1
    except (httpx.HTTPError, ValidationError, OSError, yaml.YAMLError) as e:
        logger.error("config %s failed: %s", args.action, e)
        render_error(console, msg("cli.config_failed", detail=str(e)))
        return 1


def _run_read(
    args: argparse.Namespace, root: Path, console: Console, client: httpx.Client, base: str,
) -> int:
    if args.action == "get" and not args.key:
        render_error(console, msg("cli.config_get_usage"))
        return 1
    try:
        section, path = split_key(args.key) if args.key else ("", [])
    except ValueError:
        render_error(console, msg("cli.config_key_not_found", name=args.key))
        return 1
    if args.action == "show" and path:
        render_error(console, msg("cli.config_get_usage"))
        return 1

    remote = _request(client, "GET", f"{base}/api/config")
    if remote is not None:
        config = remote.get("config") or {}
        source = msg("cli.config_source_backend", url=base)
    else:
        config = _local_config(root)
        source = msg("cli.config_source_file", path=str(root / "config.yaml"))

    value: object = config
    if section:
        value = dig(config, [section, *path])
        if value is _MISSING:
            render_error(console, msg("cli.config_key_not_found", name=args.key))
            return 1
    if args.action == "show" and not args.json:
        render_info(console, source)
    print(_format_value(value, args.json))
    return 0


def _run_set(
    args: argparse.Namespace, root: Path, console: Console, client: httpx.Client, base: str,
) -> int:
    if not args.key or args.value is None:
        render_error(console, msg("cli.config_set_usage"))
        return 1
    try:
        section, path = split_key(args.key)
    except ValueError:
        render_error(console, msg("cli.config_set_usage"))
        return 1
    value = parse_value_for(section, path, args.value)
    if not path and not isinstance(value, dict):
        render_error(console, msg("cli.config_set_usage"))
        return 1
    data = _nest(path, value)
    shown = _format_value(value, as_json=False)

    try:
        overridden = _launch_args_overrides(client, base, root, section, data)
        if overridden:
            render_error(console, msg(
                "cli.config_set_launch_args_override",
                keys=", ".join(name for name, _launch in overridden),
                launch=", ".join(dict.fromkeys(launch for _name, launch in overridden)),
            ))
            return 1
        checked = _request(
            client, "POST", f"{base}/api/config/{section}/validate", json={"data": data},
        )
        if checked is None:
            return _set_offline(args, root, console, section, data, shown)
        if not checked.get("valid", False):
            render_error(console, msg(
                "cli.config_set_invalid", name=args.key,
                errors="; ".join(checked.get("errors") or []),
            ))
            return 1
        result = _request(client, "PUT", f"{base}/api/config/{section}", json={"data": data})
    except _BackendError as e:
        if e.errors:
            render_error(console, msg(
                "cli.config_set_invalid", name=args.key, errors="; ".join(e.errors),
            ))
        else:
            render_error(console, msg("cli.config_set_refused", detail=e.detail))
        if e.status == 403 and section != "model_paths" and section not in _PRO_ONLY_SECTIONS:
            render_error(console, msg("cli.config_set_capability_hint"), level="hint")
        return 1
    if result is None:
        # validate の後に backend が落ちた。書けたかどうか分からないので直書きはしない。
        render_error(console, msg("cli.backend_not_running"))
        return 1
    render_info(console, msg("cli.config_set_done", name=args.key, value=shown))
    render_info(console, msg(
        "cli.config_set_restart_required" if result.get("restart_required", True)
        else "cli.config_set_applied_live",
    ))
    return 0


def _launch_args_overrides(
    client: httpx.Client, base: str, root: Path, section: str, data: object,
) -> list[tuple[str, str]]:
    """書こうとするキーのうち、``launch_args`` が写すために書いても効かないもの。

    ``launch_args`` の ``-c`` / ``-ngl`` / ``-np`` 等は対応するキーを上書きする (c_16 §7.2.4)。
    稼働中の API はそのキーを黙って落とし、停止中に書いても起動時に launch_args が勝つ。
    どちらも成功と表示して効かないので、CLI で先に止める。

    Returns:
        ``(書こうとしたキー, その launch_args のキー)`` の一覧。
    """
    from backend.llama_launch_args import LAUNCH_ARGS_SECTIONS, derived_keys

    full = {section: data}
    wanted: list[tuple[str, tuple[str, ...], dict]] = []
    for kind, kpath in LAUNCH_ARGS_SECTIONS.items():
        node = dig(full, list(kpath))
        if isinstance(node, dict) and set(node) - {"launch_args"}:
            wanted.append((kind, kpath, node))
    if not wanted:
        return []

    current: object = None
    out: list[tuple[str, str]] = []
    for kind, kpath, node in wanted:
        if "launch_args" in node:
            text = node["launch_args"]
        else:
            if current is None:
                remote = _request(client, "GET", f"{base}/api/config")
                current = (remote.get("config") or {}) if remote is not None else _raw_config(root)
            text = dig(current, [*kpath, "launch_args"])
        if not isinstance(text, str) or not text:
            continue
        derived = derived_keys(kind, text)
        launch = ".".join((*kpath, "launch_args"))
        out.extend((".".join((*kpath, key)), launch) for key in node if key in derived)
    return out


def _raw_config(root: Path) -> dict:
    """停止中の ``launch_args`` の読み手: 検証前の ``config.yaml`` (と Pro の上書き)。"""
    from backend.config import _read_raw_config_file

    try:
        return _read_raw_config_file(root / "config.yaml", root) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _set_offline(
    args: argparse.Namespace, root: Path, console: Console,
    section: str, data: object, shown: str,
) -> int:
    """backend 停止中の ``set``: 書き手ロックの下で ``config.yaml`` を直接書く。"""
    from backend.config import save_config_section
    from backend.edition import Edition, current_edition
    from backend.schemas import EvorefConfig

    config_path = root / "config.yaml"
    if not config_path.is_file():
        render_error(console, msg("cli.config_not_found", path=str(config_path)))
        return 1
    if section not in EvorefConfig.model_fields:
        render_error(console, msg("cli.config_section_not_found", section=section))
        return 1
    if section in _PRO_ONLY_SECTIONS and current_edition() < Edition.PRO:
        render_error(console, msg("cli.config_pro_only_section", section=section))
        return 1
    if section == "model_paths":
        blocked = _tracked_model_paths_changes(config_path, data)
        if blocked:
            render_error(console, msg(
                "cli.config_set_model_paths_migrate", keys=", ".join(blocked),
            ))
            return 1

    try:
        # config.yaml はインストール根に 1 つ。データ根に関わらず稼働中の serve を排他にする
        lock = acquire_config_write_lock(root)
    except WriterLockHeld as e:
        render_error(console, msg("cli.config_set_locked", detail=str(e)))
        return 1
    try:
        save_config_section(section, data, project_root=root, reload=False)
    except NewerConfigVersionError as e:
        render_error(console, msg(
            "cli.config_newer_version", version=e.version, current=CONFIG_VERSION,
        ))
        return 1
    except ValidationError as e:
        errors = "; ".join(str(err["msg"]) for err in e.errors())
        render_error(console, msg("cli.config_set_invalid", name=args.key, errors=errors))
        return 1
    finally:
        lock.release()
    logger.info("config.yaml updated while the backend was stopped: %s", args.key)
    render_info(console, msg("cli.config_set_done", name=args.key, value=shown))
    render_info(console, msg("cli.config_set_offline_done"))
    return 0


def _tracked_model_paths_changes(config_path: Path, data: object) -> list[str]:
    """``model_state`` が追う model_paths のキーを変える要求か (API の 403 と同じ規則)。

    これらはモデル移行でしか変えない。config を直書きすると ``model_state.json`` と
    ずれて起動時に不一致になる。
    """
    from backend.free.core.model_migration import MODEL_STATE_TRACKED_KEYS

    if not isinstance(data, dict):
        return []
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    current = raw.get("model_paths") or {}
    return sorted(
        k for k in MODEL_STATE_TRACKED_KEYS
        if k in data and str(data[k] or "") != str(current.get(k, "") or "")
    )
