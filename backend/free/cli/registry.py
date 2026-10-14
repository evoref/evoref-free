"""CLI サブコマンドの登録表 (docs/f_06 §2.9.2)

1 コマンド 1 行の :class:`CommandSpec` を並べ、``main.py`` はここを引くだけにする。
``entry`` は ``"module:func"`` の文字列で、:func:`resolve` が呼ばれたときに初めて
import する (表を読むだけでは import しない)。

- Free のコマンドは :data:`FREE_COMMANDS`。
- Pro のコマンドは ``backend/pro/cli/commands.py`` が import 時に
  :func:`register_cli_command` で登録する (create の ``register_create_hook`` と同型)。
  Free は Pro のモジュールパスを持たず、名前の集合 :data:`PRO_COMMAND_NAMES` だけを持つ。
  登録が無ければ「Pro 専用」と案内して終了コード 1。
- 撤去したコマンド :data:`RETIRED` は Free / Pro とも「撤去された」と案内して終了コード 1。

Pro の登録の読み込みは :func:`_bootstrap` が初回に 1 度だけ行う。CLI が Pro 以上で
動いているとき (:func:`backend.free.cli.cli_mode.is_cli_pro_edition`、
``EVOREF_EDITION=free`` なら偽) だけ ``backend.pro.cli.commands`` を lazy import する。
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from backend.free.cli.cli_mode import is_cli_pro_edition
from backend.log_config import get_logger

logger = get_logger("cli.registry")

type Handler = Callable[[list[str]], int]
type CommandMode = Literal["api", "offline", "dual", "local"]

#: f_06 §2.9.3 の規約 (0 成功 / 1 失敗・拒否 / 2 実行不能 / 130 中断)。
EXIT_STANDARD = "standard"
#: 既存の Pro コマンドの例外: 検証の失敗・稼働中での拒否も 2 (互換のため維持、p_04 §5.1)。
EXIT_LEGACY_VALIDATION_2 = "legacy: validation failure / refused while running = 2"


@dataclass(frozen=True)
class CommandSpec:
    """サブコマンド 1 つの登録内容。"""

    name: str
    #: ``"module:func"``。``None`` は ``main`` が自分で扱う語 (chat / create のモード指定)。
    entry: str | None
    edition: Literal["free", "pro"]
    summary_key: str
    #: api = 稼働中の API だけ / offline = 停止中だけ (単一書き手ロック) /
    #: dual = API → 繋がらなければ停止中の直接 / local = backend にもロックにも依らない
    mode: CommandMode
    #: 使う API の接頭辞 (CLI⇔API 対応表の正本)。書式は ``api_parity`` のキーと同じ
    #: (``"/api/x"`` = 全メソッド、``"GET /api/x"`` = そのメソッドだけ。どちらもパスの接頭辞)
    api: tuple[str, ...] = ()
    exit_semantics: str = EXIT_STANDARD
    #: API を介さず同等の機能を直接実装している API (例: hub status は台帳を直接読む)。書式は ``api`` と同じ
    direct_for: tuple[str, ...] = ()


def _free(
    name: str, entry: str | None, mode: CommandMode, api: tuple[str, ...] = (),
) -> CommandSpec:
    return CommandSpec(
        name=name, entry=entry, edition="free",
        summary_key=f"cli.command_summary.{name}", mode=mode, api=api,
    )


_CLI = "backend.free.cli"
#: chat / create (対話の起動・終了と REPL の /status)。/api/health は auto-serve の起動待ち
_CHAT_APIS = (
    "/api/chat", "GET /api/status", "POST /api/sessions/register", "DELETE /api/sessions",
    "/api/mode/switch", "/api/health",
)

FREE_COMMANDS: tuple[CommandSpec, ...] = (
    _free("serve", f"{_CLI}.main:_run_serve_subcmd", "local"),
    _free("chat", None, "api", _CHAT_APIS),
    _free("create", None, "api", _CHAT_APIS),
    _free("gui", f"{_CLI}.main:_run_gui", "local"),
    _free("export", f"{_CLI}.data_commands:run_export", "offline"),
    _free("import", f"{_CLI}.data_commands:run_import", "offline"),
    _free("reindex", f"{_CLI}.reindex_command:run_reindex", "api",
          ("/api/rag/reindex", "POST /api/model/reembed-facts")),
    _free("projectmap", f"{_CLI}.projectmap_command:run_projectmap", "api",
          ("/api/rag/project_map",)),
    _free("theme", f"{_CLI}.theme_command:run_theme", "api",
          ("POST /api/themes/install", "POST /api/themes/install-url", "DELETE /api/themes")),
    _free("config", f"{_CLI}.config_command:run_config", "dual", ("/api/config",)),
    # --stop-services は最初に正常停止 API を呼ぶ (pid_manager.graceful_stop_backend、c_09 §2.1)
    _free("reset", f"{_CLI}.reset_command:run_reset", "offline", ("POST /api/system/shutdown",)),
    _free("forget", f"{_CLI}.forget_command:run_forget", "offline"),
    _free("doctor", f"{_CLI}.doctor_command:run_doctor_command", "local"),
    _free("backups", f"{_CLI}.backups_command:run_backups", "local"),
    _free("tune", f"{_CLI}.tune_command:run_tune", "dual", ("/api/system/auto-tune",)),
    _free("rag", f"{_CLI}.rag_command:run_rag", "api",
          ("/api/rag/calibrate-thresholds", "/api/rag/ingest", "/api/config")),
    _free("model", f"{_CLI}.model_commands:run_model", "api", (
        "GET /api/model/state", "POST /api/model/migrate", "POST /api/model/rollback",
        "POST /api/model/reload", "GET /api/model/migration-history",
        # --component embedding の migrate / rollback / migration-history
        "/api/model/{component}", "POST /api/model/reembed-confirm",
    )),
    _free("growth", f"{_CLI}.growth_command:run_growth", "dual", (
        "GET /api/learning/growth", "POST /api/learning/growth/proposals",
        "POST /api/learning/growth/rollback",
        # regression list / retire / unretire
        "/api/learning/growth/regression",
    )),
    _free("prompts", f"{_CLI}.prompts_command:run_prompts", "api", ("/api/prompts", "/api/aux-prompts")),
    _free("fewshot", f"{_CLI}.fewshot_command:run_fewshot", "api", ("/api/learning/fewshot",)),
    _free("runs", f"{_CLI}.runs_command:run_runs", "api", ("/api/create/runs",)),
    # remove はテンプレートのパッケージ (tpl-<hex12>) だけをカートリッジの削除 API で消す
    _free("template", f"{_CLI}.template_command:run_template", "api",
          ("/api/templates", "DELETE /api/cartridges")),
    _free("status", f"{_CLI}.status_command:run_status", "api", ("GET /api/status",)),
    # llama-server の状態・起動・停止 (所有者 LlamaProcessManager の API。GUI の /api/servers は互換層)
    _free("servers", f"{_CLI}.servers_command:run_servers", "api", ("/api/model/process",)),
    # forget --id の ID を探す (/api/memory の他のルートは REPL の /memory・/pin が使う)
    _free("memory", f"{_CLI}.memory_command:run_memory", "api", ("GET /api/memory/facts",)),
)

#: Pro のコマンド名。実体の登録は ``backend/pro/cli/commands.py`` が行う。
PRO_COMMAND_NAMES: frozenset[str] = frozenset({
    "knowledge", "audit", "cartridge", "policy", "airgap", "hub", "modelfit",
})

#: 撤去したコマンド (Level 2 の撤去、2026-10-10)。
RETIRED: frozenset[str] = frozenset({"lora", "learn"})

_FREE_BY_NAME: dict[str, CommandSpec] = {s.name: s for s in FREE_COMMANDS}

#: main が argv から拾う語の全体 (Free + Pro の名前 + 撤去済み)。
COMMAND_WORDS: frozenset[str] = frozenset(_FREE_BY_NAME) | PRO_COMMAND_NAMES | RETIRED

_pro_specs: dict[str, CommandSpec] = {}
_bootstrapped: bool = False


def register_cli_command(spec: CommandSpec) -> None:
    """Pro のコマンドを登録する (``backend.pro.cli.commands`` が import 時に呼ぶ)。

    名前は :data:`PRO_COMMAND_NAMES` に在るものだけを受ける (Free の名前の集合を正本に保つ)。
    後勝ち (テストで差し替えられるようにする)。
    """
    if spec.edition != "pro" or spec.name not in PRO_COMMAND_NAMES:
        raise ValueError(f"not a Pro command name: {spec.name!r} (edition={spec.edition})")
    _pro_specs[spec.name] = spec


def reset_cli_registry() -> None:
    """テスト用: Pro の登録 (REPL のスラッシュコマンドの差し込みを含む) を消し、
    bootstrap をやり直せる状態へ戻す。"""
    global _bootstrapped
    from backend.free.cli.command_parser import reset_slash_registry

    _pro_specs.clear()
    reset_slash_registry()
    _bootstrapped = False


def _bootstrap() -> None:
    """初回だけ、Pro 以上なら Pro のコマンド登録を読み込む。"""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    if not is_cli_pro_edition():
        return
    try:
        # 初回の import が register_cli_command を呼ぶ。ImportError は Pro 未同梱 (Free 配布) の正常系。
        import backend.pro.cli.commands as pro_commands
    except ImportError:
        logger.debug("backend.pro.cli.commands not available; Pro commands are not registered")
        return
    # reset_cli_registry の後は import がキャッシュで空振りするので登録し直す
    pro_commands.register_all()


def pro_commands() -> dict[str, CommandSpec]:
    """登録済みの Pro のコマンド (Free / ``EVOREF_EDITION=free`` では空)。"""
    _bootstrap()
    return dict(_pro_specs)


def get_spec(name: str) -> CommandSpec | None:
    """名前の登録内容。Pro は登録済みのときだけ返す。"""
    if name in _FREE_BY_NAME:
        return _FREE_BY_NAME[name]
    if name in PRO_COMMAND_NAMES:
        return pro_commands().get(name)
    return None


def load_entry(entry: str) -> Handler:
    """``"module:func"`` を import して関数を返す。"""
    module_name, _, func_name = entry.partition(":")
    return getattr(importlib.import_module(module_name), func_name)


def _notice_handler(key: str, name: str) -> Handler:
    def _handler(argv: list[str]) -> int:  # noqa: ARG001
        from backend.i18n_helper import init_i18n, msg

        init_i18n()
        print(msg(key, command=name), file=sys.stderr)
        return 1

    return _handler


def resolve(name: str) -> Handler | None:
    """サブコマンド名のハンドラを返す。

    - Free のコマンド: ``entry`` の関数 (chat / create は ``None`` — main がモードとして扱う)
    - Pro のコマンド: 登録があれば ``entry`` の関数。登録が無い / 実体を import できなければ
      「Pro 専用」の案内 (終了コード 1)
    - 撤去したコマンド: 「撤去された」の案内 (終了コード 1)
    - それ以外: ``None``
    """
    if name in RETIRED:
        return _notice_handler("cli.retired_command", name)
    spec = get_spec(name)
    if name in PRO_COMMAND_NAMES:
        if spec is None:
            return _notice_handler("cli.pro_only_command", name)
        try:
            return load_entry(spec.entry or "")
        except ImportError:
            logger.debug("Pro command %s could not be imported", name)
            return _notice_handler("cli.pro_only_command", name)
    if spec is None or spec.entry is None:
        return None
    return load_entry(spec.entry)
