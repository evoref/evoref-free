"""言語パックの検証コマンド実行 (c_16 §4.5.4、段階 C-3)。

corpus の ``language/`` セクション (§4.5.3) が運ぶ ``verify`` 宣言で、実際の
処理系にコードを通す (``node --check`` / ``cargo check`` …)。**パッケージ由来の
宣言で実プロセスを起動する唯一の経路**なので、安全側の規則が本体 — 迷ったら
「実行しない」側へ倒し、判断はすべて ``log_decision`` へ 1 行残す
(f_10 §4.3、失敗 → repair へ、実行できない → 検査しない)。

Loop pillar は corpus (``backend/free/rag/``) を top-level import できない
(``backend/free/tests/test_pillar_boundary.py``)。検証済みの宣言は
:class:`VerifyCommand` (corpus 側 ``language.VerifyCommand`` の複製 —
``store.py`` の ``PROJECT_MAP_PACKAGE_KIND`` と同じ理由) で受け取り、
corpus → loop の変換は呼出側 (api 層、``chat_stream_staged.py``) が行う。

実行は既存の :class:`~backend.free.loop.action_runner.ActionRunner`
(``shell=False``) を使う。ただし ``ActionRunner._is_command_allowed`` は
ファイル名の stem が一致するかだけを見るため、verify 用には追加の関門を
このモジュールで課す:

- **承認単位は実行ファイル名ではなく argv 全体**
  (``create.staged.verify.commands``、例: ``["node", "--check", "{file}"]``)。
  パックの宣言 ``[executable, *args]`` が承認済み argv のどれかと
  完全一致 (先頭の実行ファイル名だけ大文字小文字を無視) したときだけ
  実行する。**実行ファイル名だけを allow-list にすると、PC の持ち主が
  ``python`` や ``node`` を許可した瞬間、パックはその処理系の ``-c`` /
  ``-e`` 経由で任意コードを実行する verify を宣言できてしまう** —
  「パックは選ぶだけで増やせない」が実行ファイル名にしか効いておらず、
  2026-09-20 レビューで指摘された穴 (旧 ``executables`` 設計の削除)。
- 実行ファイルの解決は ``shutil.which`` を使わず、``PATH`` の各ディレクトリを
  直接舐める。``shutil.which`` は bare name 探索時に **Windows で常に CWD
  (カレントディレクトリ) を検索対象へ先頭挿入する** (cpython
  ``shutil._win_path_needs_curdir`` — ``path=`` を明示しても効果が無い、
  2026-09-20 実装時に cpython 実装を確認)。サーバプロセスの実際の CWD が
  どこであっても検索に混ざってしまうため、ここでは PATH のディレクトリ
  だけを手で調べる。**相対パスの PATH エントリ (``.`` 等) は飛ばす** —
  相対のままだとプロセスの実際の CWD (リポジトリルート等、ワークスペースの
  祖先になりうる) 基準で解決され、CWD 経由の問題を PATH の側面から
  持ち込んでしまう。
- Windows では解決先の拡張子が ``.exe`` のときだけ実行する (``.cmd`` /
  ``.bat`` は cmd.exe 経由になり引数注入の既知クラスがある)。
- 解決先がワークスペースの中にある候補は無視して次の PATH ディレクトリへ
  進む (生成物が自分で置いた実行ファイルを踏ませない)。
- 検証対象ファイル (``{file}``) はワークスペースの中でなければ実行しない。
- **子プロセスの環境変数は最小限に絞る** (:data:`_MINIMAL_ENV_VARS`)。
  ``ActionRunner._run_command`` は ``env=`` を渡さないとサーバプロセスの
  環境を丸ごと継承する (secrets を含みうる) ため、verify はここで
  ``RunCommandAction.env`` に明示的な最小 dict を渡す。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.free.harness.action import RunCommandAction
from backend.free.loop.action_runner import ActionRunner, ActionRunnerConfig, SubprocessRunner
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("loop.staged.language_verify")

#: 失敗 detail に残す stdout+stderr の末尾バイト数 (f_10 §4.3 の記述量に合わせる)。
_DETAIL_TAIL_CHARS = 2000

#: verify の子プロセスへ渡す最小環境変数 (存在するものだけ写す)。secrets を
#: 含みうるサーバプロセスの環境を丸ごと継承させない (2026-09-20 レビュー)。
_MINIMAL_ENV_VARS: tuple[str, ...] = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOME",
    "LOCALAPPDATA", "APPDATA", "PATHEXT", "LANG", "LC_ALL",
)


@dataclass(frozen=True, slots=True)
class VerifyCommand:
    """1 件の検証コマンド (corpus 側 ``language.VerifyCommand`` の複製、c_16 §4.5.4)。

    構造検証済みの値だけを持つ — 実行時の allow-list / 解決可否はここでは
    判定しない (:func:`evaluate_verify_commands` の役目)。
    """

    id: str
    executable: str
    args: tuple[str, ...]
    timeout_sec: float
    success_exit_codes: tuple[int, ...]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _candidate_filenames(executable: str) -> tuple[str, ...]:
    if sys.platform == "win32":
        return (f"{executable}.exe",)
    return (executable,)


def resolve_verify_executable(executable: str, workspace_root: Path) -> tuple[Path | None, str]:
    """``PATH`` のディレクトリだけを直接探す (CWD 経由の暗黙解決を避ける)。

    Windows では解決先が ``.exe`` のときだけ有効。ワークスペース配下で
    見つかった候補は無視して次の ``PATH`` ディレクトリへ進む。

    ``workspace_root`` は呼出側の状態に関わらずここで必ず ``resolve()`` する
    — 候補側は ``candidate.resolve()`` で正規化 (8.3 短縮名解決込み) される
    ため、比較対象の ``workspace_root`` を未解決のまま渡す呼出しがあると
    ``%TEMP%`` の短縮名 (例: ``LONGUS~1``) と解決後の長い名前
    (``longusername``) が食い違い、ワークスペース内の候補を「外」と
    誤判定して素通りさせる (2026-09-20 実機確認で検出)。
    """
    try:
        workspace_root = workspace_root.resolve()
    except OSError:
        return None, "workspace_root could not be resolved"
    path_env = os.environ.get("PATH", "")
    if not path_env:
        return None, "PATH is empty"
    names = _candidate_filenames(executable)
    for directory in path_env.split(os.pathsep):
        directory = directory.strip()
        if not directory:
            continue
        base = Path(directory)
        if not base.is_absolute():
            # 相対エントリ (``.`` 等) はプロセスの実際の CWD 基準で解決され、
            # CWD がリポジトリルート (ワークスペースの祖先) になりうる
            # 環境では素性の知れないファイルを拾いかねない。飛ばす。
            continue
        for name in names:
            candidate = base / name
            try:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
            except OSError:
                continue
            if sys.platform == "win32" and resolved.suffix.lower() != ".exe":
                continue
            if _is_within(resolved, workspace_root):
                # 生成物が自分で置いた実行ファイルを踏まない。次の候補へ。
                continue
            if sys.platform != "win32" and not os.access(resolved, os.X_OK):
                continue
            return resolved, ""
    return None, f"executable {executable!r} was not found on PATH outside the workspace"


def _matches_approved_command(
    cmd: VerifyCommand, approved_commands: tuple[tuple[str, ...], ...],
) -> bool:
    """パックの宣言 (``[executable, *args]``) が承認済み argv と完全一致するか。

    承認単位は argv 全体 — 実行ファイル名だけの allow-list だと、
    インタプリタ (``python``/``node``) を許可した瞬間にパックが
    ``-c``/``-e`` 経由の任意コード実行を verify として宣言できてしまう
    (c_16 §4.5.4、2026-09-20 レビュー)。先頭要素 (実行ファイル名) だけ
    大文字小文字を無視し、残りの引数は完全一致 (プレースホルダ込みの
    宣言そのものを比較する — レンダー後の絶対パスとは比較しない)。
    """
    declared = (cmd.executable, *cmd.args)
    for approved in approved_commands:
        if not approved or len(declared) != len(approved):
            continue
        if declared[0].lower() != approved[0].lower():
            continue
        if tuple(declared[1:]) == tuple(approved[1:]):
            return True
    return False


def _minimal_env() -> dict[str, str]:
    """verify の子プロセスへ渡す最小環境 (存在する変数だけ写す)。"""
    return {name: os.environ[name] for name in _MINIMAL_ENV_VARS if name in os.environ}


def _render_args(args: tuple[str, ...], *, workspace: Path, target: Path) -> tuple[str, ...] | None:
    """``{file}`` / ``{workspace}`` を絶対パスへ置換する (要素単位、連結なし)。

    manifest 読み込み時 (``language.parse_verify_command``) で既に「ちょうど
    一致 or 中括弧を含まない」を検証済みだが、防御的にもう一度確認する。
    """
    out: list[str] = []
    for a in args:
        if a == "{file}":
            out.append(str(target))
        elif a == "{workspace}":
            out.append(str(workspace))
        elif "{" in a or "}" in a:
            return None
        else:
            out.append(a)
    return tuple(out)


def _log_decision(
    debug_logger: "DebugLogger | None", *,
    language_id: str, command_id: str, chosen: str, reason: str,
    scope: str, detail: str = "",
) -> None:
    if debug_logger is None:
        return
    try:
        debug_logger.log_decision(
            decision_point="language_verify",
            chosen=chosen,
            candidates=["run", "skip"],
            reason=reason,
            context={
                "language": language_id,
                "verify_id": command_id,
                "detail": detail[:500] if detail else "",
            },
            scope=scope,
        )
    except Exception as exc:  # noqa: BLE001 - 記録失敗で本処理を止めない
        logger.debug("language_verify log_decision failed: %s", exc)


def evaluate_verify_commands(
    commands: tuple[VerifyCommand, ...],
    *,
    language_id: str,
    enabled: bool,
    approved_commands: tuple[tuple[str, ...], ...],
    timeout_cap_sec: float,
    workspace_root: Path,
    target_file: Path,
    debug_logger: "DebugLogger | None" = None,
    scope: str = "request",
    runner: "SubprocessRunner | None" = None,
) -> str | None:
    """検証コマンド列を順に評価する。

    Returns:
        失敗の detail 文字列 (repair へ流す)。実行しなかった/全部通過した
        場合は ``None`` (検査しない、または成功 — どちらも生成物は採用する)。

    ``enabled=False`` のときは宣言を一切読まず、決定を 1 行だけ記録して
    ``None`` を返す。
    """
    if not enabled:
        _log_decision(
            debug_logger, language_id=language_id, command_id="*",
            chosen="skip", reason="verify_disabled", scope=scope,
        )
        return None
    if not commands:
        return None

    try:
        ws_root = workspace_root.resolve()
        target = target_file.resolve()
    except OSError:
        _log_decision(
            debug_logger, language_id=language_id, command_id="*",
            chosen="skip", reason="workspace_or_target_unresolvable", scope=scope,
        )
        return None

    if not _is_within(target, ws_root):
        _log_decision(
            debug_logger, language_id=language_id, command_id="*",
            chosen="skip", reason="target_outside_workspace", scope=scope,
        )
        return None

    for cmd in commands:
        if not _matches_approved_command(cmd, approved_commands):
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="skip", reason="command_not_approved", scope=scope,
            )
            continue

        resolved, why = resolve_verify_executable(cmd.executable, ws_root)
        if resolved is None:
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="skip", reason=why, scope=scope,
            )
            continue

        args = _render_args(cmd.args, workspace=ws_root, target=target)
        if args is None:
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="skip", reason="args_render_failed", scope=scope,
            )
            continue

        timeout = min(cmd.timeout_sec, timeout_cap_sec)
        ar_kwargs: dict = dict(
            config=ActionRunnerConfig(
                allowed_write_roots=(ws_root,),
                allowed_commands=(resolved.name,),
                command_timeout_sec=timeout,
            ),
            repo_root=ws_root,
        )
        if runner is not None:
            ar_kwargs["runner"] = runner
        ar = ActionRunner(**ar_kwargs)
        action = RunCommandAction(
            command=(str(resolved), *args), cwd=str(ws_root), env=_minimal_env(),
        )
        result = ar.run_one(action)
        meta = result.metadata or {}

        if meta.get("timeout") == "true":
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="skip", reason="timeout", scope=scope,
            )
            continue
        if meta.get("exec_error") == "true":
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="skip", reason="exec_error", scope=scope,
            )
            continue

        rc_raw = meta.get("returncode", "")
        try:
            rc: int | None = int(rc_raw) if rc_raw != "" else None
        except ValueError:
            rc = None
        if rc is None:
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="skip", reason="no_returncode", scope=scope,
            )
            continue

        if rc in cmd.success_exit_codes:
            _log_decision(
                debug_logger, language_id=language_id, command_id=cmd.id,
                chosen="pass", reason="verify_passed", scope=scope,
            )
            continue

        tail = ((result.output or "") + (result.error or ""))[-_DETAIL_TAIL_CHARS:]
        detail = f"{language_id} verify '{cmd.id}' failed (exit {rc}): {tail}"
        _log_decision(
            debug_logger, language_id=language_id, command_id=cmd.id,
            chosen="fail", reason="verify_failed", scope=scope, detail=detail,
        )
        return detail
    return None


#: 拡張子 (または source_path) → その言語の検証コマンド列。空タプルなら
#: 検証対象なし。``StagedCreateExecutor.language_verify_lookup`` の型
#: (api 層が corpus の ``CorpusStore.language_overlay()`` から組む)。
LanguageVerifyLookup = Callable[[str], tuple[VerifyCommand, ...]]


__all__ = [
    "LanguageVerifyLookup",
    "VerifyCommand",
    "evaluate_verify_commands",
    "resolve_verify_executable",
]
