"""危険コマンドパターン定義 + コマンドリテラル判定（Free / Pro 共有）"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

# バッククォートで囲まれた区間。ユーザーは「コマンド `dir E:\tmp\x` を実行して」の
# ようにコマンドを、`foo.txt` のように語の強調を、どちらもこの記法で書く。
_BACKTICK_SPAN_RE = re.compile(r"`([^`\n]+)`")


def _span_looks_like_command(span: str) -> bool:
    """バッククォート区間がコマンド呼び出しに見えるか判定する。

    保守的に「空白区切りで 2 トークン以上あり、第 1 トークンがパス・ファイル名
    でない」場合のみコマンドと見なす。``foo.txt`` / ``E:\\tmp\\x`` のような
    強調用途の単独パスは対象外に落とす。
    """
    tokens = span.split()
    if len(tokens) < 2:
        return False
    head = tokens[0]
    return not any(c in head for c in ("\\", "/", "."))


def strip_command_literals(text: str) -> str:
    """バッククォートで囲まれた **コマンド** を取り除いたテキストを返す。

    コマンドの引数として現れるパスは、読み書きの対象ではなくコマンドの一部
    である。ファイルパス抽出や書込み意図判定をこの位置のパスに反応させると、
    実行依頼がファイル操作に化ける (実インシデント 2026-07-29 ライブ監査:
    「コマンド `dir E:\\tmp\\no_such_dir_zzz` を実行して、返ってきた出力を
    そのまま報告してください」が、「出力」を書込み動詞と解釈したうえで
    コマンド引数のパスを書込み先に取り、幻覚した dir 出力 61 バイトを
    ``E:\\tmp\\no_such_dir_zzz`` へ書き込んで「完了しました」と報告した)。

    コマンド実行系の判定は生のクエリを見るため、本関数の戻り値は「パス抽出 /
    書込み意図判定」の入力としてのみ使うこと。
    """
    if "`" not in text:
        return text
    return _BACKTICK_SPAN_RE.sub(
        lambda m: " " if _span_looks_like_command(m.group(1)) else m.group(0),
        text,
    )


def extract_command_literal(text: str) -> str:
    """テキスト中の最初の「コマンドに見えるバッククォート区間」を返す。

    ``strip_command_literals`` と同じ判定基準を使う。見つからなければ空文字列
    (純粋関数)。
    """
    if "`" not in text:
        return ""
    for m in _BACKTICK_SPAN_RE.finditer(text):
        span = m.group(1).strip()
        if _span_looks_like_command(span):
            return span
    return ""


# 危険コマンドパターン（run_command_safe + EventReminderSystem で共有）
DANGEROUS_PATTERNS: list[str] = [
    r"rm\s+(-[rf]+\s+)?/",  # rm -rf /
    r"rm\s+-[rf]*\s+\*",  # rm -rf *
    r"mkfs\.",  # mkfs.ext4 etc.
    r"dd\s+.*of=",  # dd of=
    r":\(\)\{\s*:\|:&\s*\};:",  # fork bomb
    r"chmod\s+-R\s+777\s+/",  # chmod -R 777 /
    r">\s*/dev/sd",  # > /dev/sda
]

# --- run_command_readonly 用の読み取り専用ガード (allow-list) -------------------
#
# chat モードの executable query (時刻 / OS / スペック等) 専用ツール
# ``run_command_readonly`` の実行前検証。
#
# **allow-list を採用する理由**: deny-list は「read-only」を保証できない。
# 2026-07-21 の敵対的レビューで、deny-list 版は builtin ``open(path,'w')`` /
# ``tee`` + パイプ / PowerShell エイリアス (ri/sc/ni) / インタプリタ
# shell-out (powershell -c / bash -c) / mkdir / touch / cp / sed -i /
# pip3 install / ``python -c`` ペイロード抽出失敗 (attached ``-c"..."`` /
# 不均衡クォート) など多数の bypass が確認された。抜け穴は無限にあるため、
# 「許可するものだけを列挙する」allow-list に転換する:
#
#   静的に副作用が無いと検証できる Python インタプリタ呼出だけを許可する。
#   すなわち ``python -c <code>`` (AST でモジュール import / 属性呼出 /
#   builtin 名 / 代入先を検査) または ``python --version`` のみ。
#   それ以外 (他インタプリタ / シェルコマンド / パイプ / リダイレクト /
#   スクリプトファイル実行 / ``-m`` モジュール実行) は全て拒否する。
#
# ``_EXECUTABLE_QUERY_COMMANDS`` (tool_call_judge) の正規テーブルは全て
# ``python -c "..."`` か ``python --version`` に収まるため、この allow-list で
# false positive は起きない (テスト test_safety_patterns で全件通過を固定)。
#
# なお ``calculate`` ツール (builtin.py) も同様に AST allow-list で安全評価
# しており、本ガードはその設計と一貫する。

# python -c ペイロードで import してよいモジュール (読み取り専用で使う範囲)
_READONLY_SAFE_MODULES: frozenset[str] = frozenset({
    "datetime", "platform", "os", "sys", "socket", "shutil",
    "time", "math", "json", "locale",
})

# 参照 / 呼び出しを禁止する名前 (書込 / 実行 / 導入 / introspection escape)
_READONLY_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input",
    "exit", "quit", "breakpoint", "help",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "memoryview",
})

# 禁止する属性名 (許可モジュール上でも書込 / 削除 / 送信 / プロセス操作は不可)
_READONLY_FORBIDDEN_ATTRS: frozenset[str] = frozenset({
    "system", "popen", "remove", "unlink", "rmdir", "removedirs",
    "rename", "renames", "replace", "makedirs", "mkdir", "rmtree",
    "move", "copy", "copy2", "copyfile", "copytree", "copystat",
    "copymode", "chmod", "chown", "lchown", "chdir", "link", "symlink",
    "truncate", "write", "writelines", "send", "sendall", "sendto",
    "connect", "bind", "listen", "accept", "spawn", "spawnl", "spawnv",
    "spawnve", "execv", "execve", "execl", "execlp", "execvp",
    "fork", "forkpty", "kill", "killpg", "abort", "startfile",
    "putenv", "unsetenv", "open",
})


def _reject_unsafe_python_payload(payload: str) -> str | None:
    """``python -c`` ペイロードを AST で検査し、副作用があれば理由を返す。

    許可: ``_READONLY_SAFE_MODULES`` の import、それらモジュールの読み取り系
    属性呼出、print / sorted 等の無害な builtin、内包表記、単純名前への代入。
    拒否: 危険 builtin (open/eval/exec/__import__ 等)、書込/削除/送信/プロセス
    系属性 (os.remove / .write / socket.connect 等)、from-import、dunder 参照
    (introspection escape)、属性/subscript への代入 (オブジェクト変更)。
    """
    try:
        tree = ast.parse(payload, mode="exec")
    except SyntaxError as e:
        return f"python payload SyntaxError: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _READONLY_SAFE_MODULES:
                    return f"import of {alias.name!r} not allowed in readonly mode"
        elif isinstance(node, ast.ImportFrom):
            return "from-import not allowed in readonly mode"
        elif isinstance(node, ast.Name):
            if node.id in _READONLY_FORBIDDEN_NAMES or node.id.startswith("__"):
                return f"use of {node.id!r} not allowed in readonly mode"
        elif isinstance(node, ast.Attribute):
            if node.attr in _READONLY_FORBIDDEN_ATTRS or node.attr.startswith("__"):
                return f"attribute {node.attr!r} not allowed in readonly mode"
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, (ast.Attribute, ast.Subscript)):
                        return "attribute/subscript assignment not allowed in readonly mode"
    return None


def reject_readonly_violation(command: str) -> str | None:
    """``run_command_readonly`` で実行してよいコマンドか検証し、違反理由を返す。

    allow-list: ``python -c <code>`` (AST が副作用なしと検証) または
    ``python --version`` / ``python -V`` のみ許可し、それ以外は全て拒否する。
    詳細は本モジュール冒頭のコメント参照。

    Returns:
        違反理由の文字列。読み取り専用と検証できれば ``None``。
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "command is not parseable as a single shell invocation"
    if not tokens:
        return "empty command"
    base = Path(tokens[0]).name.lower()
    if not (base.startswith("python") or base in {"py", "py.exe"}):
        return f"only the python interpreter is allowed in readonly mode (got {base!r})"
    rest = tokens[1:]
    if rest in (["--version"], ["-V"]):
        return None
    payload: str | None = None
    if len(rest) == 2 and rest[0] == "-c":
        payload = rest[1]
    elif len(rest) == 1 and rest[0].startswith("-c") and len(rest[0]) > 2:
        # 密着形 python -c"code" (deny-list を素通りした過去の bypass を塞ぐ)
        payload = rest[0][2:]
    if payload is None:
        return "readonly python must be `python -c <code>` or `python --version`"
    return _reject_unsafe_python_payload(payload)
