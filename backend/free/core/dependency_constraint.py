"""ユーザーが明示した依存ライブラリの制約を決定論で扱う (pillar 非依存の共有基盤)

「標準ライブラリのみ」のような **要求文に書かれた制約** は、生成側にも検証側にも
渡っていなかった。実インシデント (2026-08-07 ライブ監査): 「標準ライブラリのみを
使ってください」と明示した依頼に対し ``import pandas`` を含むコードが生成され、
仕様書自身が「これらは標準ライブラリのみを使用し」と書いた 2 行後に
``DataFrame: pandas.DataFrame`` と自己矛盾したまま「起動可能性チェック合格」で
配信された (生成環境に pandas が入っていたので import は通ってしまう)。

制約の検出も違反の検出も LLM に判定させない。前者は要求文のパターン、後者は
``sys.stdlib_module_names`` との照合で、どちらも決定論で確定できる。
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable

#: 「標準ライブラリだけで書け」に相当する要求。``標準ライブラリ`` 単独では
#: 発火させない (「標準ライブラリの使い方を教えて」を巻き込まないため)。
_STDLIB_ONLY_RE = re.compile(
    r"標準ライブラリ(?:のみ|だけ|の範囲|に限)"
    r"|(?:外部|サードパーティ|第三者)(?:の)?(?:ライブラリ|パッケージ|モジュール|依存)"
    r"[^。\n]{0,12}(?:使わ|使用しな|用いな|無し|なし|禁止|不要)"
    r"|(?:pip\s*install|外部依存)[^。\n]{0,12}(?:不要|なし|無し|しな)"
    r"|standard\s+library\s+only"
    r"|only\s+(?:the\s+)?standard\s+library"
    r"|no\s+(?:external|third[-\s]?party)\s+(?:librar|dependenc|package|module)",
    re.IGNORECASE,
)

#: 判定対象外。``__future__`` は stdlib_module_names に含まれるが明示しておく。
_ALWAYS_ALLOWED = frozenset({"__future__", "__main__"})


def requires_stdlib_only(text: str) -> bool:
    """要求文が「標準ライブラリのみ」を指示しているかを判定する (純粋関数)。"""
    return bool(_STDLIB_ONLY_RE.search(text or ""))


def find_third_party_imports(
    code: str, local_modules: Iterable[str] = (),
) -> list[str]:
    """``code`` が import している非標準ライブラリのトップレベル名を返す。

    生成物の兄弟モジュール (``local_modules``) と相対 import は自作扱いで除外する。
    構文エラーで解析できない場合は空リスト (構文は別ゲートの責務)。

    Returns:
        重複を除いた出現順のモジュール名。標準ライブラリだけなら空リスト。
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []

    allowed = set(sys.stdlib_module_names) | _ALWAYS_ALLOWED | {
        # ``pkg/mod.py`` どちらの書き方でも兄弟を許すため stem で持つ
        str(m).rsplit("/", 1)[-1].removesuffix(".py")
        for m in local_modules
    }
    found: list[str] = []
    for node in ast.walk(tree):
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # 相対 import は必ず自作
            if node.module:
                roots = [node.module.split(".")[0]]
        for root in roots:
            if root and root not in allowed and root not in found:
                found.append(root)
    return found


__all__ = ["find_third_party_imports", "requires_stdlib_only"]
