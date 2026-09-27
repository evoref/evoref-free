"""エージェントの書込みゲート — 書いてよい範囲は依頼が決める (docs/f_03 §4.y)。

パスへ書くツール (``write_file`` / ``apply_diff``) はすべて ``ToolsRegistry.execute``
の入口でここを通る。レジストリを通らない書き手 (帳票の穴埋め) は同じ関数を明示的に呼ぶ。

以前の唯一の検査は ``..`` セグメントの拒否で、絶対パス・UNC・``\\\\?\\`` 付きはすべて
通っていた。LLM が作った引数の絶対パスを「ユーザーの明示指定」と同じに扱っていたため、
読んだファイルや取得したページの本文に仕込まれた指示が ``<data_root>/g1/store`` や
``config.yaml`` を書かせられた。

書いてよい根は 3 つ (どれかの配下なら許す):

1. 依頼文 (このターンのユーザー発話) が挙げたフォルダ (:func:`request_roots`)
2. ``outputs_dir``
3. ファイル台帳の名指しの記録 (``named=True``) の親フォルダ

データ根の ``g<N>/store`` / ``g<N>/cache`` / ``run`` とインストール根の ``config.yaml*`` /
``.venv`` / ``models`` は名指しされても書かない。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("agent.write_gate")

__all__ = [
    "WRITE_DENIED_RE",
    "WRITE_PATH_TOOLS",
    "WriteDenial",
    "check_write_target",
    "has_parent_segment",
    "is_request_named",
    "normalize_write_path",
    "request_roots",
]

#: パスへ書くツールと、そのパス引数の名前。
WRITE_PATH_TOOLS: dict[str, str] = {
    "write_file": "file_path",
    "apply_diff": "file_path",
}

#: ゲートが断ったときのツール結果 (``Error: write denied (<code>): <path>``)。
#: 最終応答で i18n ``agent.write_denied.<code>`` に写す側が読む。
WRITE_DENIED_RE = re.compile(
    r"^Error: write denied \((?P<code>outside_request_roots|protected_root|invalid_path)\): "
    r"(?P<path>.*)$",
    re.MULTILINE,
)

_IS_WINDOWS = os.name == "nt"
_SEPARATORS_RE = re.compile(r"[\\/]")
_RESERVED_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})
_INVALID_CHARS = frozenset('<>"|?*')
_GENERATION_DIR_RE = re.compile(r"g\d+")
#: 依頼文中の UNC パス (``\\server\share\...``)。ドライブ付きのパスは
#: ``delivery_roots`` が拾うので、ここは UNC だけを見る。
_UNC_PATH_RE = re.compile(
    r"\\\\[^\\/\s\"'「」()（）]+\\[^\\/\s\"'「」()（）]+(?:\\[^\s\"'「」()（）]*)?",
)


@dataclass(frozen=True, slots=True)
class WriteDenial:
    """ゲートが書込みを断った理由。"""

    code: str
    path: str

    def as_tool_result(self) -> str:
        """ツール結果の文字列 (``Error: write denied (<code>): <path>``)。"""
        return f"Error: write denied ({self.code}): {self.path}"


def has_parent_segment(path: str) -> bool:
    """``..`` セグメントを含むか (``my..file.txt`` のような名前の一部は含まない)。"""
    return ".." in _SEPARATORS_RE.split(path or "")


def normalize_write_path(raw: str) -> str | None:
    """書込み先の文字列を判定前の形に揃える。書けない形なら ``None``。

    ``~`` の展開と ``\\\\?\\`` 接頭辞の除去だけを行い、相対パスは相対のまま返す
    (錨付けは呼出側の ``anchor_relative_output_path`` の仕事)。Windows では
    ルート相対 (``\\x`` / ``/tmp/x``)・ドライブ相対 (``C:x``)・予約デバイス名・
    末尾の ``.`` / 空白・ドライブ以外の ``:`` (代替データストリーム) も断る。
    """
    text = (raw or "").strip()
    if not text:
        return None
    if _IS_WINDOWS:
        head = text.replace("/", "\\")
        if head.startswith("\\\\?\\UNC\\"):
            text = "\\\\" + text[8:]
        elif head.startswith("\\\\?\\"):
            text = text[4:]
        elif head.startswith("\\\\.\\"):
            return None
    if text == "~" or text.startswith(("~/", "~\\")):
        text = os.path.expanduser(text)
    if has_parent_segment(text):
        return None
    if not _IS_WINDOWS:
        return text
    drive, rest = os.path.splitdrive(text)
    if not drive:
        if text[:1] in ("\\", "/"):
            return None
    elif drive[:2] in ("\\\\", "//"):
        if not _SEPARATORS_RE.search(drive[2:]):
            return None  # 共有名の無い UNC
    elif not rest.startswith(("\\", "/")):
        return None
    for comp in _SEPARATORS_RE.split(rest):
        if not comp or comp == ".":
            continue
        if comp[-1] in ". " or ":" in comp:
            return None
        if any(c in _INVALID_CHARS or ord(c) < 32 for c in comp):
            return None
        if comp.split(".", 1)[0].rstrip(" ").lower() in _RESERVED_NAMES:
            return None
    return text


def _resolved_key(path: str | Path) -> str:
    """包含判定の正規形。

    実在する最も近い祖先を ``realpath`` で解決し (ジャンクション / シンボリック
    リンク / 8.3 短縮名を実名へ)、残りの成分を足して ``normcase`` する。
    """
    full = os.path.abspath(path)
    head, rest = full, []
    while not os.path.lexists(head):
        parent = os.path.dirname(head)
        if parent == head:
            break
        rest.append(os.path.basename(head))
        head = parent
    try:
        real = os.path.realpath(head)
    except (OSError, ValueError):
        real = head
    return os.path.normcase(os.path.join(real, *reversed(rest)))


def _within(key: str, root_key: str) -> bool:
    if key == root_key:
        return True
    prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
    return key.startswith(prefix)


def _relative_parts(key: str, root_key: str) -> list[str] | None:
    if not _within(key, root_key):
        return None
    rest = key[len(root_key):].strip(os.sep)
    return rest.split(os.sep) if rest else []


def _data_root() -> Path:
    """稼働中のデータ根 (``PathResolver`` の ``run/`` の親)。"""
    from backend.config import resolve_data_path

    return resolve_data_path("run_dir").parent


def _install_root() -> Path:
    from backend.config import get_project_root

    return get_project_root()


def _is_protected(key: str) -> bool:
    """名指しされても書かない場所か (データ根の ``g<N>/store|cache`` と ``run``、
    インストール根の ``config.yaml*`` / ``.venv`` / ``models``)。"""
    from backend.data_root import GENERATION_SCOPED_DIRS

    parts = _relative_parts(key, _resolved_key(_data_root()))
    if parts:
        if parts[0] == os.path.normcase("run"):
            return True
        if (
            len(parts) >= 2
            and _GENERATION_DIR_RE.fullmatch(parts[0])
            and parts[1] in {os.path.normcase(d) for d in GENERATION_SCOPED_DIRS}
        ):
            return True
    parts = _relative_parts(key, _resolved_key(_install_root()))
    if parts:
        if parts[0] in {os.path.normcase(".venv"), os.path.normcase("models")}:
            return True
        if len(parts) == 1 and parts[0].startswith(os.path.normcase("config.yaml")):
            return True
    return False


def request_roots(query: str) -> list[Path]:
    """依頼文が挙げたフォルダ (と、無ければ ``outputs_dir``)。

    ``delivery_roots`` と同じ部品だが **暗黙参照 (「その中身」→ 直近のファイル)
    は根にしない** — 直近のファイルが LLM の読んだパスなら、それを根にすると
    LLM の引数を証拠に数えることになる。UNC は依頼文に書かれているときだけ根になる。
    """
    from backend.free.agent.meta_cognitive_task_exec import delivery_roots

    roots = delivery_roots(query or "", implicit=False)
    for m in _UNC_PATH_RE.finditer(query or ""):
        candidate = Path(m.group(0).rstrip("。、,.\\/"))
        root = candidate.parent if candidate.suffix else candidate
        if root not in roots:
            roots.append(root)
    return roots


def _current_query() -> str:
    from backend.free.agent.tool_ledger import current_query

    return current_query()


def _named_ledger_roots(session_id: str | None) -> list[str]:
    from backend.free.agent.file_ledger import current_named_file_paths, named_file_paths

    paths = named_file_paths(session_id) if session_id is not None else current_named_file_paths()
    return [_resolved_key(os.path.dirname(os.path.abspath(p))) for p in paths]


def is_request_named(path: str, *, query: str | None = None, session_id: str | None = None) -> bool:
    """``path`` が依頼文由来の根 (依頼文のフォルダ / 名指しの台帳記録の親) の配下か。

    ファイル台帳の ``named`` を記録時に決めるために使う。
    """
    normalized = normalize_write_path(path)
    if normalized is None:
        return False
    key = _resolved_key(normalized)
    roots = [_resolved_key(r) for r in request_roots(_current_query() if query is None else query)]
    roots += _named_ledger_roots(session_id)
    return any(_within(key, r) for r in roots)


def check_write_target(
    raw: str, *, query: str | None = None, session_id: str | None = None,
) -> tuple[str, WriteDenial | None]:
    """書込み先を判定する。``(正規化したパス, 断った理由 or None)`` を返す。

    ``query`` / ``session_id`` を省くと、このリクエストの依頼文 (``tool_ledger``
    の宛先) とファイル台帳の宛先を使う。許したときは正規化したパス (``~`` 展開・
    ``\\\\?\\`` 除去済み) で書くこと。
    """
    from backend.config import resolve_outputs_dir

    normalized = normalize_write_path(raw)
    if normalized is None:
        return raw, WriteDenial("invalid_path", raw)
    key = _resolved_key(normalized)
    if _is_protected(key):
        return normalized, WriteDenial("protected_root", normalized)
    roots = [_resolved_key(r) for r in request_roots(_current_query() if query is None else query)]
    roots.append(_resolved_key(resolve_outputs_dir()))
    roots += _named_ledger_roots(session_id)
    if any(_within(key, r) for r in roots):
        return normalized, None
    return normalized, WriteDenial("outside_request_roots", normalized)
