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

根の中でも、**既に在るファイルの上書き** (``write_file``) は依頼文がそのファイルを書込みの
対象に挙げたときだけ (判定点 ``overwrite_target``、:mod:`backend.free.agent.overwrite_gate`)。
題材として挙がっただけのファイルを派生物 (テスト・要約・翻訳) で上書きしない
(2026-10-05 ライブ監査 T4、docs/f_03 §4.y)。
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("agent.write_gate")

__all__ = [
    "WRITE_DENIED_RE",
    "OVERWRITE_DENIAL_CODES",
    "WRITE_PATH_TOOLS",
    "WriteDenial",
    "check_overwrite_target",
    "check_write_target",
    "has_parent_segment",
    "is_request_named",
    "normalize_write_path",
    "overwrite_exempt",
    "request_named_folders",
    "request_names_file",
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
    r"^Error: write denied \((?P<code>outside_request_roots|protected_root|invalid_path"
    r"|existing_not_target|existing_not_kept)\): "
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
    _add_unc_roots(query, roots)
    return roots


def request_named_folders(query: str) -> list[Path]:
    """依頼文が **挙げた** フォルダだけ (:func:`request_roots` から ``outputs_dir`` の既定を除く)。

    裸のファイル名の探し場所 (``file_ledger.resolve_bare_filename``) とフォルダの
    台帳記録が使う。暗黙参照は根にしない (:func:`request_roots` と同じ)。
    """
    from backend.free.agent.meta_cognitive_task_exec import delivery_roots

    roots = delivery_roots(query or "", implicit=False, fallback=False)
    _add_unc_roots(query, roots)
    return roots


def request_names_file(path: str, query: str, user_texts: tuple[str, ...] | list[str] = ()) -> bool:
    """``path`` のファイルを利用者が名指したか (今の依頼文か、このセッションの user の発話)。

    書込みの素材の門 (``no_source_data``) が、計画モデルが自分で選んだ読込みを素材に
    数えないために使う (2026-10-05 ライブ監査 T6)。名前 (basename) かフルパスが文に
    書かれているときだけ真。依頼文が挙げたフォルダ (書込みの宛先の親を含む) の中に
    あるだけでは名指しに数えない — 「…\\work\\summary.md に書き出して」で、同じ
    フォルダの buggy.py を素材に数えていた (2026-10-05 レビュー H1)。名前は区切りの
    境界で照合する (``a.py`` が ``data.py`` に当たらない)。
    """
    name = os.path.basename((path or "").rstrip("\\/"))
    if not name:
        return False
    pattern = re.compile(rf"(?<![\w.\-]){re.escape(name)}(?![\w\-])", re.IGNORECASE)
    return any(text and pattern.search(text) for text in (query, *user_texts))


def _add_unc_roots(query: str, roots: list[Path]) -> None:
    for m in _UNC_PATH_RE.finditer(query or ""):
        candidate = Path(m.group(0).rstrip("。、,.\\/"))
        root = candidate.parent if candidate.suffix else candidate
        if root not in roots:
            roots.append(root)


def _current_query() -> str:
    from backend.free.agent.tool_ledger import current_query

    return current_query()


def _named_ledger_roots(session_id: str | None) -> list[str]:
    """ファイル台帳の ``named`` の記録のフォルダ (名指しのフォルダ / 名指しのファイルの親)。"""
    from backend.free.agent.file_ledger import named_folder_paths

    return [_resolved_key(os.path.abspath(p)) for p in named_folder_paths(session_id)]


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
    check_overwrite: bool = False, content: str | None = None,
) -> tuple[str, WriteDenial | None]:
    """書込み先を判定する。``(正規化したパス, 断った理由 or None)`` を返す。

    ``query`` / ``session_id`` を省くと、このリクエストの依頼文 (``tool_ledger``
    の宛先) とファイル台帳の宛先を使う。許したときは正規化したパス (``~`` 展開・
    ``\\\\?\\`` 除去済み) で書くこと。

    ``check_overwrite`` (``write_file``) なら、根の中の **既に在るファイル** の上書きを
    :func:`check_overwrite_target` に通す。``content`` は書く本文 (弱い根拠の確かめに使う)。
    """
    from backend.config import resolve_outputs_dir

    normalized = normalize_write_path(raw)
    if normalized is None:
        return raw, WriteDenial("invalid_path", raw)
    key = _resolved_key(normalized)
    if _is_protected(key):
        return normalized, WriteDenial("protected_root", normalized)
    request = _current_query() if query is None else query
    roots = [_resolved_key(r) for r in request_roots(request)]
    roots.append(_resolved_key(resolve_outputs_dir()))
    roots += _named_ledger_roots(session_id)
    if not any(_within(key, r) for r in roots):
        return normalized, WriteDenial("outside_request_roots", normalized)
    if check_overwrite:
        return normalized, check_overwrite_target(
            normalized, request, session_id=session_id, content=content,
        )
    return normalized, None


#: 上書きの門 (:func:`check_overwrite_target`) が断ったときのコード。結果に本文を添える。
OVERWRITE_DENIAL_CODES = frozenset({"existing_not_target", "existing_not_kept"})

#: 上書きの門を掛けない書込みの区間 (制作ステージの配信、docs/f_03 §4.y)。
_overwrite_exempt: ContextVar[bool] = ContextVar("write_gate_overwrite_exempt", default=False)


@contextmanager
def overwrite_exempt() -> Iterator[None]:
    """区間内の ``write_file`` に上書きの門を掛けない (根の検査はそのまま)。

    create の制作ステージの配信 (§4.4) が使う。既存ファイルは書き手が上書きの前に
    ``bk/overwrite/`` へ退避し、``delivery`` の ``replaced`` と最終応答で利用者に見せるので、
    上書きが失われも隠れもしない。作り直し・手直し (別のセッションからを含む) は既存の成果物を
    書き換えるのが仕様 (2026-10-05 独立レビュー H1 / 3 周目)。文脈変数なので区間を抜けた後・
    並走する別のタスクには効かない。
    """
    token = _overwrite_exempt.set(True)
    try:
        yield
    finally:
        _overwrite_exempt.reset(token)


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def check_overwrite_target(
    path: str, query: str, *, session_id: str | None = None, content: str | None = None,
) -> WriteDenial | None:
    """既に在るファイルの上書きを、依頼文がそのファイルを対象に挙げたときだけ許す (docs/f_03 §4.y)。

    - 書込み先が在るファイルでない (新規) / 依頼文が無い (sleep-time・テスト) → 許す
    - このリクエストで既に書いたファイル (前段のタスクが作った) → 許す
    - 判定点 ``overwrite_target`` が ``target`` → 許す。``edit`` → 書く本文が既存の本文を
      引き継いでいれば許す (確かめられなければ許す)、引き継がなければ ``existing_not_kept``。
      それ以外 → ``existing_not_target``
    - 制作ステージの配信 (:func:`overwrite_exempt` の区間) → 許す (退避と replaced の開示がある)

    依頼文は **このターンのユーザーの発話** だけ。計画のタスク文・LLM の引数は証拠に
    数えない (不変則 #15)。
    """
    from backend.free.agent.file_ledger import (
        current_session,
        last_written_path,
        session_has_file,
    )
    from backend.free.agent.output_format import anchor_relative_output_path
    from backend.free.agent.overwrite_gate import (
        EDIT_LABEL,
        TARGET_LABEL,
        overwrite_target_verdict,
        retains_existing,
    )
    from backend.free.agent.tool_ledger import written_in_request

    if _overwrite_exempt.get() or not (query or "").strip():
        return None
    target = anchor_relative_output_path(path)
    try:
        if not os.path.isfile(target):
            return None
    except (OSError, ValueError):
        return None
    if written_in_request(target):
        return None
    session = current_session() if session_id is None else session_id
    last =last_written_path(session) if session else ""
    verdict = overwrite_target_verdict(
        query, target, in_session=session_has_file(session, target),
        last_written=bool(last) and _same_file(last, target),
    )
    if verdict.fired and verdict.value == TARGET_LABEL:
        return None
    if verdict.fired and verdict.value == EDIT_LABEL:
        if retains_existing(target, content) is not False:
            return None
        logger.warning(
            "Overwrite denied: the new content does not keep the existing file (%s): %s",
            verdict.evidence, target,
        )
        return WriteDenial("existing_not_kept", path)
    logger.warning(
        "Overwrite denied: the request does not name the existing file as the target (%s): %s",
        verdict.evidence, target,
    )
    return WriteDenial("existing_not_target", path)
