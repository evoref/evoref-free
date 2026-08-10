"""プロジェクト ID 解決ロジック

EvorefMem 統合仕様 における **プロジェクトスコープ** の同定を担う
クリエイトモードでの SemMem は project スコープに物理分離されるため
(`local/memory/semantic/projects/<project_id>/`)、現在の作業ディレクトリから
安定した project_id を導出する必要がある。

決定方針:
1. 既存 alias (state.json の `project_aliases`) に該当があれば最優先で採用
2. それ以外は git remote URL を正規化してハッシュ → ``git_<sha1[:12]>``
3. git remote が無ければ作業ディレクトリの絶対パスを正規化してハッシュ →
   ``path_<sha1[:12]>``

設計原則 (CLAUDE.md / .claude/rules/backend.md):
- 純粋関数中心。git 取得のみ副作用 (subprocess) を持ち、テストでは
  ``git_runner`` をモック差し替え可能にする
- 後方互換不要
- どの経路で導出されたかを ``ProjectIdResolution.source`` でトレース可能にする
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from backend.log_config import get_logger

logger = get_logger("memory.project_resolver")


GIT_PREFIX = "git_"
PATH_PREFIX = "path_"
HASH_LEN = 12

# git URL から credentials (`user:pass@`) を除去するための正規表現
_CREDENTIALS_RE = re.compile(r"://[^/@]+@")
# 末尾の `.git` を除去
_DOT_GIT_RE = re.compile(r"\.git/?$", re.IGNORECASE)

ResolutionSource = Literal["alias", "git_remote", "path"]


@dataclass(frozen=True)
class ProjectIdResolution:
    """`resolve_project_id` の結果。

    Attributes:
        project_id: 確定したプロジェクト ID
        source: 導出経路 (``alias`` / ``git_remote`` / ``path``)
        normalized_remote: git remote URL を正規化した文字列 (取得できた場合)
        normalized_path: 作業ディレクトリの絶対パス文字列 (常に設定)
        alias_key: alias マッチした場合の alias 名
    """

    project_id: str
    source: ResolutionSource
    normalized_path: str
    normalized_remote: str | None = None
    alias_key: str | None = None


# ──────────────────────────────────────────────────────────────────────────
# 正規化ヘルパ (純粋関数)
# ──────────────────────────────────────────────────────────────────────────


def normalize_remote_url(url: str) -> str:
    """git remote URL を正規化する。

    - 前後の空白を除去
    - 認証情報 (``user@`` や ``user:pass@``) を除去 (SCP 風 ``git@host:`` の
      ``git@`` 部分も同様に除去される)
    - 末尾の ``.git`` (任意で末尾スラッシュ) を除去
    - 全体を小文字化 (大小文字差異を吸収)
    - SCP 風 (``git@github.com:owner/repo``) を URL 風に揃える

    結果は **同じリポジトリは同じ文字列** になることだけを保証し、
    HTTP/HTTPS/SSH の差は吸収しない (URL に意図的な分離があるユーザが
    意図せず alias マージされるのを防ぐため)。
    """
    if not url:
        return ""
    s = url.strip()
    # SCP 風 ``git@host:owner/repo`` を ``ssh://git@host/owner/repo`` に揃える
    if "://" not in s and ":" in s and not s.startswith("/"):
        host, _, path = s.partition(":")
        s = f"ssh://{host}/{path}"
    s = _CREDENTIALS_RE.sub("://", s)
    s = _DOT_GIT_RE.sub("", s)
    s = s.rstrip("/")
    return s.lower()


def normalize_local_path(path: Path | str) -> str:
    """作業ディレクトリパスを正規化する。

    絶対パス化 + ``Path.resolve()`` でシンボリックリンク展開 + POSIX 区切り化。
    Windows のドライブレター差 (大文字小文字) を吸収するため小文字化する。
    """
    p = Path(path)
    try:
        resolved = p.resolve(strict=False)
    except OSError:
        resolved = p.absolute()
    return resolved.as_posix().lower()


def _sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:HASH_LEN]


def compute_project_id_from_remote(remote_url: str) -> str:
    """正規化済 git remote URL から project_id を作る。

    Raises:
        ValueError: ``remote_url`` が空文字
    """
    if not remote_url:
        raise ValueError("remote_url must be non-empty")
    return f"{GIT_PREFIX}{_sha1_short(remote_url)}"


def compute_project_id_from_path(path: Path | str) -> str:
    """作業ディレクトリパスから project_id を作る (フォールバック経路)。"""
    return f"{PATH_PREFIX}{_sha1_short(normalize_local_path(path))}"


# ──────────────────────────────────────────────────────────────────────────
# git remote 取得 (副作用あり、テストでは差し替え可能)
# ──────────────────────────────────────────────────────────────────────────


GitRunner = Callable[[Path], str | None]
"""git remote URL を取得する関数の型。失敗時は None を返す。"""


def _default_git_runner(cwd: Path) -> str | None:
    """`git -C <cwd> config --get remote.origin.url` を実行する既定実装。

    git が無い、リポジトリでない、リモート未設定の場合はすべて None。
    タイムアウトは 3 秒で打ち切る (起動遅延を避けるため)。
    """
    if not cwd.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git remote lookup failed in %s: %s", cwd, exc)
        return None
    if result.returncode != 0:
        return None
    url = (result.stdout or "").strip()
    return url or None


# ──────────────────────────────────────────────────────────────────────────
# 解決本体
# ──────────────────────────────────────────────────────────────────────────


def _match_alias(
    cwd_normalized: str,
    remote_normalized: str | None,
    aliases: dict[str, str] | None,
) -> tuple[str, str] | None:
    """alias 辞書から (alias_key, project_id) を返す。

    alias 辞書は ``{alias_name: project_id}`` で、alias_name には
    正規化済 remote URL またはローカルパスを直接入れることもできる
    (ユーザが任意の人間可読な名前を与えた場合は何にもマッチしないので
    ``current_project_id`` で明示的に上書きする運用)。
    """
    if not aliases:
        return None
    candidates: list[str] = []
    if remote_normalized:
        candidates.append(remote_normalized)
    candidates.append(cwd_normalized)
    for key in candidates:
        pid = aliases.get(key)
        if pid:
            return key, pid
    return None


def resolve_project_id(
    cwd: Path | str,
    *,
    aliases: dict[str, str] | None = None,
    git_runner: GitRunner | None = None,
) -> ProjectIdResolution:
    """作業ディレクトリから project_id を解決する。

    Args:
        cwd: プロジェクトの作業ディレクトリ
        aliases: ``{normalized_remote_or_path: project_id}`` の辞書
            (state.json の ``project_aliases`` を想定)
        git_runner: git remote 取得関数。テスト用に差し替え可能。
            未指定なら `_default_git_runner` を使う。

    Returns:
        ProjectIdResolution: 確定した project_id と導出経路情報
    """
    cwd_path = Path(cwd)
    cwd_normalized = normalize_local_path(cwd_path)

    runner = git_runner or _default_git_runner
    raw_remote = runner(cwd_path)
    remote_normalized = normalize_remote_url(raw_remote) if raw_remote else None
    if remote_normalized == "":
        remote_normalized = None

    alias_hit = _match_alias(cwd_normalized, remote_normalized, aliases)
    if alias_hit is not None:
        alias_key, pid = alias_hit
        logger.debug(
            "project resolved via alias: key=%s -> id=%s", alias_key, pid,
        )
        return ProjectIdResolution(
            project_id=pid,
            source="alias",
            normalized_path=cwd_normalized,
            normalized_remote=remote_normalized,
            alias_key=alias_key,
        )

    if remote_normalized:
        pid = compute_project_id_from_remote(remote_normalized)
        logger.debug(
            "project resolved via git remote: %s -> id=%s",
            remote_normalized, pid,
        )
        return ProjectIdResolution(
            project_id=pid,
            source="git_remote",
            normalized_path=cwd_normalized,
            normalized_remote=remote_normalized,
        )

    pid = compute_project_id_from_path(cwd_path)
    logger.debug("project resolved via path fallback: %s -> id=%s", cwd_normalized, pid)
    return ProjectIdResolution(
        project_id=pid,
        source="path",
        normalized_path=cwd_normalized,
        normalized_remote=None,
    )
