"""テーマ管理ビジネスロジック"""

from __future__ import annotations

import copy
import json
import re
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from backend.config import get_config, get_path_resolver
from backend.free.themes.theme_installer import ThemeInstallResult, is_valid_theme_id
from backend.log_config import get_logger

logger = get_logger("themes.service")


# ── 戻り値型定義 ──


class ThemeActivateResult(TypedDict):
    """activate() の戻り値型"""

    theme_id: str
    name: str
    color_mode: Literal["light", "dark"]
    colors: str
    gui_layout: dict | None
    cli_theme: dict | None


class ThemeInfoResult(TypedDict):
    """_read_theme_info() / list_themes() 要素の戻り値型"""

    theme_id: str
    name: str
    version: str
    author: str
    description: str
    active: bool
    has_preview: bool
    has_cli_theme: bool


# プレビュー画像のファイル名許可パターン
_SAFE_PREVIEW_NAME = re.compile(r"^[a-zA-Z0-9_\-]+\.(png|jpe?g|gif|svg|webp)$")

# プレビュー画像の MIME タイプマッピング
_PREVIEW_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}


def _is_link_or_reparse_point(path: Path) -> bool:
    """シンボリックリンク / ジャンクション等の再解析点か (存在しなければ False)"""
    try:
        st = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _declared_file(theme_dir: Path, name: object) -> Path | None:
    """theme.json が指すファイルを theme_dir 配下に限って解決する。外を指せば None"""
    if not isinstance(name, str) or not name:
        return None
    path = theme_dir / name
    root = theme_dir.resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        return None
    return path


def _mirror_theme_to_live_config(changed: dict) -> None:
    """config.yaml に書いた ``theme`` のキーを稼働中の設定へ写す (未ロードなら何もしない)。"""
    if not changed:
        return
    try:
        live = get_config().get("theme")
    except RuntimeError:
        return
    if isinstance(live, dict):
        live.update(copy.deepcopy(changed))


class ThemeManager:
    """テーマパッケージの管理"""

    def __init__(self, themes_dir: str | Path, cfg: dict):
        self.themes_dir = Path(themes_dir)
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = cfg
        self.active_theme_id: str = cfg.get("theme", {}).get("active", "")
        self.color_mode: str = cfg.get("theme", {}).get("color_mode", "dark")

    def list_themes(self) -> list[ThemeInfoResult]:
        """全テーマのメタデータ一覧を返す"""
        themes: list[ThemeInfoResult] = []

        for entry in self.themes_dir.iterdir():
            theme_dir = self._resolve_theme_dir(entry.name)
            if theme_dir is None:
                continue
            info = self._read_theme_info(theme_dir)
            if info:
                themes.append(info)

        return themes

    async def install_from_url(self, url: str) -> ThemeInstallResult:
        """URL からテーマ ZIP をダウンロードしてインストール"""
        from backend.free.themes.theme_installer import install_from_url as _install_url

        return await _install_url(url, self.themes_dir)

    def install(self, zip_path: Path) -> ThemeInstallResult:
        """ZIP パッケージからテーマをインストール"""
        from backend.free.themes.theme_installer import install_theme

        return install_theme(zip_path, self.themes_dir)

    def activate(self, theme_id: str, color_mode: str | None = None) -> ThemeActivateResult:
        """テーマをアクティベート（設計書 §9.2 準拠）"""
        theme_dir = self._resolve_theme_dir(theme_id)
        if theme_dir is None:
            raise KeyError(f"Theme '{theme_id}' not found")

        meta = self._read_theme_meta(theme_dir)
        if meta is None:
            raise KeyError(f"Theme '{theme_id}' has no theme.json")

        # カラーモード決定
        if color_mode is None:
            color_mode = self.color_mode
        if color_mode not in ("light", "dark"):
            color_mode = "dark"

        colors_css = self._load_theme_css(theme_dir, meta, color_mode)
        layout_data = self._load_gui_layout(theme_dir, meta)
        cli_theme_data = self._load_cli_theme(theme_dir, meta)

        # 永続化に成功してから状態を更新する (readonly 等で書けなければメモリも変えない)
        self._persist_active_theme(theme_id, color_mode)
        self.active_theme_id = theme_id
        self.color_mode = color_mode

        logger.info("Theme activated: id=%s, color_mode=%s", theme_id, color_mode)
        return ThemeActivateResult(
            theme_id=theme_id,
            name=meta.get("name", theme_id),
            color_mode=color_mode,  # type: ignore[arg-type]  # validated above
            colors=colors_css,
            gui_layout=layout_data,
            cli_theme=cli_theme_data,
        )

    def uninstall(self, theme_id: str) -> None:
        """テーマをアンインストール (ID が themes_dir 直下を指さなければ ValueError)"""
        target_dir = self._theme_path(theme_id)
        if not target_dir.exists():
            raise KeyError(f"Theme '{theme_id}' not found")

        # アクティブテーマを削除する場合は残存テーマへの切替を先に永続化する。消してから
        # 書くと、書けなかったときに config が消えたテーマを指したまま残る
        if self.active_theme_id == theme_id:
            remaining = sorted(
                d.name for d in self.themes_dir.iterdir()
                if d.name != theme_id and self._resolve_theme_dir(d.name) is not None
            )
            next_active = remaining[0] if remaining else ""
            self._persist_active_theme(next_active, self.color_mode)
            self.active_theme_id = next_active

        shutil.rmtree(str(target_dir))

        logger.info("Theme uninstalled: id=%s", theme_id)

    def get_preview_path(self, theme_id: str) -> Path | None:
        """テーマプレビュー画像のパスを返す"""
        return self._get_preview_path_by_key(theme_id, "preview")

    # ── private: activate サブ関数 ──

    def _load_theme_css(
        self, theme_dir: Path, meta: dict, color_mode: str,
    ) -> str:
        """テーマ CSS を読み込んで文字列で返す"""
        colors_cfg = meta.get("colors")
        if not isinstance(colors_cfg, dict):
            colors_cfg = {}
        css_filename = colors_cfg.get(color_mode, f"colors-{color_mode}.css")
        css_path = _declared_file(theme_dir, css_filename)
        if css_path is None:
            raise ValueError(f"CSS file outside theme directory: {css_filename}")
        if not css_path.exists():
            raise ValueError(f"CSS file not found: {css_filename}")
        return css_path.read_text(encoding="utf-8")

    def _load_gui_layout(self, theme_dir: Path, meta: dict) -> dict | None:
        """gui-layout.json を読み込む。存在しなければ None"""
        layout_filename = meta.get("gui_layout", "gui-layout.json")
        layout_path = _declared_file(theme_dir, layout_filename)
        if layout_path is None or not layout_path.exists():
            return None
        return json.loads(layout_path.read_text(encoding="utf-8"))

    def _load_cli_theme(self, theme_dir: Path, meta: dict) -> dict | None:
        """cli-theme.json を読み込む。存在しない・パース失敗時は None"""
        cli_theme_filename = meta.get("cli_theme", "cli-theme.json")
        cli_theme_path = _declared_file(theme_dir, cli_theme_filename)
        if cli_theme_path is None or not cli_theme_path.exists():
            return None
        try:
            return json.loads(cli_theme_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ── private: メタ・パス解決 ──

    def _read_theme_meta(self, theme_dir: Path) -> dict | None:
        """theme.json を読み取ってメタ情報辞書を返す"""
        theme_json_path = theme_dir / "theme.json"
        if not theme_json_path.exists():
            return None
        try:
            meta = json.loads(theme_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return meta if isinstance(meta, dict) else None

    def _get_preview_path_by_key(self, theme_id: str, meta_key: str) -> Path | None:
        """指定されたメタキーでプレビュー画像パスを取得する"""
        theme_dir = self._resolve_theme_dir(theme_id)
        if theme_dir is None:
            return None

        meta = self._read_theme_meta(theme_dir)
        if meta is None:
            return None

        preview_name = meta.get(meta_key)
        if not preview_name or not _SAFE_PREVIEW_NAME.match(preview_name):
            return None

        preview_path = theme_dir / preview_name
        # パストラバーサル防止
        try:
            preview_path.resolve().relative_to(theme_dir.resolve())
        except ValueError:
            return None

        if preview_path.is_file():
            return preview_path
        return None

    def _read_theme_info(self, theme_dir: Path) -> ThemeInfoResult | None:
        """ディレクトリから theme.json を読み取ってメタ情報を返す"""
        meta = self._read_theme_meta(theme_dir)
        if meta is None:
            return None

        theme_id = theme_dir.name

        # プレビュー画像の有無
        has_preview = self.get_preview_path(theme_id) is not None

        # CLI テーマ情報
        cli_theme_filename = meta.get("cli_theme", "cli-theme.json")
        cli_theme_path = _declared_file(theme_dir, cli_theme_filename)
        has_cli_theme = cli_theme_path is not None and cli_theme_path.exists()

        return ThemeInfoResult(
            theme_id=theme_id,
            name=meta.get("name", theme_id),
            version=meta.get("version", "1.0.0"),
            author=meta.get("author", ""),
            description=meta.get("description", ""),
            active=self.active_theme_id == theme_id,
            has_preview=has_preview,
            has_cli_theme=has_cli_theme,
        )

    def theme_exists(self, theme_id: str) -> bool:
        """インストール済みのテーマか"""
        return self._resolve_theme_dir(theme_id) is not None

    def _resolve_theme_dir(self, theme_id: str) -> Path | None:
        """テーマ ID からディレクトリを解決"""
        if not theme_id:
            return None
        try:
            theme_dir = self._theme_path(theme_id)
        except ValueError:
            return None
        if theme_dir.exists() and theme_dir.is_dir():
            return theme_dir
        return None

    def _theme_path(self, theme_id: str) -> Path:
        """テーマ ID を themes_dir 直下のパスへ写す (存在は問わない)。

        ID が不正、リンク / 再解析点、または解決先が themes_dir 直下の同名でなければ
        ValueError。テーマ ID をパスにするのは必ずここを通す。
        """
        if not is_valid_theme_id(theme_id):
            raise ValueError(f"Invalid theme id: {theme_id!r}")
        theme_dir = self.themes_dir / theme_id
        if _is_link_or_reparse_point(theme_dir):
            raise ValueError(f"Theme directory is a link: {theme_id!r}")
        root = self.themes_dir.resolve()
        resolved = theme_dir.resolve()
        if resolved.parent != root or resolved.name != theme_id:
            raise ValueError(f"Theme id does not name a directory under themes_dir: {theme_id!r}")
        return theme_dir

    def _update_config(self, updater: Callable[[dict], None]) -> None:
        """config.yaml の ``theme`` 節を読み、updater で書き換えた値を ``save_config_section`` で保存する。

        updater はディスク上の ``theme`` 節 (dict) を受け取る。書込みは設定 API と同じ経路
        (版の検査・検証・コメントを残す差分書込み) を通る。readonly 起動では書かない
        (``DataReadonlyError``、API では 423)。``config_version`` がアプリより新しければ
        ``NewerConfigVersionError`` (API では 409)。書けたら変えた ``theme`` のキーを稼働中の
        設定 (``get_config()``) にも写す。

        Raises:
            RuntimeError: config.yaml の読み書き・検証に失敗した場合
        """
        import yaml
        from pydantic import ValidationError

        from backend.config import save_config_section
        from backend.config_normalize import NewerConfigVersionError
        from backend.io.readonly import guard_write

        try:
            resolver = get_path_resolver()
        except RuntimeError:
            # Config 未ロード（テスト環境など）— 永続化をスキップ
            logger.debug("PathResolver not available, skipping config persist")
            return

        config_path = resolver.root / "config.yaml"
        if not config_path.exists():
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to read config.yaml: %s", e)
            raise RuntimeError(f"Failed to read config.yaml: {e}") from e

        theme = data.get("theme")
        before = copy.deepcopy(theme) if isinstance(theme, dict) else {}
        after = copy.deepcopy(before)
        updater(after)
        changed = {k: v for k, v in after.items() if k not in before or before[k] != v}
        if not changed:
            return
        guard_write(config_path, outside_store=True)

        try:
            save_config_section("theme", changed, project_root=resolver.root, reload=False)
        except NewerConfigVersionError:
            raise
        except (ValidationError, OSError, yaml.YAMLError) as e:
            logger.error("Failed to write config.yaml: %s", e)
            raise RuntimeError(f"Failed to persist theme settings: {e}") from e

        _mirror_theme_to_live_config(changed)

    def _persist_active_theme(self, theme_id: str, color_mode: str) -> None:
        """config.yaml にアクティブテーマを永続化"""
        def updater(theme: dict) -> None:
            theme["active"] = theme_id
            theme["color_mode"] = color_mode

        self._update_config(updater)
        logger.debug("Persisted active theme to config.yaml: %s/%s", theme_id, color_mode)
