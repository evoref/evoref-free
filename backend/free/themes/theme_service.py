"""テーマ管理ビジネスロジック"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from backend.config import get_path_resolver
from backend.free.themes.theme_installer import ThemeInstallResult
from backend.io.atomic import atomic_write_text
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
    slots: dict[str, str | None] | None
    cli_modules: dict[str, str] | None
    trusted: bool
    features: dict[str, bool] | None


class ThemeInfoResult(TypedDict):
    """_read_theme_info() / list_themes() 要素の戻り値型"""

    theme_id: str
    name: str
    version: str
    author: str
    description: str
    active: bool
    trusted: bool
    has_components: bool
    component_count: int
    builtin: bool
    has_preview: bool
    has_cli_theme: bool
    has_cli_modules: bool
    cli_module_count: int
    features: dict[str, bool] | None


class ActiveCliThemeResult(TypedDict):
    """get_active_cli_theme() の戻り値型"""

    theme_id: str
    cli_theme: dict

# コンポーネントファイル名の許可パターン（パストラバーサル防止）
_SAFE_COMPONENT_NAME = re.compile(r"^[a-zA-Z0-9_\-]+\.(js|mjs)$")

# CLI モジュールファイル名の許可パターン
_SAFE_CLI_MODULE_NAME = re.compile(r"^[a-zA-Z0-9_\-]+\.py$")

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


class ThemeManager:
    """テーマパッケージの管理"""

    def __init__(self, themes_dir: str | Path, cfg: dict):
        self.themes_dir = Path(themes_dir)
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = cfg
        self.active_theme_id: str = cfg.get("theme", {}).get("active", "")
        self.color_mode: str = cfg.get("theme", {}).get("color_mode", "dark")
        self._trusted_ids: set[str] = set(cfg.get("theme", {}).get("trusted", []))

    def list_themes(self) -> list[ThemeInfoResult]:
        """全テーマのメタデータ一覧を返す"""
        themes: list[ThemeInfoResult] = []

        for entry in self.themes_dir.iterdir():
            if not entry.is_dir():
                continue
            info = self._read_theme_info(entry, builtin=False)
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
        trusted = self.is_trusted(theme_id)
        slots = self._build_slots(theme_dir, layout_data, trusted)
        cli_modules = self._build_cli_modules(theme_id, trusted)

        # 状態更新 + 永続化
        self.active_theme_id = theme_id
        self.color_mode = color_mode
        self._persist_active_theme(theme_id, color_mode)

        logger.info("Theme activated: id=%s, color_mode=%s", theme_id, color_mode)
        return ThemeActivateResult(
            theme_id=theme_id,
            name=meta.get("name", theme_id),
            color_mode=color_mode,  # type: ignore[arg-type]  # validated above
            colors=colors_css,
            gui_layout=layout_data,
            cli_theme=cli_theme_data,
            slots=slots,
            cli_modules=cli_modules,
            trusted=trusted,
            features=meta.get("features"),
        )

    def uninstall(self, theme_id: str) -> None:
        """テーマをアンインストール"""
        target_dir = self.themes_dir / theme_id
        if not target_dir.exists():
            raise KeyError(f"Theme '{theme_id}' not found")

        shutil.rmtree(str(target_dir))

        # 信頼リストからも削除
        self._trusted_ids.discard(theme_id)
        self._persist_trusted_list()

        # アクティブテーマを削除した場合は残存テーマに自動切替
        if self.active_theme_id == theme_id:
            remaining = sorted(d.name for d in self.themes_dir.iterdir() if d.is_dir())
            if remaining:
                self.active_theme_id = remaining[0]
                self._persist_active_theme(remaining[0], self.color_mode)
            else:
                self.active_theme_id = ""
                self._persist_active_theme("", self.color_mode)

        logger.info("Theme uninstalled: id=%s", theme_id)

    def is_trusted(self, theme_id: str) -> bool:
        """テーマが信頼済みかどうかを判定。

        すべてのテーマは外部扱いとし、`config.yaml` の `theme.trusted[]` に
        明示登録された ID のみ trusted として slots / cli-modules を有効化する。
        """
        return theme_id in self._trusted_ids

    def trust_theme(self, theme_id: str) -> None:
        """テーマを信頼済みとしてマーク（config.yaml に永続化）"""
        theme_dir = self._resolve_theme_dir(theme_id)
        if theme_dir is None:
            raise KeyError(f"Theme '{theme_id}' not found")
        self._trusted_ids.add(theme_id)
        self._persist_trusted_list()
        logger.info("Theme trusted: id=%s", theme_id)

    def untrust_theme(self, theme_id: str) -> None:
        """テーマの信頼を取り消し"""
        self._trusted_ids.discard(theme_id)
        self._persist_trusted_list()
        logger.info("Theme untrusted: id=%s", theme_id)

    def get_component_path(self, theme_id: str, filename: str) -> Path | None:
        """テーマコンポーネントファイルのパスを返す"""
        return self._get_safe_file_path(theme_id, "components", filename, _SAFE_COMPONENT_NAME)

    def get_preview_path(self, theme_id: str) -> Path | None:
        """テーマプレビュー画像のパスを返す"""
        return self._get_preview_path_by_key(theme_id, "preview")

    def get_active_cli_theme(self) -> ActiveCliThemeResult:
        """アクティブテーマの CLI テーマ設定を返す。

        theme.json の cli_theme フィールドで指定された JSON ファイルを読み込む。
        ファイルが存在しない場合は空の辞書を返す（呼び出し側でデフォルト値を使用）。

        Returns:
            {"theme_id": str, "cli_theme": dict}
        """
        theme_dir = self._resolve_theme_dir(self.active_theme_id)
        if theme_dir is None:
            return ActiveCliThemeResult(theme_id=self.active_theme_id, cli_theme={})

        # theme.json から cli-theme.json のファイル名を取得
        meta = self._read_theme_meta(theme_dir)
        cli_theme_filename = "cli-theme.json"
        if meta is not None:
            cli_theme_filename = meta.get("cli_theme", "cli-theme.json")

        cli_theme_path = theme_dir / cli_theme_filename
        if not cli_theme_path.exists():
            logger.debug(
                "CLI theme file not found for theme '%s': %s",
                self.active_theme_id,
                cli_theme_path,
            )
            return ActiveCliThemeResult(theme_id=self.active_theme_id, cli_theme={})

        try:
            cli_theme_data = json.loads(
                cli_theme_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read CLI theme file: %s", e)
            return ActiveCliThemeResult(theme_id=self.active_theme_id, cli_theme={})

        logger.debug(
            "Loaded CLI theme for '%s' from %s",
            self.active_theme_id,
            cli_theme_path,
        )
        return ActiveCliThemeResult(theme_id=self.active_theme_id, cli_theme=cli_theme_data)

    def list_components(self, theme_id: str) -> list[str]:
        """テーマのコンポーネントファイル一覧を返す"""
        theme_dir = self._resolve_theme_dir(theme_id)
        if theme_dir is None:
            return []
        comp_dir = theme_dir / "components"
        if not comp_dir.is_dir():
            return []
        return [
            f.name for f in comp_dir.iterdir()
            if f.is_file() and _SAFE_COMPONENT_NAME.match(f.name)
        ]

    def get_cli_module_path(self, theme_id: str, filename: str) -> Path | None:
        """CLI モジュールファイルのパスを返す"""
        return self._get_safe_file_path(theme_id, "cli-modules", filename, _SAFE_CLI_MODULE_NAME)

    def get_cli_preview_path(self, theme_id: str) -> Path | None:
        """CLI プレビュー画像のパスを返す"""
        return self._get_preview_path_by_key(theme_id, "preview_cli")

    def list_cli_modules(self, theme_id: str) -> list[str]:
        """CLI モジュールファイル一覧を返す"""
        theme_dir = self._resolve_theme_dir(theme_id)
        if theme_dir is None:
            return []
        mod_dir = theme_dir / "cli-modules"
        if not mod_dir.is_dir():
            return []
        return [
            f.name for f in mod_dir.iterdir()
            if f.is_file() and _SAFE_CLI_MODULE_NAME.match(f.name)
        ]

    # ── private: activate サブ関数 ──

    def _load_theme_css(
        self, theme_dir: Path, meta: dict, color_mode: str,
    ) -> str:
        """テーマ CSS を読み込んで文字列で返す"""
        colors_cfg = meta.get("colors", {})
        css_filename = colors_cfg.get(color_mode, f"colors-{color_mode}.css")
        css_path = theme_dir / css_filename
        if not css_path.exists():
            raise ValueError(f"CSS file not found: {css_filename}")
        return css_path.read_text(encoding="utf-8")

    def _load_gui_layout(self, theme_dir: Path, meta: dict) -> dict | None:
        """gui-layout.json を読み込む。存在しなければ None"""
        layout_filename = meta.get("gui_layout", "gui-layout.json")
        layout_path = theme_dir / layout_filename
        if not layout_path.exists():
            return None
        return json.loads(layout_path.read_text(encoding="utf-8"))

    def _load_cli_theme(self, theme_dir: Path, meta: dict) -> dict | None:
        """cli-theme.json を読み込む。存在しない・パース失敗時は None"""
        cli_theme_filename = meta.get("cli_theme", "cli-theme.json")
        cli_theme_path = theme_dir / cli_theme_filename
        if not cli_theme_path.exists():
            return None
        try:
            return json.loads(cli_theme_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _build_slots(
        self, theme_dir: Path, layout_data: dict | None, trusted: bool,
    ) -> dict | None:
        """スロット構築（trusted かつ components/ が存在する場合のみ）"""
        if not trusted or not (theme_dir / "components").is_dir():
            return None
        if layout_data and "slots" in layout_data:
            return layout_data["slots"]
        return None

    def _build_cli_modules(self, theme_id: str, trusted: bool) -> dict | None:
        """CLI モジュール一覧を構築（trusted の場合のみ）"""
        if not trusted:
            return None
        cli_modules_list = self.list_cli_modules(theme_id)
        if not cli_modules_list:
            return None
        return {m: m for m in cli_modules_list}

    # ── private: メタ・パス解決 ──

    def _read_theme_meta(self, theme_dir: Path) -> dict | None:
        """theme.json を読み取ってメタ情報辞書を返す"""
        theme_json_path = theme_dir / "theme.json"
        if not theme_json_path.exists():
            return None
        try:
            return json.loads(theme_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

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

    def _get_safe_file_path(
        self, theme_id: str, subdir: str, filename: str, pattern: re.Pattern[str],
    ) -> Path | None:
        """テーマ内のファイルパスを安全に解決する"""
        if not pattern.match(filename):
            return None
        theme_dir = self._resolve_theme_dir(theme_id)
        if theme_dir is None:
            return None
        file_path = theme_dir / subdir / filename
        # パストラバーサル防止
        try:
            file_path.resolve().relative_to((theme_dir / subdir).resolve())
        except ValueError:
            return None
        if file_path.is_file():
            return file_path
        return None

    def _read_theme_info(self, theme_dir: Path, *, builtin: bool) -> ThemeInfoResult | None:
        """ディレクトリから theme.json を読み取ってメタ情報を返す"""
        meta = self._read_theme_meta(theme_dir)
        if meta is None:
            return None

        theme_id = theme_dir.name
        has_components = (theme_dir / "components").is_dir()
        # 信頼判定: config.yaml の theme.trusted[] で明示登録された ID のみ trusted
        trusted = self.is_trusted(theme_id)
        # コンポーネント数
        component_count = len(self.list_components(theme_id)) if has_components else 0

        # プレビュー画像の有無
        has_preview = self.get_preview_path(theme_id) is not None

        # CLI テーマ情報
        cli_theme_filename = meta.get("cli_theme", "cli-theme.json")
        has_cli_theme = (theme_dir / cli_theme_filename).exists()
        has_cli_modules = (theme_dir / "cli-modules").is_dir()
        cli_module_count = len(self.list_cli_modules(theme_id)) if has_cli_modules else 0

        # features フィールド
        features = meta.get("features")

        return ThemeInfoResult(
            theme_id=theme_id,
            name=meta.get("name", theme_id),
            version=meta.get("version", "1.0.0"),
            author=meta.get("author", ""),
            description=meta.get("description", ""),
            active=self.active_theme_id == theme_id,
            trusted=trusted,
            has_components=has_components,
            component_count=component_count,
            builtin=builtin,
            has_preview=has_preview,
            has_cli_theme=has_cli_theme,
            has_cli_modules=has_cli_modules,
            cli_module_count=cli_module_count,
            features=features,
        )

    def _resolve_theme_dir(self, theme_id: str) -> Path | None:
        """テーマ ID からディレクトリを解決"""
        if not theme_id:
            return None
        theme_dir = self.themes_dir / theme_id
        if theme_dir.exists() and theme_dir.is_dir():
            return theme_dir
        return None

    def _update_config(self, updater: Callable[[dict], None]) -> None:
        """config.yaml を読み込み、updater で theme セクションを更新して書き戻す。

        Raises:
            RuntimeError: config.yaml への読み書きに失敗した場合
        """
        import yaml

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

        data.setdefault("theme", {})
        updater(data)

        try:
            atomic_write_text(
                config_path,
                yaml.dump(
                    data, default_flow_style=False,
                    allow_unicode=True, sort_keys=False,
                ),
                fsync=True,
            )
        except Exception as e:
            logger.error("Failed to write config.yaml: %s", e)
            raise RuntimeError(f"Failed to persist theme settings: {e}") from e

    def _persist_active_theme(self, theme_id: str, color_mode: str) -> None:
        """config.yaml にアクティブテーマを永続化"""
        def updater(data: dict) -> None:
            data["theme"]["active"] = theme_id
            data["theme"]["color_mode"] = color_mode

        self._update_config(updater)
        logger.debug("Persisted active theme to config.yaml: %s/%s", theme_id, color_mode)

    def _persist_trusted_list(self) -> None:
        """config.yaml に信頼済みテーマリストを永続化"""
        def updater(data: dict) -> None:
            data["theme"]["trusted"] = sorted(self._trusted_ids)

        self._update_config(updater)
        logger.debug("Persisted trusted themes: %s", self._trusted_ids)
