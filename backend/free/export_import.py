"""Free 版エクスポート / インポート マネージャ（基本版）

5カテゴリ一括エクスポート + merge モードインポートを提供する。
Pro 版は本モジュールのクラスを継承して拡張する。
"""

import json
import os
import sys
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

from backend.free.__version__ import __version__ as _FREE_VERSION
from backend.log_config import get_logger
from backend.utils import utc_compact_stamp, utc_now_dt

logger = get_logger("export_import")

# Free 版で利用可能な 5 カテゴリ
FREE_CATEGORIES = ["memory", "experience", "rag", "prompts", "cartridges"]

# カテゴリ → ローカルパスキーのマッピング（全エディション共通）
CATEGORY_PATHS: dict[str, list[str]] = {
    "memory": ["memory_dir"],
    "experience": [
        "experience_file", "eval_core_file",
    ],
    "rag": ["vectors_dir", "knowledge_dir"],
    "prompts": ["prompts_dir"],
    "lora": [
        "lora_adapter", "lora_versions_dir",
    ],
    "cartridges": ["cartridges_dir"],
    "history": ["history_dir"],
}


@dataclass
class ExportManifest:
    """エクスポートマニフェスト"""
    format_version: str = "1.0"
    exported_at: str = ""
    evoref_version: str = _FREE_VERSION
    instance_name: str = "evoref"
    source_platform: str = ""
    categories: list[str] = field(default_factory=list)
    embedding_model: dict = field(default_factory=dict)
    base_model: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    total_size_mb: float = 0.0


class ExportManager:
    """Free 版エクスポート（5カテゴリ一括）"""

    # サブクラスでオーバーライド可能
    ALLOWED_CATEGORIES = FREE_CATEGORIES

    def __init__(self, resolver, config: dict):
        self.resolver = resolver
        self.config = config

    def export_to_zip(self, output_path: Path) -> Path:
        """5カテゴリを一括で ZIP にエクスポート

        Args:
            output_path: 出力先パス（ディレクトリなら自動命名、ファイルならそのまま）

        Returns:
            生成された ZIP ファイルのパス
        """
        categories = list(self.ALLOWED_CATEGORIES)

        # 出力先決定
        if output_path.is_dir():
            # UTC 固定・末尾 Z 付与でサーバ tz に依存しない
            ts = utc_compact_stamp()
            output_path = output_path / f"evoref-export-{ts}.evoref-export.zip"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # マニフェスト生成
        manifest = self._build_manifest(categories)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            total_size = 0
            for cat in categories:
                size = self._add_category(zf, cat)
                total_size += size

            manifest.total_size_mb = total_size / (1024 * 1024)
            zf.writestr(
                "export-manifest.json",
                json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            )

        logger.info(
            "Export completed: %s (%.1f MB, categories=%s)",
            output_path, manifest.total_size_mb, categories,
        )
        return output_path

    def _build_manifest(self, categories: list[str]) -> ExportManifest:
        """export-manifest.json を生成"""
        emb_cfg = self.config.get("embedding", {})
        base_model = self.config.get("model_paths", {}).get("base_model") or ""
        instance_name = self.config.get("instance", {}).get("name", "evoref")

        return ExportManifest(
            exported_at=utc_now_dt().isoformat(),
            instance_name=instance_name,
            source_platform=sys.platform,
            categories=categories,
            embedding_model={
                "name": emb_cfg.get("model_name") or "",
                "dim": emb_cfg.get("dim", 1024),
            },
            base_model={"filename": Path(base_model).name if base_model else ""},
        )

    def _walk_category_files(
        self, category: str,
    ) -> list[tuple[Path, str]]:
        """カテゴリに属するファイルを走査し (実パス, アーカイブ名) のリストを返す"""
        results: list[tuple[Path, str]] = []
        for path_key in CATEGORY_PATHS.get(category, []):
            try:
                resolved = self.resolver.resolve_local(path_key)
            except (KeyError, ValueError):
                continue

            if not resolved.exists():
                continue

            if resolved.is_file():
                results.append((resolved, f"data/{resolved.name}"))
            elif resolved.is_dir():
                for root, _, files in os.walk(resolved):
                    for fname in files:
                        fpath = Path(root) / fname
                        rel = fpath.relative_to(resolved.parent)
                        results.append((fpath, f"data/{rel.as_posix()}"))

        return results

    def _add_category(self, zf: zipfile.ZipFile, category: str) -> int:
        """カテゴリのファイルを ZIP に追加。追加バイト数を返す。"""
        total = 0
        for fpath, arc_name in self._walk_category_files(category):
            zf.write(fpath, arc_name)
            total += fpath.stat().st_size
        return total

    def _count_category(self, category: str) -> tuple[int, float, str]:
        """カテゴリの項目数とサイズを返す"""
        entries = self._walk_category_files(category)
        items = len(entries)
        size_bytes = sum(fpath.stat().st_size for fpath, _ in entries)
        size_mb = size_bytes / (1024 * 1024)
        return items, size_mb, f"{items} files ({size_mb:.1f} MB)"

    @staticmethod
    def _validate_categories(categories: list[str], allowed: list[str]) -> None:
        """カテゴリの妥当性を検証"""
        invalid = set(categories) - set(allowed)
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")


class ImportManager:
    """Free 版インポート（merge モードのみ）"""

    def __init__(self, resolver, config: dict):
        self.resolver = resolver
        self.config = config

    def import_from_zip(self, zip_path: Path) -> dict:
        """merge モードでインポート

        Args:
            zip_path: ZIP ファイルパス

        Returns:
            インポート結果 dict
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"File not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = self._read_manifest(zf)
            compatibility = self._check_compatibility(manifest)

            # Free 版対象カテゴリのみインポート
            categories = [
                c for c in manifest.categories
                if c in FREE_CATEGORIES
            ]

            result = {
                "mode": "merge",
                "dry_run": False,
                "compatibility": compatibility,
                "results": [],
            }

            for cat in categories:
                cat_result = self._import_category(zf, cat, "merge", manifest)
                result["results"].append(cat_result)

        logger.info("Import completed: mode=merge, categories=%s", categories)
        return result

    def _read_manifest(self, zf: zipfile.ZipFile) -> ExportManifest:
        """マニフェストを読み込む"""
        try:
            raw = zf.read("export-manifest.json")
        except KeyError:
            raise ValueError("export-manifest.json not found in ZIP")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"export-manifest.json is invalid JSON: {e}")

        return ExportManifest(
            format_version=data.get("format_version", "1.0"),
            exported_at=data.get("exported_at", ""),
            evoref_version=data.get("evoref_version", ""),
            instance_name=data.get("instance_name", ""),
            source_platform=data.get("source_platform", ""),
            categories=data.get("categories", []),
            embedding_model=data.get("embedding_model", {}),
            base_model=data.get("base_model", {}),
            stats=data.get("stats", {}),
            total_size_mb=data.get("total_size_mb", 0.0),
        )

    def _check_compatibility(self, manifest: ExportManifest) -> dict:
        """埋め込みモデル互換性チェック"""
        emb_cfg = self.config.get("embedding", {})
        base_model = self.config.get("model_paths", {}).get("base_model") or ""

        exp_emb_name = manifest.embedding_model.get("name", "")
        local_emb_name = emb_cfg.get("model_name") or ""
        emb_match = exp_emb_name == local_emb_name

        exp_base = manifest.base_model.get("filename", "")
        local_base = Path(base_model).name if base_model else ""
        base_match = exp_base == local_base or not exp_base

        return {
            "embedding_match": emb_match,
            "base_model_match": base_match,
            "reembed_required": not emb_match,
        }

    def _import_category(
        self,
        zf: zipfile.ZipFile,
        category: str,
        mode: str,
        manifest: ExportManifest,  # noqa: ARG002
    ) -> dict:
        """カテゴリ単位のインポート"""
        result = {
            "category": category,
            "action": "imported",
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "warnings": [],
        }

        # ZIP 内の data/ 配下からファイルを抽出
        prefix_map: dict[str, Path] = {}
        for path_key in CATEGORY_PATHS.get(category, []):
            try:
                resolved = self.resolver.resolve_local(path_key)
                if resolved.is_file() or not resolved.exists():
                    prefix_map[f"data/{resolved.name}"] = resolved
                else:
                    prefix_map[f"data/{resolved.name}/"] = resolved
            except (KeyError, ValueError):
                continue

        # replace モード: 既存を削除（Pro 版でオーバーライド時に使用）
        if mode == "replace":
            import shutil
            for _, local_path in prefix_map.items():
                if local_path.exists():
                    if local_path.is_file():
                        local_path.unlink()
                    elif local_path.is_dir():
                        shutil.rmtree(local_path)

        # ファイル抽出
        for zip_entry in zf.namelist():
            if zip_entry == "export-manifest.json":
                continue
            if not zip_entry.startswith("data/"):
                continue

            # パストラバーサル防止
            if ".." in zip_entry:
                result["warnings"].append(f"Skipped suspicious path: {zip_entry}")
                continue

            target = self._resolve_zip_entry(zip_entry, prefix_map)
            if target is None:
                continue

            if zip_entry.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue

            # merge モード: 既存ファイルがあればスキップ
            if mode == "merge" and target.exists():
                result["skipped"] += 1
                continue

            # 抽出
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zip_entry) as src, open(target, "wb") as dst:
                dst.write(src.read())
            result["added"] += 1

        return result

    def _resolve_zip_entry(
        self,
        zip_entry: str,
        prefix_map: dict[str, Path],
    ) -> Path | None:
        """ZIP エントリをローカルパスに解決"""
        for prefix, local_base in prefix_map.items():
            if zip_entry == prefix:
                return local_base
            if zip_entry.startswith(prefix):
                rel = zip_entry[len(prefix):]
                if local_base.is_dir() or not local_base.exists():
                    return local_base / rel
                else:
                    return None
        return None
