"""CartridgeManager の registry.json 永続化

`backend.free.rag.cartridge_manager.CartridgeManager` からドメインロジックを
分離するための infra 層。`CartridgeRegistryStore` は `CartridgeInfo` の
シリアライズ / デシリアライズと `registry.json` のファイル I/O のみを担い、
カートリッジの load / unload / search / auto_load 等のドメインルールは持たない。

レイヤー責務:
- `CartridgeManager`        — ドメイン (install / load / unload / search / auto_load)
- `CartridgeRegistryStore`  — インフラ (registry.json 永続化、ファイル I/O)

このため `CartridgeRegistryStore` は import 時に `CartridgeManager` を参照せず、
`CartridgeInfo` のみに依存する (循環依存防止 + 単体テスト可能性確保)。
"""

from __future__ import annotations

import json
from dataclasses import fields as dc_fields
from pathlib import Path

from backend.io.atomic import atomic_write_text
from backend.free.rag.cartridge_manager import CartridgeInfo
from backend.log_config import get_logger

logger = get_logger("rag.cartridge_registry_store")


class CartridgeRegistryStore:
    """CartridgeManager の registry.json 純粋永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    REGISTRY_FILENAME = "registry.json"

    @staticmethod
    def serialize(registry: dict[str, CartridgeInfo]) -> list[dict]:
        """`CartridgeInfo` 辞書を JSON-serializable な list[dict] に変換する。

        フィールド順は安定 (id 昇順) で出力する。
        """
        return [_info_to_dict(info) for info in registry.values()]

    @staticmethod
    def deserialize(data: list[dict]) -> dict[str, CartridgeInfo]:
        """list[dict] から `CartridgeInfo` 辞書を再構築する。

        旧 registry.json には新フィールドが無い場合があるため、未知キーは
        落としてからインスタンス化する (後方互換維持)。`id` を持たないエントリは
        スキップする。
        """
        if not isinstance(data, list):
            return {}
        known_fields = {f.name for f in CartridgeInfo.__dataclass_fields__.values()}
        result: dict[str, CartridgeInfo] = {}
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                logger.warning("Skipping malformed registry entry: %r", item)
                continue
            filtered = {k: v for k, v in item.items() if k in known_fields}
            try:
                info = CartridgeInfo(**filtered)
            except TypeError as e:
                logger.warning("Failed to hydrate CartridgeInfo from %r: %s", item, e)
                continue
            result[info.id] = info
        return result

    @staticmethod
    def registry_path(cartridges_dir: str | Path) -> Path:
        """`registry.json` のフルパスを返す。"""
        return Path(cartridges_dir) / CartridgeRegistryStore.REGISTRY_FILENAME

    @staticmethod
    def save(registry: dict[str, CartridgeInfo], cartridges_dir: str | Path) -> None:
        """`registry` を `registry.json` に書き出す。親ディレクトリは自動作成。"""
        path = CartridgeRegistryStore.registry_path(cartridges_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = CartridgeRegistryStore.serialize(registry)
        atomic_write_text(
            path,
            json.dumps(data, ensure_ascii=False, indent=2),
            fsync=True,
        )
        logger.info("Saved cartridge registry (%d entries) to %s", len(data), path)

    @staticmethod
    def load(cartridges_dir: str | Path) -> dict[str, CartridgeInfo] | None:
        """`registry.json` から `CartridgeInfo` 辞書を読み込む。

        ファイルが存在しない場合は `None` を返す (空辞書とは区別する)。
        パース失敗時も `None` を返し、警告ログを出力する。
        呼び出し側は `None` を「ファイル未存在 = レジストリ未初期化」と解釈できる。
        """
        path = CartridgeRegistryStore.registry_path(cartridges_dir)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load cartridge registry %s: %s", path, e)
            return None
        registry = CartridgeRegistryStore.deserialize(data)
        logger.info("Loaded cartridge registry (%d entries) from %s", len(registry), path)
        return registry


# ──────────────────────────────────────────────────────────────────────────
# private serialize helper (純粋関数として保つ)
# ──────────────────────────────────────────────────────────────────────────


def _info_to_dict(info: CartridgeInfo) -> dict:
    """`CartridgeInfo` を JSON-serializable な dict に変換する (純粋関数)。

    **フィールド列挙は dataclass 由来**。手書きだと新フィールドの書き漏れで
    値が黙って落ちる (``docs_digest`` 追加時に実際に起きうる形だった)。
    """
    return {f.name: getattr(info, f.name) for f in dc_fields(info)}


