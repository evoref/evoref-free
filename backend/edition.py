"""エディション判定・プラグインレジストリ

3エディション（Free / Pro / Develop）分離の基盤モジュール。
各エディションはプラグインとしてハンドラ・ルーター・ライフサイクルフックを登録する。

- backend/free/ が必須（なければ起動不可）
- backend/pro/ が存在する場合は setup_pro() で Pro 機能を登録
- backend/develop/ が存在する場合は setup_develop() で Develop 機能を登録
  (Develop は Pro の上位互換: Develop ⊇ Pro ⊇ Free)
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("edition")


class Edition(IntEnum):
    """エディション階層 (整数値が大きいほど上位)"""
    FREE = 0
    PRO = 1
    DEVELOP = 2


# ── Free レジストリ ──
_free_registered: bool = False


def register_free() -> None:
    """Free エディションを登録する"""
    global _free_registered
    _free_registered = True
    logger.info("Free edition registered")


# ── Pro レジストリ ──
_pro_handlers: dict[str, object] = {}
_pro_routers: list = []
# Pro pillar 別 setup フック
# wire_pillars() が対応する pillar 構築フェーズで呼び出す。
_pro_pillar_setup: dict[str, object] = {}
_pro_lifecycle_shutdown = None


def register_pro_handler(name: str, handler: object) -> None:
    """Pro 機能ハンドラを登録する"""
    _pro_handlers[name] = handler
    logger.info("Pro handler registered: %s", name)


def get_pro_handler(name: str) -> object | None:
    """登録済み Pro ハンドラを取得する（未登録時は None）"""
    return _pro_handlers.get(name)


def register_pro_router(router) -> None:
    """Pro ルーターを登録する"""
    _pro_routers.append(router)


def get_pro_routers() -> list:
    """登録済み Pro ルーター一覧を取得する"""
    return list(_pro_routers)


def register_pro_pillar_setup(pillar: str, setup) -> None:
    """Pro pillar 別 setup フックを登録する

    ``wire_pillars()`` は該当 pillar の構築フェーズでこのフックを参照し、
    Pro 拡張オブジェクト (ProAssistComponents / WidgetProxyManager 等) を
    取得する。Free エディションでは登録されないため、
    :func:`get_pro_pillar_setup` が ``None`` を返す。

    Args:
        pillar: ``"gen"`` / ``"learn"`` のいずれか。
        setup: 非同期 callable (pillar ごとに異なるシグネチャ)。
    """
    _pro_pillar_setup[pillar] = setup
    logger.info("Pro pillar setup registered: %s", pillar)


def get_pro_pillar_setup(pillar: str):
    """Pro pillar 別 setup フックを取得する。未登録時は ``None``。"""
    return _pro_pillar_setup.get(pillar)


def register_pro_shutdown(shutdown) -> None:
    """Pro シャットダウンフックを登録する

    Args:
        shutdown: async callable(state) -> None
    """
    global _pro_lifecycle_shutdown
    _pro_lifecycle_shutdown = shutdown
    logger.info("Pro shutdown hook registered")


def get_pro_shutdown():
    """Pro シャットダウンフックを取得する。未登録時は ``None``。"""
    return _pro_lifecycle_shutdown


# ── Develop レジストリ ──
# Develop エディションは Pro の上位互換 (Develop ⊇ Pro ⊇ Free)。
# Pro 用 register_pro_* と完全に同形で複製した API を提供する。
_develop_handlers: dict[str, object] = {}
_develop_routers: list = []
_develop_pillar_setup: dict[str, object] = {}
_develop_lifecycle_shutdown = None


def register_develop_handler(name: str, handler: object) -> None:
    """Develop 機能ハンドラを登録する"""
    _develop_handlers[name] = handler
    logger.info("Develop handler registered: %s", name)


def get_develop_handler(name: str) -> object | None:
    """登録済み Develop ハンドラを取得する（未登録時は None）"""
    return _develop_handlers.get(name)


def register_develop_router(router) -> None:
    """Develop ルーターを登録する"""
    _develop_routers.append(router)


def get_develop_routers() -> list:
    """登録済み Develop ルーター一覧を取得する"""
    return list(_develop_routers)


def register_develop_pillar_setup(pillar: str, setup) -> None:
    """Develop pillar 別 setup フックを登録する

    Pro pillar setup と並列に呼ばれる。Free / Pro エディションでは
    登録されないため、:func:`get_develop_pillar_setup` が ``None`` を返す。

    Args:
        pillar: ``"gen"`` / ``"learn"`` / ``"loop"`` のいずれか。
        setup: 非同期 callable (pillar ごとに異なるシグネチャ)。
    """
    _develop_pillar_setup[pillar] = setup
    logger.info("Develop pillar setup registered: %s", pillar)


def get_develop_pillar_setup(pillar: str):
    """Develop pillar 別 setup フックを取得する。未登録時は ``None``。"""
    return _develop_pillar_setup.get(pillar)


def register_develop_shutdown(shutdown) -> None:
    """Develop シャットダウンフックを登録する

    Args:
        shutdown: async callable(state) -> None
    """
    global _develop_lifecycle_shutdown
    _develop_lifecycle_shutdown = shutdown
    logger.info("Develop shutdown hook registered")


def get_develop_shutdown():
    """Develop シャットダウンフックを取得する。未登録時は ``None``。"""
    return _develop_lifecycle_shutdown


# ── エディション判定 ──

def current_edition() -> Edition:
    """現在のエディションを返す (Develop > Pro > Free の優先順)"""
    if bool(_develop_handlers) or bool(_develop_routers):
        return Edition.DEVELOP
    if bool(_pro_handlers) or bool(_pro_routers):
        return Edition.PRO
    return Edition.FREE


def is_pro_or_above() -> bool:
    """Pro 以上のエディションかどうか (Develop も含む)"""
    return current_edition() >= Edition.PRO


def is_pro() -> bool:
    """Pro エディション以上が有効かどうか（Develop でも True、後方互換エイリアス）"""
    return is_pro_or_above()


def is_develop() -> bool:
    """Develop エディションが有効かどうか"""
    return current_edition() >= Edition.DEVELOP


def pro_available() -> bool:
    """Pro モジュールが存在するかどうか（import 可能性チェック）"""
    try:
        import importlib.util
        return importlib.util.find_spec("backend.pro") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def develop_available() -> bool:
    """Develop モジュールが存在するかどうか（import 可能性チェック）"""
    try:
        import importlib.util
        return importlib.util.find_spec("backend.develop") is not None
    except (ModuleNotFoundError, ValueError):
        return False


# ── ダウングレード検出 ──

_EDITION_STATE_FILE = "edition_state.json"


def check_downgrade(local_dir: Path) -> None:
    """前回のエディションと比較し、ダウングレードを検出して警告を表示する

    Args:
        local_dir: ローカルデータディレクトリ (local/)
    """
    from backend.i18n_helper import msg

    state_file = local_dir / _EDITION_STATE_FILE
    cur = current_edition()

    prev_name: str | None = None
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            prev_name = data.get("edition")
        except Exception:
            pass

    # 前回エディションを保存（次回起動時の比較用）
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({"edition": cur.name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("Failed to save edition state: %s", e)

    if prev_name is None:
        # 初回起動 — 比較対象なし
        return

    try:
        prev = Edition[prev_name]
    except KeyError:
        logger.debug("Unknown previous edition: %s", prev_name)
        return

    if cur >= prev:
        # アップグレードまたは同一 — 警告不要
        return

    # ダウングレード検出
    logger.warning(
        "Edition downgrade detected: %s -> %s",
        prev.name, cur.name,
    )
    logger.warning(msg("warning.edition.downgrade", old=prev.name, new=cur.name))

    # Pro 以上 → Free へのダウングレード: LoRA 警告
    if prev >= Edition.PRO and cur < Edition.PRO:
        logger.warning(msg("warning.edition.lora_unavailable"))

