"""設定 API"""

import asyncio
import copy

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from backend.app_state import AppState, get_app_state
from backend.config import get_config, save_config_section
from backend.schemas import EvorefConfig
from backend.edition import current_edition, Edition
from backend.free.api._error_responses import api_error
from backend.free.api.schemas import (
    ConfigFullResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ConfigValidateResponse,
    LocaleRequest,
    LocaleResponse,
    LocalesResponse,
    RuntimeInfo,
    RuntimePathRequest,
)
from backend.free.core import runtimes
from backend.i18n_helper import set_locale, get_locale, available_locales
from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

logger = get_logger("api.config")

router = APIRouter(prefix="/api/config", tags=["config"])

# Pro 専用セクション（Free では読み書き不可）
PRO_ONLY_SECTIONS = {"widget_proxy", "mode_models"}

# 機密フィールド（GET 時にマスク）。現状マスク対象なし。
SENSITIVE_FIELDS: dict[str, list[str]] = {}

# EvorefConfig で定義されたセクション名
VALID_SECTIONS = set(EvorefConfig.model_fields.keys())


def _mask_sensitive(config: dict) -> dict:
    """機密フィールドをマスクしたコピーを返す"""
    result = copy.deepcopy(config)
    for section, fields in SENSITIVE_FIELDS.items():
        if section in result and isinstance(result[section], dict):
            for field in fields:
                if field in result[section] and result[section][field]:
                    result[section][field] = "***"
    return result


def _is_pro() -> bool:
    """Pro 以上のエディション (Pro / Develop) かどうか。

    Develop は Pro の上位互換のため Pro 専用セクションも書き込み可能。
    Free のみ Pro セクションを除外する。
    """
    return current_edition() >= Edition.PRO


def _guard_model_paths_immutable(data: dict) -> None:
    """model_paths の model_state 追跡キー変更を 403 で遮断する。

    base/aux/embed のモデルは migrate API 経由でしか変更できない。
    config を直書きすると model_state.json と desync し起動時 mismatch を招くため、
    現在値と異なる追跡キーが含まれていれば 403 を返す。create_model 等の非追跡
    キーは通常通り編集可能 (現在値と一致する追跡キーが同梱されていても許可)。
    """
    from backend.free.core.model_migration import MODEL_STATE_TRACKED_KEYS

    current = get_config().get("model_paths", {}) or {}
    changed = sorted(
        k
        for k in MODEL_STATE_TRACKED_KEYS
        if k in data and str(data[k] or "") != str(current.get(k, "") or "")
    )
    if changed:
        raise api_error(
            403,
            "E0403",
            f"model_paths keys are migrate-only: {', '.join(changed)}. "
            f"Use POST /api/model/migrate or /api/model/{{component}}/migrate.",
            "api.config_model_paths_immutable",
            keys=", ".join(changed),
        )


#: 設定 API から危険側へは変えられない「能力キー」(docs/c_06 §1.5)。
#: 値がそのままコード実行・秘密の持ち出し・任意パスの読み書きにつながるので、
#: 変更は config.yaml を手で編集する (PC の持ち主だけができる経路) に限る。
#: 空文字のパスはセクション全体。
CAPABILITY_KEYS: dict[str, tuple[str, ...]] = {
    "local_paths": ("",),
    "runtime": ("fit_params_binary",),
    "llama": ("extra_args", "speculative.draft_model_path"),
    "theme": ("trusted",),
    "rag": ("project_map.roots",),
    "learning": ("cvector_seed_pairs_file",),
    "agent": ("dangerous_command_block",),
    "tools": ("fetch_url_allow_private_ip",),
    "widget_proxy": ("apis",),
    "create": ("staged.verify", "runtimes"),
}
#: 安全側と見なす真偽値 (この値への変更は許す)。
_SAFE_BOOL_VALUES: dict[tuple[str, str], bool] = {
    ("agent", "dangerous_command_block"): True,
    ("tools", "fetch_url_allow_private_ip"): False,
}
#: 要素を減らす変更だけ許すリスト (信頼の取り消し・許可の縮小)。
_SHRINK_ONLY_LISTS = frozenset({
    ("theme", "trusted"), ("llama", "extra_args"), ("rag", "project_map.roots"), ("widget_proxy", "apis"),
})
_MISSING = object()


def _dig(data: object, dotted: str) -> object:
    if not dotted:
        return data
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _is_safe_change(section: str, path: str, old: object, new: object) -> bool:
    if new == old:
        return True
    if (section, path) in _SAFE_BOOL_VALUES:
        return new is _SAFE_BOOL_VALUES[(section, path)]
    if (section, path) in _SHRINK_ONLY_LISTS and isinstance(new, list) and isinstance(old, list):
        return all(item in old for item in new)
    return False


def _guard_capability_keys(section: str, data: dict) -> None:
    """能力キーを危険側へ変える要求を 403 で拒否する (安全側への変更は通す)。"""
    paths = CAPABILITY_KEYS.get(section)
    if not paths:
        return
    current = get_config().get(section, {}) or {}
    blocked: list[str] = []
    for path in paths:
        new = _dig(data, path)
        if new is _MISSING:
            continue
        old = _dig(current, path)
        if path == "" and isinstance(new, dict) and isinstance(old, dict):
            changed = [k for k, v in new.items() if old.get(k, _MISSING) != v]
            blocked.extend(f"{section}.{k}" for k in changed)
            continue
        if not _is_safe_change(section, path, None if old is _MISSING else old, new):
            blocked.append(f"{section}.{path}" if path else section)
    if blocked:
        raise api_error(
            403,
            "E0403",
            f"capability keys cannot be changed via the API: {', '.join(sorted(blocked))}. "
            "Edit config.yaml directly.",
            "api.config_capability_key_protected",
            keys=", ".join(sorted(blocked)),
        )


# ── 固定パスのエンドポイント（{section} より先に定義） ──


@router.put("/locale", response_model=LocaleResponse)
async def update_locale(req: LocaleRequest):
    """ロケール切り替え"""
    locales = available_locales()
    if req.locale not in locales:
        raise api_error(
            400, "E0400", f"Unsupported locale: {req.locale}",
            "api.config_unsupported_locale", locale=req.locale,
        )

    set_locale(req.locale)
    logger.info("Locale changed to %s", req.locale)
    return LocaleResponse(locale=req.locale)


@router.get("/locales", response_model=LocalesResponse)
async def get_locales():
    """利用可能ロケール一覧"""
    return LocalesResponse(
        locales=available_locales(),
        current=get_locale(),
        prompt_locale=get_config().get("i18n", {}).get("prompt_locale", "ja"),
    )


# ── create の実行環境 (c_06 §1.5 の例外 / f_10 §12.4) ──
#
# ``create.runtimes.<name>`` は能力キー (汎用 PUT からは書けない) だが、ここだけは
# 保存前に ``validate_runtime_path`` を通して書ける。起動する引数は固定で設定から増やせない。


def _runtime_info(name: str, cfg: dict) -> RuntimeInfo:
    """実行環境 1 件の状態 (解決・版の取得はファイル I/O と子プロセスを伴う — スレッドで呼ぶ)。"""
    configured = str((((cfg.get("create") or {}).get("runtimes") or {}).get(name)) or "")
    found, reason = runtimes.resolve_runtime(name, cfg)
    if found is None:
        return RuntimeInfo(
            name=name, configured=configured, resolved=None, source=None, version="",
            error=reason if configured.strip() else None,
        )
    return RuntimeInfo(
        name=name, configured=configured, resolved=str(found),
        source="configured" if reason == "configured" else "path",
        version=runtimes.runtime_version(name, found), error=None,
    )


async def _runtime_info_async(name: str) -> RuntimeInfo:
    loop = asyncio.get_running_loop()
    return await run_in_executor_with_context(loop, None, _runtime_info, name, get_config())


@router.get("/runtimes", response_model=list[RuntimeInfo])
async def get_runtimes():
    """create の実行環境 (``create.runtimes.*``) の設定値・解決先・版"""
    return [await _runtime_info_async(name) for name in runtimes.RUNTIME_NAMES]


@router.put("/runtimes/{name}", response_model=RuntimeInfo)
async def update_runtime_path(name: str, req: RuntimePathRequest):
    """実行環境のパスを検証してから保存する。空文字列は設定を消す (PATH から探す)。"""
    if name not in runtimes.RUNTIME_NAMES:
        raise api_error(
            404, "E0404", f"Unknown runtime: {name}",
            "api.config_unknown_runtime", name=name,
        )
    value = ""
    if req.path.strip():
        loop = asyncio.get_running_loop()
        found, reason = await run_in_executor_with_context(
            loop, None, runtimes.validate_runtime_path, name, req.path,
        )
        if found is None:
            raise api_error(
                422, "E0422", f"create.runtimes.{name} is not usable: {reason}",
                f"api.runtime_path_invalid.{reason}", name=name, reason=reason,
            )
        value = str(found)

    try:
        save_config_section("create", {"runtimes": {name: value}})
    except ValidationError as e:
        errors = [str(err["msg"]) for err in e.errors()]
        raise HTTPException(status_code=422, detail={
            "code": "E0422", "message": str(e),
            "i18n_key": "", "context": {},
            "errors": errors,
        })
    except Exception as e:
        logger.error("Failed to save create.runtimes.%s: %s", name, e)
        raise api_error(500, "E0500", str(e))

    logger.info("create.runtimes.%s updated (%s)", name, "cleared" if not value else "set")
    return await _runtime_info_async(name)


# ── 全設定取得 ──


@router.get("", response_model=ConfigFullResponse)
async def get_full_config():
    """全設定取得（機密フィールドはマスク、Free は Pro セクション除外）"""
    config = get_config()
    masked = _mask_sensitive(config)

    is_pro = _is_pro()
    edition_name = current_edition().name.lower()

    # Free の場合 Pro 専用セクションを除外
    if not is_pro:
        for section in PRO_ONLY_SECTIONS:
            masked.pop(section, None)

    sections = [s for s in masked.keys() if isinstance(masked[s], dict)]

    return ConfigFullResponse(
        config=masked,
        sections=sections,
        edition=edition_name,
    )


# ── セクション単位のエンドポイント（パスパラメータ） ──


@router.put("/{section}", response_model=ConfigUpdateResponse)
async def update_config_section(
    section: str, req: ConfigUpdateRequest, state: AppState = Depends(get_app_state),
):
    """設定セクション更新"""
    # セクション名チェック
    if section not in VALID_SECTIONS:
        raise api_error(
            404, "E0404", f"Unknown config section: {section}",
            "api.config_unknown_section", section=section,
        )

    # Free で Pro 専用セクションへの書き込みを拒否
    if not _is_pro() and section in PRO_ONLY_SECTIONS:
        raise api_error(
            403, "E0403", f"Section '{section}' requires Pro edition",
            "api.config_pro_only", section=section,
        )

    # model_paths の model_state 追跡キーは migrate 専用 (config 直書きを遮断)
    if section == "model_paths":
        _guard_model_paths_immutable(req.data)

    # 能力キー (コード実行・秘密・任意パスにつながる値) は API から危険側へ変えない
    _guard_capability_keys(section, req.data)

    # api_key マスク値の場合は既存値を保持
    data = dict(req.data)
    if section in SENSITIVE_FIELDS:
        current = get_config().get(section, {})
        for field in SENSITIVE_FIELDS[section]:
            if field in data and data[field] == "***":
                data[field] = current.get(field, "")

    try:
        save_config_section(section, data)
    except ValidationError as e:
        errors = [str(err["msg"]) for err in e.errors()]
        raise HTTPException(status_code=422, detail={
            "code": "E0422", "message": str(e),
            "i18n_key": "", "context": {},
            "errors": errors,
        })
    except Exception as e:
        logger.error("Failed to save config section '%s': %s", section, e)
        raise api_error(500, "E0500", str(e))

    logger.info("Config section '%s' updated", section)

    # コンポーネント再生成（設定変更を即座に反映）
    await _reload_components_if_needed(section, state)

    return ConfigUpdateResponse(section=section, updated=True)


@router.post("/{section}/validate", response_model=ConfigValidateResponse)
async def validate_config_section(section: str, req: ConfigUpdateRequest):
    """設定セクションのバリデーションのみ（保存しない）"""
    if section not in VALID_SECTIONS:
        raise api_error(
            404, "E0404", f"Unknown config section: {section}",
            "api.config_unknown_section", section=section,
        )

    # 現在の設定に対象セクションだけマージして検証
    from backend.config import _deep_merge

    config = copy.deepcopy(get_config())
    if section in config and isinstance(config[section], dict):
        config[section] = _deep_merge(config[section], req.data)
    else:
        config[section] = req.data

    try:
        EvorefConfig.model_validate(config)
        return ConfigValidateResponse(section=section, valid=True)
    except ValidationError as e:
        errors = [str(err["msg"]) for err in e.errors()]
        return ConfigValidateResponse(section=section, valid=False, errors=errors)


# ── コンポーネント再生成 ──

# 設定変更時にコンポーネント再生成が必要なセクション
_RELOAD_HANDLERS: dict[str, str] = {
    "embedding": "reload_embedder",
    "instance": "reload_prompt_manager",
    "i18n": "reload_i18n",
}


async def _reload_components_if_needed(section: str, state: AppState) -> None:
    """セクション名に応じてコンポーネントを再生成する"""
    handler_name = _RELOAD_HANDLERS.get(section)
    if handler_name is None:
        return

    from backend.free.api.config.component_reload import (
        reload_embedder,
        reload_i18n,
        reload_prompt_manager,
    )

    handlers = {
        "reload_embedder": reload_embedder,
        "reload_prompt_manager": reload_prompt_manager,
        "reload_i18n": reload_i18n,
    }
    handler = handlers[handler_name]
    try:
        await handler(state)
    except Exception as e:
        logger.error("Component reload failed for section '%s': %s", section, e)
