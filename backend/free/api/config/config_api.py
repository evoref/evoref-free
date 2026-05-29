"""設定 API"""

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
)
from backend.i18n_helper import set_locale, get_locale, available_locales
from backend.log_config import get_logger

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
    )


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
    "reranker": "reload_reranker",
    "assist_model": "reload_assist_model",
    "instance": "reload_prompt_manager",
}


async def _reload_components_if_needed(section: str, state: AppState) -> None:
    """セクション名に応じてコンポーネントを再生成する"""
    handler_name = _RELOAD_HANDLERS.get(section)
    if handler_name is None:
        return

    from backend.free.api.config.component_reload import (
        reload_assist_model,
        reload_embedder,
        reload_prompt_manager,
        reload_reranker,
    )

    handlers = {
        "reload_embedder": reload_embedder,
        "reload_reranker": reload_reranker,
        "reload_assist_model": reload_assist_model,
        "reload_prompt_manager": reload_prompt_manager,
    }
    handler = handlers[handler_name]
    try:
        await handler(state)
    except Exception as e:
        logger.error("Component reload failed for section '%s': %s", section, e)
