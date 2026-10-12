"""設定 API"""

import asyncio
import copy

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from backend.app_state import AppState, get_app_state
from backend.config import get_config, save_config_section
from backend.config_normalize import NewerConfigVersionError
from backend.schemas import EvorefConfig
from backend.edition import current_edition, Edition
from backend.free.api._error_responses import api_error, api_error_detail
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
PRO_ONLY_SECTIONS = {"widget_proxy", "mode_models", "pro"}

#: スキーマに節を持たない透過キー (``extra="allow"``) のうち設定 API が知っている節。
#: ``mode_models`` は未実装でモデルのパスを値に持つため、Pro でも API からは書けない
#: (検証だけはできる)。Free では他の Pro 節と同じく 403 ``api.config_pro_only``。
_API_UNWRITABLE_SECTIONS = frozenset({"mode_models"})

# 機密フィールド（GET 時にマスク）。現状マスク対象なし。
SENSITIVE_FIELDS: dict[str, list[str]] = {}

# EvorefConfig で定義されたセクション名 + 透過キーの節
VALID_SECTIONS = set(EvorefConfig.model_fields.keys()) | _API_UNWRITABLE_SECTIONS


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
    # host: 推論の送り先 (会話・記憶の本文が載る) を外へ向けられる
    "llama": ("extra_args", "speculative.draft_model_path", "host"),
    "embedding": ("llama_host",),
    # server.host は設定画面が編集する (待ち受けの変更)。到達範囲を開く allow_remote だけ守る
    "server": ("allow_remote",),
    "rag": ("project_map.roots",),
    "agent": ("dangerous_command_block",),
    "tools": ("fetch_url_allow_private_ip", "fetch_url_enabled"),
    "widget_proxy": ("apis",),
    "create": ("staged.verify", "runtimes"),
    "pro": ("hub.export_dir", "hub.signing_key_path", "hub.subscribe_dirs", "policy.pack_path"),
}
#: 安全側と見なす真偽値 (この値への変更は許す)。
_SAFE_BOOL_VALUES: dict[tuple[str, str], bool] = {
    ("agent", "dangerous_command_block"): True,
    ("tools", "fetch_url_allow_private_ip"): False,
    ("tools", "fetch_url_enabled"): False,
    ("server", "allow_remote"): False,
}
#: 要素を減らす変更だけ許すリスト (信頼の取り消し・許可の縮小)。
_SHRINK_ONLY_LISTS = frozenset({
    ("llama", "extra_args"), ("rag", "project_map.roots"), ("widget_proxy", "apis"),
    ("pro", "hub.subscribe_dirs"),
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


#: ``launch_args`` を持つ節 (設定 API の節 → その中の位置、c_16 §7.2.4)。
_LAUNCH_ARGS_NODES: dict[str, tuple[str, tuple[str, ...]]] = {
    "llama": ("llama", ()),
    "embedding": ("embedding", ()),
    "rag": ("rerank", ("rerank",)),
}


def _launch_node(data: dict, path: tuple[str, ...]) -> dict | None:
    node: object = data
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def _guard_launch_args(section: str, data: dict) -> None:
    """``launch_args`` の変更を検査する (c_06 §1.5 の例外、c_16 §7.2.4)。

    設定 API から書けるのは性能・メモリの調整フラグだけ (:func:`api_disallowed_flags`)。ファイルや
    ネットワークに触れるフラグは 403 (config.yaml を手で編集する)。解釈できない値は 422。
    """
    from backend.llama_launch_args import LaunchArgsError, api_disallowed_flags, parse_launch_args

    spec = _LAUNCH_ARGS_NODES.get(section)
    if spec is None:
        return
    new = _launch_node(data, spec[1])
    if new is None or "launch_args" not in new:
        return
    current = _launch_node(get_config().get(section, {}) or {}, spec[1]) or {}
    if new["launch_args"] == current.get("launch_args", ""):
        return
    try:
        blocked = api_disallowed_flags(parse_launch_args(str(new["launch_args"] or "")))
    except LaunchArgsError as e:
        raise api_error(422, "E0422", str(e), "api.config_launch_args_invalid", error=str(e)) from e
    if blocked:
        raise api_error(
            403,
            "E0403",
            f"launch_args flags cannot be set via the API: {', '.join(blocked)}. Edit config.yaml directly.",
            "api.config_launch_args_protected",
            flags=", ".join(blocked),
        )


def _drop_launch_derived(section: str, data: dict) -> dict:
    """``launch_args`` が写すキー (``-c`` → ``context_size`` 等) を保存する値から除く。

    画面は検証済みの config (写した値入り) を節ごと送り返すので、そのまま保存すると写した値が
    config.yaml に固定され、``launch_args`` から消しても自動 (環境調整) に戻らなくなる。現在と新しい
    ``launch_args`` の両方が写すキーを除けば、config.yaml にはもとの値が残る。
    """
    from backend.llama_launch_args import derived_keys

    spec = _LAUNCH_ARGS_NODES.get(section)
    if spec is None:
        return data
    kind, path = spec
    new = _launch_node(data, path)
    if new is None:
        return data
    current = _launch_node(get_config().get(section, {}) or {}, path) or {}
    keys = derived_keys(kind, current.get("launch_args")) | derived_keys(kind, new.get("launch_args"))
    if not keys:
        return data
    out = copy.deepcopy(data)
    node = _launch_node(out, path)
    assert node is not None
    for key in keys:
        node.pop(key, None)
    return out


def _validation_error(e: ValidationError) -> HTTPException:
    """保存時の検証失敗を 422 ``api.config_validation_failed`` にする (``errors`` は従来の形のまま)。"""
    errors = [str(err["msg"]) for err in e.errors()]
    detail = api_error_detail(
        "E0422", str(e), "api.config_validation_failed", errors="; ".join(errors),
    )
    detail["errors"] = errors
    return HTTPException(status_code=422, detail=detail)


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
        raise _validation_error(e) from e
    except NewerConfigVersionError as e:
        raise _newer_version_error(e) from e
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
    _check_section(section)

    if section in _API_UNWRITABLE_SECTIONS:
        raise api_error(
            403, "E0403",
            f"capability keys cannot be changed via the API: {section}. Edit config.yaml directly.",
            "api.config_capability_key_protected", keys=section,
        )

    # model_paths の model_state 追跡キーは migrate 専用 (config 直書きを遮断)
    if section == "model_paths":
        _guard_model_paths_immutable(req.data)

    # 能力キー (コード実行・秘密・任意パスにつながる値) は API から危険側へ変えない
    _guard_capability_keys(section, req.data)
    _guard_launch_args(section, req.data)

    # api_key マスク値の場合は既存値を保持
    data = _drop_launch_derived(section, dict(req.data))
    if section in SENSITIVE_FIELDS:
        current = get_config().get(section, {})
        for field in SENSITIVE_FIELDS[section]:
            if field in data and data[field] == "***":
                data[field] = current.get(field, "")

    before = copy.deepcopy(get_config().get(section))
    try:
        save_config_section(section, data)
    except ValidationError as e:
        raise _validation_error(e) from e
    except NewerConfigVersionError as e:
        raise _newer_version_error(e) from e
    except Exception as e:
        logger.error("Failed to save config section '%s': %s", section, e)
        raise api_error(500, "E0500", str(e))

    logger.info("Config section '%s' updated", section)

    # コンポーネント再生成（設定変更を即座に反映）。反映できなければ再起動を求める
    applied = await _reload_components_if_needed(section, state, before)

    return ConfigUpdateResponse(
        section=section, updated=True,
        restart_required=section not in _RELOAD_HANDLERS or applied is False,
    )


def _newer_version_error(e: NewerConfigVersionError) -> HTTPException:
    """新しい版の config.yaml は書き戻さない (409)。"""
    return api_error(
        409, "E0409", str(e), "api.config_newer_version", version=e.version,
    )


def _check_section(section: str) -> None:
    """未知の節は 404、Free の Pro 専用節は 403 (PUT と validate で同じ応答にする)。"""
    if section not in VALID_SECTIONS:
        raise api_error(
            404, "E0404", f"Unknown config section: {section}",
            "api.config_unknown_section", section=section,
        )
    if not _is_pro() and section in PRO_ONLY_SECTIONS:
        raise api_error(
            403, "E0403", f"Section '{section}' requires Pro edition",
            "api.config_pro_only", section=section,
        )


@router.post("/{section}/validate", response_model=ConfigValidateResponse)
async def validate_config_section(section: str, req: ConfigUpdateRequest):
    """設定セクションのバリデーションのみ（保存しない）"""
    _check_section(section)

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

#: 埋め込み用 llama-server の起動 (``scripts/launch_llama.py`` の ``_embed_cmd``) に効くキー。
#: 変わったらクライアントを作り直しても稼働中のサーバとずれるだけなので、差し替えずに再起動を求める。
_EMBED_SERVER_KEYS = frozenset({
    "backend", "llama_host", "llama_port", "context_size", "max_length", "pooling", "gpu_layers",
    "batch_size", "ubatch_size", "threads", "slots", "cache_ram_mib", "launch_args",
})


def _changed_keys(before: object, after: object) -> set[str]:
    old = before if isinstance(before, dict) else {}
    new = after if isinstance(after, dict) else {}
    return {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}


async def _reload_components_if_needed(
    section: str, state: AppState, before: object = None,
) -> bool:
    """セクション名に応じてコンポーネントを再生成する。

    Returns:
        その場で反映できたか。False なら呼出側は ``restart_required`` を立てる
        (再生成の失敗・埋め込みサーバの起動に効くキーの変更)。
    """
    handler_name = _RELOAD_HANDLERS.get(section)
    if handler_name is None:
        return False

    if section == "embedding":
        server_keys = _changed_keys(before, get_config().get("embedding")) & _EMBED_SERVER_KEYS
        if server_keys:
            logger.warning(
                "Embedding server settings changed (%s); restart is required to apply them",
                ", ".join(sorted(server_keys)),
            )
            return False

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
        return False
    return True
