"""文書テンプレート API (c_16 §4.5.2)

`GET /api/templates` — インストール済みテンプレートの一覧。
`POST /api/templates/register` — 単一ファイルをその場で `.evocart` に組んで
install する (`/api/rag/ingest` と同じ作法)。この経路で作れるのは ``base``
だけのエントリ (体裁の継承)。``outline`` / ``fields`` は manifest を書いた
パッケージで配る。
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.app_state import AppState, get_app_state
from backend.free.api._error_responses import api_error
from backend.free.api.content._rag_helpers import (
    corpus_store_not_initialized_error,
    rag_file_empty_error,
    rag_filename_required_error,
    rag_unsupported_format_error,
)
from backend.free.rag.corpus.office_inspect import convert_template_extension
from backend.free.rag.corpus.package import (
    PackageError,
    PackageMeta,
    TEMPLATES_DIR,
    package_filename,
    write_package,
)
from backend.free.rag.corpus.templates import (
    BASE_SUFFIXES,
    TEMPLATES_FORMAT_VERSION,
    write_template_manifest,
)
from backend.log_config import get_logger

logger = get_logger("api.templates")

router = APIRouter(prefix="/api/templates", tags=["templates"])

#: 登録時に受け付ける拡張子 (本体形式 + template 変種、c_16 §4.5.2)。
_ACCEPTED_SUFFIXES = BASE_SUFFIXES | {".dotx", ".potx", ".xltx"}

#: 単一ファイル登録パッケージの版。固定値 (``/api/rag/ingest`` と同じ理由 —
#: id が本文の sha256 由来なので、同じ内容の再登録はその場で入れ直しになる)。
_REGISTER_PACKAGE_VERSION = "1.0.0"

#: ファイル名から取り除く Windows 予約文字。
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|]+')


@router.get("")
async def list_templates(state: AppState = Depends(get_app_state)) -> dict:
    """インストール済み (active 版) の全テンプレートエントリの一覧。"""
    manager = state.cartridge_manager
    if manager is None:
        raise corpus_store_not_initialized_error()
    return {"templates": manager.list_templates()}


def template_package_id(content: bytes) -> str:
    """単一ファイル登録のパッケージ id: ``tpl-`` + ファイル本体 (変換後) の sha256 の先頭 12 hex。"""
    return f"tpl-{hashlib.sha256(content).hexdigest()[:12]}"


def _safe_base_name(filename: str, suffix: str) -> str:
    stem = Path(os.path.basename(filename)).stem.strip()
    stem = _UNSAFE_NAME_RE.sub("_", stem)[:80]
    return f"{stem or 'template'}{suffix}"


@router.post("/register", status_code=201)
async def register_template(
    state: AppState = Depends(get_app_state),
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    aliases: list[str] = Form(default_factory=list),
    lang: str = Form("ja"),
) -> dict:
    """単一ファイルを体裁継承エントリ 1 つの `.evocart` にして install する。

    ``.dotx`` / ``.potx`` / ``.xltx`` は梱包前に content-type を本体形式へ
    書き換える (c_16 §4.5.2 — python-docx / python-pptx が template の
    content-type を ``ValueError`` で拒否するため)。
    """
    if not file.filename:
        raise rag_filename_required_error()
    if not doc_type.strip():
        raise api_error(
            400, "E0400", "A document type (doc_type) is required",
            "api.template_doc_type_required",
        )

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ACCEPTED_SUFFIXES:
        raise rag_unsupported_format_error(ext)

    manager = state.cartridge_manager
    if manager is None:
        raise corpus_store_not_initialized_error()

    content = await file.read()
    if not content:
        raise rag_file_empty_error()
    max_bytes = manager.max_package_bytes
    if len(content) > max_bytes:
        raise api_error(
            413, "E0413",
            f"Package too large: {len(content)} bytes (max {max_bytes})",
            "api.cartridge_package_too_large",
            size=len(content), max_size=max_bytes,
        )
    try:
        content, ext = convert_template_extension(content, ext)
    except ValueError as e:
        logger.warning("Template file %s could not be read: %s", file.filename, e)
        raise api_error(
            400, "E0400", "The template file could not be read",
            "api.template_unreadable",
        ) from e

    package_id = template_package_id(content)
    base_name = _safe_base_name(file.filename, ext)
    clean_aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]

    meta = PackageMeta(
        id=package_id,
        name=doc_type,
        version=_REGISTER_PACKAGE_VERSION,
        language=str(lang or "ja"),
        description=doc_type,
        provides=["templates"],
        requires=[f"{TEMPLATES_DIR}/{TEMPLATES_FORMAT_VERSION}"],
    )
    entry = {
        "id": "base",
        "doc_type": doc_type,
        "aliases": clean_aliases,
        "lang": str(lang or "ja"),
        "base": base_name,
    }

    with tempfile.TemporaryDirectory(prefix="evoref-template-") as tmp:
        staging = Path(tmp) / "src"
        templates_dir = staging / TEMPLATES_DIR
        templates_dir.mkdir(parents=True)
        (templates_dir / base_name).write_bytes(content)
        write_template_manifest(templates_dir, [entry])
        zip_path = Path(tmp) / package_filename(meta)
        try:
            write_package(staging, zip_path, meta)
            info = await manager.install(zip_path, internal=True)
        except PackageError as exc:
            logger.warning("Template registration rejected %s: %s", file.filename, exc)
            raise api_error(400, "E0400", str(exc)) from exc

    logger.info(
        "Registered template package %s v%s from %s (doc_type=%s)",
        info.id, info.version, file.filename, doc_type,
    )
    return {
        "key": f"{info.id}:base",
        "package": info.id,
        "version": info.version,
        "doc_type": doc_type,
        "aliases": clean_aliases,
    }


__all__ = ["router"]
