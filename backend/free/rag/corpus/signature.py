"""`.evocart` の署名検証の差込口 (c_16 §4.3 / docs/p_06)。

Free は署名を **作らず・解釈せず**、検証器の Protocol と install の差込口だけを持つ。
外から持ち込んだパッケージの ``prebuilt/`` は、登録された検証器が ``verified`` を
返したときだけ採用する。検証器が無ければ (Free 単独) 従来どおり ``docs/`` から作り直す。

検証器は Plugin Registry の handler (:data:`SIGNATURE_VERIFIER_HANDLER`) として
登録される (Pro は ``backend/pro/signed_cartridges``)。署名ブロックは
``package.json`` のトップレベル ``signature`` キーで、Free の読み手は
:attr:`PackageMeta._extra` に退避したまま書き戻す (c_16 §4.5.5 の予約)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from backend.free.rag.corpus.package import PackageMeta
from backend.log_config import get_logger

logger = get_logger("rag.corpus.signature")

#: Plugin Registry の handler 名 (``backend.edition.register_pro_handler``)。
SIGNATURE_VERIFIER_HANDLER = "cartridge_signature_verifier"

#: 検証の結果。``verified`` だけが prebuilt の採用を許す。
SignatureStatus = Literal[
    "verified",
    "unsigned",
    "untrusted_signer",
    "revoked",
    "digest_mismatch",
    "bad_signature",
    "malformed",
    "legacy_scheme",
    "verifier_error",
]


@dataclass(frozen=True, slots=True)
class SignatureVerdict:
    """1 パッケージの署名検証の結果 (理由付き)。"""

    status: SignatureStatus
    signer: str = ""
    key_id: str = ""
    #: 英語の補足 (ログ・CLI 用)。
    detail: str = ""

    @property
    def adopt_prebuilt(self) -> bool:
        """``prebuilt/`` (チャンク・埋め込み・重心) をそのまま採用してよいか。"""
        return self.status == "verified"

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "status": self.status,
            "signer": self.signer,
            "key_id": self.key_id,
            "detail": self.detail,
            "adopt_prebuilt": self.adopt_prebuilt,
        }


@runtime_checkable
class CartridgeSignatureVerifier(Protocol):
    """展開直後の版ディレクトリを検証する。

    ``directory`` は zip をそのまま展開した直後 (索引を作る前) の版ディレクトリ。
    検証器は ``meta`` の宣言を信じず、ディレクトリの中身から digest を計算し直す。
    """

    def verify_package(self, directory: Path, meta: PackageMeta) -> SignatureVerdict: ...


def registered_verifier() -> CartridgeSignatureVerifier | None:
    """登録された検証器 (無ければ ``None`` = Free 単独の挙動)。"""
    from backend.edition import get_pro_handler

    handler = get_pro_handler(SIGNATURE_VERIFIER_HANDLER)
    if handler is None:
        return None
    if not isinstance(handler, CartridgeSignatureVerifier):
        logger.warning("Ignoring %s handler without verify_package()", SIGNATURE_VERIFIER_HANDLER)
        return None
    return handler


def verify_installing_package(directory: Path, meta: PackageMeta) -> SignatureVerdict | None:
    """install 中のパッケージを検証する。検証器が無ければ ``None``。

    検証器の例外は ``verifier_error`` に畳む (prebuilt を捨てて作り直すだけで、
    install 自体は止めない)。
    """
    verifier = registered_verifier()
    if verifier is None:
        return None
    try:
        return verifier.verify_package(directory, meta)
    except Exception as e:  # noqa: BLE001 — 検証できなければ作り直す (安全側)
        logger.warning("Signature verifier failed for %s: %s", meta.id, e)
        return SignatureVerdict("verifier_error", detail=str(e))


__all__ = [
    "SIGNATURE_VERIFIER_HANDLER",
    "CartridgeSignatureVerifier",
    "SignatureStatus",
    "SignatureVerdict",
    "registered_verifier",
    "verify_installing_package",
]
