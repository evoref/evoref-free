"""組織ポリシーパックの差込口 (docs/p_09 §2)。

Free は **Protocol と hook だけ** を持つ (e_01 §0.1 原則 4)。パックの署名・適用・固定項目の
判定は Pro (``backend/pro/org_policy/``) が持ち、``setup_pro`` で
``register_pro_handler(POLICY_ENFORCER_HANDLER, enforcer)`` するだけ。

- 登録が無ければ (Free 単独・Pro でもパック未適用) :func:`policy_denial` は常に ``None`` で、
  どの呼出元も今の挙動のまま。
- 判定器の例外は握り潰して ``None`` (許可) に倒し WARNING を出す — 固定は「組織の運用の
  約束」であって安全装置ではない (利用者はファイルを直接編集できる、p_09 §6)。判定器の
  故障でチャット・編集を止めない。

判定の対象 (``target``):

- ``prompt``: そのモードの system プロンプト本文の手編集・巻き戻し・再読込・言語切替・
  成長層の採用 / 巻き戻し・Level 1 の自動採用
- ``generation_params``: そのモードの生成パラメータのデルタの採用 / 巻き戻し・自動採用
- ``tool``: ツールの選択・実行 (許可リスト)
- ``model``: ベースモデルの切替先 (GGUF の SHA-256 の許可リスト)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.log_config import get_logger

logger = get_logger("core.org_policy")

#: Plugin Registry の handler 名 (``backend.edition.register_pro_handler``)。
POLICY_ENFORCER_HANDLER = "policy_pack_enforcer"

TARGET_PROMPT = "prompt"
TARGET_GENERATION_PARAMS = "generation_params"
TARGET_TOOL = "tool"
TARGET_MODEL = "model"
TARGETS: tuple[str, ...] = (TARGET_PROMPT, TARGET_GENERATION_PARAMS, TARGET_TOOL, TARGET_MODEL)

#: 固定された資産を変えようとした (``prompt`` / ``generation_params``)。
REASON_PINNED = "pinned_by_policy"


@dataclass(frozen=True, slots=True)
class PolicyDenial:
    """固定項目を変えようとしたときの拒否の理由 (ログは英語、UI は i18n ``api.policy_denied``)。

    Attributes:
        target: :data:`TARGETS` のどれか。
        reason: 機械可読の理由 (``pinned_by_policy`` / ``tool_not_allowed`` / ``model_not_allowed`` …)。
        detail: 英語の説明。
        org / pack_version / pack_id: 固定しているパック。
    """

    target: str
    reason: str
    detail: str
    org: str = ""
    pack_version: int = 0
    pack_id: str = ""

    def context(self) -> dict[str, Any]:
        """API の構造化エラーの ``context`` に載せる値。"""
        return {
            "reason": self.reason,
            "target": self.target,
            "org": self.org,
            "pack_version": self.pack_version,
            "pack_id": self.pack_id,
        }


class PolicyPackEnforcer(Protocol):
    """固定項目の判定器 (Pro が実装して登録する)。"""

    def check(
        self, target: str, *, mode: str = "", name: str = "", path: str = "",
    ) -> PolicyDenial | None:
        """``target`` の変更・使用を拒否するなら理由を、許すなら ``None`` を返す。

        ``mode`` は ``prompt`` / ``generation_params``、``name`` は ``tool`` のツール名、
        ``path`` は ``model`` の GGUF のパス。
        """
        ...


def _enforcer() -> PolicyPackEnforcer | None:
    from backend.edition import get_pro_handler

    return get_pro_handler(POLICY_ENFORCER_HANDLER)


def policy_denial(
    target: str, *, mode: str = "", name: str = "", path: str = "",
) -> PolicyDenial | None:
    """登録された判定器に尋ねる。登録が無い・判定器が失敗したら ``None`` (今の挙動のまま)。"""
    enforcer = _enforcer()
    if enforcer is None:
        return None
    try:
        return enforcer.check(target, mode=mode, name=name, path=path)
    except Exception as exc:  # noqa: BLE001 - 判定器の故障で編集・チャットを止めない
        logger.warning("Policy enforcer failed (target=%s): %r; allowing", target, exc)
        return None


def tool_allowed(name: str) -> bool:
    """ツールを選択・実行してよいか (登録が無ければ常に真)。"""
    return policy_denial(TARGET_TOOL, name=name) is None


__all__ = [
    "POLICY_ENFORCER_HANDLER",
    "REASON_PINNED",
    "TARGETS",
    "TARGET_GENERATION_PARAMS",
    "TARGET_MODEL",
    "TARGET_PROMPT",
    "TARGET_TOOL",
    "PolicyDenial",
    "PolicyPackEnforcer",
    "policy_denial",
    "tool_allowed",
]
