"""``SemanticFact`` ↔ ``Evidence`` (kind=``fact``) の対応 (c_16 §3.2 / §4.2)。

``SemanticFact`` は **作業用インメモリ表現** で、永続形は ``Evidence`` 1 本。
:mod:`backend.free.memory.episodic.note` が ``MemoryNote`` に対してやっている
のと同じ形にしてある (ファクトの消費者は 100 ファイル以上あり、返り値の型を
変えると変更が全域に散る)。

## コアへ載せたもの

=========================  ==========================================
``SemanticFact``            ``Evidence``
=========================  ==========================================
``subject`` / ``predicate`` ``structured.subject`` / ``.predicate``
``object``                  ``structured.object`` (発話原文 = 証拠)
``statement``               ``text`` (正規化済み命題。無ければ object)
``scope``                   ``scope`` (``global`` / ``project:<id>``)
``lang``                    ``lang``
``confidence``              ``confidence``
``pinned`` / ``private``    ``pinned`` / ``private``
``superseded_by``           ``superseded_by``
``created_at``              ``as_of`` (= **発話時刻**) と ``observed_at``
``accessed_at``             ``last_used_at``
``provenances``             ``provenance[]``
``trace_id``                ``provenance[0].trace_id``
=========================  ==========================================

``created_at`` を ``as_of`` に載せるのは、抽出器が **抽出時刻ではなく発話時刻**
を入れているため (``extractors/base.py`` の ``_utterance_time``)。c_16 §3 の
``as_of`` は「内容が真だった時点」なので意味が一致する。競合の新旧比較も
``as_of`` で行う (c_16 §1 の表)。

## ``attrs`` に載せたものと消費者

キー名の SSOT は :data:`FACT_ATTR_FIELDS`。

======================= ==================================================
attrs キー               消費者
======================= ==================================================
``fact_type``            **必須**。``FACT_OWNERSHIP`` の鍵
                         (:mod:`backend.free.memory.ownership`)、注入の
                         Tier 分類、GC / 各 View の型別検索
``mode_origin``          ``views.base.merge_active_facts_across_stores``
                         (mode 別の policy / fewshot 絞り込み)、抽出器
``profile_id``           ``agent.tool_call_judge`` の索引リコール
                         (プロファイル越境の抑止)、3 キュレータ
``pin_locked_until``     ``semantic.pin_manager`` (pin の保護期間)
``auto_evolved``         ``SemanticConflictResolver``
                         (``auto_for_evolved_policies``)
``from_correction``      ``SemanticConflictResolver`` / ``MemoryInjector``
                         (訂正は確認を挟まず即採用・加点)
``failure_signature``    ``loop.failure_note`` / ``MemoryInjector``
                         (signature 一致時のみ Tier 1)
``eval_metric``          ``learning.policy_evolver`` / ``fewshot_pool``
``session_ids``          ``cli.purge_private_cmd`` / ``conflict_review``
``embed_as_query`` /     ``rag.evidence.store.embed_side_of`` — snapshot 生成
``embed_mode``           の埋め込みを query 側 + ``mode`` で行う。内部索引
                         ``idx.command.*`` の読み手 (``ToolCallJudge``) が
                         ``embed_query(..., mode=mode)`` で引くため
======================= ==================================================

## 落としたもの

======================= ===============================================
落としたフィールド        理由
======================= ===============================================
``embedding``           ベクトルは ``embeddings/<model_id>/`` (c_16 §6.1)。
                        作業用の一時値としてのみ載る (永続化しない)。
                        **どちらの側で埋め込むか** だけは
                        ``attrs.embed_as_query`` / ``attrs.embed_mode`` で
                        持ち回る
``access_count``        順位式は ``cos × freshness × confidence ×
                        store_prior`` に一本化 (c_16 §7.2)。保持順は
                        ``last_used_at`` (§5.4)
``supersedes``          ``superseded_by`` の逆写像で導出できる
                        (:meth:`SemanticStore.supersedes_of`)
``statement``           ``Evidence.text`` そのもの (二重に持たない)
``subject_aliases``     消費者ゼロ (2026-09-07 時点)
``scope_locked``        同上
``credit_score``        同上
``requires_user_review``  競合の状態は ``veracity`` (``disputed``) と
``review_status``       ``contradicts`` / ``superseded_by`` が持つ
                        (c_16 §4.2)
``_version``            ``Evidence._version`` に一本化
======================= ===============================================
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

from backend.free.memory.semantic.namespaces import namespace_of
from backend.free.memory.types import Provenance, SemanticFact
from backend.free.rag.evidence import (
    Evidence,
    compute_claim_key_structured,
    new_evidence_id,
)
from backend.log_config import get_logger
from backend.utils import format_utc, parse_utc, utc_now

logger = get_logger("memory.semantic.fact")

#: ``attrs`` へ載せるフィールド名 (``SemanticFact`` の属性名と同じ)。
#: 名前を変えないのは、既存の消費者 (``fact.<name>``) がそのまま動くようにするため。
FACT_ATTR_FIELDS: tuple[str, ...] = (
    "fact_type",
    "mode_origin",
    "profile_id",
    "pin_locked_until",
    "auto_evolved",
    "from_correction",
    "failure_signature",
    "eval_metric",
    "session_ids",
    "embed_as_query",
    "embed_mode",
    "retired_note_ids",
)

#: 既定値と同じなら ``attrs`` へ書かない (レコードを太らせない)。
#: ``fact_type`` は既定を持たない — 欠けると ownership が引けないため。
_ATTR_DEFAULTS: dict[str, Any] = {
    "mode_origin": "chat",
    "profile_id": "default",
    "pin_locked_until": None,
    "auto_evolved": False,
    "from_correction": False,
    "failure_signature": None,
    "eval_metric": None,
    "session_ids": [],
    "embed_as_query": False,
    "embed_mode": "chat",
    "retired_note_ids": [],
}

#: ``provenance[].extractor_version`` の既定 (抽出器が明示しない場合)。
FACT_BUILDER_VERSION = 1


class FactRecordError(ValueError):
    """``Evidence`` を ``SemanticFact`` として読めない (``fact_type`` 欠損等)。

    呼び出し側は **そのレコードだけ** 飛ばして件数を WARNING に出すこと
    (c_05 §0.5.2)。
    """


def _iso(epoch: float | None) -> str | None:
    """epoch 秒 → ISO 8601 UTC μs ``Z``。``None`` / 0 は ``None``。"""
    if not epoch:
        return None
    return format_utc(datetime.fromtimestamp(float(epoch), tz=UTC))


def _epoch(value: str | None, default: float = 0.0) -> float:
    """ISO 8601 (UTC) → epoch 秒。読めなければ ``default``。

    文字列のまま比較しない (c_05 §0.5) ため、読み戻しは必ずここを通す。
    """
    parsed = parse_utc(value) if value else None
    return parsed.timestamp() if parsed is not None else default


def _provenance_records(fact: SemanticFact) -> list[dict[str, Any]]:
    """``Provenance`` の列を ``Evidence.provenance`` の形へ落とす (c_16 §3.1)。

    ``captured_at`` は epoch 秒で持たれているので ISO へ揃える (時刻は 1 形式)。
    ``turn_id`` は会話由来のファクトで必須 — 欠けたら ID 連鎖が切れるので
    WARNING に出す (レコードは落とさない。抽出器の系統によっては会話由来で
    ないものもある)。
    """
    out: list[dict[str, Any]] = []
    for prov in fact.provenances or []:
        entry: dict[str, Any] = {
            "session_id": prov.session_id or "",
            "turn_id": prov.turn_id or "",
            "trace_id": prov.trace_id or "",
            "note_id": prov.note_id or "",
            "extractor": prov.extractor or "fact_extractor",
            "extractor_version": (
                int(prov.extractor_version)
                if prov.extractor_version is not None
                else FACT_BUILDER_VERSION
            ),
            "captured_at": _iso(prov.captured_at) or utc_now(),
        }
        if prov.source_id:
            # 取得単位 / 文書の識別子。落とすと ``know.*`` の claim が出所を
            # 失い、裏取り件数も verify の突合も成立しない (c_16 §3.1)。
            entry["source_id"] = prov.source_id
        if prov.mode:
            entry["mode"] = prov.mode
        if prov.project_id:
            entry["project_id"] = prov.project_id
        if prov.source:
            entry["source"] = prov.source
        if prov.model:
            entry["model"] = prov.model
        out.append(entry)
    return out


def _provenances_from(records: list[dict[str, Any]]) -> list[Provenance]:
    """``Evidence.provenance`` から ``Provenance`` の列を復元する。"""
    out: list[Provenance] = []
    for entry in records or []:
        if not isinstance(entry, dict):
            continue
        out.append(
            Provenance(
                note_id=entry.get("note_id") or None,
                session_id=entry.get("session_id") or None,
                turn_id=entry.get("turn_id") or None,
                trace_id=entry.get("trace_id") or None,
                mode=entry.get("mode") or None,
                project_id=entry.get("project_id") or None,
                source=entry.get("source") or None,
                source_id=entry.get("source_id") or None,
                captured_at=_epoch(entry.get("captured_at")),
                extractor=entry.get("extractor") or None,
                extractor_version=entry.get("extractor_version"),
                model=entry.get("model") or None,
            ),
        )
    return out


def fact_to_evidence(
    fact: SemanticFact, *, half_life_days: float | None = None,
) -> Evidence:
    """``SemanticFact`` を ``Evidence`` (kind=``fact``) にする。

    ``confidence`` は **呼出側の値をそのまま載せる**。c_16 §3.3 の決定論導出は
    ``origin`` から確度を決める規則だが、``learn.policy.*`` の confidence は
    「その方針をどれだけ信じているか」という別軸の値で、活性化閾値
    (``learning.policy.activation_min_confidence``) がこれを読む。両者を同じ
    フィールドで潰すと方針の活性化が origin で決まってしまうため、導出は
    ``know.*`` の claim (:class:`KnowledgeIngest`) に限る。

    Args:
        half_life_days: namespace 既定の半減期 (c_16 §4.2)。``know.*`` は
            domain 別、それ以外は ``None``。
    """
    subject = fact.subject or "unknown"
    object_text = fact.object or ""
    body = fact.text or object_text
    as_of = _iso(fact.created_at) or utc_now()

    attrs: dict[str, Any] = {"fact_type": str(fact.type)}
    for name in FACT_ATTR_FIELDS:
        if name == "fact_type":
            continue
        value = getattr(fact, name, None)
        if name == "session_ids":
            value = sorted(value or ())
        if value == _ATTR_DEFAULTS.get(name):
            continue
        attrs[name] = value

    return Evidence(
        id=fact.id or new_evidence_id(),
        kind="fact",
        store="semantic",
        scope=fact.scope or "global",
        text=body,
        lang=fact.lang or None,
        structured={
            "subject": subject,
            "predicate": fact.predicate or "",
            "object": object_text,
            "value": {"kind": "text", "number": None, "unit": None},
        },
        origin=fact.origin,  # type: ignore[arg-type]
        provenance=_provenance_records(fact),
        observed_at=as_of,
        as_of=as_of,
        half_life_days=half_life_days,
        confidence=float(fact.confidence),
        veracity=fact.veracity,  # type: ignore[arg-type]
        claim_key=compute_claim_key_structured(
            subject, fact.predicate or "", body,
        ),
        contradicts=list(fact.contradicts or []),
        superseded_by=fact.superseded_by,
        private=bool(fact.private),
        pinned=bool(fact.pinned),
        created_at=as_of,
        updated_at=utc_now(),
        last_used_at=_iso(fact.accessed_at),
        attrs=attrs,
        _extra=dict(fact._extra or {}),
    )


def evidence_to_fact(record: Evidence) -> SemanticFact:
    """``Evidence`` (kind=``fact``) から作業用 :class:`SemanticFact` を復元する。

    ``attrs`` の未知キーは ``_extra`` へ退避する (捨てない)。

    Raises:
        FactRecordError: ``attrs.fact_type`` が無い。``FACT_OWNERSHIP`` を
            引けないレコードは書込検証も注入の Tier 分類もできないので、
            既定値で誤魔化さずに落とす (c_05 §0.5.2)。
    """
    attrs = dict(record.attrs or {})
    fact_type = attrs.get("fact_type")
    if not fact_type:
        raise FactRecordError(f"fact evidence without attrs.fact_type: {record.id}")

    structured = record.structured if isinstance(record.structured, dict) else {}
    subject = str(structured.get("subject") or "unknown")
    predicate = str(structured.get("predicate") or "")
    object_text = str(structured.get("object") or record.text or "")
    # ``statement`` は正規化済み命題。原文と同じなら「未正規化」として ``None``
    # に戻す (``SemanticFact.text`` はどちらでも同じ本文を返す)。
    statement = record.text if record.text != object_text else None

    created_at = _epoch(record.as_of or record.observed_at)
    fact = SemanticFact(
        id=record.id,
        subject=subject,
        predicate=predicate,
        object=object_text,
        statement=statement,
        type=fact_type,  # type: ignore[arg-type]
        scope=record.scope or "global",
        lang=record.lang or "",
        provenances=_provenances_from(record.provenance),
        confidence=float(record.confidence),
        origin=record.origin,
        veracity=record.veracity,
        contradicts=list(record.contradicts or []),
        pinned=bool(record.pinned),
        superseded_by=record.superseded_by,
        created_at=created_at,
        accessed_at=_epoch(record.last_used_at, created_at),
        private=bool(record.private),
    )
    provenance = record.provenance[0] if record.provenance else {}
    fact.trace_id = str(provenance.get("trace_id") or "") or None

    known = {f.name for f in fields(SemanticFact)}
    extra = dict(record._extra or {})
    for key, value in attrs.items():
        if key == "fact_type":
            continue
        if key == "session_ids":
            fact.session_ids = set(value or ())
            continue
        if key in known:
            setattr(fact, key, value)
        else:
            extra[key] = value
    fact._extra = extra
    return fact


def fact_namespace(fact: SemanticFact) -> str:
    """``fact.subject`` の namespace (c_16 §4.2)。"""
    return namespace_of(fact.subject)


__all__ = [
    "FACT_ATTR_FIELDS",
    "FACT_BUILDER_VERSION",
    "FactRecordError",
    "evidence_to_fact",
    "fact_namespace",
    "fact_to_evidence",
]
