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

キー名・型・既定値の SSOT は ``rag/evidence/types.py`` の ``FactAttrs``
(``ATTRS_SPEC["fact"]``、台帳 lock で凍結)。:data:`FACT_ATTR_FIELDS` はそこから導く。

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
``_version``            ``Evidence._v`` (行の版) に一本化
======================= ===============================================
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any, get_args

from backend.free.memory.semantic.namespaces import namespace_of
from backend.free.memory.types import FactType, Provenance, SemanticFact
from backend.free.rag.evidence import (
    Evidence,
    compute_claim_key_structured,
    new_evidence_id,
)
from backend.free.rag.evidence.types import FactAttrs, attrs_defaults
from backend.log_config import get_logger
from backend.utils import epoch_to_utc, format_utc, parse_utc, utc_now, utc_to_epoch

logger = get_logger("memory.semantic.fact")

#: ``attrs`` へ載せるフィールド名 (``SemanticFact`` の属性名と同じ)。
#: 名前を変えないのは、既存の消費者 (``fact.<name>``) がそのまま動くようにするため。
#: SSOT は ``FactAttrs`` (``ATTRS_SPEC["fact"]``)。
FACT_ATTR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FactAttrs) if f.name != "_extra")

#: 既定値と同じなら ``attrs`` へ書かない (入れ子は既定を省いてよい、c_05 §0.5.2)。
#: ``fact_type`` は既定を持たない — 欠けると ownership が引けないため。
_ATTR_DEFAULTS: dict[str, Any] = {
    name: value for name, value in attrs_defaults("fact").items() if name != "fact_type"
}

#: ``SemanticFact`` に専用フィールドを持たないが ``attrs`` に載せる既知キー
#: (キュレータの統計と出所)。作業型では ``attrs`` の未知キーと同じく
#: ``SemanticFact._extra`` から読める (:func:`evidence_to_fact`)。``Evidence._extra``
#: に残るのは本当に未知のキー (新しい版が足したキー) だけ (c_05 §0.5.2)。
#: 既定値を持たないので、値があれば書く。
FACT_CURATOR_ATTR_FIELDS: frozenset[str] = frozenset({
    # executable_command_curator (Step 8.6)
    "command", "command_normalized", "mode", "exec_count", "success_history",
    "success_avg", "last_query", "last_executed_at",
    # url_curator (Step 8.5)
    "url", "url_normalized", "host", "fetch_count", "score_history", "score_avg",
    "score_count", "last_fetched_at",
    # assertion_curator / personal_fact_curator
    "source_note_id", "raw_utterance",
})

#: ``SemanticFact`` のフィールド → ``Evidence`` の書き先 (c_05 §1.4 の対応表)。
#:
#: 新規の組み立て (:func:`fact_to_evidence`) と既存レコードへの差分
#: (:func:`fact_patch`) はどちらもこの 1 表から組む。``None`` は永続化しない作業値。
#: ``text`` と ``claim_key`` は ``structured`` / ``text`` の変更に連動してこの中で
#: 作り直す。``_extra`` のキーは :func:`_extra_target` が ``attrs`` / ``_extra`` に
#: 振り分ける。``SemanticFact`` にフィールドを足したらここにも足す (足し忘れは
#: テストが落とす)。
FACT_EVIDENCE_FIELDS: dict[str, str | None] = {
    "id": "id",
    "subject": "structured",
    "predicate": "structured",
    "object": "structured",
    "statement": "text",
    "type": "attrs",
    "scope": "scope",
    "lang": "lang",
    "provenances": "provenance",
    "trace_id": "provenance",
    "confidence": "confidence",
    "pinned": "pinned",
    "origin": "origin",
    "veracity": "veracity",
    "contradicts": "contradicts",
    "superseded_by": "superseded_by",
    "created_at": "as_of",
    "accessed_at": "last_used_at",
    "private": "private",
    "embedding": None,
    "_extra": "_extra",
    **{name: "attrs" for name in FACT_ATTR_FIELDS if name != "fact_type"},
}

#: ``structured.value`` の既定 (新規レコードだけが使う。既存レコードの値は保つ)。
_DEFAULT_STRUCTURED_VALUE: dict[str, Any] = {"kind": "text", "number": None, "unit": None}

#: patch で変えられない書き先 (id と、``PATCHABLE_FIELDS`` に無い ``origin``)。
_IMMUTABLE_TARGETS: frozenset[str] = frozenset({"id", "origin"})

_MISSING = object()

#: ``provenance[].extractor_version`` の既定 (抽出器が明示しない場合)。
FACT_BUILDER_VERSION = 1


class FactRecordError(ValueError):
    """``Evidence`` を ``SemanticFact`` として読めない (``fact_type`` 欠損等)。

    呼び出し側は **そのレコードだけ** 飛ばして件数を WARNING に出すこと
    (c_05 §0.5.2)。
    """


#: このコードが知る ``FactType`` (``open`` な列挙、c_05 §0.5.3)。
KNOWN_FACT_TYPES: frozenset[str] = frozenset(get_args(FactType))


def is_ignored_fact_record(record: Evidence) -> bool:
    """未知の列挙値を持つ fact / claim か (保持するが使わない行、c_05 §0.4.5)。

    Evidence のコア列挙 (:attr:`Evidence.ignored`) に加え、``attrs.fact_type`` が
    未知の値の行も対象。``SemanticStore`` のロードはこれを ``_ignored`` に分け、
    索引・注入・GC に入れない。``fact_type`` の **欠損** は未知値ではなく壊れた
    レコード (:class:`FactRecordError`) として扱う。
    """
    if record.ignored:
        return True
    fact_type = (record.attrs or {}).get("fact_type")
    return bool(fact_type) and (not isinstance(fact_type, str) or fact_type not in KNOWN_FACT_TYPES)


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


def _provenance_record(prov: Provenance) -> dict[str, Any]:
    """``Provenance`` 1 件を ``Evidence.provenance`` の要素の形へ落とす (c_16 §3.1)。

    ``captured_at`` は epoch 秒で持たれているので ISO へ揃える (時刻は 1 形式)。
    """
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
    return entry


def _provenance_of(
    fact: SemanticFact, stored: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """``fact.provenances`` (+ ``trace_id``) → ``Evidence.provenance``。

    既存レコードの要素と作業型で同じに見えるものは **元の dict をそのまま使う** —
    組み直すと要素の中の未知キーや ``captured_at`` の原形が落ちる。``trace_id`` は
    ``provenance[0].trace_id`` に載せる (値があるときだけ)。
    """
    pool = [dict(p) for p in (stored or []) if isinstance(p, dict)]
    views = _provenances_from(pool)
    used = [False] * len(pool)
    out: list[dict[str, Any]] = []
    for prov in fact.provenances or []:
        entry: dict[str, Any] | None = None
        for index, view in enumerate(views):
            if not used[index] and view == prov:
                used[index] = True
                entry = pool[index]
                break
        out.append(entry if entry is not None else _provenance_record(prov))
    if out and fact.trace_id and out[0].get("trace_id") != fact.trace_id:
        out[0] = {**out[0], "trace_id": fact.trace_id}
    return out


def _attr_value(fact: SemanticFact, name: str) -> tuple[bool, Any]:
    """``attrs.<name>`` に書く値。既定値と同じなら ``(False, None)`` (書かない)。"""
    if name == "fact_type":
        return True, str(fact.type)
    value = getattr(fact, name, None)
    if name == "session_ids":
        value = sorted(value or ())
    if value == _ATTR_DEFAULTS.get(name):
        return False, None
    if name == "pin_locked_until":
        value = epoch_to_utc(value)  # epoch を永続化しない (c_05 §0.5.4)
    return True, value


def _extra_target(key: str, stored: Evidence | None) -> str:
    """``SemanticFact._extra`` のキーの書き先 (``attrs`` か ``_extra``)。

    既知の作業型フィールドと同名のキーは ``Evidence._extra`` からしか来ない
    (``attrs`` の既知キーはフィールドへ読まれる) ので ``_extra`` へ戻す。
    """
    if key == "fact_type" or key in FACT_ATTR_FIELDS:
        return "_extra"
    if key in FACT_CURATOR_ATTR_FIELDS:
        return "attrs"
    if stored is not None and key in stored.attrs:
        return "attrs"
    return "_extra"


def _core_value(fact: SemanticFact, target: str) -> Any:
    """コアフィールド 1 つの値 (対応表の単純な書き先)。"""
    match target:
        case "scope":
            return fact.scope or "global"
        case "lang":
            return fact.lang or None
        case "confidence":
            return float(fact.confidence)
        case "pinned":
            return bool(fact.pinned)
        case "private":
            return bool(fact.private)
        case "contradicts":
            return list(fact.contradicts or [])
        case "as_of":
            return _iso(fact.created_at)
        case "last_used_at":
            return _iso(fact.accessed_at)
        case _:
            return getattr(fact, target)


def _body_of(fact: SemanticFact) -> str:
    """``Evidence.text`` (正規化済み命題、無ければ発話原文)。"""
    return fact.text or (fact.object or "")


def fact_patch(
    fact: SemanticFact,
    stored: Evidence,
    names: Iterable[str],
    *,
    half_life_days: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """作業型の ``names`` の変更を、既存レコード ``stored`` への patch にする (G1)。

    :data:`FACT_EVIDENCE_FIELDS` の対応だけを書き、それ以外の ``stored`` の中身
    (``kind`` / ``valid_until`` / ``confidentiality`` / ``observed_at`` /
    ``structured.value`` / 各階層の未知キー) には触らない (c_05 §1.4)。
    読み取りビュー (:func:`evidence_to_fact`) と同じ値のフィールドは書かないので、
    差分の無い更新は空の patch になる。

    Args:
        half_life_days: ``subject`` が変わったときの namespace 既定の半減期。

    Returns:
        ``EvidenceStore.patch`` に渡す ``(fields, unset)``。

    Raises:
        ValueError: patch で変えられないフィールド (``id`` / ``origin``) の変更。
    """
    view = evidence_to_fact(stored)
    changed: list[str] = []
    for name in dict.fromkeys(names):
        target = FACT_EVIDENCE_FIELDS.get(name)
        if target is None:
            continue
        if getattr(fact, name) == getattr(view, name):
            continue
        if target in _IMMUTABLE_TARGETS:
            raise ValueError(f"cannot change {name} of an existing fact by patch")
        changed.append(name)

    fields: dict[str, Any] = {}
    unset: list[str] = []
    attrs: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for name in changed:
        target = FACT_EVIDENCE_FIELDS[name]
        if target == "attrs":
            key = "fact_type" if name == "type" else name
            present, value = _attr_value(fact, key)
            if present and stored.attrs.get(key, _MISSING) != value:
                attrs[key] = value
            elif not present and key in stored.attrs:
                unset.append(f"attrs.{key}")
        elif target == "_extra":
            old, new = view._extra or {}, fact._extra or {}
            for key in list(dict.fromkeys([*old, *new])):
                if key in new:
                    if key not in old or old[key] != new[key]:
                        dest = attrs if _extra_target(key, stored) == "attrs" else extra
                        dest[key] = new[key]
                    continue
                if key in stored.attrs and _extra_target(key, stored) == "attrs":
                    unset.append(f"attrs.{key}")
                if key in stored._extra:
                    unset.append(f"_extra.{key}")
        elif target not in ("structured", "text", "provenance"):
            value = _core_value(fact, target)
            if getattr(stored, target) != value:
                fields[target] = value

    structured_names = [n for n in changed if FACT_EVIDENCE_FIELDS[n] == "structured"]
    if structured_names:
        structured = dict(stored.structured) if isinstance(stored.structured, dict) else {}
        for name in structured_names:
            structured[name] = getattr(fact, name) or ("unknown" if name == "subject" else "")
        if structured != stored.structured:
            fields["structured"] = structured
    if "statement" in changed or "object" in changed:
        body = _body_of(fact)
        if body != stored.text:
            fields["text"] = body
    if "subject" in changed or "predicate" in changed or "text" in fields:
        structured = fields.get("structured", stored.structured) or {}
        claim_key = compute_claim_key_structured(
            structured.get("subject") or "unknown",
            structured.get("predicate") or "",
            fields.get("text", stored.text),
        )
        if claim_key != stored.claim_key:
            fields["claim_key"] = claim_key
    if "subject" in changed and half_life_days != stored.half_life_days:
        fields["half_life_days"] = half_life_days
    if "provenances" in changed or "trace_id" in changed:
        provenance = _provenance_of(fact, stored.provenance)
        if provenance != stored.provenance:
            fields["provenance"] = provenance

    if attrs:
        fields["attrs"] = attrs
    if extra:
        fields["_extra"] = extra
    return fields, unset


def _provenances_from(records: list[dict[str, Any]]) -> list[Provenance]:
    """``Evidence.provenance`` から ``Provenance`` の列を復元する。"""
    out: list[Provenance] = []
    for entry in records or []:
        if not isinstance(entry, Mapping):
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
    predicate = fact.predicate or ""
    body = _body_of(fact)
    as_of = _core_value(fact, "as_of") or utc_now()

    attrs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for name, target in FACT_EVIDENCE_FIELDS.items():
        if target != "attrs":
            continue
        key = "fact_type" if name == "type" else name
        present, value = _attr_value(fact, key)
        if present:
            attrs[key] = value
    for key, value in (fact._extra or {}).items():
        dest = attrs if _extra_target(key, None) == "attrs" else extra
        dest[key] = value

    return Evidence(
        id=fact.id or new_evidence_id(),
        kind="fact",
        store="semantic",
        scope=_core_value(fact, "scope"),
        text=body,
        lang=_core_value(fact, "lang"),
        structured={
            "subject": subject,
            "predicate": predicate,
            "object": fact.object or "",
            "value": dict(_DEFAULT_STRUCTURED_VALUE),
        },
        origin=fact.origin,  # type: ignore[arg-type]
        provenance=_provenance_of(fact, None),
        observed_at=as_of,
        as_of=as_of,
        half_life_days=half_life_days,
        confidence=_core_value(fact, "confidence"),
        veracity=fact.veracity,  # type: ignore[arg-type]
        claim_key=compute_claim_key_structured(subject, predicate, body),
        contradicts=_core_value(fact, "contradicts"),
        superseded_by=fact.superseded_by,
        private=_core_value(fact, "private"),
        pinned=_core_value(fact, "pinned"),
        created_at=as_of,
        updated_at=utc_now(),
        last_used_at=_core_value(fact, "last_used_at"),
        attrs=attrs,
        _extra=extra,
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
    if not isinstance(fact_type, str) or fact_type not in KNOWN_FACT_TYPES:
        # 未知の型を素の文字列のまま作業型へ流すと注入の Tier 分類へ漏れる。読み手の
        # 関門 (``SemanticStore`` のロード) が先に ``_ignored`` へ分けるので、ここへ
        # 届くのは関門を通らない呼び出しだけ。
        raise FactRecordError(f"unknown fact_type {fact_type!r}: {record.id}")

    structured = record.structured if record.structured is not None else {}
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
        if key == "pin_locked_until":
            value = utc_to_epoch(value)
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
    "FACT_CURATOR_ATTR_FIELDS",
    "FACT_EVIDENCE_FIELDS",
    "FactRecordError",
    "KNOWN_FACT_TYPES",
    "evidence_to_fact",
    "fact_namespace",
    "fact_patch",
    "fact_to_evidence",
    "is_ignored_fact_record",
]
