"""`know.*` の取得元 (sources) と取得単位 (items) — c_16 §4.2。

```
semantic/sources.jsonl   ks_  {id, kind, name, url_pattern, reliability, region,
                               blocked, fetch_policy{edition, interval_sec, enabled}}
semantic/items.jsonl     ki_  {id, source_id, url_hash, content_hash,
                               status: active|retracted|expired|blocked,
                               fetched_at, raw_ref}
```

claim の ``provenance[].source_id`` は ``item:ki_…``。**取得器 (``origin=web``
の書き手) は Pro 限定** で、Free 側にあるのはここまで — 台帳と、手で 1 件を
入れる :class:`KnowledgeIngest` だけ。

確度は :func:`~backend.free.rag.evidence.derive_confidence` で決めるので、
取得元の ``reliability`` (0.3〜0.9) が実質の上限になる。LLM に確度を答えさせ
ない (c_16 §3.3) のと同じ理由で、**呼出側にも答えさせない**。
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Literal

from backend.free.memory.semantic.namespaces import know_half_life_days
from backend.free.memory.types import Provenance, SemanticFact
from backend.free.rag.evidence import (
    Evidence,
    compute_claim_key_structured,
    derive_confidence,
    new_evidence_id,
)
from backend.free.rag.evidence.types import corroboration_count
from backend.io import JSONLAppendStore
from backend.log_config import get_logger
from backend.utils import parse_utc, utc_now, utc_now_dt

logger = get_logger("memory.semantic.sources")

SOURCES_FILENAME = "sources.jsonl"
ITEMS_FILENAME = "items.jsonl"

#: 取得元の種別 (c_16 §3.5 の ``claim.attrs.source_kind`` と同じ語彙)。
SourceKind = Literal[
    "official", "news", "forum", "social", "rumor", "dataset", "manual",
]

#: 取得単位の状態 (c_16 §4.2)。
ItemStatus = Literal["active", "retracted", "expired", "blocked"]

#: ``reliability`` の許容域 (c_16 §3.3)。範囲外は丸める。
RELIABILITY_MIN = 0.3
RELIABILITY_MAX = 0.9

#: 手動投入の既定の取得元。Free には取得器が無いので、``put_claim`` を
#: 出所指定なしで呼んだ分はここへ紐づく。
MANUAL_SOURCE_ID = "ks_manual"
MANUAL_SOURCE_RELIABILITY = 0.8


def new_source_id() -> str:
    """``ks_`` + 12 hex。位置カウンタを鍵にしない (c_05 §0.5)。"""
    return f"ks_{secrets.token_hex(6)}"


def new_item_id() -> str:
    """``ki_`` + 12 hex。"""
    return f"ki_{secrets.token_hex(6)}"


#: 取得の当て方 (``FetchPolicy.kind``)。``page`` = 1 URL をそのまま、
#: ``rss`` = フィードの各 entry を 1 取得単位にする。
FetchKind = Literal["page", "rss"]


@dataclass
class FetchPolicy:
    """取得器の動かし方 (Pro 限定)。Free では宣言だけ持つ。"""

    edition: str = "pro"
    interval_sec: int = 3600
    enabled: bool = False
    #: 取得の当て方。``KnowledgeSource.kind`` (出所の種別 = 確度の根拠) とは別軸。
    kind: FetchKind = "page"
    #: 実際に叩く URL。``url_pattern`` はワイルドカードを許す照合用なので、
    #: 取得先はここで別に持つ (空なら ``url_pattern`` にワイルドカードが
    #: 無いときだけそれを使う)。
    url: str = ""
    #: 未知キーの退避先 (c_05 §0.5)。入れ子の dataclass も往復で落とさない。
    _extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeSource:
    """取得元 1 件 (``sources.jsonl`` の 1 行)。"""

    id: str
    kind: SourceKind = "manual"
    name: str = ""
    url_pattern: str = ""
    reliability: float = MANUAL_SOURCE_RELIABILITY
    region: list[str] = field(default_factory=list)
    blocked: bool = False
    fetch_policy: FetchPolicy = field(default_factory=FetchPolicy)
    created_at: str = ""
    #: ``know.<domain>.<topic>`` の ``<domain>``。半減期が domain で決まる
    #: (``news`` 7 日 / ``economy`` 1 日 …) ので、取得元ごとに宣言させる。
    #: 空なら取得器が既定 domain を当てる。
    domain: str = ""
    #: 最後に取得器が回した時刻 (ISO 8601 UTC)。``fetch_policy.interval_sec``
    #: の判定に使う。台帳が後勝ちなので、別ファイルに状態を分けない。
    last_fetched_at: str = ""
    _extra: dict[str, Any] = field(default_factory=dict)

    def clamped_reliability(self) -> float:
        """``reliability`` を許容域へ丸める (c_16 §3.3)。"""
        return min(RELIABILITY_MAX, max(RELIABILITY_MIN, float(self.reliability)))


@dataclass
class KnowledgeItem:
    """取得単位 1 件 (``items.jsonl`` の 1 行)。"""

    id: str
    source_id: str = ""
    url_hash: str = ""
    content_hash: str = ""
    status: ItemStatus = "active"
    fetched_at: str = ""
    raw_ref: str | None = None
    _extra: dict[str, Any] = field(default_factory=dict)


def _to_record(obj: Any) -> dict[str, Any]:
    """dataclass → JSON レコード。キーは ``fields()`` から機械生成する。

    手書きで列挙しないこと — フィールドを足したときに書き漏れて値が消える
    (c_05 §0.5)。``_extra`` は先に展開し、既知フィールドで上書きする。
    """
    out: dict[str, Any] = dict(getattr(obj, "_extra", None) or {})
    for f in fields(obj):
        if f.name == "_extra":
            continue
        value = getattr(obj, f.name)
        if isinstance(value, FetchPolicy):
            value = _policy_to_record(value)
        out[f.name] = value
    return out


def _policy_to_record(policy: FetchPolicy) -> dict[str, Any]:
    """``FetchPolicy`` → JSON レコード。``_extra`` を展開してから既知キーを載せる。"""
    out: dict[str, Any] = dict(policy._extra or {})
    for f in fields(policy):
        if f.name == "_extra":
            continue
        out[f.name] = getattr(policy, f.name)
    return out


def _policy_from_record(data: dict[str, Any]) -> FetchPolicy:
    """JSON レコード → ``FetchPolicy``。未知キーは ``_extra`` へ退避する。

    キーは ``fields()`` から引く。手書きで列挙すると新フィールドを足したときに
    読み落として黙って既定へ倒れる (c_05 §0.5)。入れ子だからといって未知キーを
    捨ててよいわけではない — 捨てると往復で消える。
    """
    known = {f.name for f in fields(FetchPolicy)} - {"_extra"}
    extra: dict[str, Any] = dict(data.get("_extra") or {})
    kwargs = {k: v for k, v in data.items() if k in known}
    for key, value in data.items():
        if key not in known and key != "_extra":
            extra[key] = value
    return FetchPolicy(**kwargs, _extra=extra)


def _from_record(cls: type, data: dict[str, Any]) -> Any:
    """JSON レコード → dataclass。未知キーは ``_extra`` へ退避する。"""
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = dict(data.get("_extra") or {})
    for key, value in data.items():
        if key == "_extra":
            continue
        if key not in known:
            extra[key] = value
            continue
        if key == "fetch_policy" and isinstance(value, dict):
            value = _policy_from_record(value)
        kwargs[key] = value
    kwargs["_extra"] = extra
    return cls(**kwargs)


class SourceRegistry:
    """``sources.jsonl`` (追記式・後勝ち)。"""

    def __init__(self, path: Path | str) -> None:
        self._store: JSONLAppendStore[KnowledgeSource] = JSONLAppendStore(
            path,
            serialize=lambda s: json.dumps(_to_record(s), ensure_ascii=False),
            deserialize=lambda line: _from_record(KnowledgeSource, json.loads(line)),
            key_of=lambda s: s.id,
        )
        self._sources: dict[str, KnowledgeSource] = {}

    def load(self) -> None:
        self._sources = dict(self._store.load_all())

    def get(self, source_id: str) -> KnowledgeSource | None:
        return self._sources.get(source_id)

    def all(self) -> list[KnowledgeSource]:
        return list(self._sources.values())

    def put(self, source: KnowledgeSource) -> KnowledgeSource:
        """取得元を登録 / 更新する。"""
        if not source.id:
            source.id = new_source_id()
        if not source.created_at:
            source.created_at = utc_now()
        self._sources[source.id] = source
        self._store.append(source)
        self._store.maybe_compact()
        return source

    def ensure_manual(self) -> KnowledgeSource:
        """手動投入用の既定の取得元を用意する (冪等)。"""
        existing = self._sources.get(MANUAL_SOURCE_ID)
        if existing is not None:
            return existing
        return self.put(
            KnowledgeSource(
                id=MANUAL_SOURCE_ID,
                kind="manual",
                name="manual",
                reliability=MANUAL_SOURCE_RELIABILITY,
                fetch_policy=FetchPolicy(edition="free", enabled=False),
            ),
        )


class ItemRegistry:
    """``items.jsonl`` (追記式・後勝ち)。

    ``url_hash`` の索引を持つ — 取得器は取得単位ごとに「同じ取得先の既存
    item」を引くので、全件走査だと台帳が育つほど 1 サイクルが重くなる。
    """

    def __init__(self, path: Path | str) -> None:
        self._store: JSONLAppendStore[KnowledgeItem] = JSONLAppendStore(
            path,
            serialize=lambda i: json.dumps(_to_record(i), ensure_ascii=False),
            deserialize=lambda line: _from_record(KnowledgeItem, json.loads(line)),
            key_of=lambda i: i.id,
        )
        self._items: dict[str, KnowledgeItem] = {}
        self._by_url_hash: dict[str, set[str]] = {}

    def load(self) -> None:
        self._items = dict(self._store.load_all())
        self._by_url_hash = {}
        for item in self._items.values():
            self._index(item)

    def _index(self, item: KnowledgeItem) -> None:
        if item.url_hash:
            self._by_url_hash.setdefault(item.url_hash, set()).add(item.id)

    def _unindex(self, item: KnowledgeItem) -> None:
        ids = self._by_url_hash.get(item.url_hash)
        if ids is not None:
            ids.discard(item.id)
            if not ids:
                self._by_url_hash.pop(item.url_hash, None)

    def get(self, item_id: str) -> KnowledgeItem | None:
        return self._items.get(item_id)

    def all(self) -> list[KnowledgeItem]:
        return list(self._items.values())

    def by_url_hash(self, url_hash: str) -> list[KnowledgeItem]:
        """``url_hash`` が一致する取得単位 (索引経由、全件走査しない)。"""
        return [
            item for item_id in self._by_url_hash.get(url_hash, set())
            if (item := self._items.get(item_id)) is not None
        ]

    def put(self, item: KnowledgeItem) -> KnowledgeItem:
        if not item.id:
            item.id = new_item_id()
        if not item.fetched_at:
            item.fetched_at = utc_now()
        previous = self._items.get(item.id)
        if previous is not None:
            self._unindex(previous)
        self._items[item.id] = item
        self._index(item)
        self._store.append(item)
        self._store.maybe_compact()
        return item

    def set_status(self, item_id: str, status: ItemStatus) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False
        item.status = status
        self.put(item)
        return True

    def delete(self, item_id: str, *, reason: str = "gc") -> bool:
        """取得単位を台帳から落とす (tombstone、物理削除は compaction)。"""
        item = self._items.pop(item_id, None)
        if item is None:
            return False
        self._unindex(item)
        self._store.tombstone(item_id, reason=reason)
        self._store.maybe_compact()
        return True

    def gc(
        self,
        *,
        keep_days: int,
        statuses: tuple[str, ...] = ("expired", "retracted"),
        now: Any | None = None,
    ) -> int:
        """役目を終えた取得単位を落とす (c_05 §0.5.6)。

        対象は ``statuses`` の状態で、``fetched_at`` から ``keep_days`` 日を
        過ぎたもの。``keep_days <= 0`` は「消さない」。``active`` は claim の
        ``provenance`` が今も指しているので触らない。

        Returns:
            落とした件数。
        """
        if keep_days <= 0:
            return 0
        moment = now or utc_now_dt()
        cutoff = moment - timedelta(days=keep_days)
        removed = 0
        for item in list(self._items.values()):
            if item.status not in statuses:
                continue
            fetched = parse_utc(item.fetched_at)
            # 時刻が読めないものは消さない (壊れた 1 行で履歴を失わない)。
            if fetched is None or fetched >= cutoff:
                continue
            if self.delete(item.id, reason="item_retention"):
                removed += 1
        if removed:
            logger.info("Knowledge items GC: removed=%d (keep_days=%d)", removed, keep_days)
        return removed


class KnowledgeIngest:
    """``know.<domain>.<topic>`` の claim を 1 件入れる (Free 側の手動投入)。

    取得器 (``origin=web`` の自動書き手) は Pro 限定 (c_16 §4.2 / e_02)。
    ここにあるのは「取得元と取得単位を台帳へ書き、claim を 1 件作る」まで。

    確度は :func:`derive_confidence` が決める:
    ``base(origin=web) = 取得元の reliability`` × ``corroboration_bonus``。
    **呼出側に confidence を渡させない** — 確度は出所と裏取り件数から機械的に
    決まる値で、書き手の自己申告ではない (c_16 §3.3)。
    """

    def __init__(self, store: Any) -> None:
        """``store`` は :class:`~backend.free.memory.semantic.store.SemanticStore`。"""
        self._store = store

    def put_claim(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        source_id: str | None = None,
        item_id: str | None = None,
        url: str = "",
        content_hash: str = "",
        source_kind: str = "manual",
        published_at: str | None = None,
        region: list[str] | None = None,
        as_of: str | None = None,
        valid_until: str | None = None,
        value: dict[str, Any] | None = None,
        lang: str = "",
        scope: str = "global",
    ) -> SemanticFact:
        """claim (``kind="claim"``) を 1 件書き、取得単位と結ぶ。

        Args:
            subject: ``know.<domain>.<topic>``。他の namespace は ``ValueError``。
            source_id: 取得元 (``ks_…``)。省略時は手動投入の既定。
            item_id: 取得単位 (``ki_…``)。省略時はここで 1 件作る。
            value: ``{"kind", "number", "unit"}``。数量を持つ claim だけ渡す。
                省略時は ``kind="text"`` (本文のまま) になる。

        Returns:
            作られた :class:`SemanticFact` (``type="claim"``)。
        """
        if not subject.startswith("know."):
            raise ValueError(
                f"claims live under know.<domain>.<topic>, got {subject!r}",
            )
        source = (
            self._store.sources.get(source_id) if source_id
            else self._store.sources.ensure_manual()
        )
        if source is None:
            raise KeyError(f"unknown knowledge source: {source_id}")
        if source.blocked:
            raise ValueError(f"knowledge source is blocked: {source.id}")

        item = self._store.items.get(item_id) if item_id else None
        if item is None:
            item = self._store.items.put(
                KnowledgeItem(
                    id=item_id or new_item_id(),
                    source_id=source.id,
                    url_hash=hash_url(url),
                    content_hash=content_hash,
                    status="active",
                ),
            )

        observed_at = utc_now()
        stamp = as_of or published_at or observed_at
        provenance = [
            Provenance(
                source_id=f"item:{item.id}",
                extractor="knowledge_ingest",
                extractor_version=1,
                captured_at=0.0,
            ),
        ]
        confidence = derive_confidence(
            "web", 1, source_reliability=source.clamped_reliability(),
        )
        record = Evidence(
            id=new_evidence_id(),
            kind="claim",
            store="semantic",
            scope=scope,
            text=object_,
            lang=lang or None,
            structured={
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "value": value or {"kind": "text", "number": None, "unit": None},
            },
            origin="web",
            provenance=[
                {
                    "source_id": p.source_id,
                    "extractor": p.extractor,
                    "extractor_version": p.extractor_version,
                    "captured_at": observed_at,
                }
                for p in provenance
            ],
            observed_at=observed_at,
            as_of=stamp,
            valid_until=valid_until,
            half_life_days=know_half_life_days(
                subject, self._store.know_half_life,
            ),
            confidence=confidence,
            veracity="unverified",
            claim_key=compute_claim_key_structured(subject, predicate, object_),
            created_at=observed_at,
            attrs={
                "fact_type": "claim",
                "source_kind": source_kind,
                "published_at": published_at,
                "region": list(region or []),
                # 注入の出所ヘッダ (c_16 §7.4) が読む。台帳を引かずに 1 行を
                # 組めるよう、レコード側に名前を写しておく。
                "source_name": source.name or source.id,
            },
        )
        return self._store.put_record(record)

    # ── 再取得 / 期限切れ (c_16 §4.2) ──

    def claim_key_of(self, fact_id: str) -> str:
        """レコード側の ``claim_key``。``SemanticFact`` は持っていない。"""
        record = self._store.evidence.get(fact_id)
        return record.claim_key if record is not None else ""

    def find_live_claim(
        self, *, subject: str, predicate: str, object_: str, scope: str = "global",
    ) -> SemanticFact | None:
        """同じ命題 (``claim_key``) の生きている claim を 1 件引く。

        再取得のたびに同じ命題を新しいレコードで積むと、``claim_key`` が同じ
        行がストアに溜まる。読み出し側は ``claim_key`` で畳むので表には出ない
        が、確度 (裏取り件数) は分散したままになる。
        """
        wanted = compute_claim_key_structured(subject, predicate, object_)
        for fact in self._store.search_by_subject(subject, scope=scope):
            if fact.veracity == "retracted" or fact.superseded_by:
                continue
            if self.claim_key_of(fact.id) == wanted:
                return fact
        return None

    def corroborate(
        self,
        fact_id: str,
        *,
        item_id: str,
        published_at: str | None = None,
    ) -> bool:
        """既存 claim に取得単位を 1 件足して確度と時刻を更新する。

        ``update_fact`` は通さない — ``SemanticFact`` 経由で往復すると
        ``kind="claim"`` と ``attrs`` (source_kind / region / published_at) が
        ``fact`` 側の形へ潰れる。レコードを直接組み替えて put し直す。
        """
        record = self._store.evidence.get(fact_id)
        if record is None:
            return False
        marker = f"item:{item_id}"
        now = utc_now()
        provenance = [dict(p) for p in (record.provenance or [])]
        if not any(p.get("source_id") == marker for p in provenance):
            provenance.append({
                "source_id": marker,
                "extractor": "knowledge_ingest",
                "extractor_version": 1,
                "captured_at": now,
            })
        record.provenance = provenance
        record.observed_at = now
        record.updated_at = now
        if published_at and _is_newer(published_at, record.as_of):
            record.as_of = published_at
            attrs = dict(record.attrs or {})
            attrs["published_at"] = published_at
            record.attrs = attrs
        record.confidence = derive_confidence(
            "web",
            corroboration_count(provenance),
            source_reliability=self._reliability_of(item_id),
        )
        self._store.put_record(record)
        return True

    def retire_item_claims(
        self,
        item_ids: Iterable[str],
        *,
        replacements: dict[str, str] | None = None,
        reason: str = "item_expired",
    ) -> dict[str, int]:
        """期限切れになった取得単位を指す claim を畳む (c_16 §4.2)。

        取得先の内容が入れ替わると、その本文から抜いた claim の根拠は消える。
        ``items.jsonl`` 側を ``expired`` にするだけでは claim は生き続けるので、
        ここで宛先まで畳む。

        - 生きている取得単位が他にも残っている claim → 期限切れ分の
          ``provenance`` だけ落とし、裏取り件数から確度を引き直す。
        - 同じ ``<subject>|<predicate>`` の新しい claim がある → その新 claim
          で **supersede** する (更新された言明として辿れるようにする)。
        - どちらでもない → ``reason`` を付けて ``retract``。

        Returns:
            ``{"retracted", "superseded", "detached"}`` の件数。
        """
        expired = {i for i in item_ids if i}
        out = {"retracted": 0, "superseded": 0, "detached": 0}
        if not expired:
            return out
        replacements = replacements or {}
        for fact in self._store.know_facts():
            items = {
                p.source_id[len("item:"):]
                for p in fact.provenances
                if p.source_id.startswith("item:")
            }
            if not items & expired:
                continue
            if items - expired:
                if self._detach_items(fact.id, expired):
                    out["detached"] += 1
                continue
            new_id = replacements.get(f"{fact.subject}|{fact.predicate}")
            if new_id and new_id != fact.id and not fact.superseded_by:
                self._mark_superseded(fact.id, new_id)
                out["superseded"] += 1
                continue
            if self._store.retract_fact(fact.id, reason):
                out["retracted"] += 1
        return out

    def _detach_items(self, fact_id: str, expired: set[str]) -> bool:
        """``expired`` の取得単位を ``provenance`` から外し、確度を引き直す。"""
        record = self._store.evidence.get(fact_id)
        if record is None:
            return False
        original = list(record.provenance or [])
        kept = [
            p for p in original
            if str(p.get("source_id", "")).removeprefix("item:") not in expired
        ]
        if not kept or len(kept) == len(original):
            return False
        record.provenance = kept
        record.updated_at = utc_now()
        record.confidence = derive_confidence(
            "web",
            corroboration_count(kept),
            source_reliability=self._reliability_of(
                str(kept[0].get("source_id", "")).removeprefix("item:"),
            ),
        )
        self._store.put_record(record)
        return True

    def _mark_superseded(self, old_id: str, new_id: str) -> None:
        """敗者の ``superseded_by`` をレコード側で立てる。

        ``SemanticStore.supersede`` は ``update_fact`` 経由なので claim の
        ``kind`` / ``attrs`` を潰す。claim は本文ではなくレコードが正なので、
        ここで直接書く。
        """
        record = self._store.evidence.get(old_id)
        if record is None or record.superseded_by:
            return
        record.superseded_by = new_id
        record.updated_at = utc_now()
        self._store.put_record(record)

    def _reliability_of(self, item_id: str) -> float:
        """取得単位 → 取得元 → ``reliability``。辿れなければ下端。"""
        item = self._store.items.get(item_id)
        source = self._store.sources.get(item.source_id) if item else None
        if source is None:
            return RELIABILITY_MIN
        return source.clamped_reliability()


def _is_newer(candidate: str, current: str | None) -> bool:
    """``candidate`` が ``current`` より新しいか。文字列のまま比べない (c_05 §0.5)。"""
    left = parse_utc(candidate)
    if left is None:
        return False
    right = parse_utc(current or "")
    return right is None or left > right


def hash_url(url: str) -> str:
    """URL のハッシュ (``url_hash``、sha256 先頭 32 hex)。空文字は空のまま。"""
    if not url:
        return ""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "ITEMS_FILENAME",
    "MANUAL_SOURCE_ID",
    "MANUAL_SOURCE_RELIABILITY",
    "RELIABILITY_MAX",
    "RELIABILITY_MIN",
    "SOURCES_FILENAME",
    "FetchKind",
    "FetchPolicy",
    "ItemRegistry",
    "ItemStatus",
    "KnowledgeIngest",
    "KnowledgeItem",
    "KnowledgeSource",
    "SourceKind",
    "SourceRegistry",
    "hash_url",
    "new_item_id",
    "new_source_id",
]
