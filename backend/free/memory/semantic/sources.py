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

import json
import secrets
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from backend.free.memory.semantic.namespaces import know_half_life_days
from backend.free.memory.types import Provenance, SemanticFact
from backend.free.rag.evidence import (
    Evidence,
    compute_claim_key_structured,
    derive_confidence,
    new_evidence_id,
)
from backend.io import JSONLAppendStore
from backend.log_config import get_logger
from backend.utils import utc_now

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
            value = {g.name: getattr(value, g.name) for g in fields(value)}
        out[f.name] = value
    return out


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
            # キーは ``fields()`` から引く。手書きで列挙すると新フィールドを
            # 足したときに読み落として黙って既定へ倒れる (c_05 §0.5)。
            value = FetchPolicy(**{
                f.name: value[f.name]
                for f in fields(FetchPolicy) if f.name in value
            })
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
    """``items.jsonl`` (追記式・後勝ち)。"""

    def __init__(self, path: Path | str) -> None:
        self._store: JSONLAppendStore[KnowledgeItem] = JSONLAppendStore(
            path,
            serialize=lambda i: json.dumps(_to_record(i), ensure_ascii=False),
            deserialize=lambda line: _from_record(KnowledgeItem, json.loads(line)),
            key_of=lambda i: i.id,
        )
        self._items: dict[str, KnowledgeItem] = {}

    def load(self) -> None:
        self._items = dict(self._store.load_all())

    def get(self, item_id: str) -> KnowledgeItem | None:
        return self._items.get(item_id)

    def all(self) -> list[KnowledgeItem]:
        return list(self._items.values())

    def put(self, item: KnowledgeItem) -> KnowledgeItem:
        if not item.id:
            item.id = new_item_id()
        if not item.fetched_at:
            item.fetched_at = utc_now()
        self._items[item.id] = item
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
                    url_hash=_hash_url(url),
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


def _hash_url(url: str) -> str:
    """URL のハッシュ (``url_hash``)。空文字は空のまま。"""
    import hashlib

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
    "new_item_id",
    "new_source_id",
]
