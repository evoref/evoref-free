"""`Evidence` レコード型 (c_16 §3)

チャットへ注入される材料 — 会話由来ノート / 構造化事実 / 世界知識 / 文書チャンク —
を 1 型で表す。コアフィールドは凍結し、kind 固有の拡張は :attr:`Evidence.attrs`
に閉じる (c_16 §3 冒頭)。

## この型を触るときの規約 (c_05 §0.5)

- **シリアライズはキーの手書き列挙をしない**。:func:`Evidence.to_record` は
  ``dataclasses.fields()`` から機械的に組む (列挙だと新フィールドの書き漏れで
  値が黙って消える)。
- **未知キーは捨てずに** :attr:`Evidence._extra` へ退避し、書き戻しで復元する。
- **必須キー欠損はそのレコードだけ落とす** — :class:`EvidenceRecordError` を
  上げ、呼び出し側が件数を WARNING に出す。
- 時刻は ISO 8601 UTC μs ``Z`` の 1 形式 (``backend.utils.format_utc``)。
  文字列のまま辞書順で比較しない。
- ID は ``ev_`` + 12 hex のランダム値 (:func:`new_evidence_id`)。位置カウンタを
  鍵にしない。
"""

from __future__ import annotations

import hashlib
import json
import secrets
import unicodedata
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from backend.log_config import get_logger

logger = get_logger("rag.evidence.types")

#: レコード版。フィールドの意味を変える / 必須キーを増やす変更で上げる。
RECORD_VERSION = 1

# ── 列挙 (Literal + カラム用の小さな int テーブル) ───────────────────────

Kind = Literal["note", "fact", "claim", "doc_chunk"]
StoreName = Literal["episodic", "semantic", "corpus"]
Origin = Literal["user", "assistant", "tool", "document", "web", "system"]
Veracity = Literal[
    "stated", "corroborated", "unverified", "disputed", "retracted",
]
Confidentiality = Literal["normal", "secret"]
Tier = Literal["working", "short", "long"]

#: ``columns.npz`` の uint8 カラムへ落とすための文字列↔id テーブル。
#: **値は永続化されるので既存の割当を変えない** (追加は末尾へ)。
KIND_IDS: dict[str, int] = {"note": 0, "fact": 1, "claim": 2, "doc_chunk": 3}
STORE_IDS: dict[str, int] = {"episodic": 0, "semantic": 1, "corpus": 2}
ORIGIN_IDS: dict[str, int] = {
    "user": 0, "assistant": 1, "tool": 2, "document": 3, "web": 4, "system": 5,
}
VERACITY_IDS: dict[str, int] = {
    "stated": 0, "corroborated": 1, "unverified": 2, "disputed": 3,
    "retracted": 4,
}
TIER_IDS: dict[str, int] = {"working": 0, "short": 1, "long": 2}

#: 未知値 / 未設定を表す id (uint8 カラムでは 255、int16 カラムでは -1)。
UNKNOWN_U8 = 255
UNKNOWN_I16 = -1

KIND_NAMES: dict[int, str] = {v: k for k, v in KIND_IDS.items()}
STORE_NAMES: dict[int, str] = {v: k for k, v in STORE_IDS.items()}
ORIGIN_NAMES: dict[int, str] = {v: k for k, v in ORIGIN_IDS.items()}
VERACITY_NAMES: dict[int, str] = {v: k for k, v in VERACITY_IDS.items()}
TIER_NAMES: dict[int, str] = {v: k for k, v in TIER_IDS.items()}

#: 欠けたらそのレコードを落とす必須キー (c_16 §3)。
REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "_version", "id", "kind", "store", "text", "origin",
        "observed_at", "created_at",
    },
)

#: ``patch`` 事象で変更してよいフィールド (c_16 §5.2)。
PATCHABLE_FIELDS: frozenset[str] = frozenset(
    {
        "tier",  # attrs.tier のショートカット (episodic の tier 遷移)
        "veracity", "superseded_by", "contradicts", "pinned", "valid_until",
        "confidence", "as_of", "half_life_days", "claim_key", "last_used_at",
        "attrs", "private", "confidentiality", "scope", "lang",
    },
)


class EvidenceRecordError(ValueError):
    """`Evidence` レコードとして読めない (必須キー欠損 / 値域違反)。

    呼び出し側は **そのレコードだけ** 飛ばして件数を WARNING に出すこと
    (c_05 §0.5.2)。全体を落とさない。
    """


class EvidenceVersionError(EvidenceRecordError):
    """レコードの ``_version`` がコードの :data:`RECORD_VERSION` より新しい。

    「1 レコードだけ飛ばす」で済ませてはいけない唯一の読み取り失敗
    (c_05 §0.5.1)。新しい版を旧いコードで畳んで書き戻すと、知らない
    フィールドを落とした版が正になる。読み手はストアごと readonly に落とし
    (``EvidenceStore.readonly``)、``put`` / ``create_snapshot`` / prune を
    拒否すること。
    """


# ── レコード本体 ────────────────────────────────────────────────────────


@dataclass(slots=True, kw_only=True)
class Evidence:
    """注入材料の統一レコード (c_16 §3)。

    ``kw_only`` にしてあるのは、フィールドが増えても位置引数の意味が変わらない
    ようにするため。**準凍結** として扱い、更新は :func:`dataclasses.replace`
    (または :func:`apply_patch`) で新しいインスタンスを作る — snapshot 行と
    カラム配列は位置で対応しているので、その場書き換えは索引との整合を壊す。
    """

    _version: int = RECORD_VERSION
    id: str
    kind: Kind
    store: StoreName
    scope: str = "global"

    text: str
    lang: str | None = None
    #: fact / claim のみ。``{"subject", "predicate", "object", "value"}`` (§3.2)
    structured: dict[str, Any] | None = None

    origin: Origin
    provenance: list[dict[str, Any]] = field(default_factory=list)

    #: 「こちらが知った時刻」(必須)。内容が真だった時点は ``as_of``。
    observed_at: str
    as_of: str | None = None
    valid_until: str | None = None
    #: ``None`` = 減衰なし。namespace / パッケージ既定を上書きする。
    half_life_days: float | None = None

    #: :func:`derive_confidence` で決定論導出した値 (LLM に答えさせない)。
    confidence: float = 1.0
    veracity: Veracity = "stated"
    claim_key: str | None = None
    contradicts: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    private: bool = False
    confidentiality: Confidentiality = "normal"
    pinned: bool = False

    created_at: str
    updated_at: str | None = None
    last_used_at: str | None = None

    #: kind 別の拡張 (§3.5)。
    attrs: dict[str, Any] = field(default_factory=dict)
    #: 未知キーの退避先 (前方互換)。
    _extra: dict[str, Any] = field(default_factory=dict)

    # ── シリアライズ ──

    def to_record(self) -> dict[str, Any]:
        """JSON レコード (dict) にする。

        キーは :func:`dataclasses.fields` から機械生成する。**手書きで列挙
        しないこと** — フィールドを足したときに書き漏れて値が消える。
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json_line(self) -> str:
        """``records.jsonl`` / 事象ログの 1 行分 (改行を含まない JSON)。"""
        return json.dumps(self.to_record(), ensure_ascii=False)

    @property
    def namespace(self) -> str:
        """`structured.subject` の先頭要素 (``mem`` / ``know`` / ``idx`` …)。

        構造化されていないレコードは空文字。
        """
        if not isinstance(self.structured, dict):
            return ""
        subject = self.structured.get("subject")
        if not isinstance(subject, str) or not subject:
            return ""
        return subject.split(".", 1)[0]

    @property
    def tier(self) -> str:
        """episodic ノートの tier (``attrs.tier``)。無ければ空文字。"""
        value = self.attrs.get("tier")
        return value if isinstance(value, str) else ""


def from_record(data: dict[str, Any]) -> Evidence:
    """JSON レコードから :class:`Evidence` を復元する。

    - 必須キー (:data:`REQUIRED_KEYS`) が欠けていれば
      :class:`EvidenceRecordError`
    - 未知キーは ``_extra`` へ退避する (捨てない)
    - kind 別 ``attrs`` を :func:`validate_attrs` で検査する
    """
    if not isinstance(data, dict):
        raise EvidenceRecordError(f"record must be a dict, got {type(data).__name__}")

    version = data.get("_version")
    if isinstance(version, int) and version > RECORD_VERSION:
        raise EvidenceVersionError(
            f"record {data.get('id')!r} has _version {version}, newer than the "
            f"supported {RECORD_VERSION}",
        )

    missing = sorted(key for key in REQUIRED_KEYS if data.get(key) is None)
    if missing:
        raise EvidenceRecordError(f"missing required keys: {', '.join(missing)}")

    known = {f.name for f in fields(Evidence)}
    extra = dict(data.get("_extra") or {})
    for key, value in data.items():
        if key not in known:
            extra[key] = value

    kind = str(data["kind"])
    attrs = dict(data.get("attrs") or {})
    validate_attrs(kind, attrs)

    provenance = data.get("provenance") or []
    if not isinstance(provenance, list):
        raise EvidenceRecordError("provenance must be a list")
    contradicts = data.get("contradicts") or []
    if not isinstance(contradicts, list):
        raise EvidenceRecordError("contradicts must be a list")
    structured = data.get("structured")
    if structured is not None and not isinstance(structured, dict):
        raise EvidenceRecordError("structured must be an object or null")

    _require_enum("kind", kind, KIND_IDS)
    _require_enum("store", str(data["store"]), STORE_IDS)
    _require_enum("origin", str(data["origin"]), ORIGIN_IDS)
    veracity = str(data.get("veracity") or "stated")
    _require_enum("veracity", veracity, VERACITY_IDS)
    confidentiality = str(data.get("confidentiality") or "normal")
    if confidentiality not in ("normal", "secret"):
        raise EvidenceRecordError(f"invalid confidentiality: {confidentiality!r}")

    return Evidence(
        _version=int(data.get("_version") or RECORD_VERSION),
        id=str(data["id"]),
        kind=kind,  # type: ignore[arg-type]
        store=str(data["store"]),  # type: ignore[arg-type]
        scope=str(data.get("scope") or "global"),
        text=str(data["text"]),
        lang=_opt_str(data.get("lang")),
        structured=dict(structured) if structured is not None else None,
        origin=str(data["origin"]),  # type: ignore[arg-type]
        provenance=[dict(p) for p in provenance if isinstance(p, dict)],
        observed_at=str(data["observed_at"]),
        as_of=_opt_str(data.get("as_of")),
        valid_until=_opt_str(data.get("valid_until")),
        half_life_days=_opt_float(data.get("half_life_days")),
        confidence=float(data.get("confidence", 1.0)),
        veracity=veracity,  # type: ignore[arg-type]
        claim_key=_opt_str(data.get("claim_key")),
        contradicts=[str(c) for c in contradicts],
        superseded_by=_opt_str(data.get("superseded_by")),
        private=bool(data.get("private", False)),
        confidentiality=confidentiality,  # type: ignore[arg-type]
        pinned=bool(data.get("pinned", False)),
        created_at=str(data["created_at"]),
        updated_at=_opt_str(data.get("updated_at")),
        last_used_at=_opt_str(data.get("last_used_at")),
        attrs=attrs,
        _extra=extra,
    )


def _require_enum(name: str, value: str, table: dict[str, int]) -> None:
    if value not in table:
        raise EvidenceRecordError(
            f"invalid {name}: {value!r} (expected one of {sorted(table)})",
        )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise EvidenceRecordError(f"not a number: {value!r}") from e


def new_evidence_id() -> str:
    """``ev_`` + 12 hex のランダム ID を発番する。

    位置カウンタを使わない — 削除後に再発行されると、既存レコードを別内容で
    上書きする (2026-09-05 監査で VectorStore の chunk id が実際にそうなった)。
    """
    return f"ev_{secrets.token_hex(6)}"


# ── kind 別 attrs の検査 (c_16 §3.5) ────────────────────────────────────

#: 全 kind 共通の任意 attrs — 埋め込み側の宣言 (c_16 §3.5 / §6.1)。読み手は
#: :func:`backend.free.rag.evidence.store.embed_side_of` (snapshot 生成時の
#: 埋め込み) だけで、kind に依らず効く。
EMBED_SIDE_ATTRS: frozenset[str] = frozenset({"embed_as_query", "embed_mode"})

#: kind → (必須 attrs, 任意 attrs)。未知キーは **落とさず通す** (前方互換)。
#: 値域違反だけを :class:`EvidenceRecordError` にする。
ATTRS_SPEC: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "note": (
        frozenset(),
        frozenset({"tier", "turn_index", "summary_of", "mode"}),
    ),
    "fact": (frozenset(), EMBED_SIDE_ATTRS),
    "claim": (
        frozenset(),
        frozenset({"source_kind", "published_at", "region"}),
    ),
    "doc_chunk": (
        frozenset({"package_id", "package_version", "doc_id", "position"}),
        frozenset({"heading"}) | EMBED_SIDE_ATTRS,
    ),
}

#: claim の ``attrs.source_kind`` に許す値 (§3.5)。
CLAIM_SOURCE_KINDS: frozenset[str] = frozenset(
    {"official", "news", "forum", "social", "rumor", "dataset", "manual"},
)

#: note の ``attrs.mode``。
NOTE_MODES: frozenset[str] = frozenset({"chat", "coding"})


def validate_attrs(kind: str, attrs: dict[str, Any]) -> None:
    """kind 別 ``attrs`` を検査する (違反は :class:`EvidenceRecordError`)。

    未知キーは **通す** — 新しい版が足したフィールドを旧版が弾くと、読めた
    はずのレコードが丸ごと落ちる。値域が壊れている既知フィールドだけを弾く。
    """
    spec = ATTRS_SPEC.get(kind)
    if spec is None:
        raise EvidenceRecordError(f"unknown kind: {kind!r}")
    required, _optional = spec
    missing = sorted(k for k in required if attrs.get(k) in (None, ""))
    if missing:
        raise EvidenceRecordError(
            f"{kind}.attrs missing required keys: {', '.join(missing)}",
        )

    # 埋め込み側の宣言は kind 共通 (§3.5)。型を外すと snapshot 生成が黙って
    # document 側へ倒れるので、ここで弾いておく。
    embed_as_query = attrs.get("embed_as_query")
    if embed_as_query is not None and not isinstance(embed_as_query, bool):
        raise EvidenceRecordError("attrs.embed_as_query must be a bool")
    embed_mode = attrs.get("embed_mode")
    if embed_mode is not None and not isinstance(embed_mode, str):
        raise EvidenceRecordError("attrs.embed_mode must be a string")

    if kind == "note":
        tier = attrs.get("tier")
        if tier is not None and tier not in TIER_IDS:
            raise EvidenceRecordError(f"invalid note tier: {tier!r}")
        mode = attrs.get("mode")
        if mode is not None and mode not in NOTE_MODES:
            raise EvidenceRecordError(f"invalid note mode: {mode!r}")
        turn_index = attrs.get("turn_index")
        if turn_index is not None and not isinstance(turn_index, int):
            raise EvidenceRecordError("note turn_index must be an int")
        summary_of = attrs.get("summary_of")
        if summary_of is not None and not isinstance(summary_of, list):
            raise EvidenceRecordError("note summary_of must be a list")
    elif kind == "claim":
        source_kind = attrs.get("source_kind")
        if source_kind is not None and source_kind not in CLAIM_SOURCE_KINDS:
            raise EvidenceRecordError(f"invalid claim source_kind: {source_kind!r}")
        region = attrs.get("region")
        if region is not None and not isinstance(region, list):
            raise EvidenceRecordError("claim region must be a list")
    elif kind == "doc_chunk":
        position = attrs.get("position")
        if not isinstance(position, int):
            raise EvidenceRecordError("doc_chunk position must be an int")


# ── claim_key (c_16 §3.4) ───────────────────────────────────────────────


#: 句読点カテゴリだが **落とさない** 文字。``%`` を消すと「30%」と「30」が
#: 同じ鍵になり、違う主張が 1 件に畳まれる (2026-09-05 監査で桁区切りの
#: 「1,280」が 280 に化けたのと同じ事故の形)。通貨記号や ``°`` は Unicode
#: 上 Symbol なのでそもそも除去対象ではない。
_KEPT_PUNCTUATION: frozenset[str] = frozenset("%‰")


def normalize_claim_text(text: str) -> str:
    """畳み込み鍵用の正規化: NFKC → 空白・句読点除去 → 小文字化。

    全角/半角 (``Ｄ７`` → ``d7``)、濁点の合成、句読点の有無、語間の空白の
    揺れを吸収する。数量の意味を担う記号 (:data:`_KEPT_PUNCTUATION` と
    Symbol カテゴリ) は残す。
    """
    normalized = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for ch in normalized:
        if ch.isspace():
            continue
        if ch in _KEPT_PUNCTUATION:
            out.append(ch)
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("Z"):
            continue
        out.append(ch)
    return "".join(out).lower()


def compute_claim_key(text: str) -> str:
    """本文から畳み込み鍵 (sha256 hex) を作る。

    同じ ``claim_key`` を持つレコードは注入時に 1 件へ畳む (c_16 §7.3)。
    """
    return hashlib.sha256(
        normalize_claim_text(text).encode("utf-8"),
    ).hexdigest()


def compute_claim_key_structured(
    subject: str, predicate: str, obj: Any,
) -> str:
    """fact / claim の畳み込み鍵を ``subject|predicate|正規化 object`` で作る。

    subject / predicate は識別子なので空白除去 + 小文字化だけ、object は本文と
    同じ :func:`normalize_claim_text` を通す。
    """
    subject_key = unicodedata.normalize("NFKC", str(subject)).strip().lower()
    predicate_key = unicodedata.normalize("NFKC", str(predicate)).strip().lower()
    object_key = normalize_claim_text("" if obj is None else str(obj))
    payload = f"{subject_key}|{predicate_key}|{object_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def claim_hash64(claim_key: str | None) -> int:
    """``claim_key`` の先頭 64bit を uint64 として返す (カラム用、§5.3)。

    ``None`` / 不正値は 0。0 は「鍵なし」を表すので畳み込み対象から外すこと。
    """
    if not claim_key:
        return 0
    try:
        raw = bytes.fromhex(claim_key[:16])
    except ValueError:
        return 0
    if len(raw) < 8:
        return 0
    return int.from_bytes(raw[:8], "big", signed=False)


# ── confidence (c_16 §3.3) ──────────────────────────────────────────────

#: origin 別の base 値。``web`` は ``sources.jsonl`` の reliability を使うため
#: テーブルに入れない (:func:`derive_confidence` 参照)。
ORIGIN_BASE_CONFIDENCE: dict[str, float] = {
    "user": 1.0,
    "tool": 0.9,
    "document": 0.8,
    "system": 0.8,
    "assistant": 0.5,
}

#: ``origin=web`` で reliability が分からないときの保守的な既定 (範囲の下端)。
WEB_DEFAULT_RELIABILITY = 0.3

#: reliability の許容域 (c_16 §3.3)。
RELIABILITY_MIN = 0.3
RELIABILITY_MAX = 0.9


def corroboration_bonus(corroboration: int) -> float:
    """``min(1.0, 0.7 + 0.1 × 独立出所数)`` (c_16 §3.3)。"""
    return min(1.0, 0.7 + 0.1 * max(0, int(corroboration)))


def derive_confidence(
    origin: str,
    corroboration: int,
    source_reliability: float | None = None,
) -> float:
    """`confidence` を決定論で導出する (LLM に確度を答えさせない)。

    ``confidence = base(origin) × corroboration_bonus × source_reliability``

    ``origin=web`` は base 自体が ``sources.jsonl`` の reliability
    (0.3〜0.9)。二重に掛けないよう、web では乗数側を 1.0 とする。
    reliability 未知の web は :data:`WEB_DEFAULT_RELIABILITY` (下端) に倒す。
    """
    if origin == "web":
        reliability = (
            WEB_DEFAULT_RELIABILITY if source_reliability is None
            else float(source_reliability)
        )
        base = min(RELIABILITY_MAX, max(RELIABILITY_MIN, reliability))
        multiplier = 1.0
    else:
        base = ORIGIN_BASE_CONFIDENCE.get(origin, 0.5)
        multiplier = 1.0 if source_reliability is None else float(source_reliability)
    value = base * corroboration_bonus(corroboration) * multiplier
    return round(min(1.0, max(0.0, value)), 6)


def corroboration_count(provenance: list[dict[str, Any]] | None) -> int:
    """独立出所数 (裏取り件数) を provenance から **読み込み時に** 数える。

    保存しない (c_16 §3.1)。1 エントリにつき ``source_id`` があればそれを、
    無ければ ``session_id`` を出所とみなし、ユニーク数を返す。どちらも無い
    エントリは数えない (出所が特定できない = 裏取りにならない)。
    """
    if not provenance:
        return 0
    sources: set[str] = set()
    for entry in provenance:
        if not isinstance(entry, dict):
            continue
        value = entry.get("source_id") or entry.get("session_id")
        if isinstance(value, str) and value:
            sources.add(value)
    return len(sources)


__all__ = [
    "ATTRS_SPEC",
    "CLAIM_SOURCE_KINDS",
    "EMBED_SIDE_ATTRS",
    "KIND_IDS",
    "KIND_NAMES",
    "NOTE_MODES",
    "ORIGIN_BASE_CONFIDENCE",
    "ORIGIN_IDS",
    "ORIGIN_NAMES",
    "PATCHABLE_FIELDS",
    "RECORD_VERSION",
    "REQUIRED_KEYS",
    "STORE_IDS",
    "STORE_NAMES",
    "TIER_IDS",
    "TIER_NAMES",
    "UNKNOWN_I16",
    "UNKNOWN_U8",
    "VERACITY_IDS",
    "VERACITY_NAMES",
    "Confidentiality",
    "Evidence",
    "EvidenceRecordError",
    "EvidenceVersionError",
    "Kind",
    "Origin",
    "StoreName",
    "Tier",
    "Veracity",
    "claim_hash64",
    "compute_claim_key",
    "compute_claim_key_structured",
    "corroboration_bonus",
    "corroboration_count",
    "derive_confidence",
    "from_record",
    "new_evidence_id",
    "normalize_claim_text",
    "validate_attrs",
]
