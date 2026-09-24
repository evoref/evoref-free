"""`Evidence` レコード型 (c_16 §3)

チャットへ注入される材料 — 会話由来ノート / 構造化事実 / 世界知識 / 文書チャンク —
を 1 型で表す。コアフィールドは凍結し、kind 固有の拡張は :attr:`Evidence.attrs`
に閉じる (c_16 §3 冒頭)。

## この型を触るときの規約 (c_05 §0.5)

- **シリアライズはキーの手書き列挙をしない**。:func:`from_record` /
  :meth:`Evidence.to_record` は前計算表のコーデック (:mod:`backend.io.codec`) を
  なぞる (列挙だと新フィールドの書き漏れで値が黙って消える)。
- **未知キーは捨てずに階層ごとの ``_extra`` へ退避**し、書き戻しでその階層の
  トップへ戻す — レコード直下は :attr:`Evidence._extra`、``provenance`` の要素・
  ``structured``・``structured.value`` は型付きの入れ子 (:class:`ProvenanceEntry` /
  :class:`Structured` / :class:`StructuredValue`) の ``_extra``。kind 別 ``attrs`` は
  キー単位で patch される素の dict のまま持ち、未知キーはその dict のトップに
  そのまま残る (既知キーの型は :data:`ATTRS_SCHEMAS` の表で検査する)。メモリ上の
  ``_extra`` は空なら ``None``。
- 入れ子の型付き階層は読み手に dict と同じ面 (``.get`` / ``[k]`` / ``in``) を見せる。
- **必須キー欠損はそのレコードだけ落とす** — :class:`EvidenceRecordError` を
  上げ、呼び出し側が件数を WARNING に出す。
- **既知の版で未知の列挙値** (kind / store / origin / veracity / confidentiality /
  note の tier / mode) は落とさない。行を原形のまま保持し、使わない
  (:attr:`Evidence.ignored`。除外の関門は ``is_active`` / ``active_mask`` と
  ``SemanticStore`` のロードの 2 箇所だけ、c_05 §0.4.5 / §0.5.3)。書き手側
  (``EvidenceStore.put``) は未知値を拒否する。
- 時刻は ISO 8601 UTC μs ``Z`` の 1 形式 (``backend.utils.format_utc``)。
  文字列のまま辞書順で比較しない。
- ID は ``ev_`` + 16 hex のランダム値 (:func:`new_evidence_id`、ID 台帳
  ``backend.io.id_registry``)。位置カウンタを鍵にしない。
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar, Literal

from backend.io import jsoncodec
from backend.io.codec import CodecError, codec_for, intern_str, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import fits_id_column, new_id
from backend.log_config import get_logger

logger = get_logger("rag.evidence.types")

#: レコード版。フィールドの意味を変える / 必須キーを増やす変更で上げる。
RECORD_VERSION = 1

# ── 列挙 (Literal + カラム用の小さな int テーブル) ───────────────────────

Kind = Literal["note", "fact", "claim", "doc_chunk", "doc_pseudo_query", "code_node"]
StoreName = Literal["episodic", "semantic", "corpus"]
Origin = Literal["user", "assistant", "tool", "document", "web", "system"]
Veracity = Literal[
    "stated", "corroborated", "unverified", "disputed", "retracted",
]
Confidentiality = Literal["normal", "secret"]
Tier = Literal["working", "short", "long"]
#: claim (``kind="claim"``) の ``structured.predicate`` — **誰が言ったか** の 4 段 (c_16 §3.3)。
#: 書き手は Pro の取得器だが語彙は Free に置く (Pro は import して使う、
#: 抽出のプロンプトとアルゴリズムは Pro に残す)。
ClaimPredicate = Literal["states", "reports", "rumors", "measures"]

#: ``columns.npz`` の uint8 カラムへ落とすための文字列↔id テーブル。
#: **値は永続化されるので既存の割当を変えない** (追加は末尾へ)。
KIND_IDS: dict[str, int] = {
    "note": 0, "fact": 1, "claim": 2, "doc_chunk": 3, "doc_pseudo_query": 4,
    "code_node": 5,
}
STORE_IDS: dict[str, int] = {"episodic": 0, "semantic": 1, "corpus": 2}
ORIGIN_IDS: dict[str, int] = {
    "user": 0, "assistant": 1, "tool": 2, "document": 3, "web": 4, "system": 5,
}
VERACITY_IDS: dict[str, int] = {
    "stated": 0, "corroborated": 1, "unverified": 2, "disputed": 3,
    "retracted": 4,
}
TIER_IDS: dict[str, int] = {"working": 0, "short": 1, "long": 2}
#: ``confidentiality`` の既知値 (カラムは持たず ``flags`` の secret ビットで表す)。
CONFIDENTIALITY_VALUES: frozenset[str] = frozenset({"normal", "secret"})

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
        "_v", "id", "kind", "store", "text", "origin",
        "observed_at", "created_at",
    },
)

#: ``patch`` 事象で変更してよいフィールド (c_16 §5.2 / G1 の patch op)。
#:
#: - ``attrs`` / ``_extra`` はキー単位で重ねる (浅いマージ)。``attrs`` の中の階層の
#:   ``_extra`` もキーとして重なる。``text`` / ``structured`` / ``provenance`` は丸ごと置換。
#: - **値の削除は null ではなく明示の ``unset``** (:data:`UNSETTABLE_PREFIXES`)。null は
#:   「値が無い」を書く (``superseded_by=None`` 等)。
#: - ``touch`` 事象も ``last_used_at`` を必ず運ぶ (superseded の保持期間の起点)。
PATCHABLE_FIELDS: frozenset[str] = frozenset(
    {
        "tier",  # attrs.tier のショートカット (episodic の tier 遷移)
        "veracity", "superseded_by", "contradicts", "pinned", "valid_until",
        "confidence", "as_of", "half_life_days", "claim_key", "last_used_at",
        "attrs", "private", "confidentiality", "scope", "lang",
        "text", "structured", "provenance", "_extra",
    },
)

#: ``unset`` で **キーを消せる** 入れ子 (``"attrs.<key>"`` / ``"_extra.<key>"``)。
#: それ以外の ``unset`` の名前は :data:`PATCHABLE_FIELDS` のフィールド名で、既定値へ戻す。
UNSETTABLE_PREFIXES: frozenset[str] = frozenset({"attrs", "_extra"})


class EvidenceRecordError(ValueError):
    """`Evidence` レコードとして読めない (必須キー欠損 / 値域違反)。

    呼び出し側は **そのレコードだけ** 飛ばして件数を WARNING に出すこと
    (c_05 §0.5.2)。全体を落とさない。
    """


class EvidenceVersionError(EvidenceRecordError):
    """レコード (または事象の行) の ``_v`` がコードの版より新しい。

    「1 レコードだけ飛ばす」で済ませてはいけない唯一の読み取り失敗
    (c_05 §0.5.1)。新しい版を旧いコードで畳んで書き戻すと、知らない
    フィールドを落とした版が正になる。読み手はストアごと readonly に落とし
    (``EvidenceStore.readonly``)、``put`` / ``create_snapshot`` / prune を
    拒否すること。
    """


# ── 型付きの入れ子 (c_16 §3.1 / §3.2、c_05 §0.5.2) ──────────────────────


class _NestedLevel(Mapping[str, Any]):
    """型付きの入れ子の階層を、読み手には dict と同じ面で見せる。

    キーは値のある (``None`` でない) 既知フィールドと ``_extra`` のキー — 書き出し
    (既定値を省く) と同じ集合。``.get`` / ``[k]`` / ``in`` / ``dict(...)`` / ``==``
    (dict とも比べられる) はこの集合で動く。
    """

    __slots__ = ()
    _KEYS: ClassVar[frozenset[str]] = frozenset()
    _ORDER: ClassVar[tuple[str, ...]] = ()

    def __getitem__(self, key: str) -> Any:
        if key in self._KEYS:
            value = getattr(self, key)
            if value is not None:
                return value
            raise KeyError(key)
        extra = getattr(self, "_extra")
        if extra and key in extra:
            return extra[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._KEYS:
            value = getattr(self, key)
            return default if value is None else value
        extra = getattr(self, "_extra")
        if extra:
            return extra.get(key, default)
        return default

    def __iter__(self) -> Iterator[str]:
        for name in self._ORDER:
            if getattr(self, name) is not None:
                yield name
        extra = getattr(self, "_extra")
        if extra:
            for key in extra:
                if key not in self._KEYS:
                    yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


def plain_value(value: Any) -> Any:
    """型付きの入れ子 (``ProvenanceEntry`` 等) を書き出しと同じ素の dict / list に戻す。

    patch 事象の ``fields`` はそのまま JSON に書かれるので、常駐レコードから写した
    階層オブジェクトを渡されても素の形にしてから追記する。
    """
    if isinstance(value, _NestedLevel):
        return codec_for(type(value)).encode(value)
    if isinstance(value, list):
        return [plain_value(v) for v in value]
    if isinstance(value, dict):
        return {k: plain_value(v) for k, v in value.items()}
    return value


def _level_keys(cls: type) -> type:
    """階層クラスの既知キー (フィールド順) を ``_NestedLevel`` の表に載せる。"""
    names = tuple(f.name for f in fields(cls) if f.name != "_extra")
    cls._ORDER = names  # type: ignore[attr-defined]
    cls._KEYS = frozenset(names)  # type: ignore[attr-defined]
    return cls


@_level_keys
@persisted(omit_defaults=True)
@dataclass(slots=True, kw_only=True, eq=False)
class StructuredValue(_NestedLevel):
    """``structured.value`` — 値の種類と数量 (c_16 §3.2)。"""

    kind: Literal["text", "number", "date", "list"] | None = None
    number: float | None = None
    unit: str | None = None
    _extra: dict[str, Any] | None = None


@_level_keys
@persisted(omit_defaults=True, intern=("subject", "predicate"))
@dataclass(slots=True, kw_only=True, eq=False)
class Structured(_NestedLevel):
    """fact / claim の ``structured`` (c_16 §3.2)。namespace は ``subject`` の先頭要素。"""

    subject: str | None = None
    predicate: str | None = None
    #: 発話原文 (証拠)。提示・比較の本文は ``Evidence.text``
    object: str | None = None
    value: StructuredValue | None = None
    _extra: dict[str, Any] | None = None


@_level_keys
@persisted(omit_defaults=True, intern=("extractor", "model", "mode"))
@dataclass(slots=True, kw_only=True, eq=False)
class ProvenanceEntry(_NestedLevel):
    """``provenance`` の 1 要素 (c_16 §3.1)。値の無いキーは書かない。"""

    session_id: str | None = None
    turn_id: str | None = None
    trace_id: str | None = None
    note_id: str | None = None
    #: 文書 / 取得単位 (``doc:<pkg>/<doc>`` / ``item:ki_…``)
    source_id: str | None = None
    extractor: str | None = None
    extractor_version: int | None = None
    captured_at: str | None = None
    mode: str | None = None
    project_id: str | None = None
    source: str | None = None
    model: str | None = None
    _extra: dict[str, Any] | None = None


def _nested(cls: type, value: Any) -> Any:
    """書き手が素の dict で渡した入れ子を型付きの階層にする。"""
    if type(value) is not dict:
        return value
    try:
        return codec_for(cls).decode(value)
    except CodecError as e:
        raise EvidenceRecordError(str(e)) from e


# ── レコード本体 ────────────────────────────────────────────────────────


@persisted(intern=("scope", "lang"))
@dataclass(slots=True, kw_only=True)
class Evidence:
    """注入材料の統一レコード (c_16 §3)。

    ``kw_only`` にしてあるのは、フィールドが増えても位置引数の意味が変わらない
    ようにするため。**準凍結** として扱い、更新は :func:`dataclasses.replace`
    (または :func:`apply_patch`) で新しいインスタンスを作る — snapshot 行と
    カラム配列は位置で対応しているので、その場書き換えは索引との整合を壊す。

    常駐は Evidence 1 本 (G1 設計 §17.4): ``slots`` で、低カーディナリティの文字列
    (scope / lang / structured の subject・predicate / provenance の extractor・model・
    mode / attrs の tier・mode・fact_type) は読み込み時に 1 つの実体へ寄せる。
    """

    #: 行の版 (c_05 §0.5.1 の ``_v``)
    _v: int = RECORD_VERSION
    id: str
    kind: Kind
    store: StoreName
    scope: str = "global"

    text: str
    lang: str | None = None
    #: fact / claim のみ (§3.2)
    structured: Structured | None = None

    origin: Origin
    provenance: list[ProvenanceEntry] = field(default_factory=list)

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

    #: kind 別の拡張 (§3.5)。キー単位で patch するので素の dict のまま持ち、
    #: 既知キーの型は :data:`ATTRS_SCHEMAS` の表で検査する。
    attrs: dict[str, Any] = field(default_factory=dict)
    #: 未知キーの退避先 (前方互換)。書き出しでレコード直下へ戻す。空なら ``None``。
    _extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # 書き手が素の dict で渡した入れ子を型付きの階層へ (読み手の decode は型付き済み)
        structured = self.structured
        if type(structured) is dict:
            self.structured = _nested(Structured, structured)
        for entry in self.provenance:
            if type(entry) is dict:
                self.provenance = [_nested(ProvenanceEntry, p) for p in self.provenance]
                break
        if self._extra is not None and not self._extra:
            self._extra = None

    # ── シリアライズ ──

    def to_record(self) -> dict[str, Any]:
        """JSON レコード (dict) にする。

        キーは前計算表 (``dataclasses.fields()`` から機械生成) をなぞる。**手書きで
        列挙しないこと** — フィールドを足したときに書き漏れて値が消える。未知キーは
        各階層のトップへ戻す (``_extra`` という名のキーは書かない)。
        """
        return _EVIDENCE_CODEC.encode(self)

    def to_json_line(self) -> str:
        """``records.jsonl`` / 事象ログの 1 行分 (改行を含まない JSON)。"""
        return jsoncodec.dumps(self.to_record())

    @property
    def namespace(self) -> str:
        """`structured.subject` の先頭要素 (``mem`` / ``know`` / ``idx`` …)。

        構造化されていないレコードは空文字。
        """
        structured = self.structured
        if structured is None:
            return ""
        subject = structured.get("subject")
        if not isinstance(subject, str) or not subject:
            return ""
        return intern_str(subject.split(".", 1)[0])

    @property
    def tier(self) -> str:
        """episodic ノートの tier (``attrs.tier``)。無ければ空文字。"""
        value = self.attrs.get("tier")
        return value if isinstance(value, str) else ""

    def unknown_enums(self) -> tuple[str, ...]:
        """このコードが知らない列挙値を持つフィールド名 (無ければ空)。

        読み手の規則 (c_05 §0.4.5): 未知値の行は原形のまま保持し、検索・注入・
        学習・GC・件数上限の対象にしない。``closed`` / ``open`` の区別は書き手の
        版上げの義務だけで、読み手の挙動は同じ。
        """
        unknown = [
            name for name, value, table in (
                ("kind", self.kind, KIND_IDS),
                ("store", self.store, STORE_IDS),
                ("origin", self.origin, ORIGIN_IDS),
                ("veracity", self.veracity, VERACITY_IDS),
                ("confidentiality", self.confidentiality, CONFIDENTIALITY_VALUES),
            )
            if value not in table
        ]
        if self.kind == "note":
            tier = self.attrs.get("tier")
            if tier is not None and (not isinstance(tier, str) or tier not in TIER_IDS):
                unknown.append("attrs.tier")
            mode = self.attrs.get("mode")
            if mode is not None and (not isinstance(mode, str) or mode not in NOTE_MODES):
                unknown.append("attrs.mode")
        return tuple(unknown)

    @property
    def ignored(self) -> bool:
        """未知の列挙値を持つ (保持するが使わない行、c_05 §0.4.5)。"""
        return bool(self.unknown_enums())


_EVIDENCE_CODEC = codec_for(Evidence)


def from_record(data: dict[str, Any]) -> Evidence:
    """JSON レコードから :class:`Evidence` を復元する (前計算表のコーデック)。

    - 版がコードより新しければ :class:`EvidenceVersionError` (飛ばさない)
    - 必須キー (:data:`REQUIRED_KEYS`) の欠損・null、既知フィールドの型違いは
      :class:`EvidenceRecordError` (呼出側がそのレコードだけ飛ばして数える)
    - 未知キーは各階層の ``_extra`` へ退避する (捨てない)。G0 の書き手が入れ子の
      ``_extra`` として書いた未知キーはレコード直下の ``_extra`` へ開く
    - kind 別 ``attrs`` の既知キーは :data:`ATTRS_SCHEMAS` の表で **形だけ** 検査する
    - 未知の列挙値は値のまま保持する (落とさない。:attr:`Evidence.ignored`)
    """
    if not isinstance(data, dict):
        raise EvidenceRecordError(f"record must be a dict, got {type(data).__name__}")

    version = data.get("_v")
    if version is None:
        raise EvidenceRecordError("missing required keys: _v")
    if isinstance(version, int) and version > RECORD_VERSION:
        raise EvidenceVersionError(
            f"record {data.get('id')!r} has _v {version}, newer than the "
            f"supported {RECORD_VERSION}",
        )

    try:
        record = _EVIDENCE_CODEC.decode(data)
    except CodecError as e:
        raise EvidenceRecordError(str(e)) from e

    extra = record._extra
    if extra is not None and "_extra" in extra:
        record._extra = _unfold_g0_extra(extra)
    _share_equal_stamps(record)
    if record.attrs:
        record.attrs = _decode_attrs(record.kind, record.attrs)
    elif ATTRS_SPEC.get(record.kind, _NO_ATTRS)[0]:
        validate_attrs(record.kind, record.attrs, values=False)  # 必須 attrs の欠損
    return record


def _share_equal_stamps(record: Evidence) -> None:
    """1 行の中で同じ値の時刻を 1 つの実体にする (常駐メモリ、G1 設計 §17.4)。

    書き手は ``observed_at`` / ``as_of`` / ``created_at`` / ``last_used_at`` /
    ``provenance[].captured_at`` に同じ時刻を入れることが多い。時刻は高カーディナリティ
    なので全体の intern 表には入れず、行の中だけで寄せる。
    """
    base = record.observed_at
    if record.as_of == base:
        record.as_of = base
    if record.created_at == base:
        record.created_at = base
    if record.last_used_at == base:
        record.last_used_at = base
    if record.updated_at == base:
        record.updated_at = base
    for entry in record.provenance:
        if entry.captured_at == base:
            entry.captured_at = base


def _unfold_g0_extra(extra: dict[str, Any]) -> dict[str, Any] | None:
    """G0 の書き手が ``"_extra": {...}`` と入れ子で書いた未知キーを直下へ開く。"""
    out = {k: v for k, v in extra.items() if k != "_extra"}
    nested = extra["_extra"]
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key not in _EVIDENCE_CODEC.known:  # 既知フィールドが勝つ
                out.setdefault(key, value)
    else:
        out["_extra"] = nested
    return out or None


def check_writable(record: Evidence) -> None:
    """書き手側の検査 (発番時): 未知の列挙値・列に入らない id・attrs の値域を拒否する。

    読み手は未知値の行を保持して使わない (:attr:`Evidence.ignored`) が、このコードが
    自分で未知値を書くのは誤り (``closed`` の列挙に値を足すなら版上げ、c_05 §0.5.3)。
    """
    if not fits_id_column(record.id):
        raise EvidenceRecordError(
            f"evidence id {record.id!r} does not fit the id column (ASCII, <= 24 bytes)",
        )
    unknown = record.unknown_enums()
    if unknown:
        raise EvidenceRecordError(
            f"evidence {record.id} has unknown enum value(s): {', '.join(unknown)}",
        )
    validate_attrs(record.kind, record.attrs)


def new_evidence_id() -> str:
    """``ev_`` + 16 hex のランダム ID を発番する (c_05 §0.5.5)。

    位置カウンタを使わない — 削除後に再発行されると、既存レコードを別内容で
    上書きする (2026-09-05 監査で VectorStore の chunk id が実際にそうなった)。
    """
    return new_id("ev_")


# ── kind 別 attrs の検査 (c_16 §3.5) ────────────────────────────────────

#: 全 kind 共通の任意 attrs — 埋め込み側の宣言 (c_16 §3.5 / §6.1)。読み手は
#: :func:`backend.free.rag.evidence.store.embed_side_of` (snapshot 生成時の
#: 埋め込み) だけで、kind に依らず効く。
EMBED_SIDE_ATTRS: frozenset[str] = frozenset({"embed_as_query", "embed_mode"})

#: kind 別 attrs の表 (キー名・型・既定値の SSOT、台帳 lock R1 で凍結)。
#:
#: attrs はキー単位で patch するので **メモリ上は素の dict** のまま持ち、この表は
#: 読み込み時の型検査と intern、書き手の既定値 (入れ子は既定と同じ値を書かなくて
#: よい、c_05 §0.5.2) にだけ使う。表に無いキーはその dict のトップに原形のまま
#: 残る (この階層の ``_extra``)。キーを足すときはフィールドを 1 行足す (既定値を
#: 持つ任意フィールドの追加は同じ版のまま許される、c_05 §0.4.4)。


@dataclass(slots=True, kw_only=True, eq=False)
class _EmbedSideAttrs:
    """全 kind 共通の任意 attrs — 埋め込み側の宣言 (c_16 §3.5 / §6.1)。"""

    embed_as_query: bool = False
    embed_mode: str = "chat"
    _extra: dict[str, Any] | None = None


@persisted(omit_defaults=True)
@dataclass(slots=True, kw_only=True, eq=False)
class NoteAttrs(_EmbedSideAttrs):
    """note の骨格 (c_16 §3.5)。sleep-time 工程の作業欄 (``NOTE_ATTR_FIELDS``) は表の外。"""

    tier: Literal["working", "short", "long"] | None = None
    turn_index: int | None = None
    summary_of: list[str] | None = None
    mode: Literal["chat", "coding"] | None = None


@persisted(
    omit_defaults=True,
    intern=("fact_type", "mode_origin", "profile_id", "embed_mode"),
)
@dataclass(slots=True, kw_only=True, eq=False)
class FactAttrs(_EmbedSideAttrs):
    """fact の attrs (c_16 §3.5)。名前は ``SemanticFact`` の属性名と同じ。

    ``fact_type`` は fact の層で必須 (欠けた行は ``FactRecordError``)。Evidence の
    読み込みでは落とさない。
    """

    fact_type: str | None = None
    mode_origin: str = "chat"
    profile_id: str = "default"
    pin_locked_until: str | None = None
    auto_evolved: bool = False
    from_correction: bool = False
    failure_signature: str | None = None
    eval_metric: dict[str, Any] | None = None
    session_ids: list[str] = field(default_factory=list)
    retired_note_ids: list[str] = field(default_factory=list)


@persisted(omit_defaults=True, intern=("fact_type",))
@dataclass(slots=True, kw_only=True, eq=False)
class ClaimAttrs(_EmbedSideAttrs):
    """claim (``know.*``) の attrs (c_16 §3.5)。``source_kind`` は自由文。"""

    fact_type: str | None = None
    source_kind: str | None = None
    published_at: str | None = None
    region: list[str] | None = None
    #: 注入の出所ヘッダ (§7.4) の表示名
    source_name: str | None = None


@persisted(omit_defaults=True)
@dataclass(slots=True, kw_only=True, eq=False)
class DocChunkAttrs(_EmbedSideAttrs):
    """doc_chunk の attrs (c_16 §3.5)。"""

    package_id: str
    package_version: str
    doc_id: str
    position: int
    heading: str | None = None


@persisted(omit_defaults=True)
@dataclass(slots=True, kw_only=True, eq=False)
class DocPseudoQueryAttrs(_EmbedSideAttrs):
    """疑似クエリ (f_01 §6) の attrs。対象チャンクの id を持ち、埋め込みは query 側。"""

    target_id: str
    package_id: str
    from_hint: bool = False


@persisted(omit_defaults=True)
@dataclass(slots=True, kw_only=True, eq=False)
class CodeNodeAttrs(_EmbedSideAttrs):
    """ProjectMap の code グラフノード (c_16 §3.5 / §4.4)。``node_type`` は自由文。"""

    package_id: str
    package_version: str
    node_type: str
    path: str
    name: str | None = None
    qualname: str | None = None
    lang: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    parent_id: str | None = None
    signature: str | None = None
    external_imports: list[str] | None = None
    fan_in: int | None = None
    fan_out: int | None = None


#: kind → attrs の表。
ATTRS_SCHEMAS: dict[str, type] = {
    "note": NoteAttrs,
    "fact": FactAttrs,
    "claim": ClaimAttrs,
    "doc_chunk": DocChunkAttrs,
    "doc_pseudo_query": DocPseudoQueryAttrs,
    "code_node": CodeNodeAttrs,
}


#: kind → attrs の表のコーデック (読み込みごとに引き直さない)。
_ATTRS_CODECS = {kind: codec_for(cls) for kind, cls in ATTRS_SCHEMAS.items()}


def attrs_defaults(kind: str) -> dict[str, Any]:
    """kind の attrs の既定値 (必須キーは含まない)。書き手が既定と同じ値を省くのに使う。"""
    out: dict[str, Any] = {}
    for f in fields(ATTRS_SCHEMAS[kind]):
        if f.name == "_extra":
            continue
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
    return out


def _attrs_spec(cls: type) -> tuple[frozenset[str], frozenset[str]]:
    names = [f for f in fields(cls) if f.name != "_extra"]
    required = frozenset(
        f.name for f in names if f.default is MISSING and f.default_factory is MISSING
    )
    return required, frozenset(f.name for f in names) - required


#: kind → (必須 attrs, 任意 attrs)。:data:`ATTRS_SCHEMAS` から導く。未知キーは
#: **落とさず通す** (前方互換)。値域違反だけを :class:`EvidenceRecordError` にする。
ATTRS_SPEC: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    kind: _attrs_spec(cls) for kind, cls in ATTRS_SCHEMAS.items()
}
_NO_ATTRS: tuple[frozenset[str], frozenset[str]] = (frozenset(), frozenset())

#: claim の ``attrs.source_kind`` に許す値 (§3.5)。
CLAIM_SOURCE_KINDS: frozenset[str] = frozenset(
    {"official", "news", "forum", "social", "rumor", "dataset", "manual"},
)

#: note の ``attrs.mode``。
NOTE_MODES: frozenset[str] = frozenset({"chat", "coding"})

#: code_node の ``attrs.node_type`` に許す値 (c_16 §3.5)。``component`` は
#: マークアップ / SFC (Svelte / Vue) の単一ファイルコンポーネント (c_16 §4.4)。
CODE_NODE_TYPES: frozenset[str] = frozenset({
    "file", "class", "function", "method", "component",
})


def validate_attrs(kind: str, attrs: dict[str, Any], *, values: bool = True) -> None:
    """kind 別 ``attrs`` を検査する (違反は :class:`EvidenceRecordError`)。

    未知キーは **通す** — 新しい版が足したフィールドを旧版が弾くと、読めた
    はずのレコードが丸ごと落ちる。値域が壊れている既知フィールドだけを弾く。

    ``values=False`` は読み手の検査 (:func:`from_record`): 形 (型・必須キー) だけを
    見て、列挙の値 (未知の kind / tier / mode) と自由文 (claim の ``source_kind`` /
    code_node の ``node_type``) の値は問わない (未知値の行は保持して使わない、
    c_05 §0.4.5 / §0.5.3)。
    """
    if kind not in ATTRS_SCHEMAS:
        if not values:
            return
        raise EvidenceRecordError(f"unknown kind: {kind!r}")
    _check_attrs(kind, attrs)

    if not values:
        return
    if kind == "note":
        tier = attrs.get("tier")
        if tier is not None and tier not in TIER_IDS:
            raise EvidenceRecordError(f"invalid note tier: {tier!r}")
        mode = attrs.get("mode")
        if mode is not None and mode not in NOTE_MODES:
            raise EvidenceRecordError(f"invalid note mode: {mode!r}")
    elif kind == "claim":
        source_kind = attrs.get("source_kind")
        if source_kind is not None and source_kind not in CLAIM_SOURCE_KINDS:
            raise EvidenceRecordError(f"invalid claim source_kind: {source_kind!r}")
    elif kind == "code_node":
        node_type = attrs.get("node_type")
        if node_type not in CODE_NODE_TYPES:
            raise EvidenceRecordError(f"invalid code_node node_type: {node_type!r}")


def _check_attrs(kind: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """必須キーと既知キーの型を表で検査し、intern 済みの dict を返す。


    埋め込み側の宣言 (kind 共通) の型を外すと snapshot 生成が黙って document 側へ
    倒れるので、ここで弾く。
    """
    if not isinstance(attrs, dict):
        raise EvidenceRecordError(f"{kind}.attrs must be an object")
    required = ATTRS_SPEC[kind][0]
    if required:
        missing = sorted(k for k in required if attrs.get(k) in (None, ""))
        if missing:
            raise EvidenceRecordError(
                f"{kind}.attrs missing required keys: {', '.join(missing)}",
            )
    try:
        return _ATTRS_CODECS[kind].check_mapping(attrs)
    except CodecError as e:
        raise EvidenceRecordError(f"{kind}.attrs: {e}") from e


def _decode_attrs(kind: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """読み手の attrs: 既知の kind は表で検査・intern、未知の kind は形を問わず保持。"""
    if kind not in _ATTRS_CODECS:
        return attrs
    return _check_attrs(kind, attrs)


#: Evidence レコードの形式 (c_05 §0.7.1)。列挙の ``closed`` / ``open`` は書き手の
#: 義務 (``closed`` へ値を足すなら版上げ) で、読み手はどちらも「未知値の行は保持して
#: 使わない」(§0.4.5 / §0.5.3)。``fact_type`` は ``attrs.fact_type``、``namespace`` は
#: ``structured.subject`` の先頭要素、``tier`` / ``mode`` は note の ``attrs``。
#: ``records`` の表 (中核 + 型付きの入れ子 + kind 別 attrs) は lock (R1) で凍結する。
EVIDENCE_RECORD_FORMAT = register_format(FormatSpec(
    format_id="evidence.record",
    version=RECORD_VERSION,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/memory/<store>/snapshot/<ver>/records.jsonl",
    retention="per store (c_16 §5.4)",
    export=True,
    encodings=("jsonl",),
    enums={
        "kind": "open",
        "store": "closed",
        "origin": "open",
        "veracity": "closed",
        "confidentiality": "closed",
        "tier": "closed",
        "mode": "open",
        "fact_type": "open",
        "namespace": "open",
    },
    records=(Evidence, *ATTRS_SCHEMAS.values()),
))


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
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("source_id") or entry.get("session_id")
        if isinstance(value, str) and value:
            sources.add(value)
    return len(sources)


__all__ = [
    "ATTRS_SCHEMAS",
    "ATTRS_SPEC",
    "CLAIM_SOURCE_KINDS",
    "CODE_NODE_TYPES",
    "CONFIDENTIALITY_VALUES",
    "EMBED_SIDE_ATTRS",
    "KIND_IDS",
    "KIND_NAMES",
    "NOTE_MODES",
    "ORIGIN_BASE_CONFIDENCE",
    "ORIGIN_IDS",
    "ORIGIN_NAMES",
    "PATCHABLE_FIELDS",
    "UNSETTABLE_PREFIXES",
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
    "ClaimAttrs",
    "ClaimPredicate",
    "CodeNodeAttrs",
    "Confidentiality",
    "DocChunkAttrs",
    "DocPseudoQueryAttrs",
    "Evidence",
    "EvidenceRecordError",
    "EvidenceVersionError",
    "FactAttrs",
    "Kind",
    "NoteAttrs",
    "Origin",
    "ProvenanceEntry",
    "StoreName",
    "Structured",
    "StructuredValue",
    "Tier",
    "Veracity",
    "attrs_defaults",
    "check_writable",
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
